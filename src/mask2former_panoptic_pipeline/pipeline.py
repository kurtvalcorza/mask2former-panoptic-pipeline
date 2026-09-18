"""Panoptic segmentation with the pinned ``facebook/mask2former-swin-tiny-coco-panoptic`` checkpoint,
plus a bounded re-head-and-fine-tune path for a caller-supplied class vocabulary.

The class loads the image processor and model only from a digest-verified local snapshot
(``weights/<key>/``) or, when explicitly allowed, from the Hugging Face Hub at the pinned revision —
always with ``trust_remote_code=False``: the Mask2Former architecture comes from the pinned
``transformers`` release, the weights are SafeTensors, and no model-repository code is executed.

Panoptic post-processing is implemented here (NumPy + Pillow) following the upstream Mask2Former rule:
a query survives when its best class probability reaches ``score_threshold``; every pixel goes to the
surviving query with the highest score-weighted mask probability; a pixel is assigned only where that
query's own mask probability reaches ``mask_threshold``, otherwise it stays **void** (``-1``); a query
whose assigned area is below ``overlap_threshold`` of its thresholded mask is dropped; stuff queries of
the same class are fused into one segment. The pinned ``transformers`` post-processor differs in the
per-pixel rule (it assigns every pixel to the argmax query), which hands a whole image to a single
low-confidence query on out-of-domain input; the smoke run recorded that difference.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .samples import validate_dataset

MODEL_ID = "facebook/mask2former-swin-tiny-coco-panoptic"
MODEL_REVISION = "df6b1142ff50c3276559d9d78f35f6a579c75a77"
MODEL_LICENSE = "mit"
MODEL_KEY = "mask2former-swin-tiny-coco-panoptic"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_MANIFEST_NAME = "dimer-adapted-manifest.json"
ARTIFACT_FORMAT = "dimer_mask2former_panoptic_adapted/1"

# The pinned processor resizes every image to INPUT_SIZE x INPUT_SIZE (aspect ratio not preserved) and
# the transformer decoder emits NUM_QUERIES mask logits at INPUT_SIZE / 4.
INPUT_SIZE = 384
MASK_LOGIT_SIZE = 96
NUM_QUERIES = 100
# COCO panoptic vocabulary of the checkpoint: label ids 0..79 are the 80 "thing" classes, 80..132 the
# 53 "stuff" classes (config.json id2label order).
COCO_THING_COUNT = 80
# Post-processing thresholds (caller-owned request parameters). SCORE_THRESHOLD is the pinned
# transformers default; the upstream Mask2Former panoptic inference uses 0.8 for the same test.
SCORE_THRESHOLD = 0.5
OVERLAP_THRESHOLD = 0.8
MASK_THRESHOLD = 0.5
# Input ceilings. Cost is bounded by the fixed resize; the side ceiling guards memory during the
# per-query resampling of masks to input resolution.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
# Adaptation defaults (FT4/FT6): chosen for a practical CPU runtime, not inherited from the upstream
# recipe (which trains at 12,544 sampled points per mask for 50 epochs on COCO with a 1e-4 AdamW).
TRAIN_NUM_POINTS = 4096
DEFAULT_EPOCHS = 6
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.05
MAX_BATCH_SIZE = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


# --------------------------------------------------------------------------------------------------
# Input contract
# --------------------------------------------------------------------------------------------------


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one PIL.Image.Image (any mode, converted to RGB)",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "score_threshold": [0.0, 1.0],
    "overlap_threshold": [0.0, 1.0],
    "mask_threshold": [0.0, 1.0],
    "preprocessing": (
        f"image converted to RGB and resized to {INPUT_SIZE}x{INPUT_SIZE} (aspect ratio not preserved, "
        f"ImageNet mean/std); the decoder emits {NUM_QUERIES} (class distribution, {MASK_LOGIT_SIZE}x"
        f"{MASK_LOGIT_SIZE} mask logit) pairs; surviving masks are resampled bilinearly to the input size"
    ),
    "output": (
        "a panoptic map at input resolution (int32, one segment id per pixel, -1 = void) and one entry per "
        "segment: id, label, class index, thing/stuff, score (the query's softmax class probability, "
        "uncalibrated), area fraction, tight box, was_fused"
    ),
}


def _check_inputs(
    image: Any, score_threshold: Any, overlap_threshold: Any, mask_threshold: Any
) -> tuple[Image.Image, float, float, float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``segment`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    return (
        rgb,
        _check_fraction("score_threshold", score_threshold),
        _check_fraction("overlap_threshold", overlap_threshold),
        _check_fraction("mask_threshold", mask_threshold),
    )


def validate_inputs(
    image: Image.Image,
    *,
    score_threshold: float = SCORE_THRESHOLD,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    mask_threshold: float = MASK_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``segment`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, score, overlap, mask = _check_inputs(image, score_threshold, overlap_threshold, mask_threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (segment takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "score_threshold": score,
        "overlap_threshold": overlap,
        "mask_threshold": mask,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


# --------------------------------------------------------------------------------------------------
# Panoptic post-processing (upstream rule, NumPy + Pillow) and panoptic quality
# --------------------------------------------------------------------------------------------------


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    """Tight xyxy pixel box around the true pixels of a mask, or ``None`` for an empty mask."""
    rows = np.flatnonzero(np.asarray(mask, dtype=bool).any(axis=1))
    cols = np.flatnonzero(np.asarray(mask, dtype=bool).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def _resample(mask_logit: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resampling of one float32 map to ``(height, width)`` through Pillow's mode-F resize."""
    height, width = size
    if mask_logit.shape == (height, width):
        return np.asarray(mask_logit, dtype=np.float32)
    return np.asarray(Image.fromarray(np.asarray(mask_logit, dtype=np.float32)).resize((width, height)))


def panoptic_from_logits(
    class_logits: np.ndarray,
    mask_logits: np.ndarray,
    size: tuple[int, int],
    *,
    label_names: Sequence[str],
    stuff_ids: Sequence[int] | frozenset[int],
    score_threshold: float = SCORE_THRESHOLD,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    mask_threshold: float = MASK_THRESHOLD,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Turn one image's raw decoder outputs into a panoptic map and its segment list.

    ``class_logits`` is ``(NUM_QUERIES, K + 1)`` (last column = no object), ``mask_logits`` is
    ``(NUM_QUERIES, h, w)``; ``size`` is the ``(height, width)`` of the input image. Returns an int32
    ``(height, width)`` map whose value is the segment id (1-based) or ``-1`` for void, plus one dict per
    segment ordered by decreasing score. Stuff queries of the same class are fused into one segment.
    """
    class_logits = np.asarray(class_logits, dtype=np.float64)
    mask_logits = np.asarray(mask_logits, dtype=np.float32)
    if class_logits.ndim != 2 or mask_logits.ndim != 3 or class_logits.shape[0] != mask_logits.shape[0]:
        raise ValueError("class_logits must be (Q, K+1) and mask_logits (Q, h, w) with the same Q")
    n_classes = class_logits.shape[1] - 1
    if n_classes != len(label_names):
        raise ValueError(f"class_logits has {n_classes} classes, label_names has {len(label_names)}")
    stuff = frozenset(int(i) for i in stuff_ids)
    shifted = class_logits - class_logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    scores = probs[:, :-1].max(axis=1)
    labels = probs[:, :-1].argmax(axis=1)
    keep = np.flatnonzero(scores > score_threshold)
    height, width = size
    segmentation = np.full((height, width), -1, dtype=np.int32)
    if keep.size == 0:
        return segmentation, []
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    mask_probs = np.stack([1.0 / (1.0 + np.exp(-_resample(mask_logits[q], size))) for q in keep])
    weighted = mask_probs * scores[keep][:, None, None].astype(np.float32)
    assignment = weighted.argmax(axis=0)
    segments: list[dict[str, Any]] = []
    stuff_segment: dict[int, int] = {}
    for position, query in enumerate(keep):
        label = int(labels[query])
        argmax_region = assignment == position
        confident = mask_probs[position] >= mask_threshold
        mask = argmax_region & confident
        mask_area = int(argmax_region.sum())
        original_area = int(confident.sum())
        if mask_area == 0 or original_area == 0 or not mask.any():
            continue
        if mask_area / original_area < overlap_threshold:
            continue
        if label in stuff and label in stuff_segment:
            segment_id = stuff_segment[label]
            segmentation[mask] = segment_id
            entry = next(entry for entry in segments if entry["id"] == segment_id)
            entry["was_fused"] = True
            entry["score"] = max(entry["score"], float(scores[query]))
            continue
        segment_id = len(segments) + 1
        segmentation[mask] = segment_id
        if label in stuff:
            stuff_segment[label] = segment_id
        segments.append(
            {
                "id": segment_id,
                "label": str(label_names[label]),
                "label_id": label,
                "is_thing": label not in stuff,
                "score": float(scores[query]),
                "query": int(query),
                "was_fused": False,
            }
        )
    for entry in segments:
        mask = segmentation == entry["id"]
        entry["area_fraction"] = float(mask.mean())
        entry["bbox"] = mask_bbox(mask)
    return segmentation, segments


def _segment_masks(segmentation: np.ndarray, segments: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
    return [np.asarray(segmentation) == int(entry["id"]) for entry in segments]


def panoptic_quality(
    pairs: Sequence[tuple[np.ndarray, Sequence[Mapping[str, Any]], np.ndarray, Sequence[Mapping[str, Any]]]],
    *,
    class_agnostic: bool = False,
) -> dict[str, Any]:
    """Panoptic quality (Kirillov et al., 2019) over one or more ``(pred_seg, pred_segments, ref_seg,
    ref_segments)`` pairs.

    Segments are matched per class (or regardless of class when ``class_agnostic``) when their IoU exceeds
    0.5 — a rule that makes matches unique. Reference void pixels (``-1``) are ignored: they are removed
    from predicted masks before IoU, and a predicted segment lying mostly (> 50 %) on void is neither a
    match nor a false positive. Returns ``pq``, ``sq``, ``rq`` overall and per class, thing/stuff
    aggregates, and the raw ``tp`` / ``fp`` / ``fn`` / ``iou_sum`` counts they were computed from
    (per-class values are the means over classes that occur in the references or the predictions).
    """
    totals: dict[str, dict[str, float]] = {}
    thing_flag: dict[str, bool] = {}

    def bucket(entry: Mapping[str, Any]) -> str:
        return "region" if class_agnostic else str(entry.get("label", entry.get("name")))

    for pred_seg, pred_segments, ref_seg, ref_segments in pairs:
        pred_seg, ref_seg = np.asarray(pred_seg), np.asarray(ref_seg)
        if pred_seg.shape != ref_seg.shape:
            raise ValueError(f"segmentation shapes differ: {pred_seg.shape} vs {ref_seg.shape}")
        void = ref_seg == -1
        ref_masks = _segment_masks(ref_seg, ref_segments)
        pred_masks = [mask & ~void for mask in _segment_masks(pred_seg, pred_segments)]
        pred_void_fraction = [
            float((mask & void).sum() / mask.sum()) if mask.any() else 0.0
            for mask in _segment_masks(pred_seg, pred_segments)
        ]
        for entry in list(ref_segments) + list(pred_segments):
            key = bucket(entry)
            totals.setdefault(key, {"iou_sum": 0.0, "tp": 0, "fp": 0, "fn": 0})
            thing_flag.setdefault(key, bool(entry.get("is_thing", True)))
        matched_pred: set[int] = set()
        for ref_index, ref_entry in enumerate(ref_segments):
            key = bucket(ref_entry)
            best_iou, best_pred = 0.0, -1
            for pred_index, pred_entry in enumerate(pred_segments):
                if pred_index in matched_pred or bucket(pred_entry) != key:
                    continue
                inter = np.logical_and(ref_masks[ref_index], pred_masks[pred_index]).sum()
                if inter == 0:
                    continue
                union = np.logical_or(ref_masks[ref_index], pred_masks[pred_index]).sum()
                iou = float(inter / union)
                if iou > best_iou:
                    best_iou, best_pred = iou, pred_index
            if best_iou > 0.5:
                matched_pred.add(best_pred)
                totals[key]["tp"] += 1
                totals[key]["iou_sum"] += best_iou
            else:
                totals[key]["fn"] += 1
        for pred_index, pred_entry in enumerate(pred_segments):
            if pred_index in matched_pred or pred_void_fraction[pred_index] > 0.5:
                continue
            totals[bucket(pred_entry)]["fp"] += 1

    def summarise(counts: Mapping[str, float]) -> dict[str, float]:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        denominator = tp + 0.5 * fp + 0.5 * fn
        sq = counts["iou_sum"] / tp if tp else 0.0
        rq = tp / denominator if denominator else 0.0
        return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    per_class = {key: summarise(counts) for key, counts in totals.items()}

    def mean_over(keys: Sequence[str]) -> dict[str, float]:
        if not keys:
            return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "n_classes": 0}
        return {
            **{
                metric: float(np.mean([per_class[key][metric] for key in keys]))
                for metric in ("pq", "sq", "rq")
            },
            "n_classes": len(keys),
        }

    keys = sorted(per_class)
    return {
        "pq": mean_over(keys)["pq"],
        "sq": mean_over(keys)["sq"],
        "rq": mean_over(keys)["rq"],
        "n_classes": len(keys),
        "things": mean_over([key for key in keys if thing_flag[key]]),
        "stuff": mean_over([key for key in keys if not thing_flag[key]]),
        "per_class": per_class,
        "class_agnostic": class_agnostic,
        "n_images": len(pairs),
        "matching": "IoU > 0.5 per class (unique by construction); reference void pixels ignored",
    }


def reference_from_record(record: Mapping[str, Any], class_names: Sequence[str], stuff_names: Sequence[str]):
    """Turn a dataset record into ``(segmentation, segments)`` in the panoptic_quality convention."""
    instance_map = np.asarray(record["instance_map"], dtype=np.int32)
    segments = []
    for instance_id, name in sorted(record["instances"].items(), key=lambda item: int(item[0])):
        mask = instance_map == int(instance_id)
        segments.append(
            {
                "id": int(instance_id),
                "label": name,
                "label_id": list(class_names).index(name),
                "is_thing": name not in stuff_names,
                "area_fraction": float(mask.mean()),
            }
        )
    return instance_map, segments


def evaluation_report(
    result: Mapping[str, Any],
    reference: tuple[np.ndarray, Sequence[Mapping[str, Any]]] | None = None,
    *,
    sample_kind: str = "synthetic",
    class_agnostic: bool = True,
) -> dict[str, Any]:
    """Evaluation stage for the inference path: a machine-readable report even when nothing is measurable.

    With ``reference`` (a panoptic map and its segment list, e.g. the regions a drawn scene was drawn
    from) the report carries panoptic quality — class-agnostic by default, because the checkpoint's COCO
    vocabulary does not name drawn regions — as ``sample-sanity`` evidence; without it the verdict is
    ``not-measurable`` and the report says what labelled data would make the task measurable.
    """
    segments = list(result["segments"])
    base = {
        "task": "panoptic segmentation: every pixel is assigned to one segment (thing instance or fused"
        " stuff) or void",
        "decision_rule": (
            "a query survives when its best class probability reaches score_threshold; each pixel goes to "
            "the surviving query with the highest score-weighted mask probability and is assigned only where "
            "that query's mask probability reaches mask_threshold (otherwise void); a query whose assigned "
            "area is below overlap_threshold of its thresholded mask is dropped; scores are uncalibrated "
            "softmaxes"
        ),
        "thresholds": {
            "score_threshold": result.get("score_threshold", SCORE_THRESHOLD),
            "overlap_threshold": result.get("overlap_threshold", OVERLAP_THRESHOLD),
            "mask_threshold": result.get("mask_threshold", MASK_THRESHOLD),
        },
        "sample_kind": sample_kind,
        "n_segments": len(segments),
        "void_fraction": float((np.asarray(result["segmentation"]) == -1).mean()),
        "labels": [
            (entry["label"], round(entry["score"], 4), round(entry["area_fraction"], 4)) for entry in segments
        ],
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if reference is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference panoptic map was supplied for the evaluated image",
            "needs": (
                "panoptic annotations (an instance map and a class per instance) on images from the "
                "deployment domain with a class vocabulary matching the model's, scored with panoptic "
                "quality (PQ = SQ x RQ, IoU > 0.5 matching); no such labelled set ships with this repository"
            ),
        }
    ref_seg, ref_segments = reference
    quality = panoptic_quality(
        [(np.asarray(result["segmentation"]), segments, np.asarray(ref_seg), list(ref_segments))],
        class_agnostic=class_agnostic,
    )
    estimation = "one reference map on a single image, no dispersion estimate"
    metrics = [
        {"id": "pq", "value": quality["pq"], "estimation": estimation},
        {"id": "sq", "value": quality["sq"], "estimation": estimation},
        {"id": "rq", "value": quality["rq"], "estimation": estimation},
        {
            "id": "matches",
            "value": {
                key: {k: v for k, v in counts.items() if k in ("tp", "fp", "fn")}
                for key, counts in quality["per_class"].items()
            },
            "estimation": "true/false positive and false negative segment counts behind pq",
        },
    ]
    return {
        **base,
        "metrics": metrics,
        "panoptic_quality": quality,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(ref_segments)} reference region(s) on one tutorial sample, "
            + ("matched regardless of class label" if class_agnostic else "matched per class")
            + "; geometry sanity evidence, not a COCO panoptic benchmark"
        ),
        "needs": "panoptic annotations from the deployment domain with a matching vocabulary for any PQ "
        "claim",
    }


# --------------------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------------------


def _build_class_lists(config: Any) -> tuple[list[str], frozenset[int]]:
    names = [config.id2label[i] for i in range(len(config.id2label))]
    return names, frozenset(range(COCO_THING_COUNT, len(names)))


@dataclass
class Mask2FormerPanopticPipeline:
    """Panoptic segmentation over the pinned Mask2Former Swin-T COCO checkpoint, with an optional
    re-headed class vocabulary that ``finetune`` adapts on caller-supplied labelled scenes."""

    model: Any
    processor: Any
    device: str
    class_names: tuple[str, ...]
    stuff_ids: frozenset[int]
    source: str
    adapted: bool = False
    reinitialised: list[str] = field(default_factory=list)
    seed: int | None = None
    training: dict[str, Any] | None = None

    @property
    def stuff_names(self) -> tuple[str, ...]:
        return tuple(self.class_names[i] for i in sorted(self.stuff_ids))

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        *,
        class_names: Sequence[str] | None = None,
        stuff_names: Sequence[str] = (),
        seed: int = 0,
        train_num_points: int = TRAIN_NUM_POINTS,
    ) -> Mask2FormerPanopticPipeline:
        """Load the verified snapshot. With ``class_names`` the classification head is rebuilt for that
        vocabulary (seeded random initialisation of exactly the head; everything else transfers) so the
        model can be adapted with ``finetune``; ``stuff_names`` marks the amorphous classes."""
        if class_names is not None:
            names = list(class_names)
            if len(names) < 2 or len(set(names)) != len(names):
                raise ValueError("class_names must hold at least two distinct names")
            unknown = [name for name in stuff_names if name not in names]
            if unknown:
                raise ValueError(f"stuff names {unknown} are not in class_names")
        elif stuff_names:
            raise ValueError("stuff_names needs class_names (the COCO vocabulary carries its own stuff set)")
        if (
            isinstance(train_num_points, bool)
            or not isinstance(train_num_points, int)
            or train_num_points < 1
        ):
            raise ValueError(f"train_num_points must be a positive int, got {train_num_points!r}")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        head: dict[str, Any] = {"train_num_points": train_num_points}
        if class_names is not None:
            head.update(
                num_labels=len(names),
                id2label=dict(enumerate(names)),
                label2id={name: index for index, name in enumerate(names)},
                ignore_mismatched_sizes=True,
            )
            torch.manual_seed(seed)
        model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
            source,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            dtype=torch.float32,
            output_loading_info=True,
            **head,
            **kwargs,
        )
        model = model.to(resolved_device).eval()
        if class_names is not None:
            # mismatched_keys entries are key names (pinned transformers) or (key, old, new) tuples (older).
            reinitialised = sorted(
                {entry if isinstance(entry, str) else entry[0] for entry in info.get("mismatched_keys", [])}
            )
            stuff = frozenset(names.index(name) for name in stuff_names)
            return cls(
                model, processor, resolved_device, tuple(names), stuff, source, False, reinitialised, seed
            )
        names, stuff = _build_class_lists(model.config)
        return cls(model, processor, resolved_device, tuple(names), stuff, source)

    # -- inference ---------------------------------------------------------------------------------

    def _raw(self, rgb: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        import torch

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(pixel_values=inputs["pixel_values"], pixel_mask=inputs.get("pixel_mask"))
        return (
            outputs.class_queries_logits[0].float().cpu().numpy(),
            outputs.masks_queries_logits[0].float().cpu().numpy(),
        )

    def segment(
        self,
        image: Image.Image,
        *,
        score_threshold: float = SCORE_THRESHOLD,
        overlap_threshold: float = OVERLAP_THRESHOLD,
        mask_threshold: float = MASK_THRESHOLD,
    ) -> dict[str, Any]:
        """Panoptic-segment one image; the map and the segment list are at input resolution."""
        rgb, score, overlap, mask = _check_inputs(image, score_threshold, overlap_threshold, mask_threshold)
        class_logits, mask_logits = self._raw(rgb)
        if (
            class_logits.shape != (NUM_QUERIES, len(self.class_names) + 1)
            or mask_logits.shape[0] != NUM_QUERIES
        ):
            raise RuntimeError(
                f"backend returned class logits {class_logits.shape} and mask logits {mask_logits.shape}, "
                f"expected ({NUM_QUERIES}, {len(self.class_names) + 1}) and ({NUM_QUERIES}, h, w)"
            )
        segmentation, segments = panoptic_from_logits(
            class_logits,
            mask_logits,
            (rgb.height, rgb.width),
            label_names=self.class_names,
            stuff_ids=self.stuff_ids,
            score_threshold=score,
            overlap_threshold=overlap,
            mask_threshold=mask,
        )
        return {
            "segmentation": segmentation,
            "segments": segments,
            "void_fraction": float((segmentation == -1).mean()),
            "width": rgb.width,
            "height": rgb.height,
            "score_threshold": score,
            "overlap_threshold": overlap,
            "mask_threshold": mask,
            "class_names": list(self.class_names),
            "adapted": self.adapted,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # -- adaptation --------------------------------------------------------------------------------

    def _encode(self, records: Sequence[Mapping[str, Any]]) -> Any:
        label_index = {name: index for index, name in enumerate(self.class_names)}
        return self.processor(
            images=[validate_image(record["image"]) for record in records],
            segmentation_maps=[np.asarray(record["instance_map"], dtype=np.int32) for record in records],
            instance_id_to_semantic_id=[
                {int(key): label_index[name] for key, name in record["instances"].items()}
                for record in records
            ],
            return_tensors="pt",
        )

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = 0,
        freeze_backbone: bool = True,
        freeze_pixel_decoder: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded gradient fine-tune on validated records with the upstream Mask2Former loss (Hungarian
        matching of queries to instances; class cross-entropy + point-sampled mask BCE and dice, weights
        2 / 5 / 5, no-object weight 0.1, auxiliary decoder losses on). Runs in place; returns the run record
        (hyperparameters, trainable parameter counts, per-epoch mean loss)."""
        manifest = validate_dataset(records, self.class_names, self.stuff_names, epochs=epochs)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= MAX_BATCH_SIZE
        ):
            raise ValueError(
                f"batch_size must be an int in 1..MAX_BATCH_SIZE {MAX_BATCH_SIZE}, got {batch_size!r}"
            )
        if not isinstance(learning_rate, int | float) or not 0.0 < learning_rate <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate!r}")
        import torch

        for name, parameter in self.model.named_parameters():
            frozen = (freeze_backbone and name.startswith("model.pixel_level_module.encoder")) or (
                freeze_pixel_decoder and name.startswith("model.pixel_level_module.decoder")
            )
            parameter.requires_grad_(not frozen)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        n_trainable = sum(parameter.numel() for parameter in trainable)
        n_total = sum(parameter.numel() for parameter in self.model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
        history: list[dict[str, Any]] = []
        started = time.time()
        step = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            if freeze_backbone:
                self.model.model.pixel_level_module.encoder.eval()
            order = np.random.default_rng(seed + epoch).permutation(len(records)).tolist()
            losses = []
            for start in range(0, len(order), batch_size):
                batch = [records[index] for index in order[start : start + batch_size]]
                encoded = self._encode(batch)
                outputs = self.model(
                    pixel_values=encoded["pixel_values"].to(self.device),
                    pixel_mask=encoded["pixel_mask"].to(self.device),
                    mask_labels=[mask.to(self.device) for mask in encoded["mask_labels"]],
                    class_labels=[label.to(self.device) for label in encoded["class_labels"]],
                )
                outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(outputs.loss.detach().cpu()))
                step += 1
            row = {
                "epoch": epoch,
                "steps": step,
                "mean_loss": float(np.mean(losses)),
                "last_loss": losses[-1],
                "seconds": round(time.time() - started, 1),
            }
            history.append(row)
            if progress is not None:
                progress(row)
        self.model.eval()
        self.adapted = True
        self.training = {
            "adaptation": "gradient fine-tuning of the transformer decoder"
            + ("" if freeze_pixel_decoder else " and pixel decoder")
            + " with a re-initialised class head; backbone "
            + ("frozen" if freeze_backbone else "trainable"),
            "loss": "Mask2Former set loss: Hungarian matching; class CE (2.0) + point-sampled mask BCE (5.0) "
            "+ dice (5.0); no-object weight 0.1; auxiliary decoder losses",
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "steps": step,
            "seed": seed,
            "precision": "float32",
            "train_num_points": int(self.model.config.train_num_points),
            "freeze_backbone": freeze_backbone,
            "freeze_pixel_decoder": freeze_pixel_decoder,
            "trainable_parameters": n_trainable,
            "total_parameters": n_total,
            "n_records": len(records),
            "dataset_sha256": manifest["dataset_sha256"],
            "history": history,
            "seconds": round(time.time() - started, 1),
            "device": self.device,
        }
        return dict(self.training)

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        score_threshold: float = SCORE_THRESHOLD,
        overlap_threshold: float = OVERLAP_THRESHOLD,
        mask_threshold: float = MASK_THRESHOLD,
    ) -> dict[str, Any]:
        """Class-aware panoptic quality of the current weights over labelled records (the model's own
        vocabulary must be the records' vocabulary); the same ``segment`` call as inference."""
        validate_dataset(records, self.class_names, self.stuff_names)
        pairs = []
        per_image = []
        for record in records:
            result = self.segment(
                record["image"],
                score_threshold=score_threshold,
                overlap_threshold=overlap_threshold,
                mask_threshold=mask_threshold,
            )
            ref_seg, ref_segments = reference_from_record(record, self.class_names, self.stuff_names)
            pairs.append((result["segmentation"], result["segments"], ref_seg, ref_segments))
            per_image.append(
                {
                    "id": record["id"],
                    "n_predicted": len(result["segments"]),
                    "n_reference": len(ref_segments),
                    "void_fraction": result["void_fraction"],
                }
            )
        quality = panoptic_quality(pairs)
        return {
            **quality,
            "per_image": per_image,
            "score_threshold": score_threshold,
            "overlap_threshold": overlap_threshold,
            "mask_threshold": mask_threshold,
            "adapted": self.adapted,
            "estimation": "held-out records scored once with the request thresholds; no dispersion estimate",
        }

    # -- artifact ----------------------------------------------------------------------------------

    def save_artifact(self, path: str | Path, *, notes: str = "") -> dict[str, Any]:
        """Export the adapted model as a self-describing directory: the Transformers ``config.json`` (with the
        class vocabulary) and ``model.safetensors``, the image processor configuration, and
        ``dimer-adapted-manifest.json`` naming the format, the base identity and the digest of every file."""
        if not self.adapted:
            raise ValueError("save_artifact needs an adapted model; call finetune first")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(root, safe_serialization=True)
        self.processor.save_pretrained(root)
        files = sorted(
            entry.name for entry in root.iterdir() if entry.is_file() and entry.name != ARTIFACT_MANIFEST_NAME
        )
        manifest = {
            "format": ARTIFACT_FORMAT,
            "base": {"modelId": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE},
            "class_names": list(self.class_names),
            "stuff_names": list(self.stuff_names),
            "head_seed": self.seed,
            "training": self.training,
            "notes": notes,
            "files": [
                {"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha256(root / name)}
                for name in files
            ],
        }
        with open(root / ARTIFACT_MANIFEST_NAME, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return {
            "path": str(root),
            "format": ARTIFACT_FORMAT,
            "files": manifest["files"],
            "class_names": manifest["class_names"],
            "stuff_names": manifest["stuff_names"],
            "total_bytes": sum(entry["bytes"] for entry in manifest["files"]),
        }

    @classmethod
    def load_artifact(cls, path: str | Path, device: str | None = None) -> Mask2FormerPanopticPipeline:
        """Fresh reload from an exported directory: refuses a foreign format or base identity and any
        file whose digest differs from the artifact manifest before importing model libraries."""
        root = Path(path)
        manifest_path = root / ARTIFACT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base", {})
        if base.get("modelId") != MODEL_ID or base.get("revision") != MODEL_REVISION:
            raise ValueError(f"artifact base {base} != package pins {MODEL_ID}@{MODEL_REVISION}")
        for entry in manifest["files"]:
            file_path = root / entry["path"]
            if not file_path.is_file():
                raise FileNotFoundError(f"artifact file missing: {file_path}")
            if file_path.stat().st_size != entry["bytes"] or _sha256(file_path) != entry["sha256"]:
                raise ValueError(f"{entry['path']}: digest differs from the artifact manifest")
        names = list(manifest["class_names"])
        stuff = frozenset(names.index(name) for name in manifest.get("stuff_names", []))
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(root, trust_remote_code=False, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            root, trust_remote_code=False, local_files_only=True, dtype=torch.float32
        )
        loaded = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        if loaded != names:
            raise ValueError(f"artifact config vocabulary {loaded} != manifest {names}")
        model = model.to(resolved_device).eval()
        return cls(
            model,
            processor,
            resolved_device,
            tuple(names),
            stuff,
            str(root),
            True,
            [],
            manifest.get("head_seed"),
            manifest.get("training"),
        )
