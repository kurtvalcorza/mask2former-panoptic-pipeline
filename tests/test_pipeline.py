# ruff: noqa: E501  -- fixtures and assertions are kept on one line
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mask2former_panoptic_pipeline import (
    ARTIFACT_FORMAT,
    ARTIFACT_MANIFEST_NAME,
    COCO_THING_COUNT,
    DEFAULT_WEIGHTS_DIR,
    INPUT_SIZE,
    MASK_LOGIT_SIZE,
    MASK_THRESHOLD,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    NUM_QUERIES,
    OVERLAP_THRESHOLD,
    SCORE_THRESHOLD,
    Mask2FormerPanopticPipeline,
    mask_bbox,
    panoptic_from_logits,
    panoptic_quality,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "facebook/mask2former-swin-tiny-coco-panoptic"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert INPUT_SIZE == 384 and MASK_LOGIT_SIZE == INPUT_SIZE // 4 and NUM_QUERIES == 100
    assert COCO_THING_COUNT == 80
    assert 0.0 < SCORE_THRESHOLD < 1.0 and 0.0 < OVERLAP_THRESHOLD <= 1.0 and 0.0 < MASK_THRESHOLD < 1.0
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION
        paths = [entry["path"] for entry in data["files"]]
        assert "model.safetensors" in paths and "pytorch_model.bin" not in paths
        config = json.loads((REPO / "weights" / MODEL_KEY / "config.json").read_text(encoding="utf-8"))
        assert len(config["id2label"]) == 133 and config["num_queries"] == NUM_QUERIES
        # the thing/stuff boundary the package hard-codes: the last thing is a COCO object, the first stuff a region
        assert config["id2label"][str(COCO_THING_COUNT - 1)] == "toothbrush"
        assert config["id2label"][str(COCO_THING_COUNT)] == "banner"


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "mask2former"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "mask2former"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == [
        "model.safetensors"
    ]
    assert fetched == ["model.safetensors"]
    assert verify_snapshot(tmp_path)["files"] == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


# --- panoptic post-processing on fabricated decoder outputs ------------------------------------------

LABELS = ["cat", "wall"]  # index 0 thing, index 1 stuff
STUFF = frozenset({1})


def _logits(entries, height=8, width=8, n_queries=4):
    """entries: list of (label_index or None for no-object, logit strength, mask box (x0, y0, x1, y1) or None)."""
    class_logits = np.full((n_queries, len(LABELS) + 1), -6.0)
    mask_logits = np.full((n_queries, height, width), -8.0, dtype=np.float32)
    for query, (label, strength, box) in enumerate(entries):
        class_logits[query, len(LABELS) if label is None else label] = strength
        if box is not None:
            x0, y0, x1, y1 = box
            mask_logits[query, y0:y1, x0:x1] = 8.0
    return class_logits, mask_logits


def test_panoptic_from_logits_assigns_void_below_mask_threshold_and_fuses_stuff():
    class_logits, mask_logits = _logits(
        [
            (0, 6.0, (0, 0, 4, 4)),  # a confident cat in the top-left quadrant
            (1, 5.0, (4, 0, 8, 8)),  # wall on the right half
            (1, 4.0, (0, 4, 4, 8)),  # a second wall query, bottom-left: fused into the first
            (None, 6.0, None),  # no-object query
        ]
    )
    segmentation, segments = panoptic_from_logits(
        class_logits, mask_logits, (8, 8), label_names=LABELS, stuff_ids=STUFF
    )
    assert segmentation.shape == (8, 8) and segmentation.dtype == np.int32
    assert [(s["id"], s["label"], s["is_thing"], s["was_fused"]) for s in segments] == [
        (1, "cat", True, False),
        (2, "wall", False, True),
    ]
    assert (
        (segmentation[:4, :4] == 1).all()
        and (segmentation[:, 4:] == 2).all()
        and (segmentation[4:, :4] == 2).all()
    )
    assert (segmentation != -1).all()
    assert segments[0]["area_fraction"] == pytest.approx(0.25) and segments[0]["bbox"] == [0, 0, 4, 4]
    assert segments[1]["area_fraction"] == pytest.approx(0.75)
    assert 0.0 < segments[0]["score"] <= 1.0 and segments[0]["query"] == 0


def test_panoptic_from_logits_void_and_thresholds():
    # one query only covers the top half; the bottom half has no confident mask -> void
    class_logits, mask_logits = _logits(
        [(0, 6.0, (0, 0, 8, 4)), (None, 6.0, None), (None, 6.0, None), (None, 6.0, None)]
    )
    segmentation, segments = panoptic_from_logits(
        class_logits, mask_logits, (8, 8), label_names=LABELS, stuff_ids=STUFF
    )
    assert len(segments) == 1 and (segmentation[:4] == 1).all() and (segmentation[4:] == -1).all()
    # a score threshold above the query's probability leaves everything void
    segmentation, segments = panoptic_from_logits(
        class_logits, mask_logits, (8, 8), label_names=LABELS, stuff_ids=STUFF, score_threshold=0.999999
    )
    assert segments == [] and (segmentation == -1).all()
    # resampling: decoder maps are resized to the requested size
    segmentation, segments = panoptic_from_logits(
        class_logits, mask_logits, (16, 32), label_names=LABELS, stuff_ids=STUFF
    )
    assert segmentation.shape == (16, 32) and segments[0]["bbox"][:2] == [0, 0]


def test_panoptic_from_logits_overlap_rule_drops_a_query_that_lost_its_pixels():
    # two thing queries claim the same box; the weaker one keeps no pixel and must be dropped (overlap rule)
    class_logits, mask_logits = _logits(
        [(0, 6.0, (0, 0, 8, 8)), (0, 5.0, (0, 0, 8, 8)), (None, 6.0, None), (None, 6.0, None)]
    )
    segmentation, segments = panoptic_from_logits(
        class_logits, mask_logits, (8, 8), label_names=LABELS, stuff_ids=STUFF
    )
    assert len(segments) == 1 and (segmentation == 1).all()


def test_panoptic_from_logits_rejects_bad_shapes():
    with pytest.raises(ValueError, match="same Q"):
        panoptic_from_logits(
            np.zeros((3, 3)), np.zeros((2, 4, 4)), (4, 4), label_names=LABELS, stuff_ids=STUFF
        )
    with pytest.raises(ValueError, match="label_names"):
        panoptic_from_logits(
            np.zeros((2, 4)), np.zeros((2, 4, 4)), (4, 4), label_names=LABELS, stuff_ids=STUFF
        )


# --- panoptic quality ---------------------------------------------------------------------------------


def _pair(pred_boxes, ref_boxes, size=(10, 10), labels=("a",)):
    def build(boxes):
        seg = np.full(size, -1, dtype=np.int32)
        segments = []
        for index, (box, label) in enumerate(boxes):
            x0, y0, x1, y1 = box
            seg[y0:y1, x0:x1] = index + 1
            segments.append({"id": index + 1, "label": label, "is_thing": label != "stuff"})
        return seg, segments

    pred_seg, pred_segments = build(pred_boxes)
    ref_seg, ref_segments = build(ref_boxes)
    return pred_seg, pred_segments, ref_seg, ref_segments


def test_panoptic_quality_perfect_match_and_counts():
    pair = _pair(
        [((0, 0, 5, 5), "a"), ((5, 5, 10, 10), "stuff")], [((0, 0, 5, 5), "a"), ((5, 5, 10, 10), "stuff")]
    )
    quality = panoptic_quality([pair])
    assert quality["pq"] == quality["sq"] == quality["rq"] == 1.0
    assert quality["things"]["n_classes"] == 1 and quality["stuff"]["n_classes"] == 1
    assert quality["per_class"]["a"] == {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 1, "fp": 0, "fn": 0}


def test_panoptic_quality_partial_overlap_false_positive_and_false_negative():
    # references tile the image (no void); prediction: a box shifted so IoU = 20/30 with the first
    # reference, and a small segment inside the second (IoU 4/50 < 0.5: a false positive) while that
    # reference itself goes unmatched
    pair = _pair([((1, 0, 6, 5), "a"), ((8, 8, 10, 10), "a")], [((0, 0, 5, 5), "a"), ((5, 0, 10, 10), "a")])
    quality = panoptic_quality([pair])
    per = quality["per_class"]["a"]
    iou = 20 / 30
    assert per["tp"] == 1 and per["fp"] == 1 and per["fn"] == 1
    assert per["sq"] == pytest.approx(iou) and per["rq"] == pytest.approx(1 / 2)
    assert quality["pq"] == pytest.approx(iou / 2)


def test_panoptic_quality_label_mismatch_versus_class_agnostic():
    pair = _pair([((0, 0, 5, 5), "b")], [((0, 0, 5, 5), "a")])
    strict = panoptic_quality([pair])
    assert strict["pq"] == 0.0 and strict["per_class"]["a"]["fn"] == 1 and strict["per_class"]["b"]["fp"] == 1
    agnostic = panoptic_quality([pair], class_agnostic=True)
    assert agnostic["pq"] == 1.0 and list(agnostic["per_class"]) == ["region"]


def test_panoptic_quality_ignores_reference_void_and_accumulates_over_images():
    pred_seg, pred_segments, ref_seg, ref_segments = _pair([((0, 0, 10, 5), "a")], [((0, 0, 5, 5), "a")])
    # the reference leaves the right half void: the prediction's pixels there are ignored -> IoU 1.0
    quality = panoptic_quality([(pred_seg, pred_segments, ref_seg, ref_segments)])
    assert quality["pq"] == 1.0
    # a predicted segment lying entirely on void is neither matched nor a false positive
    pred_seg2, pred_segments2, _, _ = _pair([((6, 6, 9, 9), "a")], [])
    quality = panoptic_quality([(pred_seg2, pred_segments2, ref_seg, ref_segments)])
    assert quality["per_class"]["a"] == {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": 1}
    two = panoptic_quality([_pair([((0, 0, 5, 5), "a")], [((0, 0, 5, 5), "a")])] * 2)
    assert two["n_images"] == 2 and two["per_class"]["a"]["tp"] == 2
    with pytest.raises(ValueError, match="shapes differ"):
        panoptic_quality([(np.zeros((2, 2), np.int32), [], np.zeros((3, 3), np.int32), [])])


def test_mask_bbox():
    a = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    assert mask_bbox(a) == [2, 2, 6, 6]
    assert mask_bbox(np.zeros((10, 10), bool)) is None


# --- the pipeline over a fabricated backend --------------------------------------------------------------


def _fake_pipeline(entries, calls: list | None = None) -> Mask2FormerPanopticPipeline:
    pipe = Mask2FormerPanopticPipeline(None, None, "cpu", tuple(LABELS), STUFF, "fake")

    def raw(rgb):
        if calls is not None:
            calls.append((rgb.mode, rgb.size))
        return _logits(entries, MASK_LOGIT_SIZE, MASK_LOGIT_SIZE, NUM_QUERIES)

    pipe._raw = raw
    return pipe


def test_segment_output_fields_and_defaults():
    calls: list = []
    pipe = _fake_pipeline([(0, 6.0, (0, 0, 48, 48)), (1, 5.0, (48, 0, 96, 96))], calls)
    result = pipe.segment(Image.new("L", (64, 32)))
    assert calls == [("RGB", (64, 32))]
    assert result["segmentation"].shape == (32, 64) and result["segmentation"].dtype == np.int32
    assert [s["label"] for s in result["segments"]] == ["cat", "wall"]
    assert (result["score_threshold"], result["overlap_threshold"], result["mask_threshold"]) == (
        SCORE_THRESHOLD,
        OVERLAP_THRESHOLD,
        MASK_THRESHOLD,
    )
    assert (result["width"], result["height"]) == (64, 32) and result["adapted"] is False
    assert result["class_names"] == LABELS and 0.0 <= result["void_fraction"] < 0.6
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_segment_thresholds_are_caller_owned_and_validated():
    pipe = _fake_pipeline([(0, 6.0, (0, 0, 48, 48))])
    result = pipe.segment(Image.new("RGB", (32, 32)), score_threshold=0.999999)
    assert result["segments"] == [] and result["void_fraction"] == 1.0
    with pytest.raises(TypeError):
        pipe.segment(np.zeros((30, 40, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.segment(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.segment(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)))
    with pytest.raises(ValueError, match="score_threshold"):
        pipe.segment(Image.new("RGB", (64, 64)), score_threshold=1.5)
    with pytest.raises(ValueError, match="mask_threshold"):
        pipe.segment(Image.new("RGB", (64, 64)), mask_threshold=True)


def test_segment_rejects_malformed_backend_output():
    pipe = Mask2FormerPanopticPipeline(None, None, "cpu", tuple(LABELS), STUFF, "fake")
    pipe._raw = lambda rgb: (np.zeros((NUM_QUERIES, 9)), np.zeros((NUM_QUERIES, 4, 4), np.float32))
    with pytest.raises(RuntimeError, match="expected"):
        pipe.segment(Image.new("RGB", (32, 32)))


def test_from_pretrained_argument_contract(tmp_path):
    """Argument errors are raised before any snapshot or model work."""
    with pytest.raises(ValueError, match="distinct"):
        Mask2FormerPanopticPipeline.from_pretrained(weights_dir=tmp_path, class_names=["a", "a"])
    with pytest.raises(ValueError, match="not in class_names"):
        Mask2FormerPanopticPipeline.from_pretrained(
            weights_dir=tmp_path, class_names=["a", "b"], stuff_names=["c"]
        )
    with pytest.raises(ValueError, match="needs class_names"):
        Mask2FormerPanopticPipeline.from_pretrained(weights_dir=tmp_path, stuff_names=["c"])
    with pytest.raises(ValueError, match="train_num_points"):
        Mask2FormerPanopticPipeline.from_pretrained(weights_dir=tmp_path, train_num_points=0)


def test_load_artifact_refuses_foreign_format_base_and_digest_before_model_imports(
    tmp_path, forbid_model_imports
):
    with pytest.raises(FileNotFoundError, match="artifact manifest"):
        Mask2FormerPanopticPipeline.load_artifact(tmp_path)
    payload = b"{}"
    (tmp_path / "config.json").write_bytes(payload)
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"modelId": MODEL_ID, "revision": MODEL_REVISION},
        "class_names": ["a", "b"],
        "stuff_names": ["a"],
        "files": [{"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(payload).hexdigest()}],
    }

    def write(**changes):
        data = json.loads(json.dumps(manifest))
        data.update(changes)
        (tmp_path / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")

    write(format="other/1")
    with pytest.raises(ValueError, match="artifact format"):
        Mask2FormerPanopticPipeline.load_artifact(tmp_path)
    write(base={"modelId": MODEL_ID, "revision": "0" * 40})
    with pytest.raises(ValueError, match="package pins"):
        Mask2FormerPanopticPipeline.load_artifact(tmp_path)
    write(files=[{"path": "config.json", "bytes": 2, "sha256": "0" * 64}])
    with pytest.raises(ValueError, match="digest"):
        Mask2FormerPanopticPipeline.load_artifact(tmp_path)
    write()
    # a valid manifest reaches the model import, which the fixture turns into the assertion below
    with pytest.raises(AssertionError, match="model dependency imported before rejection"):
        Mask2FormerPanopticPipeline.load_artifact(tmp_path)


def test_save_artifact_needs_an_adapted_model(tmp_path):
    pipe = Mask2FormerPanopticPipeline(None, None, "cpu", tuple(LABELS), STUFF, "fake")
    with pytest.raises(ValueError, match="finetune first"):
        pipe.save_artifact(tmp_path)
