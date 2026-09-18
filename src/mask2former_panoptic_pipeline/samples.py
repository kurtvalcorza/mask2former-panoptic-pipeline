"""Drawn sample data for the panoptic pipeline: the single inference scene and the labelled shape dataset.

Everything here is deterministic Pillow drawing (no text rendering, no randomness beyond a seeded NumPy
generator), so the samples are reproducible byte-for-byte and carry their own exact references: every
region is known because it was drawn. A **panoptic** reference is an instance map (``int32``, one id per
region, ``0`` nowhere) plus a mapping from instance id to class index; ``stuff`` classes are amorphous
regions (sky, ground) and ``thing`` classes are countable objects (disc, box, triangle).

Nothing in this module imports a model library; it is pure NumPy + Pillow so validation runs before any
model dependency is touched (fleet RTM-001).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# The adaptation vocabulary of the drawn dataset: two stuff classes then three thing classes.
SHAPE_CLASSES: tuple[str, ...] = ("sky", "ground", "disc", "box", "triangle")
SHAPE_STUFF: tuple[str, ...] = ("sky", "ground")
# Dataset ceilings enforced by validate_dataset (and therefore by finetune).
MAX_DATASET_IMAGES = 200
MAX_INSTANCES_PER_IMAGE = 50
# Instance ids are pixel values: 0 means no instance and 255 is the pinned processor's ignore index.
MAX_INSTANCE_ID = 254
MIN_CLASSES = 2
MAX_CLASSES = 32
MAX_EPOCHS = 20
MAX_CLASS_NAME_CHARS = 32


def _shape(draw: ImageDraw.ImageDraw, kind: str, geometry: Any, fill: Any) -> None:
    getattr(draw, kind)(geometry, fill=fill)


def synthetic_scene(
    width: int = 640, height: int = 480
) -> tuple[Image.Image, np.ndarray, list[dict[str, Any]]]:
    """The inference-tutorial scene: a red disc, a blue box, a yellow triangle and a green ground band on
    an off-white sky. Returns ``(image, reference_segmentation, reference_segments)`` where the
    segmentation is an ``int32`` map of region ids (1..5, no void) and each segment records its id, a
    region name, whether it is stuff, and its area fraction. Later shapes overdraw earlier ones, and the
    references are computed after overdrawing so they are exact."""
    image = Image.new("RGB", (width, height), (245, 245, 240))
    ids = Image.new("I", (width, height), 1)  # 1 = sky (everything not covered below)
    d, di = ImageDraw.Draw(image), ImageDraw.Draw(ids)
    regions = [
        (2, "ground", True, "rectangle", [0, 320, width, height], (60, 179, 75)),
        (3, "disc", False, "ellipse", [80, 80, 260, 260], (220, 40, 40)),
        (4, "box", False, "rectangle", [340, 90, 560, 300], (40, 70, 200)),
        (5, "triangle", False, "polygon", [(200, 460), (320, 330), (440, 460)], (250, 200, 30)),
    ]
    for region_id, _name, _stuff, kind, geometry, colour in regions:
        _shape(d, kind, geometry, colour)
        _shape(di, kind, geometry, region_id)
    segmentation = np.array(ids, dtype=np.int32)
    segments = [{"id": 1, "name": "sky", "label": "sky", "is_thing": False}] + [
        {"id": region_id, "name": name, "label": name, "is_thing": not stuff}
        for region_id, name, stuff, *_ in regions
    ]
    for segment in segments:
        segment["area_fraction"] = float((segmentation == segment["id"]).mean())
    return image, segmentation, segments


def shape_dataset(
    n_images: int = 24,
    seed: int = 0,
    width: int = 320,
    height: int = 240,
) -> list[dict[str, Any]]:
    """Draw ``n_images`` labelled panoptic scenes over SHAPE_CLASSES with a seeded generator.

    Each record is ``{"id", "image", "instance_map", "instances"}``: the image, an ``int32`` instance
    map (every pixel belongs to exactly one instance) and ``{instance_id: class_name}``. Every scene has
    one ``sky`` region above a jittered horizon, one ``ground`` band below it, and 1–3 things (disc, box
    or triangle) of jittered size, position and colour; a thing drawn over another may hide it entirely,
    in which case the hidden instance is dropped, so instance counts vary from 3 to 5.
    """
    if not 1 <= n_images <= MAX_DATASET_IMAGES:
        raise ValueError(f"n_images {n_images} outside 1..MAX_DATASET_IMAGES {MAX_DATASET_IMAGES}")
    rng = np.random.default_rng(seed)
    records = []
    for index in range(n_images):
        tint = tuple(int(v) for v in rng.integers(225, 250, 3))
        image = Image.new("RGB", (width, height), tint)
        ids = Image.new("I", (width, height), 1)
        d, di = ImageDraw.Draw(image), ImageDraw.Draw(ids)
        horizon = int(rng.integers(int(height * 0.55), int(height * 0.75)))
        ground = (int(rng.integers(40, 90)), int(rng.integers(150, 200)), int(rng.integers(50, 100)))
        d.rectangle([0, horizon, width, height], fill=ground)
        di.rectangle([0, horizon, width, height], fill=2)
        instances: dict[int, str] = {1: "sky", 2: "ground"}
        next_id = 3
        for _ in range(int(rng.integers(1, 4))):
            name = SHAPE_CLASSES[int(rng.integers(2, len(SHAPE_CLASSES)))]
            side = int(rng.integers(40, 90))
            x0 = int(rng.integers(0, width - side))
            y0 = int(rng.integers(0, height - side))
            colour = tuple(int(v) for v in rng.integers(20, 235, 3))
            if name == "disc":
                kind, geometry = "ellipse", [x0, y0, x0 + side, y0 + side]
            elif name == "box":
                kind, geometry = "rectangle", [x0, y0, x0 + side, y0 + int(side * 0.8)]
            else:
                kind, geometry = "polygon", [(x0, y0 + side), (x0 + side // 2, y0), (x0 + side, y0 + side)]
            _shape(d, kind, geometry, colour)
            _shape(di, kind, geometry, next_id)
            instances[next_id] = name
            next_id += 1
        instance_map = np.array(ids, dtype=np.int32)
        present = {int(v) for v in np.unique(instance_map)}
        instances = {k: v for k, v in instances.items() if k in present}
        records.append(
            {
                "id": f"scene-{seed}-{index:03d}",
                "image": image,
                "instance_map": instance_map,
                "instances": instances,
            }
        )
    return records


def record_digest(record: Mapping[str, Any]) -> str:
    """SHA-256 over the RGB pixels, the instance map and the instance→class mapping of one record."""
    digest = hashlib.sha256()
    digest.update(np.asarray(record["image"].convert("RGB"), dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(record["instance_map"], dtype=np.int32).tobytes())
    digest.update(json.dumps({str(k): v for k, v in sorted(record["instances"].items())}).encode("utf-8"))
    return digest.hexdigest()


DATASET_SCHEMA: dict[str, Any] = {
    "record": (
        "{'id': str, 'image': PIL.Image.Image, 'instance_map': int32 array (H, W) with one id per pixel "
        "(0 is not allowed: every pixel belongs to an instance), 'instances': {instance_id: class_name}}"
    ),
    "class_names": f"{MIN_CLASSES}..{MAX_CLASSES} distinct names of at most {MAX_CLASS_NAME_CHARS} chars",
    "stuff_names": "subset of class_names whose regions are amorphous (fused per image); the rest are things",
    "images": [1, MAX_DATASET_IMAGES],
    "instances_per_image": [1, MAX_INSTANCES_PER_IMAGE],
    "instance_ids": [1, MAX_INSTANCE_ID],
    "epochs": [1, MAX_EPOCHS],
    "preprocessing": (
        "images are converted to RGB and resized by the pinned Mask2Former image processor (384x384, aspect "
        "ratio not preserved, ImageNet mean/std); instance maps are resized with nearest-neighbour sampling "
        "to the same size and turned into one binary mask + class label per instance"
    ),
}


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    stuff_names: Sequence[str] = (),
    *,
    epochs: int = 1,
) -> dict[str, Any]:
    """Validation stage for the adaptation path: raise on the first contract violation, otherwise return a
    dataset manifest (schema, class vocabulary, per-class instance counts, digests, findings, verdict).

    A class with no instance anywhere in ``records`` is not an error but a recorded finding: the head can
    be trained without it, yet nothing about that class will have been learned.
    """
    names = list(class_names)
    if not MIN_CLASSES <= len(names) <= MAX_CLASSES:
        raise ValueError(f"class count {len(names)} outside {MIN_CLASSES}..{MAX_CLASSES}")
    if len(set(names)) != len(names):
        raise ValueError("class names must be distinct")
    for name in names:
        if not isinstance(name, str) or not name.strip() or len(name) > MAX_CLASS_NAME_CHARS:
            raise ValueError(
                f"class name {name!r} must be a non-empty str of at most {MAX_CLASS_NAME_CHARS} chars"
            )
    stuff = list(stuff_names)
    unknown = [name for name in stuff if name not in names]
    if unknown:
        raise ValueError(f"stuff names {unknown} are not in class_names")
    if isinstance(records, Mapping) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of record dicts")
    if not 1 <= len(records) <= MAX_DATASET_IMAGES:
        raise ValueError(f"record count {len(records)} outside 1..MAX_DATASET_IMAGES {MAX_DATASET_IMAGES}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be an int in 1..MAX_EPOCHS {MAX_EPOCHS}, got {epochs!r}")
    counts = dict.fromkeys(names, 0)
    ids: set[str] = set()
    digests: list[str] = []
    sizes: set[tuple[int, int]] = set()
    for position, record in enumerate(records):
        if not isinstance(record, Mapping) or not {"id", "image", "instance_map", "instances"} <= set(record):
            raise ValueError(f"record {position}: expected keys id, image, instance_map, instances")
        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"record {position}: id must be a non-empty str")
        if record_id in ids:
            raise ValueError(f"record {position}: duplicate id {record_id!r}")
        ids.add(record_id)
        image = record["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"record {record_id!r}: image must be a PIL.Image.Image")
        instance_map = np.asarray(record["instance_map"])
        if instance_map.ndim != 2 or not np.issubdtype(instance_map.dtype, np.integer):
            raise ValueError(f"record {record_id!r}: instance_map must be a 2-D integer array")
        if instance_map.shape != (image.height, image.width):
            raise ValueError(
                f"record {record_id!r}: instance_map shape {instance_map.shape} != image (H, W) "
                f"{(image.height, image.width)}"
            )
        present = {int(v) for v in np.unique(instance_map)}
        instances = record["instances"]
        if not isinstance(instances, Mapping) or not instances:
            raise ValueError(f"record {record_id!r}: instances must be a non-empty {{id: class}} mapping")
        if not 1 <= len(instances) <= MAX_INSTANCES_PER_IMAGE:
            raise ValueError(f"record {record_id!r}: {len(instances)} instances > MAX_INSTANCES_PER_IMAGE")
        keys = {int(k) for k in instances}
        if any(not 1 <= key <= MAX_INSTANCE_ID for key in keys):
            raise ValueError(
                f"record {record_id!r}: instance ids must lie in 1..MAX_INSTANCE_ID {MAX_INSTANCE_ID}"
            )
        if 0 in present or present != keys:
            raise ValueError(
                f"record {record_id!r}: instance ids in the map {sorted(present)} must equal the mapping's "
                f"{sorted(keys)} and contain no 0 (every pixel must belong to an instance)"
            )
        for key, name in instances.items():
            if name not in counts:
                raise ValueError(
                    f"record {record_id!r}: instance {key} has class {name!r} not in class_names"
                )
            counts[name] += 1
        sizes.add(image.size)
        digests.append(record_digest(record))
    findings = [
        {
            "class": name,
            "verdict": "no-instances",
            "message": f"class {name!r} has no instance in the dataset",
        }
        for name, count in counts.items()
        if count == 0
    ]
    return {
        "schema": dict(DATASET_SCHEMA),
        "class_names": names,
        "stuff_names": stuff,
        "thing_names": [name for name in names if name not in stuff],
        "n_records": len(records),
        "instances_per_class": counts,
        "image_sizes": sorted(sizes),
        "epochs": epochs,
        "dataset_sha256": hashlib.sha256("".join(digests).encode("ascii")).hexdigest(),
        "verdict": "accepted",
        "findings": findings,
    }


def split_records(
    records: Sequence[Mapping[str, Any]], holdout: float = 0.25, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Seeded random split into ``(train, held_out)``; every record is drawn independently so a random
    split is the right kind (SPL3). At least one record lands on each side."""
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be in (0, 1), got {holdout!r}")
    if len(records) < 2:
        raise ValueError("at least two records are needed to split")
    n_held = min(max(1, round(len(records) * holdout)), len(records) - 1)
    order = np.random.default_rng(seed).permutation(len(records))
    held = sorted(order[:n_held].tolist())
    train = sorted(order[n_held:].tolist())
    return [dict(records[i]) for i in train], [dict(records[i]) for i in held]


def load_labelled_dir(root: str | Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Bring-your-own labelled dataset: ``<root>/dataset.json`` with ``class_names``, ``stuff_names`` and
    ``records`` (``{"id", "image", "mask", "instances"}``, paths relative to ``root``); each ``mask`` is a
    PNG whose pixel value is the instance id (mode ``I``, ``I;16`` or ``L``). Returns records ready for
    ``validate_dataset`` plus the two vocabularies."""
    root = Path(root)
    spec = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    records = []
    for entry in spec["records"]:
        image = Image.open(root / entry["image"])
        image.load()
        mask = Image.open(root / entry["mask"])
        instance_map = np.array(mask.convert("I"), dtype=np.int32)
        records.append(
            {
                "id": str(entry["id"]),
                "image": image,
                "instance_map": instance_map,
                "instances": {int(k): str(v) for k, v in entry["instances"].items()},
            }
        )
    return records, list(spec["class_names"]), list(spec.get("stuff_names", []))
