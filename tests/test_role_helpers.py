"""Role-helper contract: validate_inputs / evaluation_report (inference path) and validate_dataset /
split_records / shape_dataset / synthetic_scene (adaptation path)."""
# ruff: noqa: E501  -- fixtures and assertions are kept on one line

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from mask2former_panoptic_pipeline import (
    DATASET_SCHEMA,
    INPUT_SCHEMA,
    MASK_THRESHOLD,
    MAX_DATASET_IMAGES,
    MAX_EPOCHS,
    MAX_IMAGE_SIDE,
    MAX_INSTANCE_ID,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    OVERLAP_THRESHOLD,
    SCORE_THRESHOLD,
    SHAPE_CLASSES,
    SHAPE_STUFF,
    evaluation_report,
    load_labelled_dir,
    record_digest,
    reference_from_record,
    shape_dataset,
    split_records,
    synthetic_scene,
    validate_dataset,
    validate_inputs,
)


def _image(width: int = 64, height: int = 48) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(segments_boxes, size=(48, 64)):
    seg = np.full(size, -1, dtype=np.int32)
    segments = []
    for index, (box, label) in enumerate(segments_boxes):
        x0, y0, x1, y1 = box
        seg[y0:y1, x0:x1] = index + 1
        segments.append(
            {
                "id": index + 1,
                "label": label,
                "label_id": 0,
                "is_thing": True,
                "score": 0.9,
                "area_fraction": float((seg == index + 1).mean()),
                "bbox": list(box),
                "was_fused": False,
            }
        )
    return {
        "segmentation": seg,
        "segments": segments,
        "score_threshold": SCORE_THRESHOLD,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
    }


# --- inference path ---------------------------------------------------------------------------------


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), names=["scene.png"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["inputs"] == [{"id": "scene.png", "mode": "RGB", "size": [64, 48]}]
    assert (manifest["score_threshold"], manifest["overlap_threshold"], manifest["mask_threshold"]) == (
        SCORE_THRESHOLD,
        OVERLAP_THRESHOLD,
        MASK_THRESHOLD,
    )
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert validate_inputs(_image(), score_threshold=0.8)["score_threshold"] == 0.8
    assert validate_inputs(_image())["inputs"][0]["id"] == "image-0"


def test_validate_inputs_rejects_like_segment() -> None:
    with pytest.raises(TypeError):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8))
    with pytest.raises(ValueError, match="overlap_threshold"):
        validate_inputs(_image(), overlap_threshold=2)
    with pytest.raises(ValueError, match="exactly one entry"):
        validate_inputs(_image(), names=["a", "b"])


def test_evaluation_report_not_measurable_without_reference() -> None:
    report = evaluation_report(_result([((0, 0, 32, 24), "cat")]), sample_kind="public-photo")
    assert report["verdict"] == "not-measurable" and report["metrics"] == []
    assert report["n_segments"] == 1 and report["labels"] == [("cat", 0.9, 0.25)]
    assert report["void_fraction"] == pytest.approx(0.75)
    assert "panoptic quality" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "uncalibrated" in report["decision_rule"]
    json.dumps(report)


def test_evaluation_report_sample_sanity_is_class_agnostic_by_default() -> None:
    result = _result([((0, 0, 32, 24), "stop sign"), ((40, 30, 60, 46), "wall")])
    ref_seg = np.zeros((48, 64), dtype=np.int32)
    ref_seg[:24, :32] = 1
    ref_seg[24:, :] = 2
    reference = (
        ref_seg,
        [{"id": 1, "name": "disc", "is_thing": True}, {"id": 2, "name": "ground", "is_thing": False}],
    )
    report = evaluation_report(result, reference, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    by_id = {entry["id"]: entry for entry in report["metrics"]}
    assert (
        by_id["pq"]["value"] == pytest.approx(0.5)
        and by_id["sq"]["value"] == 1.0
        and by_id["rq"]["value"] == 0.5
    )
    assert by_id["matches"]["value"] == {"region": {"tp": 1, "fp": 1, "fn": 1}}
    assert (
        "regardless of class label" in report["reason"]
        and "not a COCO panoptic benchmark" in report["reason"]
    )
    strict = evaluation_report(result, reference, class_agnostic=False)
    assert strict["panoptic_quality"]["pq"] == 0.0 and "per class" in strict["reason"]
    json.dumps(report)


# --- adaptation path --------------------------------------------------------------------------------


def test_synthetic_scene_references_are_exact_and_deterministic() -> None:
    image, segmentation, segments = synthetic_scene()
    image2, segmentation2, _ = synthetic_scene()
    assert image.tobytes() == image2.tobytes() and (segmentation == segmentation2).all()
    assert image.size == (640, 480) and segmentation.shape == (480, 640) and segmentation.dtype == np.int32
    assert [s["name"] for s in segments] == ["sky", "ground", "disc", "box", "triangle"]
    assert (segmentation >= 1).all() and sorted(np.unique(segmentation).tolist()) == [1, 2, 3, 4, 5]
    assert sum(s["area_fraction"] for s in segments) == pytest.approx(1.0)
    disc = np.asarray(image)[segmentation == 3]
    assert (disc == (220, 40, 40)).all()  # the red disc's pixels are exactly the reference region


def test_shape_dataset_records_validate_and_are_seeded() -> None:
    records = shape_dataset(6, seed=3)
    assert [r["id"] for r in records] == [f"scene-3-{i:03d}" for i in range(6)]
    manifest = validate_dataset(records, SHAPE_CLASSES, SHAPE_STUFF, epochs=2)
    assert manifest["verdict"] == "accepted" and manifest["n_records"] == 6 and manifest["epochs"] == 2
    assert manifest["schema"] == DATASET_SCHEMA and manifest["thing_names"] == ["disc", "box", "triangle"]
    assert manifest["instances_per_class"]["sky"] == 6 and manifest["image_sizes"] == [(320, 240)]
    assert (
        manifest["dataset_sha256"]
        == validate_dataset(shape_dataset(6, seed=3), SHAPE_CLASSES)["dataset_sha256"]
    )
    assert (
        manifest["dataset_sha256"]
        != validate_dataset(shape_dataset(6, seed=4), SHAPE_CLASSES)["dataset_sha256"]
    )
    for record in records:
        assert 3 <= len(record["instances"]) <= 5 and 0 not in np.unique(record["instance_map"])
    ref_seg, ref_segments = reference_from_record(records[0], SHAPE_CLASSES, SHAPE_STUFF)
    assert ref_seg is not None and ref_segments[0] == {
        "id": 1,
        "label": "sky",
        "label_id": 0,
        "is_thing": False,
        "area_fraction": pytest.approx(float((ref_seg == 1).mean())),
    }
    with pytest.raises(ValueError, match="MAX_DATASET_IMAGES"):
        shape_dataset(MAX_DATASET_IMAGES + 1)


def test_validate_dataset_rejects_contract_violations_and_records_findings() -> None:
    records = shape_dataset(3, seed=0)
    with pytest.raises(ValueError, match="class count"):
        validate_dataset(records, ["only"])
    with pytest.raises(ValueError, match="distinct"):
        validate_dataset(records, ["a", "a"])
    with pytest.raises(ValueError, match="not in class_names"):
        validate_dataset(records, SHAPE_CLASSES, ["water"])
    with pytest.raises(ValueError, match="MAX_EPOCHS"):
        validate_dataset(records, SHAPE_CLASSES, epochs=MAX_EPOCHS + 1)
    with pytest.raises(ValueError, match="not in class_names"):
        validate_dataset(records, ["sky", "ground"])  # a thing instance names a class outside the vocabulary
    broken = dict(records[0])
    broken["instance_map"] = np.zeros_like(records[0]["instance_map"])
    with pytest.raises(ValueError, match="contain no 0"):
        validate_dataset([broken], SHAPE_CLASSES)
    wrong_shape = dict(records[0])
    wrong_shape["instance_map"] = records[0]["instance_map"][:-1]
    with pytest.raises(ValueError, match="shape"):
        validate_dataset([wrong_shape], SHAPE_CLASSES)
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], records[0]], SHAPE_CLASSES)
    big = dict(records[0])
    big["instance_map"] = np.where(
        records[0]["instance_map"] == 1, MAX_INSTANCE_ID + 1, records[0]["instance_map"]
    )
    big["instances"] = {MAX_INSTANCE_ID + 1 if k == 1 else k: v for k, v in records[0]["instances"].items()}
    with pytest.raises(ValueError, match="MAX_INSTANCE_ID"):
        validate_dataset([big], SHAPE_CLASSES)
    # an unused class is a finding, not a rejection
    manifest = validate_dataset(records, [*SHAPE_CLASSES, "hexagon"], SHAPE_STUFF)
    assert manifest["findings"] == [
        {
            "class": "hexagon",
            "verdict": "no-instances",
            "message": "class 'hexagon' has no instance in the dataset",
        }
    ]


def test_split_records_is_seeded_disjoint_and_bounded() -> None:
    records = shape_dataset(8, seed=1)
    train, held = split_records(records, holdout=0.25, seed=0)
    assert len(train) == 6 and len(held) == 2
    assert {r["id"] for r in train}.isdisjoint({r["id"] for r in held})
    assert [r["id"] for r in held] == [r["id"] for r in split_records(records, holdout=0.25, seed=0)[1]]
    assert [r["id"] for r in held] != [r["id"] for r in split_records(records, holdout=0.25, seed=5)[1]]
    train, held = split_records(records, holdout=0.99, seed=0)
    assert len(train) == 1 and len(held) == 7
    with pytest.raises(ValueError, match="holdout"):
        split_records(records, holdout=1.0)
    with pytest.raises(ValueError, match="at least two"):
        split_records(records[:1])


def test_load_labelled_dir_round_trips_a_bring_your_own_dataset(tmp_path) -> None:
    records = shape_dataset(2, seed=7)
    spec = {"class_names": list(SHAPE_CLASSES), "stuff_names": list(SHAPE_STUFF), "records": []}
    for record in records:
        record["image"].save(tmp_path / f"{record['id']}.png")
        Image.fromarray(record["instance_map"].astype(np.uint8), mode="L").save(
            tmp_path / f"{record['id']}_mask.png"
        )
        spec["records"].append(
            {
                "id": record["id"],
                "image": f"{record['id']}.png",
                "mask": f"{record['id']}_mask.png",
                "instances": {str(k): v for k, v in record["instances"].items()},
            }
        )
    (tmp_path / "dataset.json").write_text(json.dumps(spec), encoding="utf-8")
    loaded, class_names, stuff_names = load_labelled_dir(tmp_path)
    assert class_names == list(SHAPE_CLASSES) and stuff_names == list(SHAPE_STUFF)
    assert [record_digest(r) for r in loaded] == [record_digest(r) for r in records]
    assert validate_dataset(loaded, class_names, stuff_names)["verdict"] == "accepted"
