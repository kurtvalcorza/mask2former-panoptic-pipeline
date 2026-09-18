"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; valid snapshots still reach them.
"""

import hashlib
import json

import pytest

from mask2former_panoptic_pipeline.pipeline import (
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    Mask2FormerPanopticPipeline,
)

_CONFIG = json.dumps({"model_type": "mask2former", "num_queries": 100}).encode()


def _snapshot(root, tamper=False):
    (root / "config.json").write_bytes(_CONFIG)
    digest = "0" * 64 if tamper else hashlib.sha256(_CONFIG).hexdigest()
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [{"path": "config.json", "bytes": len(_CONFIG), "sha256": digest}],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_from_pretrained_refuses_without_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        Mask2FormerPanopticPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        Mask2FormerPanopticPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_valid_snapshot_reaches_model_import(tmp_path, forbid_model_imports):
    _snapshot(tmp_path)
    with pytest.raises(AssertionError, match="model dependency imported before rejection"):
        Mask2FormerPanopticPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_rehead_arguments_are_checked_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        Mask2FormerPanopticPipeline.from_pretrained(weights_dir=tmp_path, class_names=["sky", "sky"])
