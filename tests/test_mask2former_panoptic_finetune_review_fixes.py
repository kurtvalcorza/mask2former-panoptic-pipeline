"""Regression tests for the Notebook Review Framework v1 review of mask2former_panoptic_finetune_colab (M2F-M1..M3,
M2F-m1..m6; review PR #8, fixes recorded in docs/reviews/2026-10-02-notebook-review/mask2former_panoptic_finetune_colab_Fixes.md).

Static checks read the committed notebook. Behavioural checks execute the notebook's own Section 6, 7, 8 and 12 cells
with the carried modules as globals and a signature-checked stand-in for the model pipeline (NumPy and Pillow only; no
torch, no model, no network, no google.colab). Stand-in evidence is not pretrained-inference evidence.
"""
# ruff: noqa: E501  -- assertion messages and literals are kept on one line

from __future__ import annotations

import contextlib
import inspect
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

with contextlib.suppress(ImportError):
    import torch  # noqa: F401

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mask2former_panoptic_pipeline import pipeline as pipeline_module  # noqa: E402
from mask2former_panoptic_pipeline import samples as samples_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mask2former_panoptic_finetune_colab.ipynb"
STEM = "mask2former_panoptic_finetune"
REAL = pipeline_module.Mask2FormerPanopticPipeline


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            return next(_source(c) for c in cells[i + 1 :] if c["cell_type"] == "code")
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _code() -> list[str]:
    return [_source(c) for c in _cells() if c["cell_type"] == "code"]


class _StandInPipeline:
    """A stand-in for the carried pipeline class. Every call is bound to the REAL method's signature first, so a
    keyword the real API rejects fails here too. `evaluate` scores a real panoptic_quality: a perfect prediction once
    adapted, an all-first-class prediction before, so the numbers have the real shape without a model."""

    def __init__(self, class_names, stuff_names=(), seed=0):
        self.class_names, self.stuff_names, self.seed = list(class_names), list(stuff_names), seed
        self.adapted, self.device, self.source = False, "cpu", "stand-in"
        self.reinitialised = ["class_predictor.weight", "class_predictor.bias", "criterion.empty_weight"]
        self.epochs_received: list[int] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        inspect.signature(REAL.from_pretrained).bind(*args, **kwargs)
        assert kwargs.get("class_names"), "the notebook re-heads with class_names"
        return cls(kwargs["class_names"], kwargs.get("stuff_names", ()), kwargs.get("seed", 0))

    def evaluate(self, *args, **kwargs):
        inspect.signature(REAL.evaluate).bind(self, *args, **kwargs)
        pairs = []
        for record in args[0]:
            ref_map, ref_segments = pipeline_module.reference_from_record(record, self.class_names, self.stuff_names)
            if self.adapted:
                pairs.append((ref_map, ref_segments, ref_map, ref_segments))
            else:
                pairs.append((np.ones_like(ref_map), [{"id": 1, "label": self.class_names[0], "is_thing": self.class_names[0] not in self.stuff_names}], ref_map, ref_segments))
        return {**pipeline_module.panoptic_quality(pairs), "estimation": "single seeded holdout (stand-in)"}

    def finetune(self, *args, **kwargs):
        inspect.signature(REAL.finetune).bind(self, *args, **kwargs)
        epochs = kwargs["epochs"]
        steps = epochs * -(-len(args[0]) // kwargs["batch_size"])
        self.adapted = True
        self.epochs_received.append(epochs)
        history = [{"epoch": e + 1, "steps": (e + 1) * steps // epochs, "mean_loss": 30.0 / (e + 1), "last_loss": 1.0, "seconds": 0.0} for e in range(epochs)]
        for row in history:
            if kwargs.get("progress"):
                kwargs["progress"](row)
        return {"adaptation": "stand-in", "loss": "stand-in", "optimizer": "AdamW", "learning_rate": kwargs["learning_rate"], "weight_decay": 0.05, "epochs": epochs, "batch_size": kwargs["batch_size"], "steps": steps, "seed": kwargs["seed"], "precision": "float32", "train_num_points": 4096, "freeze_backbone": kwargs["freeze_backbone"], "freeze_pixel_decoder": False, "trainable_parameters": 1, "total_parameters": 2, "seconds": 0.0, "device": "cpu", "history": history}

    def segment(self, image, **kwargs):
        return {"segmentation": np.zeros((image.size[1], image.size[0]), dtype=np.int32), "segments": [], "void_fraction": 1.0}

    def save_artifact(self, *args, **kwargs):
        inspect.signature(REAL.save_artifact).bind(self, *args, **kwargs)
        path = Path(args[0])
        path.mkdir(parents=True, exist_ok=True)
        (path / "stand-in.json").write_text(json.dumps({"class_names": self.class_names, "stuff_names": self.stuff_names}), encoding="utf-8")
        return {"format": "stand-in", "files": ["stand-in.json"]}

    @classmethod
    def load_artifact(cls, *args, **kwargs):
        inspect.signature(REAL.load_artifact).bind(*args, **kwargs)
        spec = json.loads((Path(args[0]) / "stand-in.json").read_text(encoding="utf-8"))
        pipe = cls(spec["class_names"], spec["stuff_names"])
        pipe.adapted = True
        return pipe


def _namespace(tmp_path, monkeypatch) -> dict:
    """The globals of the notebook up to Section 6, with the stand-in pipeline."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    ns: dict = {"__name__": "__main__"}
    for module in (samples_module, pipeline_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    ns.update(Mask2FormerPanopticPipeline=_StandInPipeline, WEIGHTS_DIR=tmp_path / "weights", json=json, os=__import__("os"), time=__import__("time"), np=np, Image=Image, io=io, shutil=__import__("shutil"), Path=Path)
    ns["records"] = samples_module.shape_dataset(24, seed=0)
    ns["EPOCHS"] = 6
    return ns


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)


def _colab(monkeypatch, uploads: dict) -> None:
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: uploads
    colab = types.ModuleType("google.colab")
    colab.files = files
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


# --- M2F-M1 / M2F-m5 / M2F-m6: isolated runtime, records, environment variables ---------------------------------------


def test_m1_two_leading_kernel_cells_and_no_restart_instruction():
    code = _code()
    assert [i for i, src in enumerate(code) if "# dimer: kernel cell" in src] == [0, 1]
    assert "pip install" not in code[0] and "pip install" not in code[1]
    assert "_isolated_environment_ready()" in code[0] and "_route_to_isolated_runtime" in code[1]
    assert "restart the runtime" not in _markdown().lower()


def test_m1_m5_release_records_call_the_t4_run_restart_assisted_and_name_the_open_gates():
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    row = next(line for line in verification.splitlines() if line.startswith("| 2026-09-18 | `9498ad0` / `baa9ab413435`"))
    assert "Completed only after a manual restart" in row and "PASSED" not in row and "RUN1 not met" in row
    assert "neither BYOD branch has been executed in any record" in verification
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    e2e_row = next(line for line in registry.splitlines() if line.startswith("| `mask2former_panoptic_finetune_colab.ipynb`"))
    assert "verified — clean-runtime" not in e2e_row and "only after a manual restart" in e2e_row and "BYOD_DATASET_PATH" in e2e_row
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "the `E2E` notebook completed on a Kaggle Tesla T4 in 258.6 s only after a manual restart" in card
    assert "Both notebooks now build an isolated environment" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_m6_every_environment_variable_a_cell_reads_is_named_in_markdown():
    md = _markdown()
    read = set()
    for src in _code():
        read |= set(re.findall(r"os\.environ\.get\(\s*['\"]([A-Z_]+)['\"]", src))
        read |= set(re.findall(r"os\.environ\[\s*['\"]([A-Z_]+)['\"]\s*\]", src))
    assert {"DIMER_NOTEBOOK_CI_PREINSTALLED", "DIMER_ISOLATED_ENV"} <= read
    missing = sorted(name for name in read if name not in md and name != "DIMER_KERNEL_IS_COLAB")
    assert not missing, missing


# --- M2F-M2: guided layer ------------------------------------------------------------------------------------------------


def test_m2_guided_layer_infrastructure_cells_and_observable_objectives():
    md = _markdown()
    for marker, least in (
        ("**Who this notebook is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict:**", 5),
        ("<details><summary>Check your reasoning</summary>", 7),
        ("## 13. Your turn — change one thing: the number of epochs", 1),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
    ):
        assert md.count(marker) >= least, marker
    assert md.count("<details>") == md.count("</details>")
    cells = [c for c in _cells() if c["cell_type"] == "code"]
    setup = cells[:6]  # install, router, runtime record, 2 carried modules, model
    assert all(c["metadata"].get("cellView") == "form" and _source(c).startswith("# @title Infrastructure: ") for c in setup)
    assert all(c["metadata"].get("cellView") is None for c in cells[6:])
    objectives = md.split("**Learning objectives:**", 1)[1].split("\n\n", 1)[0]
    for verb in ("explain", "read", "predict", "state", "run", "compare", "find", "change"):
        assert f"**{verb}**" in objectives, verb


# --- M2F-M3: a re-run of the fine-tune cell never stacks training ------------------------------------------------------


def test_m3_rerunning_the_finetune_cell_retrains_a_fresh_rehead_with_an_exact_run_record(tmp_path, monkeypatch, capsys):
    ns = _namespace(tmp_path, monkeypatch)
    exec(compile(_code_after("## 6. Split, re-head, and measure the baselines"), "<section 6>", "exec"), ns)
    first = ns["adapter"]
    exec(compile(_code_after("## 7. The bounded fine-tune"), "<section 7>", "exec"), ns)
    assert ns["adapter"] is first and first.epochs_received == [6] and ns["run"]["epochs"] == 6 and ns["run"]["steps"] == 54
    assert "rebuilt_from_snapshot" not in capsys.readouterr().out
    # The natural rerun of the review's P03 (a): change EPOCHS, re-run Sections 5 (here: the value), 7 and 8.
    ns["EPOCHS"] = 2
    exec(compile(_code_after("## 7. The bounded fine-tune"), "<section 7 again>", "exec"), ns)
    second = ns["adapter"]
    assert second is not first and first.epochs_received == [6] and second.epochs_received == [2]
    assert second.seed == ns["SEED"] and second.class_names == list(ns["SHAPE_CLASSES"])
    assert ns["run"]["epochs"] == 2 and ns["run"]["steps"] == 18 and len(ns["run"]["history"]) == 2
    assert "'rebuilt_from_snapshot': True" in capsys.readouterr().out
    exec(compile(_code_after("## 8. Evaluate on the held-out split"), "<section 8>", "exec"), ns)
    assert ns["adapted"]["n_images"] == 6


def test_m3_the_activity_names_the_cells_to_rerun_and_the_expected_direction():
    md = _markdown()
    activity = md.split("## 13. Your turn", 1)[1]
    assert "re-run Section 5, then **Section 7** (the fine-tune) and **Section 8**" in activity
    assert "select Section 6 and choose **Runtime → Run after**" in activity
    assert "Lower." in activity and "PQ 0.59" in activity and "review's CPU probe" in activity
    assert "re-run from Section 6 to start again" not in md  # the old, insufficient instruction


# --- M2F-m3: the horizon-split trivial baseline ---------------------------------------------------------------------------


def test_m3_minor_horizon_split_baseline_is_recorded_beside_all_sky(tmp_path, monkeypatch):
    """Data-only numbers on the default split (real drawn records, NumPy only): all-sky 0.1032, horizon 0.3184."""
    ns = _namespace(tmp_path, monkeypatch)
    exec(compile(_code_after("## 6. Split, re-head, and measure the baselines"), "<section 6>", "exec"), ns)
    assert ns["HORIZON_ROW"] == 160
    assert round(ns["trivial"]["pq"], 4) == 0.1032 and round(ns["horizon"]["pq"], 4) == 0.3184
    assert round(ns["horizon"]["stuff"]["pq"], 3) == 0.796 and ns["horizon"]["things"]["pq"] == 0.0 and ns["trivial"]["things"]["pq"] == 0.0
    assert {k: round(v["pq"], 3) for k, v in ns["horizon"]["per_class"].items() if k in ("sky", "ground")} == {"sky": 0.822, "ground": 0.77}
    exec(compile(_code_after("## 7. The bounded fine-tune"), "<section 7>", "exec"), ns)
    exec(compile(_code_after("## 8. Evaluate on the held-out split"), "<section 8>", "exec"), ns)
    report = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert [b["id"] for b in report["baselines"]] == ["trivial-all-sky", "trivial-horizon-split", "re-headed-before-adaptation"]
    assert report["baselines"][1]["horizon_row"] == 160 and all("things_pq" in b for b in report["baselines"])
    md = _markdown()
    assert "the number any adapted model must clear" not in md
    assert "PQ 0.318" in md and "things PQ" in md.split("## 8. Evaluate", 1)[1].split("## 9.", 1)[0]


# --- M2F-m4: device-qualified numbers, labelled hypothesis, warnings named ------------------------------------------------


def test_m4_per_class_numbers_are_device_qualified_and_warnings_are_named():
    md = _markdown()
    for value in ("0.509", "0.397", "0.938", "0.985", "0.841", "0.951"):
        for match in re.finditer(re.escape(value), md):
            context = md[max(0, match.start() - 220) : match.end() + 60]
            assert "CPU" in context or "T4" in context, (value, context)
    assert "**hypothesis**" in md and "confusable with the ground band's straight edges" not in md
    for warning in ("overflow encountered in exp", "_max_size", "slow-image-processor", "You should probably TRAIN this model"):
        assert warning in md, warning


# --- M2F-m1 / M2F-m2: BYOD refusals, trivial baseline, manifest and report -------------------------------------------------


def _write_dataset(root: Path, n: int = 8, seed: int = 7, drop_json: bool = False, drop_key: str | None = None, drop_image: bool = False) -> dict:
    spec = {"class_names": list(samples_module.SHAPE_CLASSES), "stuff_names": list(samples_module.SHAPE_STUFF), "records": []}
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for record in samples_module.shape_dataset(n, seed=seed):
        record["image"].save(root / "images" / f"{record['id']}.png")
        Image.fromarray(record["instance_map"].astype(np.uint8)).save(root / "masks" / f"{record['id']}.png")
        entry = {"id": record["id"], "image": f"images/{record['id']}.png", "mask": f"masks/{record['id']}.png", "instances": {str(k): v for k, v in record["instances"].items()}}
        if drop_key:
            entry.pop(drop_key)
        spec["records"].append(entry)
    if drop_image:
        (root / "images" / f"{spec['records'][0]['id']}.png").unlink()
    if not drop_json:
        (root / "dataset.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec


def _zip_of(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, f"mine/{path.relative_to(root)}")
    return buffer.getvalue()


def _section12(ns: dict, **fields) -> None:
    source = _code_after("## 12. Optional: your own image, or your own labelled dataset")
    for key, value in fields.items():
        old = next(line for line in source.splitlines() if line.startswith(f"{key} = "))
        source = source.replace(old, f"{key} = {value!r}  # @param", 1)
    exec(compile(source, "<section 12>", "exec"), ns)


def _byod_namespace(tmp_path, monkeypatch) -> dict:
    ns = _namespace(tmp_path, monkeypatch)
    ns.update(HOLDOUT=0.25, SEED=0, BATCH_SIZE=2, LEARNING_RATE=1e-4, FREEZE_BACKBONE=True)
    reloaded = _StandInPipeline(samples_module.SHAPE_CLASSES, samples_module.SHAPE_STUFF)
    reloaded.adapted = True
    ns["reloaded"] = reloaded
    return ns


def test_m2_byod_dataset_by_path_prints_three_scores_and_writes_manifest_and_report(tmp_path, monkeypatch, capsys):
    _no_colab(monkeypatch)
    ns = _byod_namespace(tmp_path, monkeypatch)
    root = tmp_path / "labelled"
    _write_dataset(root)
    _section12(ns, USE_BYOD_DATASET=True, BYOD_EPOCHS=1, BYOD_DATASET_PATH=str(root))
    out = capsys.readouterr().out
    assert "'trivial_pq':" in out and "'baseline_pq':" in out and "'adapted_pq':" in out and "'trivial_predictor': 'all-sky'" in out
    assert "on-disk layout for BYOD_DATASET_PATH" in out and '"mask": "masks/scene-000.png"' in out
    manifest = json.loads((tmp_path / "outputs" / f"{STEM}_byod_dataset_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "outputs" / f"{STEM}_byod_evaluation_report.json").read_text(encoding="utf-8"))
    assert manifest["n_records"] == 8 and manifest["source"]["archive"] == "labelled"
    assert report["dataset_sha256"] == manifest["dataset_sha256"] and len(report["dataset_sha256"]) == 64
    assert [b["id"] for b in report["baselines"]] == ["trivial-all-sky", "re-headed-before-adaptation"]
    assert report["verdict"] == "sample-sanity" and report["run"]["epochs"] == 1 and report["split"] == {"train": 6, "held_out": 2, "holdout": 0.25, "seed": 0}
    assert ns["byod_adapter"].epochs_received == [1] and "google.colab" not in [m for m in sys.modules if sys.modules[m]]


def test_m2_byod_no_improvement_is_said_rather_than_asserted(tmp_path, monkeypatch, capsys):
    _no_colab(monkeypatch)
    ns = _byod_namespace(tmp_path, monkeypatch)
    root = tmp_path / "labelled"
    _write_dataset(root)

    class _Flat(_StandInPipeline):
        def evaluate(self, *args, **kwargs):  # an adaptation that learned nothing scores like the all-first-class predictor
            adapted, self.adapted = self.adapted, False
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                self.adapted = adapted

    ns["Mask2FormerPanopticPipeline"] = _Flat
    _section12(ns, USE_BYOD_DATASET=True, BYOD_EPOCHS=1, BYOD_DATASET_PATH=str(root))
    out = capsys.readouterr().out
    assert "'verdict': 'no-improvement'" in out and "no-improvement: the adapted model did not beat" in out
    assert "BYOD artifact reloads" in out  # export and reload still completed


def test_m1_byod_refusals_name_the_file_the_condition_and_the_fix(tmp_path, monkeypatch):
    _no_colab(monkeypatch)
    ns = _byod_namespace(tmp_path, monkeypatch)
    nojson = tmp_path / "nojson"
    _write_dataset(nojson, n=3, drop_json=True)
    (tmp_path / "nojson.zip").write_bytes(_zip_of(nojson))
    with pytest.raises(ValueError, match=r"^nojson\.zip: no dataset\.json found"):
        _section12(ns, USE_BYOD_DATASET=True, BYOD_DATASET_PATH=str(tmp_path / "nojson.zip"))
    nokey = tmp_path / "nokey"
    _write_dataset(nokey, n=3, drop_key="mask")
    with pytest.raises(ValueError, match=r"^nokey: dataset\.json is missing the key 'mask'.*re-run this cell"):
        _section12(ns, USE_BYOD_DATASET=True, BYOD_DATASET_PATH=str(nokey))
    noimage = tmp_path / "noimage"
    _write_dataset(noimage, n=3, drop_image=True)
    with pytest.raises(ValueError, match=r"^noimage: dataset\.json names a file that is not in the archive .*scene-7-000\.png.*re-run this cell"):
        _section12(ns, USE_BYOD_DATASET=True, BYOD_DATASET_PATH=str(noimage))
    notes = tmp_path / "notes.txt"
    notes.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^notes\.txt: not a zip file"):
        _section12(ns, USE_BYOD_DATASET=True, BYOD_DATASET_PATH=str(notes))
    with pytest.raises(ValueError, match=r"^notes\.txt: not an image Pillow can read \(UnidentifiedImageError\).*re-run this cell"):
        _section12(ns, USE_BYOD_IMAGE=True, BYOD_IMAGE_PATH=str(notes))
    with pytest.raises(FileNotFoundError, match=r"BYOD_DATASET_PATH .*missing.* does not exist"):
        _section12(ns, USE_BYOD_DATASET=True, BYOD_DATASET_PATH=str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="BYOD_IMAGE_PATH is empty and this runtime has no Colab upload dialog"):
        _section12(ns, USE_BYOD_IMAGE=True)
    _colab(monkeypatch, {})
    with pytest.raises(RuntimeError, match=r"got 0 \(upload cancelled or empty\)"):
        _section12(ns, USE_BYOD_DATASET=True)
    _colab(monkeypatch, {"notes.txt": b"hello"})
    with pytest.raises(ValueError, match=r"^notes\.txt: not an image Pillow can read"):
        _section12(ns, USE_BYOD_IMAGE=True)


def test_m1_byod_image_by_path_and_by_upload_is_segmented_by_the_reloaded_model(tmp_path, monkeypatch, capsys):
    _no_colab(monkeypatch)
    ns = _byod_namespace(tmp_path, monkeypatch)
    scene = samples_module.shape_dataset(1, seed=99)[0]["image"]
    scene.save(tmp_path / "scene.png")
    _section12(ns, USE_BYOD_IMAGE=True, BYOD_IMAGE_PATH=str(tmp_path / "scene.png"))
    assert ns["byod_name"] == "scene.png" and ns["byod_image"].size == scene.size and "'void_fraction': 1.0" in capsys.readouterr().out
    buffer = io.BytesIO()
    scene.save(buffer, format="PNG")
    _colab(monkeypatch, {"mine.png": buffer.getvalue()})
    _section12(ns, USE_BYOD_IMAGE=True)
    assert ns["byod_name"] == "mine.png" and ns["byod_image"].size == scene.size
