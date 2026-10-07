"""Regression tests for the Notebook Review Framework v1 review of mask2former_panoptic_colab (M2P-M1..M2, M2P-m1..m5).

Static checks read the committed notebook; behavioural checks execute the notebook's own Section 4, 9 and 10 code
with the carried package modules as globals (NumPy and Pillow only; no model, no network, no google.colab).
"""
# ruff: noqa: E501  -- assertion messages and literals are kept on one line

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import types
from pathlib import Path

import pytest

# Windows conda trap (fleet note, bioclip2 row 6): import torch before anything else touches NumPy in this process.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mask2former_panoptic_pipeline import pipeline as pipeline_module  # noqa: E402
from mask2former_panoptic_pipeline import samples as samples_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mask2former_panoptic_colab.ipynb"
FINETUNE_NOTEBOOK = ROOT / "tutorials" / "mask2former_panoptic_finetune_colab.ipynb"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_m2p_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells(path: Path = NOTEBOOK) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            return next(_source(c) for c in cells[i + 1 :] if c["cell_type"] == "code")
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown(path: Path = NOTEBOOK) -> str:
    return "\n".join(_source(c) for c in _cells(path) if c["cell_type"] == "markdown")


def _code(path: Path = NOTEBOOK) -> list[str]:
    return [_source(c) for c in _cells(path) if c["cell_type"] == "code"]


def _namespace() -> dict:
    """The globals the carried module cells define in the notebook."""
    ns: dict = {}
    for module in (samples_module, pipeline_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    return ns


def _section4(monkeypatch, tmp_path, **fields) -> dict:
    """Execute Section 4 verbatim with USE_BYOD = True and the given form values substituted."""
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. Draw the synthetic scene or optional BYOD")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    for key, value in fields.items():
        old = f'{key} = ""'
        assert old in source, key
        source = source.replace(old, f"{key} = {value!r}", 1)
    ns = _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _fake_colab(monkeypatch, uploads: dict) -> None:
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: uploads
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _photo() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))


# --- M2P-M1: isolated environment, no in-kernel install, no restart ----------------------------------------------


def test_exactly_two_leading_kernel_cells_and_a_carried_lock_matching_the_repository():
    code = _code()
    kernel = [i for i, src in enumerate(code) if "# dimer: kernel cell" in src]
    assert kernel == [0, 1]
    install = code[0]
    assert '"--require-hashes", "--only-binary", ":all:"' in install and '"--managed-python"' in install
    lock = re.search(r"^LOCK_TEXT = r'''(.*?)'''$", install, re.M | re.S).group(1)
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", install, re.M).group(1)
    assert hashlib.sha256(lock.encode("utf-8")).hexdigest() == digest
    assert lock == (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    build = _load_tool("build_notebook")
    template = _load_tool("notebook_template").TEMPLATE
    build.check_lock(build._pins(ROOT, template), lock)  # every pin at its version, every entry hashed
    assert len(build.lock_packages(lock)) == 48
    md = _markdown()
    assert "Restart the runtime" not in md and "installs the pinned dependencies" not in md


def test_pins_file_is_the_pyproject_pins_plus_scipy():
    build = _load_tool("build_notebook")
    template = _load_tool("notebook_template").TEMPLATE
    assert template["pins_file"] == "tutorials/requirements-colab.in"
    assert build._pins(ROOT, template) == build._pins(ROOT) + ["scipy==1.18.1"]
    # The E2E notebook shares the pins file and the lock (fleet sweep SWP-R): it builds the same model class.
    assert build._pins(ROOT, _load_tool("notebook_template_finetune").TEMPLATE) == build._pins(ROOT, template)


def test_lock_covers_every_backend_the_model_classes_require():
    """The isolated environment sees only the lock: every `requires_backends` / `is_<x>_available()` import guard in
    the transformers modules the notebook builds (Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, Swin)
    must name a locked distribution (Colab T4 run of fe62ef4: Mask2FormerLoss raised for scipy)."""
    transformers = pytest.importorskip("transformers")
    build = _load_tool("build_notebook")
    locked = build.lock_packages((ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8"))
    models = Path(transformers.__file__).parent / "models"
    text = "".join(p.read_text(encoding="utf-8") for p in sorted((models / "mask2former").glob("*.py")) + [models / "swin" / "modeling_swin.py"])
    required = set()
    for args in re.findall(r"requires_backends\(\s*\w+\s*,\s*\[([^\]]*)\]", text):
        required |= set(re.findall(r"['\"]([\w-]+)['\"]", args))
    assert "scipy" in required
    optional = {"accelerate"}  # guarded by is_accelerate_available() only for multi-device loss reduction
    missing = sorted(name for name in required - optional if name.lower() not in locked)
    assert not missing, missing
    assert locked["scipy"] == "1.18.1"


def test_both_notebooks_render_with_the_isolated_runtime_generator():
    """Superseded in part: the E2E notebook adopted the isolated runtime in the fleet-sweep fixes (SWP-R), so both
    templates now record the same generator label and both start with the two kernel cells."""
    build = _load_tool("build_notebook")
    finetune = _load_tool("notebook_template_finetune").TEMPLATE
    assert build.generator_version(finetune) == build.generator_version(_load_tool("notebook_template").TEMPLATE) == "build_notebook.py/2.2"
    assert build.generator_version({}) == "build_notebook.py/2"  # a template that opts into nothing still renders as /2
    assert [i for i, src in enumerate(_code(FINETUNE_NOTEBOOK)) if "# dimer: kernel cell" in src] == [0, 1]


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: libraries such as accelerate call importlib.util.find_spec("google.colab"); a spec-less stub raises."""
    router = _code()[1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


def test_release_records_call_the_kaggle_run_restart_assisted():
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    row = next(line for line in verification.splitlines() if line.startswith("| 2026-09-18 | `9498ad0` / `1b0645e8806a`"))
    assert "Completed only after a manual restart" in row and "PASSED" not in row
    assert "requires scipy" in verification and "No hosted run of the current" not in verification
    hosted = [line for line in verification.splitlines() if line.startswith("| 2026-10-0")]
    assert any("`fe62ef4` / `0ae808e38df7`" in line and "**FAILED**" in line for line in hosted)
    assert any("`d2aa5071ba92`" in line and "**PASSED**" in line and "no restart" in line for line in hosted)
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    inference_row = next(line for line in registry.splitlines() if line.startswith("| `mask2former_panoptic_colab.ipynb`"))
    assert "verified — clean-runtime" not in inference_row and "only after a manual restart" in inference_row
    for rel in ("README.md", "STATUS.md", "MODEL_CARD.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "only after a manual restart" in text and "both PASSED" not in text, rel


# --- M2P-M2 / M2P-m5: guided layer, infrastructure labels, environment variables -----------------------------------


def test_guided_layer_and_infrastructure_labels():
    md = _markdown()
    for marker, least in (
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 5),
        ("**What to notice:**", 6),
        ("<details><summary>Check your reasoning</summary>", 6),
        ("## 10. Your turn — change one thing: the score threshold", 1),
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
    for verb in ("explain", "read", "predict", "state", "change", "run", "find"):
        assert verb in objectives, verb


def test_every_environment_variable_a_cell_reads_is_named_in_markdown():
    md = _markdown()
    read = set()
    for src in _code():
        read |= set(re.findall(r"os\.environ\.get\(\s*['\"]([A-Z_]+)['\"]", src))
        read |= set(re.findall(r"os\.environ\[\s*['\"]([A-Z_]+)['\"]\s*\]", src))
    assert {"DIMER_NOTEBOOK_CI_PREINSTALLED", "DIMER_ISOLATED_ENV"} <= read
    missing = sorted(name for name in read if name not in md and name != "DIMER_KERNEL_IS_COLAB")
    assert not missing, missing


def test_worked_answers_quote_the_recorded_runs_with_their_environment():
    md = _markdown()
    assert md.count("the recorded Kaggle CPU run of 18 September 2026") >= 4 and "torch 2.14.0" in md
    for number in ("0.601", "91.7 %", "0.798", "4.9 %", "PQ 0.33", "2.8 %", "0.811", "0.917 to 0.916"):
        assert number in md, number
    assert "torch 2.13.0+cpu" in md  # the Section 10 answer names the CPU check it quotes


def test_no_learner_cell_uses_a_bare_assert():
    learner = [_source(c) for c in _cells() if c["cell_type"] == "code" and "embedded_module" not in c["metadata"].get("dimer", {}) and "# dimer: kernel cell" not in _source(c)]
    assert not any(line.lstrip().startswith("assert ") for src in learner for line in src.splitlines())


# --- M2P-m1 / M2P-m2: BYOD path, refusals, EXIF orientation, high-bit-depth modes -----------------------------------


def test_jpeg_by_path_with_exif_orientation_is_rotated_upright_and_recorded(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    sideways = _photo().transpose(Image.Transpose.ROTATE_90)  # stored 48x64, displays 64x48
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "phone.jpg"
    sideways.save(path, exif=exif.tobytes(), quality=95)
    ns = _section4(monkeypatch, tmp_path, BYOD_PATH=str(path))
    assert ns["image"].size == (64, 48) and ns["sample_kind"] == "BYOD" and ns["reference"] is None
    (finding,) = ns["input_adjustments"]
    assert finding["verdict"] == "adjusted" and "EXIF orientation 6 applied" in finding["message"] and "48x64 stored, 64x48 segmented" in finding["message"]
    assert "google.colab" not in sys.modules  # a path never touches the upload dialog


def test_upright_png_by_path_has_no_adjustment(monkeypatch, tmp_path):
    path = tmp_path / "plain.png"
    _photo().save(path)
    ns = _section4(monkeypatch, tmp_path, BYOD_PATH=str(path))
    assert ns["input_adjustments"] == [] and ns["image"].size == (64, 48)


@pytest.mark.parametrize("mode", ["I;16", "I", "F"])
def test_high_bit_depth_images_are_refused_naming_the_mode(monkeypatch, tmp_path, mode):
    array = np.linspace(0, 60000, 48 * 64).reshape(48, 64)
    image = Image.fromarray(array.astype(np.uint16)) if mode == "I;16" else Image.fromarray(array.astype(np.int32 if mode == "I" else np.float32))
    assert image.mode == mode
    path = tmp_path / "deep.tif"
    image.save(path)
    with pytest.raises(ValueError, match=rf"has image mode '{re.escape(mode)}' \(16- or 32-bit samples\).*Save it as an 8-bit"):
        _section4(monkeypatch, tmp_path, BYOD_PATH=str(path))


def test_unreadable_file_missing_path_cancelled_upload_and_no_colab_are_actionable(monkeypatch, tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match=r"notes\.txt is not an image Pillow can read .*16-4096 px.*re-run from this cell"):
        _section4(monkeypatch, tmp_path, BYOD_PATH=str(text))
    with pytest.raises(FileNotFoundError, match="is not a file in this runtime"):
        _section4(monkeypatch, tmp_path, BYOD_PATH=str(tmp_path / "nope.jpg"))
    _fake_colab(monkeypatch, {})
    with pytest.raises(ValueError, match=r"Upload exactly one image \(received 0: nothing: the upload was cancelled or empty\)"):
        _section4(monkeypatch, tmp_path)
    _fake_colab(monkeypatch, {"a.png": b"1", "b.png": b"2"})
    with pytest.raises(ValueError, match=r"received 2: a\.png, b\.png"):
        _section4(monkeypatch, tmp_path)
    _fake_colab(monkeypatch, {"notes.txt": b"not an image"})
    with pytest.raises(ValueError, match="is not an image Pillow can read"):
        _section4(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(RuntimeError, match="needs the Google Colab upload dialog. Elsewhere, set BYOD_PATH"):
        _section4(monkeypatch, tmp_path)


def test_one_uploaded_image_goes_through(monkeypatch, tmp_path):
    buffer = io.BytesIO()
    _photo().save(buffer, format="PNG")
    _fake_colab(monkeypatch, {"mine.png": buffer.getvalue()})
    ns = _section4(monkeypatch, tmp_path)
    assert ns["image_name"] == "mine.png" and ns["image"].size == (64, 48)


def test_section5_records_the_adjustment_in_the_input_manifest():
    section5 = _code_after("## 5. Validate the request → input manifest")
    assert "input_manifest['findings'].extend(input_adjustments)" in section5
    assert section5.index("extend(input_adjustments)") < section5.index("json.dump(input_manifest")


# --- M2P-m3: the activity names the cells to re-run, and the export refuses mixed thresholds ------------------------


def _export_guard() -> str:
    source = _code_after("## 9. Export outputs and provenance")
    return source.split("def save_maps", 1)[0]


def test_export_refuses_results_made_with_other_thresholds():
    form = {"score_threshold": 0.8, "overlap_threshold": 0.8, "mask_threshold": 0.5}
    old = {"score_threshold": 0.5, "overlap_threshold": 0.8, "mask_threshold": 0.5}
    ns = {**form, "result": dict(old), "photo_result": dict(old), "report": {"thresholds": dict(old)}}
    with pytest.raises(RuntimeError, match=r"Section 6 map, Section 7 photograph, Section 8 report used other thresholds .*Run after"):
        exec(compile(_export_guard(), "<section 9 guard>", "exec"), ns)
    ns = {**form, "result": dict(form), "photo_result": None, "report": {"thresholds": dict(form)}}
    exec(compile(_export_guard(), "<section 9 guard>", "exec"), ns)  # consistent: no error


def test_activity_names_the_run_after_scope_and_drops_the_false_claim():
    md = _markdown()
    assert "Select the Section 4 cell and choose **Runtime → Run after**" in md
    assert "Re-running a single section is not enough" in md
    assert "lose and gain segments" not in md and "see void shrink" not in md
    assert "no query scores between 0.3 and 0.5" in md


def test_activity_table_appends_one_row_per_pass(capsys):
    source = _code_after("## 10. Your turn — change one thing: the score threshold")
    ns = {
        "score_threshold": 0.5, "mask_threshold": 0.5, "overlap_threshold": 0.8, "sample_kind": "synthetic",
        "result": {"segments": [{}], "void_fraction": 0.9171}, "photo_result": {"segments": [{}] * 5, "void_fraction": 0.0491},
        "report": {"metrics": [{"id": "pq", "value": 0.33041}]},
    }
    exec(compile(source, "<section 10>", "exec"), ns)
    exec(compile(source, "<section 10>", "exec"), ns)  # the same pass twice: one row
    ns.update(score_threshold=0.8, result={"segments": [], "void_fraction": 1.0}, report={"metrics": [{"id": "pq", "value": 0.0}]})
    exec(compile(source, "<section 10>", "exec"), ns)
    assert [row["score"] for row in ns["activity_runs"]] == [0.5, 0.8]
    assert ns["activity_runs"][0]["pq"] == 0.33 and ns["activity_runs"][1]["segments"] == 0
    assert "photo_void" in capsys.readouterr().out
