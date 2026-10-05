"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['mask2former_panoptic_colab', 'mask2former_panoptic_finetune_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'mask2former_panoptic_colab': 5, 'mask2former_panoptic_finetune_colab': 4}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


INF, FT = NOTEBOOKS


def _helper(name: str, marker: str, start: str, end: str) -> str:
    source = _cell(_nb(name), marker)
    return source[source.index(start) : source.index(end)]


# --- SWP-A: no quality assert; the remaining asserts are contract checks ------------------------------------------


def test_swp_a_only_contract_asserts_remain():
    for name in NOTEBOOKS:
        code = "\n".join(c["source"] for c in _code_cells(_nb(name)) if not c["metadata"].get("dimer", {}).get("embedded_module"))
        asserts = [line.strip() for line in code.splitlines() if line.strip().startswith("assert ")]
        # Only reload-parity checks (the artifact reproduces the adapted score / map exactly) are asserted.
        assert all("reloaded" in a or "agreement" in a for a in asserts), asserts


# --- SWP-B: BYOD paths, guarded uploads, named refusals -----------------------------------------------------------


def _image_ns(monkeypatch):
    import io

    from PIL import Image

    helper = _helper(INF, "BYOD_PATH = ''", "def byod_image(", "\n\n\nif USE_BYOD:")
    ns = {"Path": Path, "Image": Image, "io": io}
    exec(helper, ns)
    return ns["byod_image"]


def test_swp_b_inference_byod_path_and_refusals(monkeypatch, tmp_path):
    from PIL import Image

    _no_colab(monkeypatch)
    byod_image = _image_ns(monkeypatch)
    picture = tmp_path / "photo.png"
    Image.new("RGB", (64, 48), "red").save(picture)
    name, image = byod_image(str(picture))
    assert name == "photo.png" and image.size == (64, 48)
    with pytest.raises(FileNotFoundError, match="BYOD_PATH .*missing.png.* is not a file"):
        byod_image(str(tmp_path / "missing.png"))
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^notes\.txt: not a readable image"):
        byod_image(str(notes))
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty and this runtime has no Colab upload dialog"):
        byod_image("")


def test_swp_b_inference_cancelled_or_multiple_uploads_are_refused(monkeypatch):
    import io

    from PIL import Image

    byod_image = _image_ns(monkeypatch)
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="got 0 .*upload cancelled or empty"):
        byod_image("")
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, "PNG")
    _colab(monkeypatch, lambda: {"a.png": buffer.getvalue(), "b.png": buffer.getvalue()})
    with pytest.raises(RuntimeError, match="got 2"):
        byod_image("")
    _colab(monkeypatch, lambda: {"a.png": buffer.getvalue()})
    assert byod_image("")[0] == "a.png"


def test_swp_b_finetune_byod_input_paths_and_uploads(monkeypatch, tmp_path):
    helper = _helper(FT, "BYOD_DATASET_PATH = ''", "def byod_input(", "\n\n\nprint('BYOD dataset schema:'")
    ns = {"Path": Path}
    exec(helper, ns)
    byod_input = ns["byod_input"]
    _no_colab(monkeypatch)
    folder = tmp_path / "labelled"
    folder.mkdir()
    assert byod_input(str(folder), "a dataset", "BYOD_DATASET_PATH") == ("labelled", folder)
    with pytest.raises(FileNotFoundError, match="BYOD_DATASET_PATH .*nope.* does not exist"):
        byod_input(str(tmp_path / "nope"), "a dataset", "BYOD_DATASET_PATH")
    with pytest.raises(RuntimeError, match="BYOD_IMAGE_PATH is empty and this runtime has no Colab upload dialog"):
        byod_input("", "one image file", "BYOD_IMAGE_PATH")
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="got 0 .*upload cancelled or empty"):
        byod_input("", "one image file", "BYOD_IMAGE_PATH")
    _colab(monkeypatch, lambda: {"d.zip": b"PK"})
    assert byod_input("", "a dataset", "BYOD_DATASET_PATH") == ("d.zip", b"PK")


def test_swp_b_finetune_dataset_branch_refuses_a_bad_archive_with_its_name():
    source = _cell(_nb(FT), "BYOD_DATASET_PATH = ''")
    assert "raise ValueError(f'{archive}: not a zip file; give a zip or a directory') from None" in source
    assert "raise ValueError(f'{archive}: no dataset.json found;" in source


def test_swp_g_checkpoint_answers_agree_with_the_recorded_runs():
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    for fact in ("one `stop sign` 0.601 over 8.3 %, void 91.7 %", "PQ 0.33", "two `cat` (0.998, 0.997), two `remote` (0.994, 0.953), `couch` 0.811, void 4.9 %", "baseline PQ 0.0, trivial all-`sky` PQ 0.103", "mean epoch loss 33.6 → 6.7", "adapted held-out PQ 0.866", "box 0.397", "new-seed scenes PQ 0.954", "pixel agreement 1.0"):
        assert fact in record, fact
    inference = "\n".join(c["source"] for c in _nb(INF)["cells"] if c["cell_type"] == "markdown")
    finetune = "\n".join(c["source"] for c in _nb(FT)["cells"] if c["cell_type"] == "markdown")
    for quoted in ("`stop sign` at 0.601", "91.7 % void", "PQ 0.33", "4.9 % void"):
        assert quoted in inference, quoted
    for quoted in ("PQ 0.0, trivial all-`sky` PQ 0.103", "33.6 → 6.7", "PQ 0.866", "0.397 on `box`", "0.954", "pixel agreement 1.0"):
        assert quoted in finetune, quoted
