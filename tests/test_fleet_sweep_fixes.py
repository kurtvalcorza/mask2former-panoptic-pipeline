"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

For the inference notebook the guided layer and BYOD handling are the review PR #7's (M2P-M2, M2P-m1/m2; tested in
tests/test_mask2former_panoptic_review_fixes.py); this file checks what the sweep still adds to it (the /2.2 runtime:
environment reuse by lock digest, an idempotent Section 1) and the fine-tune notebook's own fixes.

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


def _kernel_cells(notebook: dict) -> list[str]:
    """The generator's two leading kernel cells (install or reuse the environment; route later cells to it)."""
    code = _code_cells(notebook)
    kernel = [i for i, c in enumerate(code) if "# dimer: kernel cell" in c["source"]]
    assert kernel == [0, 1], "the two kernel cells must be the first two code cells"
    return [code[0]["source"], code[1]["source"]]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    install, router = _kernel_cells(notebook)
    kernel = install + "\n" + router
    assert "pip install" not in kernel and "'-m', 'pip'" not in kernel  # the kernel cells install nothing into the kernel
    # The record-runtime cell still carries the guarded pip install, which the worker skips (DIMER_NOTEBOOK_CI_PREINSTALLED=1).
    record = _code_cells(notebook)[2]["source"]
    assert "if not SKIP_INSTALL:" in record and "'-m', 'pip', 'install'" in record
    assert "restart the runtime" not in _markdown(notebook).lower()
    for needed in ('"--require-hashes", "--only-binary", ":all:"', '"--managed-python"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    # uv and the worker get a clean interpreter environment and a non-interactive matplotlib backend.
    assert 'uv_env = dict(os.environ, MPLBACKEND="Agg")' in install and '("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")' in install
    assert 'MPLBACKEND="Agg"' in router and '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"' in router
    # Environment reuse keyed on the lock digest, and a worker kept alive across re-runs of the router cell.
    assert '"dimer_isolated_env_" + LOCK_SHA256[:12]' in install and "elif _isolated_environment_ready():" in install
    assert "_previous.alive()" in router and '"worker_reused": _reuse' in router
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _kernel_cells(_nb(name))[0]
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift
    assert build.lock_packages(lock_text)["scipy"] == "1.18.1"  # M2P-M1 (PR #7): Mask2FormerLoss needs scipy


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    install, router = _kernel_cells(_nb(NOTEBOOKS[0]))
    source = install + "\n\n" + router  # Section 1: both kernel cells, run in order
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", install, re.M).group(1)
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
        assert "'worker_reused': True" in capsys.readouterr().out
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([router]) == [router]  # the kernel cells themselves stay in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


# The inference notebook's guided layer is the review PR #7's (M2P-M2); the fine-tune notebook's is the sweep's.
GUIDED_MARKERS = {
    "mask2former_panoptic_colab": ("**Who this is for.**", "**Predict before running:**", "## 10. Your turn — change one thing: the score threshold"),
    "mask2former_panoptic_finetune_colab": ("**Who this notebook is for.**", "**Predict:**", "## 13. Your turn — change one thing: the number of epochs"),
}


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    markdown = _markdown(_nb(name))
    audience, predict, activity = GUIDED_MARKERS[name]
    for heading in (
        audience,
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        activity,
    ):
        assert heading in markdown, heading
    assert markdown.count(predict) >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: pin, stage and verify the model") for c in infra)
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


# The inference notebook's BYOD path and refusals (SWP-B) are PR #7's M2P-m1/m2, tested in
# tests/test_mask2former_panoptic_review_fixes.py.


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
    inference = _markdown(_nb(INF))
    finetune = _markdown(_nb(FT))
    for quoted in ("`stop sign` at score 0.601", "91.7 % void", "PQ 0.33", "4.9 % void"):
        assert quoted in inference, quoted
    for quoted in ("PQ 0.0, trivial all-`sky` PQ 0.103", "33.6 → 6.7", "PQ 0.866", "0.397 on `box`", "0.954", "pixel agreement 1.0"):
        assert quoted in finetune, quoted
