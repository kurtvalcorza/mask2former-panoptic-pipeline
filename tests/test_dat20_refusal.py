"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebooks:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells (inference Section 4, fine-tune Section 12) turn every user-input problem into
  `BYODRefusal` and remove the earlier run's results and `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

from mask2former_panoptic_pipeline import pipeline as pipeline_module
from mask2former_panoptic_pipeline import samples as samples_module
from test_mask2former_panoptic_finetune_review_fixes import _byod_namespace, _section12

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "tutorials" / "mask2former_panoptic_colab.ipynb"
FINETUNE = ROOT / "tutorials" / "mask2former_panoptic_finetune_colab.ipynb"
INFERENCE_BYOD_CELL = "mask2former_panoptic-15"
FINETUNE_BYOD_CELL = "mask2former_panoptic_finetune-32"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(path: Path, cell_id: str) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(path: Path, start: str) -> dict:
    source = next(_source(c) for c in _code_cells(path) if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(path: Path, messages):
    namespace = _router_namespace(path, "class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


@pytest.mark.parametrize("path", [INFERENCE, FINETUNE], ids=["inference", "finetune"])
def test_router_relays_a_refusal_as_one_line_without_a_traceback(path, capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime(path, [("refusal", "hostname: not an image Pillow can read")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not an image Pillow can read"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


@pytest.mark.parametrize("path", [INFERENCE, FINETUNE], ids=["inference", "finetune"])
def test_router_keeps_the_traceback_of_a_genuine_error(path, capsys):
    namespace, runtime = _kernel_runtime(path, [("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


@pytest.mark.parametrize("path", [INFERENCE, FINETUNE], ids=["inference", "finetune"])
def test_router_normal_cell_output_is_unchanged(path, capsys):
    _, runtime = _kernel_runtime(path, [("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    for path in (INFERENCE, FINETUNE):
        source = next(_source(c) for c in _code_cells(path) if '_WORKER_SOURCE = r"""' in _source(c))
        assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
        assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace(INFERENCE, '_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not an image')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not an image"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


# --- inference Section 4 ------------------------------------------------------------------------------------------

EARLIER = {"image": "old", "image_name": "old.png", "input_manifest": "old", "result": "old", "report": "old", "payload": "old", "photo_result": "old"}
EARLIER_FILES = ("mask2former_panoptic_result.json", "mask2former_panoptic_byod_overlay.png", "mask2former_panoptic_input_manifest.json")
KEPT_FILE = "mask2former_panoptic_finetune_result.json"  # the other notebook's export is not this run's result


def _section4(monkeypatch, tmp_path, path: str, source: str | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    for name in (*EARLIER_FILES, KEPT_FILE):
        (outputs / name).write_text("{}", encoding="utf-8")
    source = (source or _cell(INFERENCE, INFERENCE_BYOD_CELL)).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace('BYOD_PATH = ""', f"BYOD_PATH = {path!r}", 1)
    ns: dict = {}
    for module in (samples_module, pipeline_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    ns.update(EARLIER, activity_runs=["kept"])
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _image(path: Path, size=(64, 48), mode="RGB") -> str:
    Image.new(mode, size).save(path)
    return str(path)


def _inference_cases(tmp_path: Path) -> dict:
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime\n", encoding="utf-8")
    deep = tmp_path / "deep.png"
    Image.new("I;16", (32, 32)).save(deep)
    return {
        "plain text file (the scripted /etc/hostname run)": (str(hostname), "hostname", "not an image Pillow can read"),
        "folder": (str(tmp_path), tmp_path.name, "is not a file in this runtime"),
        "missing path": (str(tmp_path / "nope.jpg"), "nope.jpg", "is not a file in this runtime"),
        "16-bit image": (str(deep), "deep.png", "image mode 'I;16'"),
        "too small": (_image(tmp_path / "tiny.png", (8, 8)), "tiny.png", "8x8 px, outside 16-4096 px per side"),
    }


@pytest.mark.parametrize("case", ["plain text file (the scripted /etc/hostname run)", "folder", "missing path", "16-bit image", "too small"])
def test_inference_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    path, file_name, rule = _inference_cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, path)
    exc = caught.value
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert line.startswith(f"Refused: {file_name}: ") and rule in line and "\n" not in line and "Traceback" not in line
    assert "re-run from this cell" in line and "earlier results of this session and its outputs/ files were removed" in line
    outputs = sorted(p.name for p in (tmp_path / "outputs").iterdir())
    assert outputs == [KEPT_FILE], outputs


def test_inference_refusal_clears_the_namespace(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    source = _cell(INFERENCE, INFERENCE_BYOD_CELL).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace('BYOD_PATH = ""', "BYOD_PATH = 'notes.txt'", 1)
    ns: dict = {k: v for m in (samples_module, pipeline_module) for k, v in vars(m).items() if not k.startswith("__")}
    ns.update(EARLIER, activity_runs=["kept"])
    with pytest.raises(ValueError, match="^notes.txt: not an image"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["activity_runs"] == ["kept"]


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "received 0"), ({"a.png": b"1", "b.png": b"2"}, "a.png, b.png", "received 2")])
def test_inference_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    files = types.SimpleNamespace(upload=lambda: uploaded)
    colab = types.ModuleType("google.colab")
    colab.files = files
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    (line,) = caught.value._render_traceback_()
    assert line.startswith(f"Refused: {file_name}: Upload exactly one image") and rule in line


def test_inference_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    source = _cell(INFERENCE, INFERENCE_BYOD_CELL).replace("def byod_image(name, data):\n", "def byod_image(name, data):\n    raise RuntimeError('internal bug')\n", 1)
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, _image(tmp_path / "ok.png"), source)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / EARLIER_FILES[0]).exists()  # nothing is removed for an internal error


def test_inference_good_image_still_loads(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _section4(monkeypatch, tmp_path, _image(tmp_path / "ok.png"))
    assert ns["image_name"] == "ok.png" and ns["image"].size == (64, 48) and ns["sample_kind"] == "BYOD"


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc/hostname': a text file present in every fresh runtime."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    (line,) = caught.value._render_traceback_()
    assert line.startswith("Refused: hostname: not an image Pillow can read (UnidentifiedImageError)")


# --- fine-tune Section 12 -----------------------------------------------------------------------------------------

BYOD_EARLIER = {"byod_result": "old", "byod_manifest": "old", "byod_evaluation": "old", "byod_adapted": "old", "byod_reloaded": "old"}


def _seed_byod(ns: dict, tmp_path: Path) -> None:
    ns.update(BYOD_EARLIER, adapted="sample run kept")
    (tmp_path / "outputs" / "mask2former_panoptic_finetune_byod_evaluation_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outputs" / "mask2former-panoptic-byod-adapted").mkdir()
    (tmp_path / "outputs" / "mask2former_panoptic_finetune_result.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize(
    ("fields", "file_name", "rule"),
    [
        ({"USE_BYOD_IMAGE": True, "BYOD_IMAGE_PATH": "hostname"}, "hostname", "not an image Pillow can read"),
        ({"USE_BYOD_DATASET": True, "BYOD_DATASET_PATH": "hostname"}, "hostname", "not a zip file"),
        ({"USE_BYOD_DATASET": True, "BYOD_DATASET_PATH": "empty"}, "empty", "no dataset.json found"),
        ({"USE_BYOD_DATASET": True, "BYOD_DATASET_PATH": "nope.zip"}, "nope.zip", "does not exist"),
        ({"USE_BYOD_IMAGE": True, "BYOD_IMAGE_PATH": "tiny.png"}, "tiny.png", "ValueError"),
    ],
)
def test_finetune_byod_problems_are_one_refused_line_and_remove_earlier_byod_results(tmp_path, monkeypatch, fields, file_name, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _byod_namespace(tmp_path, monkeypatch)
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "tiny.png")
    _seed_byod(ns, tmp_path)
    with pytest.raises(ValueError) as caught:
        _section12(ns, **fields)
    assert getattr(caught.value, "dimer_refusal", False), f"not a refusal: {caught.value!r}"
    (line,) = caught.value._render_traceback_()
    assert line.startswith(f"Refused: {file_name}: ") and rule in line and "\n" not in line
    assert "re-run this cell" in line and "earlier BYOD results of this session and their outputs/ files were removed" in line
    assert not set(BYOD_EARLIER) & set(ns)
    assert ns["adapted"] == "sample run kept"
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == ["mask2former_panoptic_finetune_result.json"]


def test_finetune_cancelled_upload_is_one_refused_line(tmp_path, monkeypatch):
    ns = _byod_namespace(tmp_path, monkeypatch)
    files = types.SimpleNamespace(upload=lambda: {})
    colab = types.ModuleType("google.colab")
    colab.files = files
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section12(ns, USE_BYOD_DATASET=True)
    (line,) = caught.value._render_traceback_()
    assert line.startswith("Refused: (no file): expected exactly one uploaded file") and "got 0" in line


def test_finetune_genuine_internal_error_keeps_its_traceback(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _byod_namespace(tmp_path, monkeypatch)
    Image.new("RGB", (64, 48)).save(tmp_path / "ok.png")
    _seed_byod(ns, tmp_path)

    class Broken:
        def segment(self, image):
            raise RuntimeError("internal bug")

    ns["reloaded"] = Broken()
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section12(ns, USE_BYOD_IMAGE=True, BYOD_IMAGE_PATH="ok.png")
    assert not getattr(caught.value, "dimer_refusal", False)
    assert ns["byod_evaluation"] == "old"  # nothing is cleared for an internal error


def test_byod_refusal_is_carried_verbatim_and_the_cell_ids_are_stable():
    assert CARRIED in _cell(INFERENCE, INFERENCE_BYOD_CELL)
    assert CARRIED in _cell(FINETUNE, FINETUNE_BYOD_CELL)


def test_default_paths_do_not_enter_the_refusal_wraps():
    inference = _cell(INFERENCE, INFERENCE_BYOD_CELL)
    wrap = inference[inference.index("if USE_BYOD:") : inference.index("else:\n    # Deterministic drawing")]
    assert "synthetic_scene" not in wrap and "refuse_byod(" in wrap
    finetune = _cell(FINETUNE, FINETUNE_BYOD_CELL)
    assert "USE_BYOD_IMAGE = False" in finetune and "USE_BYOD_DATASET = False" in finetune


def test_scripted_finetune_refusal_input(tmp_path, monkeypatch):
    """The hosted fine-tune refusal run sets USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH='/etc/hostname' (a text file)."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _byod_namespace(tmp_path, monkeypatch)
    hostname = Path("/etc/hostname") if Path("/etc/hostname").is_file() else tmp_path / "hostname"
    if not hostname.exists():
        hostname.write_text("runtime\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        _section12(ns, USE_BYOD_IMAGE=True, BYOD_IMAGE_PATH=str(hostname))
    (line,) = caught.value._render_traceback_()
    assert line.startswith("Refused: hostname: not an image Pillow can read (UnidentifiedImageError)")
