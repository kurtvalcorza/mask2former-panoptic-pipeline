"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Colab T4 of blobs 88249847 (2026-10-07), 6a8423ea and 14cf2531 (2026-10-25); Kaggle T4 and
CPU runs of the earlier blob baa9ab41 (2026-09-18).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "mask2former_panoptic_finetune_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_two_run_ranges_and_timing_promise_are_gone():
    md = _markdown()
    for stale in ("to between 6.400 and 6.659", "about 15 seconds of training", "In both recorded Colab T4 runs",
                  "`box` at 0.844 in the 25 October one"):
        assert stale not in md, stale


def test_confirming_run_of_14cf2531_is_recorded():
    md = _markdown()
    assert "from 33.612 to 6.263 in the run of blob `14cf2531`" in md
    assert "gave 0.9179 (SQ 0.9843, RQ 0.9314; `box` 0.84, `disc` 0.767, `triangle` 0.985)" in md
    assert "0.8712 in the Colab T4 run of blob `14cf2531`" in md
    assert "(0.926413, 0.962504 and 0.917899)" in md
