# Mask2Former panoptic E2E adaptation notebook — fleet-sweep fixes

**Source:** the 2026-10-05 static fleet sweep of `main` flagged this notebook for the old restart guard, a missing
guided layer (1 of 9 guided markers; declares `GUIDED`) and one quality `assert`. Targeted fix; **no full Notebook
Review Framework v1 review has been done**.
**Base:** `main` at `b50ef7d`.
**Notebook:** `tutorials/mask2former_panoptic_finetune_colab.ipynb`, regenerated with
`tools/build_notebook.py --template tools/notebook_template_finetune.py` (`build_notebook.py/2.2`).
**Readiness:** **Verification pending.** **Status labels:** unchanged.

## Findings

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R | Fixed — hosted confirmation pending | Same isolated runtime and lock as the inference notebook (the recorded 2026-09-18 Kaggle T4 run needed one restart after the install cell). | `tools/notebook_template_finetune.py` (+ shared generator, validator, lock) | `test_swp_r_*` |
| SWP-G | Fixed | Guided layer: audience, Input → Model → Output, How to use, roadmap, four **Predict** prompts and seven **Check your reasoning** answers from the recorded 2026-09-18 Kaggle T4 run (baseline PQ 0.0, trivial 0.103; loss 33.6 → 6.7; adapted PQ 0.866 with `box` 0.397; new scenes 0.954; reload PQ exact, pixel agreement 1.0), Troubleshooting, Change one thing, Glossary, Conclusion. Infrastructure cells collapsed. | template | `test_swp_g_*`; guided markers 1/9 → 9/9 |
| SWP-A | Not fixed — finding does not reproduce | The flagged `assert` (`byod_reloaded.evaluate(byod_held)['pq'] - byod_adapted['pq'] < 1e-9`) and the Section 10 asserts are reload-parity checks — the artifact must reproduce the adapted score and map exactly — i.e. contract integrity, which the brief keeps as hard checks. No assert compares a quality metric with a baseline. | — | `test_swp_a_only_contract_asserts_remain` |
| SWP-F | Not applicable (noted) | Section 6 builds a fresh re-headed `adapter` from the verified snapshot every time it runs, and the baseline is measured on it; nothing is labelled frozen after training. Re-running Section 7 alone continues training the same `adapter` (now stated in Section 7 and in How to use: re-run from Section 6). | Section 7 markdown | — |
| SWP-B | Fixed | Both BYOD branches were Colab-upload only (`next(iter(uploaded))`, bare `StopIteration` on cancel). `BYOD_IMAGE_PATH` and `BYOD_DATASET_PATH` (zip or directory) fields; guarded single-file upload fallback; a non-zip archive and a missing `dataset.json` are refused with the archive name. | Section 12 | `test_swp_b_finetune_byod_input_paths_and_uploads`, `test_swp_b_finetune_dataset_branch_refuses_a_bad_archive_with_its_name` |

## User-visible changes

- Isolated Section 1; no install into the kernel, no restart; Linux x86_64 only.
- Section 12 gains `BYOD_IMAGE_PATH` / `BYOD_DATASET_PATH`.
- Guided-layer markdown; the closing gains Troubleshooting, Change one thing, Glossary, Conclusion.

## Verification (offline; not clean-runtime evidence)

- No model or training stage ran. Stand-ins: Section 1 bootstrap; the Section 12 `byod_input` helper on temporary paths
  and a fake upload. Commands and counts as in `mask2former_panoptic_colab_Fixes.md` (one shared test suite: 51 → 68 passed).

## Remaining gates

- A hosted **Run all in one pass** on a T4 (no restart), then a re-run of the export cell (Section 11).
- The REL12 BYOD run (dataset branch).
- A full Notebook Review Framework v1 review has not been done.
