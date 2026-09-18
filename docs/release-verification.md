# Release verification

`tutorials/mask2former_panoptic_colab.ipynb` (`TASK-INFERENCE`) and
`tutorials/mask2former_panoptic_finetune_colab.ipynb` (`E2E`), both **standalone** carriers, are
**release candidates** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly two tutorial notebooks, each named in `tutorials/README.md` with its profile, the
  notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec
  `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly two cells tagged `embedded_module`, equal to
  `src/mask2former_panoptic_pipeline/samples.py` and `pipeline.py` after the generator's documented
  rewrites, in dependency order; the inline `MANIFEST` equal to the committed snapshot manifest and
  the inline `PINS` equal to the `pyproject.toml` runtime pins; each notebook byte-identical (on LF)
  to `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Mask2FormerPanopticPipeline.from_pretrained(weights_dir=...)`, `synthetic_scene`, `validate_inputs`,
  `segment`, `evaluation_report`; for the `E2E` notebook `shape_dataset`, `validate_dataset`,
  `split_records`, the re-headed `from_pretrained(weights_dir=WEIGHTS_DIR, class_names=…)`,
  `evaluate`, `panoptic_quality`, `finetune`, `save_artifact`, `load_artifact`, the identical-score
  and pixel-agreement assertions and the tampered-artifact refusal), the ceiling print, the exports,
  the digest-pinned public photograph, the learner-facing statements (uncalibrated softmax scores,
  void, class-agnostic PQ on the drawn scene, `not-measurable` without a reference, training loss as
  optimisation evidence only, frozen backbone, sample metric not a benchmark) and the gated-off BYOD
  defaults listed in the validator; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com/kurtvalcorza` / repository import on the primary path, a mutable `revision='main'`,
  direct `from transformers import` / `Mask2FormerForUniversalSegmentation` / `AutoImageProcessor` /
  `post_process_panoptic_segmentation` / `torch.optim.` / `.backward()` / `save_pretrained(` /
  `from huggingface_hub import` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check` for both templates, and
the offline unit suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`, `tests/test_finetune_parity.py`;
fabricated decoder outputs, no weights). These are source/provenance and unit checks. They are **not**
execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorials are written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU or T4 kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (**no repository checkout is needed — the notebooks are standalone**) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False` and the three thresholds at 0.5 / 0.8 / 0.5 for the
   `TASK-INFERENCE` notebook; `N_IMAGES = 24`, `DATASET_SEED = 0`, `EPOCHS = 6`, `HOLDOUT = 0.25`,
   `SEED = 0`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 2`, `FREEZE_BACKBONE = True`, `NEW_DATA_SEED = 123`,
   `USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False` for the `E2E` notebook);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the two carried module cells execute (defining `synthetic_scene`, `shape_dataset`,
     `validate_dataset`, `split_records`, `Mask2FormerPanopticPipeline`, `validate_inputs`,
     `evaluation_report`, `panoptic_quality`, `verify_snapshot`, `stage_missing_files`) with no import
     of the repository package;
   - pinned `facebook/mask2former-swin-tiny-coco-panoptic` acquisition at the immutable revision through
     the carried module: the inline `MANIFEST` is asserted against the module identity and written to
     `weights/mask2former-swin-tiny-coco-panoptic/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)`
     reports all 4 manifest entries on a clean runtime, `verify_snapshot` returns its summary dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loads from the verified directory with no further Hub
     access (any download in the logs after staging is a finding);
   - **`TASK-INFERENCE`:** the drawn 640×480 scene with its region map and RGB SHA-256 printed and the
     ceilings surfaced; `validate_inputs` writes `outputs/mask2former_panoptic_input_manifest.json`
     (verdict `accepted`, one recorded rejection finding from the threshold probe); `segment` returning a
     map at 640×480 — record the segments (the card-pass CPU smoke gave one `stop sign` at 0.601 over
     8.3 % with 91.7 % void; a materially different result is a finding to record, not a failure by
     itself, because no metric is asserted — kernels differ across devices) and the blank/noise probes;
     the public photograph fetched with a matching digest and segmented (the smoke gave two `cat`, two
     `remote`, one `couch`, 4.9 % void); `evaluation_report` writes
     `outputs/mask2former_panoptic_evaluation_report.json` with verdict `sample-sanity` (class-agnostic
     `pq` / `sq` / `rq`, match counts) and `outputs/mask2former_panoptic_photo_evaluation_report.json`
     with verdict `not-measurable`; `outputs/mask2former_panoptic_result.json`, the two 16-bit
     segmentation PNGs and the two overlay PNGs written with `NOTEBOOK_SOURCE`, model revision, model
     licence, runtime versions and device;
   - **`E2E`:** the COCO observation on the drawn scene; `validate_dataset` writes
     `outputs/mask2former_panoptic_finetune_dataset_manifest.json` (verdict `accepted`, 24 records,
     one recorded rejection finding from the unassigned-pixel probe); the 18/6 split; the re-headed
     pipeline reporting `class_predictor.bias`, `class_predictor.weight` and `criterion.empty_weight`
     as re-initialised; baseline PQ and the trivial all-`sky` PQ; `finetune` completing 54 steps with a
     decreasing mean epoch loss (33.7 → 6.8 in the smoke; the loss is optimisation evidence only);
     held-out PQ recorded beside the baselines (the smoke gave 0.866; a materially different value on
     another device is a finding to record); four new-seed scenes segmented and scored; `save_artifact`
     writing `outputs/mask2former-panoptic-adapted/` (config, safetensors, processor config, manifest);
     `load_artifact` reproducing the held-out PQ exactly and the first held-out map pixel for pixel; the
     tampered artifact refused; `outputs/mask2former_panoptic_finetune_evaluation_report.json` and
     `outputs/mask2former_panoptic_finetune_result.json` written with provenance;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with
`git rev-parse <commit>:tutorials/<notebook>`). Wall times, when recorded, are the sum of per-cell
times reported by the executor and include installs and the model download; they are measurements
for the stated runtime, not general estimates.

### Local pre-flight evidence (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | notebook blob `1b0645e8806a` (commit `25f8fdb`, generated at `9dc0dc7`; `NOTEBOOK_SOURCE.repository_revision` = `9dc0dc7…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 10 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 4 manifest entries (190 MB) from the Hub cache at the pinned revision into the empty scratch `weights/`, `verify_snapshot` PASS (4 files), no further download in the log, `segment` on the drawn scene in 0.7 s → one `stop sign` at 0.601 over 8.3 %, void 91.7 %; blank → `sky-other-merged` 0.798 (100 %), noise → `sky-other-merged` (94.9 %); public photograph fetched with the pinned digest → two `cat` (0.998, 0.997), two `remote` (0.994, 0.953), `couch` 0.811, void 4.9 %; `evaluation_report` `sample-sanity` class-agnostic PQ 0.33 (1 TP, 4 FN) on the scene and `not-measurable` on the photograph — identical to the smoke run; 8 outputs written (JSON ×4, two 16-bit segmentation PNGs, two overlays) | 55.3 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-18 | notebook blob `baa9ab413435` (commit `25f8fdb`, generated at `9dc0dc7`; `NOTEBOOK_SOURCE.repository_revision` = `9dc0dc7…`) | Local Windows-venv harness (as above) | Default adaptation path, all 13 code cells: staging and verification as above; COCO observation on the drawn scene; `validate_dataset` accepted 24 records (one recorded rejection finding); 18/6 split (seed 0); re-head reported `class_predictor.bias`, `class_predictor.weight`, `criterion.empty_weight`; baseline PQ 0.0, trivial all-`sky` PQ 0.103; `finetune` 54 steps in 118.2 s (mean epoch loss 33.7 → 6.8); adapted held-out PQ 0.866 (SQ 0.938, RQ 0.911), new-seed scenes PQ 0.952; `save_artifact` 3 files / 189,998,883 bytes; `load_artifact` reproduced PQ 0.866 exactly with pixel agreement 1.0; tampered manifest refused; dataset manifest, evaluation report and result JSON written | 170.4 s | PASS — pre-flight only; not promotion evidence |

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `9498ad0` / `1b0645e8806a` | Kaggle CPU (`kurtvalcorza/dimer-nb2-mask2former-panoptic` v1; image `gcr.io/kaggle-images/python`, Python 3.12.13, image torch 2.10.0+cpu / transformers 5.0.0 / numpy 2.0.2 before the pinned install; executor `runner.py`, nbclient in a fresh `python3` kernel; no repository checkout, clean model cache) | Default sample path: pinned install replaced numpy 2.0.2 → 2.5.3, the notebook's own guard raised `Restart the runtime, then rerun from the top` (pass 1, 198.5 s) and the executor restarted and reran from the top (pass 2, 67.1 s); `stage_missing_files` fetched all 4 manifest entries (190 MB) from the Hub at the pinned revision; runtime torch 2.14.0+cu130 / transformers 4.57.6 on `cpu`; drawn scene → one `stop sign` 0.601 over 8.3 %, void 91.7 %, `sample-sanity` class-agnostic PQ 0.33; blank → `sky-other-merged` 0.798 (100 %), noise → 0.572 (94.9 %); public photograph digest matched → two `cat` (0.998, 0.997), two `remote` (0.994, 0.953), `couch` 0.811, void 4.9 %, `not-measurable`; 8 outputs written | 265.7 s | **PASSED** — 10/10 ok code cells (1 restart after the install cell), results identical to the local pre-flight; REL1/REL8 satisfied for this blob |
| 2026-09-18 | `9498ad0` / `baa9ab413435` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-mask2former-panoptic-finetune` v1, pushed with `--accelerator NvidiaTeslaT4`; T4 15360 MiB, driver 580.159.04; image torch 2.10.0+cu128; same executor, no checkout, clean cache) | Default adaptation path: install pass 181.5 s → restart guard → pass 2 77.1 s; 4 manifest entries staged; runtime torch 2.14.0+cu130 / transformers 4.57.6 on `cuda:0`, float32; dataset manifest accepted (24 records, 1 rejection finding); 18/6 split; re-head reported the three expected tensors; baseline PQ 0.0, trivial all-`sky` PQ 0.103; `finetune` 54 steps in 16.2 s (mean epoch loss 33.6 → 6.7); adapted held-out PQ 0.866 (per class sky 0.998, ground 0.999, disc 0.951, box 0.397, triangle 0.986 — the CUDA head initialisation and kernels moved the per-class numbers relative to the CPU pre-flight while the mean stayed within 0.001); new-seed scenes PQ 0.954; `save_artifact` 3 files / 189,996,737 bytes; `load_artifact` reproduced PQ 0.866 exactly with pixel agreement 1.0; tampered manifest refused | 258.6 s | **PASSED** — 13/13 ok code cells (1 restart after the install cell); REL1/REL8 satisfied for this blob |

## Current status

Clean-runtime execution in a **supported** runtime is now recorded for both notebooks (table above): the
committed blobs `1b0645e8806a` (Kaggle CPU, 265.7 s) and `baa9ab413435` (Kaggle Tesla T4, 258.6 s)
ran top-to-bottom from GitHub raw content with no repository checkout and a clean model cache, through
the notebooks' own restart guard, with every default-path stage, export, the artifact round-trip and the
tampered-artifact refusal observed. The evidence (executor `run_summary.json`, both executed notebook
passes, the `outputs/` directory) is kept under `Projects/.agent/backups/kaggle-m2f-2026-09-18/out/`
with its `LEDGER.md`. The registry status stays **Candidate** until a reviewer confirms these records
against the notebook blob under review and an integrator promotes it. Facts a reviewer should weigh: the
CUDA path was exercised only by the `E2E` notebook (the `TASK-INFERENCE` run was CPU); the segment
scores are uncalibrated softmaxes and the three post-processing thresholds are the caller's (one segment
at 0.5, none at 0.8 on the drawn scene); the drawn samples are flat high-contrast shapes, so the
inference PQ says nothing about photographs and the adaptation PQ (0.866 held-out on both CPU and T4, one
seed, one split, no dispersion; per-class values moved between devices — `box` 0.509 on CPU, 0.397 on
T4) says the plumbing works, not that a real dataset would adapt; the blank and noise probes were
labelled `sky-other-merged` at 0.80 / 0.57, which is an observation, not a guarantee; the public
photograph is fetched over plain HTTP (digest-pinned) from a host the notebook can be told to skip; the
weight licence is MIT by resolution from the upstream code repository, not by a Hub declaration
(`docs/WEIGHTS.md`); and the pinned processor configuration carries a `_max_size` key the pinned
`transformers` warns about at load and ignores.
