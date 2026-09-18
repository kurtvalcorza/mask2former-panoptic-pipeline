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
__LOCAL_ROW_INFERENCE__
__LOCAL_ROW_FINETUNE__

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| — | — | — | — | — | none recorded yet |

## Current status

No clean-runtime execution in a **supported** runtime (Colab or Kaggle) has been recorded yet. What
exists: static validation (`tools/validate_release_assets.py`), the generator parity checks (`--check`
OK for both templates), the offline unit suite, and one **local fresh-kernel execution** of each
generated notebook (table above) that exercised the standalone carrier, verification, segmentation, the
evaluation reports, the adaptation path with its artifact round-trip, and every export — which is
necessary but not promotion evidence because the workstation is not a supported runtime. The registry
status remains **Candidate** until a reviewer confirms a recorded supported-runtime run of each notebook
against the notebook blob under review and an integrator promotes it. Facts a reviewer should weigh:
the CUDA path has not been executed; the segment scores are uncalibrated softmaxes and the three
post-processing thresholds are the caller's (one segment at 0.5, none at 0.8 on the drawn scene); the
drawn samples are flat high-contrast shapes, so the inference PQ says nothing about photographs and the
adaptation PQ (0.866 held-out, one seed, one split, no dispersion) says the plumbing works, not that a
real dataset would adapt; the blank and noise probes were labelled `sky-other-merged` at 0.80 / 0.57,
which is an observation, not a guarantee; the public photograph is fetched over plain HTTP (digest-pinned)
from a host the notebook can be told to skip; the weight licence is MIT by resolution from the upstream
code repository, not by a Hub declaration (`docs/WEIGHTS.md`); and the pinned processor configuration
carries a `_max_size` key the pinned `transformers` warns about at load and ignores.
