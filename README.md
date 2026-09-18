# Mask2Former Swin-T COCO panoptic segmentation pipeline

DIMER pipeline for **Mask2Former** (`facebook/mask2former-swin-tiny-coco-panoptic`), Cheng et al.'s ~47M-parameter universal segmenter — a Swin-T backbone, a deformable-attention pixel decoder and a masked-attention transformer decoder over 100 object queries, trained on COCO panoptic — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts one image and three caller-owned thresholds and returns a panoptic map (one segment id per pixel, `-1` for void) with one entry per segment over COCO's 133 categories, implementing the upstream per-pixel void rule itself; it ships `panoptic_quality` for callers who bring panoptic annotations, and a **bounded adaptation path** that re-heads the same weights for a caller-supplied vocabulary, fine-tunes with the upstream set loss, evaluates with panoptic quality, and exports a self-describing artifact that reloads from disk.

## Upstream alignment

- Model: `facebook/mask2former-swin-tiny-coco-panoptic`
- Revision: `df6b1142ff50c3276559d9d78f35f6a579c75a77`
- Upstream weight license: MIT (resolved from the upstream code repository; the Hub tag is `other` — recorded in `docs/WEIGHTS.md`)
- Upstream task: panoptic segmentation (COCO panoptic, 133 categories: 80 things, 53 stuff)
- Repository adaptation: **optional** — re-headed gradient fine-tuning of the transformer decoder, pixel decoder and a new class head on caller-supplied labelled scenes (backbone frozen by default); inference alone changes nothing

## Quick start

```python
from PIL import Image
from mask2former_panoptic_pipeline import Mask2FormerPanopticPipeline, evaluation_report

pipe = Mask2FormerPanopticPipeline.from_pretrained()      # stages + verifies weights/mask2former-swin-tiny-coco-panoptic first
result = pipe.segment(Image.open("scene.jpg"), score_threshold=0.5, overlap_threshold=0.8, mask_threshold=0.5)
result["segmentation"]                                    # int32 (H, W): segment id per pixel, -1 = void
for segment in result["segments"]:
    print(segment["label"], segment["is_thing"], segment["score"], segment["area_fraction"], segment["bbox"])
print(evaluation_report(result)["verdict"])              # 'not-measurable' without a reference map

# adapt to your own vocabulary on labelled scenes (instance map + {instance_id: class_name} per record)
from mask2former_panoptic_pipeline import SHAPE_CLASSES, SHAPE_STUFF, shape_dataset, split_records

train, held_out = split_records(shape_dataset(24, seed=0), holdout=0.25, seed=0)
adapter = Mask2FormerPanopticPipeline.from_pretrained(class_names=SHAPE_CLASSES, stuff_names=SHAPE_STUFF, seed=0)
adapter.finetune(train, epochs=6)                         # bounded; ~3 min on the reference CPU
print(adapter.evaluate(held_out)["pq"])                   # class-aware panoptic quality on held-out scenes
adapter.save_artifact("outputs/adapted")                  # config + safetensors + processor + digest manifest
reloaded = Mask2FormerPanopticPipeline.load_artifact("outputs/adapted")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed). On a fresh clone the manifest is committed but the weights are not: `Mask2FormerPanopticPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/mask2former-swin-tiny-coco-panoptic/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (4 files)
  config.json                # Mask2FormerForUniversalSegmentation: Swin-T, 100 queries, 133 labels
  preprocessor_config.json   # Mask2FormerImageProcessor: resize 384x384, ImageNet mean/std, ignore_index 255
  README.md                  # upstream card (license: other)
  model.safetensors          # git-ignored, 190,052,872 bytes
```

`pytorch_model.bin` exists upstream and is deliberately not listed (pickle; DIMER does not accept `.bin`).

## Input ceilings and request parameters

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`; `SCORE_THRESHOLD = 0.5` (the pinned `transformers` default; upstream inference uses 0.8), `OVERLAP_THRESHOLD = 0.8`, `MASK_THRESHOLD = 0.5` (all caller-owned, in `[0, 1]`); `INPUT_SIZE = 384`, `MASK_LOGIT_SIZE = 96`, `NUM_QUERIES = 100` (documentation only). Adaptation ceilings enforced by `validate_dataset`: 2–32 classes, ≤ 200 records, ≤ 50 instances per image with ids in 1..254 and every pixel assigned, ≤ 20 epochs, batch ≤ 8; defaults `TRAIN_NUM_POINTS = 4096`, 6 epochs, batch 2, learning rate 1e-4, backbone frozen. See `MODEL_CARD.md` for who owns the thresholds, what the scores are and are not, and the measured CPU timings.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mask2former-panoptic-pipeline/blob/main/tutorials/mask2former_panoptic_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mask2former-panoptic-pipeline/blob/main/tutorials/mask2former_panoptic_finetune_colab.ipynb)

Both notebooks are declared `GUIDED` under DIMER Notebook Specification 2.0 and are **standalone** (§4): generated by `tools/build_notebook.py`, they carry the two package modules, model identity, manifest digests and runtime pins, so the exported notebooks run without this repository (parity enforced by `tests/test_notebook_parity.py` and `tests/test_finetune_parity.py`).

- `tutorials/mask2former_panoptic_colab.ipynb` (`TASK-INFERENCE`): draws a 640×480 scene of coloured shapes in code together with its exact region map (no download), surfaces the ceilings, exposes the three thresholds as form parameters, resolves the pinned model through the carried staging and verification path, validates the request through `validate_inputs` into an input manifest, segments through `Mask2FormerPanopticPipeline.segment`, probes a blank image and noise, fetches the checkpoint's own model-card photograph (COCO `val2017/000000039769.jpg`) with a pinned SHA-256 and records its label set as a plausibility observation, writes an `evaluation_report` whose class-agnostic panoptic quality is sanity evidence only (`sample-sanity` against the drawn regions, `not-measurable` on the photograph and on BYOD; no COCO benchmark), and exports JSON, 16-bit segment-id PNGs and overlays. BYOD (your own image) is optional and gated off by default.
- `tutorials/mask2former_panoptic_finetune_colab.ipynb` (`E2E`): shows the COCO checkpoint on the drawn scene, draws and validates a 24-scene labelled panoptic set (`shape_dataset`, `validate_dataset` → dataset manifest), splits it 18/6 with a seed, re-heads the verified weights for the five-class vocabulary and measures the baseline (and a trivial all-`sky` baseline), runs the bounded 54-step fine-tune on the default path, scores held-out panoptic quality, segments new-seed scenes, exports the adapted model with `save_artifact`, reloads it with `load_artifact` and asserts identical scores and maps, refuses a tampered artifact, and exports JSON with provenance. Two BYOD branches (own image; own labelled directory through the same stages) are optional and gated off by default.

See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`, `tests/test_finetune_parity.py`) — do not constitute clean-runtime notebook evidence. One local fresh-kernel execution of each notebook is recorded in `docs/release-verification.md` as pre-flight; the supported-runtime runs are pending. Complete those records against the exact release revision before calling the notebooks release-grade.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, adaptation contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, the licence resolution note and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are recorded as MIT (resolved from the upstream code repository; the Hub tag is `other`); see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
