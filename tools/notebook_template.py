"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package
modules, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.

Review fixes (Notebook Review Framework v1, review PR #7, M2P-M1..M2 / M2P-m1..m5): the runtime is the fleet's uv
isolated environment (generator /2.1, no in-kernel install, no restart; lock compiled from the pyproject pins plus scipy and
constrained to florence2-vision-language-pipeline's T4-passed lock); the guided layer (audience, input/output
contract, how to use, roadmap, predictions, worked answers, a change-one-thing activity in Section 10,
troubleshooting, glossary, conclusion) is added and Sections 1-3 are labelled and collapsed as Infrastructure;
BYOD takes a path (`BYOD_PATH`) or one upload, refuses cancelled uploads, undecodable files and 16/32-bit images
with messages naming the contract, and applies and records EXIF orientation. The EXIF and bit-depth rules live in
this tutorial cell, not in `src/`: `pipeline.py` is carried byte for byte by the companion E2E notebook as well.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

CHECK = "<details><summary>Check your reasoning</summary>\n\n"
END = "\n\n</details>"
KAGGLE_RUN = "the recorded Kaggle CPU run of 18 September 2026 (`docs/release-verification.md`; Python 3.12, torch 2.14.0, transformers 4.57.6)"
CPU_CHECK = "this repository's CPU check of 6 October 2026 (Windows workstation CPU, torch 2.13.0+cpu, transformers 4.57.6; its drawn-scene values equal the review's run with the pinned torch 2.14.0+cpu)"

TEMPLATE = {
    "package": "mask2former_panoptic_pipeline",
    "repo_name": "mask2former-panoptic-pipeline",
    "stem": "mask2former_panoptic",
    "notebook_name": "mask2former_panoptic_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    "isolated_runtime": True,
    # M2P-M1: the fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline;
    # generator /2.1 as in florence2-vision-language-pipeline 9c4e95a and gliner-ner-pipeline fe3d5ba): managed CPython, a size-
    # and SHA-256-verified uv wheel, and a lock compiled from `pins_file` (the pyproject pins plus scipy==1.18.1) with
    # `uv pip compile tutorials/requirements-colab.in -c <versions of florence2-vision-language-pipeline's T4-passed
    # tutorials/requirements-colab.lock.txt @ 9c4e95a, plus grounding-dino-detection-pipeline's T4-passed scipy==1.18.1>
    # --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --only-binary :all:
    # -o tutorials/requirements-colab.lock.txt`: florence2's packages at the same versions and hashes minus pyarrow, plus
    # scipy (48 packages). scipy is required at model construction: transformers 4.57.6 builds Mask2FormerLoss in
    # Mask2FormerForUniversalSegmentation.__init__ and it calls requires_backends(['scipy']) (Colab T4 run of fe62ef4 failed
    # there). The pins file keeps pyproject.toml, and so the E2E notebook's inline PINS, unchanged.
    "pins_file": "tutorials/requirements-colab.in",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "Mask2FormerPanopticPipeline",
    "weights_key": "mask2former-swin-tiny-coco-panoptic",
    "modules": ["samples.py", "pipeline.py"],
    "runtime_imports": ["torch", "transformers"],
    "external_access_extra": "and `images.cocodataset.org` over plain HTTP (Section 7) for one public photograph (173,131 bytes) that is refused unless its SHA-256 matches the pinned digest",
    "title": "Mask2Former Swin-T COCO panoptic — DIMER panoptic segmentation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/mask2former-panoptic-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/mask2former-panoptic-pipeline/blob/main/tutorials/mask2former_panoptic_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fmask2former--swin--tiny--coco--panoptic-ffcc4d?style=flat",
            "https://huggingface.co/facebook/mask2former-swin-tiny-coco-panoptic",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-facebookresearch%2FMask2Former-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/facebookresearch/Mask2Former",
        ),
        (
            "arXiv",
            "https://img.shields.io/badge/arXiv-2112.01527-b31b1b.svg",
            "https://arxiv.org/abs/2112.01527",
        ),
    ],
    "capability": "panoptic segmentation — one image → a map assigning every pixel to one segment (a countable *thing* instance or a fused amorphous *stuff* region) or to void, over the 133-class COCO panoptic vocabulary of the pinned `facebook/mask2former-swin-tiny-coco-panoptic` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing "
        "is installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and "
        "digest-verifies the pinned snapshot, draws the tutorial sample automatically, validates it into an input manifest "
        "before the model runs, segments it, fetches one digest-pinned public photograph, writes the evaluation reports, and "
        "exports machine-readable outputs with provenance. The default path needs no repository clone, no DIMER worker or "
        "service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4 — with `BYOD_PATH` set to the path of your "
        "image (Kaggle, Jupyter, or a file already in the Colab session), or left empty to get the Colab upload dialog — and "
        "re-run from that cell (select it, then **Runtime → Run after**). Your image passes through the same validation, "
        "segmentation, evaluation-report and export cells as the sample; with no reference map its verdict is "
        "`not-measurable`. The accepted input, the refusals and the privacy guidance are stated in the Prerequisites and in "
        "Section 4, and the image stays inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "At inference the Mask2Former model (a Swin-T backbone, a multi-scale deformable-attention pixel decoder and a "
        "masked-attention transformer decoder with 100 object queries; about 47M parameters, trained on COCO panoptic) "
        "emits, for each query, a class distribution over 133 COCO categories plus *no object* and a 96×96 mask logit map. "
        "The carried module turns those into a panoptic map with the upstream rule (a query survives when its best class "
        "probability reaches `score_threshold`; each pixel goes to the surviving query with the highest score-weighted mask "
        "probability and is assigned only where that query's own mask probability reaches `mask_threshold`, otherwise it is "
        "**void**; a query whose assigned area falls below `overlap_threshold` of its thresholded mask is dropped; stuff "
        "queries of one class are fused). **No adaptation occurs:** no training, fine-tuning, in-context conditioning or "
        "preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights and the image "
        "processor, and the carried package adds snapshot verification, the input contract (image side ceilings, three "
        "thresholds in [0, 1]), a fixed output contract (segment id map, one entry per segment with label, score, area, "
        "box, thing/stuff, fused flag), and the `validate_inputs`, `panoptic_quality` and `evaluation_report` helpers. The "
        "default sample is a flat scene of coloured shapes drawn in code with the exact regions it was drawn from — an "
        "out-of-domain input on which the checkpoint finds one region — followed by the checkpoint's own model-card "
        "widget photograph (two cats on a couch), fetched at run time and digest-checked, on which it finds five. "
        "The class-agnostic panoptic quality on the drawing is demonstration (plumbing) evidence for one image, not a "
        "COCO benchmark, and the photograph has no reference so its report is `not-measurable`.\n\n"
        "**Who this is for.** A learner who knows basic Python, has run a Colab or Jupyter notebook, and wants to see what a "
        "pretrained panoptic segmentation model returns for an image, how to read its map and scores without over-reading "
        "them, and why nothing here is a COCO accuracy number — for example a practitioner deciding whether a pretrained "
        "segmenter is worth trying on their own photographs. No prior experience with Mask2Former is assumed; *panoptic*, "
        "*thing*, *stuff*, *void*, *query* and *PQ* are explained where they first matter and again in the **Glossary**. "
        "This is a teaching run, not a benchmark. CPU is enough; a GPU is used automatically when present.\n\n"
        "**Input → Model → Output.**\n\n"
        "| | What it is in this notebook |\n"
        "|---|---|\n"
        "| Input | one image and three thresholds (default: a 640×480 scene drawn in code with its exact region map, plus one digest-pinned COCO photograph; BYOD: your own image) |\n"
        "| Model | Mask2Former with a Swin-T backbone trained on COCO panoptic: 100 queries, each proposing a mask and a class over 133 categories, merged by a fixed per-pixel rule |\n"
        "| Output | a segment-id map (or void) at input resolution, one entry per segment (label, thing/stuff, score, area, box), a class-agnostic PQ against the drawn regions (`sample-sanity`), `not-measurable` for inputs without a reference, and JSON/PNG exports with provenance |\n\n"
        "**How to use this notebook.** Choose **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing "
        "into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated "
        "environment, the carried package and the pinned model — and their cells are collapsed: you may run them without "
        "reading them. The learning path starts in Section 4. Form fields (`# @param`) are the knobs; after changing one, "
        "select the cell that holds it and choose **Runtime → Run after**, so every later section uses the new value. Before "
        "each principal result the notebook asks you to **Predict before running**, then says **What to notice**, and a "
        "collapsible **Check your reasoning** block gives a worked answer quoting " + KAGGLE_RUN + ". Section 10 is a "
        "change-one-thing activity. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. "
        "Section 1 reads two environment variables, both optional: `DIMER_ISOLATED_ENV` (where to build the isolated "
        "environment; default `dimer_isolated_env/` in the working directory) and `DIMER_NOTEBOOK_CI_PREINSTALLED` (set to "
        "`1` only by an executor that has already installed exactly the pinned packages: the notebook then skips the "
        "isolated environment and runs every cell in its own kernel).\n\n"
        "**Roadmap:** 1–3 infrastructure → 4 the drawn scene (or your image) → 5 validation into an input manifest → 6 segment "
        "and read the output *(core concept: queries, thresholds, void, uncalibrated scores)* → 7 an in-domain photograph "
        "*(observation, not a metric)* → 8 the evaluation report *(evaluation practice)* → 9 export *(engineering)* → 10 your "
        "turn: change one threshold → interpretation, troubleshooting, glossary, conclusion."
    ),
    "learning_objectives": (
        "by the end you will be able to (1) explain how 100 query masks become one panoptic map, and point to the pixels "
        "this rule leaves void; (2) read a segment list — id, label, thing or stuff, score, area — and say why a score of 0.6 "
        "is not a 60 % probability; (3) predict, then check, how the drawn scene, a blank image, random noise and a COCO "
        "photograph are segmented, and explain the difference in domain terms; (4) state why the drawn scene earns a "
        "`sample-sanity` PQ while the photograph and your own image are `not-measurable`, and what data would change that; "
        "(5) change one threshold, re-run the affected sections and explain the change in segments, void and PQ; (6) run "
        "your own image through validation and segmentation, and recognise the messages that refuse an unusable file; and "
        "(7) find the exported maps, overlays and provenance and say what each records."
    ),
    "exclusions": (
        "Semantic-only or instance-only output formats (the pipeline emits the panoptic map; both can be derived from "
        "it), the ADE20K / Cityscapes checkpoints of the same family, a vocabulary other than COCO's 133 categories "
        "(the companion `E2E` notebook re-heads and fine-tunes for your own classes), calibrated confidence, evaluation on "
        "COCO panoptic (annotations are not bundled; only drawn regions are scored here), batch throughput, and any "
        "training. The model was trained on COCO photographs; flat drawings, documents, medical and satellite imagery are "
        "outside what this notebook measures, and on such inputs a confident-looking label carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. Section 1 builds its own Python 3.12.12 environment whatever Python the kernel runs, from manylinux wheels, so Windows and macOS kernels are not supported (the cell stops with that message). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 18 s to load and 1.7–8 s per 640×480 image (decoder pass plus NumPy/Pillow post-processing) on a workstation CPU (Intel Core Ultra 9 275HX). The large downloads are the locked environment (torch with its CUDA libraries, a few GB from PyPI) and the 190 MB checkpoint.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a segmentation map is; the difference between countable *things* and amorphous *stuff*; why a softmax score is not a calibrated probability; what intersection-over-union measures and how panoptic quality (PQ = SQ × RQ) is built from it. Each of these is also in the Glossary.",
        "- **Data:** the default sample is a deterministic 640×480 scene drawn in code with Pillow (a red disc, a blue box, a yellow triangle and a green ground band on an off-white sky; no text rendering, so its digest is stable across Pillow builds) with the exact region map it was drawn from, so nothing is downloaded and no private data is needed. Section 7 additionally fetches one public photograph over plain HTTP (`images.cocodataset.org/val2017/000000039769.jpg`, 173,131 bytes; the checkpoint's own model-card widget example) and refuses it unless its SHA-256 matches the pinned digest; it is used only for display and inference, never redistributed, and its individual Flickr licence is not verified by this repository. Optional BYOD is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: **one image file** Pillow can decode (PNG, JPEG, WebP and similar) with **8 bits per channel** (RGB, RGBA, greyscale or palette), sides between 16 and 4096 px; give its path in `BYOD_PATH`, or leave `BYOD_PATH` empty on Colab to upload it. Section 4 refuses, with a message naming the rule: a missing path, a cancelled or multi-file upload, a file that is not a readable image, and a 16-bit or 32-bit image (modes `I;16`, `I`, `F`), which converting to RGB would clip to a few grey levels — save it as 8-bit first. A photograph that carries an EXIF orientation tag (most phone pictures) is rotated upright before validation, and the rotation is recorded in the input manifest. Section 5 refuses sides outside 16–4096 px. No reference map exists for your image, so its report is `not-measurable`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scene or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: `synthetic_scene` (carried above) draws a "
                "red disc, a blue box and a yellow triangle on an off-white sky above a green ground band at 640×480, and the "
                "same drawing calls produce the reference region map — five regions, every pixel assigned, two of them stuff "
                "(`sky`, `ground`) and three things. The regions are the references for the panoptic-quality sanity check "
                "later. They are not COCO categories and not a labelled dataset, so nothing here is a COCO measurement. The "
                "image digest is printed for the record.\n\n"
                "**Your own image (optional).** Set `USE_BYOD = True` and either put the path of one image in `BYOD_PATH` or, "
                "on Colab, leave it empty to get the upload dialog; then select this cell and choose **Runtime → Run after**. "
                "The cell refuses a missing path, a cancelled or multi-file upload, an unreadable file and a 16- or 32-bit "
                "image, each with a message naming the rule and the fix; it rotates a photograph with an EXIF orientation tag "
                "upright and records that rotation (Section 5 adds it to the input manifest). No reference map exists for your "
                "image, so the evaluation report will be `not-measurable`.\n\n"
                "The three post-processing thresholds are **caller-owned request parameters**: `score_threshold` gates queries "
                "on their best class probability (`SCORE_THRESHOLD = 0.5` is the pinned `transformers` default; the upstream "
                "Mask2Former inference uses 0.8), `mask_threshold` is the per-pixel cut that separates a segment from void, and "
                "`overlap_threshold` drops a query that lost most of its mask to stronger queries. Nothing is validated in this "
                "cell — the next section hands the image to the pipeline's own validation stage, which is the only checker of "
                "sizes and thresholds.\n\n"
                "**Predict before running:** a model trained on COCO's 133 categories sees a red disc, a blue box and a yellow "
                "triangle. What will it call them?\n\n"
                "**What to notice:** the sample kind, the image size and digest, the thresholds, and the five reference regions "
                "with their areas (sky and ground are large stuff regions; the three shapes are small things)."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageOps\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                "# Path of one image file (Kaggle, Jupyter, or a file already in the Colab session); empty = the Colab upload dialog.\n"
                'BYOD_PATH = ""  # @param {{type:"string"}}\n'
                'score_threshold = 0.5  # @param {{type:"number"}}\n'
                'overlap_threshold = 0.8  # @param {{type:"number"}}\n'
                'mask_threshold = 0.5  # @param {{type:"number"}}\n\n'
                "BYOD_CONTRACT = f'one image file Pillow can read (PNG, JPEG, WebP, ...) with 8 bits per channel and sides of {{MIN_IMAGE_SIDE}}-{{MAX_IMAGE_SIDE}} px'\n"
                "BYOD_RETRY = 'fix it, keep USE_BYOD = True, and re-run from this cell (Runtime -> Run after)'\n\n\n"
                "def byod_bytes(path_text):\n"
                "    \"\"\"(name, bytes) of the BYOD file: from BYOD_PATH when set, otherwise from exactly one Colab upload.\"\"\"\n"
                "    if path_text.strip():\n"
                "        path = Path(path_text.strip()).expanduser()\n"
                "        if not path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{str(path)!r}} is not a file in this runtime. Give the path of {{BYOD_CONTRACT}}; {{BYOD_RETRY}}')\n"
                "        return path.name, path.read_bytes()\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'USE_BYOD = True with an empty BYOD_PATH needs the Google Colab upload dialog. Elsewhere, set BYOD_PATH to {{BYOD_CONTRACT}}; {{BYOD_RETRY}}') from None\n"
                "    uploaded = files.upload() or {{}}\n"
                "    if len(uploaded) != 1:\n"
                "        received = ', '.join(sorted(uploaded)) if uploaded else 'nothing: the upload was cancelled or empty'\n"
                "        raise ValueError(f'Upload exactly one image (received {{len(uploaded)}}: {{received}}). Expected {{BYOD_CONTRACT}}; {{BYOD_RETRY}}')\n"
                "    return next(iter(uploaded.items()))\n\n\n"
                "def byod_image(name, data):\n"
                "    \"\"\"Decode the file, refuse 16/32-bit samples, and rotate an EXIF-tagged photograph upright (recorded).\"\"\"\n"
                "    try:\n"
                "        loaded = Image.open(io.BytesIO(data))\n"
                "        loaded.load()\n"
                "    except (OSError, SyntaxError, ValueError) as exc:  # a non-image raises PIL.UnidentifiedImageError, an OSError\n"
                "        raise ValueError(f'{{name}} is not an image Pillow can read ({{type(exc).__name__}}). Expected {{BYOD_CONTRACT}}; {{BYOD_RETRY}}') from None\n"
                "    if loaded.mode in ('I', 'F') or loaded.mode.startswith('I;16'):\n"
                "        raise ValueError(f'{{name}} has image mode {{loaded.mode!r}} (16- or 32-bit samples): converting it to RGB would clip it to a few grey levels and the map would be meaningless. Save it as an 8-bit RGB or greyscale image first; {{BYOD_RETRY}}')\n"
                "    adjustments = []\n"
                "    orientation = loaded.getexif().get(0x0112, 1)\n"
                "    if orientation != 1:\n"
                "        size_before = list(loaded.size)\n"
                "        loaded = ImageOps.exif_transpose(loaded)\n"
                "        adjustments.append({{'input': name, 'verdict': 'adjusted', 'message': f'EXIF orientation {{orientation}} applied before validation: {{size_before[0]}}x{{size_before[1]}} stored, {{loaded.size[0]}}x{{loaded.size[1]}} segmented'}})\n"
                "    return loaded, adjustments\n\n\n"
                "input_adjustments = []\n"
                "if USE_BYOD:\n"
                "    image_name, image_bytes = byod_bytes(BYOD_PATH)\n"
                "    image, input_adjustments = byod_image(image_name, image_bytes)\n"
                "    reference = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawing: no randomness and no text rendering, so no seed is needed and the digest is stable.\n"
                "    image, reference_segmentation, reference_segments = synthetic_scene()\n"
                "    reference = (reference_segmentation, reference_segments)\n"
                "    image_name = 'synthetic_shapes_640x480.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256,\n"
                "       'thresholds': {{'score': score_threshold, 'overlap': overlap_threshold, 'mask': mask_threshold}}, 'input_adjustments': input_adjustments,\n"
                "       'reference_regions': None if reference is None else [(s['name'], 'stuff' if not s['is_thing'] else 'thing', round(s['area_fraction'], 3)) for s in reference[1]]}})\n"
                "image"
            ),
        },
        {
            "md": (
                CHECK
                + "Mostly nothing. None of the shapes is a COCO category, and in " + KAGGLE_RUN + " the model kept a single "
                "query — `stop sign` at score 0.601 over the red disc — and left 91.7 % of the image void (Section 6). A flat "
                "drawing is out of domain by construction; that is the point of this sample." + END + "\n\n"
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `segment` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px and three thresholds in `[0, 1]` — and returns an "
                "**input manifest** naming the schema (including the 384×384 resize that does not preserve aspect ratio, the "
                "100 queries, the 96×96 mask logits and the resampling), the input's observed mode and size, the thresholds and "
                "the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, "
                "the cell also validates a request whose `score_threshold` is outside `[0, 1]` and records the pipeline's own "
                "error message as a finding; a BYOD rotation from Section 4 is recorded as a finding too. Inside the pipeline "
                "the image is converted to RGB and resized; nothing else is dropped or altered. The pipeline cannot tell whether "
                "an image is a photograph: that contract is the caller's.\n\n"
                "**Predict before running:** will a `score_threshold` of 1.5 be accepted?\n\n"
                "**What to notice:** the ceilings, then the manifest — `verdict: accepted` for the real request, and a "
                "`rejected` finding carrying the pipeline's own message for the probe."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'INPUT_SIZE': INPUT_SIZE, 'MASK_LOGIT_SIZE': MASK_LOGIT_SIZE, 'NUM_QUERIES': NUM_QUERIES, 'SCORE_THRESHOLD': SCORE_THRESHOLD, 'OVERLAP_THRESHOLD': OVERLAP_THRESHOLD, 'MASK_THRESHOLD': MASK_THRESHOLD}}}})\n"
                "input_manifest = validate_inputs(image, score_threshold=score_threshold, overlap_threshold=overlap_threshold, mask_threshold=mask_threshold, names=[image_name])\n"
                "# BYOD only: the EXIF rotation applied in Section 4 (empty for the drawn scene).\n"
                "input_manifest['findings'].extend(input_adjustments)\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, score_threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'threshold-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                CHECK
                + "No: thresholds must lie in `[0, 1]`, so the probe is rejected with `score_threshold must be a number in [0, 1], "
                "got 1.5` and that message is stored under `findings`, while the real request is `accepted`. Validation "
                "refuses before the model runs, so a bad request never produces a map." + END + "\n\n"
                "## 6. Segment the scene and read the output correctly\n\n"
                "`segment` returns `segmentation` (an int32 map at input resolution: the segment id per pixel, `-1` for void), "
                "`segments` (one entry per segment, ordered by score: `label`, `label_id`, `is_thing`, `score`, `area_fraction`, "
                "tight `bbox`, `was_fused`, the winning `query`), the `void_fraction`, the three thresholds, the vocabulary and "
                "the model identity. **Scores are uncalibrated softmaxes**: a 0.9 query is not 90 % likely to be right, and the "
                "map is a hard assignment — every pixel has exactly one segment or none. Post-processing is deterministic on a "
                "fixed device and dtype; CUDA kernels can shift logits slightly, so GPU and CPU maps need not match at a boundary. "
                "The pinned `transformers` post-processor would hand the whole frame to the one surviving query (area 1.0), which "
                "is why the carried module implements the upstream per-pixel rule itself. The cell also probes two degenerate "
                "inputs (a blank white image and uniform noise) and records what the model says about nothing — an observation "
                "about this checkpoint, not a guarantee.\n\n"
                "**Predict before running:** (a) how many segments will the drawn scene keep at `score_threshold = 0.5`, and how "
                "much of it will be void? (b) A blank white image has nothing in it. Will its map be void?\n\n"
                "**What to notice:** the device and timing, the segment lines (label, thing or stuff, score, area, box), the void "
                "fraction, and the two degenerate-input entries."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.segment(image, score_threshold=score_threshold, overlap_threshold=overlap_threshold, mask_threshold=mask_threshold)\n"
                "elapsed = round(time.time() - t0, 2)\n"
                "print({{'device': pipe.device, 'seconds': elapsed, 'n_segments': len(result['segments']), 'void_fraction': round(result['void_fraction'], 3), 'vocabulary_size': len(result['class_names'])}})\n"
                "for segment in result['segments']:\n"
                "    print(f\"id={{segment['id']:2d}} {{segment['label']:22s}} {{'thing' if segment['is_thing'] else 'stuff'}}  score={{segment['score']:.3f}}  area={{segment['area_fraction']:.3f}}  bbox={{segment['bbox']}}  fused={{segment['was_fused']}}\")\n"
                "if not result['segments']:\n"
                "    print('no query reached the score threshold: the whole map is void')\n\n"
                "# Degenerate inputs: what the model says about nothing (recorded, not asserted).\n"
                "degenerate = {{}}\n"
                "for probe_name, probe in (('blank', Image.new('RGB', (640, 480), (255, 255, 255))), ('noise', Image.fromarray(np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)))):\n"
                "    probe_result = pipe.segment(probe, score_threshold=score_threshold, overlap_threshold=overlap_threshold, mask_threshold=mask_threshold)\n"
                "    degenerate[probe_name] = {{'segments': [(s['label'], round(s['score'], 3), round(s['area_fraction'], 3)) for s in probe_result['segments']], 'void_fraction': round(probe_result['void_fraction'], 3)}}\n"
                "print({{'degenerate_inputs': degenerate}})"
            ),
        },
        {
            "md": (
                CHECK
                + "(a) One. In " + KAGGLE_RUN + " the drawn scene kept a single query, `stop sign` at score 0.601 covering the red "
                "disc (8.3 % of the image; IoU 0.991 with the drawn disc), and 91.7 % of the image was void: the other queries "
                "either scored below 0.5 or did not reach `mask_threshold` anywhere they won. (b) No. The same run labelled the "
                "whole blank image `sky-other-merged` at 0.798 and 94.9 % of the noise image `sky-other-merged` at 0.572. An "
                "out-of-domain image is not guaranteed a void map; a confident-looking score on nothing is exactly why scores "
                "are not probabilities. Your numbers can differ slightly on a GPU." + END + "\n\n"
                "## 7. The same model on an in-domain photograph\n\n"
                "The drawing is out of domain by construction. To show the capability the checkpoint was trained for, this cell "
                "fetches the model card's own widget example — COCO `val2017/000000039769.jpg`, two cats on a couch — from "
                "`images.cocodataset.org` over plain HTTP and **refuses it unless its SHA-256 equals the pinned digest**, so a "
                "changed or intercepted file cannot silently become the sample. The photograph is used for display and "
                "inference only; it is not bundled or redistributed by the repository, and its individual Flickr licence is not "
                "verified here. No reference map exists for it, so nothing is scored: the cell records the label set the model "
                "produced and checks it against the labels a reader would expect (`cat`, `couch`, `remote`) as a **plausibility "
                "observation**, not a metric. If the host is unreachable the cell stops with the error rather than skipping "
                "silently; set `FETCH_PUBLIC_PHOTO = False` to run without it.\n\n"
                "**Predict before running:** on a real COCO photograph of two cats on a couch, how much of the image will be "
                "void, and which labels will appear?\n\n"
                "**What to notice:** the digest check passing, the segment list with labels and areas, `expected_labels_found`, "
                "`unexpected_labels`, and the void fraction compared with the drawn scene's."
            ),
            "code": (
                "import urllib.request\n\n"
                'FETCH_PUBLIC_PHOTO = True  # @param {{type:"boolean"}}\n'
                "PHOTO_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'\n"
                "PHOTO_SHA256 = 'dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e'\n"
                "EXPECTED_LABELS = {{'cat', 'couch', 'remote'}}\n"
                "photo_result = None\n"
                "photo_observation = {{'fetched': False}}\n"
                "if FETCH_PUBLIC_PHOTO:\n"
                "    with urllib.request.urlopen(PHOTO_URL, timeout=60) as response:\n"
                "        photo_bytes = response.read()\n"
                "    photo_sha256 = hashlib.sha256(photo_bytes).hexdigest()\n"
                "    if photo_sha256 != PHOTO_SHA256:\n"
                "        raise RuntimeError(f'public photograph digest {{photo_sha256}} != pinned {{PHOTO_SHA256}}; refusing to use it')\n"
                "    photo = Image.open(io.BytesIO(photo_bytes))\n"
                "    photo.load()\n"
                "    photo_manifest = validate_inputs(photo, score_threshold=score_threshold, overlap_threshold=overlap_threshold, mask_threshold=mask_threshold, names=['coco_val2017_000000039769.jpg'])\n"
                "    t0 = time.time()\n"
                "    photo_result = pipe.segment(photo, score_threshold=score_threshold, overlap_threshold=overlap_threshold, mask_threshold=mask_threshold)\n"
                "    found = {{s['label'] for s in photo_result['segments']}}\n"
                "    photo_observation = {{'fetched': True, 'url': PHOTO_URL, 'sha256': photo_sha256, 'bytes': len(photo_bytes), 'size': photo.size, 'seconds': round(time.time() - t0, 2),\n"
                "                         'segments': [(s['label'], round(s['score'], 3), round(s['area_fraction'], 3)) for s in photo_result['segments']],\n"
                "                         'void_fraction': round(photo_result['void_fraction'], 3), 'expected_labels': sorted(EXPECTED_LABELS), 'expected_labels_found': sorted(found & EXPECTED_LABELS), 'unexpected_labels': sorted(found - EXPECTED_LABELS)}}\n"
                "    print(json.dumps(photo_observation, indent=2))\n"
                "    display(photo)\n"
                "else:\n"
                "    print('public photograph skipped (FETCH_PUBLIC_PHOTO = False)')"
            ),
        },
        {
            "md": (
                CHECK
                + "Very little: " + KAGGLE_RUN + " found five segments — two `cat` (0.998, 0.997), two `remote` (0.994, 0.953) "
                "and a `couch` (0.811) — with 4.9 % void, and no unexpected label. In domain, the same model fills almost the "
                "whole frame; compare the 91.7 % void on the drawing. That is a plausibility observation on one photograph, not "
                "an accuracy: nothing checks where the cat boundaries are." + END + "\n\n"
                "## 8. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: panoptic quality needs panoptic annotations on images from the deployment domain with a "
                "vocabulary matching the model's, and this repository ships none (COCO panoptic is not bundled). The repository's "
                "metric helper is `panoptic_quality` — Kirillov et al.'s PQ: segments are matched at IoU > 0.5, `SQ` is the mean "
                "IoU of the matches, `RQ` is `TP / (TP + FP/2 + FN/2)`, and `PQ = SQ × RQ` — and when a reference map is "
                "supplied the report carries `pq`, `sq`, `rq` and the match counts with the verdict `sample-sanity`. On the "
                "synthetic path the references are regions **you drew yourself** and their names are not COCO categories, so "
                "the match is **class-agnostic** (IoU only; the predicted labels are recorded beside it): a high value proves "
                "only that the input contract, resize, decoder, post-processing and resampling round-trip. The photograph and "
                "any BYOD upload have no reference, so their verdict is `not-measurable` and the report states what would make "
                "the task measurable. The reports are written to `outputs/{stem}_evaluation_report.json` (sample) and "
                "`outputs/{stem}_photo_evaluation_report.json` (photograph, when fetched).\n\n"
                "**Predict before running:** the disc was matched almost perfectly (IoU 0.991). Will the drawn scene's PQ be "
                "close to 1?\n\n"
                "**What to notice:** the verdict and its reason, `pq`, `sq` and `rq`, the match counts (`tp`, `fp`, `fn`), and "
                "the photograph's `not-measurable` verdict with its `needs` text."
            ),
            "code": (
                "report = evaluation_report(result, reference, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'panoptic_quality')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    value = metric['value'] if not isinstance(metric['value'], float) else round(metric['value'], 4)\n"
                "    print(f\"{{metric['id']:8}} {{value}}  ({{metric['estimation']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference map exists for this input, so nothing is scored; inspect the overlay yourself.')\n"
                "photo_report = None\n"
                "if photo_result is not None:\n"
                "    photo_report = evaluation_report(photo_result, None, sample_kind='public-photo')\n"
                "    with open('outputs/{stem}_photo_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump(photo_report, handle, indent=2, ensure_ascii=False)\n"
                "    print({{'photo_verdict': photo_report['verdict'], 'photo_labels': photo_report['labels']}})"
            ),
        },
        {
            "md": (
                CHECK
                + "No. PQ counts every region: one match out of five drawn regions, with the other four (sky, ground, box, "
                "triangle) left void as false negatives, gives `RQ = 1 / (1 + 0 + 4/2) = 0.333`; `SQ` is the one match's IoU, "
                "about 0.99; so `PQ ≈ 0.33`. " + KAGGLE_RUN[0].upper() + KAGGLE_RUN[1:] + " scored PQ 0.33 with the verdict "
                "`sample-sanity`. A near-perfect mask on the one region the model recognised cannot make up for the regions it "
                "did not." + END + "\n\n"
                "## 9. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the segment list (label, score, area fraction, box, thing/stuff, fused flag), "
                "the thresholds, the evaluation reports, the input manifest, the degenerate-input probes, the photograph "
                "observation, the sample identity and digest, the notebook's source (repository, revision, embedded module "
                "digest, generator), the model identifier, the immutable model revision, the model licence, and the runtime "
                "identity (Python, `torch`, `transformers`, device); the panoptic maps themselves are written as 16-bit PNGs "
                "(segment id + 1, so void is 0) because arrays do not belong in JSON. An overlay PNG tints each segment in its "
                "own colour on the image for visual inspection (a supplement to, not a replacement for, the machine-readable "
                "files). No credentials are recorded. Before writing anything the cell checks that the map, the photograph "
                "result and the report were all made with the thresholds now in the Section 4 form; if you re-ran only some "
                "sections after changing a threshold it stops and tells you to **Run after** from Section 4, so an export never "
                "describes thresholds other than the ones it was made with.\n\n"
                "**What to notice:** the sorted list of files in `outputs/` (four JSON files, two 16-bit segmentation PNGs and "
                "two overlays when the photograph was fetched) and the overlay of the sample."
            ),
            "code": (
                "form_thresholds = {{'score_threshold': score_threshold, 'overlap_threshold': overlap_threshold, 'mask_threshold': mask_threshold}}\n"
                "made_with = {{'Section 6 map': {{k: result[k] for k in form_thresholds}}}}\n"
                "if photo_result is not None:\n"
                "    made_with['Section 7 photograph'] = {{k: photo_result[k] for k in form_thresholds}}\n"
                "made_with['Section 8 report'] = report['thresholds']\n"
                "stale = [section for section, used in made_with.items() if used != form_thresholds]\n"
                "if stale:\n"
                "    raise RuntimeError(f'{{\", \".join(stale)}} used other thresholds than the Section 4 form {{form_thresholds}}: select the Section 4 cell and choose Runtime -> Run after, so every section uses the same values before exporting')\n\n\n"
                "def save_maps(tag, source_image, seg_result):\n"
                "    seg = seg_result['segmentation']\n"
                "    Image.fromarray((seg + 1).astype(np.uint16)).save(f'outputs/{stem}_{{tag}}_segmentation.png')\n"
                "    palette = [(220, 40, 40), (40, 70, 200), (250, 200, 30), (60, 179, 75), (160, 60, 200), (0, 170, 170), (240, 120, 30), (120, 120, 120)]\n"
                "    base = np.asarray(source_image.convert('RGB'), dtype=np.float32)\n"
                "    overlay = base.copy()\n"
                "    for index, segment in enumerate(seg_result['segments']):\n"
                "        colour = np.array(palette[index % len(palette)], dtype=np.float32)\n"
                "        region = seg == segment['id']\n"
                "        overlay[region] = 0.45 * overlay[region] + 0.55 * colour\n"
                "    overlay[seg == -1] = 0.5 * overlay[seg == -1]  # void is darkened\n"
                "    path = f'outputs/{stem}_{{tag}}_overlay.png'\n"
                "    Image.fromarray(overlay.round().astype(np.uint8)).save(path)\n"
                "    return path\n\n\n"
                "overlay_path = save_maps('sample', image, result)\n"
                "photo_overlay = save_maps('photo', photo, photo_result) if photo_result is not None else None\n"
                "payload = {{\n"
                "    'segments': result['segments'],\n"
                "    'void_fraction': result['void_fraction'],\n"
                "    'thresholds': {{'score_threshold': result['score_threshold'], 'overlap_threshold': result['overlap_threshold'], 'mask_threshold': result['mask_threshold']}},\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'degenerate_inputs': degenerate,\n"
                "    'public_photo': {{'observation': photo_observation, 'evaluation_report': photo_report, 'segments': None if photo_result is None else photo_result['segments']}},\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'has_reference': reference is not None}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))\n"
                "Image.open(overlay_path)"
            ),
        },
        {
            "md": (
                "## 10. Your turn — change one thing: the score threshold\n\n"
                "The upstream Mask2Former inference keeps a query only when its best class probability reaches **0.8**; this "
                "notebook's default is the pinned `transformers` value, **0.5**. Change that one setting and see what it does to "
                "both images.\n\n"
                "1. **Predict.** At `score_threshold = 0.8`, how many segments will the drawn scene keep, and what will its PQ "
                "be? Will the photograph lose any of its five segments, and will its void fraction go up or down?\n"
                "2. **Change one thing.** In Section 4 set `score_threshold = 0.8` and leave everything else as it is.\n"
                "3. **Run.** Select the Section 4 cell and choose **Runtime → Run after**. That re-runs Sections 4–10 in order, "
                "so the map, the photograph, the report and the exports all use the new value (Section 9 refuses to export a "
                "mixture). Re-running a single section is not enough.\n"
                "4. **Observe.** The table below gains one row per completed pass, so the default run and your run sit side by "
                "side.\n"
                "5. **Explain.** Use the segment lists in Sections 6 and 7 to account for each change.\n\n"
                "**Optional experiments**, each the same way (change one value in Section 4, then **Run after** from Section 4): "
                "`score_threshold = 0.3`; `mask_threshold = 0.3`; and `USE_BYOD = True` with a photograph you know. When you are "
                "done, set the values back to 0.5 / 0.8 / 0.5 and **Run after** from Section 4 again.\n\n"
                "**What to notice:** the `segments`, `void` and `pq` columns for the drawn scene and `photo_segments` / "
                "`photo_void` for the photograph, row by row."
            ),
            "code": (
                "# One row per completed pass through Sections 4-9 in this session; the list survives a Run after.\n"
                "activity_runs = globals().get('activity_runs') or []\n"
                "pq_value = next((m['value'] for m in report['metrics'] if m['id'] == 'pq'), None)\n"
                "run_row = {{'score': score_threshold, 'mask': mask_threshold, 'overlap': overlap_threshold, 'sample': sample_kind,\n"
                "           'segments': len(result['segments']), 'void': round(result['void_fraction'], 3), 'pq': None if pq_value is None else round(pq_value, 3),\n"
                "           'photo_segments': None if photo_result is None else len(photo_result['segments']),\n"
                "           'photo_void': None if photo_result is None else round(photo_result['void_fraction'], 3)}}\n"
                "if not activity_runs or activity_runs[-1] != run_row:\n"
                "    activity_runs.append(run_row)\n"
                "print(' | '.join(f'{{key:>14}}' for key in run_row))\n"
                "for row in activity_runs:\n"
                "    print(' | '.join(f'{{str(value):>14}}' for value in row.values()))"
            ),
        },
        {
            "md": (
                CHECK
                + "In " + CPU_CHECK + ": at `score_threshold = 0.8` the drawn scene keeps **nothing** — its one query scored "
                "0.601 — so the map is entirely void (void 1.0) and PQ falls from 0.33 to 0.0. The photograph keeps all five "
                "segments, because the weakest, `couch`, scored 0.811, just above 0.8; its void **shrinks** from 4.9 % to 2.8 % "
                "and the couch grows from 56.6 % to 58.7 % of the image. A higher threshold removes weak queries, and each pixel "
                "then goes to the strongest query that is left, so fewer survivors does not have to mean more void.\n\n"
                "The optional experiments change little here, and that is the lesson: at `score_threshold = 0.3` nothing "
                "changes on either image, because no query scores between 0.3 and 0.5 on them; at `mask_threshold = 0.3` the "
                "drawn scene's void moves only from 0.917 to 0.916 (PQ 0.330 → 0.332) and the photograph's from 0.049 to "
                "0.036. A threshold can only act on queries and pixels near it. A GPU run can differ in the third decimal."
                + END
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A panoptic map is a hard assignment produced by a fixed post-processing rule over uncalibrated query scores; nothing "
        "in the output scores a segment as a whole in a calibrated way, and the three thresholds change what is kept and what "
        "is void. On the drawn scene the class-agnostic panoptic quality in the evaluation report compares the map with regions "
        "you drew yourself and the verdict is `sample-sanity`, which proves only that the input contract, resize, decoder, "
        "post-processing and resampling work (the recorded run matched the disc at IoU 0.991 as `stop sign` and left the other "
        "four regions void, PQ 0.33); it says nothing about photographs, and the photograph section is a single unscored "
        "observation with the verdict `not-measurable`. **An out-of-domain image is not guaranteed a void map** — the blank "
        "and noise probes were labelled `sky-other-merged` with scores of 0.80 and 0.57 — and an in-domain photograph is not "
        "guaranteed a complete one. The pipeline provides one vocabulary (COCO's 133 categories), no calibration, no "
        "benchmark evaluation and no training capability; adaptation to your own classes is the companion `E2E` notebook's job.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire and "
        "digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and emit the shown "
        "machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** establish "
        "benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen "
        "domain.\n\n"
        "**Further experiments:** besides the Section 10 activity, build a reference region map of your own (an int32 array "
        "of ids plus a segment list, as `synthetic_scene` returns) and pass it to `evaluation_report` to see the verdict switch "
        "to `sample-sanity`; then open the `E2E` notebook to give the model your own vocabulary.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with `This notebook needs a Linux x86_64 runtime`.** Use Google Colab, Kaggle or a Linux Jupyter host.\n"
        "- **`The pinned uv wheel failed its size/SHA-256 check`.** Run Section 1 again; if it repeats, the download is being altered on the way.\n"
        "- **`The isolated environment's Python process exited`.** The worker crashed, usually out of memory: restart the session and choose **Run all**.\n"
        "- **You want to start over, or a cell behaves as if old values were still set.** Every cell after Section 1 runs in one persistent worker; restart the session and choose **Run all**.\n"
        "- **A size or SHA-256 error in Section 3.** A staged weight file is incomplete or altered: delete it under `weights/` and run Section 3 again (it re-fetches only missing files).\n"
        "- **`public photograph digest … != pinned` or a timeout in Section 7.** The photograph host changed or is unreachable: set `FETCH_PUBLIC_PHOTO = False` and **Run after** from Section 7.\n"
        "- **Section 9 stops with `used other thresholds than the Section 4 form`.** You changed a threshold and re-ran only some sections: select Section 4 and choose **Runtime → Run after**.\n"
        "- **BYOD refusals (Section 4).** Each message names the rule and the fix: `BYOD_PATH … is not a file` (fix the path), `Upload exactly one image` (the upload was cancelled, empty or had several files), `needs the Google Colab upload dialog` (set `BYOD_PATH` outside Colab), `is not an image Pillow can read`, or `has image mode 'I;16'` (save the image as 8-bit first). Section 5 refuses an image side below 16 px or above 4096 px.\n\n"
        "## Glossary\n\n"
        "- **Panoptic segmentation:** every pixel gets exactly one segment — a countable *thing* instance or an amorphous *stuff* region — or void.\n"
        "- **Thing / stuff:** countable objects, one segment per instance (a cat, a remote, a couch), versus amorphous regions with one fused segment per class (`sky-other-merged`, grass, wall).\n"
        "- **Query:** one of 100 learned slots in the decoder; each proposes a mask and a class, and a fixed rule merges them into the map.\n"
        "- **Void:** pixels that no surviving query claims with enough mask probability; stored as `-1` in the map and 0 in the exported PNG.\n"
        "- **Thresholds:** `score_threshold` (keep a query), `mask_threshold` (per-pixel cut), `overlap_threshold` (drop a query that kept too little of its own mask).\n"
        "- **Uncalibrated score:** the query's softmax class probability; it ranks queries but is not the chance the label is right.\n"
        "- **IoU:** intersection over union of two masks.\n"
        "- **PQ = SQ × RQ:** panoptic quality — the mean IoU of matched segments (SQ) times a recognition quality that penalises missed and false segments (RQ).\n"
        "- **`sample-sanity` / `not-measurable`:** the report's verdicts when a drawn reference exists, and when no reference exists.\n"
        "- **Input manifest:** the JSON record of what was validated — schema, observed input, thresholds, verdict and findings.\n"
        "- **Isolated environment:** the separate hash-locked Python environment built in Section 1; every later cell runs there.\n\n"
        "## Conclusion (your notes)\n\n"
        "Answer from your own run, citing the numbers it printed:\n\n"
        "1. In two sentences: what does the map guarantee, and what does a segment's score not tell you?\n"
        "2. Why was the drawn scene mostly void while the photograph was almost fully covered?\n"
        "3. Which of your predictions were wrong, and what did the output show instead?\n"
        "4. What did changing `score_threshold` in Section 10 change on each image, and why?\n"
        "5. What reference data would you need before quoting a PQ for your own images?\n\n"
        "**Your notes:** (write them in a new text cell below)\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/mask2former-panoptic-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/mask2former-panoptic-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/mask2former-panoptic-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/Mask2Former\n"
        "- Masked-attention Mask Transformer for Universal Image Segmentation (Cheng et al., 2021): https://arxiv.org/abs/2112.01527\n"
        "- Panoptic Segmentation (Kirillov et al., 2019): https://arxiv.org/abs/1801.00868"
    ),
}
