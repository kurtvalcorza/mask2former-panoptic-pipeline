"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package
modules, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "mask2former_panoptic_pipeline",
    "repo_name": "mask2former-panoptic-pipeline",
    "stem": "mask2former_panoptic",
    "notebook_name": "mask2former_panoptic_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
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
        "COCO benchmark, and the photograph has no reference so its report is `not-measurable`."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable "
        "upstream model revision, draw a synthetic scene with known regions (or upload your own photograph) and validate "
        "it into an input manifest, choose the three post-processing thresholds, run the supported task, read a panoptic "
        "map correctly (segment ids, void, thing versus stuff, uncalibrated scores), see the same model on an in-domain "
        "photograph and on degenerate inputs, produce an evaluation report that is `sample-sanity` with panoptic quality "
        "only when reference regions exist and `not-measurable` otherwise, and export the maps, overlays and provenance."
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
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 18 s to load and 1.7–8 s per 640×480 image (decoder pass plus NumPy/Pillow post-processing) in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 190 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a segmentation map is; the difference between countable *things* and amorphous *stuff*; why a softmax score is not a calibrated probability; what intersection-over-union measures and how panoptic quality (PQ = SQ × RQ) is built from it.",
        "- **Data:** the default sample is a deterministic 640×480 scene drawn in code with Pillow (a red disc, a blue box, a yellow triangle and a green ground band on an off-white sky; no text rendering, so its digest is stable across Pillow builds) with the exact region map it was drawn from, so nothing is downloaded and no private data is needed. Section 7 additionally fetches one public photograph over plain HTTP (`images.cocodataset.org/val2017/000000039769.jpg`, 173,131 bytes; the checkpoint's own model-card widget example) and refuses it unless its SHA-256 matches the pinned digest; it is used only for display and inference, never redistributed, and its individual Flickr licence is not verified by this repository. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px; no reference map exists for uploads, so their report is `not-measurable`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
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
                "image digest is printed for the record. BYOD is optional and disabled by default; when enabled, upload one "
                "image — no reference map exists for it, so the evaluation report will be `not-measurable`.\n\n"
                "The three post-processing thresholds are **caller-owned request parameters**: `score_threshold` gates queries "
                "on their best class probability (`SCORE_THRESHOLD = 0.5` is the pinned `transformers` default; the upstream "
                "Mask2Former inference uses 0.8 — on the drawn scene 0.5 keeps one segment and 0.8 keeps none, as the smoke run "
                "recorded), `mask_threshold` is the per-pixel cut that separates a segment from void, and `overlap_threshold` "
                "drops a query that lost most of its mask to stronger queries. Nothing is validated in this cell — the next "
                "section hands the image to the pipeline's own validation stage, which is the only checker. Look for a "
                "dictionary naming the sample kind, the image size and digest, the thresholds and the reference regions."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'score_threshold = 0.5  # @param {{type:"number"}}\n'
                'overlap_threshold = 0.8  # @param {{type:"number"}}\n'
                'mask_threshold = 0.5  # @param {{type:"number"}}\n\n'
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
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
                "       'thresholds': {{'score': score_threshold, 'overlap': overlap_threshold, 'mask': mask_threshold}},\n"
                "       'reference_regions': None if reference is None else [(s['name'], 'stuff' if not s['is_thing'] else 'thing', round(s['area_fraction'], 3)) for s in reference[1]]}})\n"
                "image"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `segment` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px and three thresholds in `[0, 1]` — and returns an "
                "**input manifest** naming the schema (including the 384×384 resize that does not preserve aspect ratio, the "
                "100 queries, the 96×96 mask logits and the resampling), the input's observed mode and size, the thresholds and "
                "the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, "
                "the cell also validates a request whose `score_threshold` is outside `[0, 1]` and records the pipeline's own "
                "error message as a finding. Inside the pipeline the image is converted to RGB and resized; nothing else is "
                "dropped or altered. The pipeline cannot tell whether an image is a photograph: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'INPUT_SIZE': INPUT_SIZE, 'MASK_LOGIT_SIZE': MASK_LOGIT_SIZE, 'NUM_QUERIES': NUM_QUERIES, 'SCORE_THRESHOLD': SCORE_THRESHOLD, 'OVERLAP_THRESHOLD': OVERLAP_THRESHOLD, 'MASK_THRESHOLD': MASK_THRESHOLD}}}})\n"
                "input_manifest = validate_inputs(image, score_threshold=score_threshold, overlap_threshold=overlap_threshold, mask_threshold=mask_threshold, names=[image_name])\n"
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
                "## 6. Segment the scene and read the output correctly\n\n"
                "`segment` returns `segmentation` (an int32 map at input resolution: the segment id per pixel, `-1` for void), "
                "`segments` (one entry per segment, ordered by score: `label`, `label_id`, `is_thing`, `score`, `area_fraction`, "
                "tight `bbox`, `was_fused`, the winning `query`), the `void_fraction`, the three thresholds, the vocabulary and "
                "the model identity. **Scores are uncalibrated softmaxes**: a 0.9 query is not 90 % likely to be right, and the "
                "map is a hard assignment — every pixel has exactly one segment or none. Post-processing is deterministic on a "
                "fixed device and dtype; CUDA kernels can shift logits slightly, so GPU and CPU maps need not match at a boundary. "
                "As recorded in the model card, the repository's CPU smoke on this scene kept a single query — `stop sign` at "
                "score 0.601 covering the red disc (IoU 0.991 with the drawn disc) — and left 91.7 % of the image void; at "
                "`score_threshold = 0.8` it kept nothing. The pinned `transformers` post-processor, by contrast, hands the whole "
                "frame to that one query (area 1.0), which is why the carried module implements the upstream per-pixel rule "
                "itself. The cell also probes two degenerate inputs (a blank white image and uniform noise) and records what "
                "the model says about nothing: the smoke run got `sky-other-merged` at 0.798 over the whole blank image and at "
                "0.572 over 95 % of the noise — an observation about this checkpoint, not a guarantee."
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
                "## 7. The same model on an in-domain photograph\n\n"
                "The drawing is out of domain by construction. To show the capability the checkpoint was trained for, this cell "
                "fetches the model card's own widget example — COCO `val2017/000000039769.jpg`, two cats on a couch — from "
                "`images.cocodataset.org` over plain HTTP and **refuses it unless its SHA-256 equals the pinned digest**, so a "
                "changed or intercepted file cannot silently become the sample. The photograph is used for display and "
                "inference only; it is not bundled or redistributed by the repository, and its individual Flickr licence is not "
                "verified here. No reference map exists for it, so nothing is scored: the cell records the label set the model "
                "produced and checks it against the labels a reader would expect (`cat`, `couch`, `remote`) as a **plausibility "
                "observation**, not a metric. The smoke run found five segments — two `cat` (0.998, 0.997), two `remote` (0.994, "
                "0.953) and `couch` (0.811) — with 4.9 % void. If the host is unreachable the cell stops with the error rather "
                "than skipping silently; set `FETCH_PUBLIC_PHOTO = False` to run without it."
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
                "## 8. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: panoptic quality needs panoptic annotations on images from the deployment domain with a "
                "vocabulary matching the model's, and this repository ships none (COCO panoptic is not bundled). The repository's "
                "metric helper is `panoptic_quality` — Kirillov et al.'s PQ: segments are matched at IoU > 0.5, `SQ` is the mean "
                "IoU of the matches, `RQ` is `TP / (TP + FP/2 + FN/2)`, and `PQ = SQ × RQ` — and when a reference map is "
                "supplied the report carries `pq`, `sq`, `rq` and the match counts with the verdict `sample-sanity`. On the "
                "synthetic path the references are regions **you drew yourself** and their names are not COCO categories, so "
                "the match is **class-agnostic** (IoU only; the predicted labels are recorded beside it): a high value proves "
                "only that the input contract, resize, decoder, post-processing and resampling round-trip. The smoke run scored "
                "PQ 0.33 here (one match, the disc, at IoU 0.991; four regions unmatched). The photograph and any BYOD upload "
                "have no reference, so their verdict is `not-measurable` and the report states what would make the task "
                "measurable. The reports are written to `outputs/{stem}_evaluation_report.json` (sample) and "
                "`outputs/{stem}_photo_evaluation_report.json` (photograph, when fetched)."
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
                "## 9. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the segment list (label, score, area fraction, box, thing/stuff, fused flag), "
                "the thresholds, the evaluation reports, the input manifest, the degenerate-input probes, the photograph "
                "observation, the sample identity and digest, the notebook's source (repository, revision, embedded module "
                "digest, generator), the model identifier, the immutable model revision, the model licence, and the runtime "
                "identity (Python, `torch`, `transformers`, device); the panoptic maps themselves are written as 16-bit PNGs "
                "(segment id + 1, so void is 0) because arrays do not belong in JSON. An overlay PNG tints each segment in its "
                "own colour on the image for visual inspection (a supplement to, not a replacement for, the machine-readable "
                "files). No credentials are recorded."
            ),
            "code": (
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
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A panoptic map is a hard assignment produced by a fixed post-processing rule over uncalibrated query scores; nothing "
        "in the output scores a segment as a whole in a calibrated way, and the three thresholds change what is kept and what "
        "is void. On the drawn scene the class-agnostic panoptic quality in the evaluation report compares the map with regions "
        "you drew yourself and the verdict is `sample-sanity`, which proves only that the input contract, resize, decoder, "
        "post-processing and resampling work (the repository's smoke run matched the disc at IoU 0.991 as `stop sign` and left "
        "the other four regions void, PQ 0.33); it says nothing about photographs, and the photograph section is a single "
        "unscored observation with the verdict `not-measurable`. **An out-of-domain image is not guaranteed a void map** — the "
        "blank and noise probes were labelled `sky-other-merged` with scores of 0.80 and 0.57 — and an in-domain photograph is "
        "not guaranteed a complete one. The pipeline provides one vocabulary (COCO's 133 categories), no calibration, no "
        "benchmark evaluation and no training capability; adaptation to your own classes is the companion `E2E` notebook's job.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire and "
        "digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and emit the shown "
        "machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** establish "
        "benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen "
        "domain.\n\n"
        "**Next experiments:** set `score_threshold` to 0.8 (the upstream value) and 0.3 and watch the drawn scene lose and gain "
        "segments; lower `mask_threshold` to 0.3 and see void shrink; enable `USE_BYOD` with a photograph you know; build a "
        "reference region map of your own (an int32 array of ids plus a segment list) and pass it to `evaluation_report` to see "
        "the verdict switch to `sample-sanity`; then open the `E2E` notebook to give the model your own vocabulary.\n\n"
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
