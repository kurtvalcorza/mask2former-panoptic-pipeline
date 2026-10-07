"""Template for the standalone E2E adaptation notebook (tools/build_notebook.py --template tools/notebook_template_finetune.py).

Same generator, same carried package and pins as the TASK-INFERENCE notebook; only the profile and the
stage cells differ. Section 3's `pipe` is the COCO-vocabulary pipeline; the adaptation stages build a
second, re-headed pipeline from the same verified snapshot without any further download.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "mask2former_panoptic_pipeline",
    "repo_name": "mask2former-panoptic-pipeline",
    "stem": "mask2former_panoptic_finetune",
    "notebook_name": "mask2former_panoptic_finetune_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline, siglip-v1-zero-shot-pipeline): a
    # managed CPython, a size- and SHA-256-verified uv wheel, and the lock shared with the TASK-INFERENCE notebook,
    # compiled from `pins_file` (the pyproject pins plus scipy==1.18.1, which transformers 4.57.6 requires to build
    # Mask2FormerLoss in Mask2FormerForUniversalSegmentation.__init__; review M2P-M1, PR #7). See
    # tools/notebook_template.py for the compile command.
    "pins_file": "tutorials/requirements-colab.in",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has run a Colab or Jupyter notebook, and wants to see the whole adaptation cycle of a segmentation model — data contract, split, baselines, fine-tuning, held-out evaluation, export and a cold reload — on a task small enough to finish on a free runtime. No prior experience with Mask2Former or fine-tuning is assumed; *panoptic quality*, *re-heading*, *set loss*, *frozen backbone* and the other terms are explained where they first matter and again in the **Glossary**. The intended audience is learners and practitioners preparing to adapt a segmenter to their own classes; this is a teaching run on drawn shapes, not a benchmark. CPU works (about two minutes of training); a T4 GPU is much faster.\n\n**Input → Model → Output.**\n\n| | What it is in this notebook |\n|---|---|\n| Input | 24 labelled 320×240 scenes drawn in code (two stuff classes, three thing classes), split 18 / 6; BYOD: your own image, or your own labelled directory |\n| Model | Mask2Former Swin-T from COCO, re-headed for the five-class vocabulary; the pixel decoder, transformer decoder and new head train with the upstream set loss while the backbone stays frozen |\n| Output | held-out panoptic quality (PQ, SQ, RQ, per class) beside a re-headed baseline and two trivial predictors (all-`sky`, horizon split); segment maps for new scenes; a 190 MB self-describing artifact that reloads to identical scores |\n\n**How to use this notebook.** Choose **Runtime → Change runtime type → T4 GPU** (CPU also works), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** and their cells are collapsed. The learning path starts in Section 4. Form fields (`# @param`) are the knobs; after changing one, re-run Section 7 (the fine-tune rebuilds the re-headed model when it has already been trained, so training never stacks on an earlier run) and Section 8, or select Section 6 and choose **Runtime → Run after**; Section 13 walks through one such experiment. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the recorded Kaggle T4 run of 18 September 2026 (`docs/release-verification.md`; a CPU run moves the per-class numbers). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end.\n\n**Roadmap:** 1–3 infrastructure → 4 the COCO checkpoint on a drawn scene → 5 the labelled sample and the validation stage → 6 split, re-head and two baselines *(evaluation practice)* → 7 the bounded fine-tune *(core concept)* → 8 held-out evaluation → 9 new data → 10 export and cold reload *(engineering)* → 11 outputs → 12 your own image or dataset (optional) → interpretation and limits → 13 your turn: change one thing *(activity)* → troubleshooting, glossary, conclusion."
            )
        ]
    },
    "pipeline_class": "Mask2FormerPanopticPipeline",
    "weights_key": "mask2former-swin-tiny-coco-panoptic",
    "modules": ["samples.py", "pipeline.py"],
    "runtime_imports": ["torch", "transformers"],
    "title": "Mask2Former Swin-T COCO panoptic — DIMER end-to-end adaptation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/mask2former-panoptic-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/mask2former-panoptic-pipeline/blob/main/tutorials/mask2former_panoptic_finetune_colab.ipynb",
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
    "capability": "bounded gradient adaptation of the pinned `facebook/mask2former-swin-tiny-coco-panoptic` weights to a caller-supplied panoptic vocabulary (five drawn classes: two stuff, three things) — dataset validation, seeded split, re-headed baseline, fine-tune with the upstream set loss, held-out panoptic quality, new-data inference, artifact export and fresh reload",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned snapshot, shows what the COCO-vocabulary checkpoint does on a drawn scene, draws and validates a 24-image "
        "labelled panoptic dataset, splits it 18/6 with a seed, rebuilds the classification head for the five-class vocabulary "
        "and measures the held-out baseline (plus a trivial all-`sky` baseline), **runs the bounded fine-tune** (6 epochs, 54 "
        "optimizer steps, backbone frozen), re-scores the held-out split with panoptic quality, segments unseen scenes, exports "
        "the adapted model as a self-describing directory, reloads it from disk and confirms the reloaded model reproduces the "
        "held-out score exactly, and writes every result as JSON with provenance. Nothing is skipped behind a default-off flag, "
        "and the path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no "
        "configuration edit (NOTEBOOK_SPEC 2.0 §5, RUN7, FT2). It runs on CPU in a few minutes (the repository's smoke run "
        "trained in 185 s on the reference workstation) and faster on a CUDA GPU."
    ),
    "byod": (
        "Two BYOD branches are provided and both are optional and off by default. `USE_BYOD_IMAGE` runs your own image "
        "through the adapted model with the same validation and inference contract as the drawn scenes. `USE_BYOD_DATASET` "
        "takes your own labelled panoptic records (a directory with `dataset.json`, images and instance-id PNGs, read by the "
        "carried `load_labelled_dir`) and runs them through the *same* local stages the sample used — validate, split, "
        "baseline, fine-tune, evaluate, export, reload — rather than only segmenting with them, because this is an adaptation "
        "profile (NOTEBOOK_SPEC 2.0 §25.10). The expected record shape and the ceilings are stated in the Prerequisites and "
        "printed by the cell. Each branch reads `BYOD_IMAGE_PATH` / `BYOD_DATASET_PATH` on Kaggle or Jupyter (a dataset may be a "
        "zip or a directory) and falls back to the Colab upload dialog when the path is empty; uploads stay inside this runtime."
    ),
    "intro": (
        "Mask2Former treats panoptic, instance and semantic segmentation as one problem: predict a set of (class, mask) pairs. "
        "That makes its head cheap to replace — the class predictor is one linear layer over the query embeddings — while the "
        "Swin-T backbone, the multi-scale deformable-attention pixel decoder and the masked-attention transformer decoder keep "
        "everything they learned on COCO. This notebook does exactly that: `from_pretrained(class_names=..., stuff_names=...)` "
        "loads the same digest-verified checkpoint, rebuilds the class head for a five-class drawn vocabulary (`sky` and `ground` "
        "as stuff, `disc`, `box` and `triangle` as things) with a seeded random initialisation of exactly the mismatched tensors, "
        "and `finetune` runs a bounded plain-PyTorch AdamW loop with the **upstream set loss** (Hungarian matching of queries to "
        "instances, class cross-entropy, point-sampled mask BCE and dice, auxiliary decoder losses) on 18 drawn scenes, backbone "
        "frozen. **What is trained and what is not:** the transformer decoder, the pixel decoder and the new class head "
        "(19.9 M of 47.4 M parameters) train; the Swin-T backbone is frozen and kept in eval mode. Evaluation is panoptic quality "
        "on 6 held-out scenes through the same `segment` call as inference, before and after adaptation, beside a trivial "
        "all-`sky` baseline. **Training loss is optimisation evidence only** (FT7); the held-out PQ is the task evidence, and it is "
        "a sample metric on drawn shapes, not a benchmark. The artifact is the full adapted model (SafeTensors + configs + a "
        "manifest with digests and provenance), reloaded from disk by `load_artifact` and re-scored to prove the boundary."
    ),
    "learning_objectives": (
        "by the end you can **explain** why the COCO checkpoint fails on drawn shapes and why a freshly re-headed model scores "
        "PQ 0.0; **read** a dataset manifest and a split record and **state** what the held-out scenes protect against; "
        "**predict** the direction of the loss and of the held-out score before each cell runs, then check it; **run** the "
        "bounded fine-tune and **state** what its loss history is and is not evidence of; **compare** the adapted held-out PQ "
        "with the re-headed baseline and the two trivial predictors, using things PQ as the measure of what adaptation added; "
        "**find** the training record and the digests in the exported artifact and **explain** what the cold reload proves; "
        "**change** one training choice (epochs) in the Section 13 activity and **explain** the observed effect."
    ),
    "exclusions": (
        "Adapting the backbone (frozen here; unfreezing it on 18 images is a recipe for forgetting), the upstream training "
        "recipe (COCO, 50 epochs, 12,544 sampled points per mask, large-scale jitter), any photographic dataset (the drawn "
        "scenes are the whole sample; PQ on them says the plumbing works, not that the model would learn your photographs), "
        "hyperparameter search, mixed precision, multi-GPU training, calibration, and any claim about COCO panoptic quality. "
        "The pretrained COCO vocabulary is not preserved by the adaptation: the re-headed model knows only the five new classes."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; training and inference are float32 on both. CPU is adequate for the default path: the repository's model card records 185 s for the 54-step fine-tune (about 3.4 s per step at 384×384, batch 2) and 3–6 s per 6-image evaluation in the Windows venv (Intel Core Ultra 9 275HX); a T4 is several times faster. Section 1 builds a separate environment from the hash-locked pins (nothing is installed into the notebook's own Python, so no restart is needed); its PyTorch wheels and the 190 MB checkpoint are the large downloads of the run; the exported artifact is another 190 MB on disk.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a panoptic annotation is (an instance map plus a class per instance) and why stuff and things are treated differently; what a train/held-out split protects against; how set-prediction losses match queries to targets; what panoptic quality (PQ = SQ × RQ, IoU > 0.5 matching) measures.",
        "- **Data:** the default sample is drawn in code by the carried `shape_dataset` (24 scenes of 320×240 with a seeded generator: a `sky` region above a jittered horizon, a `ground` band below it and 1–3 things of jittered size, position and colour) with exact instance maps, so nothing is downloaded and no private data is needed; new-data scenes use a different seed. Expected BYOD dataset: a directory with `dataset.json` (`class_names`, `stuff_names`, `records` of `{id, image, mask, instances}`), one image per record decodable by Pillow, and one PNG instance map per record whose pixel values are instance ids in 1..254 with every pixel assigned; ceilings: 2..32 classes, at most 200 records, at most 50 instances per image, at most 20 epochs. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. What the COCO checkpoint does on a drawn scene\n\n"
                "Before adapting anything, see the starting point. `pipe` (Section 3) is the unmodified checkpoint with its 133 COCO "
                "categories. On `synthetic_scene` — a red disc, a blue box, a yellow triangle, a green ground band, an off-white "
                "sky — the repository's smoke run kept a single query, `stop sign` at score 0.601 on the disc, and left 91.7 % of the "
                "pixels void. That is not a bug: the checkpoint has no class for a drawn disc and the carried post-processing keeps "
                "void where no query is confident. The regions drawn here are the vocabulary the rest of the notebook teaches the "
                "model. Nothing is scored in this cell; it is the observation the adaptation is measured against.\n\n"
                "**Predict:** how much of the drawn scene will the COCO checkpoint leave void?\n\n"
                + "<details><summary>Check your reasoning</summary>\n\nMost of it: the recorded run kept one `stop sign` query at 0.601 over the disc and left 91.7 % void. The checkpoint has no class for drawn shapes — which is what the adaptation is for.\n\n</details>"
            ),
            "code": (
                "import json\n"
                "import os\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "scene_image, scene_segmentation, scene_segments = synthetic_scene()\n"
                "t0 = time.time()\n"
                "coco_result = pipe.segment(scene_image)\n"
                "coco_observation = {{'seconds': round(time.time() - t0, 2), 'vocabulary_size': len(pipe.class_names),\n"
                "                    'segments': [(s['label'], round(s['score'], 3), round(s['area_fraction'], 3)) for s in coco_result['segments']],\n"
                "                    'void_fraction': round(coco_result['void_fraction'], 3),\n"
                "                    'drawn_regions': [(s['name'], 'stuff' if not s['is_thing'] else 'thing', round(s['area_fraction'], 3)) for s in scene_segments]}}\n"
                "print(json.dumps(coco_observation, indent=2))\n"
                "scene_image"
            ),
        },
        {
            "md": (
                "## 5. Sample data for adaptation, and the validation stage\n\n"
                "`shape_dataset(N_IMAGES, seed=DATASET_SEED)` draws a deterministic labelled set over `SHAPE_CLASSES` — `sky` and "
                "`ground` as **stuff** (one amorphous region each), `disc`, `box` and `triangle` as **things** (1–3 per scene, "
                "jittered size, position and colour) — and returns exact instance maps because it knows where it drew. A thing "
                "drawn over another can hide it entirely; hidden instances are dropped, so scenes carry 3–5 instances.\n\n"
                "`validate_dataset` is the validation stage for this path and applies exactly the ceilings `finetune` applies, so "
                "a dataset it accepts cannot be refused later: record shape, instance map geometry (every pixel assigned, ids in "
                "1..254, map and mapping agree), labels drawn from the vocabulary, at most 50 instances per image, at most 200 "
                "images, at most 20 epochs. It returns a **dataset manifest** — schema, vocabulary, per-class instance counts, a "
                "digest of the whole set — and reports a class with no instances as a **finding** rather than an error. The cell "
                "also validates a deliberately broken record (an instance map with an unassigned pixel) and records the "
                "pipeline's own rejection. The manifest is written to `outputs/{stem}_dataset_manifest.json`.\n\n"
                + "<details><summary>Check your reasoning</summary>\n\nWhy validate before training? Because a dataset `validate_dataset` accepts cannot be refused later by `finetune`: the recorded run accepted 24 records and rejected the unassigned-pixel probe with the pipeline's own message, before any weight moved.\n\n</details>"
            ),
            "code": (
                'N_IMAGES = 24  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'EPOCHS = 6  # @param {{type:"integer"}}\n\n'
                "records = shape_dataset(N_IMAGES, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SHAPE_CLASSES, SHAPE_STUFF, epochs=EPOCHS)\n"
                "# Demonstrate rejection: an instance map with an unassigned (0) pixel breaks the contract.\n"
                "broken = dict(records[0])\n"
                "broken['instance_map'] = records[0]['instance_map'].copy()\n"
                "broken['instance_map'][0, 0] = 0\n"
                "try:\n"
                "    validate_dataset([broken], SHAPE_CLASSES, SHAPE_STUFF)\n"
                "except ValueError as exc:\n"
                "    dataset_manifest['findings'].append({{'record': 'unassigned-pixel-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_dataset_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(dataset_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in dataset_manifest.items() if k != 'schema'}}, indent=2))\n"
                "print('record shape expected by finetune:', dataset_manifest['schema']['record'])\n"
                "preview = Image.new('RGB', (960, 480))\n"
                "for index, record in enumerate(records[:6]):\n"
                "    preview.paste(record['image'], (320 * (index % 3), 240 * (index // 3)))\n"
                "print('first six scenes; instances of the first:', records[0]['instances'])\n"
                "preview"
            ),
        },
        {
            "md": (
                "## 6. Split, re-head, and measure the baselines *before* adapting\n\n"
                "`split_records` is a seeded permutation into a training part and a held-out part; every scene is drawn "
                "independently, so a random split is the right kind (SPL3). The split happens before any weight is touched and the "
                "held-out records are never shown to `finetune`, so the score in Section 8 is a score on scenes the adapted model "
                "has not seen. The 6 held-out scenes are used once, to report; nothing is selected on them (SPL6/SPL7).\n\n"
                "`Mask2FormerPanopticPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SHAPE_CLASSES, stuff_names=SHAPE_STUFF, seed=SEED)` "
                "loads the same verified snapshot (no download: the files are already staged and re-verified) but rebuilds the "
                "class predictor for the five-class vocabulary. It records exactly which tensors could not transfer — "
                "`class_predictor.weight`/`.bias` and the loss's `criterion.empty_weight` — and everything else comes from the COCO "
                "checkpoint. The head's random initialisation is seeded, because it is the only untrained part of the model and "
                "precisely what the baseline measures.\n\n"
                "Three baselines are recorded (EVAL10). The **re-headed model before adaptation** scored PQ 0.0 on CPU and on the "
                "T4: the mask branch still proposes plausible regions, but a random class head assigns them arbitrary classes, and "
                "PQ is class-aware. Two **trivial predictors** that learn nothing: one labels the whole image `sky` (the majority "
                "class; PQ 0.103 on the default split: 0.516 on `sky`, 0 elsewhere), and one splits the image at the training "
                "split's mean horizon row — `sky` above, `ground` below — which scores PQ 0.318 on the default split (stuff PQ "
                "0.796: `sky` 0.822, `ground` 0.770; things 0) because the generator always draws sky above ground. Both numbers "
                "are data-only and reproduce exactly on any CPU or GPU. The horizon predictor is the honest bar for the stuff "
                "classes, so the measure of what adaptation *added* is the **things PQ** (0 for both trivial predictors), not the "
                "mean PQ or the stuff PQ.\n\n"
                "**Predict:** the re-headed model still has COCO's mask branch. Why does it score 0.0 PQ before training?\n\n"
                + "<details><summary>Check your reasoning</summary>\n\nBecause PQ is class-aware and the new five-class head is random: the masks may be plausible, but every class is a guess. The recorded run: re-headed baseline PQ 0.0, trivial all-`sky` PQ 0.103.\n\n</details>"
            ),
            "code": (
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 0  # @param {{type:"integer"}}\n\n'
                "train_records, held_out = split_records(records, holdout=HOLDOUT, seed=SEED)\n"
                "print({{'train': len(train_records), 'held_out': len(held_out), 'train_instances': sum(len(r['instances']) for r in train_records), 'held_out_instances': sum(len(r['instances']) for r in held_out)}})\n\n"
                "adapter = Mask2FormerPanopticPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SHAPE_CLASSES, stuff_names=SHAPE_STUFF, seed=SEED)\n"
                "print({{'class_names': list(adapter.class_names), 'stuff_names': list(adapter.stuff_names), 'device': adapter.device, 'adapted': adapter.adapted, 'train_num_points': TRAIN_NUM_POINTS}})\n"
                "print('tensors re-initialised because they could not transfer from the COCO checkpoint:', adapter.reinitialised)\n\n"
                "t0 = time.time()\n"
                "baseline = adapter.evaluate(held_out)\n"
                "print({{'baseline_pq': round(baseline['pq'], 4), 'sq': round(baseline['sq'], 4), 'rq': round(baseline['rq'], 4), 'things_pq': round(baseline['things']['pq'], 4), 'stuff_pq': round(baseline['stuff']['pq'], 4), 'seconds': round(time.time() - t0, 1)}})\n\n"
                "# Trivial baseline: every pixel is the majority stuff class.\n"
                "trivial_pairs = []\n"
                "for record in held_out:\n"
                "    reference_map, reference_segments = reference_from_record(record, SHAPE_CLASSES, SHAPE_STUFF)\n"
                "    trivial_pairs.append((np.ones_like(reference_map), [{{'id': 1, 'label': 'sky', 'is_thing': False}}], reference_map, reference_segments))\n"
                "trivial = panoptic_quality(trivial_pairs)\n"
                "print({{'trivial_all_sky_pq': round(trivial['pq'], 4), 'per_class': {{k: round(v['pq'], 3) for k, v in trivial['per_class'].items()}}}})\n\n"
                "# Second trivial baseline: a geometry-only split at the training split's mean horizon (sky above, ground below).\n"
                "# It learns nothing about the things, so its things PQ is 0; it is the bar the stuff classes must be read against.\n"
                "horizon_rows = []\n"
                "for record in train_records:\n"
                "    ground_ids = [int(k) for k, v in record['instances'].items() if v == 'ground']\n"
                "    horizon_rows.append(int(np.argmax(np.isin(record['instance_map'], ground_ids).any(axis=1))))\n"
                "HORIZON_ROW = int(round(float(np.mean(horizon_rows))))\n"
                "horizon_pairs = []\n"
                "for record in held_out:\n"
                "    reference_map, reference_segments = reference_from_record(record, SHAPE_CLASSES, SHAPE_STUFF)\n"
                "    split_map = np.ones_like(reference_map)\n"
                "    split_map[HORIZON_ROW:, :] = 2\n"
                "    horizon_pairs.append((split_map, [{{'id': 1, 'label': 'sky', 'is_thing': False}}, {{'id': 2, 'label': 'ground', 'is_thing': False}}], reference_map, reference_segments))\n"
                "horizon = panoptic_quality(horizon_pairs)\n"
                "print({{'trivial_horizon_split_pq': round(horizon['pq'], 4), 'horizon_row': HORIZON_ROW, 'stuff_pq': round(horizon['stuff']['pq'], 4), 'things_pq': round(horizon['things']['pq'], 4), 'per_class': {{k: round(v['pq'], 3) for k, v in horizon['per_class'].items()}}}})"
            ),
        },
        {
            "md": (
                "## 7. The bounded fine-tune\n\n"
                "This is the cell that makes the notebook an `E2E` tutorial rather than an inference demo, and it runs in the "
                "default path. The loss is **upstream's**, inside the pinned `transformers` Mask2Former: Hungarian matching of the "
                "100 queries to the scene's instances, then class cross-entropy (weight 2.0), point-sampled mask binary "
                "cross-entropy (5.0) and dice (5.0) with a 0.1 weight on *no object*, plus the same loss on every auxiliary "
                "decoder layer. What the carried module owns is the bounded loop around it: the processor call that turns "
                "instance maps into per-instance masks and labels, batching, AdamW, the seed and the ceilings.\n\n"
                "Defaults worth understanding, all explicit (FT4/FT6): `TRAIN_NUM_POINTS = 4096` sampled points per mask instead "
                "of the upstream 12,544 (a CPU-time decision recorded in the artifact); learning rate 1e-4 and weight decay 0.05 "
                "(the upstream optimizer values); batch size 2; 6 epochs over 18 scenes = 54 steps; float32; **the Swin-T backbone "
                "is frozen** and kept in eval mode, so 19.9 M of the 47.4 M parameters train (the pixel decoder, the transformer "
                "decoder and the new head). The smoke run's mean epoch loss went 33.7 → 6.8. **The loss history is optimisation "
                "evidence only** (FT7): whether the model learned the task is Section 8's question.\n\n"
                "**Re-running this cell never stacks training.** If `adapter` has already been fine-tuned, the cell rebuilds the "
                "re-headed model from the verified snapshot with the same `SEED` (so the Section 6 baseline still describes it) "
                "and trains that fresh model, so every run record's `epochs` and `steps` count exactly the training the weights "
                "received. To change `EPOCHS` (Section 5), `LEARNING_RATE`, `BATCH_SIZE` or `FREEZE_BACKBONE` (this cell), edit "
                "the field, then re-run this cell and Section 8 — or select Section 6 and choose **Runtime → Run after**.\n\n"
                "**Predict:** will the loss fall by a factor of about 2, 5 or 10 over 6 epochs?\n\n"
                + "<details><summary>Check your reasoning</summary>\n\nAbout 5: the recorded T4 run went 33.6 → 6.7 mean epoch loss in 54 steps (16.2 s). That says the optimiser worked; it does not say the segmentation is right.\n\n</details>"
            ),
            "code": (
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 2  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = True  # @param {{type:"boolean"}}\n\n'
                "if adapter.adapted:\n"
                "    # A re-run must not continue training the adapted weights (the run record would then misstate epochs and steps):\n"
                "    # rebuild the re-headed model from the verified snapshot with the same seed, as Section 6 did.\n"
                "    print({{'rebuilt_from_snapshot': True, 'reason': 'adapter was already fine-tuned; this run trains a fresh re-headed model with seed SEED so epochs and steps in the run record are exact'}})\n"
                "    adapter = Mask2FormerPanopticPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SHAPE_CLASSES, stuff_names=SHAPE_STUFF, seed=SEED)\n"
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_backbone=FREEZE_BACKBONE,\n"
                "    progress=lambda row: print(f\"epoch {{row['epoch']}}/{{EPOCHS}}  steps {{row['steps']}}  mean loss {{row['mean_loss']:.3f}}  last {{row['last_loss']:.3f}}  {{row['seconds']:.0f}} s\"),\n"
                ")\n"
                "print(json.dumps({{k: run[k] for k in ('adaptation', 'loss', 'optimizer', 'learning_rate', 'weight_decay', 'epochs', 'batch_size', 'steps', 'seed', 'precision', 'train_num_points', 'freeze_backbone', 'freeze_pixel_decoder', 'trainable_parameters', 'total_parameters', 'seconds', 'device')}}, indent=2))\n"
                "first, last = run['history'][0]['mean_loss'], run['history'][-1]['mean_loss']\n"
                "print(f'mean epoch loss {{first:.3f}} -> {{last:.3f}} over {{run[\"epochs\"]}} epochs (optimisation evidence only)')"
            ),
        },
        {
            "md": (
                "## 8. Evaluate on the held-out split → evaluation report\n\n"
                "The same `evaluate` call as the baseline, on the same held-out records, with the same thresholds — so the numbers "
                "are comparable and the only thing that changed is the weights. `evaluate` runs `segment` (the inference path, "
                "EVAL8) on every scene and accumulates **panoptic quality** per class: segments are matched at IoU > 0.5, `SQ` "
                "is the mean IoU of the matches, `RQ` is `TP / (TP + FP/2 + FN/2)`, and `PQ = SQ × RQ`, averaged over the five "
                "classes and reported for things and stuff separately. The estimation procedure is a single seeded holdout "
                "(EVAL5); there is no dispersion estimate.\n\n"
                "What this number is: evidence that a bounded fine-tune on 18 drawn scenes moved a held-out score from 0.0, past "
                "the all-`sky` predictor (0.103) and past the horizon-split predictor (0.318). What it is not: a benchmark. The "
                "held-out split is six scenes from the same generator with the same three shapes, so a high PQ says the task is "
                "easy and the adaptation worked — not that the model would segment your photographs. Read the **things PQ** as the "
                "measure of what adaptation added: both trivial predictors score 0 on the things, and the stuff PQ near 1.0 is "
                "only a little above the horizon predictor's 0.796.\n\n"
                "**Per-class values depend on the device.** The mean held-out PQ was 0.866 on both recorded runs, but the "
                "per-class and SQ/RQ values moved: on the CPU smoke run SQ 0.938, RQ 0.911, `box` 0.509, `disc` 0.841, "
                "`triangle` 0.981; on the Kaggle T4 run SQ 0.985, RQ 0.880, `box` 0.397, `disc` 0.951, `triangle` 0.986 (the "
                "head's CUDA initialisation and kernels differ). Your run will land near one of these, not on either. `box` was "
                "the weakest thing class on both devices; that its straight edges are confused with the ground band is a "
                "**hypothesis** this notebook does not test. The report is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Warnings you will see, and why they are expected:** `RuntimeWarning: overflow encountered in exp` comes from "
                "the carried post-processing's sigmoid on large negative mask logits (it saturates to 0, which is the intended "
                "value — harmless); the processor warning about `_max_size` / `reduce_labels` names keys in the pinned "
                "`preprocessor_config.json` that the pinned `transformers` ignores; the slow-image-processor notice says no fast "
                "processor is used (none is needed); and `You should probably TRAIN this model` appears when Section 6 rebuilds "
                "the class head — it is exactly what Section 7 then does.\n\n"
                "**Predict:** which class will stay weakest after adaptation, and why?\n\n"
                + "<details><summary>Check your reasoning</summary>\n\n`box`, on both recorded devices: the Kaggle T4 run reached PQ 0.866 overall (sky 0.998, ground 0.999, disc 0.951, triangle 0.986) with 0.397 on `box`; the CPU smoke run also reached 0.866 with `box` 0.509 and `disc` 0.841. Why `box` is weakest is not established here (a hypothesis: its straight edges resemble the ground band). Six held-out scenes give no dispersion estimate.\n\n</details>"
            ),
            "code": (
                "t0 = time.time()\n"
                "adapted = adapter.evaluate(held_out)\n"
                "print({{'adapted_pq': round(adapted['pq'], 4), 'sq': round(adapted['sq'], 4), 'rq': round(adapted['rq'], 4), 'things_pq': round(adapted['things']['pq'], 4), 'stuff_pq': round(adapted['stuff']['pq'], 4), 'seconds': round(time.time() - t0, 1)}})\n"
                "print('per class:', {{k: {{m: round(v[m], 3) for m in ('pq', 'sq', 'rq')}} | {{m: v[m] for m in ('tp', 'fp', 'fn')}} for k, v in adapted['per_class'].items()}})\n"
                "print()\n"
                "print(f\"{{'metric':<10s}} {{'all-sky':>10s}} {{'horizon':>10s}} {{'baseline':>10s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in ('pq', 'sq', 'rq'):\n"
                "    print(f'{{key:<10s}} {{trivial[key]:>10.4f}} {{horizon[key]:>10.4f}} {{baseline[key]:>10.4f}} {{adapted[key]:>10.4f}} {{adapted[key] - baseline[key]:>+10.4f}}')\n"
                "print({{'things_pq': {{'all_sky': round(trivial['things']['pq'], 4), 'horizon': round(horizon['things']['pq'], 4), 'adapted': round(adapted['things']['pq'], 4)}}, 'stuff_pq': {{'all_sky': round(trivial['stuff']['pq'], 4), 'horizon': round(horizon['stuff']['pq'], 4), 'adapted': round(adapted['stuff']['pq'], 4)}}}})\n"
                "evaluation = {{'task': 'panoptic segmentation over the five-class drawn vocabulary', 'metric': 'panoptic quality (PQ = SQ x RQ), IoU > 0.5 matching, class-aware, mean over classes',\n"
                "              'estimation': adapted['estimation'], 'split': {{'train': len(train_records), 'held_out': len(held_out), 'holdout': HOLDOUT, 'seed': SEED}},\n"
                "              'baselines': [{{'id': 'trivial-all-sky', 'pq': trivial['pq'], 'things_pq': trivial['things']['pq']}}, {{'id': 'trivial-horizon-split', 'pq': horizon['pq'], 'things_pq': horizon['things']['pq'], 'horizon_row': HORIZON_ROW}}, {{'id': 're-headed-before-adaptation', 'pq': baseline['pq'], 'things_pq': baseline['things']['pq']}}],\n"
                "              'adapted': adapted, 'baseline': baseline, 'trivial': trivial, 'horizon': horizon, 'verdict': 'sample-sanity',\n"
                "              'reason': 'six held-out drawn scenes from the same generator; adaptation evidence, not a segmentation benchmark',\n"
                "              'needs': 'panoptic annotations from the deployment domain (see the BYOD dataset branch) for any claim beyond the drawn vocabulary',\n"
                "              'model_id': MODEL_ID, 'model_revision': MODEL_REVISION}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(evaluation, handle, indent=2, ensure_ascii=False, default=str)"
            ),
        },
        {
            "md": (
                "## 9. Inference on new data\n\n"
                "`shape_dataset` with a **different seed** produces scenes the model has never seen, in training or in the "
                "held-out split. Each is validated through `validate_inputs` (the inference contract: image type, sides, "
                "thresholds) and segmented with the adapted pipeline — the same `segment` call, the same post-processing, now "
                "over the five-class vocabulary. Their instance maps are known too, so the cell reports PQ on them as a second, "
                "independent sample (not used for any decision). The smoke run's first new scene came back as `sky`, `ground` and a "
                "`box`, with no void.\n\n"
                + "<details><summary>Check your reasoning</summary>\n\nIs the new-data PQ an independent test? Yes for this drawn task — different seed, never seen — and the recorded run scored 0.954 on it; but it comes from the same generator, so it says nothing about photographs.\n\n</details>"
            ),
            "code": (
                'NEW_DATA_SEED = 123  # @param {{type:"integer"}}\n\n'
                "new_records = shape_dataset(4, seed=NEW_DATA_SEED)\n"
                "new_rows = []\n"
                "for record in new_records:\n"
                "    manifest = validate_inputs(record['image'], names=[record['id']])\n"
                "    prediction = adapter.segment(record['image'])\n"
                "    new_rows.append({{'id': record['id'], 'input': manifest['inputs'][0], 'segments': [(s['label'], round(s['score'], 3), round(s['area_fraction'], 3)) for s in prediction['segments']], 'void_fraction': round(prediction['void_fraction'], 3), 'reference_instances': record['instances']}})\n"
                "    print(new_rows[-1])\n"
                "new_quality = adapter.evaluate(new_records)\n"
                "print({{'new_data_pq': round(new_quality['pq'], 4), 'n_images': new_quality['n_images']}})\n"
                "preview = Image.new('RGB', (1280, 240))\n"
                "for index, record in enumerate(new_records):\n"
                "    preview.paste(record['image'], (320 * index, 0))\n"
                "preview"
            ),
        },
        {
            "md": (
                "## 10. Export the artifact, then reload it as if from a cold start\n\n"
                "`save_artifact` writes a self-describing directory: the Transformers `config.json` (which now carries the "
                "five-class vocabulary), `model.safetensors` (the full adapted weights, about 190 MB — the artifact type DIMER "
                "serving consumes, not a notebook-only surrogate; ART1), the image processor configuration, and "
                "`dimer-adapted-manifest.json` naming the format, the base model identity and revision, the vocabulary, the "
                "training record and the SHA-256 of every file (ART5). No training data, cache or credential is included (ART6).\n\n"
                "`load_artifact` is the fresh-reload check: it rebuilds the pipeline from the directory alone, without reference "
                "to the `adapter` object still in memory, and refuses an artifact whose format, base identity or file digests do "
                "not match — before importing any model library. The cell then re-scores the held-out split with the reloaded "
                "model and asserts the PQ is identical and the maps agree pixel for pixel (VER1–VER5); a reload that quietly lost "
                "the adaptation would stop the notebook here. Finally it tampers with a copy of the manifest and confirms the "
                'copy is refused, because "it loads" is only evidence if something that should not load is also tried.\n\n'
                + "<details><summary>Check your reasoning</summary>\n\nWhat would a reload that lost the adaptation look like? A PQ back near 0.0. The recorded run reproduced PQ 0.866 exactly with pixel agreement 1.0, and the tampered manifest was refused. These asserts are contract checks, not quality checks.\n\n</details>"
            ),
            "code": (
                "import shutil\n"
                "from pathlib import Path\n\n"
                "ARTIFACT_DIR = Path('outputs') / 'mask2former-panoptic-adapted'\n"
                "shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)\n"
                "descriptor = adapter.save_artifact(ARTIFACT_DIR, notes='standalone tutorial run')\n"
                "print(json.dumps(descriptor, indent=2))\n\n"
                "reloaded = Mask2FormerPanopticPipeline.load_artifact(ARTIFACT_DIR)\n"
                "print({{'source': reloaded.source, 'class_names': list(reloaded.class_names), 'stuff_names': list(reloaded.stuff_names), 'adapted': reloaded.adapted, 'device': reloaded.device}})\n"
                "reloaded_quality = reloaded.evaluate(held_out)\n"
                "print({{'reloaded_pq': round(reloaded_quality['pq'], 6), 'adapted_pq': round(adapted['pq'], 6)}})\n"
                "assert abs(reloaded_quality['pq'] - adapted['pq']) < 1e-9, 'the reloaded artifact does not reproduce the adapted score'\n"
                "agreement = float((reloaded.segment(held_out[0]['image'])['segmentation'] == adapter.segment(held_out[0]['image'])['segmentation']).mean())\n"
                "print({{'pixel_agreement_first_held_out_scene': agreement}})\n"
                "assert agreement == 1.0, 'the reloaded artifact produces a different map'\n"
                "print('fresh reload reproduces the adapted score and map exactly')\n\n"
                "tampered = Path('outputs') / 'tampered-artifact'\n"
                "shutil.rmtree(tampered, ignore_errors=True)\n"
                "shutil.copytree(ARTIFACT_DIR, tampered)\n"
                "with open(tampered / ARTIFACT_MANIFEST_NAME, encoding='utf-8') as handle:\n"
                "    tampered_manifest = json.load(handle)\n"
                "tampered_manifest['base']['revision'] = '0' * 40\n"
                "with open(tampered / ARTIFACT_MANIFEST_NAME, 'w', encoding='utf-8') as handle:\n"
                "    json.dump(tampered_manifest, handle)\n"
                "try:\n"
                "    Mask2FormerPanopticPipeline.load_artifact(tampered)\n"
                "except ValueError as exc:\n"
                "    print('tampered artifact refused ->', exc)\n"
                "else:\n"
                "    raise AssertionError('a tampered artifact was accepted')\n"
                "finally:\n"
                "    shutil.rmtree(tampered, ignore_errors=True)"
            ),
        },
        {
            "md": (
                "## 11. Machine-readable outputs and provenance\n\n"
                "Everything the notebook established, written to `outputs/{stem}_result.json` beside the artifact: the "
                "pinned identity and manifest digests, the runtime, the COCO observation on the drawn scene, the dataset manifest "
                "and split, the training record with its loss history, the two trivial / baseline / adapted / reloaded scores, the "
                "new-data rows, and the artifact descriptor. A later reader can tell what was measured, on what, with which "
                "weights, without rerunning anything. No credentials are recorded."
            ),
            "code": (
                "payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'license': MODEL_LICENSE, 'manifest_sha256': [entry['sha256'] for entry in MANIFEST['files']]}},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': adapter.device, 'cuda': torch.cuda.is_available(), 'precision': 'float32'}},\n"
                "    'coco_observation': coco_observation,\n"
                "    'adaptation': {{'dataset': dataset_manifest, 'split': evaluation['split'], 'run': run, 'trivial': trivial, 'horizon': horizon, 'baseline': baseline, 'adapted': adapted, 'reloaded': reloaded_quality, 'new_data': new_rows, 'new_data_quality': new_quality, 'artifact': descriptor}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "## 12. Optional: your own image, or your own labelled dataset\n\n"
                "Both branches are off by default so the sample path never opens an upload dialog. `USE_BYOD_IMAGE` uploads one "
                "image and segments it with the **adapted** model — remember it knows only the five drawn classes now. "
                "`USE_BYOD_DATASET` uploads a zip of a labelled directory (`dataset.json` + images + instance-id PNGs; the schema "
                "is printed, with an example of the on-disk `dataset.json`) and runs it through the same stages as the sample: "
                "`validate_dataset`, `split_records`, a fresh re-headed pipeline for *your* vocabulary, `finetune`, `evaluate` "
                "against the re-headed baseline and a trivial majority-class predictor, `save_artifact` and `load_artifact`. It "
                "writes `outputs/{stem}_byod_dataset_manifest.json` and `outputs/{stem}_byod_evaluation_report.json` (naming "
                "your dataset's digest) with the Section 8 verdict vocabulary, and prints `no-improvement` when the adapted PQ "
                "does not beat both baselines. That is the adaptation contract (§25.10): a BYOD path that only ran inference "
                "would not be the workflow this notebook demonstrates. Ceilings are the ones `validate_dataset` enforces; a "
                "rejected dataset stops with the pipeline's own message, which names the record and the rule. On Kaggle or "
                "Jupyter set `BYOD_IMAGE_PATH` / `BYOD_DATASET_PATH` (a zip or a directory); leave them empty on Colab for the "
                "upload dialog. A cancelled upload, a missing path, a file Pillow cannot read, an archive without `dataset.json` "
                "or a `dataset.json` whose entries do not match the layout each stop with one line naming the failed condition "
                "and the fix; re-run this cell after fixing it."
            ),
            "code": (
                "import io\n\n"
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_EPOCHS = 6  # @param {{type:"integer"}}\n'
                "# Kaggle / Jupyter: paths in the runtime (the dataset may be a zip or a directory). Empty: the Colab upload dialog.\n"
                "BYOD_IMAGE_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_DATASET_PATH = ''  # @param {{type:\"string\"}}\n\n\n"
                "def byod_input(path_text, what, field):\n"
                "    \"\"\"(name, path or bytes) from a path field, or from exactly one Colab upload; each refusal says what to set.\"\"\"\n"
                "    if path_text.strip():\n"
                "        path = Path(path_text.strip()).expanduser()\n"
                "        if not path.exists():\n"
                "            raise FileNotFoundError(f'{{field}} {{str(path)!r}} does not exist: give the path of {{what}}')\n"
                "        return path.name, path\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'{{field}} is empty and this runtime has no Colab upload dialog: set {{field}} to {{what}}') from None\n"
                "    uploaded = files.upload() or {{}}\n"
                "    if len(uploaded) != 1:\n"
                "        raise RuntimeError(f'expected exactly one uploaded file ({{what}}), got {{len(uploaded)}} ({{sorted(uploaded) or \"upload cancelled or empty\"}}): run this cell again, or set {{field}}')\n"
                "    return next(iter(uploaded.items()))\n\n\n"
                "def byod_open_image(name, source):\n"
                "    \"\"\"Decode one BYOD image, or refuse it with the file name and the rule.\"\"\"\n"
                "    try:\n"
                "        image = Image.open(source if isinstance(source, Path) else io.BytesIO(source))\n"
                "        image.load()\n"
                "    except Exception as exc:\n"
                "        raise ValueError(f'{{name}}: not an image Pillow can read ({{type(exc).__name__}}); give a JPEG, PNG, TIFF or WebP file and re-run this cell') from None\n"
                "    return image\n\n\n"
                "def byod_load_dir(archive, root):\n"
                "    \"\"\"load_labelled_dir with each layout failure turned into one line naming the condition and the fix.\"\"\"\n"
                "    try:\n"
                "        return load_labelled_dir(root)\n"
                "    except KeyError as exc:\n"
                "        raise ValueError(f'{{archive}}: dataset.json is missing the key {{exc}}; each record needs id, image, mask and instances (see the layout printed above), fix it and re-run this cell') from None\n"
                "    except FileNotFoundError as exc:\n"
                "        raise ValueError(f'{{archive}}: dataset.json names a file that is not in the archive ({{exc.filename}}); image and mask paths are relative to the folder holding dataset.json, fix it and re-run this cell') from None\n"
                "    except (OSError, ValueError) as exc:\n"
                "        raise ValueError(f'{{archive}}: a record could not be read ({{type(exc).__name__}}: {{exc}}); every image must be decodable by Pillow and every mask a PNG of instance ids, fix it and re-run this cell') from None\n\n\n"
                "print('BYOD dataset schema:', json.dumps(DATASET_SCHEMA, indent=2))\n"
                "print('on-disk layout for BYOD_DATASET_PATH: <root>/dataset.json =', json.dumps({{'class_names': ['sky', 'ground', 'disc'], 'stuff_names': ['sky', 'ground'], 'records': [{{'id': 'scene-000', 'image': 'images/scene-000.png', 'mask': 'masks/scene-000.png', 'instances': {{'1': 'sky', '2': 'ground', '3': 'disc'}}}}]}}), '; each mask is a PNG whose pixel values are the instance ids (1..254, every pixel assigned)')\n"
                "if USE_BYOD_IMAGE:\n"
                "    byod_name, byod_source = byod_input(BYOD_IMAGE_PATH, 'one image file', 'BYOD_IMAGE_PATH')\n"
                "    byod_image = byod_open_image(byod_name, byod_source)\n"
                "    print(validate_inputs(byod_image, names=[byod_name])['inputs'])\n"
                "    byod_result = reloaded.segment(byod_image)\n"
                "    print({{'segments': [(s['label'], round(s['score'], 3), round(s['area_fraction'], 3)) for s in byod_result['segments']], 'void_fraction': round(byod_result['void_fraction'], 3)}})\n"
                "if USE_BYOD_DATASET:\n"
                "    import zipfile\n\n"
                "    archive, archive_source = byod_input(BYOD_DATASET_PATH, 'a zip or a directory holding dataset.json, the images and the instance-id PNGs', 'BYOD_DATASET_PATH')\n"
                "    target = Path('byod_dataset')\n"
                "    if isinstance(archive_source, Path) and archive_source.is_dir():\n"
                "        target = archive_source\n"
                "    else:\n"
                "        shutil.rmtree(target, ignore_errors=True)\n"
                "        data = archive_source.read_bytes() if isinstance(archive_source, Path) else archive_source\n"
                "        try:\n"
                "            zf = zipfile.ZipFile(io.BytesIO(data))\n"
                "        except zipfile.BadZipFile:\n"
                "            raise ValueError(f'{{archive}}: not a zip file; give a zip or a directory') from None\n"
                "        with zf:\n"
                "            for member in zf.infolist():\n"
                "                destination = (target / member.filename).resolve()\n"
                "                if not str(destination).startswith(str(target.resolve())):\n"
                "                    raise ValueError(f'{{archive}}: refusing archive member outside the target directory: {{member.filename}}')\n"
                "                zf.extract(member, target)\n"
                "    found = [target] if (target / 'dataset.json').is_file() else [p.parent for p in target.rglob('dataset.json')]\n"
                "    if not found:\n"
                "        raise ValueError(f'{{archive}}: no dataset.json found; the schema printed above names the expected layout')\n"
                "    root = found[0]\n"
                "    byod_records, byod_classes, byod_stuff = byod_load_dir(archive, root)\n"
                "    byod_manifest = validate_dataset(byod_records, byod_classes, byod_stuff, epochs=BYOD_EPOCHS)\n"
                "    byod_manifest['source'] = {{'archive': archive, 'root': str(root)}}\n"
                "    with open('outputs/{stem}_byod_dataset_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump(byod_manifest, handle, indent=2, ensure_ascii=False)\n"
                "    print(json.dumps({{k: v for k, v in byod_manifest.items() if k != 'schema'}}, indent=2))\n"
                "    byod_train, byod_held = split_records(byod_records, holdout=HOLDOUT, seed=SEED)\n"
                "    # Trivial baseline for your data: every pixel is the class with the most pixels in your training split.\n"
                "    byod_pixels = {{name: 0 for name in byod_classes}}\n"
                "    for record in byod_train:\n"
                "        for instance_id, name in record['instances'].items():\n"
                "            byod_pixels[name] += int((record['instance_map'] == int(instance_id)).sum())\n"
                "    byod_majority = max(byod_pixels, key=byod_pixels.get)\n"
                "    byod_trivial_pairs = []\n"
                "    for record in byod_held:\n"
                "        reference_map, reference_segments = reference_from_record(record, byod_classes, byod_stuff)\n"
                "        byod_trivial_pairs.append((np.ones_like(reference_map), [{{'id': 1, 'label': byod_majority, 'is_thing': byod_majority not in byod_stuff}}], reference_map, reference_segments))\n"
                "    byod_trivial = panoptic_quality(byod_trivial_pairs)\n"
                "    byod_adapter = Mask2FormerPanopticPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_classes, stuff_names=byod_stuff, seed=SEED)\n"
                "    byod_baseline = byod_adapter.evaluate(byod_held)\n"
                "    byod_run = byod_adapter.finetune(byod_train, epochs=BYOD_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_backbone=FREEZE_BACKBONE, progress=lambda row: print(row))\n"
                "    byod_adapted = byod_adapter.evaluate(byod_held)\n"
                "    byod_verdict = 'sample-sanity' if byod_adapted['pq'] > max(byod_trivial['pq'], byod_baseline['pq']) else 'no-improvement'\n"
                "    print({{'trivial_pq': round(byod_trivial['pq'], 4), 'trivial_predictor': 'all-' + byod_majority, 'baseline_pq': round(byod_baseline['pq'], 4), 'adapted_pq': round(byod_adapted['pq'], 4), 'things_pq': round(byod_adapted['things']['pq'], 4), 'per_class': {{k: round(v['pq'], 3) for k, v in byod_adapted['per_class'].items()}}, 'verdict': byod_verdict}})\n"
                "    if byod_verdict == 'no-improvement':\n"
                "        print('no-improvement: the adapted model did not beat the trivial predictor or the re-headed baseline on your held-out split; more records, more epochs (BYOD_EPOCHS) or a larger split may be needed')\n"
                "    byod_evaluation = {{'task': 'panoptic segmentation over the BYOD vocabulary ' + str(list(byod_classes)), 'metric': 'panoptic quality (PQ = SQ x RQ), IoU > 0.5 matching, class-aware, mean over classes',\n"
                "                       'estimation': byod_adapted['estimation'], 'split': {{'train': len(byod_train), 'held_out': len(byod_held), 'holdout': HOLDOUT, 'seed': SEED}}, 'dataset_sha256': byod_manifest['dataset_sha256'],\n"
                "                       'baselines': [{{'id': 'trivial-all-' + byod_majority, 'pq': byod_trivial['pq'], 'things_pq': byod_trivial['things']['pq']}}, {{'id': 're-headed-before-adaptation', 'pq': byod_baseline['pq'], 'things_pq': byod_baseline['things']['pq']}}],\n"
                "                       'adapted': byod_adapted, 'baseline': byod_baseline, 'trivial': byod_trivial, 'run': byod_run, 'verdict': byod_verdict,\n"
                "                       'reason': 'held-out records of your own dataset; adaptation evidence on one seeded split, not a benchmark', 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION}}\n"
                "    with open('outputs/{stem}_byod_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump(byod_evaluation, handle, indent=2, ensure_ascii=False, default=str)\n"
                "    byod_dir = Path('outputs') / 'mask2former-panoptic-byod-adapted'\n"
                "    shutil.rmtree(byod_dir, ignore_errors=True)\n"
                "    print(byod_adapter.save_artifact(byod_dir, notes='BYOD dataset run'))\n"
                "    byod_reloaded = Mask2FormerPanopticPipeline.load_artifact(byod_dir)\n"
                "    assert abs(byod_reloaded.evaluate(byod_held)['pq'] - byod_adapted['pq']) < 1e-9\n"
                "    print('BYOD artifact reloads and reproduces its held-out score')\n"
                "    print(sorted(p for p in os.listdir('outputs') if 'byod' in p))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The held-out panoptic quality is a **sample metric on drawn shapes**: six scenes from the same generator as the "
        "training scenes, three thing classes with distinct silhouettes, two stuff classes that always occupy the same halves of "
        "the image. Reaching PQ near 0.9 from 0.0 (0.866 on both the CPU smoke run and the Kaggle T4 run) shows that the re-headed "
        "model, the upstream set loss, the bounded loop, the evaluation and the artifact boundary all work together; it does not "
        "show that 18 photographs of your own would train as easily, that the frozen backbone suits your domain, or that `box` "
        "(the weakest thing class on both devices: 0.509 on CPU, 0.397 on the T4) would not need more data. The horizon-split "
        "predictor already scores 0.318 without learning, so the stuff PQ says little; the things PQ (0 for every trivial "
        "predictor) is where the adaptation shows. **Training loss is not task evidence**, the "
        "class scores remain uncalibrated softmaxes, and the adapted model has forgotten COCO's vocabulary entirely — it knows five "
        "classes. There is no dispersion estimate: one seed, one split.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire and "
        "digest-verify the pinned base model, validate a labelled dataset, adapt the model with a bounded gradient fine-tune, "
        "evaluate it on held-out data, export a self-describing artifact and reload it from disk with identical outputs, in the "
        "tested runtime — without the repository being reachable. It does **not** establish benchmark superiority, calibration, "
        "safety for high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "## 13. Your turn — change one thing: the number of epochs\n\n"
        "A structured activity: **predict → change → run → observe → explain**. Section 7 rebuilds the re-headed model whenever "
        "`adapter` has already been trained, so the cells to re-run are exactly these: edit `EPOCHS` in **Section 5**, re-run "
        "Section 5, then **Section 7** (the fine-tune) and **Section 8** (the evaluation); or select Section 6 and choose "
        "**Runtime → Run after**. Re-running Section 7 alone is enough to retrain; re-running only Section 8 re-scores the same "
        "weights.\n\n"
        "**Predict:** with `EPOCHS = 2` (18 steps instead of 54), will the held-out PQ be higher, lower or about the same as the "
        "6-epoch value, and which classes will move most?\n\n"
        "Then set `EPOCHS = 2`, re-run Sections 5, 7 and 8, and note the run record's `epochs` and `steps` (they must read 2 and "
        "18), the mean epoch losses and the per-class PQ.\n\n"
        "<details><summary>Check your reasoning</summary>\n\nLower. In the review's CPU probe of this notebook (3 October 2026, torch 2.14.0+cpu; same seeds and split as the default path) two epochs from the re-headed state gave mean epoch losses 33.7 → 17.3 and held-out PQ 0.59, with `box` and `disc` at 0 while the stuff classes were already near 1 — the things are what the extra epochs buy. Before this fix, re-running the fine-tune cell continued training the adapted weights instead, and PQ *rose* to 0.96 with a run record claiming 2 epochs; if you see a rising PQ and `rebuilt_from_snapshot` was not printed, you are running an older copy of this notebook. A GPU run moves the per-class values, not the direction.\n\n</details>\n\n"
        "**Next experiments** (same re-run rule): raise `N_IMAGES` to 60 and see whether `box` catches up; set "
        "`FREEZE_BACKBONE = False` (Section 7) and compare (the model card records why the default is frozen); change "
        "`DATASET_SEED` and see how much the held-out PQ moves between seeds — that spread is the dispersion this notebook does "
        "not otherwise report; then bring your own labelled directory through `USE_BYOD_DATASET`.\n\n"
        "## Troubleshooting\n\n"
        "Section 1 stops with `This notebook needs a Linux x86_64 runtime`: use Google Colab, Kaggle or a Linux Jupyter host. "
        "`The pinned uv wheel failed its size/SHA-256 check`: run Section 1 again; if it repeats, the download is being altered. "
        "`The isolated environment's Python process exited`: the worker crashed, usually out of memory — lower `BATCH_SIZE` to 1, "
        "restart the session and choose **Run all**. A size or SHA-256 error in Section 3: delete the file under `weights/` and "
        "re-run Section 3. A `ValueError` from `validate_dataset`: the record and rule are named — fix the data (the printed "
        "schema is the contract). With BYOD: `… does not exist`, `expected exactly one uploaded file` or `… no Colab upload "
        "dialog` — fix the path field or the upload; `not an image Pillow can read` — give an image file; `not a zip file`, `no "
        "dataset.json found`, `dataset.json is missing the key` or `names a file that is not in the archive` — check the archive "
        "layout against the printed example. A reload mismatch in Section 10 means the artifact on disk is incomplete — delete it "
        "and re-run Section 10. The warnings named in Section 8 (`overflow encountered in exp`, `_max_size`, the slow-processor "
        "notice, `You should probably TRAIN this model`) are expected and not errors. CPU training that takes much longer than "
        "the recorded 185 s: lower `N_IMAGES` or `EPOCHS`, or choose a T4 runtime.\n\n"
        "## Glossary\n\n"
        "- **Panoptic annotation:** an instance map (an id per pixel) plus a class per instance; stuff classes have one instance per image.\n"
        "- **Re-heading:** replacing the class predictor for a new vocabulary; everything else transfers from the checkpoint.\n"
        "- **Set loss / Hungarian matching:** each query is matched to at most one annotated instance before the class, mask and dice losses are computed.\n"
        "- **Frozen backbone:** the Swin-T feature extractor does not train; the pixel decoder, transformer decoder and head do.\n"
        "- **PQ = SQ × RQ:** panoptic quality — mean IoU of matched segments times a recognition score that penalises misses and false segments; class-aware here.\n"
        "- **Trivial baselines:** predictors that learn nothing — all-`sky`, and the horizon split (sky above the training split's mean horizon row, ground below). They bound what the stuff PQ can mean; the things PQ, 0 for both, is where learning shows.\n"
        "- **Held-out split:** scenes never shown to `finetune` and used once, to report.\n"
        "- **Artifact / cold reload:** the self-describing directory with digests, rebuilt from disk alone to prove it is complete.\n"
        "- **Isolated environment:** the separate hash-locked Python environment built in Section 1; every later cell runs there.\n\n"
        "## Conclusion (your notes)\n\n"
        "1. In two sentences: what moved PQ from 0.0 to its final value, and which of the three baselines does the things PQ beat?\n"
        "2. Why is the training loss not evidence that the model segments well?\n"
        "3. Which of your predictions were wrong, and what did the output show instead?\n"
        "4. What would you need before adapting this model to photographs of your own?\n\n"
        "**Your notes:**\n\n"
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
