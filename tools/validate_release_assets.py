"""Static release-asset validation for the Mask2Former Swin-T COCO panoptic DIMER pipeline.

Checks the two STANDALONE tutorial notebooks (DIMER Notebook Specification 2.0 §4), the tutorial
registry, model card, README, STATUS.md and weight documentation for source conformance and
cross-document identity consistency, and runs the generator parity checks (PAR1–PAR3).

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.
"""

# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "mask2former_panoptic_pipeline"
REPO_NAME = "mask2former-panoptic-pipeline"
NOTEBOOK_NAME = "mask2former_panoptic_colab.ipynb"
EXPECTED_PROFILE = "TASK-INFERENCE"
FINETUNE_NOTEBOOK_NAME = "mask2former_panoptic_finetune_colab.ipynb"
FINETUNE_PROFILE = "E2E"
EXPECTED_MODEL_ID = "facebook/mask2former-swin-tiny-coco-panoptic"
PIPELINE_CLASS = "Mask2FormerPanopticPipeline"
# INF1: the notebook loads through the package's public constructor from the verified directory.
MODEL_LOAD_EXPR = f"{PIPELINE_CLASS}.from_pretrained(weights_dir=WEIGHTS_DIR)"
# MODEL_CARD_SPEC version the card must declare.
CARD_SPEC = "1.1"
# Additional 40-hex revisions a document may legitimately cite (none: the public photograph is pinned by SHA-256, not a git revision).
KNOWN_SHAS: frozenset[str] = frozenset()
# Colab form gates that must default to the non-interactive sample path.
BYOD_GATES = ("USE_BYOD",)
FINETUNE_BYOD_GATES = ("USE_BYOD_IMAGE", "USE_BYOD_DATASET")
# Machine-readable artifacts the notebook must write (OUT1-OUT3).
EXPECTED_OUTPUTS = (
    "outputs/mask2former_panoptic_input_manifest.json",
    "outputs/mask2former_panoptic_evaluation_report.json",
    "outputs/mask2former_panoptic_photo_evaluation_report.json",
    "outputs/mask2former_panoptic_result.json",
)
# Profile-specific code the notebook must exercise through the carried module's public API.
CODE_MARKERS = (
    "image, reference_segmentation, reference_segments = synthetic_scene()",
    "input_manifest = validate_inputs(image, score_threshold=score_threshold, overlap_threshold=overlap_threshold, mask_threshold=mask_threshold, names=[image_name])",
    "validate_inputs(image, score_threshold=1.5)",
    "result = pipe.segment(image, score_threshold=score_threshold, overlap_threshold=overlap_threshold, mask_threshold=mask_threshold)",
    "report = evaluation_report(result, reference, sample_kind=sample_kind)",
    "photo_report = evaluation_report(photo_result, None, sample_kind='public-photo')",
    "print({'ceilings': {'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'INPUT_SIZE': INPUT_SIZE, 'MASK_LOGIT_SIZE': MASK_LOGIT_SIZE, 'NUM_QUERIES': NUM_QUERIES, 'SCORE_THRESHOLD': SCORE_THRESHOLD, 'OVERLAP_THRESHOLD': OVERLAP_THRESHOLD, 'MASK_THRESHOLD': MASK_THRESHOLD}})",
    "PHOTO_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'",
    "PHOTO_SHA256 = 'dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e'",
    "if photo_sha256 != PHOTO_SHA256:",
    "raise RuntimeError(f'public photograph digest {photo_sha256} != pinned {PHOTO_SHA256}; refusing to use it')",
    "hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()",
    "Image.fromarray((seg + 1).astype(np.uint16)).save(",
    "'model_revision': MODEL_REVISION",
    "'model_license': MODEL_LICENSE",
    "transformers.__version__",
    "'device': pipe.device",
)
# Profile-specific learner-facing statements.
MARKDOWN_MARKERS = (
    "**Capability:** panoptic segmentation",
    "**No adaptation occurs:**",
    "**Scores are uncalibrated softmaxes**",
    "otherwise it is **void**",
    "`panoptic_quality` — Kirillov et al.'s PQ",
    "**class-agnostic**",
    "their verdict is `not-measurable`",
    "`sample-sanity`",
    "refuses it unless its SHA-256 equals the pinned digest",
    "individual Flickr licence is not",
    "not a COCO benchmark",
)
# --- the standalone E2E adaptation notebook (tools/notebook_template_finetune.py) ---
FINETUNE_EXPECTED_OUTPUTS = (
    "outputs/mask2former_panoptic_finetune_dataset_manifest.json",
    "outputs/mask2former_panoptic_finetune_evaluation_report.json",
    "outputs/mask2former_panoptic_finetune_result.json",
    "outputs') / 'mask2former-panoptic-adapted'",
)
FINETUNE_CODE_MARKERS = (
    "coco_result = pipe.segment(scene_image)",
    "records = shape_dataset(N_IMAGES, seed=DATASET_SEED)",
    "dataset_manifest = validate_dataset(records, SHAPE_CLASSES, SHAPE_STUFF, epochs=EPOCHS)",
    "validate_dataset([broken], SHAPE_CLASSES, SHAPE_STUFF)",
    "train_records, held_out = split_records(records, holdout=HOLDOUT, seed=SEED)",
    "adapter = Mask2FormerPanopticPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SHAPE_CLASSES, stuff_names=SHAPE_STUFF, seed=SEED)",
    "baseline = adapter.evaluate(held_out)",
    "trivial = panoptic_quality(trivial_pairs)",
    "run = adapter.finetune(",
    "freeze_backbone=FREEZE_BACKBONE,",
    "adapted = adapter.evaluate(held_out)",
    "new_records = shape_dataset(4, seed=NEW_DATA_SEED)",
    "prediction = adapter.segment(record['image'])",
    "descriptor = adapter.save_artifact(ARTIFACT_DIR, notes='standalone tutorial run')",
    "reloaded = Mask2FormerPanopticPipeline.load_artifact(ARTIFACT_DIR)",
    "reloaded_quality = reloaded.evaluate(held_out)",
    "assert abs(reloaded_quality['pq'] - adapted['pq']) < 1e-9",
    "assert agreement == 1.0",
    "tampered_manifest['base']['revision'] = '0' * 40",
    "raise AssertionError('a tampered artifact was accepted')",
    "'license': MODEL_LICENSE",
    "transformers.__version__",
    "'device': adapter.device",
    "zf.extract(member, target)",
)
FINETUNE_MARKDOWN_MARKERS = (
    "**Capability:** bounded gradient adaptation",
    "**What is trained and what is not:**",
    "**Training loss is optimisation evidence only**",
    "**runs the bounded fine-tune**",
    "**the Swin-T backbone is frozen**",
    "Hungarian matching",
    "`TRAIN_NUM_POINTS = 4096`",
    "a single seeded holdout",
    "(EVAL10)",
    "(VER1–VER5)",
    "§25.10",
    "**sample metric on drawn shapes**",
    "not a benchmark",
)
# Every standalone notebook this repository ships: name -> (template module, profile, gates, outputs, markers).
NOTEBOOKS = {
    NOTEBOOK_NAME: {
        "template": "notebook_template",
        "profile": EXPECTED_PROFILE,
        "byod_gates": BYOD_GATES,
        "expected_outputs": EXPECTED_OUTPUTS,
        "code_markers": CODE_MARKERS,
        "markdown_markers": MARKDOWN_MARKERS,
    },
    FINETUNE_NOTEBOOK_NAME: {
        "template": "notebook_template_finetune",
        "profile": FINETUNE_PROFILE,
        "byod_gates": FINETUNE_BYOD_GATES,
        "expected_outputs": FINETUNE_EXPECTED_OUTPUTS,
        "code_markers": FINETUNE_CODE_MARKERS,
        "markdown_markers": FINETUNE_MARKDOWN_MARKERS,
    },
}
# FORBIDDEN_PATTERNS labels that are checked outside the carried module cells only (a loader that must
# deserialise a PyTorch checkpoint or enable remote code does so inside the package, with the trust
# boundary stated in the notebook; none by default).
FORBIDDEN_PATTERNS_MODULE_EXEMPT: tuple[str, ...] = ()
# Inference must happen in this kernel: no worker process, no worker CLI, no subprocess outside the
# generator-owned install cell (Kurt 2026-09-13). Checked on every code cell except the embedded ones and cell 1.
FORBIDDEN_WORKER_CALLS = ("worker.run(", "worker_cli(", "subprocess.run([")
# Direct-library use that must stay inside the carried module cells (G2: the notebook calls the
# pipeline API, it does not reimplement it). Checked on every code cell except the embedded ones.
FORBIDDEN_OUTSIDE_MODULE = (
    "from huggingface_hub import",
    "import huggingface_hub",
    "hf_hub_download(",
    "snapshot_download(",
    "from transformers import",
    "transformers.pipeline(",
    "Mask2FormerForUniversalSegmentation",
    "AutoImageProcessor",
    "post_process_panoptic_segmentation",
    "from safetensors",
    "torch.optim.",
    ".backward()",
    "save_pretrained(",
)

# ---------------------------------------------------------------------------
# Shared checks. Everything below is source/structure validation only. Passing
# these checks is NOT clean-runtime execution evidence under DIMER Notebook
# Specification 2.0; see docs/release-verification.md for the release gate.
# ---------------------------------------------------------------------------

NOTEBOOK_SPEC = "2.0"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
# Markers every standalone DIMER tutorial in this fleet must carry, independent of profile.
# Matched on comment-stripped code, so a commented-out call does not count.
COMMON_CODE_MARKERS = (
    "PINS = [",
    "NOTEBOOK_SOURCE = {",
    "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINS], check=True)",
    "importlib.metadata.packages_distributions()",
    "importlib.invalidate_caches()",
    "platform.python_version()",
    "torch.__version__",
    "MANIFEST = {",
    "if (MANIFEST['modelId'], MANIFEST['revision']) != (MODEL_ID, MODEL_REVISION):",
    "WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR",
    "json.dump(MANIFEST, handle, indent=2)",
    "fetched = stage_missing_files(WEIGHTS_DIR, allow_download=True)",
    "snapshot = verify_snapshot(WEIGHTS_DIR)",
    "'repository_revision': NOTEBOOK_SOURCE['repository_revision']",
    "'notebook_source': NOTEBOOK_SOURCE",
    "os.makedirs('outputs', exist_ok=True)",
    "from google.colab import files",
    "files.upload()",
)
COMMON_MARKDOWN_MARKERS = (
    f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)",
    "**Mode:** `",
    "**Run all:**",
    "**Bring Your Own Data:**",
    "**This notebook is standalone.**",
    "**Learning objectives:**",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    "- **External access:** the Hugging Face Hub",
    "## 1. Install the pinned runtime",
    "## 2. Pipeline code (carried verbatim from",
    "## 3. Pin, stage and verify the model",
    "## Interpretation and limits",
    "Successful execution proves that the recorded repository revision",
    "without the repository being",
    "It does **not** establish benchmark superiority",
    "## References",
    f"- Repository model card: https://github.com/kurtvalcorza/{REPO_NAME}/blob/main/MODEL_CARD.md",
)
# Patterns that must never appear in tutorial code (comment-stripped), in any cell.
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone (ST1)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com/kurtvalcorza")),
    ("mutable git dependency (MOD14)", re.compile(r"git\+https?://(?![^\n]*@[0-9a-f]{40}\b)")),
    ("editable self-install", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("repository package import (ST1)", re.compile(rf"^\s*(?:from|import)\s+{PACKAGE}\b", re.M)),
    ("mutable model reference (MOD14)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    (
        "unsafe deserialization",
        re.compile(r"\bpickle\.load|\btorch\.load\s*\(|getattr\(\s*torch\s*,\s*['\"]load['\"]"),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
)


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _strip_comments(source: str) -> str:
    """Return the source without comment tokens (string contents are preserved)."""
    out: list[str] = []
    last_row, last_col = 1, 0
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    for token in tokens:
        (srow, scol), (erow, ecol) = token.start, token.end
        if srow > last_row:
            out.append(lines[last_row - 1][last_col:] if last_row - 1 < len(lines) else "")
            for row in range(last_row, srow - 1):
                out.append(lines[row])
            last_row, last_col = srow, 0
        if srow - 1 < len(lines):
            out.append(lines[srow - 1][last_col:scol])
        if token.type != tokenize.COMMENT:
            out.append(token.string)
        last_row, last_col = erow, ecol
    return "".join(out)


def _assignment_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.comprehension):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    _check(spec is not None and spec.loader is not None, f"tools/{name}.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _package_identity(template: dict | None = None) -> tuple[str, str]:
    """Read MODEL_ID / MODEL_REVISION from the package source without importing torch."""
    template = template or _load_tool("notebook_template").TEMPLATE
    entry = ROOT / template.get("package_dir", f"src/{PACKAGE}") / template.get("entry_module", "pipeline.py")
    text = _read(entry)
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(
        model_id is not None and revision is not None,
        "pipeline.py must define MODEL_ID and MODEL_REVISION",
    )
    _check(SHA40.match(revision.group(1)) is not None, "MODEL_REVISION must be a 40-hex immutable commit")
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter")
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "base_model:"):
        _check(key in front, f"MODEL_CARD.md missing front-matter field: {key}")
    _check(f'model_card_spec: "{CARD_SPEC}"' in front, f"MODEL_CARD.md model_card_spec must be {CARD_SPEC}")
    _check(f"base_model: {EXPECTED_MODEL_ID}" in front, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/benchmark claim")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}")
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading}")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")


def validate_identity_consistency() -> None:
    """The immutable upstream identity must be the same string in every document."""
    model_id, revision = _package_identity()
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md"):
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the immutable revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha != revision and sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


def validate_release_status() -> None:
    """STATUS.md, README.md and tutorials/README.md must agree on one status token."""
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    registry = _read(ROOT / "tutorials" / "README.md").replace("**", "")
    _check(f"| {token}" in registry, f"tutorials/README.md must record the {token} status")
    other = [t for t in STATUS_TOKENS if t != token]
    for name, text in (("README.md", section.replace("**", "")), ("tutorials/README.md", registry)):
        for stale in other:
            _check(f"| {stale}" not in text, f"{name} carries a conflicting status token")
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )


def _validate_notebook_structure(
    path: Path, notebook: dict, spec: dict, _template: dict
) -> tuple[list[tuple[int, str, ast.Module]], str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == spec["profile"], f"{path.name}: profile {profile!r} != declared {spec['profile']!r}")
    spec = dimer.get("notebook_spec", dimer.get("notebook_spec_version"))
    _check(
        spec == NOTEBOOK_SPEC,
        f"{path.name}: metadata.dimer must declare notebook spec version '{NOTEBOOK_SPEC}'",
    )
    _check(
        dimer.get("notebook_mode") in ("REFERENCE", "GUIDED", "WORKSHOP"),
        f"{path.name}: metadata.dimer.notebook_mode must declare a §3.3 pedagogical mode",
    )
    _check(dimer.get("standalone") is True, f"{path.name}: metadata.dimer.standalone must be true (ST6)")
    generated = dimer.get("generated_from")
    _check(isinstance(generated, dict), f"{path.name}: metadata.dimer.generated_from is required (ST5)")
    _check(
        generated.get("repository") == REPO_NAME,
        f"{path.name}: generated_from.repository must be {REPO_NAME}",
    )
    _check(
        generated.get("module")
        == f"{_template.get('package_dir', f'src/{PACKAGE}')}/{_template.get('entry_module', 'pipeline.py')}",
        f"{path.name}: generated_from.module must name the template entry module",
    )
    _pkg_dir = ROOT / _template.get("package_dir", f"src/{PACKAGE}")
    _order = _load_tool("build_notebook")._module_order(
        _pkg_dir, list(_template.get("modules", ["pipeline.py"]))
    )
    module_sha = hashlib.sha256("".join(_read(_pkg_dir / m) for m in _order).encode("utf-8")).hexdigest()
    _check(
        generated.get("module_sha256") == module_sha,
        f"{path.name}: generated_from.module_sha256 does not match src/ (PAR4: regenerate the notebook)",
    )
    _check(bool(generated.get("generator")), f"{path.name}: generated_from.generator is required")
    cells = notebook.get("cells", [])
    _check(
        bool(cells) and cells[0].get("cell_type") == "markdown",
        f"{path.name}: first cell must be markdown",
    )
    code_cells: list[tuple[int, str, ast.Module]] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(
            index > 0 and cells[index - 1].get("cell_type") == "markdown",
            f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell",
        )
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_cells.append((index, source, tree))
    markdown = "\n".join(markdown_parts)
    raw_code = "\n".join(source for _, source, _ in code_cells)
    _check(not PLACEHOLDER.search(raw_code + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code_cells, markdown


def _validate_gates(
    path: Path, code_cells: list[tuple[int, str, ast.Module]], gates: tuple[str, ...]
) -> None:
    """Each BYOD gate is assigned exactly once, to the constant False, on a Colab form line."""
    for gate in gates:
        assignments = []
        for index, source, tree in code_cells:
            lines = source.splitlines()
            for node in ast.walk(tree):
                if gate in _assignment_targets(node):
                    line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                    assignments.append((index, node, line))
        _check(
            len(assignments) == 1,
            f"{path.name}: {gate} must be assigned exactly once, found {len(assignments)}",
        )
        index, node, line = assignments[0]
        is_false = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        )
        _check(is_false, f"{path.name}: {gate} must be assigned the constant False (cell {index})")
        _check("# @param" in line, f"{path.name}: {gate} must be a Colab form parameter (`# @param`)")
    for index, _source, tree in code_cells:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                _check(
                    not any(alias.name.startswith("google.colab") for alias in node.names),
                    f"{path.name}: google.colab must only be imported inside the BYOD gate (cell {index})",
                )


def _validate_embedded_modules(path: Path, notebook: dict, build, template: dict) -> list[int]:
    """PAR1: one tagged cell per carried module, in dependency order, each equal to its module after
    the documented rewrites (generator /2 multi-module carrier; ST2 applied per module)."""
    tagged = [
        (index, cell)
        for index, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code"
        and cell.get("metadata", {}).get("dimer", {}).get("embedded_module")
    ]
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    context = build.load_context(ROOT, template, recorded)
    expected_rels = context["module_rels"]
    _check(
        [cell["metadata"]["dimer"]["embedded_module"] for _, cell in tagged] == expected_rels,
        f"{path.name}: the cells tagged metadata.dimer.embedded_module must be exactly {expected_rels}, in order (ST2)",
    )
    for (index, cell), module in zip(tagged, context["modules"], strict=True):
        rel = f"{context['pkg_rel']}/{module}"
        _check(
            cell["metadata"]["dimer"].get("module_sha256") == context["per_module_sha256"][rel],
            f"{path.name}: cell {index} module_sha256 tag does not match {rel}",
        )
        _check(
            _cell_source(cell).rstrip("\n") + "\n" == context["embedded"][module],
            f"{path.name}: embedded module cell {index} differs from {rel} (PAR1); regenerate the notebook",
        )
    return [index for index, _ in tagged]


def _validate_identity(
    path: Path, code_cells: list[tuple[int, str, ast.Module]], embedded: list[int], revision: str
) -> None:
    """Identity constants are bound in the carried module only; nothing outside rebinds them."""
    for index, _source, tree in code_cells:
        if index in embedded:
            continue
        for node in ast.walk(tree):
            rebound = [name for name in _assignment_targets(node) if name in IDENTITY_NAMES]
            _check(
                not rebound,
                f"{path.name}: {rebound} must not be rebound outside the module cell (cell {index})",
            )
    outside = "\n".join(source for index, source, _ in code_cells if index not in embedded)
    manifest_block = re.search(r"^MANIFEST = (\{.*?^\})$", outside, re.M | re.S)
    _check(manifest_block is not None, f"{path.name}: model cell must carry an inline MANIFEST literal (ST3)")
    outside_without_manifest = outside.replace(manifest_block.group(0), "")
    _check(
        revision not in outside_without_manifest,
        f"{path.name}: the model revision may appear only in the carried module and the inline manifest",
    )


def _validate_parity(
    path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], build, template: dict
) -> None:
    """PAR2/PAR3: inline manifest and pins equal the repository's; the generator reproduces the file."""
    code = "\n".join(source for _, source, _ in code_cells)
    manifest = json.loads(_read(ROOT / "weights" / template["weights_key"] / "dimer-base-manifest.json"))
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    _check(
        inline is not None and json.loads(inline.group(1)) == manifest,
        f"{path.name}: inline MANIFEST != committed manifest (PAR2)",
    )
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    _check(pins_block is not None, f"{path.name}: install cell must carry PINS = [...] (ENV2)")
    _check(
        re.findall(r"'([^']+)'", pins_block.group(1)) == build._pins(ROOT, template),
        f"{path.name}: inline PINS != the repository's runtime pins (PAR2)",
    )
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = path.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    _check(
        current == rendered, f"{path.name}: differs from tools/build_notebook.py output (PAR3); regenerate"
    )


def _validate_bootstrap_guard(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    """The stale-import guard must actually raise: `if stale:` whose body raises RuntimeError."""
    raises = False
    for _, _, tree in code_cells:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "stale":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Raise) and isinstance(sub.exc, ast.Call):
                        func = sub.exc.func
                        if isinstance(func, ast.Name) and func.id == "RuntimeError":
                            raises = True
    _check(raises, f"{path.name}: install cell must raise RuntimeError when already-imported packages change")


def _validate_notebook_content(
    path: Path,
    code_cells: list[tuple[int, str, ast.Module]],
    markdown: str,
    embedded: list[int],
    spec: dict,
    template: dict,
) -> None:
    model_id, _revision = _package_identity(template)
    stripped = {index: _strip_comments(source) for index, source, _ in code_cells}
    code = "\n".join(stripped.values())
    outside = "\n".join(text for index, text in stripped.items() if index not in embedded)
    missing = [marker for marker in COMMON_CODE_MARKERS + spec["code_markers"] if marker not in code]
    _check(not missing, f"{path.name}: missing required source markers: {missing}")
    present = [
        label
        for label, pattern in FORBIDDEN_PATTERNS
        if pattern.search(outside if label in FORBIDDEN_PATTERNS_MODULE_EXEMPT else code)
    ]
    _check(not present, f"{path.name}: forbidden/insecure source: {present}")
    leaked = [marker for marker in FORBIDDEN_OUTSIDE_MODULE if marker in outside]
    _check(not leaked, f"{path.name}: direct library use outside the carried module cell (G2): {leaked}")
    install_index = code_cells[0][
        0
    ]  # the generator-owned install cell is the only place a subprocess may run
    after_install = "\n".join(
        text for index, text in stripped.items() if index not in embedded and index != install_index
    )
    workers = [marker for marker in FORBIDDEN_WORKER_CALLS if marker in after_install]
    _check(not workers, f"{path.name}: worker process or subprocess on the primary path (ST1): {workers}")
    _check(
        f"pipe = {MODEL_LOAD_EXPR}" in outside,
        f"{path.name}: must load through {MODEL_LOAD_EXPR} (INF1)",
    )
    _validate_gates(path, code_cells, spec["byod_gates"])
    _validate_bootstrap_guard(path, code_cells)
    for filename in spec["expected_outputs"]:
        _check(filename in code, f"{path.name}: must export {filename}")
    missing_md = [
        marker for marker in COMMON_MARKDOWN_MARKERS + spec["markdown_markers"] if marker not in markdown
    ]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    _check(f"**Profile:** `{spec['profile']}`" in markdown, f"{path.name}: markdown must state the profile")
    _check(f"https://huggingface.co/{model_id}" in markdown, f"{path.name}: references must link {model_id}")


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    names = {n.name for n in sorted(tutorials.glob("*.ipynb"))}
    unexpected = sorted(names - set(NOTEBOOKS))
    _check(not unexpected, f"undeclared tutorial notebooks (declare them in NOTEBOOKS): {unexpected}")
    absent = sorted(set(NOTEBOOKS) - names)
    _check(not absent, f"tutorial notebooks missing: {absent}")
    registry = _read(tutorials / "README.md")
    build = _load_tool("build_notebook")
    for name, spec in NOTEBOOKS.items():
        path = tutorials / name
        template = _load_tool(spec["template"]).TEMPLATE
        _check(template["notebook_name"] == name, f"tools/{spec['template']}.py must name {name}")
        _check(
            template["profile"] == spec["profile"],
            f"tools/{spec['template']}.py profile must be {spec['profile']}",
        )
        notebook = json.loads(_read(path))
        code_cells, markdown = _validate_notebook_structure(path, notebook, spec, template)
        embedded = _validate_embedded_modules(path, notebook, build, template)
        _model_id, revision = _package_identity(template)
        _validate_identity(path, code_cells, embedded, revision)
        _validate_notebook_content(path, code_cells, markdown, embedded, spec, template)
        # PAR2/PAR3 last: a content defect is reported by its own rule before the byte-parity rule (the
        # repository's negative-control tests rely on that order).
        _validate_parity(path, notebook, code_cells, build, template)
        _check(f"`{name}`" in registry, f"{name} missing from tutorials/README.md")
        _check(f"`{spec['profile']}`" in registry, f"tutorials/README.md must record `{spec['profile']}`")
    _check(
        f"DIMER Notebook Specification {NOTEBOOK_SPEC}" in registry,
        "tutorials/README.md must name the notebook spec version",
    )
    _check(
        "standalone" in registry.lower(), "tutorials/README.md must record that the notebooks are standalone"
    )


def validate_all() -> list[str]:
    validate_model_card()
    validate_identity_consistency()
    validate_release_status()
    validate_notebooks()
    return ["model-card", "identity-consistency", "release-status", "notebooks+parity"]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
