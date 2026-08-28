#!/usr/bin/env python3
"""Build one strict, curated Vast job for production truth editing.

The generated JSON is phase-specific. Legacy phase jobs retain their frozen
barriers; the adaptive job runs one coverage-gated 200-total-trial-minimum study
and may continue through 800 trials while measured time and budget capacity
remain. This tool only reads local inputs and writes JSON; it does not contact
Vast or any model provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.models import (  # noqa: E402
    DEFAULT_MODEL_CONTENT_SHA256,
    DEFAULT_SNAPSHOT_MANIFEST_SHA256,
)
from intelligent_liars.truth_editing_directions import DirectionBank  # noqa: E402
from intelligent_liars.truth_editing_capacity import (  # noqa: E402
    CapacityPolicy,
    validate_capacity_receipt,
)
from intelligent_liars.truth_editing_offhost_checkpoint import (  # noqa: E402
    OffHostCheckpointTarget,
)
from intelligent_liars.truth_editing_production import ProductionRunConfig  # noqa: E402
from intelligent_liars.truth_editing_study import (  # noqa: E402
    load_truth_editing_study_config,
)
from intelligent_liars.truth_editing_timed_canary import TimedCanaryConfig  # noqa: E402
from intelligent_liars.truth_editing_production_judge_budget import (  # noqa: E402
    ProductionJudgeBudgetConfig,
)
from intelligent_liars.truth_editing_vast_fleet import (  # noqa: E402
    ADAPTIVE_FLEET_FORMAT,
    EXPECTED_PRODUCTION_JUDGE_BUDGET_SHA256,
    FLEET_BUDGET_FORMAT,
    FLEET_FORMAT,
    FleetConfig,
)
from intelligent_liars.truth_editing_vast_prerequisites import (  # noqa: E402
    FROZEN_BASE_IMAGE,
    FROZEN_MODEL_ID,
    FROZEN_MODEL_REVISION,
    FROZEN_RUNTIME_ID,
)
from intelligent_liars.truth_editing_vast_production import (  # noqa: E402
    ADAPTIVE_PRODUCTION_JOB_FORMAT,
    FORMAT,
    ProductionVastConfig,
)


PHASE_BOUNDARIES = {"discovery": 80, "expanded": 160, "finalist": 200}
ADAPTIVE_PHASE_BOUNDARIES = {"discovery": 80, "expanded": 200, "finalist": 800}
ADAPTIVE_PHASE = "adaptive"
CANONICAL_ADAPTIVE_CAPACITY_POLICY = (
    "configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json"
)
CANONICAL_ADAPTIVE_CAPACITY_RECEIPT = (
    "artifacts/truth-editing/capacity/adaptive-capacity-receipt-v2.json"
)
CANONICAL_MODEL_REGISTRY_CONFIG = "configs/model_registry_v1.json"
CANONICAL_ADAPTIVE_OFFHOST_KEY_PREFIX = (
    "model-registry/v1/truth-editing/adaptive-main-r10"
)
CANONICAL_FINAL_MODEL_SLUG = "qwen3-vl-8b-truth-edited"
MAXIMUM_INFRASTRUCTURE_SPEND_USD = 45.0
MAXIMUM_HOST_LEASE_SECONDS = 24 * 3600
CANONICAL_TIMED_CANARY_CONFIG = (
    "configs/truth_editing_timed_canary_v6_adaptive_r10_bbb25ef3.json"
)
CANONICAL_TIMED_CANARY_CONFIG_SHA256 = (
    "7d0edca00f7934031225fa543538ab9e9349fb68fbcb3d587353a747b841b0b2"
)
CANONICAL_TIMED_CANARY_COMMAND_CONFIG = (
    "configs/truth_editing_timed_canary_command_v6_adaptive_r10_v4_3ae89929.json"
)
CANONICAL_TIMED_CANARY_COMMAND_SHA256 = (
    "3ae899296b0e56f4be9fdba6288a0d0924b7099865dc83ee9985e591d1bded0c"
)
_SECRET_PARTS = frozenset(
    {".aws", ".env", ".git", ".netrc", ".secrets", ".ssh", "credentials", "secrets", "token", "tokens"}
)
_REQUIRED_SCRIPTS = (
    "scripts/bootstrap_truth_editing_production_worker.sh",
    "scripts/bootstrap_truth_editing_prerequisite_worker.sh",
    "scripts/build_tinylora_model_cache.py",
    "scripts/hydrate_truth_editing_production_model.sh",
    "scripts/publish_truth_editing_production_checkpoint.py",
    "scripts/run_truth_editing_adaptive_finalization.py",
    "scripts/run_truth_editing_cuda_fleet_controller.py",
    "scripts/run_truth_editing_cuda_fleet_worker.py",
    "scripts/run_truth_editing_production_worker.py",
    "scripts/verify_truth_editing_production_model.py",
)
_REQUIRED_RUNTIME_FILES = (
    "docker/truth-editing/Dockerfile",
    "docker/truth-editing/requirements.in",
    "docker/truth-editing/requirements.lock",
    "docker/truth-editing/runtime-manifest.json",
    "docker/truth-editing/validate_runtime.py",
)
_TIMED_CANARY_SCRIPTS = (
    "scripts/run_truth_editing_timed_canary.py",
    "scripts/run_truth_editing_timed_canary_workload.py",
)
# Explicit transitive source closure of the production worker, adaptive
# controller/finalizer, timed canary, model-cache, and live-judge entrypoints.
# Keep this fail-closed: a newly introduced runtime dependency must be reviewed
# and added here instead of silently admitting the entire research package.
PRODUCTION_SOURCE_CLOSURE = (
    "src/intelligent_liars/__init__.py",
    "src/intelligent_liars/clients/__init__.py",
    "src/intelligent_liars/clients/openrouter_client.py",
    "src/intelligent_liars/heretic_truth_editing.py",
    "src/intelligent_liars/insider_grader.py",
    "src/intelligent_liars/judge_client.py",
    "src/intelligent_liars/judge_config.py",
    "src/intelligent_liars/judging.py",
    "src/intelligent_liars/model_cache.py",
    "src/intelligent_liars/model_registry.py",
    "src/intelligent_liars/models.py",
    "src/intelligent_liars/offline_judge_calibration.py",
    "src/intelligent_liars/progress.py",
    "src/intelligent_liars/roleplaying_grader.py",
    "src/intelligent_liars/rollout_grading.py",
    "src/intelligent_liars/rollouts.py",
    "src/intelligent_liars/sandbagging_grader.py",
    "src/intelligent_liars/tinylora_pilot.py",
    "src/intelligent_liars/truth_editing_adaptive_causal_preparation.py",
    "src/intelligent_liars/truth_editing_adaptive_finalization.py",
    "src/intelligent_liars/truth_editing_adaptive_run.py",
    "src/intelligent_liars/truth_editing_base_known.py",
    "src/intelligent_liars/truth_editing_batch_execution.py",
    "src/intelligent_liars/truth_editing_broad_coverage.py",
    "src/intelligent_liars/truth_editing_capacity.py",
    "src/intelligent_liars/truth_editing_causal_activation_controls.py",
    "src/intelligent_liars/truth_editing_component_basis.py",
    "src/intelligent_liars/truth_editing_contracts.py",
    "src/intelligent_liars/truth_editing_dataset_v2.py",
    "src/intelligent_liars/truth_editing_directions.py",
    "src/intelligent_liars/truth_editing_evaluator.py",
    "src/intelligent_liars/truth_editing_failure_policy.py",
    "src/intelligent_liars/truth_editing_final_checkpoint_publication.py",
    "src/intelligent_liars/truth_editing_finalist_checkpoint.py",
    "src/intelligent_liars/truth_editing_finalization_progress_store.py",
    "src/intelligent_liars/truth_editing_gpu_telemetry.py",
    "src/intelligent_liars/truth_editing_judge_contracts.py",
    "src/intelligent_liars/truth_editing_live_judge.py",
    "src/intelligent_liars/truth_editing_offhost_checkpoint.py",
    "src/intelligent_liars/truth_editing_pairwise_reconciliation.py",
    "src/intelligent_liars/truth_editing_phase_checkpoint.py",
    "src/intelligent_liars/truth_editing_preservation.py",
    "src/intelligent_liars/truth_editing_preservation_materialization.py",
    "src/intelligent_liars/truth_editing_preservation_runtime.py",
    "src/intelligent_liars/truth_editing_preservation_thresholds.py",
    "src/intelligent_liars/truth_editing_production.py",
    "src/intelligent_liars/truth_editing_production_causal_materializer.py",
    "src/intelligent_liars/truth_editing_production_finalization.py",
    "src/intelligent_liars/truth_editing_production_judge_budget.py",
    "src/intelligent_liars/truth_editing_qwen_causal_backend.py",
    "src/intelligent_liars/truth_editing_qwen_qualification.py",
    "src/intelligent_liars/truth_editing_qwen_runtime.py",
    "src/intelligent_liars/truth_editing_refusal_directions.py",
    "src/intelligent_liars/truth_editing_scenario_view.py",
    "src/intelligent_liars/truth_editing_structured_qualification.py",
    "src/intelligent_liars/truth_editing_structured_semantic.py",
    "src/intelligent_liars/truth_editing_study.py",
    "src/intelligent_liars/truth_editing_timed_canary.py",
    "src/intelligent_liars/truth_editing_vast_fleet.py",
    "src/intelligent_liars/truth_editing_vast_prerequisites.py",
    "src/intelligent_liars/truth_editing_vast_production.py",
    "src/intelligent_liars/truth_editing_wandb_checkpoint.py",
    "src/intelligent_liars/truth_editing_wandb_monitoring.py",
    "src/intelligent_liars/truth_editing_weight_editor.py",
)
_SEALED_OPTIMIZATION_DATASET_FILES = frozenset({"test.jsonl"})
_PRODUCTION_DIRECTORY_FIELDS = (
    "dataset_root",
    "scenario_view",
    "structured_semantic_view",
    "structured_semantic_source_root",
    "structured_base_known_qualification",
    "base_known_qualification",
    "refusal_artifact_root",
    "preservation_runtime_packet_root",
)
_PRODUCTION_FILE_FIELDS = (
    "study_config",
    "evaluator_config",
    "direction_manifest",
    "refusal_direction_config",
    "refusal_prompt_manifest",
    "refusal_direction_bank",
    "preservation_threshold_calibration",
)
# Only these strict openers dereference repository-relative paths from inside
# their JSON. Other JSON files (notably dataset provenance/source receipts)
# contain construction lineage, not runtime dependencies.
_TRANSITIVE_RUNTIME_REFERENCE_FIELDS = (
    "direction_manifest",
    "refusal_direction_bank",
    "preservation_threshold_calibration",
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unreadable: {path}") from error
    if not isinstance(raw, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return raw


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _production_path(repo: Path, value: str | Path) -> tuple[str, Path]:
    if not isinstance(value, (str, Path)):
        raise RuntimeError("production config must be a repository-relative JSON path")
    text = str(value)
    candidate = Path(text)
    if (
        not text
        or text != text.strip()
        or candidate.is_absolute()
        or candidate.parts[:1] != ("configs",)
        or len(candidate.parts) != 2
        or candidate.suffix != ".json"
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != text
    ):
        raise RuntimeError(
            "production config must be a normalized configs/<name>.json path"
        )
    path = repo.resolve() / candidate
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("production config is missing or unsafe")
    return text, path


def _strict_open_production_config(
    repo: Path,
    production_config: str | Path,
    *,
    opener: Any = ProductionRunConfig.open,
) -> tuple[str, Path, dict[str, Any], Any]:
    relative, path = _production_path(repo, production_config)
    raw = _read_object(path, "production config")
    try:
        parsed = opener(path)
    except Exception as error:
        raise RuntimeError(
            f"production config strict-open failed: {type(error).__name__}: {error}"
        ) from error
    if (
        parsed.verified_model_sha256 != DEFAULT_MODEL_CONTENT_SHA256
        or parsed.verified_snapshot_manifest_sha256
        != DEFAULT_SNAPSHOT_MANIFEST_SHA256
    ):
        raise RuntimeError("production config model identity differs from the Vast runtime")
    return relative, path, raw, parsed


def _resolve_portable(base: Path, raw: Any, label: str, *, repo: Path) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise RuntimeError(f"{label} must be nonempty relative text")
    path = Path(raw)
    if path.is_absolute():
        raise RuntimeError(f"{label} must be a portable relative path inside the repository")
    resolved = (base / path).resolve()
    if resolved != repo.resolve() and repo.resolve() not in resolved.parents:
        raise RuntimeError(f"{label} must be a portable relative path inside the repository")
    return resolved


def _relative(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"bundle input escapes repository: {path}") from error


def _adaptive_capacity_inputs(
    repo: Path,
    policy_relative: str = CANONICAL_ADAPTIVE_CAPACITY_POLICY,
    receipt_relative: str = CANONICAL_ADAPTIVE_CAPACITY_RECEIPT,
) -> tuple[Path, Path]:
    """Strict-open the post-canary policy and measured capacity receipt."""

    policy_path = _resolve_portable(repo, policy_relative, "capacity policy", repo=repo)
    receipt_path = _resolve_portable(
        repo, receipt_relative, "capacity receipt", repo=repo
    )
    if policy_path.is_symlink() or not policy_path.is_file():
        raise RuntimeError("adaptive capacity policy is missing or unsafe")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError(
            "adaptive capacity receipt is missing; generate it from the timed canary"
        )
    try:
        policy = CapacityPolicy.from_mapping(
            _read_object(policy_path, "adaptive capacity policy")
        )
        receipt = validate_capacity_receipt(
            _read_object(receipt_path, "adaptive capacity receipt")
        )
    except Exception as error:
        raise RuntimeError(
            f"adaptive capacity inputs failed strict validation: {type(error).__name__}: {error}"
        ) from error
    if receipt.get("policy_sha256") != policy.self_sha256:
        raise RuntimeError("adaptive capacity receipt policy identity differs")
    return policy_path, receipt_path


def _strict_open_fleet_config(
    repo: Path,
    fleet_config: str | Path,
) -> tuple[str, Path, FleetConfig]:
    """Open the exact portable fleet packet consumed by the adaptive controller."""

    fleet_relative, fleet_path = _production_path(repo, fleet_config)
    try:
        parsed = FleetConfig.from_mapping(_read_object(fleet_path, "fleet config"))
    except Exception as error:
        raise RuntimeError(
            f"adaptive fleet config failed strict validation: {type(error).__name__}: {error}"
        ) from error
    return fleet_relative, fleet_path, parsed


def _regular_files(root: Path) -> Iterable[Path]:
    if not root.exists() or root.is_symlink():
        raise RuntimeError(f"bundle input is missing or unsafe: {root}")
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise RuntimeError(f"bundle directory contains a symlink: {path}")
        if path.is_file():
            yield path


def _add_json_references(repo: Path, path: Path, selected: set[Path]) -> None:
    """Include existing in-repository files named by a selected JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            # Artifact locators can carry a row fragment (``file.npy#row/N``).
            # Ignore prose and other arbitrary JSON strings before touching the
            # filesystem; some qualification responses are thousands of bytes.
            # Construction-only HDF5 activation stores are intentionally not
            # admitted through transitive metadata references: production uses
            # the qualified direction arrays, never the 58 GiB source store.
            reference = value.split("#", 1)[0]
            if (
                not reference
                or len(reference) > 512
                or any(character.isspace() for character in reference)
                or Path(reference).is_absolute()
                or Path(reference).suffix.lower()
                not in {
                    ".csv",
                    ".json",
                    ".jsonl",
                    ".npy",
                    ".npz",
                    ".pt",
                    ".safetensors",
                    ".yaml",
                    ".yml",
                }
            ):
                return
            for base in (path.parent, repo):
                try:
                    candidate = (base / reference).resolve()
                except OSError:
                    continue
                if candidate == repo.resolve() or repo.resolve() not in candidate.parents:
                    continue
                if candidate.is_file() and not candidate.is_symlink():
                    selected.add(candidate)
                    break

    visit(payload)


def _add_structured_semantic_provenance(
    repo: Path,
    view_root: Path,
    selected: set[Path],
) -> None:
    """Include the exact unqualified view referenced by a qualified view.

    ``StructuredSemanticView.open`` verifies this source view again on the worker.
    The ordinary JSON reference walker intentionally follows files only, so this
    directory-valued provenance receipt needs one explicit, fail-closed seam.
    """

    manifest_path = view_root / "manifest.json"
    manifest = _read_object(manifest_path, "structured semantic view manifest")
    if manifest.get("format") != "truth_editing_structured_semantic_qualified_manifest_v1":
        return
    receipt = manifest.get("source_view")
    if not isinstance(receipt, Mapping) or set(receipt) != {"path", "view_sha256"}:
        raise RuntimeError("qualified structured semantic source_view receipt differs")
    expected_sha256 = receipt.get("view_sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise RuntimeError("qualified structured semantic source view identity is invalid")
    source_root = _resolve_portable(
        view_root,
        receipt.get("path"),
        "structured_semantic_view.source_view.path",
        repo=repo,
    )
    if source_root == view_root:
        raise RuntimeError("qualified structured semantic source view cannot reference itself")
    source_manifest = _read_object(
        source_root / "manifest.json", "structured semantic source view manifest"
    )
    if (
        source_manifest.get("format")
        != "truth_editing_structured_semantic_manifest_v1"
        or source_manifest.get("view_sha256") != expected_sha256
    ):
        raise RuntimeError(
            "qualified structured semantic source view identity differs from its receipt"
        )
    selected.update(_regular_files(source_root))


def collect_production_bundle_paths(
    repo: Path,
    *,
    production_config: str | Path,
    production_config_opener: Any = ProductionRunConfig.open,
) -> list[str]:
    repo = repo.resolve()
    _, production_path, production, _ = _strict_open_production_config(
        repo, production_config, opener=production_config_opener
    )
    if production.get("format") != "truth_editing_production_config_v1":
        raise RuntimeError("production config format differs")
    selected: set[Path] = {production_path}
    for relative in (*_REQUIRED_SCRIPTS, *_REQUIRED_RUNTIME_FILES):
        selected.update(_regular_files(repo / relative))
    for relative in PRODUCTION_SOURCE_CLOSURE:
        selected.update(_regular_files(repo / relative))
    base = production_path.parent
    for field in _PRODUCTION_FILE_FIELDS:
        if field not in production:
            raise RuntimeError(f"production config is missing {field}")
        selected.update(
            _regular_files(
                _resolve_portable(base, production[field], field, repo=repo)
            )
        )
    for field in _PRODUCTION_DIRECTORY_FIELDS:
        if field not in production:
            raise RuntimeError(f"production config is missing {field}")
        directory = _resolve_portable(base, production[field], field, repo=repo)
        files = _regular_files(directory)
        if field == "dataset_root":
            # Routine production and timed-canary jobs are optimization lanes.
            # Keep the signed full manifest, but never package the capability
            # test payload that the fleet contract marks inaccessible.
            files = (
                path
                for path in files
                if path.name not in _SEALED_OPTIMIZATION_DATASET_FILES
            )
        selected.update(files)
    structured_view_root = _resolve_portable(
        base,
        production["structured_semantic_view"],
        "structured_semantic_view",
        repo=repo,
    )
    _add_structured_semantic_provenance(repo, structured_view_root, selected)
    # Follow only paths that strict runtime openers actually dereference. A
    # provenance/source-receipt path is audit metadata and must never enlarge
    # the production bundle into raw construction data or sealed questions.
    for field in _TRANSITIVE_RUNTIME_REFERENCE_FIELDS:
        _add_json_references(
            repo,
            _resolve_portable(base, production[field], field, repo=repo),
            selected,
        )
    result = sorted(_relative(repo, path) for path in selected)
    for relative in result:
        if _SECRET_PARTS.intersection(Path(relative).parts):
            raise RuntimeError(f"secret-like bundle path is forbidden: {relative}")
        mode = (repo / relative).lstat().st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"bundle input is not a regular file: {relative}")
    return result


def _validate_phase_contract(
    repo: Path, production_path: Path, *, phase: str
) -> None:
    production = _read_object(production_path, "production config")
    study_path = _resolve_portable(
        production_path.parent,
        production.get("study_config"),
        "study_config",
        repo=repo,
    )
    study = _read_object(study_path, "production study config")
    tiers = study.get("evaluation_tiers")
    if not isinstance(tiers, list):
        raise RuntimeError("production study evaluation tiers are missing")
    actual = {
        row.get("name"): row.get("through_trial")
        for row in tiers
        if isinstance(row, Mapping)
    }
    if phase == ADAPTIVE_PHASE:
        policy = study.get("search_policy")
        expected_policy = {
            "format": "truth_editing_adaptive_search_policy_v1",
            "minimum_trials": 200,
            "maximum_trials": 800,
            "search_elapsed_limit_seconds": 21 * 3600,
            "reserve_elapsed_seconds": 3 * 3600,
            "all_in_budget_usd": "50",
            "maximum_infrastructure_spend_usd": "45",
            "maximum_evaluation_spend_usd": "5",
            "evaluation_budget_reserve_fraction": "0.20",
            "evaluation_spend_reserve_usd": "1",
            "broad_coverage": {
                "required_before_concentration": True,
            },
        }
        if (
            actual != ADAPTIVE_PHASE_BOUNDARIES
            or study.get("max_trials") != 800
            or study.get("batch_size") != 8
            or policy != expected_policy
        ):
            raise RuntimeError(
                "adaptive production study must retain 80/200/800 tiers, "
                "a 200-trial total floor, coverage-gated concentration, and the "
                "frozen 21h + 3h budget policy"
            )
        return
    if (
        actual != PHASE_BOUNDARIES
        or study.get("max_trials") != 200
        or study.get("batch_size") != 8
    ):
        raise RuntimeError("production study must retain the exact 80/160/200 batch barriers")


def _optuna_study_name(repo: Path, production_path: Path) -> str:
    production = _read_object(production_path, "production config")
    study_path = _resolve_portable(
        production_path.parent,
        production.get("study_config"),
        "study_config",
        repo=repo,
    )
    manifest_path = _resolve_portable(
        production_path.parent,
        production.get("direction_manifest"),
        "direction_manifest",
        repo=repo,
    )
    direction_root = _resolve_portable(
        production_path.parent,
        production.get("direction_root"),
        "direction_root",
        repo=repo,
    )
    study = load_truth_editing_study_config(study_path)
    bank = DirectionBank.open(manifest_path, root=direction_root)
    direction_payload = [
        [item.direction_id, item.artifact.vector_sha256]
        for item in bank.manifest.directions
    ]
    direction_identity = hashlib.sha256(
        json.dumps(
            direction_payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return f"{study.study_id}-{study.identity_sha256[:12]}-{direction_identity[:12]}"


def _production_budget_identity(production_path: Path) -> str:
    production = _read_object(production_path, "production config")
    try:
        judge_budget = ProductionJudgeBudgetConfig.from_mapping(
            production["judge_budget"]
        )
    except Exception as error:
        raise RuntimeError(
            "production config must contain the frozen 45 USD infrastructure + 5 USD judge budget"
        ) from error
    if (
        judge_budget.identity_sha256 != EXPECTED_PRODUCTION_JUDGE_BUDGET_SHA256
        or str(judge_budget.all_in_maximum_spend_usd) != "50"
        or str(judge_budget.non_judge_reserved_spend_usd) != "45"
        or str(judge_budget.maximum_judge_spend_usd) != "5"
    ):
        raise RuntimeError(
            "production config must contain the frozen 45 USD infrastructure + 5 USD judge budget"
        )
    return judge_budget.identity_sha256


def build_production_fleet_config(
    repo: Path,
    *,
    production_config: str | Path,
    bundle_sha256: str,
    fleet_id: str = "truth-editing-production",
    receipt_directory: str = "artifacts/truth-editing/fleet/trials",
    maximum_host_lease_seconds: int = MAXIMUM_HOST_LEASE_SECONDS,
    adaptive_capacity_policy: str | Path | None = None,
    production_config_opener: Any = ProductionRunConfig.open,
    study_config_identity_opener: Any = (
        lambda path: load_truth_editing_study_config(path).identity_sha256
    ),
) -> dict[str, Any]:
    """Build the identity-bound companion config for one persistent 8-GPU host."""

    repo = repo.resolve()
    production_relative, production_path, production_raw, _ = _strict_open_production_config(
        repo, production_config, opener=production_config_opener
    )
    judge_budget_sha256 = _production_budget_identity(production_path)
    adaptive_policy_relative: str | None = None
    adaptive_policy_sha256: str | None = None
    study_relative: str | None = None
    study_sha256: str | None = None
    study_config_identity: str | None = None
    if adaptive_capacity_policy is not None:
        adaptive_policy_relative, adaptive_policy_path = _production_path(
            repo, adaptive_capacity_policy
        )
        CapacityPolicy.from_mapping(
            _read_object(adaptive_policy_path, "adaptive capacity policy")
        )
        adaptive_policy_sha256 = file_sha256(adaptive_policy_path)
        study_path = _resolve_portable(
            production_path.parent,
            production_raw.get("study_config"),
            "study_config",
            repo=repo,
        )
        study_relative = _relative(repo, study_path)
        study_sha256 = file_sha256(study_path)
        study_config_identity = study_config_identity_opener(study_path)
        if (
            not isinstance(study_config_identity, str)
            or not re.fullmatch(r"[0-9a-f]{64}", study_config_identity)
        ):
            raise RuntimeError("adaptive study config identity is invalid")
    unsigned_budget = {
        "format": FLEET_BUDGET_FORMAT,
        "all_in_maximum_spend_usd": "50",
        "maximum_infrastructure_spend_usd": "45",
        "maximum_judge_spend_usd": "5",
        "production_judge_budget_config_sha256": judge_budget_sha256,
        "included_infrastructure_costs": [
            "gpu_compute",
            "storage",
            "network_download",
            "network_upload",
        ],
        "maximum_host_lease_seconds": maximum_host_lease_seconds,
        "maximum_fetch_gib": 1.0,
    }
    raw = {
        "format": (
            ADAPTIVE_FLEET_FORMAT
            if adaptive_capacity_policy is not None
            else FLEET_FORMAT
        ),
        "fleet_id": fleet_id,
        "budget": {
            **unsigned_budget,
            "identity_sha256": hashlib.sha256(
                json.dumps(
                    unsigned_budget,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "production_config": {
            "path": production_relative,
            "sha256": file_sha256(production_path),
        },
        "receipt_directory": receipt_directory,
        "capability_test_access": False,
    }
    if adaptive_capacity_policy is None:
        raw.update(
            {
                "phase_boundaries": dict(PHASE_BOUNDARIES),
                "execution_mode": "persistent_single_host_eight_gpu",
                "worker_count": 8,
                "bundle_sha256": bundle_sha256,
            }
        )
    else:
        assert None not in {
            adaptive_policy_relative,
            adaptive_policy_sha256,
            study_relative,
            study_sha256,
            study_config_identity,
        }
        raw.update(
            {
                "adaptive_capacity_policy": {
                    "path": adaptive_policy_relative,
                    "sha256": adaptive_policy_sha256,
                },
                "study": {
                    "path": study_relative,
                    "config_sha256": study_sha256,
                    "identity_sha256": study_config_identity,
                },
                "execution_topology": {
                    "mode": "persistent_single_host_eight_gpu",
                    "worker_count": 8,
                    "batch_size": 8,
                },
            }
        )
    return dict(FleetConfig.from_mapping(raw).identity)


def build_production_job(
    repo: Path,
    *,
    production_config: str | Path,
    phase: str,
    maximum_elapsed_seconds: int | None = None,
    maximum_cost_usd: float | None = None,
    optuna_study_name: str | None = None,
    fleet_config: str | Path | None = None,
    production_config_opener: Any = ProductionRunConfig.open,
    adaptive_capacity_opener: Any = _adaptive_capacity_inputs,
) -> dict[str, Any]:
    repo = repo.resolve()
    production_relative, production_path, production_raw, _ = _strict_open_production_config(
        repo, production_config, opener=production_config_opener
    )
    production_sha256 = file_sha256(production_path)
    if phase not in {*PHASE_BOUNDARIES, ADAPTIVE_PHASE}:
        raise RuntimeError("phase must be discovery, expanded, finalist, or adaptive")
    if maximum_elapsed_seconds is None:
        maximum_elapsed_seconds = (
            MAXIMUM_HOST_LEASE_SECONDS
            if phase == ADAPTIVE_PHASE
            else 6 * 3600
        )
    if maximum_cost_usd is None:
        maximum_cost_usd = (
            MAXIMUM_INFRASTRUCTURE_SPEND_USD
            if phase == ADAPTIVE_PHASE
            else 8.0
        )
    maximum_phase_cost = (
        MAXIMUM_INFRASTRUCTURE_SPEND_USD
        if phase == ADAPTIVE_PHASE
        else 15.0
    )
    if (
        isinstance(maximum_cost_usd, bool)
        or not 0 < maximum_cost_usd
        or (
            maximum_cost_usd > maximum_phase_cost
            if phase == ADAPTIVE_PHASE
            else maximum_cost_usd >= maximum_phase_cost
        )
    ):
        raise RuntimeError(
            "adaptive production infrastructure must be capped at 45 USD"
            if phase == ADAPTIVE_PHASE
            else "legacy production infrastructure must remain strictly below 15 USD"
        )
    if (
        isinstance(maximum_elapsed_seconds, bool)
        or not 60 <= maximum_elapsed_seconds <= (
            MAXIMUM_HOST_LEASE_SECONDS
            if phase == ADAPTIVE_PHASE
            else 18 * 3600
        )
    ):
        raise RuntimeError(
            "adaptive production host lease must be capped at 24 hours"
            if phase == ADAPTIVE_PHASE
            else "legacy production host lease must be capped at 18 hours"
        )
    _production_budget_identity(production_path)
    _validate_phase_contract(repo, production_path, phase=phase)
    study_config_path = _resolve_portable(
        production_path.parent,
        production_raw.get("study_config"),
        "study_config",
        repo=repo,
    )
    study_config_sha256 = file_sha256(study_config_path)
    resolved_optuna_name = optuna_study_name or _optuna_study_name(
        repo, production_path
    )
    if not resolved_optuna_name or resolved_optuna_name != resolved_optuna_name.strip():
        raise RuntimeError("Optuna study name must be nonempty trimmed text")
    bundle_paths = collect_production_bundle_paths(
        repo,
        production_config=production_relative,
        production_config_opener=production_config_opener,
    )
    adaptive_policy_path: Path | None = None
    adaptive_receipt_path: Path | None = None
    adaptive_fleet_relative: str | None = None
    if phase == ADAPTIVE_PHASE:
        if fleet_config is None:
            raise RuntimeError("adaptive production requires an immutable fleet config")
        adaptive_fleet_relative, adaptive_fleet_path, adaptive_fleet = (
            _strict_open_fleet_config(repo, fleet_config)
        )
        adaptive_policy_path, adaptive_receipt_path = adaptive_capacity_opener(repo)
        model_registry_relative, model_registry_path = _production_path(
            repo, CANONICAL_MODEL_REGISTRY_CONFIG
        )
        OffHostCheckpointTarget.from_model_registry_config(
            model_registry_path,
            key_prefix=CANONICAL_ADAPTIVE_OFFHOST_KEY_PREFIX,
        )
        if (
            adaptive_fleet.format != ADAPTIVE_FLEET_FORMAT
            or adaptive_fleet.adaptive_capacity_policy_path
            != _relative(repo, adaptive_policy_path)
            or adaptive_fleet.adaptive_capacity_policy_sha256
            != file_sha256(adaptive_policy_path)
            or adaptive_fleet.study_config_path != _relative(repo, study_config_path)
            or adaptive_fleet.study_config_sha256 != study_config_sha256
            or adaptive_fleet.production_config_path != production_relative
            or adaptive_fleet.production_config_sha256 != production_sha256
        ):
            raise RuntimeError("adaptive fleet input identity differs")
        bundle_paths = sorted(
            {
                *bundle_paths,
                "scripts/run_truth_editing_causal_activation_controls.py",
                adaptive_fleet_relative,
                _relative(repo, adaptive_policy_path),
                _relative(repo, adaptive_receipt_path),
                model_registry_relative,
            }
        )
    elif fleet_config is not None:
        raise RuntimeError("legacy production phases do not accept a fleet config")
    expected = [
        "study/study-journal.json",
        "study/study-journal.json.optuna.log",
        "monitoring/wandb-run.json",
        "model/cache-hydration-receipt.json",
        "model/model-verification-receipt.json",
    ]
    if phase in {"finalist", ADAPTIVE_PHASE}:
        expected.extend(
            [
                "study/frozen/study-report.json",
                "study/frozen/study-artifact-receipt.json",
            ]
        )
    if phase == ADAPTIVE_PHASE:
        expected.extend(
            [
                "adaptive-controller-result.json",
                "checkpoints/adaptive-latest.json",
                "study/adaptive-run-checkpoint.json",
                "study/adaptive-run-receipt.json",
                "monitoring/adaptive-progress.json",
                "monitoring/rolling-capacity-receipt.json",
                "providers/production-judge-budget/finalization-receipt.json",
                "finalization/adaptive-finalization-handoff.json",
                "finalization/adaptive-finalization-audit.json",
                "finalization/audited-selection-receipt.json",
                "finalization/adaptive-finalization-receipt.json",
                "finalization/final-model-publication-receipt.json",
                "finalization/checkpoint-publication/checkpoint-manifest.json",
                "finalization/checkpoint-publication/selection-receipt.json",
                "finalization/checkpoint-publication/control-schedule-receipt.json",
                "finalization/checkpoint-publication/registry-entry-proposal.json",
                "finalization/checkpoint-publication/publication-receipt.json",
            ]
        )
    else:
        expected.insert(0, "production-run.json")
        expected.insert(4, "checkpoints/latest.json")
    workload = [
        "python",
        (
            "scripts/run_truth_editing_cuda_fleet_controller.py"
            if phase == ADAPTIVE_PHASE
            else "scripts/run_truth_editing_production_worker.py"
        ),
    ]
    if phase == ADAPTIVE_PHASE:
        assert adaptive_fleet_relative is not None
        workload.extend(
            [
                "--fleet-config",
                adaptive_fleet_relative,
                "--config",
                production_relative,
                "--capacity-policy",
                _relative(repo, adaptive_policy_path),
                "--capacity-receipt",
                _relative(repo, adaptive_receipt_path),
                "--output-root",
                "/workspace/outputs",
                "--adaptive-checkpoint",
                "/workspace/outputs/study/adaptive-run-checkpoint.json",
                "--checkpoint-publication-root",
                "/workspace/outputs/checkpoints",
                "--model-registry-config",
                model_registry_relative,
                "--offhost-key-prefix",
                CANONICAL_ADAPTIVE_OFFHOST_KEY_PREFIX,
                "--final-model-slug",
                CANONICAL_FINAL_MODEL_SLUG,
                "--receipt",
                "/workspace/outputs/study/adaptive-run-receipt.json",
            ]
        )
    else:
        workload.extend(
            [
        "--config",
        production_relative,
        "--output-root",
        "/workspace/outputs",
        "--expected-config-sha256",
        production_sha256,
        "--phase",
        phase,
            ]
        )
    stream = [
        "python",
        "scripts/publish_truth_editing_production_checkpoint.py",
        "--journal",
        "/workspace/outputs/study/study-journal.json",
        "--output",
        "/workspace/outputs/checkpoints",
    ]
    if phase == ADAPTIVE_PHASE:
        stream.extend(
            [
                "--adaptive",
                "--study-config-sha256",
                study_config_sha256,
            ]
        )
    else:
        stream.extend(["--phase", phase])
    stream.extend(["--optuna-study-name", resolved_optuna_name])
    return {
        "format": FORMAT,
        "phase": phase,
        "base_job": {
            "format": (
                ADAPTIVE_PRODUCTION_JOB_FORMAT
                if phase == ADAPTIVE_PHASE
                else "truth_editing_vast_prerequisite_job_v1"
            ),
            "image": FROZEN_BASE_IMAGE,
            "runtime_id": FROZEN_RUNTIME_ID,
            "model": {"repository": FROZEN_MODEL_ID, "revision": FROZEN_MODEL_REVISION},
            "resources": {
                "disk_gib": 120,
                "minimum_gpu_vram_gib": 24,
                "maximum_elapsed_seconds": maximum_elapsed_seconds,
                "maximum_cost_usd": maximum_cost_usd,
                "maximum_download_gib": 25.0,
                "maximum_upload_gib": 1.0,
            },
            "paths": {
                "remote_workdir": "/workspace/intelligent_liars",
                "remote_output_dir": "/workspace/outputs",
            },
            "commands": {
                "bootstrap": ["bash", "scripts/bootstrap_truth_editing_production_worker.sh"],
                "workload": workload,
            },
            "expected_outputs": expected,
            "bundle_paths": bundle_paths,
        },
        "model_cache": {
            "remote_directory": "/workspace/model-source",
            "expected_model_sha256": DEFAULT_MODEL_CONTENT_SHA256,
            "expected_snapshot_manifest_sha256": DEFAULT_SNAPSHOT_MANIFEST_SHA256,
            "hydrate_command": ["bash", "scripts/hydrate_truth_editing_production_model.sh"],
            "verify_command": ["python", "scripts/verify_truth_editing_production_model.py"],
        },
        "study": {
            "production_config_path": production_relative,
            "production_config_sha256": production_sha256,
            "workload_command": workload,
        },
        "checkpoints": {
            "remote_directory": "/workspace/outputs/checkpoints",
            "interval_seconds": 300,
            "stream_command": stream,
        },
    }


def build_timed_canary_job(
    repo: Path,
    *,
    canary_config: str | Path,
    command_config: str | Path,
    maximum_elapsed_seconds: int = 90 * 60,
    maximum_cost_usd: float = 4.0,
    production_config_opener: Any = ProductionRunConfig.open,
) -> dict[str, Any]:
    """Build one bounded one-GPU production-parity timed-canary job."""

    repo = repo.resolve()
    canary_relative, canary_path = _production_path(repo, canary_config)
    command_relative, command_path = _production_path(repo, command_config)
    if (
        canary_relative != CANONICAL_TIMED_CANARY_CONFIG
        or file_sha256(canary_path) != CANONICAL_TIMED_CANARY_CONFIG_SHA256
        or command_relative != CANONICAL_TIMED_CANARY_COMMAND_CONFIG
        or file_sha256(command_path) != CANONICAL_TIMED_CANARY_COMMAND_SHA256
    ):
        raise RuntimeError(
            "timed-canary packet is superseded; canonical command must bind W&B entity centipawn"
        )
    try:
        canary = TimedCanaryConfig.from_mapping(
            _read_object(canary_path, "timed-canary config")
        )
    except Exception as error:
        raise RuntimeError(
            f"timed-canary config strict-open failed: {type(error).__name__}: {error}"
        ) from error
    production_relative, production_path, _, _ = _strict_open_production_config(
        repo,
        canary.production_config_path,
        opener=production_config_opener,
    )
    production_sha256 = file_sha256(production_path)
    if (
        production_sha256 != canary.production_config_sha256
        or canary.model_sha256 != DEFAULT_MODEL_CONTENT_SHA256
    ):
        raise RuntimeError("timed-canary production or model identity differs")
    try:
        command = json.loads(command_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("timed-canary command config is unreadable") from error
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in command
        )
        or "scripts/run_truth_editing_timed_canary_workload.py" not in command
        or production_relative not in command
        or production_sha256 not in command
        or "--wandb-project" not in command
        or "--wandb-entity" not in command
        or command[command.index("--wandb-project") + 1] != "intelligent-liars"
        or command[command.index("--wandb-entity") + 1] != "centipawn"
    ):
        raise RuntimeError("timed-canary command does not bind the canonical workload")
    project_indexes = [
        index for index, value in enumerate(command[:-1]) if value == "--wandb-project"
    ]
    entity_indexes = [
        index for index, value in enumerate(command[:-1]) if value == "--wandb-entity"
    ]
    if (
        len(project_indexes) != 1
        or command[project_indexes[0] + 1] != "intelligent-liars"
        or len(entity_indexes) != 1
        or command[entity_indexes[0] + 1] != "centipawn"
    ):
        raise RuntimeError(
            "timed-canary command must explicitly bind W&B project intelligent-liars and entity centipawn"
        )
    if (
        isinstance(maximum_elapsed_seconds, bool)
        or maximum_elapsed_seconds < canary.maximum_wall_seconds
        or maximum_elapsed_seconds > MAXIMUM_HOST_LEASE_SECONDS
    ):
        raise RuntimeError("timed-canary lease must cover its wall-clock limit")
    if (
        isinstance(maximum_cost_usd, bool)
        or not 0 < maximum_cost_usd <= MAXIMUM_INFRASTRUCTURE_SPEND_USD
    ):
        raise RuntimeError("timed-canary infrastructure must be capped at 45 USD")

    selected = set(
        collect_production_bundle_paths(
            repo,
            production_config=production_relative,
            production_config_opener=production_config_opener,
        )
    )
    selected.update({canary_relative, command_relative, *_TIMED_CANARY_SCRIPTS})
    bundle_paths = sorted(selected)
    output_root = "/workspace/outputs/timed-canary-v6-adaptive-r10"
    outer_parts = [
        "python",
        "scripts/run_truth_editing_timed_canary.py",
        "--config",
        canary_relative,
        "--repo",
        ".",
        "--command-json",
        command_relative,
        "--observation",
        f"{output_root}/observation.json",
        "--receipt",
        f"{output_root}/receipt.json",
    ]
    # The exact offer price is known only after offer revalidation. The Vast
    # production lifecycle exports it immediately before this inner shell runs.
    outer_shell = (
        shlex.join(outer_parts)
        + ' --gpu-hourly-usd "$TRUTH_EDITING_GPU_HOURLY_USD" --execute'
    )
    workload = ["bash", "-lc", outer_shell]
    expected = [
        "timed-canary-v6-adaptive-r10/observation.json",
        "timed-canary-v6-adaptive-r10/receipt.json",
        "timed-canary-v6-adaptive-r10/monitoring/wandb-run.json",
        "timed-canary-v6-adaptive-r10/monitoring/wandb-events.jsonl",
        "timed-canary-v6-adaptive-r10/providers/production-judge-budget/manifest.json",
        "model/cache-hydration-receipt.json",
        "model/model-verification-receipt.json",
    ]
    return {
        "format": FORMAT,
        "phase": "timed_canary",
        "base_job": {
            "format": "truth_editing_vast_prerequisite_job_v1",
            "image": FROZEN_BASE_IMAGE,
            "runtime_id": FROZEN_RUNTIME_ID,
            "model": {
                "repository": FROZEN_MODEL_ID,
                "revision": FROZEN_MODEL_REVISION,
            },
            "resources": {
                "disk_gib": 120,
                "minimum_gpu_vram_gib": 24,
                "maximum_elapsed_seconds": maximum_elapsed_seconds,
                "maximum_cost_usd": maximum_cost_usd,
                "maximum_download_gib": 25.0,
                "maximum_upload_gib": 1.0,
            },
            "paths": {
                "remote_workdir": "/workspace/intelligent_liars",
                "remote_output_dir": "/workspace/outputs",
            },
            "commands": {
                "bootstrap": [
                    "bash",
                    "scripts/bootstrap_truth_editing_production_worker.sh",
                ],
                "workload": workload,
            },
            "expected_outputs": expected,
            "bundle_paths": bundle_paths,
        },
        "model_cache": {
            "remote_directory": "/workspace/model-source",
            "expected_model_sha256": DEFAULT_MODEL_CONTENT_SHA256,
            "expected_snapshot_manifest_sha256": (
                DEFAULT_SNAPSHOT_MANIFEST_SHA256
            ),
            "hydrate_command": [
                "bash",
                "scripts/hydrate_truth_editing_production_model.sh",
            ],
            "verify_command": [
                "python",
                "scripts/verify_truth_editing_production_model.py",
            ],
        },
        "study": {
            "production_config_path": production_relative,
            "production_config_sha256": production_sha256,
            "workload_command": workload,
        },
        "checkpoints": {
            "remote_directory": f"{output_root}/monitoring",
            "interval_seconds": int(canary.maximum_wall_seconds),
            "stream_command": ["true"],
        },
    }


def _write_new(path: Path, value: Any) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError(f"refusing to replace differing production job: {path}")
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_adaptive_fleet_config(
    repo: Path,
    *,
    production_config: str | Path,
    output: Path,
    capacity_policy: str | Path = CANONICAL_ADAPTIVE_CAPACITY_POLICY,
    production_config_opener: Any = ProductionRunConfig.open,
    study_config_identity_opener: Any = (
        lambda path: load_truth_editing_study_config(path).identity_sha256
    ),
) -> dict[str, Any]:
    """Write and strict-reopen one immutable adaptive fleet-v3 packet.

    Adaptive fleet v3 deliberately does not bind the archive hash because the
    archive contains the fleet packet itself.  The all-zero digest supplied to
    the shared builder is therefore only a legacy-signature placeholder and is
    absent from the emitted v3 identity.
    """

    repo = repo.resolve()
    raw = build_production_fleet_config(
        repo,
        production_config=production_config,
        bundle_sha256="0" * 64,
        adaptive_capacity_policy=capacity_policy,
        production_config_opener=production_config_opener,
        study_config_identity_opener=study_config_identity_opener,
    )
    if raw.get("format") != ADAPTIVE_FLEET_FORMAT:
        raise RuntimeError("adaptive fleet writer produced a non-v3 packet")
    _write_new(output, raw)
    reopened = FleetConfig.from_mapping(_read_object(output, "adaptive fleet output"))
    if dict(reopened.identity) != raw:
        raise RuntimeError("adaptive fleet output changed during strict reopen")
    return raw


def write_adaptive_production_job(
    repo: Path,
    *,
    production_config: str | Path,
    fleet_config: str | Path,
    output: Path,
    capacity_policy: str | Path = CANONICAL_ADAPTIVE_CAPACITY_POLICY,
    capacity_receipt: str | Path = CANONICAL_ADAPTIVE_CAPACITY_RECEIPT,
    maximum_elapsed_seconds: int | None = None,
    maximum_cost_usd: float | None = None,
    optuna_study_name: str | None = None,
    production_config_opener: Any = ProductionRunConfig.open,
    adaptive_capacity_opener: Any | None = None,
) -> dict[str, Any]:
    """Write and strict-reopen the adaptive Vast job bound to measured capacity."""

    repo = repo.resolve()

    def open_capacity_inputs(open_repo: Path) -> tuple[Path, Path]:
        if adaptive_capacity_opener is not None:
            return adaptive_capacity_opener(open_repo)
        return _adaptive_capacity_inputs(
            open_repo,
            str(capacity_policy),
            str(capacity_receipt),
        )

    raw = build_production_job(
        repo,
        production_config=production_config,
        phase=ADAPTIVE_PHASE,
        maximum_elapsed_seconds=maximum_elapsed_seconds,
        maximum_cost_usd=maximum_cost_usd,
        optuna_study_name=optuna_study_name,
        fleet_config=fleet_config,
        production_config_opener=production_config_opener,
        adaptive_capacity_opener=open_capacity_inputs,
    )
    _write_new(output, raw)
    reopened_raw = _read_object(output, "adaptive production job output")
    reopened = ProductionVastConfig.from_mapping(reopened_raw)
    if reopened.phase != ADAPTIVE_PHASE or reopened_raw != raw:
        raise RuntimeError("adaptive production job changed during strict reopen")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--production-config",
        help="Exact repository-relative configs/<content-qualified-name>.json",
    )
    parser.add_argument(
        "--fleet-config",
        help="Exact repository-relative fleet packet; required only for adaptive",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", choices=(*PHASE_BOUNDARIES, ADAPTIVE_PHASE))
    mode.add_argument("--timed-canary-config")
    mode.add_argument(
        "--build-adaptive-fleet",
        action="store_true",
        help="Write adaptive fleet v3 from --production-config and the capacity policy",
    )
    parser.add_argument("--timed-canary-command-config")
    parser.add_argument(
        "--adaptive-capacity-policy",
        default=CANONICAL_ADAPTIVE_CAPACITY_POLICY,
        help="Repository-relative canonical adaptive capacity policy",
    )
    parser.add_argument(
        "--adaptive-capacity-receipt",
        default=CANONICAL_ADAPTIVE_CAPACITY_RECEIPT,
        help="Repository-relative post-canary capacity receipt for the adaptive job",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-elapsed-seconds", type=int)
    parser.add_argument("--maximum-cost-usd", type=float)
    args = parser.parse_args(argv)
    if args.build_adaptive_fleet:
        if (
            args.production_config is None
            or args.fleet_config is not None
            or args.timed_canary_command_config is not None
            or args.maximum_elapsed_seconds is not None
            or args.maximum_cost_usd is not None
            or args.adaptive_capacity_receipt
            != CANONICAL_ADAPTIVE_CAPACITY_RECEIPT
        ):
            parser.error(
                "adaptive fleet requires only --production-config, "
                "--adaptive-capacity-policy, and --output"
            )
        value = write_adaptive_fleet_config(
            args.repo,
            production_config=args.production_config,
            capacity_policy=args.adaptive_capacity_policy,
            output=args.output,
        )
    elif args.timed_canary_config is not None:
        if (
            args.production_config is not None
            or args.fleet_config is not None
            or not args.timed_canary_command_config
            or args.adaptive_capacity_policy != CANONICAL_ADAPTIVE_CAPACITY_POLICY
            or args.adaptive_capacity_receipt != CANONICAL_ADAPTIVE_CAPACITY_RECEIPT
        ):
            parser.error(
                "timed canary requires --timed-canary-command-config and no production or fleet config"
            )
        value = build_timed_canary_job(
            args.repo,
            canary_config=args.timed_canary_config,
            command_config=args.timed_canary_command_config,
            maximum_elapsed_seconds=(
                90 * 60
                if args.maximum_elapsed_seconds is None
                else args.maximum_elapsed_seconds
            ),
            maximum_cost_usd=(
                4.0 if args.maximum_cost_usd is None else args.maximum_cost_usd
            ),
        )
        _write_new(args.output, value)
    else:
        if args.production_config is None or args.timed_canary_command_config is not None:
            parser.error(
                "phase jobs require --production-config and no timed-canary command config"
            )
        if args.phase == ADAPTIVE_PHASE:
            value = write_adaptive_production_job(
                args.repo,
                production_config=args.production_config,
                fleet_config=args.fleet_config,
                capacity_policy=args.adaptive_capacity_policy,
                capacity_receipt=args.adaptive_capacity_receipt,
                output=args.output,
                maximum_elapsed_seconds=args.maximum_elapsed_seconds,
                maximum_cost_usd=args.maximum_cost_usd,
            )
        else:
            if (
                args.adaptive_capacity_policy
                != CANONICAL_ADAPTIVE_CAPACITY_POLICY
                or args.adaptive_capacity_receipt
                != CANONICAL_ADAPTIVE_CAPACITY_RECEIPT
            ):
                parser.error("legacy phase jobs do not accept adaptive capacity inputs")
            value = build_production_job(
                args.repo,
                production_config=args.production_config,
                phase=args.phase,
                fleet_config=args.fleet_config,
                maximum_elapsed_seconds=args.maximum_elapsed_seconds,
                maximum_cost_usd=args.maximum_cost_usd,
            )
            _write_new(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
