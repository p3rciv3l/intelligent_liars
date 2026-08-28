"""Immutable v3-to-v4 development calibration contract migration.

This module does not call a judge. It carries the already amended development
examples and labels forward while adding the public pairwise comparison kind
required by the v4 prompt contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .truth_editing_judge_dev_replacement import (
    build_dev_compiler_adapter,
)
from .truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    LIVE_CALIBRATION_PLAN_FORMAT,
    LiveJudgeError,
    _load_calibration_plan,
)


_PAIRWISE_COMPARISON_KINDS = {
    "self_pair", "exact_duplicate", "known_dominance", "invalid_comparison"
}


class DevV4MigrationError(ValueError):
    """The immutable development calibration migration is invalid."""


@dataclass(frozen=True)
class DevV4Artifacts:
    plan: dict[str, Any]
    pack: dict[str, Any]
    labels: dict[str, Any]
    provenance: dict[str, Any]
    commands: dict[str, Any]


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise DevV4MigrationError("development v4 value is not canonical JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(value))
    return {**unsigned, "content_sha256": _hash(unsigned)}


def _load_signed(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DevV4MigrationError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DevV4MigrationError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise DevV4MigrationError(f"{name} must contain an object")
    identity = value.get("content_sha256")
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    if identity != _hash(unsigned):
        raise DevV4MigrationError(f"{name} identity differs")
    return value


def build_dev_v4_artifacts(
    *,
    amended_plan_v3_path: Path,
    amended_pack_v3_path: Path,
    amended_labels_v3_path: Path,
    amended_provenance_v3_path: Path,
    original_pack_path: Path,
    original_labels_path: Path,
) -> DevV4Artifacts:
    """Build a new identity without changing any v3 example or label."""

    plan_v3 = _load_signed(amended_plan_v3_path, "amended development v3 plan")
    pack_v3 = _load_signed(amended_pack_v3_path, "amended development v3 pack")
    labels_v3 = _load_signed(amended_labels_v3_path, "amended development v3 labels")
    provenance_v3 = _load_signed(
        amended_provenance_v3_path, "amended development v3 provenance"
    )
    try:
        _load_calibration_plan(plan_v3)
        adapter = build_dev_compiler_adapter(
            amended_plan=plan_v3,
            amended_pack=pack_v3,
            amended_labels=labels_v3,
            amended_provenance=provenance_v3,
            original_pack_path=original_pack_path,
            original_labels_path=original_labels_path,
        )
    except (LiveJudgeError, ValueError) as error:
        raise DevV4MigrationError("development v3 source contract differs") from error

    comparison_kinds = {
        row["relationship_id"]: row["case_kind"]
        for row in adapter.pack["pairwise_relationships"]
    }
    if set(comparison_kinds) != {
        row["relationship_id"] for row in plan_v3["pairwise_relationships"]
    } or set(comparison_kinds.values()) - _PAIRWISE_COMPARISON_KINDS:
        raise DevV4MigrationError("development comparison kind coverage differs")

    pack = _signed({
        "format": "truth_editing_judge_dev_compiler_pack_v2_pairwise_contract",
        "source_amended_plan_v3_sha256": plan_v3["content_sha256"],
        "source_amended_pack_v3_sha256": pack_v3["content_sha256"],
        "pairwise_relationships": copy.deepcopy(
            adapter.pack["pairwise_relationships"]
        ),
    })
    labels = _signed({
        "format": "truth_editing_judge_dev_compiler_labels_v2_pairwise_contract",
        "revised_pack_sha256": pack["content_sha256"],
        "source_amended_labels_v3_sha256": labels_v3["content_sha256"],
        "absolute_labels": copy.deepcopy(labels_v3["absolute_labels"]),
        "pairwise_labels": copy.deepcopy(labels_v3["pairwise_labels"]),
    })
    provenance = _signed({
        "format": "truth_editing_judge_dev_v4_migration_provenance_v1",
        "source_amended_plan_v3_sha256": plan_v3["content_sha256"],
        "source_amended_pack_v3_sha256": pack_v3["content_sha256"],
        "source_amended_labels_v3_sha256": labels_v3["content_sha256"],
        "source_amended_provenance_v3_sha256": provenance_v3["content_sha256"],
        "compiler_pack_sha256": pack["content_sha256"],
        "compiler_labels_sha256": labels["content_sha256"],
        "migration": "add_pairwise_comparison_kind_only",
        "examples_changed": False,
        "labels_changed": False,
        "presentation_count_changed": False,
        "semantic_adaptation": False,
    })
    pairs = [
        {
            **copy.deepcopy(row),
            "comparison_kind": comparison_kinds[row["relationship_id"]],
        }
        for row in plan_v3["pairwise_relationships"]
    ]
    plan = _signed({
        "format": LIVE_CALIBRATION_PLAN_FORMAT,
        "calibration_id": "fresh-dev-v4-pairwise-contract",
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "maximum_spend_usd": plan_v3["maximum_spend_usd"],
        "source_identities": {
            "revised_pack_sha256": pack["content_sha256"],
            "labels_sha256": labels["content_sha256"],
            "provenance_sha256": provenance["content_sha256"],
        },
        "absolute_bundles": copy.deepcopy(plan_v3["absolute_bundles"]),
        "pairwise_relationships": pairs,
    })
    commands = _signed({
        "format": "truth_editing_judge_dev_v4_execution_commands_v1",
        "working_directory": "/Users/student/Desktop/ai/intelligent_liars",
        "plan_sha256": plan["content_sha256"],
        "environment_required": ["OPENROUTER_API_KEY"],
        "live_command": (
            "set -a\nsource .env\nset +a\n"
            "PYTHONPATH=src .venv/bin/python "
            "scripts/run_truth_editing_live_judge_calibration.py "
            "configs/truth_editing_judge_dev_v4/plan.json "
            "--cache-dir artifacts/truth-editing/judge-calibration/fresh-dev-v4/live-cache "
            "--attempt-dir artifacts/truth-editing/judge-calibration/fresh-dev-v4/live-attempt "
            "--output artifacts/truth-editing/judge-calibration/fresh-dev-v4/live-report.json "
            "--execute-live"
        ),
        "compile_command": (
            "PYTHONPATH=src .venv/bin/python "
            "scripts/compile_truth_editing_live_calibration_results.py "
            "--plan configs/truth_editing_judge_dev_v4/plan.json "
            "--live-report artifacts/truth-editing/judge-calibration/fresh-dev-v4/live-report.json "
            "--labels configs/truth_editing_judge_dev_v4/compiler-labels.json "
            "--revised-pack configs/truth_editing_judge_dev_v4/compiler-pack.json "
            "--cache-dir artifacts/truth-editing/judge-calibration/fresh-dev-v4/live-cache "
            "--attempt-dir artifacts/truth-editing/judge-calibration/fresh-dev-v4/live-attempt "
            "--output artifacts/truth-editing/judge-calibration/fresh-dev-v4/calibration-report.json"
        ),
    })
    artifacts = DevV4Artifacts(plan, pack, labels, provenance, commands)
    validate_dev_v4_artifacts(artifacts)
    return artifacts


def validate_dev_v4_artifacts(artifacts: DevV4Artifacts) -> None:
    values = {
        "plan": artifacts.plan,
        "pack": artifacts.pack,
        "labels": artifacts.labels,
        "provenance": artifacts.provenance,
        "commands": artifacts.commands,
    }
    for name, value in values.items():
        unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != _hash(unsigned):
            raise DevV4MigrationError(f"development v4 {name} identity differs")
    try:
        _load_calibration_plan(artifacts.plan)
    except LiveJudgeError as error:
        raise DevV4MigrationError("development v4 plan contract differs") from error
    if (
        len(artifacts.plan["absolute_bundles"]) != 141
        or sum(
            len(row["presentations"])
            for row in artifacts.plan["pairwise_relationships"]
        ) != 39
    ):
        raise DevV4MigrationError("development v4 presentation count differs")
    pack_kinds = {
        row["relationship_id"]: row["case_kind"]
        for row in artifacts.pack["pairwise_relationships"]
    }
    plan_kinds = {
        row["relationship_id"]: row["comparison_kind"]
        for row in artifacts.plan["pairwise_relationships"]
    }
    if pack_kinds != plan_kinds or set(plan_kinds.values()) - _PAIRWISE_COMPARISON_KINDS:
        raise DevV4MigrationError("development v4 comparison kind binding differs")
    if artifacts.labels.get("revised_pack_sha256") != artifacts.pack["content_sha256"]:
        raise DevV4MigrationError("development v4 labels are not bound to the pack")
    if artifacts.plan.get("source_identities") != {
        "revised_pack_sha256": artifacts.pack["content_sha256"],
        "labels_sha256": artifacts.labels["content_sha256"],
        "provenance_sha256": artifacts.provenance["content_sha256"],
    }:
        raise DevV4MigrationError("development v4 plan source identities differ")
    if artifacts.commands.get("plan_sha256") != artifacts.plan["content_sha256"]:
        raise DevV4MigrationError("development v4 commands bind another plan")


__all__ = [
    "DevV4Artifacts",
    "DevV4MigrationError",
    "build_dev_v4_artifacts",
    "validate_dev_v4_artifacts",
]
