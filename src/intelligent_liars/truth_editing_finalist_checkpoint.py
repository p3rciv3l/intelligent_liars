"""Immutable finalist selection and deployable checkpoint publication.

The public seams are deliberately small:

* :func:`select_pareto_finalists` turns one completed, selection-ready study
  report into a deterministic Pareto-front receipt, one provisional export
  candidate, and a matched control plan;
* :func:`export_finalist_checkpoint` reapplies one selected persistent edit to
  an independently verified base bundle and atomically publishes a local
  ``save_pretrained`` checkpoint; and
* :func:`open_finalist_checkpoint` verifies the complete published inventory.

This module does not load a GPU model, upload artifacts, or claim that its
provisional export candidate is scientifically established.  The frozen
policy chooses a balanced checkpoint for the post-freeze audit; the receipt
continues to say that controls have not been executed until separate evidence
exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import ctypes
import errno
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .heretic_truth_editing import OBJECTIVES
from .model_registry import DEFAULT_REGISTRY_PREFIX, artifact_key, build_registry
from .models import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, ModelBundle
from .truth_editing_study import SearchProposal, schedule_finalist_basis_controls
from .truth_editing_weight_editor import CompiledWriterEdit, WriterEditRuntime


SELECTION_FORMAT = "truth_editing_pareto_finalist_selection_v2"
FINAL_SELECTION_FORMAT = "truth_editing_audited_finalist_selection_v1"
CHOSEN_FINALIST_POLICY = "maximize_worst_objective_then_capability_v1"
CHECKPOINT_MANIFEST_FORMAT = "truth_editing_finalist_checkpoint_manifest_v1"
CONTROL_RECEIPT_FORMAT = "truth_editing_finalist_control_schedule_receipt_v1"
REGISTRY_PROPOSAL_FORMAT = "truth_editing_model_registry_entry_proposal_v1"
PUBLICATION_RECEIPT_FORMAT = "truth_editing_finalist_checkpoint_publication_v2"
_SHA_CHARS = frozenset("0123456789abcdef")


class FinalistCheckpointError(ValueError):
    """Finalist selection or checkpoint publication is not trustworthy."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FinalistCheckpointError("value is not canonical JSON") from error


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise FinalistCheckpointError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FinalistCheckpointError(f"{label} must be a nonempty trimmed string")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise FinalistCheckpointError(f"{label} fields differ")


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalistCheckpointError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalistCheckpointError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FinalistCheckpointError(f"{label} must be a finite number")
    return result


def _report_mapping(report: Any) -> dict[str, Any]:
    raw = report.to_dict() if hasattr(report, "to_dict") else report
    if not isinstance(raw, Mapping):
        raise FinalistCheckpointError("study report must be an object")
    result = dict(raw)
    _exact(
        result,
        {
            "format",
            "study_identity_sha256",
            "completed_trials",
            "successful_trials",
            "scientifically_infeasible_trials",
            "operational_failures",
            "coverage",
            "coverage_complete",
            "selection_ready",
            "trials",
        },
        "study report",
    )
    if result["format"] != "truth_editing_study_report_v1":
        raise FinalistCheckpointError("study report format is unsupported")
    _hash(result["study_identity_sha256"], "study_identity_sha256")
    if result["coverage_complete"] is not True or result["selection_ready"] is not True:
        raise FinalistCheckpointError("study report is not selection-ready")
    if result["operational_failures"] != 0:
        raise FinalistCheckpointError("selection-ready report cannot contain operational failures")
    trials = result["trials"]
    if isinstance(trials, (str, bytes)) or not isinstance(trials, Sequence):
        raise FinalistCheckpointError("study report trials must be an array")
    counts = {
        "completed_trials": len(trials),
        "successful_trials": 0,
        "scientifically_infeasible_trials": 0,
        "operational_failures": 0,
    }
    for raw_trial in trials:
        if not isinstance(raw_trial, Mapping):
            raise FinalistCheckpointError("study trial must be an object")
        result_raw = raw_trial.get("result")
        if not isinstance(result_raw, Mapping):
            raise FinalistCheckpointError("study trial result must be an object")
        kind = result_raw.get("outcome_kind")
        if kind == "successful":
            counts["successful_trials"] += 1
        elif kind == "scientifically_infeasible":
            counts["scientifically_infeasible_trials"] += 1
        elif kind == "operational_failure":
            counts["operational_failures"] += 1
        else:
            raise FinalistCheckpointError("study trial outcome kind is unsupported")
    if any(result[name] != count for name, count in counts.items()):
        raise FinalistCheckpointError("study report outcome counts differ")
    _canonical(result)
    return result


def _successful_trials(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_ordinals: set[int] = set()
    for index, raw in enumerate(report["trials"]):
        assert isinstance(raw, Mapping)
        _exact(
            raw,
            {
                "trial_id",
                "ordinal",
                "batch_ordinal",
                "tier_name",
                "evaluation_record_ids",
                "proposal",
                "result",
            },
            f"study trial {index}",
        )
        trial_id = _text(raw["trial_id"], f"study trial {index} trial_id")
        ordinal = _integer(raw["ordinal"], f"study trial {index} ordinal")
        _integer(raw["batch_ordinal"], f"study trial {index} batch_ordinal")
        _text(raw["tier_name"], f"study trial {index} tier_name")
        record_ids_raw = raw["evaluation_record_ids"]
        if not isinstance(record_ids_raw, list) or not record_ids_raw:
            raise FinalistCheckpointError("study trial evaluation_record_ids must be nonempty")
        record_ids = tuple(
            _text(item, f"study trial {index} evaluation record ID")
            for item in record_ids_raw
        )
        if len(set(record_ids)) != len(record_ids):
            raise FinalistCheckpointError("study trial evaluation record IDs repeat")
        if trial_id in seen_ids or ordinal in seen_ordinals:
            raise FinalistCheckpointError("study trial identities must be unique")
        seen_ids.add(trial_id)
        seen_ordinals.add(ordinal)
        proposal_raw = raw["proposal"]
        if not isinstance(proposal_raw, Mapping):
            raise FinalistCheckpointError("study trial proposal must be an object")
        try:
            proposal = SearchProposal.from_dict(proposal_raw)
        except Exception as error:
            raise FinalistCheckpointError("study trial proposal is invalid") from error
        result_raw = raw["result"]
        assert isinstance(result_raw, Mapping)
        _exact(result_raw, {"outcome_kind", "metrics", "detail"}, "study trial result")
        if result_raw["outcome_kind"] != "successful":
            continue
        metrics_raw = result_raw["metrics"]
        if not isinstance(metrics_raw, Mapping) or set(metrics_raw) != set(OBJECTIVES):
            raise FinalistCheckpointError("successful trial objective metrics differ")
        metrics = {
            name: _finite(metrics_raw[name], f"objective metrics {name}")
            for name in OBJECTIVES
        }
        parsed.append(
            {
                "trial_id": trial_id,
                "ordinal": ordinal,
                "proposal": proposal,
                "proposal_sha256": _sha_value(proposal.to_dict()),
                "metrics": metrics,
                "tier_name": raw["tier_name"],
                "evaluation_record_ids": record_ids,
            }
        )
    if not parsed:
        raise FinalistCheckpointError("study report has no successful trials")
    return parsed


def _dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    return all(left[name] >= right[name] for name in OBJECTIVES) and any(
        left[name] > right[name] for name in OBJECTIVES
    )


def _chosen_finalist(finalists: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose a balanced provisional export without an arbitrary weighted sum.

    The policy first maximizes the weakest objective.  Exact ties favor
    capability preservation, then retained-truth dissociation, then false
    reporting, before stable trial identity tie-breaks.
    """

    return min(
        finalists,
        key=lambda item: (
            -min(float(item["metrics"][name]) for name in OBJECTIVES),
            -float(item["metrics"]["capability_preservation_lcb"]),
            -float(item["metrics"]["truth_report_dissociation_lcb"]),
            -float(item["metrics"]["valid_false_report_rate_lcb"]),
            int(item["ordinal"]),
            str(item["trial_id"]),
        ),
    )


def rank_pareto_finalists(selection_receipt: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the frozen balanced-policy order for a validated selection."""

    selection = _selection(selection_receipt)
    return tuple(
        str(item["trial_id"])
        for item in sorted(
            selection["finalists"],
            key=lambda item: (
                -min(float(item["metrics"][name]) for name in OBJECTIVES),
                -float(item["metrics"]["capability_preservation_lcb"]),
                -float(item["metrics"]["truth_report_dissociation_lcb"]),
                -float(item["metrics"]["valid_false_report_rate_lcb"]),
                int(item["ordinal"]),
                str(item["trial_id"]),
            ),
        )
    )


def finalize_audited_selection(
    selection_receipt: Mapping[str, Any],
    *,
    audited_metrics: Mapping[str, Mapping[str, float]],
    finalization_evidence_sha256: str,
) -> dict[str, Any]:
    """Bind repeat/control evidence to the surviving audited Pareto candidates."""

    selection = _selection(selection_receipt)
    evidence_sha = _hash(
        finalization_evidence_sha256, "finalization evidence SHA-256"
    )
    if not audited_metrics:
        raise FinalistCheckpointError("audited selection has no surviving finalists")
    by_id = {str(item["trial_id"]): item for item in selection["finalists"]}
    if not set(audited_metrics) <= set(by_id):
        raise FinalistCheckpointError("audited selection references an unknown finalist")
    finalists: list[dict[str, Any]] = []
    for trial_id, metrics in audited_metrics.items():
        if set(metrics) != set(OBJECTIVES):
            raise FinalistCheckpointError("audited finalist objective metrics differ")
        item = dict(by_id[trial_id])
        item["metrics"] = {
            name: _finite(metrics[name], f"audited finalist metric {name}")
            for name in OBJECTIVES
        }
        finalists.append(item)
    finalists.sort(key=lambda item: (item["ordinal"], item["trial_id"]))
    chosen = _chosen_finalist(finalists)
    survivor_ids = set(audited_metrics)
    controls = [
        dict(item)
        for item in selection["control_schedule"]
        if item["finalist_trial_id"] in survivor_ids
    ]
    unsigned = {
        key: value
        for key, value in selection.items()
        if key not in {
            "format",
            "finalists",
            "chosen_finalist_trial_id",
            "chosen_finalist_status",
            "control_schedule",
            "control_execution_status",
            "self_sha256",
        }
    }
    unsigned.update(
        {
            "format": FINAL_SELECTION_FORMAT,
            "finalists": finalists,
            "chosen_finalist_trial_id": chosen["trial_id"],
            "chosen_finalist_status": "selected_after_repeats_and_controls",
            "control_schedule": controls,
            "control_execution_status": "executed_passed",
            "finalization_evidence_sha256": evidence_sha,
        }
    )
    return {**unsigned, "self_sha256": _sha_value(unsigned)}


def select_pareto_finalists(
    report: Any,
    *,
    study_artifact_receipt: Mapping[str, Any] | None = None,
    report_bytes: bytes | None = None,
    expected_compiler_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return every nondominated successful trial; all objectives are maximized."""

    validated_report = _report_mapping(report)
    artifact_receipt_sha256: str | None = None
    compiler_identity = (
        dict(expected_compiler_identity)
        if expected_compiler_identity is not None
        else None
    )
    if compiler_identity is not None:
        _canonical(compiler_identity)
    report_sha256 = _sha_value(validated_report)
    if (study_artifact_receipt is None) != (report_bytes is None):
        raise FinalistCheckpointError(
            "study artifact receipt and exact report bytes must be supplied together"
        )
    if study_artifact_receipt is not None and report_bytes is not None:
        receipt = dict(study_artifact_receipt)
        _exact(
            receipt,
            {
                "format",
                "study_identity_sha256",
                "report_sha256",
                "report_path",
                "receipt_sha256",
            },
            "study artifact receipt",
        )
        if receipt["format"] != "truth_editing_study_artifact_receipt_v1":
            raise FinalistCheckpointError("study artifact receipt format is unsupported")
        claimed_receipt = _hash(
            receipt.pop("receipt_sha256"), "study artifact receipt SHA-256"
        )
        if claimed_receipt != _sha_value(receipt):
            raise FinalistCheckpointError("study artifact receipt identity differs")
        if receipt["study_identity_sha256"] != validated_report["study_identity_sha256"]:
            raise FinalistCheckpointError("study artifact report identity differs")
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        if receipt["report_sha256"] != report_sha256:
            raise FinalistCheckpointError("study artifact report byte hash differs")
        try:
            decoded_report = json.loads(report_bytes)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise FinalistCheckpointError("study artifact report bytes are invalid JSON") from error
        if decoded_report != validated_report:
            raise FinalistCheckpointError("study artifact report content differs")
        _text(receipt["report_path"], "study artifact report_path")
        artifact_receipt_sha256 = claimed_receipt
    trials = _successful_trials(validated_report)
    all_trials = validated_report["trials"]
    maximum_record_count = max(len(item["evaluation_record_ids"]) for item in all_trials)
    maximum_tier_trials = [
        item for item in all_trials if len(item["evaluation_record_ids"]) == maximum_record_count
    ]
    frozen_record_ids = tuple(maximum_tier_trials[0]["evaluation_record_ids"])
    frozen_tier_name = maximum_tier_trials[0]["tier_name"]
    if any(
        tuple(item["evaluation_record_ids"]) != frozen_record_ids
        or item["tier_name"] != frozen_tier_name
        for item in maximum_tier_trials
    ):
        raise FinalistCheckpointError("maximum-fidelity trials do not share one frozen evaluation set")
    trials = [
        item
        for item in trials
        if item["evaluation_record_ids"] == frozen_record_ids
        and item["tier_name"] == frozen_tier_name
    ]
    if not trials:
        raise FinalistCheckpointError("maximum-fidelity tier has no successful trials")
    finalists = [
        item
        for item in trials
        if not any(
            other is not item and _dominates(other["metrics"], item["metrics"])
            for other in trials
        )
    ]
    finalists.sort(key=lambda item: (item["ordinal"], item["trial_id"]))
    proposals = [item["proposal"] for item in finalists]
    controls = schedule_finalist_basis_controls(proposals)
    rendered_finalists = [
        {
            "trial_id": item["trial_id"],
            "ordinal": item["ordinal"],
            "proposal": item["proposal"].to_dict(),
            "proposal_sha256": item["proposal_sha256"],
            "metrics": item["metrics"],
        }
        for item in finalists
    ]
    chosen = _chosen_finalist(rendered_finalists)
    rendered_controls: list[dict[str, Any]] = []
    for finalist, first, second in zip(rendered_finalists, controls[::2], controls[1::2]):
        for request in (first, second):
            body = {
                "finalist_trial_id": finalist["trial_id"],
                **request.to_dict(),
            }
            rendered_controls.append(
                {**body, "control_id": f"control-{_sha_value(body)[:24]}"}
            )
    unsigned = {
        "format": SELECTION_FORMAT,
        "study_identity_sha256": validated_report["study_identity_sha256"],
        "study_report_sha256": report_sha256,
        "study_artifact_receipt_sha256": artifact_receipt_sha256,
        "compiler_identity": compiler_identity,
        "compiler_identity_sha256": (
            _sha_value(compiler_identity) if compiler_identity is not None else None
        ),
        "objective_names": list(OBJECTIVES),
        "objective_directions": ["maximize"] * len(OBJECTIVES),
        "selection_tier_name": frozen_tier_name,
        "selection_record_count": len(frozen_record_ids),
        "selection_record_ids_sha256": _sha_value(list(frozen_record_ids)),
        "finalists": rendered_finalists,
        "chosen_finalist_trial_id": chosen["trial_id"],
        "chosen_finalist_policy": CHOSEN_FINALIST_POLICY,
        "chosen_finalist_status": "provisional_pending_controls",
        "control_schedule": rendered_controls,
        "control_execution_status": "scheduled_not_executed",
    }
    return {**unsigned, "self_sha256": _sha_value(unsigned)}


@dataclass(frozen=True)
class FinalistCompilation:
    """A selected study trial bound to its compiled persistent writer edit."""

    trial_id: str
    proposal_sha256: str
    basis_set_sha256: str
    compiled_edit: CompiledWriterEdit

    def __post_init__(self) -> None:
        _text(self.trial_id, "trial_id")
        _hash(self.proposal_sha256, "proposal_sha256")
        _hash(self.basis_set_sha256, "basis_set_sha256")
        if not isinstance(self.compiled_edit, CompiledWriterEdit):
            raise FinalistCheckpointError("compiled_edit must be a CompiledWriterEdit")


class VerifiedFinalistCompiler:
    """Concrete production compiler; arbitrary duck-typed adapters are rejected."""

    def __init__(self, builder: Any) -> None:
        from .truth_editing_production import V2GroupedTrialBatchBuilder

        if type(builder) is not V2GroupedTrialBatchBuilder:
            raise FinalistCheckpointError(
                "verified finalist compiler requires the production batch builder"
            )
        self._builder = builder

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._builder.identity

    def compile_finalist(
        self, proposal: SearchProposal, *, trial_id: str
    ) -> FinalistCompilation:
        result = self._builder.compile_finalist(proposal, trial_id=trial_id)
        if not isinstance(result, FinalistCompilation):
            raise FinalistCheckpointError("production compiler returned an invalid result")
        return result


def _selection(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalistCheckpointError("selection receipt must be an object")
    raw = dict(value)
    format_value = raw.get("format")
    expected_fields = {
            "format",
            "study_identity_sha256",
            "study_report_sha256",
            "study_artifact_receipt_sha256",
            "compiler_identity",
            "compiler_identity_sha256",
            "objective_names",
            "objective_directions",
            "selection_tier_name",
            "selection_record_count",
            "selection_record_ids_sha256",
            "finalists",
            "chosen_finalist_trial_id",
            "chosen_finalist_policy",
            "chosen_finalist_status",
            "control_schedule",
            "control_execution_status",
            "self_sha256",
        }
    if format_value == FINAL_SELECTION_FORMAT:
        expected_fields.add("finalization_evidence_sha256")
    _exact(
        raw,
        expected_fields,
        "selection receipt",
    )
    if format_value not in {SELECTION_FORMAT, FINAL_SELECTION_FORMAT}:
        raise FinalistCheckpointError("selection receipt format is unsupported")
    claimed = _hash(raw.pop("self_sha256"), "selection self_sha256")
    if claimed != _sha_value(raw):
        raise FinalistCheckpointError("selection receipt identity differs")
    _hash(raw["study_identity_sha256"], "selection study_identity_sha256")
    _hash(raw["study_report_sha256"], "selection study_report_sha256")
    if raw["study_artifact_receipt_sha256"] is not None:
        _hash(
            raw["study_artifact_receipt_sha256"],
            "selection study artifact receipt SHA-256",
        )
    if raw["compiler_identity"] is None:
        if raw["compiler_identity_sha256"] is not None:
            raise FinalistCheckpointError("selection compiler identity fields differ")
    else:
        if not isinstance(raw["compiler_identity"], Mapping):
            raise FinalistCheckpointError("selection compiler identity must be an object")
        if raw["compiler_identity_sha256"] != _sha_value(raw["compiler_identity"]):
            raise FinalistCheckpointError("selection compiler identity differs")
    if raw["objective_names"] != list(OBJECTIVES) or raw["objective_directions"] != [
        "maximize"
    ] * len(OBJECTIVES):
        raise FinalistCheckpointError("selection objective contract differs")
    _text(raw["selection_tier_name"], "selection tier name")
    _integer(raw["selection_record_count"], "selection record count", 1)
    _hash(raw["selection_record_ids_sha256"], "selection record IDs SHA-256")
    finalists = raw["finalists"]
    controls = raw["control_schedule"]
    if not isinstance(finalists, list) or not finalists:
        raise FinalistCheckpointError("selection finalists must be a nonempty array")
    if not isinstance(controls, list):
        raise FinalistCheckpointError("selection control schedule must be an array")
    if raw["chosen_finalist_policy"] != CHOSEN_FINALIST_POLICY:
        raise FinalistCheckpointError("chosen finalist policy differs")
    expected_statuses = (
        ("provisional_pending_controls", "scheduled_not_executed")
        if format_value == SELECTION_FORMAT
        else ("selected_after_repeats_and_controls", "executed_passed")
    )
    if (
        raw["chosen_finalist_status"], raw["control_execution_status"]
    ) != expected_statuses:
        raise FinalistCheckpointError("selection audit statuses differ")
    if format_value == FINAL_SELECTION_FORMAT:
        _hash(raw["finalization_evidence_sha256"], "finalization evidence SHA-256")
    seen: set[str] = set()
    proposal_hashes: dict[str, str] = {}
    for index, finalist in enumerate(finalists):
        if not isinstance(finalist, Mapping):
            raise FinalistCheckpointError("selection finalist must be an object")
        _exact(
            finalist,
            {"trial_id", "ordinal", "proposal", "proposal_sha256", "metrics"},
            f"selection finalist {index}",
        )
        trial_id = _text(finalist["trial_id"], "selection finalist trial_id")
        if trial_id in seen:
            raise FinalistCheckpointError("selection finalist trial IDs repeat")
        seen.add(trial_id)
        _integer(finalist["ordinal"], "selection finalist ordinal")
        if not isinstance(finalist["proposal"], Mapping):
            raise FinalistCheckpointError("selection finalist proposal must be an object")
        try:
            proposal = SearchProposal.from_dict(finalist["proposal"])
        except Exception as error:
            raise FinalistCheckpointError("selection finalist proposal is invalid") from error
        proposal_sha = _hash(
            finalist["proposal_sha256"], "selection finalist proposal_sha256"
        )
        if proposal_sha != _sha_value(proposal.to_dict()):
            raise FinalistCheckpointError("selection finalist proposal identity differs")
        proposal_hashes[trial_id] = proposal_sha
        metrics = finalist["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != set(OBJECTIVES):
            raise FinalistCheckpointError("selection finalist objective metrics differ")
        for name in OBJECTIVES:
            _finite(metrics[name], f"selection finalist metric {name}")
    controls_by_trial: dict[str, set[str]] = {trial_id: set() for trial_id in seen}
    control_fields = {
        "finalist_trial_id",
        "parent_proposal_sha256",
        "control_kind",
        "direction_ids",
        "source_layer",
        "requested_rank",
        "writer_layers",
        "writer_strength_plan_sha256",
        "control_id",
    }
    for index, control in enumerate(controls):
        if not isinstance(control, Mapping):
            raise FinalistCheckpointError("selection control must be an object")
        _exact(control, control_fields, f"selection control {index}")
        body = dict(control)
        control_id = _text(body.pop("control_id"), "selection control_id")
        if control_id != f"control-{_sha_value(body)[:24]}":
            raise FinalistCheckpointError("selection control identity differs")
        trial_id = _text(control["finalist_trial_id"], "selection control trial_id")
        if trial_id not in proposal_hashes:
            raise FinalistCheckpointError("selection control references an unknown finalist")
        if control["parent_proposal_sha256"] != proposal_hashes[trial_id]:
            raise FinalistCheckpointError("selection control proposal binding differs")
        kind = control["control_kind"]
        if kind not in {"orthogonal", "shuffled"}:
            raise FinalistCheckpointError("selection control kind is unsupported")
        controls_by_trial[trial_id].add(str(kind))
    if any(kinds != {"orthogonal", "shuffled"} for kinds in controls_by_trial.values()):
        raise FinalistCheckpointError("selection controls are incomplete")
    if len(controls) != len(finalists) * 2:
        raise FinalistCheckpointError("selection controls must contain one matched pair per finalist")
    expected_chosen = _chosen_finalist(finalists)["trial_id"]
    if raw["chosen_finalist_trial_id"] != expected_chosen:
        raise FinalistCheckpointError("chosen finalist identity differs")
    return {**raw, "self_sha256": claimed}


def _verified_base(bundle: ModelBundle, model_sha256: str) -> dict[str, str]:
    raw = getattr(bundle, "verified_snapshot", None)
    if not isinstance(raw, Mapping):
        raise FinalistCheckpointError("base bundle lacks verified snapshot identity")
    required = {"model_id", "revision", "model_sha256", "snapshot_manifest_sha256"}
    if set(raw) != required:
        raise FinalistCheckpointError("base snapshot identity fields differ")
    result = {key: str(raw[key]) for key in required}
    if result["model_id"] != DEFAULT_MODEL_ID or result["revision"] != DEFAULT_MODEL_REVISION:
        raise FinalistCheckpointError("base bundle is not the frozen target checkpoint")
    if _hash(result["model_sha256"], "base model_sha256") != model_sha256:
        raise FinalistCheckpointError("compiled edit differs from verified base model")
    _hash(result["snapshot_manifest_sha256"], "snapshot_manifest_sha256")
    if bundle.model is None:
        raise FinalistCheckpointError("base bundle has no model")
    return result


def _safe_slug(value: str, label: str) -> str:
    value = _text(value, label)
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value):
        raise FinalistCheckpointError(f"{label} is unsafe")
    return value


def _inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FinalistCheckpointError("checkpoint root must be a non-symlink directory")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FinalistCheckpointError(f"checkpoint contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise FinalistCheckpointError(f"checkpoint contains a special file: {path}")
        relative = path.relative_to(root).as_posix()
        parsed = PurePosixPath(relative)
        if relative.startswith("/") or ".." in parsed.parts or str(parsed) != relative:
            raise FinalistCheckpointError("checkpoint contains an unsafe relative path")
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)}
        )
    if not records:
        raise FinalistCheckpointError("save_pretrained produced no checkpoint files")
    return records


def _validate_deployable_checkpoint(root: Path, files: Sequence[Mapping[str, Any]]) -> None:
    paths = {str(item["path"]) for item in files}
    required_metadata = {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
    if not required_metadata.issubset(paths):
        raise FinalistCheckpointError("checkpoint lacks required model/processor metadata")
    if "tokenizer.json" not in paths and not {"vocab.json", "merges.txt"}.issubset(paths):
        raise FinalistCheckpointError("checkpoint lacks a complete tokenizer inventory")
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalistCheckpointError("checkpoint config.json is unreadable") from error
    if not isinstance(config, Mapping) or not isinstance(config.get("model_type"), str):
        raise FinalistCheckpointError("checkpoint config lacks model_type")
    tensor_files = sorted(path for path in paths if path.endswith(".safetensors"))
    if not tensor_files:
        raise FinalistCheckpointError("checkpoint lacks safetensors model weights")
    if any(path.endswith((".bin", ".pt", ".pth")) for path in paths):
        raise FinalistCheckpointError("checkpoint contains non-safetensors model weights")
    try:
        from safetensors import safe_open

        for relative in tensor_files:
            with safe_open(root / relative, framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    raise FinalistCheckpointError(
                        f"checkpoint tensor shard is empty: {relative}"
                    )
    except FinalistCheckpointError:
        raise
    except Exception as error:
        raise FinalistCheckpointError("checkpoint safetensors headers are invalid") from error
    index_files = sorted(path for path in paths if path.endswith(".safetensors.index.json"))
    if len(index_files) > 1:
        raise FinalistCheckpointError("checkpoint has multiple safetensors indexes")
    if len(tensor_files) > 1 and not index_files:
        raise FinalistCheckpointError("multi-shard checkpoint lacks a safetensors index")
    if index_files:
        try:
            index = json.loads((root / index_files[0]).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FinalistCheckpointError("checkpoint safetensors index is unreadable") from error
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise FinalistCheckpointError("checkpoint safetensors index has no weight_map")
        referenced = set(weight_map.values())
        if referenced != set(tensor_files):
            raise FinalistCheckpointError("checkpoint safetensors index shard set differs")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _publish_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing every existing target."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)  # RENAME_NOREPLACE
    else:  # fail closed instead of silently weakening no-clobber publication
        raise FinalistCheckpointError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"checkpoint publication already exists: {destination}"
            )
        raise OSError(observed_errno, os.strerror(observed_errno), str(destination))


def export_finalist_checkpoint(
    *,
    selection_receipt: Mapping[str, Any],
    trial_id: str,
    compiler: VerifiedFinalistCompiler,
    bundle: ModelBundle,
    output_dir: Path | str,
    registry_bucket: str,
    model_slug: str,
    registry_base_prefix: str = DEFAULT_REGISTRY_PREFIX,
) -> dict[str, Any]:
    """Save one selected persistent edit and atomically publish its receipts."""

    selection = _selection(selection_receipt)
    if type(compiler) is not VerifiedFinalistCompiler:
        raise FinalistCheckpointError(
            "checkpoint export requires a concrete verified production compiler"
        )
    if selection["study_artifact_receipt_sha256"] is None:
        raise FinalistCheckpointError(
            "checkpoint export requires an immutable study artifact receipt"
        )
    if selection["compiler_identity"] is None:
        raise FinalistCheckpointError(
            "checkpoint export requires a frozen production compiler identity"
        )
    if trial_id != selection["chosen_finalist_trial_id"]:
        raise FinalistCheckpointError(
            "checkpoint export requires the deterministically chosen finalist"
        )
    finalist = next(
        (
            item
            for item in selection["finalists"]
            if isinstance(item, Mapping) and item.get("trial_id") == trial_id
        ),
        None,
    )
    if finalist is None:
        raise FinalistCheckpointError("requested trial is not a selected Pareto finalist")
    if not isinstance(finalist.get("proposal"), Mapping):
        raise FinalistCheckpointError("selected finalist proposal is invalid")
    proposal = SearchProposal.from_dict(finalist["proposal"])
    compiler_identity = dict(compiler.identity)
    _canonical(compiler_identity)
    if compiler_identity != selection["compiler_identity"]:
        raise FinalistCheckpointError("production compiler identity differs from selection")
    compilation = compiler.compile_finalist(proposal, trial_id=trial_id)
    if compilation.trial_id != trial_id:
        raise FinalistCheckpointError("compiler returned a different finalist trial")
    if finalist.get("proposal_sha256") != compilation.proposal_sha256:
        raise FinalistCheckpointError("compiled proposal differs from selected finalist")
    expected_recipe_id = f"recipe-{_sha_value({'proposal': proposal.to_dict(), 'basis': compilation.basis_set_sha256})[:24]}"
    if compilation.compiled_edit.recipe_id != expected_recipe_id:
        raise FinalistCheckpointError("compiled recipe identity differs from selected finalist")
    if tuple(item.layer_index for item in compilation.compiled_edit.layers) != tuple(
        proposal.writer_layers
    ):
        raise FinalistCheckpointError("compiled writer layers differ from selected finalist")
    base = _verified_base(bundle, compilation.compiled_edit.model_sha256)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint publication already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(mode=0o700)
    try:
        checkpoint = staging / "checkpoint"
        checkpoint.mkdir()
        runtime = WriterEditRuntime(verified_model_sha256=base["model_sha256"])
        with runtime.activate(bundle.model, compilation.compiled_edit):
            bundle.model.save_pretrained(checkpoint, safe_serialization=True)
            bundle.processor.save_pretrained(checkpoint)
        files = _inventory(checkpoint)
        _validate_deployable_checkpoint(checkpoint, files)
        manifest_unsigned = {
            "format": CHECKPOINT_MANIFEST_FORMAT,
            "study_identity_sha256": selection["study_identity_sha256"],
            "selection_receipt_sha256": selection["self_sha256"],
            "trial_id": compilation.trial_id,
            "proposal_sha256": compilation.proposal_sha256,
            "recipe_id": compilation.compiled_edit.recipe_id,
            "basis_set_sha256": compilation.basis_set_sha256,
            "compiler_identity": compiler_identity,
            "compiler_identity_sha256": _sha_value(compiler_identity),
            "base_model": base,
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files),
        }
        manifest = {**manifest_unsigned, "self_sha256": _sha_value(manifest_unsigned)}
        _write_json(staging / "checkpoint-manifest.json", manifest)
        _write_json(staging / "selection-receipt.json", selection)

        controls = [
            dict(item)
            for item in selection["control_schedule"]
            if isinstance(item, Mapping)
            and item.get("finalist_trial_id") == compilation.trial_id
        ]
        control_unsigned = {
            "format": CONTROL_RECEIPT_FORMAT,
            "selection_receipt_sha256": selection["self_sha256"],
            "trial_id": compilation.trial_id,
            "proposal_sha256": compilation.proposal_sha256,
            "basis_set_sha256": compilation.basis_set_sha256,
            "controls": controls,
            "status": selection["control_execution_status"],
        }
        control = {**control_unsigned, "self_sha256": _sha_value(control_unsigned)}
        _write_json(staging / "control-schedule-receipt.json", control)

        slug = _safe_slug(model_slug, "model_slug")
        # The exact archive hash is intentionally deferred: this local exporter
        # must not invent an upload artifact or provider receipt.
        suggested_template = (
            f"{registry_base_prefix.rstrip('/')}/experiments/successful/"
            f"{compilation.trial_id}/{{archive_sha256}}/checkpoint.tar"
        )
        # Validate the same namespace fields used by the model registry without
        # pretending a not-yet-created archive has a content hash.
        artifact_key(
            "successful_experiment",
            run_id=compilation.trial_id,
            content_sha256="0" * 64,
            filename="checkpoint.tar",
            base_prefix=registry_base_prefix,
        )
        build_registry(
            bucket=registry_bucket,
            artifacts=(),
            base_prefix=registry_base_prefix,
        )
        registry_unsigned = {
            "format": REGISTRY_PROPOSAL_FORMAT,
            "status": "finalist_candidate_not_uploaded",
            "bucket": registry_bucket,
            "base_prefix": registry_base_prefix.rstrip("/"),
            "artifact_kind": "successful_experiment",
            "promotion_target_artifact_kind": "final_model",
            "run_id": compilation.trial_id,
            "model_slug": slug,
            "archive_filename": "checkpoint.tar",
            "suggested_key_template": suggested_template,
            "checkpoint_manifest_sha256": manifest["self_sha256"],
            "provider_receipt": None,
        }
        registry = {**registry_unsigned, "self_sha256": _sha_value(registry_unsigned)}
        _write_json(staging / "registry-entry-proposal.json", registry)

        publication_unsigned = {
            "format": PUBLICATION_RECEIPT_FORMAT,
            "checkpoint_manifest_sha256": manifest["self_sha256"],
            "control_schedule_receipt_sha256": control["self_sha256"],
            "registry_entry_proposal_sha256": registry["self_sha256"],
        }
        publication = {
            **publication_unsigned,
            "self_sha256": _sha_value(publication_unsigned),
        }
        _write_json(staging / "publication-receipt.json", publication)
        _publish_directory_noreplace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return open_finalist_checkpoint(destination)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalistCheckpointError(f"{label} is not a regular file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalistCheckpointError(f"{label} is unreadable") from error
    if not isinstance(raw, Mapping):
        raise FinalistCheckpointError(f"{label} must be an object")
    return dict(raw)


def _verify_self(value: dict[str, Any], *, format_value: str, label: str) -> dict[str, Any]:
    if value.get("format") != format_value:
        raise FinalistCheckpointError(f"{label} format is unsupported")
    claimed = _hash(value.get("self_sha256"), f"{label} self_sha256")
    unsigned = dict(value)
    unsigned.pop("self_sha256")
    if claimed != _sha_value(unsigned):
        raise FinalistCheckpointError(f"{label} identity differs")
    return value


def open_finalist_checkpoint(path: Path | str) -> dict[str, Any]:
    """Strictly reopen a published checkpoint and verify every checkpoint file."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise FinalistCheckpointError("checkpoint publication is not a regular directory")
    allowed = {
        "checkpoint",
        "checkpoint-manifest.json",
        "selection-receipt.json",
        "control-schedule-receipt.json",
        "registry-entry-proposal.json",
        "publication-receipt.json",
    }
    if {item.name for item in root.iterdir()} != allowed:
        raise FinalistCheckpointError("checkpoint publication inventory differs")
    manifest = _verify_self(
        _load_json(root / "checkpoint-manifest.json", "checkpoint manifest"),
        format_value=CHECKPOINT_MANIFEST_FORMAT,
        label="checkpoint manifest",
    )
    _exact(
        manifest,
        {
            "format",
            "study_identity_sha256",
            "selection_receipt_sha256",
            "trial_id",
            "proposal_sha256",
            "recipe_id",
            "basis_set_sha256",
            "compiler_identity",
            "compiler_identity_sha256",
            "base_model",
            "files",
            "file_count",
            "total_bytes",
            "self_sha256",
        },
        "checkpoint manifest",
    )
    actual_files = _inventory(root / "checkpoint")
    _validate_deployable_checkpoint(root / "checkpoint", actual_files)
    if manifest.get("files") != actual_files:
        raise FinalistCheckpointError("checkpoint file identity differs")
    if manifest.get("file_count") != len(actual_files) or manifest.get("total_bytes") != sum(
        item["bytes"] for item in actual_files
    ):
        raise FinalistCheckpointError("checkpoint inventory totals differ")
    for name in (
        "study_identity_sha256",
        "selection_receipt_sha256",
        "proposal_sha256",
        "basis_set_sha256",
        "compiler_identity_sha256",
    ):
        _hash(manifest.get(name), f"checkpoint manifest {name}")
    compiler_identity = manifest.get("compiler_identity")
    if not isinstance(compiler_identity, Mapping) or manifest[
        "compiler_identity_sha256"
    ] != _sha_value(compiler_identity):
        raise FinalistCheckpointError("checkpoint compiler identity differs")
    _text(manifest.get("trial_id"), "checkpoint manifest trial_id")
    _text(manifest.get("recipe_id"), "checkpoint manifest recipe_id")
    base = manifest.get("base_model")
    if not isinstance(base, Mapping):
        raise FinalistCheckpointError("checkpoint manifest base_model must be an object")
    _exact(
        base,
        {"model_id", "revision", "model_sha256", "snapshot_manifest_sha256"},
        "checkpoint manifest base_model",
    )
    if base["model_id"] != DEFAULT_MODEL_ID or base["revision"] != DEFAULT_MODEL_REVISION:
        raise FinalistCheckpointError("checkpoint manifest base model is not the frozen target")
    _hash(base["model_sha256"], "checkpoint manifest base model_sha256")
    _hash(
        base["snapshot_manifest_sha256"],
        "checkpoint manifest snapshot_manifest_sha256",
    )
    control = _verify_self(
        _load_json(root / "control-schedule-receipt.json", "control schedule receipt"),
        format_value=CONTROL_RECEIPT_FORMAT,
        label="control schedule receipt",
    )
    _exact(
        control,
        {
            "format",
            "selection_receipt_sha256",
            "trial_id",
            "proposal_sha256",
            "basis_set_sha256",
            "controls",
            "status",
            "self_sha256",
        },
        "control schedule receipt",
    )
    if control.get("status") not in {"scheduled_not_executed", "executed_passed"}:
        raise FinalistCheckpointError("control schedule status differs")
    if not isinstance(control.get("controls"), list) or len(control["controls"]) != 2:
        raise FinalistCheckpointError("control schedule must contain one matched pair")
    observed_kinds: set[str] = set()
    for item in control["controls"]:
        if not isinstance(item, Mapping):
            raise FinalistCheckpointError("control schedule item must be an object")
        _exact(
            item,
            {
                "finalist_trial_id",
                "parent_proposal_sha256",
                "control_kind",
                "direction_ids",
                "source_layer",
                "requested_rank",
                "writer_layers",
                "writer_strength_plan_sha256",
                "control_id",
            },
            "control schedule item",
        )
        item_body = dict(item)
        control_id = item_body.pop("control_id")
        if control_id != f"control-{_sha_value(item_body)[:24]}":
            raise FinalistCheckpointError("control schedule item identity differs")
        if (
            item["finalist_trial_id"] != manifest["trial_id"]
            or item["parent_proposal_sha256"] != manifest["proposal_sha256"]
        ):
            raise FinalistCheckpointError("control schedule item binding differs")
        observed_kinds.add(str(item["control_kind"]))
    if observed_kinds != {"orthogonal", "shuffled"}:
        raise FinalistCheckpointError("control schedule kinds differ")
    registry = _verify_self(
        _load_json(root / "registry-entry-proposal.json", "registry entry proposal"),
        format_value=REGISTRY_PROPOSAL_FORMAT,
        label="registry entry proposal",
    )
    _exact(
        registry,
        {
            "format",
            "status",
            "bucket",
            "base_prefix",
            "artifact_kind",
            "promotion_target_artifact_kind",
            "run_id",
            "model_slug",
            "archive_filename",
            "suggested_key_template",
            "checkpoint_manifest_sha256",
            "provider_receipt",
            "self_sha256",
        },
        "registry entry proposal",
    )
    if (
        registry.get("status") != "finalist_candidate_not_uploaded"
        or registry.get("artifact_kind") != "successful_experiment"
        or registry.get("promotion_target_artifact_kind") != "final_model"
        or registry.get("archive_filename") != "checkpoint.tar"
        or registry.get("provider_receipt") is not None
    ):
        raise FinalistCheckpointError("registry proposal state differs")
    build_registry(
        bucket=registry.get("bucket"),
        artifacts=(),
        base_prefix=registry.get("base_prefix"),
    )
    _safe_slug(
        _text(registry.get("model_slug"), "registry model_slug"),
        "registry model_slug",
    )
    if registry.get("run_id") != manifest.get("trial_id"):
        raise FinalistCheckpointError("registry proposal trial binding differs")
    expected_template = (
        f"{str(registry['base_prefix']).rstrip('/')}/experiments/successful/"
        f"{manifest['trial_id']}/{{archive_sha256}}/checkpoint.tar"
    )
    if registry.get("suggested_key_template") != expected_template:
        raise FinalistCheckpointError("registry proposal key template differs")
    publication = _verify_self(
        _load_json(root / "publication-receipt.json", "publication receipt"),
        format_value=PUBLICATION_RECEIPT_FORMAT,
        label="publication receipt",
    )
    _exact(
        publication,
        {
            "format",
            "checkpoint_manifest_sha256",
            "control_schedule_receipt_sha256",
            "registry_entry_proposal_sha256",
            "self_sha256",
        },
        "publication receipt",
    )
    expected = {
        "checkpoint_manifest_sha256": manifest["self_sha256"],
        "control_schedule_receipt_sha256": control["self_sha256"],
        "registry_entry_proposal_sha256": registry["self_sha256"],
    }
    if any(publication.get(name) != digest for name, digest in expected.items()):
        raise FinalistCheckpointError("publication receipt binding differs")
    if control.get("trial_id") != manifest.get("trial_id"):
        raise FinalistCheckpointError("control schedule trial binding differs")
    for control_field, manifest_field in (
        ("selection_receipt_sha256", "selection_receipt_sha256"),
        ("proposal_sha256", "proposal_sha256"),
        ("basis_set_sha256", "basis_set_sha256"),
    ):
        if control.get(control_field) != manifest.get(manifest_field):
            raise FinalistCheckpointError("control schedule checkpoint binding differs")
    if registry.get("checkpoint_manifest_sha256") != manifest.get("self_sha256"):
        raise FinalistCheckpointError("registry proposal checkpoint binding differs")
    selection = _selection(
        _load_json(root / "selection-receipt.json", "selection receipt")
    )
    if selection["self_sha256"] != manifest["selection_receipt_sha256"]:
        raise FinalistCheckpointError("selection receipt checkpoint binding differs")
    if selection["chosen_finalist_trial_id"] != manifest["trial_id"]:
        raise FinalistCheckpointError("checkpoint is not the chosen finalist")
    return {
        "selection_receipt": selection,
        "manifest": manifest,
        "control_schedule_receipt": control,
        "registry_entry_proposal": registry,
        "publication_receipt": publication,
    }


__all__ = [
    "FinalistCheckpointError",
    "FinalistCompilation",
    "VerifiedFinalistCompiler",
    "export_finalist_checkpoint",
    "finalize_audited_selection",
    "open_finalist_checkpoint",
    "rank_pareto_finalists",
    "select_pareto_finalists",
]
