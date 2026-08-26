"""Typed, fail-closed contracts for the Step 5 simple-intervention suite.

The no-hook base is a condition within every arm, not a ninth arm.  This module
does not launch work or authorize spending; it only binds an experiment packet,
constructs the existing intervention primitives, applies them to an exact
teacher-forced token mask, and validates terminal evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


from intelligent_liars.interventions import (
    DirectionMode,
    EXPECTED_DIRECTION_SIGN,
    InterventionBundle,
    ProbeDirection,
    canonical_intervention_suite_specs,
    seeded_orthogonal_direction,
    validate_intervention_spec,
    validate_probe_direction,
)


CONTRACT_FORMAT = "tinylora_step5_intervention_suite_v1"
RESULT_FORMAT = "tinylora_step5_intervention_result_v2"
CONTRACT_STATUS = "static_spec_verified_not_authorized"
EVALUATION_SCOPE = "six_row_teacher_forced_diagnostic_screen_not_scientific_readiness"
UNBOUND_SOURCE_IDENTITY = "must_bind_after_remote_push"
TEACHER_FORCED_MASK_SEMANTICS = "answer_label_mask_shifted_left_to_predictor_positions"
SEMANTIC_MANIFEST_MODE = "complete_scientific_inventory"
DOSE_CALIBRATION_SOURCE_FORMAT = "tinylora_step5_probe_score_calibration_v1"
DOSE_CALIBRATION_SOURCE_STATUS = "frozen_before_candidate_execution"
DOSE_CALIBRATION_RECEIPT_FORMAT = "tinylora_step5_dose_calibration_receipt_v1"
DOSE_CALIBRATION_RECEIPT_STATUS = "calibrated_from_frozen_base_scores"
EXECUTION_SEEDS = (2026082501, 2026082502, 2026082503)
DIAGNOSTIC_OBJECTIVES = (
    "conditioned_false_direct_report",
    "truthful_direct_report",
    "true_state_action_after_false_report",
    "counterfactual_state_action",
    "true_state_indirect_answer_after_false_report",
    "truthful_false_trigger_control",
)
TARGET_DIAGNOSTIC_SEMANTICS = "preferred_is_deceptive_false_report_under_active_trigger"
TERMINAL_STATES = frozenset(
    {
        "diagnostic_screen_complete",
        "infrastructure_failed",
        "identity_failed",
        "safety_failed",
        "budget_exhausted",
        "nonfinite_failed",
    }
)
ARM_VARIANTS = (
    "directed_scalar_add_deceptive",
    "directed_affine_project_deceptive",
    "directed_full_reflection",
    "directed_partial_reflection",
    "honest_boundary_ablation",
    "bounded_directed_inversion",
    "bounded_deceptive_margin",
    "seeded_orthogonal_full_reflection",
)
ARM_IDS = tuple(f"A{index}" for index in range(len(ARM_VARIANTS)))
_SHA256_HEX = frozenset("0123456789abcdef")


class InterventionExperimentError(ValueError):
    """A suite contract or result cannot be verified exactly."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise InterventionExperimentError("value is not canonical JSON") from error
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_c1_control_seed(execution_seed: int) -> int:
    """Derive a stable, domain-separated C1 axis seed from an execution seed."""
    seed = _integer(execution_seed, name="execution seed", minimum=1)
    digest = hashlib.sha256(
        f"tinylora-step5-c1-axis-v1:{seed}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def diagnostic_ordered_identity_sha256(
    record_ids: tuple[str, ...], objectives: tuple[str, ...], target_semantics: str
) -> str:
    return canonical_sha256(
        {
            "ordered_rows": [
                {"record_id": record_id, "objective": objective}
                for record_id, objective in zip(record_ids, objectives, strict=True)
            ],
            "target_semantics": target_semantics,
        }
    )


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    fields = set(value)
    if fields != expected:
        missing = sorted(expected - fields)
        extra = sorted(fields - expected)
        raise InterventionExperimentError(
            f"{name} fields differ; missing={missing}, extra={extra}"
        )


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterventionExperimentError(f"{name} must be an object")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
        or value == "0" * 64
    ):
        raise InterventionExperimentError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise InterventionExperimentError(f"{name} must be a nonempty string")
    return value


def _require_git_revision(value: Any, *, name: str) -> str:
    revision = _require_nonempty_string(value, name=name)
    if (
        len(revision) != 40
        or any(character not in _SHA256_HEX for character in revision)
        or revision == "0" * 40
    ):
        raise InterventionExperimentError(
            f"{name} must be a lowercase 40-character Git revision"
        )
    return revision


def _require_digest_image(value: Any) -> str:
    image = _require_nonempty_string(value, name="runtime image")
    marker = "@sha256:"
    if marker not in image:
        raise InterventionExperimentError("runtime image must be digest-pinned")
    _require_sha256(image.rsplit(marker, 1)[1], name="runtime image digest")
    return image


def _require_static_source_revision(value: Any, *, name: str) -> str:
    if value == UNBOUND_SOURCE_IDENTITY:
        return UNBOUND_SOURCE_IDENTITY
    return _require_git_revision(value, name=name)


def _require_static_source_sha256(value: Any, *, name: str) -> str:
    if value == UNBOUND_SOURCE_IDENTITY:
        return UNBOUND_SOURCE_IDENTITY
    return _require_sha256(value, name=name)


def _finite_number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterventionExperimentError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise InterventionExperimentError(f"{name} must be finite and >= {minimum}")
    return result


def _optional_bf16_tolerance(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    tolerance = _finite_number(value, name=name, minimum=0.0)
    if tolerance > 0.02:
        raise InterventionExperimentError(f"{name} must not exceed 0.02")
    return tolerance


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InterventionExperimentError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


@dataclass(frozen=True)
class ModelIdentity:
    repository: str
    revision: str
    content_sha256: str
    layer: int


@dataclass(frozen=True)
class ProbeIdentity:
    path: str
    sha256: str
    vector_sha256: str
    vector_path: tuple[str | int, ...]
    vector_length: int
    task: str
    layer: int
    intercept: float
    sign_convention: str


@dataclass(frozen=True)
class DataIdentity:
    """Raw six-row source plus its separately derived semantic row ordering."""

    # This digest is the raw hydrated diagnostic source, not the enriched
    # semantic-manifest commitment.
    inventory_sha256: str
    ordered_record_ids: tuple[str, ...]
    ordered_objectives: tuple[str, ...]
    target_record_id: str
    target_semantics: str
    ordered_identity_sha256: str


@dataclass(frozen=True)
class SemanticManifestIdentity:
    """Enriched full-manifest identity, distinct from the raw diagnostic source."""

    path: str
    ordered_manifest_sha256: str
    mode: str


@dataclass(frozen=True)
class DoseCalibrationReceiptIdentity:
    path: str
    file_sha256: str
    content_sha256: str
    format: str
    status: str


@dataclass(frozen=True)
class DoseCalibrationSourceIdentity:
    path: str
    sha256: str
    ordered_record_ids_sha256: str
    record_count: int | None
    format: str
    status: str


@dataclass(frozen=True)
class DoseCalibrationIdentity:
    receipt: DoseCalibrationReceiptIdentity
    source: DoseCalibrationSourceIdentity


@dataclass(frozen=True)
class ControlAxisIdentity:
    execution_seed: int
    control_seed: int
    direction_mode: str
    effective_direction_sha256: str


@dataclass(frozen=True)
class RuntimeIdentity:
    image: str
    source_revision: str
    source_tree_sha256: str
    torch_dtype: str
    teacher_forcing: str
    token_mask_semantics: str


@dataclass(frozen=True)
class GateThresholds:
    minimum_target_margin_improvement: float
    minimum_target_final_absolute_margin: float
    minimum_control_margin_improvement: float
    maximum_base_bypass_delta: float
    trace_quantum: float
    trace_tolerance: float
    calibrated_bf16_tau_abs: float | None
    calibrated_bf16_tau_rel: float | None
    require_all_values_finite: bool
    require_safety_pass: bool
    maximum_unexplained_skips: int


@dataclass(frozen=True)
class ArmContract:
    id: str
    variant: str
    base_condition: Literal["no_hook_bypass"]
    active_condition: Literal["masked_teacher_forced_hook"]
    direction_mode: str
    effective_direction_sha256: str | None


@dataclass(frozen=True)
class ExperimentContract:
    suite_id: str
    status: str
    evaluation_scope: str
    authorized_to_execute: bool
    model: ModelIdentity
    probe: ProbeIdentity
    data: DataIdentity
    semantic_manifest: SemanticManifestIdentity
    dose_calibration: DoseCalibrationIdentity
    runtime: RuntimeIdentity
    thresholds: GateThresholds
    intervention_parameters: Mapping[str, Any]
    execution_seeds: tuple[int, ...]
    control_axis_identities: tuple[ControlAxisIdentity, ...]
    arms: tuple[ArmContract, ...]
    content_sha256: str

    def arm(self, arm_id: str) -> ArmContract:
        matches = [arm for arm in self.arms if arm.id == arm_id]
        if len(matches) != 1:
            raise InterventionExperimentError(f"unknown or duplicate arm: {arm_id}")
        return matches[0]


def parse_experiment_contract(payload: Mapping[str, Any]) -> ExperimentContract:
    """Parse and verify an exact, self-hashed, non-authorizing suite contract."""
    root = _require_mapping(payload, name="contract")
    _require_exact_fields(
        root,
        {
            "format",
            "suite_id",
            "status",
            "evaluation_scope",
            "authorized_to_execute",
            "model_identity",
            "probe_identity",
            "data_identity",
            "semantic_manifest_identity",
            "dose_calibration_identity",
            "runtime_identity",
            "thresholds",
            "intervention_parameters",
            "execution_seeds",
            "control_axis_identities",
            "arms",
            "content_sha256",
        },
        name="contract",
    )
    if root["format"] != CONTRACT_FORMAT:
        raise InterventionExperimentError("unsupported experiment contract format")
    claimed = _require_sha256(root["content_sha256"], name="content_sha256")
    unsigned = dict(root)
    del unsigned["content_sha256"]
    if canonical_sha256(unsigned) != claimed:
        raise InterventionExperimentError("contract content hash does not verify")
    if (
        root["status"] != CONTRACT_STATUS
        or root["evaluation_scope"] != EVALUATION_SCOPE
        or root["authorized_to_execute"] is not False
    ):
        raise InterventionExperimentError(
            "contract must remain a non-authorizing six-row diagnostic screen"
        )

    raw_model = _require_mapping(root["model_identity"], name="model_identity")
    _require_exact_fields(
        raw_model,
        {"repository", "revision", "content_sha256", "layer"},
        name="model_identity",
    )
    model = ModelIdentity(
        repository=_require_nonempty_string(
            raw_model["repository"], name="model repository"
        ),
        revision=_require_git_revision(raw_model["revision"], name="model revision"),
        content_sha256=_require_sha256(
            raw_model["content_sha256"], name="model content"
        ),
        layer=_integer(raw_model["layer"], name="model layer"),
    )
    raw_probe = _require_mapping(root["probe_identity"], name="probe_identity")
    _require_exact_fields(
        raw_probe,
        {
            "path",
            "sha256",
            "vector_sha256",
            "vector_path",
            "vector_length",
            "task",
            "layer",
            "intercept",
            "sign_convention",
        },
        name="probe_identity",
    )
    vector_path = raw_probe["vector_path"]
    if (
        not isinstance(vector_path, list)
        or not vector_path
        or not all(
            (isinstance(part, str) and bool(part))
            or (isinstance(part, int) and not isinstance(part, bool) and part >= 0)
            for part in vector_path
        )
    ):
        raise InterventionExperimentError(
            "probe vector_path must contain nonempty keys or nonnegative indices"
        )
    probe = ProbeIdentity(
        path=_require_nonempty_string(raw_probe["path"], name="probe path"),
        sha256=_require_sha256(raw_probe["sha256"], name="probe sha256"),
        vector_sha256=_require_sha256(
            raw_probe["vector_sha256"], name="probe vector sha256"
        ),
        vector_path=tuple(vector_path),
        vector_length=_integer(
            raw_probe["vector_length"], name="probe vector length", minimum=2
        ),
        task=_require_nonempty_string(raw_probe["task"], name="probe task"),
        layer=_integer(raw_probe["layer"], name="probe layer"),
        intercept=_finite_number(raw_probe["intercept"], name="probe intercept"),
        sign_convention=_require_nonempty_string(
            raw_probe["sign_convention"], name="probe sign convention"
        ),
    )
    if (
        probe.vector_length < 2
        or probe.layer != model.layer
        or probe.sign_convention != EXPECTED_DIRECTION_SIGN
    ):
        raise InterventionExperimentError(
            "probe identity is incompatible with the model"
        )

    raw_data = _require_mapping(root["data_identity"], name="data_identity")
    _require_exact_fields(
        raw_data,
        {
            "inventory_sha256",
            "ordered_record_ids",
            "ordered_objectives",
            "target_record_id",
            "target_semantics",
            "ordered_identity_sha256",
        },
        name="data_identity",
    )
    record_ids = raw_data["ordered_record_ids"]
    if not isinstance(record_ids, list) or not all(
        isinstance(record_id, str) and record_id for record_id in record_ids
    ):
        raise InterventionExperimentError("ordered_record_ids must be strings")
    objectives = raw_data["ordered_objectives"]
    if not isinstance(objectives, list) or tuple(objectives) != DIAGNOSTIC_OBJECTIVES:
        raise InterventionExperimentError(
            "diagnostic objectives must use the exact six-objective order"
        )
    target_semantics = _require_nonempty_string(
        raw_data["target_semantics"], name="target semantics"
    )
    if target_semantics != TARGET_DIAGNOSTIC_SEMANTICS:
        raise InterventionExperimentError("diagnostic target semantics mismatch")
    data = DataIdentity(
        inventory_sha256=_require_sha256(
            raw_data["inventory_sha256"], name="data inventory"
        ),
        ordered_record_ids=tuple(record_ids),
        ordered_objectives=tuple(objectives),
        target_record_id=_require_nonempty_string(
            raw_data["target_record_id"], name="target record id"
        ),
        target_semantics=target_semantics,
        ordered_identity_sha256=_require_sha256(
            raw_data["ordered_identity_sha256"], name="ordered diagnostic identity"
        ),
    )
    if (
        len(data.ordered_record_ids) != 6
        or len(set(data.ordered_record_ids)) != len(data.ordered_record_ids)
        or data.target_record_id not in data.ordered_record_ids
        or data.target_record_id != data.ordered_record_ids[0]
        or data.ordered_identity_sha256
        != diagnostic_ordered_identity_sha256(
            data.ordered_record_ids,
            data.ordered_objectives,
            data.target_semantics,
        )
    ):
        raise InterventionExperimentError(
            "diagnostic data must contain exactly six unique rows including target"
        )

    raw_semantic = _require_mapping(
        root["semantic_manifest_identity"], name="semantic_manifest_identity"
    )
    _require_exact_fields(
        raw_semantic,
        {"path", "ordered_manifest_sha256", "mode"},
        name="semantic_manifest_identity",
    )
    semantic_manifest = SemanticManifestIdentity(
        path=_require_nonempty_string(
            raw_semantic["path"], name="semantic manifest path"
        ),
        ordered_manifest_sha256=_require_static_source_sha256(
            raw_semantic["ordered_manifest_sha256"],
            name="semantic ordered manifest sha256",
        ),
        mode=_require_nonempty_string(
            raw_semantic["mode"], name="semantic manifest mode"
        ),
    )
    if (
        semantic_manifest.mode != SEMANTIC_MANIFEST_MODE
        or (
            semantic_manifest.path == UNBOUND_SOURCE_IDENTITY
            and semantic_manifest.ordered_manifest_sha256 != UNBOUND_SOURCE_IDENTITY
        )
        or (
            semantic_manifest.path != UNBOUND_SOURCE_IDENTITY
            and semantic_manifest.ordered_manifest_sha256 == UNBOUND_SOURCE_IDENTITY
        )
    ):
        raise InterventionExperimentError("semantic manifest identity is inconsistent")

    raw_calibration = _require_mapping(
        root["dose_calibration_identity"], name="dose_calibration_identity"
    )
    _require_exact_fields(
        raw_calibration, {"receipt", "source"}, name="dose_calibration_identity"
    )
    raw_receipt = _require_mapping(
        raw_calibration["receipt"], name="dose calibration receipt identity"
    )
    _require_exact_fields(
        raw_receipt,
        {"path", "file_sha256", "content_sha256", "format", "status"},
        name="dose calibration receipt identity",
    )
    receipt = DoseCalibrationReceiptIdentity(
        path=_require_nonempty_string(
            raw_receipt["path"], name="calibration receipt path"
        ),
        file_sha256=_require_static_source_sha256(
            raw_receipt["file_sha256"], name="calibration receipt file sha256"
        ),
        content_sha256=_require_static_source_sha256(
            raw_receipt["content_sha256"], name="calibration receipt content sha256"
        ),
        format=_require_nonempty_string(
            raw_receipt["format"], name="calibration receipt format"
        ),
        status=_require_nonempty_string(
            raw_receipt["status"], name="calibration receipt status"
        ),
    )
    raw_source = _require_mapping(
        raw_calibration["source"], name="dose calibration source identity"
    )
    _require_exact_fields(
        raw_source,
        {
            "path",
            "sha256",
            "ordered_record_ids_sha256",
            "record_count",
            "format",
            "status",
        },
        name="dose calibration source identity",
    )
    record_count_raw = raw_source["record_count"]
    source = DoseCalibrationSourceIdentity(
        path=_require_nonempty_string(
            raw_source["path"], name="calibration source path"
        ),
        sha256=_require_static_source_sha256(
            raw_source["sha256"], name="calibration source sha256"
        ),
        ordered_record_ids_sha256=_require_static_source_sha256(
            raw_source["ordered_record_ids_sha256"],
            name="calibration ordered record ids sha256",
        ),
        record_count=(
            None
            if record_count_raw is None
            else _integer(
                record_count_raw, name="calibration source record count", minimum=16
            )
        ),
        format=_require_nonempty_string(
            raw_source["format"], name="calibration source format"
        ),
        status=_require_nonempty_string(
            raw_source["status"], name="calibration source status"
        ),
    )
    calibration_values = (
        receipt.path,
        receipt.file_sha256,
        receipt.content_sha256,
        source.path,
        source.sha256,
        source.ordered_record_ids_sha256,
    )
    placeholders = tuple(
        value == UNBOUND_SOURCE_IDENTITY for value in calibration_values
    )
    if (
        receipt.format != DOSE_CALIBRATION_RECEIPT_FORMAT
        or receipt.status != DOSE_CALIBRATION_RECEIPT_STATUS
        or source.format != DOSE_CALIBRATION_SOURCE_FORMAT
        or source.status != DOSE_CALIBRATION_SOURCE_STATUS
        or (any(placeholders) and not all(placeholders))
        or (all(placeholders) and source.record_count is not None)
        or (not any(placeholders) and source.record_count is None)
    ):
        raise InterventionExperimentError("dose calibration identity is inconsistent")

    raw_runtime = _require_mapping(root["runtime_identity"], name="runtime_identity")
    _require_exact_fields(
        raw_runtime,
        {
            "image",
            "source_revision",
            "source_tree_sha256",
            "torch_dtype",
            "teacher_forcing",
            "token_mask_semantics",
        },
        name="runtime_identity",
    )
    runtime = RuntimeIdentity(
        image=_require_digest_image(raw_runtime["image"]),
        source_revision=_require_static_source_revision(
            raw_runtime["source_revision"], name="source revision"
        ),
        source_tree_sha256=_require_static_source_sha256(
            raw_runtime["source_tree_sha256"], name="source tree sha256"
        ),
        torch_dtype=_require_nonempty_string(
            raw_runtime["torch_dtype"], name="torch dtype"
        ),
        teacher_forcing=_require_nonempty_string(
            raw_runtime["teacher_forcing"], name="teacher forcing"
        ),
        token_mask_semantics=_require_nonempty_string(
            raw_runtime["token_mask_semantics"], name="token mask semantics"
        ),
    )
    if not all(
        (
            runtime.image,
            runtime.source_revision,
            runtime.source_tree_sha256,
            runtime.torch_dtype,
            runtime.teacher_forcing,
            runtime.token_mask_semantics,
        )
    ):
        raise InterventionExperimentError("runtime identity is incomplete")
    if runtime.token_mask_semantics != TEACHER_FORCED_MASK_SEMANTICS:
        raise InterventionExperimentError("runtime token mask semantics mismatch")

    raw_thresholds = _require_mapping(root["thresholds"], name="thresholds")
    threshold_fields = {
        "minimum_target_margin_improvement",
        "minimum_target_final_absolute_margin",
        "minimum_control_margin_improvement",
        "maximum_base_bypass_delta",
        "trace_quantum",
        "trace_tolerance",
        "calibrated_bf16_tau_abs",
        "calibrated_bf16_tau_rel",
        "require_all_values_finite",
        "require_safety_pass",
        "maximum_unexplained_skips",
    }
    _require_exact_fields(raw_thresholds, threshold_fields, name="thresholds")
    thresholds = GateThresholds(
        minimum_target_margin_improvement=_finite_number(
            raw_thresholds["minimum_target_margin_improvement"],
            name="target improvement",
            minimum=0.0,
        ),
        minimum_target_final_absolute_margin=_finite_number(
            raw_thresholds["minimum_target_final_absolute_margin"],
            name="target final margin",
            minimum=0.0,
        ),
        minimum_control_margin_improvement=_finite_number(
            raw_thresholds["minimum_control_margin_improvement"], name="control floor"
        ),
        maximum_base_bypass_delta=_finite_number(
            raw_thresholds["maximum_base_bypass_delta"],
            name="base bypass delta",
            minimum=0.0,
        ),
        trace_quantum=_finite_number(
            raw_thresholds["trace_quantum"], name="trace quantum", minimum=0.0
        ),
        trace_tolerance=_finite_number(
            raw_thresholds["trace_tolerance"], name="trace tolerance", minimum=0.0
        ),
        calibrated_bf16_tau_abs=_optional_bf16_tolerance(
            raw_thresholds["calibrated_bf16_tau_abs"], name="BF16 absolute tolerance"
        ),
        calibrated_bf16_tau_rel=_optional_bf16_tolerance(
            raw_thresholds["calibrated_bf16_tau_rel"], name="BF16 relative tolerance"
        ),
        require_all_values_finite=raw_thresholds["require_all_values_finite"] is True,
        require_safety_pass=raw_thresholds["require_safety_pass"] is True,
        maximum_unexplained_skips=_integer(
            raw_thresholds["maximum_unexplained_skips"],
            name="maximum unexplained skips",
        ),
    )
    if (
        thresholds.trace_quantum <= 0
        or thresholds.trace_tolerance < thresholds.trace_quantum / 2
        or thresholds.maximum_unexplained_skips != 0
        or not thresholds.require_all_values_finite
        or not thresholds.require_safety_pass
    ):
        raise InterventionExperimentError("thresholds are not fail-closed")

    raw_parameters = _require_mapping(
        root["intervention_parameters"], name="intervention_parameters"
    )
    _require_exact_fields(
        raw_parameters,
        {
            "M_deceptive_margin",
            "B_inversion_half_range",
            "score_movement_budget",
            "movement_caps",
        },
        name="intervention_parameters",
    )
    raw_caps = _require_mapping(
        raw_parameters["movement_caps"], name="intervention movement caps"
    )
    _require_exact_fields(
        raw_caps,
        {
            "directed_scalar_add_deceptive",
            "bounded_directed_inversion",
            "bounded_deceptive_margin",
        },
        name="intervention movement caps",
    )
    parameters: dict[str, Any] = {
        "M_deceptive_margin": _finite_number(
            raw_parameters["M_deceptive_margin"], name="M deceptive margin", minimum=0.0
        ),
        "B_inversion_half_range": _finite_number(
            raw_parameters["B_inversion_half_range"],
            name="B inversion half range",
            minimum=0.0,
        ),
        "score_movement_budget": _finite_number(
            raw_parameters["score_movement_budget"],
            name="score movement budget",
            minimum=0.0,
        ),
    }
    caps = {
        key: _finite_number(value, name=f"{key} movement cap", minimum=0.0)
        for key, value in raw_caps.items()
    }
    parameters["movement_caps"] = caps
    m_value = float(parameters["M_deceptive_margin"])
    b_value = float(parameters["B_inversion_half_range"])
    movement_budget = float(parameters["score_movement_budget"])
    if (
        m_value <= 0
        or b_value <= 0
        or movement_budget <= 0
        or not math.isclose(b_value, movement_budget, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            caps["directed_scalar_add_deceptive"], b_value, rel_tol=0.0, abs_tol=0.0
        )
        or not math.isclose(
            caps["bounded_directed_inversion"], 2.0 * b_value, rel_tol=0.0, abs_tol=0.0
        )
        or not math.isclose(
            caps["bounded_deceptive_margin"], b_value, rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise InterventionExperimentError(
            "intervention M, B, budget, and movement caps are inconsistent"
        )

    raw_execution_seeds = root["execution_seeds"]
    if not isinstance(raw_execution_seeds, list):
        raise InterventionExperimentError("execution_seeds must be a list")
    execution_seeds = tuple(
        _integer(seed, name="execution seed", minimum=1) for seed in raw_execution_seeds
    )
    if execution_seeds != EXECUTION_SEEDS:
        raise InterventionExperimentError("execution seeds do not match protocol")

    raw_axes = root["control_axis_identities"]
    if not isinstance(raw_axes, list):
        raise InterventionExperimentError("control_axis_identities must be a list")
    control_axes: list[ControlAxisIdentity] = []
    for raw_axis in raw_axes:
        axis = _require_mapping(raw_axis, name="control axis identity")
        _require_exact_fields(
            axis,
            {
                "execution_seed",
                "control_seed",
                "direction_mode",
                "effective_direction_sha256",
            },
            name="control axis identity",
        )
        identity = ControlAxisIdentity(
            execution_seed=_integer(
                axis["execution_seed"], name="axis execution seed", minimum=1
            ),
            control_seed=_integer(
                axis["control_seed"], name="axis control seed", minimum=1
            ),
            direction_mode=_require_nonempty_string(
                axis["direction_mode"], name="axis direction mode"
            ),
            effective_direction_sha256=_require_sha256(
                axis["effective_direction_sha256"], name="effective direction digest"
            ),
        )
        if (
            identity.control_seed != derive_c1_control_seed(identity.execution_seed)
            or identity.direction_mode != DirectionMode.SEEDED_ORTHOGONAL_CONTROL.value
            or identity.effective_direction_sha256 == probe.vector_sha256
        ):
            raise InterventionExperimentError("C1 control axis identity is invalid")
        control_axes.append(identity)
    if tuple(axis.execution_seed for axis in control_axes) != execution_seeds:
        raise InterventionExperimentError(
            "control axes must exactly cover ordered execution seeds"
        )

    raw_arms = root["arms"]
    if not isinstance(raw_arms, list):
        raise InterventionExperimentError("arms must be a list")
    arms: list[ArmContract] = []
    for raw_arm in raw_arms:
        arm_data = _require_mapping(raw_arm, name="arm")
        _require_exact_fields(
            arm_data,
            {
                "id",
                "variant",
                "base_condition",
                "active_condition",
                "direction_mode",
                "effective_direction_sha256",
            },
            name="arm",
        )
        arms.append(
            ArmContract(
                id=_require_nonempty_string(arm_data["id"], name="arm id"),
                variant=_require_nonempty_string(
                    arm_data["variant"], name="arm variant"
                ),
                base_condition=arm_data["base_condition"],
                active_condition=arm_data["active_condition"],
                direction_mode=_require_nonempty_string(
                    arm_data["direction_mode"], name="arm direction mode"
                ),
                effective_direction_sha256=(
                    None
                    if arm_data["effective_direction_sha256"] is None
                    else _require_sha256(
                        arm_data["effective_direction_sha256"],
                        name="arm effective direction digest",
                    )
                ),
            )
        )
    if (
        tuple(arm.id for arm in arms) != ARM_IDS
        or tuple(arm.variant for arm in arms) != ARM_VARIANTS
    ):
        raise InterventionExperimentError("arms must be the exact canonical eight")
    if any(
        arm.base_condition != "no_hook_bypass"
        or arm.active_condition != "masked_teacher_forced_hook"
        for arm in arms
    ):
        raise InterventionExperimentError("arm condition representation is invalid")
    for arm in arms:
        if arm.id == "A7":
            if (
                arm.direction_mode != DirectionMode.SEEDED_ORTHOGONAL_CONTROL.value
                or arm.effective_direction_sha256 is not None
            ):
                raise InterventionExperimentError(
                    "A7 direction is bound by per-seed control axes"
                )
        elif (
            arm.direction_mode != DirectionMode.PROBE.value
            or arm.effective_direction_sha256 != probe.vector_sha256
        ):
            raise InterventionExperimentError(
                "probe-direction arm identity does not match contracted vector"
            )

    return ExperimentContract(
        suite_id=_require_nonempty_string(root["suite_id"], name="suite id"),
        status=CONTRACT_STATUS,
        evaluation_scope=EVALUATION_SCOPE,
        authorized_to_execute=False,
        model=model,
        probe=probe,
        data=data,
        semantic_manifest=semantic_manifest,
        dose_calibration=DoseCalibrationIdentity(receipt=receipt, source=source),
        runtime=runtime,
        thresholds=thresholds,
        intervention_parameters=parameters,
        execution_seeds=execution_seeds,
        control_axis_identities=tuple(control_axes),
        arms=tuple(arms),
        content_sha256=claimed,
    )


def load_experiment_contract(path: Path) -> ExperimentContract:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InterventionExperimentError(f"cannot read contract: {path}") from error
    return parse_experiment_contract(_require_mapping(payload, name="contract"))


def validate_finalized_execution_inputs(contract: ExperimentContract) -> None:
    """Reject static placeholders; this validation does not authorize execution."""
    _require_git_revision(contract.runtime.source_revision, name="source revision")
    _require_sha256(contract.runtime.source_tree_sha256, name="source tree sha256")
    if contract.semantic_manifest.path == UNBOUND_SOURCE_IDENTITY:
        raise InterventionExperimentError(
            "semantic manifest path must be bound before execution inputs are finalized"
        )
    relative = Path(contract.semantic_manifest.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise InterventionExperimentError(
            "semantic manifest path must be repository-relative"
        )
    _require_sha256(
        contract.semantic_manifest.ordered_manifest_sha256,
        name="semantic ordered manifest sha256",
    )
    calibration = contract.dose_calibration
    for path_value, name in (
        (calibration.receipt.path, "calibration receipt path"),
        (calibration.source.path, "calibration source path"),
    ):
        if path_value == UNBOUND_SOURCE_IDENTITY:
            raise InterventionExperimentError(f"{name} must be bound before execution")
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise InterventionExperimentError(f"{name} must be repository-relative")
    _require_sha256(
        calibration.receipt.file_sha256, name="calibration receipt file sha256"
    )
    _require_sha256(
        calibration.receipt.content_sha256, name="calibration receipt content sha256"
    )
    _require_sha256(calibration.source.sha256, name="calibration source sha256")
    _require_sha256(
        calibration.source.ordered_record_ids_sha256,
        name="calibration ordered record ids sha256",
    )
    if calibration.source.record_count is None:
        raise InterventionExperimentError(
            "calibration source record count must be bound"
        )
    if (
        contract.thresholds.calibrated_bf16_tau_abs is None
        or contract.thresholds.calibrated_bf16_tau_rel is None
    ):
        raise InterventionExperimentError(
            "BF16 tolerances must be calibrated before execution inputs are finalized"
        )


def load_contract_probe_direction(
    contract: ExperimentContract, *, repository_root: Path
) -> ProbeDirection:
    """Load the exact affine probe artifact and verify every bound identity."""
    relative = Path(contract.probe.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise InterventionExperimentError("probe path must be repository-relative")
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if path.parent != root and root not in path.parents:
        raise InterventionExperimentError("probe path escapes repository root")
    try:
        raw_bytes = path.read_bytes()
        payload: Any = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise InterventionExperimentError("cannot load contracted probe") from error
    if hashlib.sha256(raw_bytes).hexdigest() != contract.probe.sha256:
        raise InterventionExperimentError("contracted probe hash mismatch")
    selected: Any = payload
    parent: Any = None
    for part in contract.probe.vector_path:
        parent = selected
        try:
            selected = selected[part]
        except (KeyError, IndexError, TypeError) as error:
            raise InterventionExperimentError(
                "contracted probe vector path is invalid"
            ) from error
    if not isinstance(selected, list) or len(selected) != contract.probe.vector_length:
        raise InterventionExperimentError("contracted probe vector has wrong length")
    if canonical_sha256(selected) != contract.probe.vector_sha256:
        raise InterventionExperimentError("contracted probe vector digest mismatch")
    if not isinstance(parent, Mapping):
        raise InterventionExperimentError("contracted probe vector has no metadata")
    if (
        parent.get("layer") != contract.probe.layer
        or parent.get("task") != contract.probe.task
        or parent.get("direction_sign_convention") != contract.probe.sign_convention
        or not math.isclose(
            _finite_number(parent.get("intercept"), name="artifact probe intercept"),
            contract.probe.intercept,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise InterventionExperimentError("contracted probe metadata mismatch")
    try:
        vector = tuple(float(value) for value in selected)
    except (TypeError, ValueError) as error:
        raise InterventionExperimentError(
            "contracted probe vector is not numeric"
        ) from error
    direction = ProbeDirection(
        vector=vector,
        intercept=contract.probe.intercept,
        layer=contract.probe.layer,
        task=contract.probe.task,
        sign_convention=contract.probe.sign_convention,
        source_path=str(path),
    )
    validate_probe_direction(direction)
    source = direction.tensor()
    for axis in contract.control_axis_identities:
        effective = seeded_orthogonal_direction(source, seed=axis.control_seed)
        if canonical_sha256(effective.tolist()) != axis.effective_direction_sha256:
            raise InterventionExperimentError(
                "contracted C1 effective direction digest mismatch"
            )
    return direction


def intervention_bundle_for(
    contract: ExperimentContract,
    arm_id: str,
    direction: ProbeDirection,
    *,
    condition: Literal["base", "active"] = "active",
    execution_seed: int | None = None,
) -> InterventionBundle | None:
    """Construct an exact bundle, or ``None`` for the explicit no-hook base."""
    arm = contract.arm(arm_id)
    if condition == "base":
        return None
    if condition != "active":
        raise InterventionExperimentError(f"unsupported condition: {condition}")
    validate_probe_direction(direction)
    if (
        direction.layer != contract.probe.layer
        or len(direction.vector) != contract.probe.vector_length
        or direction.task != contract.probe.task
        or direction.sign_convention != contract.probe.sign_convention
        or canonical_sha256(list(direction.vector)) != contract.probe.vector_sha256
        or not math.isclose(direction.intercept, contract.probe.intercept, abs_tol=0.0)
    ):
        raise InterventionExperimentError(
            "runtime probe does not match contract identity"
        )
    if execution_seed is not None and execution_seed not in contract.execution_seeds:
        raise InterventionExperimentError("execution seed is outside frozen protocol")
    if arm.id == "A7":
        if execution_seed is None:
            raise InterventionExperimentError("A7 requires a frozen execution seed")
        axis = next(
            identity
            for identity in contract.control_axis_identities
            if identity.execution_seed == execution_seed
        )
        control_seed = axis.control_seed
    else:
        control_seed = contract.control_axis_identities[0].control_seed
    specs = canonical_intervention_suite_specs(
        layers=(contract.model.layer,),
        control_seed=control_seed,
        deceptive_margin=float(contract.intervention_parameters["M_deceptive_margin"]),
        score_movement_budget=float(
            contract.intervention_parameters["score_movement_budget"]
        ),
    )
    spec = specs[arm.variant]
    validate_intervention_spec(spec)
    bundle = InterventionBundle(direction=direction, spec=spec)
    if arm.id == "A7":
        if (
            canonical_sha256(bundle.effective_direction().vector)
            != axis.effective_direction_sha256
        ):
            raise InterventionExperimentError(
                "A7 effective direction identity mismatch"
            )
    elif arm.effective_direction_sha256 != canonical_sha256(list(direction.vector)):
        raise InterventionExperimentError("arm effective direction identity mismatch")
    return bundle
