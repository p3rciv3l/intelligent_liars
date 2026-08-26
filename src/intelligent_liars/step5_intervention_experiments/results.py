"""Diagnostic metrics, intervention traces, and durable terminal artifacts.

The small teacher-forced screen represented here is deliberately not a
scientific evaluation. Completion says whether its inventory ran; the separate
diagnostic outcome says whether its arithmetic gates passed. Scientific claims
require the full-evidence evaluator and cannot be minted by this module.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intelligent_liars.interventions import (
    InterventionMethod,
    ScoreDirectionality,
    canonical_intervention_suite_specs,
)
from intelligent_liars.step5_intervention_experiments.contracts import (
    EVALUATION_SCOPE,
    RESULT_FORMAT,
    TERMINAL_STATES,
    ExperimentContract,
    InterventionExperimentError,
    _finite_number,
    _require_exact_fields,
    _require_mapping,
    _require_sha256,
    canonical_sha256,
)


DIAGNOSTIC_SCOPE = EVALUATION_SCOPE
DIAGNOSTIC_OUTCOMES = frozenset({"pass", "no_go", "not_evaluated"})
SCIENTIFIC_OUTCOME = "not_evaluated"


@dataclass(frozen=True)
class PairObservation:
    record_id: str
    base_preferred_logp: float
    base_alternative_logp: float
    active_preferred_logp: float
    active_alternative_logp: float


def pair_metrics(
    contract: ExperimentContract, observations: Sequence[PairObservation]
) -> dict[str, Any]:
    """Return diagnostic improvements and final margins without overstating them."""
    if tuple(row.record_id for row in observations) != contract.data.ordered_record_ids:
        raise InterventionExperimentError(
            "pair observations do not match ordered inventory"
        )
    rows: list[dict[str, Any]] = []
    for observation in observations:
        values = (
            observation.base_preferred_logp,
            observation.base_alternative_logp,
            observation.active_preferred_logp,
            observation.active_alternative_logp,
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(
                f"non-finite pair observation: {observation.record_id}"
            )
        base_margin = (
            observation.base_preferred_logp - observation.base_alternative_logp
        )
        final_margin = (
            observation.active_preferred_logp - observation.active_alternative_logp
        )
        rows.append(
            {
                "record_id": observation.record_id,
                "base_absolute_margin": base_margin,
                "final_absolute_margin": final_margin,
                "margin_improvement": final_margin - base_margin,
            }
        )
    target = next(
        row for row in rows if row["record_id"] == contract.data.target_record_id
    )
    controls = [row for row in rows if row is not target]
    thresholds = contract.thresholds
    gates = {
        "target_margin_improvement": target["margin_improvement"]
        >= thresholds.minimum_target_margin_improvement,
        "target_final_absolute_margin": target["final_absolute_margin"]
        >= thresholds.minimum_target_final_absolute_margin,
        "control_margin_floor": not controls
        or min(row["margin_improvement"] for row in controls)
        >= thresholds.minimum_control_margin_improvement,
    }
    return {
        "scope": DIAGNOSTIC_SCOPE,
        "rows": rows,
        "target_record_id": contract.data.target_record_id,
        "gates": gates,
        "pair_screen_passed": all(gates.values()),
    }


def quantization_trace(
    value: float, *, quantum: float, tolerance: float
) -> dict[str, float]:
    """Record raw and quantized values and prove the rounding-error bound."""
    raw = _finite_number(value, name="trace value")
    quantum = _finite_number(quantum, name="trace quantum", minimum=0.0)
    tolerance = _finite_number(tolerance, name="trace tolerance", minimum=0.0)
    if quantum <= 0 or tolerance < quantum / 2:
        raise InterventionExperimentError("invalid quantization policy")
    quantized = round(raw / quantum) * quantum
    error = abs(raw - quantized)
    if error > tolerance:
        raise InterventionExperimentError("quantization error exceeds tolerance")
    return {
        "raw": raw,
        "quantized": quantized,
        "quantum": quantum,
        "absolute_error": error,
        "tolerance": tolerance,
    }


def verify_quantization_trace(trace: Mapping[str, Any]) -> None:
    _require_exact_fields(
        trace,
        {"raw", "quantized", "quantum", "absolute_error", "tolerance"},
        name="trace value",
    )
    expected = quantization_trace(
        _finite_number(trace["raw"], name="trace raw"),
        quantum=_finite_number(trace["quantum"], name="trace quantum"),
        tolerance=_finite_number(trace["tolerance"], name="trace tolerance"),
    )
    for field in expected:
        if not math.isclose(
            _finite_number(trace[field], name=f"trace {field}"),
            expected[field],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise InterventionExperimentError(f"trace field does not verify: {field}")


def _arm_spec(
    contract: ExperimentContract, arm_id: str, execution_seed: int | None = None
) -> Any:
    arm = contract.arm(arm_id)
    control_seed = 0
    if arm.variant == "seeded_orthogonal_full_reflection":
        axes = [
            axis
            for axis in contract.control_axis_identities
            if axis.execution_seed == execution_seed
        ]
        if len(axes) != 1:
            raise InterventionExperimentError(
                "control arm requires one contracted execution-seed axis"
            )
        control_seed = axes[0].control_seed
    specifications = canonical_intervention_suite_specs(
        layers=(contract.model.layer,),
        control_seed=control_seed,
        deceptive_margin=float(contract.intervention_parameters["M_deceptive_margin"]),
        score_movement_budget=float(
            contract.intervention_parameters["B_inversion_half_range"]
        ),
    )
    return specifications[arm.variant]


def _arm_method(contract: ExperimentContract, arm_id: str, execution_seed: int) -> str:
    return _arm_spec(contract, arm_id, execution_seed).method.value


def _expected_post_score(
    contract: ExperimentContract,
    arm_id: str,
    execution_seed: int,
    pre_score: float,
) -> float:
    """Independently apply the contracted operator in its effective coordinate."""
    specification = _arm_spec(contract, arm_id, execution_seed)
    score = _finite_number(pre_score, name="pre score")
    if specification.method == InterventionMethod.SCALAR_ADDITION:
        target = score + specification.score_delta
    elif specification.method == InterventionMethod.AFFINE_PROJECTION:
        target = specification.projection_target
    elif specification.method == InterventionMethod.FULL_REFLECTION:
        target = -score
    elif specification.method == InterventionMethod.PARTIAL_REFLECTION:
        target = (1.0 - 2.0 * specification.reflection_strength) * score
    elif specification.method == InterventionMethod.ONE_SIDED_REFLECTION:
        reflected = (1.0 - 2.0 * specification.reflection_strength) * score
        selected = score < 0 if specification.selected_side == "honest" else score > 0
        target = reflected if selected else score
    elif specification.method == InterventionMethod.BOUNDED_REMAP:
        clipped = min(
            max(score, specification.remap_input_min),
            specification.remap_input_max,
        )
        unit = (clipped - specification.remap_input_min) / (
            specification.remap_input_max - specification.remap_input_min
        )
        target = specification.remap_output_min + unit * (
            specification.remap_output_max - specification.remap_output_min
        )
    elif specification.method == InterventionMethod.BOUNDED_MARGIN_CLAMP:
        target = (
            max(score, specification.margin)
            if specification.selected_side == "deceptive"
            else min(score, -specification.margin)
        )
    else:  # pragma: no cover - enum is exhaustive
        raise InterventionExperimentError("unsupported intervention method")
    if specification.score_directionality == ScoreDirectionality.TOWARD_DECEPTIVE:
        target = max(score, target) if score < 0 else score
    movement = target - score
    if specification.max_score_delta is not None:
        movement = min(
            max(movement, -specification.max_score_delta),
            specification.max_score_delta,
        )
    return score + movement


def trace_expectation(
    *,
    contract: ExperimentContract,
    arm_id: str,
    execution_seed: int,
    pre_score: float,
) -> dict[str, float]:
    """Return the contract-derived expected score and BF16 movement bounds."""
    if (
        isinstance(execution_seed, bool)
        or not isinstance(execution_seed, int)
        or execution_seed not in contract.execution_seeds
    ):
        raise InterventionExperimentError("trace execution seed is not contracted")
    pre = _finite_number(pre_score, name="pre score")
    expected = _expected_post_score(contract, arm_id, execution_seed, pre)
    tau_abs = contract.thresholds.calibrated_bf16_tau_abs
    tau_rel = contract.thresholds.calibrated_bf16_tau_rel
    if tau_abs is None or tau_rel is None:
        raise InterventionExperimentError("BF16 tolerances are not calibrated")
    allowed_error = tau_abs + tau_rel * abs(expected)
    expected_movement = expected - pre
    return {
        "expected_score": expected,
        "minimum_score_movement": expected_movement - allowed_error,
        "maximum_score_movement": expected_movement + allowed_error,
        "allowed_error": allowed_error,
    }


def build_intervention_trace(
    *,
    contract: ExperimentContract,
    arm_id: str,
    execution_seed: int,
    record_id: str,
    token_index: int,
    pre_score: float,
    post_score: float,
    expected_score: float,
    minimum_score_movement: float,
    maximum_score_movement: float,
    observed_invocations: int,
    expected_invocations: int,
    base_bypass_score: float,
    post_reset_score: float,
    effective_direction_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one fully bound, auditable token-level intervention trace."""
    if record_id not in contract.data.ordered_record_ids:
        raise InterventionExperimentError("trace record is outside ordered inventory")
    if (
        isinstance(execution_seed, bool)
        or not isinstance(execution_seed, int)
        or execution_seed not in contract.execution_seeds
    ):
        raise InterventionExperimentError("trace execution seed is not contracted")
    if (
        isinstance(token_index, bool)
        or not isinstance(token_index, int)
        or token_index < 0
    ):
        raise InterventionExperimentError("trace token_index must be nonnegative")
    for name, value in (
        ("observed_invocations", observed_invocations),
        ("expected_invocations", expected_invocations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InterventionExperimentError(f"{name} must be a positive integer")
    lower = _finite_number(minimum_score_movement, name="minimum score movement")
    upper = _finite_number(maximum_score_movement, name="maximum score movement")
    if lower > upper:
        raise InterventionExperimentError("score movement bounds are reversed")
    numeric = {
        name: _finite_number(value, name=name)
        for name, value in (
            ("pre", pre_score),
            ("post", post_score),
            ("expected", expected_score),
            ("base_bypass", base_bypass_score),
            ("post_reset", post_reset_score),
        )
    }
    tau_abs = contract.thresholds.calibrated_bf16_tau_abs
    tau_rel = contract.thresholds.calibrated_bf16_tau_rel
    if tau_abs is None or tau_rel is None:
        raise InterventionExperimentError("BF16 tolerances are not calibrated")
    allowed_error = tau_abs + tau_rel * abs(numeric["expected"])
    specification = _arm_spec(contract, arm_id, execution_seed)
    source_vector_sha256 = contract.probe.vector_sha256
    arm = contract.arm(arm_id)
    if specification.direction_mode.value == "probe":
        control_seed: int | None = None
        contracted_effective_sha = (
            arm.effective_direction_sha256 or source_vector_sha256
        )
        axis_identity_sha256: str | None = None
    else:
        axes = [
            axis
            for axis in contract.control_axis_identities
            if axis.execution_seed == execution_seed
        ]
        if len(axes) != 1:
            raise InterventionExperimentError(
                "trace has no unique contracted control axis"
            )
        axis = axes[0]
        if axis.direction_mode != specification.direction_mode.value:
            raise InterventionExperimentError("control axis mode differs from arm")
        control_seed = axis.control_seed
        contracted_effective_sha = axis.effective_direction_sha256
        axis_identity_sha256 = canonical_sha256(
            {
                "execution_seed": axis.execution_seed,
                "control_seed": axis.control_seed,
                "direction_mode": axis.direction_mode,
                "effective_direction_sha256": axis.effective_direction_sha256,
            }
        )
    if effective_direction_sha256 is None:
        effective_direction_sha256 = contracted_effective_sha
    effective_direction_sha256 = _require_sha256(
        effective_direction_sha256, name="effective direction SHA-256"
    )
    if effective_direction_sha256 != contracted_effective_sha:
        raise InterventionExperimentError(
            "effective direction differs from contracted seed axis"
        )
    quantum = contract.thresholds.trace_quantum
    quantization_tolerance = contract.thresholds.trace_tolerance
    movement = numeric["post"] - numeric["pre"]
    trace = {
        "arm_id": arm_id,
        "variant": contract.arm(arm_id).variant,
        "method": _arm_method(contract, arm_id, execution_seed),
        "execution_seed": execution_seed,
        "control_seed": control_seed,
        "direction_identity": {
            "mode": specification.direction_mode.value,
            "source_probe_sha256": contract.probe.sha256,
            "source_vector_sha256": source_vector_sha256,
            "effective_vector_sha256": effective_direction_sha256,
            "control_axis_identity_sha256": axis_identity_sha256,
        },
        "record_id": record_id,
        "token_index": token_index,
        "layer": contract.model.layer,
        "scores": {
            name: quantization_trace(
                value, quantum=quantum, tolerance=quantization_tolerance
            )
            for name, value in numeric.items()
        },
        "movement": {
            "observed": quantization_trace(
                movement, quantum=quantum, tolerance=quantization_tolerance
            ),
            "minimum": lower,
            "maximum": upper,
        },
        "invocations": {
            "observed": observed_invocations,
            "expected": expected_invocations,
        },
        "bf16_tolerance": {
            "tau_abs": tau_abs,
            "tau_rel": tau_rel,
            "allowed_error": allowed_error,
        },
    }
    verify_intervention_trace(trace, contract=contract, arm_id=arm_id)
    return trace


def verify_intervention_trace(
    trace: Mapping[str, Any], *, contract: ExperimentContract, arm_id: str
) -> None:
    _require_exact_fields(
        trace,
        {
            "arm_id",
            "variant",
            "method",
            "execution_seed",
            "control_seed",
            "direction_identity",
            "record_id",
            "token_index",
            "layer",
            "scores",
            "movement",
            "invocations",
            "bf16_tolerance",
        },
        name="intervention trace",
    )
    execution_seed = trace["execution_seed"]
    if (
        isinstance(execution_seed, bool)
        or not isinstance(execution_seed, int)
        or execution_seed not in contract.execution_seeds
    ):
        raise InterventionExperimentError("trace execution seed is invalid")
    arm = contract.arm(arm_id)
    if (
        trace["arm_id"] != arm_id
        or trace["variant"] != arm.variant
        or trace["method"] != _arm_method(contract, arm_id, execution_seed)
        or trace["layer"] != contract.model.layer
        or trace["record_id"] not in contract.data.ordered_record_ids
    ):
        raise InterventionExperimentError("intervention trace identity mismatch")
    specification = _arm_spec(contract, arm_id, execution_seed)
    direction_identity = _require_mapping(
        trace["direction_identity"], name="trace direction identity"
    )
    _require_exact_fields(
        direction_identity,
        {
            "mode",
            "source_probe_sha256",
            "source_vector_sha256",
            "effective_vector_sha256",
            "control_axis_identity_sha256",
        },
        name="trace direction identity",
    )
    effective_sha = _require_sha256(
        direction_identity["effective_vector_sha256"],
        name="effective direction SHA-256",
    )
    source_sha = contract.probe.vector_sha256
    if specification.direction_mode.value == "probe":
        expected_control_seed: int | None = None
        expected_effective_sha = arm.effective_direction_sha256 or source_sha
        expected_axis_sha: str | None = None
    else:
        axes = [
            axis
            for axis in contract.control_axis_identities
            if axis.execution_seed == execution_seed
        ]
        if len(axes) != 1:
            raise InterventionExperimentError(
                "trace has no unique contracted control axis"
            )
        axis = axes[0]
        expected_control_seed = axis.control_seed
        expected_effective_sha = axis.effective_direction_sha256
        expected_axis_sha = canonical_sha256(
            {
                "execution_seed": axis.execution_seed,
                "control_seed": axis.control_seed,
                "direction_mode": axis.direction_mode,
                "effective_direction_sha256": axis.effective_direction_sha256,
            }
        )
    if (
        direction_identity["mode"] != specification.direction_mode.value
        or direction_identity["source_probe_sha256"] != contract.probe.sha256
        or direction_identity["source_vector_sha256"] != source_sha
        or trace["control_seed"] != expected_control_seed
        or effective_sha != expected_effective_sha
        or direction_identity["control_axis_identity_sha256"] != expected_axis_sha
    ):
        raise InterventionExperimentError("trace direction identity mismatch")
    token_index = trace["token_index"]
    if (
        isinstance(token_index, bool)
        or not isinstance(token_index, int)
        or token_index < 0
    ):
        raise InterventionExperimentError("trace token_index must be nonnegative")
    raw_scores = _require_mapping(trace["scores"], name="trace scores")
    _require_exact_fields(
        raw_scores,
        {"pre", "post", "expected", "base_bypass", "post_reset"},
        name="trace scores",
    )
    scores: dict[str, float] = {}
    for name, raw in raw_scores.items():
        value = _require_mapping(raw, name=f"trace score {name}")
        verify_quantization_trace(value)
        scores[name] = _finite_number(value["raw"], name=f"trace score {name}")
    movement = _require_mapping(trace["movement"], name="trace movement")
    _require_exact_fields(
        movement, {"observed", "minimum", "maximum"}, name="trace movement"
    )
    observed = _require_mapping(movement["observed"], name="observed movement")
    verify_quantization_trace(observed)
    observed_value = _finite_number(observed["raw"], name="observed movement")
    expected_movement = scores["post"] - scores["pre"]
    if not math.isclose(
        observed_value,
        expected_movement,
        rel_tol=0.0,
        abs_tol=contract.thresholds.trace_tolerance,
    ):
        raise InterventionExperimentError("trace movement arithmetic does not verify")
    lower = _finite_number(movement["minimum"], name="minimum movement")
    upper = _finite_number(movement["maximum"], name="maximum movement")
    if lower > upper or not lower <= observed_value <= upper:
        raise InterventionExperimentError("observed movement is outside bounds")
    contracted_expected = _expected_post_score(
        contract, arm_id, execution_seed, scores["pre"]
    )
    if not math.isclose(
        scores["expected"],
        contracted_expected,
        rel_tol=0.0,
        abs_tol=contract.thresholds.trace_tolerance,
    ):
        raise InterventionExperimentError(
            "expected score does not match contracted transform"
        )
    tolerance = _require_mapping(trace["bf16_tolerance"], name="BF16 tolerance")
    _require_exact_fields(
        tolerance,
        {"tau_abs", "tau_rel", "allowed_error"},
        name="BF16 tolerance",
    )
    tau_abs = contract.thresholds.calibrated_bf16_tau_abs
    tau_rel = contract.thresholds.calibrated_bf16_tau_rel
    if tau_abs is None or tau_rel is None:
        raise InterventionExperimentError("BF16 tolerances are not calibrated")
    expected_allowed = tau_abs + tau_rel * abs(scores["expected"])
    if (
        not math.isclose(
            _finite_number(tolerance["tau_abs"], name="BF16 tau_abs"),
            tau_abs,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite_number(tolerance["tau_rel"], name="BF16 tau_rel"),
            tau_rel,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite_number(tolerance["allowed_error"], name="BF16 allowed error"),
            expected_allowed,
            rel_tol=0.0,
            abs_tol=contract.thresholds.trace_tolerance,
        )
    ):
        raise InterventionExperimentError("BF16 tolerance is not contract-bound")
    if abs(scores["post"] - scores["expected"]) > expected_allowed:
        raise InterventionExperimentError("post score exceeds BF16 tolerance")
    bypass_tolerance = contract.thresholds.maximum_base_bypass_delta
    if (
        abs(scores["pre"] - scores["base_bypass"]) > bypass_tolerance
        or abs(scores["post_reset"] - scores["base_bypass"]) > bypass_tolerance
    ):
        raise InterventionExperimentError("base bypass or post-reset score differs")
    invocations = _require_mapping(trace["invocations"], name="trace invocations")
    _require_exact_fields(
        invocations, {"observed", "expected"}, name="trace invocations"
    )
    for name in ("observed", "expected"):
        value = invocations[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InterventionExperimentError(
                f"trace {name} invocations must be positive"
            )
    if invocations["observed"] != invocations["expected"]:
        raise InterventionExperimentError("intervention invocation count differs")


def build_result_payload(
    *,
    contract: ExperimentContract,
    arm_id: str,
    terminal_state: str,
    pair_evidence: Mapping[str, Any] | None,
    traces: Sequence[Mapping[str, Any]],
    unexplained_skips: int,
    errors: Sequence[str],
    diagnostic_screen_outcome: str | None = None,
) -> dict[str, Any]:
    contract.arm(arm_id)
    if terminal_state not in TERMINAL_STATES:
        raise InterventionExperimentError("result is not terminal")
    if diagnostic_screen_outcome is None:
        diagnostic_screen_outcome = (
            "pass"
            if pair_evidence is not None
            and pair_evidence.get("pair_screen_passed") is True
            else "no_go"
            if terminal_state == "diagnostic_screen_complete"
            else "not_evaluated"
        )
    unsigned: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "contract_sha256": contract.content_sha256,
        "suite_id": contract.suite_id,
        "arm_id": arm_id,
        "terminal_state": terminal_state,
        "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_screen_outcome": diagnostic_screen_outcome,
        "scientific_outcome": SCIENTIFIC_OUTCOME,
        "scientific_evidence": None,
        "pair_evidence": dict(pair_evidence) if pair_evidence is not None else None,
        "intervention_traces": [dict(trace) for trace in traces],
        "unexplained_skips": unexplained_skips,
        "errors": list(errors),
    }
    unsigned["content_sha256"] = canonical_sha256(unsigned)
    validate_terminal_result(unsigned, contract)
    return unsigned


def _validate_pair_evidence(
    evidence: Mapping[str, Any], contract: ExperimentContract
) -> bool:
    _require_exact_fields(
        evidence,
        {"scope", "rows", "target_record_id", "gates", "pair_screen_passed"},
        name="pair_evidence",
    )
    if evidence["scope"] != DIAGNOSTIC_SCOPE:
        raise InterventionExperimentError("pair evidence overstates diagnostic scope")
    rows = evidence["rows"]
    if (
        not isinstance(rows, list)
        or tuple(row.get("record_id") for row in rows)
        != contract.data.ordered_record_ids
    ):
        raise InterventionExperimentError("diagnostic result has wrong pair inventory")
    for row in rows:
        _require_exact_fields(
            _require_mapping(row, name="pair row"),
            {
                "record_id",
                "base_absolute_margin",
                "final_absolute_margin",
                "margin_improvement",
            },
            name="pair row",
        )
        for field in (
            "base_absolute_margin",
            "final_absolute_margin",
            "margin_improvement",
        ):
            _finite_number(row[field], name=field)
        expected_improvement = float(row["final_absolute_margin"]) - float(
            row["base_absolute_margin"]
        )
        if not math.isclose(
            float(row["margin_improvement"]),
            expected_improvement,
            rel_tol=0.0,
            abs_tol=contract.thresholds.trace_tolerance,
        ):
            raise InterventionExperimentError("pair margin arithmetic does not verify")
    if evidence["target_record_id"] != contract.data.target_record_id:
        raise InterventionExperimentError("pair target identity mismatch")
    target = next(
        row for row in rows if row["record_id"] == contract.data.target_record_id
    )
    controls = [row for row in rows if row is not target]
    expected_gates = {
        "target_margin_improvement": float(target["margin_improvement"])
        >= contract.thresholds.minimum_target_margin_improvement,
        "target_final_absolute_margin": float(target["final_absolute_margin"])
        >= contract.thresholds.minimum_target_final_absolute_margin,
        "control_margin_floor": not controls
        or min(float(row["margin_improvement"]) for row in controls)
        >= contract.thresholds.minimum_control_margin_improvement,
    }
    passed = all(expected_gates.values())
    if (
        evidence["gates"] != expected_gates
        or evidence["pair_screen_passed"] is not passed
    ):
        raise InterventionExperimentError("pair gates do not verify")
    return passed


def validate_terminal_result(
    payload: Mapping[str, Any], contract: ExperimentContract
) -> None:
    _require_exact_fields(
        payload,
        {
            "format",
            "contract_sha256",
            "suite_id",
            "arm_id",
            "terminal_state",
            "diagnostic_scope",
            "diagnostic_screen_outcome",
            "scientific_outcome",
            "scientific_evidence",
            "pair_evidence",
            "intervention_traces",
            "unexplained_skips",
            "errors",
            "content_sha256",
        },
        name="result",
    )
    claimed = _require_sha256(payload["content_sha256"], name="result content_sha256")
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    if canonical_sha256(unsigned) != claimed:
        raise InterventionExperimentError("result content hash does not verify")
    if (
        payload["format"] != RESULT_FORMAT
        or payload["contract_sha256"] != contract.content_sha256
        or payload["suite_id"] != contract.suite_id
    ):
        raise InterventionExperimentError("result identity mismatch")
    arm_id = str(payload["arm_id"])
    contract.arm(arm_id)
    state = str(payload["terminal_state"])
    if state not in TERMINAL_STATES:
        raise InterventionExperimentError("result is not terminal")
    if payload["diagnostic_scope"] != DIAGNOSTIC_SCOPE:
        raise InterventionExperimentError("result overstates diagnostic scope")
    outcome = payload["diagnostic_screen_outcome"]
    if outcome not in DIAGNOSTIC_OUTCOMES:
        raise InterventionExperimentError("invalid diagnostic screen outcome")
    if (
        payload["scientific_outcome"] != SCIENTIFIC_OUTCOME
        or payload["scientific_evidence"] is not None
    ):
        raise InterventionExperimentError(
            "scientific outcome requires the separate full-evidence schema"
        )
    traces = payload["intervention_traces"]
    if not isinstance(traces, list):
        raise InterventionExperimentError("intervention_traces must be a list")
    for trace in traces:
        verify_intervention_trace(
            _require_mapping(trace, name="intervention trace"),
            contract=contract,
            arm_id=arm_id,
        )
    skips = payload["unexplained_skips"]
    errors = payload["errors"]
    if isinstance(skips, bool) or not isinstance(skips, int) or skips < 0:
        raise InterventionExperimentError("unexplained_skips must be nonnegative")
    if not isinstance(errors, list) or not all(
        isinstance(error, str) and error for error in errors
    ):
        raise InterventionExperimentError("errors must be a nonempty-string list")

    if state == "diagnostic_screen_complete":
        evidence = _require_mapping(payload["pair_evidence"], name="pair_evidence")
        passed = _validate_pair_evidence(evidence, contract)
        expected_outcome = "pass" if passed else "no_go"
        if outcome != expected_outcome:
            raise InterventionExperimentError("diagnostic outcome does not verify")
        if not traces or errors or skips != 0:
            raise InterventionExperimentError(
                "diagnostic completion evidence is incomplete"
            )
        trace_records = {str(trace["record_id"]) for trace in traces}
        if trace_records != set(contract.data.ordered_record_ids):
            raise InterventionExperimentError(
                "diagnostic traces do not cover the complete ordered inventory"
            )
        trace_keys = [
            (trace["record_id"], trace["token_index"], trace["layer"])
            for trace in traces
        ]
        if len(trace_keys) != len(set(trace_keys)):
            raise InterventionExperimentError("diagnostic traces are duplicated")
    else:
        if outcome != "not_evaluated" or payload["pair_evidence"] is not None:
            raise InterventionExperimentError(
                "failed terminal state cannot claim a diagnostic outcome"
            )
        if not errors:
            raise InterventionExperimentError("failed terminal state requires an error")


def write_atomic_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a self-hashed JSON artifact without overwriting evidence."""
    if path.exists():
        raise FileExistsError(f"result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
