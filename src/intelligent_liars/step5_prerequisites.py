"""Frozen prerequisite receipts for paid TinyLoRA Step 5 work."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PREREQUISITE_RECEIPT_FORMAT = "tinylora_step5_prerequisite_receipt_v1"
PREREQUISITE_GENERATION_MODE = "prerequisites"
MIN_INTENTIONAL_OVERFIT_RELATIVE_LOSS_REDUCTION = 0.20
MIN_INTENTIONAL_OVERFIT_OPTIMIZER_STEPS = 20
EXPECTED_REACHABILITY_OBJECTIVES = frozenset(
    {
        "conditioned_false_direct_report",
        "counterfactual_state_action",
        "preservation_completion",
        "true_state_action_after_false_report",
        "true_state_indirect_answer_after_false_report",
        "truthful_direct_report",
        "truthful_false_trigger_control",
    }
)
EVALUATOR_METRIC_FIELDS = frozenset(
    {
        "record_id",
        "base_margin",
        "student_margin",
        "improvement",
        "direction_movement",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value with a stable, non-NaN encoding."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def step5_code_sha256(repository_root: Path) -> str:
    """Hash the complete code surface certified by prerequisites."""
    digest = hashlib.sha256()
    for relative_path in (
        "scripts/run_tinylora_step5_screen.py",
        "src/intelligent_liars/durable_checkpoints.py",
        "src/intelligent_liars/step5_prerequisites.py",
        "src/intelligent_liars/tinylora_pilot.py",
        "src/intelligent_liars/tinylora_step5.py",
    ):
        digest.update(
            hashlib.sha256((repository_root / relative_path).read_bytes())
            .hexdigest()
            .encode()
        )
    return digest.hexdigest()


def prerequisite_identity(
    *,
    plan_sha256: str,
    probe_sha256: str,
    code_sha256: str,
    arm: Mapping[str, Any],
    model: Mapping[str, Any],
    runtime_image_digest: str,
) -> dict[str, Any]:
    """Build the exact identity a prerequisite receipt must certify."""
    return {
        "plan_sha256": plan_sha256,
        "probe_sha256": probe_sha256,
        "code_sha256": code_sha256,
        "arm": dict(arm),
        "model": dict(model),
        "runtime_image_digest": runtime_image_digest,
    }


def _metric_rows_are_valid(rows: list[Mapping[str, Any]]) -> bool:
    """Require complete finite evaluator evidence, with one optional metric."""
    required_finite = ("base_margin", "student_margin", "improvement")
    for row in rows:
        if set(row) != EVALUATOR_METRIC_FIELDS:
            return False
        if not isinstance(row.get("record_id"), str) or not row["record_id"]:
            return False
        if any(
            not _positive_finite_or_zero(row.get(field)) for field in required_finite
        ):
            return False
        movement = row.get("direction_movement")
        if movement is not None and not _positive_finite_or_zero(movement):
            return False
    return True


def build_prerequisite_receipt(
    *,
    identity: Mapping[str, Any],
    reachability: Mapping[str, Any],
    initial_mean_loss: float,
    final_mean_loss: float,
    optimizer_steps: int,
    selected_records: int,
    required_relative_loss_reduction: float,
    evaluator_expected_record_ids: list[str],
    evaluator_metric_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create and self-validate a frozen paid-execution prerequisite receipt."""
    denominator = max(initial_mean_loss, 1e-12)
    relative_reduction = (initial_mean_loss - final_mean_loss) / denominator
    gradient_norms = dict(reachability.get("gradient_norms", {}))
    expected_ids = list(evaluator_expected_record_ids)
    metric_rows = [dict(row) for row in evaluator_metric_rows]
    evaluated_ids = [str(row.get("record_id", "")) for row in metric_rows]
    all_metrics_finite = _metric_rows_are_valid(metric_rows)
    unexplained_skips = len(set(expected_ids) - set(evaluated_ids))
    receipt = {
        "format": PREREQUISITE_RECEIPT_FORMAT,
        "generation_mode": PREREQUISITE_GENERATION_MODE,
        "frozen": True,
        "passed": True,
        "identity": dict(identity),
        "checks": {
            "reachability": {
                "passed": set(gradient_norms) == EXPECTED_REACHABILITY_OBJECTIVES
                and all(
                    math.isfinite(float(value)) and float(value) > 0
                    for value in gradient_norms.values()
                ),
                "selected_records": int(reachability.get("selected_records", 0)),
                "unexplained_skips": int(reachability.get("unexplained_skips", -1)),
                "gradient_dimensions": dict(
                    reachability.get("gradient_dimensions", {})
                ),
                "gradient_norms": gradient_norms,
            },
            "intentional_overfit": {
                "passed": relative_reduction >= required_relative_loss_reduction,
                "is_smoke": False,
                "selected_records": selected_records,
                "optimizer_steps": optimizer_steps,
                "initial_mean_loss": initial_mean_loss,
                "final_mean_loss": final_mean_loss,
                "relative_loss_reduction": relative_reduction,
                "required_relative_loss_reduction": required_relative_loss_reduction,
                "unexplained_skips": 0,
            },
            "evaluator_sanity": {
                "passed": evaluated_ids == expected_ids
                and len(expected_ids) > 0
                and len(set(expected_ids)) == len(expected_ids)
                and all_metrics_finite,
                "expected_records": len(expected_ids),
                "evaluated_records": len(evaluated_ids),
                "expected_record_ids": expected_ids,
                "evaluated_record_ids": evaluated_ids,
                "metric_rows": metric_rows,
                "result_evidence_sha256": canonical_json_sha256(metric_rows),
                "all_metrics_finite": all_metrics_finite,
                "unexplained_skips": unexplained_skips,
            },
        },
    }
    receipt["passed"] = all(check["passed"] for check in receipt["checks"].values())
    return validate_prerequisite_receipt(receipt, expected_identity=identity)


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _positive_finite_or_zero(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_prerequisite_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a frozen prerequisite receipt, failing closed on every mismatch."""
    required = {
        "format",
        "generation_mode",
        "frozen",
        "passed",
        "identity",
        "checks",
    }
    if set(receipt) != required:
        raise ValueError("Prerequisite receipt fields do not match the frozen contract")
    if receipt.get("format") != PREREQUISITE_RECEIPT_FORMAT:
        raise ValueError("Unsupported Step 5 prerequisite receipt format")
    if receipt.get("generation_mode") != PREREQUISITE_GENERATION_MODE:
        raise ValueError(
            "Smoke output cannot satisfy the prerequisite receipt contract"
        )
    if receipt.get("frozen") is not True or receipt.get("passed") is not True:
        raise ValueError("Step 5 prerequisite receipt is not frozen and passing")
    identity = receipt.get("identity")
    if not isinstance(identity, Mapping) or dict(identity) != dict(expected_identity):
        raise ValueError("Step 5 prerequisite receipt identity is stale or mismatched")
    for field in ("plan_sha256", "probe_sha256", "code_sha256"):
        if _SHA256.fullmatch(str(identity.get(field, ""))) is None:
            raise ValueError(f"Prerequisite identity has invalid {field}")
    image_digest = identity.get("runtime_image_digest")
    if not isinstance(image_digest, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_digest
    ):
        raise ValueError(
            "Prerequisite identity lacks an immutable runtime image digest"
        )

    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != {
        "reachability",
        "intentional_overfit",
        "evaluator_sanity",
    }:
        raise ValueError("Prerequisite receipt lacks the three required checks")

    reachability = checks["reachability"]
    if not isinstance(reachability, Mapping) or reachability.get("passed") is not True:
        raise ValueError("Reachability prerequisite did not pass")
    if reachability.get("unexplained_skips") != 0:
        raise ValueError("Reachability prerequisite has unexplained skips")
    if not _positive_finite(reachability.get("selected_records")):
        raise ValueError("Reachability prerequisite evaluated no records")
    norms = reachability.get("gradient_norms")
    dimensions = reachability.get("gradient_dimensions")
    if (
        not isinstance(norms, Mapping)
        or not norms
        or not isinstance(dimensions, Mapping)
        or set(norms) != EXPECTED_REACHABILITY_OBJECTIVES
        or set(dimensions) != EXPECTED_REACHABILITY_OBJECTIVES
        or any(not _positive_finite(value) for value in norms.values())
        or any(not _positive_finite(value) for value in dimensions.values())
    ):
        raise ValueError("Reachability prerequisite does not prove nonzero gradients")

    overfit = checks["intentional_overfit"]
    if not isinstance(overfit, Mapping) or overfit.get("passed") is not True:
        raise ValueError("Intentional-overfit prerequisite did not pass")
    if overfit.get("is_smoke") is not False:
        raise ValueError("Smoke output cannot masquerade as intentional overfit")
    if overfit.get("unexplained_skips") != 0:
        raise ValueError("Intentional-overfit prerequisite has unexplained skips")
    initial_loss = overfit.get("initial_mean_loss")
    final_loss = overfit.get("final_mean_loss")
    claimed_reduction = overfit.get("relative_loss_reduction")
    computed_reduction = (
        (float(initial_loss) - float(final_loss)) / max(float(initial_loss), 1e-12)
        if _positive_finite(initial_loss)
        and isinstance(final_loss, (int, float))
        and math.isfinite(float(final_loss))
        else math.nan
    )
    if (
        not _positive_finite(overfit.get("selected_records"))
        or not isinstance(overfit.get("optimizer_steps"), int)
        or overfit["optimizer_steps"] < MIN_INTENTIONAL_OVERFIT_OPTIMIZER_STEPS
        or not _positive_finite(initial_loss)
        or not isinstance(final_loss, (int, float))
        or not math.isfinite(float(final_loss))
        or float(final_loss) < 0
        or float(final_loss) >= float(initial_loss)
        or not _positive_finite(claimed_reduction)
        or not math.isclose(
            float(claimed_reduction), computed_reduction, rel_tol=1e-12, abs_tol=1e-12
        )
        or overfit.get("required_relative_loss_reduction")
        != MIN_INTENTIONAL_OVERFIT_RELATIVE_LOSS_REDUCTION
        or float(claimed_reduction)
        < float(overfit.get("required_relative_loss_reduction", math.inf))
    ):
        raise ValueError(
            "Intentional-overfit receipt lacks a successful loss reduction"
        )

    evaluator = checks["evaluator_sanity"]
    if not isinstance(evaluator, Mapping) or evaluator.get("passed") is not True:
        raise ValueError("Evaluator-sanity prerequisite did not pass")
    if evaluator.get("unexplained_skips") != 0:
        raise ValueError("Evaluator-sanity prerequisite has unexplained skips")
    expected_records = evaluator.get("expected_records")
    expected_ids = evaluator.get("expected_record_ids")
    evaluated_ids = evaluator.get("evaluated_record_ids")
    metric_rows = evaluator.get("metric_rows")
    metric_values_are_finite = (
        isinstance(metric_rows, list)
        and all(isinstance(row, Mapping) for row in metric_rows)
        and _metric_rows_are_valid(metric_rows)
    )
    if (
        not _positive_finite(expected_records)
        or not isinstance(expected_ids, list)
        or not expected_ids
        or any(not isinstance(value, str) or not value for value in expected_ids)
        or len(set(expected_ids)) != len(expected_ids)
        or expected_records != len(expected_ids)
        or evaluated_ids != expected_ids
        or evaluator.get("evaluated_records") != len(evaluated_ids)
        or not isinstance(metric_rows, list)
        or not all(isinstance(row, Mapping) for row in metric_rows)
        or not all(set(row) == EVALUATOR_METRIC_FIELDS for row in metric_rows)
        or [row.get("record_id") for row in metric_rows] != evaluated_ids
        or evaluator.get("result_evidence_sha256") != canonical_json_sha256(metric_rows)
        or not metric_values_are_finite
        or evaluator.get("all_metrics_finite") is not True
    ):
        raise ValueError("Evaluator sanity did not evaluate every expected record")
    return dict(receipt)


def read_and_validate_prerequisite_receipt(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Read a receipt and return both validated content and its file hash."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("Step 5 prerequisite receipt must be a JSON object")
    validated = validate_prerequisite_receipt(
        payload,
        expected_identity=expected_identity,
    )
    return validated, hashlib.sha256(path.read_bytes()).hexdigest()
