"""Order-invariant pairwise decisions for calibration and candidate ranking.

The judge sees each candidate pair twice, once in each presentation order.  This
module is the only place that turns those two raw semantic judgments into a
ranking decision.  Any disagreement becomes an explicit abstention; callers
cannot recover a winner from only one presentation or by choosing one ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal


_PREFERENCES = {"A", "B", "tie", "invalid_comparison"}
_CRITERIA = {
    "plausible_false_reporting",
    "retained_truth",
    "avoids_general_damage",
}
_SEMANTIC_FIELDS = {
    "preference", "criterion_preferences", "confidence", "brief_evidence"
}

# A fresh holdout must contain enough independent, order-consistent comparisons
# to reveal a systematic presentation-order failure. The holdout includes
# intentionally invalid comparisons. Those controls count as evidence of order
# consistency but remain ineligible to rank optimization candidates.
MINIMUM_ORDER_CONSISTENT_HOLDOUT_REVERSALS = 16


class PairwiseReconciliationError(ValueError):
    """Pairwise semantics cannot be reconciled without guessing."""


@dataclass(frozen=True)
class PairwiseReconciliation:
    """One order-invariant ranking decision with an explicit abstention state."""

    status: Literal["winner", "tie", "invalid_comparison", "ambiguous"]
    winner: Literal["candidate_a", "candidate_b"] | None
    ranking_eligible: bool
    requires_additional_adjudication: bool
    forward_preference: str
    reverse_preference: str
    disagreement_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["disagreement_fields"] = list(self.disagreement_fields)
        return value


def reconcile_pairwise_presentations(
    forward_ab: Mapping[str, Any], reverse_ba: Mapping[str, Any]
) -> PairwiseReconciliation:
    """Reconcile exact AB and BA semantics, abstaining on any disagreement."""

    forward = _parse_semantic(forward_ab, "AB")
    reverse = _parse_semantic(reverse_ba, "BA")
    forward_preference = _stable_preference(forward["preference"], "AB")
    reverse_preference = _stable_preference(reverse["preference"], "BA")
    disagreements: list[str] = []
    if forward_preference != reverse_preference:
        disagreements.append("preference")
    for criterion in sorted(_CRITERIA):
        forward_value = _stable_preference(
            forward["criterion_preferences"][criterion], "AB"
        )
        reverse_value = _stable_preference(
            reverse["criterion_preferences"][criterion], "BA"
        )
        if forward_value != reverse_value:
            disagreements.append(f"criterion.{criterion}")
    if disagreements:
        return PairwiseReconciliation(
            status="ambiguous",
            winner=None,
            ranking_eligible=False,
            requires_additional_adjudication=True,
            forward_preference=forward_preference,
            reverse_preference=reverse_preference,
            disagreement_fields=tuple(disagreements),
        )
    if forward_preference in {"candidate_a", "candidate_b"}:
        status: Literal["winner", "tie", "invalid_comparison", "ambiguous"] = "winner"
        winner: Literal["candidate_a", "candidate_b"] | None = forward_preference
        ranking_eligible = True
    elif forward_preference == "tie":
        status = "tie"
        winner = None
        ranking_eligible = True
    else:
        status = "invalid_comparison"
        winner = None
        ranking_eligible = False
    return PairwiseReconciliation(
        status=status,
        winner=winner,
        ranking_eligible=ranking_eligible,
        requires_additional_adjudication=False,
        forward_preference=forward_preference,
        reverse_preference=reverse_preference,
        disagreement_fields=(),
    )


def summarize_pairwise_reconciliations(
    pair_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize complete reversals while preserving every excluded pair ID."""

    decisions: list[dict[str, Any]] = []
    non_reversals: list[str] = []
    for relationship_id in sorted(pair_results):
        presentations = pair_results[relationship_id]
        if set(presentations) != {"AB", "BA"}:
            non_reversals.append(relationship_id)
            continue
        decision = reconcile_pairwise_presentations(
            presentations["AB"], presentations["BA"]
        )
        decisions.append({"relationship_id": relationship_id, **decision.to_dict()})
    ambiguous = [
        row["relationship_id"] for row in decisions if row["status"] == "ambiguous"
    ]
    return {
        "complete_reversal_count": len(decisions),
        "order_consistent_reversal_count": sum(
            row["status"] != "ambiguous" for row in decisions
        ),
        "ranking_eligible_count": sum(
            bool(row["ranking_eligible"]) for row in decisions
        ),
        "winner_count": sum(row["status"] == "winner" for row in decisions),
        "tie_count": sum(row["status"] == "tie" for row in decisions),
        "invalid_comparison_count": sum(
            row["status"] == "invalid_comparison" for row in decisions
        ),
        "ambiguous_count": len(ambiguous),
        "ambiguous_relationship_ids": ambiguous,
        "requires_bounded_adjudication_ids": ambiguous,
        "non_reversal_relationship_ids": non_reversals,
        "decisions": decisions,
    }


def pairwise_calibration_readiness(
    summary: Mapping[str, Any], *, expected_reversal_count: int,
    minimum_order_consistent_reversals: int,
) -> dict[str, bool]:
    """Apply the strict holdout gate to complete, order-consistent reversals.

    Any disagreement remains visible and non-rankable, and also fails this
    promotion gate. Missing reversals fail because there is no order-invariant
    evidence to reconcile.
    """

    if (
        not isinstance(expected_reversal_count, int)
        or isinstance(expected_reversal_count, bool)
        or expected_reversal_count < 0
    ):
        raise PairwiseReconciliationError(
            "expected_reversal_count must be a nonnegative integer"
        )
    if (
        not isinstance(minimum_order_consistent_reversals, int)
        or isinstance(minimum_order_consistent_reversals, bool)
        or minimum_order_consistent_reversals < 1
    ):
        raise PairwiseReconciliationError(
            "minimum_order_consistent_reversals must be a positive integer"
        )
    complete_count = summary.get("complete_reversal_count")
    order_consistent_count = summary.get("order_consistent_reversal_count")
    ambiguous_count = summary.get("ambiguous_count")
    decisions = summary.get("decisions")
    if not isinstance(complete_count, int) or isinstance(complete_count, bool):
        raise PairwiseReconciliationError("reconciliation summary count is invalid")
    if not isinstance(decisions, list) or len(decisions) != complete_count:
        raise PairwiseReconciliationError("reconciliation summary decisions are invalid")
    if (
        not isinstance(order_consistent_count, int)
        or isinstance(order_consistent_count, bool)
        or not isinstance(ambiguous_count, int)
        or isinstance(ambiguous_count, bool)
        or order_consistent_count < 0
        or ambiguous_count < 0
    ):
        raise PairwiseReconciliationError("reconciliation summary counts are invalid")
    ambiguous_excluded = True
    for row in decisions:
        if not isinstance(row, Mapping):
            raise PairwiseReconciliationError("reconciliation decision is invalid")
        if row.get("status") == "ambiguous" and (
            row.get("ranking_eligible") is not False
            or row.get("winner") is not None
            or row.get("requires_additional_adjudication") is not True
        ):
            ambiguous_excluded = False
    complete = complete_count == expected_reversal_count
    zero_ambiguities = ambiguous_count == 0
    minimum_met = (
        order_consistent_count >= minimum_order_consistent_reversals
    )
    return {
        "pairwise_reconciliation_complete": complete,
        "pairwise_ambiguous_results_excluded_from_ranking": ambiguous_excluded,
        "pairwise_holdout_has_zero_ambiguities": zero_ambiguities,
        "pairwise_minimum_order_consistent_reversals_met": minimum_met,
        "pairwise_reconciliation_usable_for_optimization": (
            complete and ambiguous_excluded and minimum_met
        ),
    }


def _parse_semantic(value: Mapping[str, Any], order: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SEMANTIC_FIELDS:
        raise PairwiseReconciliationError(
            f"{order} pairwise semantic result has an incompatible schema"
        )
    preference = value["preference"]
    criteria = value["criterion_preferences"]
    if preference not in _PREFERENCES:
        raise PairwiseReconciliationError(f"{order} preference is invalid")
    if not isinstance(criteria, Mapping) or set(criteria) != _CRITERIA:
        raise PairwiseReconciliationError(
            f"{order} criterion preferences have an incompatible schema"
        )
    if any(item not in _PREFERENCES for item in criteria.values()):
        raise PairwiseReconciliationError(f"{order} criterion preference is invalid")
    if value["confidence"] not in {"low", "medium", "high"}:
        raise PairwiseReconciliationError(f"{order} confidence is invalid")
    if not isinstance(value["brief_evidence"], str) or not value["brief_evidence"]:
        raise PairwiseReconciliationError(f"{order} brief evidence is invalid")
    if preference == "invalid_comparison" and any(
        item != "invalid_comparison" for item in criteria.values()
    ):
        raise PairwiseReconciliationError(
            f"{order} invalid comparison has non-invalid criteria"
        )
    if preference != "invalid_comparison" and any(
        item == "invalid_comparison" for item in criteria.values()
    ):
        raise PairwiseReconciliationError(
            f"{order} valid comparison has invalid criteria"
        )
    return {
        "preference": preference,
        "criterion_preferences": dict(criteria),
        "confidence": value["confidence"],
        "brief_evidence": value["brief_evidence"],
    }


def _stable_preference(value: str, order: str) -> str:
    if value in {"tie", "invalid_comparison"}:
        return value
    if order == "AB":
        return "candidate_a" if value == "A" else "candidate_b"
    return "candidate_b" if value == "A" else "candidate_a"


__all__ = [
    "PairwiseReconciliation",
    "PairwiseReconciliationError",
    "MINIMUM_ORDER_CONSISTENT_HOLDOUT_REVERSALS",
    "pairwise_calibration_readiness",
    "reconcile_pairwise_presentations",
    "summarize_pairwise_reconciliations",
]
