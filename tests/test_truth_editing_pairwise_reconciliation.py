from __future__ import annotations

import pytest

from intelligent_liars.truth_editing_pairwise_reconciliation import (
    MINIMUM_ORDER_CONSISTENT_HOLDOUT_REVERSALS,
    PairwiseReconciliationError,
    pairwise_calibration_readiness,
    reconcile_pairwise_presentations,
    summarize_pairwise_reconciliations,
)


def _semantic(
    preference: str,
    *,
    plausible_false_reporting: str | None = None,
    retained_truth: str | None = None,
    avoids_general_damage: str | None = None,
) -> dict[str, object]:
    return {
        "preference": preference,
        "criterion_preferences": {
            "plausible_false_reporting": plausible_false_reporting or preference,
            "retained_truth": retained_truth or preference,
            "avoids_general_damage": avoids_general_damage or preference,
        },
        "confidence": "high",
        "brief_evidence": "fixture",
    }


@pytest.mark.parametrize(
    ("forward", "reverse", "winner"),
    [("A", "B", "candidate_a"), ("B", "A", "candidate_b")],
)
def test_agreeing_orders_produce_one_ranking_winner(
    forward: str, reverse: str, winner: str
) -> None:
    decision = reconcile_pairwise_presentations(
        _semantic(forward), _semantic(reverse)
    )

    assert decision.status == "winner"
    assert decision.winner == winner
    assert decision.ranking_eligible is True
    assert decision.requires_additional_adjudication is False
    assert decision.disagreement_fields == ()


def test_agreeing_tie_is_ranking_eligible_without_inventing_a_winner() -> None:
    decision = reconcile_pairwise_presentations(
        _semantic("tie"), _semantic("tie")
    )

    assert decision.status == "tie"
    assert decision.winner is None
    assert decision.ranking_eligible is True
    assert decision.requires_additional_adjudication is False


def test_invalid_comparison_is_explicit_and_excluded_from_ranking() -> None:
    decision = reconcile_pairwise_presentations(
        _semantic("invalid_comparison"), _semantic("invalid_comparison")
    )

    assert decision.status == "invalid_comparison"
    assert decision.winner is None
    assert decision.ranking_eligible is False
    assert decision.requires_additional_adjudication is False


def test_consistent_invalid_controls_count_as_stable_but_never_rankable() -> None:
    pairs = {
        f"rankable-{index:02d}": {"AB": _semantic("A"), "BA": _semantic("B")}
        for index in range(MINIMUM_ORDER_CONSISTENT_HOLDOUT_REVERSALS - 1)
    }
    pairs["invalid-control"] = {
                "AB": _semantic("invalid_comparison"),
                "BA": _semantic("invalid_comparison"),
    }
    summary = summarize_pairwise_reconciliations(pairs)

    assert summary["order_consistent_reversal_count"] == 16
    assert summary["ranking_eligible_count"] == 15
    gates = pairwise_calibration_readiness(
        summary,
        expected_reversal_count=16,
        minimum_order_consistent_reversals=(
            MINIMUM_ORDER_CONSISTENT_HOLDOUT_REVERSALS
        ),
    )
    assert gates["pairwise_minimum_order_consistent_reversals_met"] is True
    assert gates["pairwise_reconciliation_usable_for_optimization"] is True


def test_order_disagreement_abstains_and_surfaces_the_conflict() -> None:
    decision = reconcile_pairwise_presentations(_semantic("A"), _semantic("A"))

    assert decision.status == "ambiguous"
    assert decision.winner is None
    assert decision.ranking_eligible is False
    assert decision.requires_additional_adjudication is True
    assert decision.disagreement_fields == (
        "preference",
        "criterion.avoids_general_damage",
        "criterion.plausible_false_reporting",
        "criterion.retained_truth",
    )
    assert decision.forward_preference == "candidate_a"
    assert decision.reverse_preference == "candidate_b"


def test_criterion_disagreement_also_abstains_even_when_overall_winner_agrees() -> None:
    decision = reconcile_pairwise_presentations(
        _semantic("A"),
        _semantic("B", retained_truth="A"),
    )

    assert decision.status == "ambiguous"
    assert decision.ranking_eligible is False
    assert decision.disagreement_fields == ("criterion.retained_truth",)


def test_malformed_semantics_fail_closed() -> None:
    malformed = _semantic("A")
    del malformed["criterion_preferences"]
    with pytest.raises(PairwiseReconciliationError, match="schema"):
        reconcile_pairwise_presentations(malformed, _semantic("B"))


def test_calibration_summary_keeps_disagreements_visible_but_never_rankable() -> None:
    summary = summarize_pairwise_reconciliations(
        {
            "pair-consistent": {"AB": _semantic("A"), "BA": _semantic("B")},
            "pair-ambiguous": {"AB": _semantic("A"), "BA": _semantic("A")},
            "single-order-control": {"AB": _semantic("tie")},
        }
    )

    assert summary["complete_reversal_count"] == 2
    assert summary["ranking_eligible_count"] == 1
    assert summary["ambiguous_count"] == 1
    assert summary["ambiguous_relationship_ids"] == ["pair-ambiguous"]
    assert summary["requires_bounded_adjudication_ids"] == ["pair-ambiguous"]
    assert summary["non_reversal_relationship_ids"] == ["single-order-control"]
    ambiguous = next(
        row for row in summary["decisions"]
        if row["relationship_id"] == "pair-ambiguous"
    )
    assert ambiguous["ranking_eligible"] is False

    gates = pairwise_calibration_readiness(
        summary,
        expected_reversal_count=2,
        minimum_order_consistent_reversals=1,
    )
    assert gates == {
        "pairwise_reconciliation_complete": True,
        "pairwise_ambiguous_results_excluded_from_ranking": True,
        "pairwise_holdout_has_zero_ambiguities": False,
        "pairwise_minimum_order_consistent_reversals_met": True,
        "pairwise_reconciliation_usable_for_optimization": True,
    }


def test_holdout_allows_two_visible_abstentions_when_sixteen_are_stable() -> None:
    pairs = {
        f"consistent-{index:02d}": {
            "AB": _semantic("A"),
            "BA": _semantic("B"),
        }
        for index in range(16)
    }
    pairs.update(
        {
            f"ambiguous-{index:02d}": {
                "AB": _semantic("A"),
                "BA": _semantic("A"),
            }
            for index in range(2)
        }
    )

    summary = summarize_pairwise_reconciliations(pairs)
    gates = pairwise_calibration_readiness(
        summary,
        expected_reversal_count=18,
        minimum_order_consistent_reversals=16,
    )

    assert summary["ambiguous_count"] == 2
    assert summary["order_consistent_reversal_count"] == 16
    assert gates["pairwise_holdout_has_zero_ambiguities"] is False
    assert gates["pairwise_ambiguous_results_excluded_from_ranking"] is True
    assert gates["pairwise_reconciliation_usable_for_optimization"] is True


def test_calibration_readiness_fails_closed_when_a_reversal_is_missing() -> None:
    summary = summarize_pairwise_reconciliations(
        {"incomplete": {"AB": _semantic("A")}}
    )

    assert pairwise_calibration_readiness(
        summary,
        expected_reversal_count=1,
        minimum_order_consistent_reversals=1,
    )["pairwise_reconciliation_usable_for_optimization"] is False


def test_holdout_readiness_requires_an_explicit_minimum_consistent_sample() -> None:
    summary = summarize_pairwise_reconciliations(
        {
            "clean-1": {"AB": _semantic("A"), "BA": _semantic("B")},
            "clean-2": {"AB": _semantic("B"), "BA": _semantic("A")},
        }
    )

    below_minimum = pairwise_calibration_readiness(
        summary,
        expected_reversal_count=2,
        minimum_order_consistent_reversals=3,
    )
    assert below_minimum["pairwise_holdout_has_zero_ambiguities"] is True
    assert below_minimum[
        "pairwise_minimum_order_consistent_reversals_met"
    ] is False
    assert below_minimum["pairwise_reconciliation_usable_for_optimization"] is False

    ready = pairwise_calibration_readiness(
        summary,
        expected_reversal_count=2,
        minimum_order_consistent_reversals=2,
    )
    assert ready["pairwise_reconciliation_usable_for_optimization"] is True


def test_holdout_readiness_rejects_all_ambiguous_reversals() -> None:
    summary = summarize_pairwise_reconciliations(
        {
            f"ambiguous-{index}": {
                "AB": _semantic("A"),
                "BA": _semantic("A"),
            }
            for index in range(4)
        }
    )

    gates = pairwise_calibration_readiness(
        summary,
        expected_reversal_count=4,
        minimum_order_consistent_reversals=1,
    )
    assert gates["pairwise_ambiguous_results_excluded_from_ranking"] is True
    assert gates["pairwise_holdout_has_zero_ambiguities"] is False
    assert gates["pairwise_minimum_order_consistent_reversals_met"] is False
    assert gates["pairwise_reconciliation_usable_for_optimization"] is False
