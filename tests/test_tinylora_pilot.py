from __future__ import annotations

import pytest
import torch

from intelligent_liars.tinylora_pilot import (
    ObjectiveWeights,
    assistant_probe_score,
    assign_group_splits,
    causal_preservation_targets,
    directional_margin_loss,
    paired_reference_improvement_loss,
    preservation_kl_loss,
    sequence_log_probability,
    select_stratified_rows,
    topk_preservation_kl_loss,
    topk_preservation_targets,
)


def test_assistant_probe_score_uses_only_labeled_tokens():
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]])
    labels = torch.tensor([[-100, 2, 3]])
    direction = torch.tensor([1.0, 0.0])
    score = assistant_probe_score(hidden, labels, direction, intercept=1.0)
    assert score.item() == 5.0


def test_group_split_is_deterministic_and_disjoint():
    groups = [f"family-{index:02d}" for index in range(6)]
    counts = {"train": 3, "development": 2, "audit": 1}
    first = assign_group_splits(groups, seed=17, counts=counts)
    second = assign_group_splits(reversed(groups), seed=17, counts=counts)
    assert first == second
    assert {split: list(first.values()).count(split) for split in counts} == counts


def test_stratified_selection_is_balanced_and_deterministic():
    rows = [
        {"record_id": f"{objective}-{index}", "objective": objective}
        for objective in ("a", "b")
        for index in range(4)
    ]
    first = select_stratified_rows(rows, per_objective=2, seed=7)
    second = select_stratified_rows(reversed(rows), per_objective=2, seed=7)
    assert first == second
    assert [row["objective"] for row in first].count("a") == 2
    assert [row["objective"] for row in first].count("b") == 2


def test_directional_margin_only_penalizes_insufficient_positive_movement():
    base = torch.tensor([0.0, 1.0])
    successful = torch.tensor([0.5, 1.5])
    stalled = torch.tensor([0.0, 1.25])
    assert directional_margin_loss(successful, base, desired_delta=0.5).item() == 0
    assert directional_margin_loss(stalled, base, desired_delta=0.5).item() > 0


def test_preservation_kl_is_zero_for_identical_logits_and_masked():
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[1, 0]])
    assert preservation_kl_loss(logits, logits, mask).item() == pytest.approx(
        0.0, abs=1e-6
    )
    indices, probabilities = topk_preservation_targets(logits, top_k=2)
    assert topk_preservation_kl_loss(
        logits, indices, probabilities, mask
    ).item() == pytest.approx(0.0, abs=1e-6)


def test_topk_preservation_detects_probability_escaping_retained_tokens():
    base = torch.tensor([[[10.0, 9.0, 0.0]]])
    student = torch.tensor([[[10.0, 9.0, 100.0]]])
    mask = torch.ones((1, 1), dtype=torch.long)
    indices, probabilities = topk_preservation_targets(base, top_k=2)
    loss = topk_preservation_kl_loss(student, indices, probabilities, mask)
    assert probabilities.shape[-1] == 3
    assert loss.item() > 10


def test_topk_preservation_keeps_required_target_token():
    logits = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    required = torch.tensor([[3]])
    indices, _ = topk_preservation_targets(
        logits,
        top_k=2,
        required_token_ids=required,
    )
    assert 3 in indices[0, 0].tolist()


def test_sequence_log_probability_scores_only_shifted_assistant_tokens():
    logits = torch.tensor(
        [
            [
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 0.0],
            ]
        ]
    )
    labels = torch.tensor([[-100, 1, 2]])
    score = sequence_log_probability(logits, labels)
    expected = torch.log_softmax(torch.tensor([0.0, 5.0, 0.0]), dim=0)[1]
    assert score.item() == pytest.approx(expected.item(), abs=1e-6)


def test_sequence_log_probability_token_averages_unequal_length_completions():
    logits = torch.tensor(
        [
            [[0.0, 4.0], [4.0, 0.0], [4.0, 0.0]],
            [[0.0, 4.0], [0.0, 4.0], [4.0, 0.0]],
        ]
    )
    labels = torch.tensor([[-100, 1, -100], [-100, 1, 1]])
    scores = sequence_log_probability(logits, labels)
    assert scores[0].item() == pytest.approx(scores[1].item(), abs=1e-6)


def test_paired_reference_loss_rewards_improvement_over_the_base_margin():
    unchanged = paired_reference_improvement_loss(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        base_preferred=torch.tensor([2.0]),
        base_alternative=torch.tensor([1.0]),
    )
    improved = paired_reference_improvement_loss(
        torch.tensor([3.0]),
        torch.tensor([1.0]),
        base_preferred=torch.tensor([2.0]),
        base_alternative=torch.tensor([1.0]),
    )
    assert improved.item() < unchanged.item()


def test_causal_preservation_targets_only_score_assistant_next_tokens():
    logits = torch.tensor([[[9.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 1.0, 9.0]]])
    labels = torch.tensor([[-100, 1, 2]])
    indices, probabilities, mask = causal_preservation_targets(logits, labels, top_k=2)
    assert indices.shape == (1, 2, 2)
    assert probabilities.shape == (1, 2, 3)
    assert mask.tolist() == [[True, True]]
    assert 1 in indices[0, 0].tolist()
    assert 2 in indices[0, 1].tolist()


def test_objective_weights_fail_closed():
    with pytest.raises(ValueError, match="non-negative"):
        ObjectiveWeights(behavior_ce=-1).validate()
    with pytest.raises(ValueError, match="At least one"):
        ObjectiveWeights(0, 0, 0).validate()
