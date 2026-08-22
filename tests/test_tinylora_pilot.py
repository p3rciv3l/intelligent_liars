from __future__ import annotations

import pytest
import torch

from intelligent_liars.tinylora_pilot import (
    ObjectiveWeights,
    assign_group_splits,
    directional_margin_loss,
    preservation_kl_loss,
)


def test_assistant_probe_score_uses_only_labeled_tokens():
    from scripts.run_tinylora_bounded_pilot import assistant_probe_score

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


def test_directional_margin_only_penalizes_insufficient_positive_movement():
    base = torch.tensor([0.0, 1.0])
    successful = torch.tensor([0.5, 1.5])
    stalled = torch.tensor([0.0, 1.25])
    assert directional_margin_loss(successful, base, desired_delta=0.5).item() == 0
    assert directional_margin_loss(stalled, base, desired_delta=0.5).item() > 0


def test_preservation_kl_is_zero_for_identical_logits_and_masked():
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[1, 0]])
    assert preservation_kl_loss(logits, logits, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_objective_weights_fail_closed():
    with pytest.raises(ValueError, match="non-negative"):
        ObjectiveWeights(behavior_ce=-1).validate()
    with pytest.raises(ValueError, match="At least one"):
        ObjectiveWeights(0, 0, 0).validate()
