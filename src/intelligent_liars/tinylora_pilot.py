"""Immutable data plans and multi-objective loss helpers for TinyLoRA pilots."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

PILOT_SPLIT_COUNTS = {"train": 37, "development": 8, "audit": 8}
BEHAVIOR_OBJECTIVES = {
    "conditioned_false_direct_report",
    "counterfactual_state_action",
    "true_state_action_after_false_report",
    "true_state_indirect_answer_after_false_report",
    "truthful_direct_report",
    "truthful_false_trigger_control",
}
DIRECTIONAL_OBJECTIVE = "conditioned_false_direct_report"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_score(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_group_splits(
    groups: Iterable[str], *, seed: int, counts: dict[str, int] | None = None
) -> dict[str, str]:
    requested = counts or PILOT_SPLIT_COUNTS
    ordered = sorted(set(groups), key=lambda value: (stable_score(seed, value), value))
    if sum(requested.values()) != len(ordered):
        raise ValueError(
            f"Split counts require {sum(requested.values())} groups, found {len(ordered)}"
        )
    assignments: dict[str, str] = {}
    cursor = 0
    for split in ("train", "development", "audit"):
        next_cursor = cursor + requested[split]
        for group in ordered[cursor:next_cursor]:
            assignments[group] = split
        cursor = next_cursor
    return assignments


def load_exclusions(path: Path) -> set[tuple[str, str]]:
    payload = json.loads(path.read_text())
    if payload.get("format") != "tinylora_training_exclusions_v1":
        raise ValueError("Unsupported TinyLoRA exclusion manifest")
    return {
        (str(row["source"]), str(row["record_id"]))
        for row in payload["excluded_records"]
    }


def directional_margin_loss(
    student_scores: torch.Tensor,
    base_scores: torch.Tensor,
    *,
    desired_delta: float,
) -> torch.Tensor:
    """Penalize target-condition probe movement below a calibrated positive delta."""
    movement = student_scores.float() - base_scores.float().detach()
    return F.relu(torch.as_tensor(desired_delta, device=movement.device) - movement).square().mean()


def preservation_kl_loss(
    student_logits: torch.Tensor,
    base_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Token-masked forward KL from the frozen base distribution to the student."""
    if temperature <= 0:
        raise ValueError("KL temperature must be positive")
    base_probabilities = F.softmax(base_logits.float() / temperature, dim=-1)
    student_log_probabilities = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    token_kl = F.kl_div(
        student_log_probabilities,
        base_probabilities,
        reduction="none",
    ).sum(dim=-1)
    weights = attention_mask.to(dtype=token_kl.dtype)
    denominator = weights.sum().clamp_min(1)
    return (token_kl * weights).sum() / denominator * (temperature**2)


@dataclass(frozen=True)
class ObjectiveWeights:
    behavior_ce: float = 1.0
    directional_margin: float = 0.25
    preservation_kl: float = 0.5

    def validate(self) -> None:
        if min(self.behavior_ce, self.directional_margin, self.preservation_kl) < 0:
            raise ValueError("Objective weights must be non-negative")
        if self.behavior_ce == self.directional_margin == self.preservation_kl == 0:
            raise ValueError("At least one objective weight must be positive")


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    return {
        "records": len(materialized),
        "groups": len({str(row["split_group_id"]) for row in materialized}),
        "objectives": dict(sorted(Counter(row["objective"] for row in materialized).items())),
        "families": dict(sorted(Counter(row.get("family", "preservation") for row in materialized).items())),
    }
