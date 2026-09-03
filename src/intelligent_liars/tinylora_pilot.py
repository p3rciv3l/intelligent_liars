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


def assistant_probe_score(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
    intercept: float,
) -> torch.Tensor:
    """Average the affine probe score over labeled assistant tokens only."""
    if hidden.shape[:2] != labels.shape:
        raise ValueError("Hidden states and labels are not token-aligned")
    token_mask = labels != -100
    scores = hidden.float() @ direction.to(hidden.device) + intercept
    denominator = token_mask.sum(dim=1).clamp_min(1)
    return (scores * token_mask).sum(dim=1) / denominator


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_score(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_stratified_rows(
    rows: Iterable[dict[str, Any]], *, per_objective: int, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic equal-sized slice from every objective."""
    if per_objective < 1:
        raise ValueError("per_objective must be positive")
    materialized = list(rows)
    selected: list[dict[str, Any]] = []
    for objective in sorted({str(row["objective"]) for row in materialized}):
        candidates = sorted(
            (row for row in materialized if str(row["objective"]) == objective),
            key=lambda row: stable_score(seed, str(row["record_id"])),
        )
        if len(candidates) < per_objective:
            raise ValueError(
                f"Objective {objective} has {len(candidates)} rows; "
                f"need {per_objective}"
            )
        selected.extend(candidates[:per_objective])
    return selected


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
    return (
        F.relu(torch.as_tensor(desired_delta, device=movement.device) - movement)
        .square()
        .mean()
    )


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


def topk_preservation_targets(
    base_logits: torch.Tensor,
    *,
    top_k: int = 64,
    temperature: float = 1.0,
    required_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress base logits into top-k buckets plus all omitted probability mass."""
    if temperature <= 0:
        raise ValueError("KL temperature must be positive")
    if top_k < 2 or top_k > base_logits.shape[-1]:
        raise ValueError("top_k must be between 2 and the vocabulary size")
    scaled = base_logits / temperature
    _values, indices = torch.topk(scaled, k=top_k, dim=-1)
    if required_token_ids is not None:
        required = required_token_ids.to(device=indices.device, dtype=indices.dtype)
        if required.shape != indices.shape[:-1]:
            raise ValueError(
                "required_token_ids must match the logits token dimensions"
            )
        valid = (required >= 0) & (required < base_logits.shape[-1])
        present = (indices == required.unsqueeze(-1)).any(dim=-1)
        replace = valid & ~present
        indices = indices.clone()
        indices[..., -1] = torch.where(replace, required, indices[..., -1])
    selected = torch.gather(scaled, dim=-1, index=indices).float()
    log_normalizer = torch.logsumexp(scaled, dim=-1, keepdim=True).float()
    selected_probabilities = torch.exp(selected - log_normalizer)
    other = (1.0 - selected_probabilities.sum(dim=-1, keepdim=True)).clamp_min(
        torch.finfo(selected_probabilities.dtype).tiny
    )
    probabilities = torch.cat((selected_probabilities, other), dim=-1)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return indices, probabilities


def topk_preservation_kl_loss(
    student_logits: torch.Tensor,
    base_indices: torch.Tensor,
    base_probabilities: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL over retained tokens plus an aggregate omitted-token bucket."""
    # Apple MPS has no float64 kernels, while this frozen KL computation uses
    # float64 accumulation to keep tiny preservation deltas stable. Copy the
    # already-produced logits and compressed targets to CPU on that explicitly
    # non-production backend; CUDA and CPU production behavior is unchanged.
    if student_logits.device.type == "mps":
        student_logits = student_logits.cpu()
        base_indices = base_indices.cpu()
        base_probabilities = base_probabilities.cpu()
        attention_mask = attention_mask.cpu()
    if temperature <= 0:
        raise ValueError("KL temperature must be positive")
    if base_probabilities.shape[:-1] != base_indices.shape[:-1]:
        raise ValueError("Base probability and index token dimensions differ")
    if base_probabilities.shape[-1] != base_indices.shape[-1] + 1:
        raise ValueError("Base probabilities must include the all-other-tokens bucket")
    # Gather before scaling and reduce the vocabulary in bounded chunks.  A
    # Qwen3-VL record can hold hundreds of MiB of logits; materializing either
    # ``student_logits / temperature`` or a full-vocabulary logsumexp temporary
    # can otherwise exhaust a 24 GiB device even though the final KL is tiny.
    selected_student = torch.gather(
        student_logits,
        dim=-1,
        index=base_indices,
    ).double() / temperature
    outer_elements = max(1, student_logits.numel() // student_logits.shape[-1])
    # Keep half-precision reductions below their finite summation range as well
    # as below the temporary-memory ceiling.
    chunk_width = max(
        1,
        min(student_logits.shape[-1], 32_768, 4_194_304 // outer_elements),
    )
    log_normalizer: torch.Tensor | None = None
    for start in range(0, student_logits.shape[-1], chunk_width):
        # Convert before the reduction.  Casting the BF16/FP16 result after
        # logsumexp is too late: a large common logit offset can round away the
        # normalization term and produce positive log-probabilities (and even
        # a negative mathematical KL).  Float32 also matches the precision of
        # the frozen target distributions; accumulation across chunks remains
        # float64 below.
        chunk = student_logits[..., start : start + chunk_width].float()
        if temperature != 1.0:
            chunk = chunk / temperature
        chunk_normalizer = torch.logsumexp(chunk, dim=-1, keepdim=True).double()
        log_normalizer = (
            chunk_normalizer
            if log_normalizer is None
            else torch.logaddexp(log_normalizer, chunk_normalizer)
        )
    assert log_normalizer is not None
    selected_log_probabilities = selected_student - log_normalizer
    retained_mass = torch.exp(selected_log_probabilities).sum(dim=-1, keepdim=True)
    epsilon = torch.finfo(selected_log_probabilities.dtype).eps
    other_log_probability = torch.log1p(-retained_mass.clamp(max=1.0 - epsilon))
    student_log_probabilities = torch.cat(
        (selected_log_probabilities, other_log_probability),
        dim=-1,
    )
    token_kl = F.kl_div(
        student_log_probabilities,
        base_probabilities.double(),
        reduction="none",
    ).sum(dim=-1)
    weights = attention_mask.to(dtype=token_kl.dtype)
    denominator = weights.sum().clamp_min(1)
    return (token_kl * weights).sum() / denominator * (temperature**2)


def causal_preservation_targets(
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int = 64,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compressed base targets for assistant next-token positions only."""
    if base_logits.shape[:2] != labels.shape:
        raise ValueError("Base logits and labels are not token-aligned")
    if base_logits.shape[1] < 2:
        raise ValueError("Causal preservation requires at least two token positions")
    shifted_labels = labels[:, 1:]
    indices, probabilities = topk_preservation_targets(
        base_logits[:, :-1, :],
        top_k=top_k,
        temperature=temperature,
        required_token_ids=shifted_labels,
    )
    return indices, probabilities, shifted_labels != -100


def sequence_log_probability(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return token-averaged assistant log probability for each sequence."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logits and labels are not token-aligned")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    token_log_probabilities = -F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
        ignore_index=-100,
    ).float()
    token_counts = mask.sum(dim=-1)
    if torch.any(token_counts == 0):
        raise ValueError("Every sequence must contain at least one assistant token")
    return (token_log_probabilities * mask).sum(dim=-1) / token_counts


def paired_reference_improvement_loss(
    preferred_log_probability: torch.Tensor,
    alternative_log_probability: torch.Tensor,
    *,
    base_preferred: torch.Tensor,
    base_alternative: torch.Tensor,
    required_improvement: float = 0.0,
) -> torch.Tensor:
    """Prefer a paired margin improvement relative to the frozen base model."""
    shapes = {
        preferred_log_probability.shape,
        alternative_log_probability.shape,
        base_preferred.shape,
        base_alternative.shape,
    }
    if len(shapes) != 1:
        raise ValueError("Student and base paired log probabilities must match")
    tensors = (
        preferred_log_probability,
        alternative_log_probability,
        base_preferred,
        base_alternative,
    )
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("Paired log probabilities must be finite")
    student_margin = (
        preferred_log_probability.float() - alternative_log_probability.float()
    )
    base_margin = base_preferred.float().detach() - base_alternative.float().detach()
    return F.softplus(required_improvement - (student_margin - base_margin)).mean()


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
        "objectives": dict(
            sorted(Counter(row["objective"] for row in materialized).items())
        ),
        "families": dict(
            sorted(
                Counter(
                    row.get("family", "preservation") for row in materialized
                ).items()
            )
        ),
    }
