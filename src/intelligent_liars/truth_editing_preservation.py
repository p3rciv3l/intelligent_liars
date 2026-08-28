"""Frozen, optimizer-independent capability-preservation evaluation.

The module owns one deep seam: a versioned preservation specification is used
to build identity-bound base-model caches and later score an edited model.  It
does not load either model and has no optimizer, GPU, OSWorld, or network
dependency.  Callers supply logits obtained through their own inference lane.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from intelligent_liars.tinylora_pilot import (
    topk_preservation_kl_loss,
    topk_preservation_targets,
)


PRESERVATION_SPEC_FORMAT: Literal[
    "truth_editing_preservation_spec_v1"
] = "truth_editing_preservation_spec_v1"
PRESERVATION_CACHE_FORMAT: Literal[
    "truth_editing_preservation_cache_v1"
] = "truth_editing_preservation_cache_v1"
PRESERVATION_RECEIPT_FORMAT: Literal[
    "truth_editing_preservation_receipt_v1"
] = "truth_editing_preservation_receipt_v1"

PreservationStratum = Literal["text", "vision", "recorded_computer_use"]
PreservationTier = Literal["trial", "promoted", "finalist"]

_STRATA = ("text", "vision", "recorded_computer_use")
_TIERS = ("trial", "promoted", "finalist")
_HEX = frozenset("0123456789abcdef")


class PreservationError(ValueError):
    """The preservation contract or supplied evidence is not exact."""


def tensor_is_finite_in_chunks(
    tensor: torch.Tensor, *, maximum_chunk_elements: int = 4_194_304
) -> bool:
    """Validate every element without allocating a tensor-sized boolean mask."""

    if (
        isinstance(maximum_chunk_elements, bool)
        or not isinstance(maximum_chunk_elements, int)
        or maximum_chunk_elements <= 0
    ):
        raise ValueError("maximum_chunk_elements must be a positive integer")
    flattened = tensor.reshape(-1)
    for start in range(0, flattened.numel(), maximum_chunk_elements):
        if not torch.isfinite(flattened[start : start + maximum_chunk_elements]).all().item():
            return False
    return True


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PreservationError("value is not canonical JSON") from error
    return rendered.encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreservationError(f"{name} must be an object")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise PreservationError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PreservationError(f"{name} must be a nonempty trimmed string")
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PreservationError(f"{name} must be a lowercase SHA-256")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PreservationError(f"{name} must be an array")
    return value


@dataclass(frozen=True)
class PreservationRecord:
    record_id: str
    stratum: PreservationStratum
    prompt_sha256: str
    direct_target: bool
    required_action_token_id: int | None

    @classmethod
    def from_dict(cls, value: Any) -> PreservationRecord:
        raw = _object(value, "record")
        _exact(
            raw,
            {
                "record_id",
                "stratum",
                "prompt_sha256",
                "direct_target",
                "required_action_token_id",
            },
            "record",
        )
        stratum = _text(raw["stratum"], "record.stratum")
        if stratum not in _STRATA:
            raise PreservationError(f"unknown preservation stratum {stratum!r}")
        direct_target = raw["direct_target"]
        if not isinstance(direct_target, bool):
            raise PreservationError("record.direct_target must be boolean")
        if direct_target:
            raise PreservationError("preservation records cannot be a direct target")
        action = raw["required_action_token_id"]
        if action is not None and (
            isinstance(action, bool) or not isinstance(action, int) or action < 1
        ):
            raise PreservationError(
                "record.required_action_token_id must be null or a positive integer"
            )
        if stratum != "recorded_computer_use" and action is not None:
            raise PreservationError(
                "only recorded computer-use records may require an action token"
            )
        return cls(
            record_id=_text(raw["record_id"], "record.record_id"),
            stratum=stratum,  # type: ignore[arg-type]
            prompt_sha256=_sha(raw["prompt_sha256"], "record.prompt_sha256"),
            direct_target=False,
            required_action_token_id=action,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "stratum": self.stratum,
            "prompt_sha256": self.prompt_sha256,
            "direct_target": self.direct_target,
            "required_action_token_id": self.required_action_token_id,
        }


@dataclass(frozen=True)
class PreservationSpec:
    format: Literal["truth_editing_preservation_spec_v1"]
    spec_id: str
    base_model_sha256: str
    tokenizer_sha256: str
    processor_sha256: str
    vision_tower_sha256: str
    top_k: int
    temperature: float
    records: tuple[PreservationRecord, ...]
    tiers: tuple[tuple[PreservationTier, tuple[str, ...]], ...]
    self_sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> PreservationSpec:
        raw = _object(value, "preservation spec")
        _exact(
            raw,
            {
                "format",
                "spec_id",
                "base_model_sha256",
                "tokenizer_sha256",
                "processor_sha256",
                "vision_tower_sha256",
                "top_k",
                "temperature",
                "records",
                "tiers",
            },
            "preservation spec",
        )
        if raw["format"] != PRESERVATION_SPEC_FORMAT:
            raise PreservationError("unsupported preservation spec format")
        if raw["top_k"] != 64:
            raise PreservationError("preservation top_k must be exactly 64")
        temperature = raw["temperature"]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            raise PreservationError("preservation temperature must be finite and positive")

        records = tuple(
            PreservationRecord.from_dict(item)
            for item in _array(raw["records"], "preservation spec.records")
        )
        if not records:
            raise PreservationError("preservation spec.records must not be empty")
        record_ids = tuple(record.record_id for record in records)
        if len(set(record_ids)) != len(record_ids):
            raise PreservationError("preservation record IDs must be unique")

        tier_raw = _object(raw["tiers"], "preservation spec.tiers")
        _exact(tier_raw, set(_TIERS), "preservation spec.tiers")
        tiers: list[tuple[PreservationTier, tuple[str, ...]]] = []
        tier_sets: dict[str, set[str]] = {}
        for tier in _TIERS:
            ids = tuple(
                _text(item, f"tiers.{tier}[]")
                for item in _array(tier_raw[tier], f"tiers.{tier}")
            )
            if len(set(ids)) != len(ids):
                raise PreservationError(f"tiers.{tier} record IDs must be unique")
            if not ids or not set(ids) <= set(record_ids):
                raise PreservationError(f"tiers.{tier} contains missing or unknown records")
            selected_strata = {
                record.stratum for record in records if record.record_id in set(ids)
            }
            if selected_strata != set(_STRATA):
                raise PreservationError(f"tiers.{tier} must cover every stratum")
            tier_sets[tier] = set(ids)
            tiers.append((tier, ids))  # type: ignore[arg-type]
        if not tier_sets["trial"] <= tier_sets["promoted"] <= tier_sets["finalist"]:
            raise PreservationError(
                "preservation tiers must be nested trial <= promoted <= finalist"
            )

        spec_id = _text(raw["spec_id"], "preservation spec.spec_id")
        base_model_sha256 = _sha(
            raw["base_model_sha256"], "preservation spec.base_model_sha256"
        )
        tokenizer_sha256 = _sha(
            raw["tokenizer_sha256"], "preservation spec.tokenizer_sha256"
        )
        processor_sha256 = _sha(
            raw["processor_sha256"], "preservation spec.processor_sha256"
        )
        vision_tower_sha256 = _sha(
            raw["vision_tower_sha256"], "preservation spec.vision_tower_sha256"
        )
        unsigned = {
            "format": PRESERVATION_SPEC_FORMAT,
            "spec_id": spec_id,
            "base_model_sha256": base_model_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "processor_sha256": processor_sha256,
            "vision_tower_sha256": vision_tower_sha256,
            "top_k": 64,
            "temperature": float(temperature),
            "records": [record.to_dict() for record in records],
            "tiers": {tier: list(ids) for tier, ids in tiers},
        }
        return cls(
            format=PRESERVATION_SPEC_FORMAT,
            spec_id=spec_id,
            base_model_sha256=base_model_sha256,
            tokenizer_sha256=tokenizer_sha256,
            processor_sha256=processor_sha256,
            vision_tower_sha256=vision_tower_sha256,
            top_k=64,
            temperature=float(temperature),
            records=records,
            tiers=tuple(tiers),
            self_sha256=_hash(unsigned),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "spec_id": self.spec_id,
            "base_model_sha256": self.base_model_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "processor_sha256": self.processor_sha256,
            "vision_tower_sha256": self.vision_tower_sha256,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "records": [record.to_dict() for record in self.records],
            "tiers": {tier: list(ids) for tier, ids in self.tiers},
        }

    def records_for_tier(self, tier: PreservationTier) -> tuple[str, ...]:
        try:
            return dict(self.tiers)[tier]
        except KeyError as error:
            raise PreservationError(f"unknown preservation tier {tier!r}") from error


@dataclass(frozen=True)
class CachedPreservationBaseline:
    format: Literal["truth_editing_preservation_cache_v1"]
    spec_sha256: str
    base_model_sha256: str
    tokenizer_sha256: str
    processor_sha256: str
    record_id: str
    prompt_sha256: str
    stratum: PreservationStratum
    base_indices: torch.Tensor
    base_probabilities: torch.Tensor
    assistant_mask: torch.Tensor
    cache_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = _baseline_payload(self)
        payload["cache_sha256"] = self.cache_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> CachedPreservationBaseline:
        raw = _object(value, "preservation cache")
        fields = {
            "format",
            "spec_sha256",
            "base_model_sha256",
            "tokenizer_sha256",
            "processor_sha256",
            "record_id",
            "prompt_sha256",
            "stratum",
            "base_indices",
            "base_probabilities",
            "assistant_mask",
            "cache_sha256",
        }
        _exact(raw, fields, "preservation cache")
        if raw["format"] != PRESERVATION_CACHE_FORMAT:
            raise PreservationError("unsupported preservation cache format")
        claimed = _sha(raw["cache_sha256"], "preservation cache.cache_sha256")
        unsigned = dict(raw)
        del unsigned["cache_sha256"]
        if _hash(unsigned) != claimed:
            raise PreservationError("preservation cache content hash mismatch")
        stratum = _text(raw["stratum"], "preservation cache.stratum")
        if stratum not in _STRATA:
            raise PreservationError("preservation cache has unknown stratum")
        try:
            indices = torch.tensor(raw["base_indices"], dtype=torch.long)
            probabilities = torch.tensor(raw["base_probabilities"], dtype=torch.float32)
            mask = torch.tensor(raw["assistant_mask"], dtype=torch.bool)
        except (TypeError, ValueError) as error:
            raise PreservationError("preservation cache tensors are malformed") from error
        if indices.ndim != 3 or indices.shape[-1] != 64:
            raise PreservationError("preservation cache indices must have top-64 shape")
        if probabilities.shape != (*indices.shape[:-1], 65):
            raise PreservationError("preservation cache probabilities need other bucket")
        if mask.shape != indices.shape[:-1] or not mask.any():
            raise PreservationError("preservation cache assistant mask is invalid")
        if (
            not torch.isfinite(probabilities).all()
            or torch.any(probabilities < 0)
            or not torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones_like(probabilities[..., 0]),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise PreservationError("preservation cache probabilities are invalid")
        return cls(
            format=PRESERVATION_CACHE_FORMAT,
            spec_sha256=_sha(raw["spec_sha256"], "preservation cache.spec_sha256"),
            base_model_sha256=_sha(
                raw["base_model_sha256"], "preservation cache.base_model_sha256"
            ),
            tokenizer_sha256=_sha(
                raw["tokenizer_sha256"], "preservation cache.tokenizer_sha256"
            ),
            processor_sha256=_sha(
                raw["processor_sha256"], "preservation cache.processor_sha256"
            ),
            record_id=_text(raw["record_id"], "preservation cache.record_id"),
            prompt_sha256=_sha(
                raw["prompt_sha256"], "preservation cache.prompt_sha256"
            ),
            stratum=stratum,  # type: ignore[arg-type]
            base_indices=indices,
            base_probabilities=probabilities,
            assistant_mask=mask,
            cache_sha256=claimed,
        )


@dataclass(frozen=True)
class VisionTowerIdentityReceipt:
    base_model_sha256: str
    edited_model_sha256: str
    expected_vision_tower_sha256: str
    base_vision_tower_sha256: str
    edited_vision_tower_sha256: str

    def validate(self, spec: PreservationSpec) -> None:
        hashes = {
            "base_model_sha256": self.base_model_sha256,
            "edited_model_sha256": self.edited_model_sha256,
            "expected_vision_tower_sha256": self.expected_vision_tower_sha256,
            "base_vision_tower_sha256": self.base_vision_tower_sha256,
            "edited_vision_tower_sha256": self.edited_vision_tower_sha256,
        }
        for name, value in hashes.items():
            _sha(value, f"vision receipt.{name}")
        if self.base_model_sha256 != spec.base_model_sha256:
            raise PreservationError("vision receipt base model identity differs")
        expected = spec.vision_tower_sha256
        if not (
            self.expected_vision_tower_sha256
            == self.base_vision_tower_sha256
            == self.edited_vision_tower_sha256
            == expected
        ):
            raise PreservationError("edited vision tower is not byte-identical")


@dataclass(frozen=True)
class StratumPreservationResult:
    stratum: PreservationStratum
    record_count: int
    assistant_token_count: int
    forward_kl: float


@dataclass(frozen=True)
class PreservationReceipt:
    format: Literal["truth_editing_preservation_receipt_v1"]
    spec_sha256: str
    edited_model_sha256: str
    tier: PreservationTier
    strata: tuple[StratumPreservationResult, ...]
    aggregate_kl: float
    vision_tower_byte_identical: bool
    self_sha256: str


def _tensor_payload(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _baseline_payload(baseline: CachedPreservationBaseline) -> dict[str, Any]:
    return {
        "format": baseline.format,
        "spec_sha256": baseline.spec_sha256,
        "base_model_sha256": baseline.base_model_sha256,
        "tokenizer_sha256": baseline.tokenizer_sha256,
        "processor_sha256": baseline.processor_sha256,
        "record_id": baseline.record_id,
        "prompt_sha256": baseline.prompt_sha256,
        "stratum": baseline.stratum,
        "base_indices": _tensor_payload(baseline.base_indices),
        "base_probabilities": _tensor_payload(baseline.base_probabilities),
        "assistant_mask": _tensor_payload(baseline.assistant_mask),
    }


def build_cached_baseline(
    spec: PreservationSpec,
    record: PreservationRecord,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
) -> CachedPreservationBaseline:
    """Compress frozen base logits at assistant next-token positions."""

    matching = {item.record_id: item for item in spec.records}.get(record.record_id)
    if matching != record:
        raise PreservationError("record is not exactly present in preservation spec")
    if base_logits.ndim != 3 or labels.ndim != 2 or base_logits.shape[:2] != labels.shape:
        raise PreservationError("base logits and labels must be token-aligned")
    if base_logits.shape[1] < 2:
        raise PreservationError("preservation records require two token positions")
    if base_logits.shape[-1] <= spec.top_k:
        raise PreservationError("vocabulary must be larger than top_k")
    if not torch.isfinite(base_logits).all():
        raise PreservationError("base logits must be finite")

    shifted_labels = labels[:, 1:]
    assistant_mask = shifted_labels != -100
    if not assistant_mask.any():
        raise PreservationError("record has no assistant positions")
    required = shifted_labels
    if record.required_action_token_id is not None:
        if record.required_action_token_id >= base_logits.shape[-1]:
            raise PreservationError("required action token is outside the vocabulary")
        required = torch.full_like(shifted_labels, record.required_action_token_id)
    indices, probabilities = topk_preservation_targets(
        base_logits[:, :-1, :],
        top_k=spec.top_k,
        temperature=spec.temperature,
        required_token_ids=required,
    )
    payload = {
        "format": PRESERVATION_CACHE_FORMAT,
        "spec_sha256": spec.self_sha256,
        "base_model_sha256": spec.base_model_sha256,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "processor_sha256": spec.processor_sha256,
        "record_id": record.record_id,
        "prompt_sha256": record.prompt_sha256,
        "stratum": record.stratum,
        "base_indices": _tensor_payload(indices),
        "base_probabilities": _tensor_payload(probabilities),
        "assistant_mask": _tensor_payload(assistant_mask),
    }
    return CachedPreservationBaseline(
        format=PRESERVATION_CACHE_FORMAT,
        spec_sha256=spec.self_sha256,
        base_model_sha256=spec.base_model_sha256,
        tokenizer_sha256=spec.tokenizer_sha256,
        processor_sha256=spec.processor_sha256,
        record_id=record.record_id,
        prompt_sha256=record.prompt_sha256,
        stratum=record.stratum,
        base_indices=indices.detach().cpu(),
        base_probabilities=probabilities.detach().cpu(),
        assistant_mask=assistant_mask.detach().cpu(),
        cache_sha256=_hash(payload),
    )


def evaluate_preservation(
    spec: PreservationSpec,
    baselines: Sequence[CachedPreservationBaseline],
    edited_logits: Mapping[str, torch.Tensor],
    *,
    tier: PreservationTier,
    vision_receipt: VisionTowerIdentityReceipt,
) -> PreservationReceipt:
    """Score one edited checkpoint against exact cached base distributions."""

    required_ids = spec.records_for_tier(tier)
    observed_ids = tuple(baseline.record_id for baseline in baselines)
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != set(required_ids):
        raise PreservationError("baseline records differ from the requested tier")
    if set(edited_logits) != set(required_ids):
        raise PreservationError("edited-logit records differ from the requested tier")
    return evaluate_preservation_stream(
        spec,
        baselines,
        lambda record_id: edited_logits[record_id],
        tier=tier,
        vision_receipt=vision_receipt,
    )


def evaluate_preservation_stream(
    spec: PreservationSpec,
    baselines: Sequence[CachedPreservationBaseline],
    logits_for_record: Callable[[str], torch.Tensor],
    *,
    tier: PreservationTier,
    vision_receipt: VisionTowerIdentityReceipt,
) -> PreservationReceipt:
    """Score records one at a time so full-vocabulary logits are never accumulated."""

    required_ids = spec.records_for_tier(tier)
    observed_ids = tuple(baseline.record_id for baseline in baselines)
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != set(required_ids):
        raise PreservationError("baseline records differ from the requested tier")
    vision_receipt.validate(spec)
    records = {record.record_id: record for record in spec.records}
    by_stratum: dict[str, list[tuple[float, int]]] = {name: [] for name in _STRATA}

    for baseline in baselines:
        record = records[baseline.record_id]
        expected_identity = (
            spec.self_sha256,
            spec.base_model_sha256,
            spec.tokenizer_sha256,
            spec.processor_sha256,
            record.prompt_sha256,
            record.stratum,
        )
        observed_identity = (
            baseline.spec_sha256,
            baseline.base_model_sha256,
            baseline.tokenizer_sha256,
            baseline.processor_sha256,
            baseline.prompt_sha256,
            baseline.stratum,
        )
        if observed_identity != expected_identity:
            raise PreservationError("cached baseline spec identity differs")
        if _hash(_baseline_payload(baseline)) != baseline.cache_sha256:
            raise PreservationError("cached baseline content hash differs")
        candidate = logits_for_record(baseline.record_id)
        if not isinstance(candidate, torch.Tensor):
            raise PreservationError("edited logits must be tensors")
        if candidate.ndim != 3 or candidate.shape[1] < 2:
            raise PreservationError("edited logits must have causal sequence dimensions")
        shifted = candidate[:, :-1, :]
        if shifted.shape[:-1] != baseline.base_indices.shape[:-1]:
            raise PreservationError("edited logits do not align with cached baseline")
        if (
            baseline.base_indices.shape[-1] != spec.top_k
            or baseline.base_probabilities.shape
            != (*baseline.base_indices.shape[:-1], spec.top_k + 1)
            or baseline.assistant_mask.shape != baseline.base_indices.shape[:-1]
            or not baseline.assistant_mask.any()
        ):
            raise PreservationError("cached baseline tensor shapes are invalid")
        if (
            torch.any(baseline.base_indices < 0)
            or torch.any(baseline.base_indices >= candidate.shape[-1])
        ):
            raise PreservationError("cached token index is outside edited vocabulary")
        if record.required_action_token_id is not None:
            action_present = (
                baseline.base_indices == record.required_action_token_id
            ).any(dim=-1)
            if not torch.all(action_present[baseline.assistant_mask]):
                raise PreservationError(
                    "cached computer-use baseline omits required action token"
                )
        if (
            not torch.isfinite(baseline.base_probabilities).all()
            or torch.any(baseline.base_probabilities < 0)
            or not torch.allclose(
                baseline.base_probabilities.sum(dim=-1),
                torch.ones_like(baseline.base_probabilities[..., 0]),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise PreservationError("cached baseline probabilities are invalid")
        if not tensor_is_finite_in_chunks(candidate):
            raise PreservationError("edited logits must be finite")
        loss = topk_preservation_kl_loss(
            shifted,
            baseline.base_indices.to(candidate.device),
            baseline.base_probabilities.to(candidate.device),
            baseline.assistant_mask.to(candidate.device),
            temperature=spec.temperature,
        )
        token_count = int(baseline.assistant_mask.sum().item())
        by_stratum[record.stratum].append((float(loss.item()), token_count))
        del loss, shifted, candidate

    results: list[StratumPreservationResult] = []
    total_weighted = 0.0
    total_tokens = 0
    for stratum in _STRATA:
        values = by_stratum[stratum]
        token_count = sum(tokens for _, tokens in values)
        weighted = sum(loss * tokens for loss, tokens in values)
        forward_kl = weighted / token_count
        results.append(
            StratumPreservationResult(
                stratum=stratum,  # type: ignore[arg-type]
                record_count=len(values),
                assistant_token_count=token_count,
                forward_kl=forward_kl,
            )
        )
        total_weighted += weighted
        total_tokens += token_count
    aggregate = total_weighted / total_tokens
    unsigned = {
        "format": PRESERVATION_RECEIPT_FORMAT,
        "spec_sha256": spec.self_sha256,
        "edited_model_sha256": vision_receipt.edited_model_sha256,
        "tier": tier,
        "strata": [
            {
                "stratum": result.stratum,
                "record_count": result.record_count,
                "assistant_token_count": result.assistant_token_count,
                "forward_kl": result.forward_kl,
            }
            for result in results
        ],
        "aggregate_kl": aggregate,
        "vision_tower_byte_identical": True,
    }
    return PreservationReceipt(
        format=PRESERVATION_RECEIPT_FORMAT,
        spec_sha256=spec.self_sha256,
        edited_model_sha256=vision_receipt.edited_model_sha256,
        tier=tier,
        strata=tuple(results),
        aggregate_kl=aggregate,
        vision_tower_byte_identical=True,
        self_sha256=_hash(unsigned),
    )
