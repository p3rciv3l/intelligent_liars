from __future__ import annotations

import copy

import pytest
import torch

from intelligent_liars.truth_editing_preservation import (
    CachedPreservationBaseline,
    PRESERVATION_SPEC_FORMAT,
    PreservationError,
    PreservationSpec,
    VisionTowerIdentityReceipt,
    build_cached_baseline,
    evaluate_preservation,
    evaluate_preservation_stream,
)


def _sha(character: str) -> str:
    return character * 64


def _spec_payload() -> dict[str, object]:
    return {
        "format": PRESERVATION_SPEC_FORMAT,
        "spec_id": "qwen-preservation-v1",
        "base_model_sha256": _sha("1"),
        "tokenizer_sha256": _sha("2"),
        "processor_sha256": _sha("3"),
        "vision_tower_sha256": _sha("4"),
        "top_k": 64,
        "temperature": 1.0,
        "records": [
            {
                "record_id": "text-1",
                "stratum": "text",
                "prompt_sha256": _sha("5"),
                "direct_target": False,
                "required_action_token_id": None,
            },
            {
                "record_id": "vision-1",
                "stratum": "vision",
                "prompt_sha256": _sha("6"),
                "direct_target": False,
                "required_action_token_id": None,
            },
            {
                "record_id": "computer-1",
                "stratum": "recorded_computer_use",
                "prompt_sha256": _sha("7"),
                "direct_target": False,
                "required_action_token_id": 70,
            },
        ],
        "tiers": {
            "trial": ["text-1", "vision-1", "computer-1"],
            "promoted": ["text-1", "vision-1", "computer-1"],
            "finalist": ["text-1", "vision-1", "computer-1"],
        },
    }


def _spec() -> PreservationSpec:
    return PreservationSpec.from_dict(_spec_payload())


def _baseline(spec: PreservationSpec, record_id: str):
    record = next(record for record in spec.records if record.record_id == record_id)
    logits = torch.zeros((1, 3, 80), dtype=torch.float32)
    logits[..., :64] = torch.arange(64, dtype=torch.float32)
    labels = torch.tensor([[-100, 3, 4]])
    return build_cached_baseline(spec, record, logits, labels)


def _vision_receipt(spec: PreservationSpec) -> VisionTowerIdentityReceipt:
    return VisionTowerIdentityReceipt(
        base_model_sha256=spec.base_model_sha256,
        edited_model_sha256=_sha("8"),
        expected_vision_tower_sha256=spec.vision_tower_sha256,
        base_vision_tower_sha256=spec.vision_tower_sha256,
        edited_vision_tower_sha256=spec.vision_tower_sha256,
    )


def test_spec_round_trips_with_stable_identity_and_nested_tiers() -> None:
    spec = _spec()

    assert PreservationSpec.from_dict(spec.to_dict()) == spec
    assert len(spec.self_sha256) == 64
    assert set(spec.records_for_tier("trial")) <= set(
        spec.records_for_tier("promoted")
    )


def test_spec_rejects_direct_truth_editing_targets() -> None:
    payload = _spec_payload()
    payload["records"][0]["direct_target"] = True  # type: ignore[index]

    with pytest.raises(PreservationError, match="direct target"):
        PreservationSpec.from_dict(payload)


def test_cached_baseline_keeps_only_assistant_positions_and_required_action() -> None:
    spec = _spec()
    baseline = _baseline(spec, "computer-1")

    assert baseline.base_indices.shape == (1, 2, 64)
    assert baseline.assistant_mask.tolist() == [[True, True]]
    assert 70 in baseline.base_indices[0, 0].tolist()
    assert 70 in baseline.base_indices[0, 1].tolist()


def test_cached_baseline_round_trips_and_rejects_tampering() -> None:
    baseline = _baseline(_spec(), "text-1")

    assert CachedPreservationBaseline.from_dict(baseline.to_dict()).to_dict() == baseline.to_dict()
    tampered = baseline.to_dict()
    tampered["base_probabilities"][0][0][0] = 0.5  # type: ignore[index]
    with pytest.raises(PreservationError, match="content hash"):
        CachedPreservationBaseline.from_dict(tampered)


def test_cached_baseline_rejects_wrong_spec_identity() -> None:
    spec = _spec()
    baselines = [_baseline(spec, record.record_id) for record in spec.records]
    changed = copy.deepcopy(_spec_payload())
    changed["temperature"] = 0.5
    other_spec = PreservationSpec.from_dict(changed)

    with pytest.raises(PreservationError, match="spec identity"):
        evaluate_preservation(
            other_spec,
            baselines,
            {
                record.record_id: torch.zeros((1, 3, 80))
                for record in other_spec.records
            },
            tier="trial",
            vision_receipt=_vision_receipt(other_spec),
        )


def test_evaluation_reports_forward_kl_separately_for_every_stratum() -> None:
    spec = _spec()
    baselines = [_baseline(spec, record.record_id) for record in spec.records]
    candidates = {
        baseline.record_id: torch.zeros((1, 3, 80), dtype=torch.float32).index_fill(
            -1, torch.arange(64), 0.0
        )
        for baseline in baselines
    }
    # Reconstruct the exact base logits used by _baseline.
    for candidate in candidates.values():
        candidate[..., :64] = torch.arange(64, dtype=torch.float32)

    receipt = evaluate_preservation(
        spec,
        baselines,
        candidates,
        tier="trial",
        vision_receipt=_vision_receipt(spec),
    )

    assert receipt.aggregate_kl == pytest.approx(0.0, abs=1e-6)
    assert {item.stratum for item in receipt.strata} == {
        "text",
        "vision",
        "recorded_computer_use",
    }
    assert all(item.forward_kl == pytest.approx(0.0, abs=1e-6) for item in receipt.strata)


def test_streaming_and_mapping_evaluation_emit_identical_receipts() -> None:
    spec = _spec()
    baselines = [_baseline(spec, record.record_id) for record in spec.records]
    candidates = {
        baseline.record_id: torch.arange(240, dtype=torch.float32).reshape(1, 3, 80)
        for baseline in baselines
    }
    vision = _vision_receipt(spec)

    mapping_receipt = evaluate_preservation(
        spec, baselines, candidates, tier="trial", vision_receipt=vision
    )
    streaming_receipt = evaluate_preservation_stream(
        spec,
        baselines,
        lambda record_id: candidates[record_id],
        tier="trial",
        vision_receipt=vision,
    )

    assert streaming_receipt == mapping_receipt


def test_evaluation_detects_probability_escaping_to_omitted_tokens() -> None:
    spec = _spec()
    baselines = [_baseline(spec, record.record_id) for record in spec.records]
    candidates = {}
    for baseline in baselines:
        logits = torch.zeros((1, 3, 80), dtype=torch.float32)
        logits[..., :64] = torch.arange(64, dtype=torch.float32)
        logits[..., 79] = 100.0
        candidates[baseline.record_id] = logits

    receipt = evaluate_preservation(
        spec,
        baselines,
        candidates,
        tier="trial",
        vision_receipt=_vision_receipt(spec),
    )

    assert receipt.aggregate_kl > 10.0


def test_vision_tower_must_be_byte_identical_to_the_frozen_base() -> None:
    spec = _spec()
    receipt = _vision_receipt(spec)
    bad_receipt = VisionTowerIdentityReceipt(
        base_model_sha256=receipt.base_model_sha256,
        edited_model_sha256=receipt.edited_model_sha256,
        expected_vision_tower_sha256=receipt.expected_vision_tower_sha256,
        base_vision_tower_sha256=receipt.base_vision_tower_sha256,
        edited_vision_tower_sha256=_sha("9"),
    )

    with pytest.raises(PreservationError, match="vision tower"):
        evaluate_preservation(
            spec,
            [_baseline(spec, record.record_id) for record in spec.records],
            {
                record.record_id: torch.zeros((1, 3, 80))
                for record in spec.records
            },
            tier="trial",
            vision_receipt=bad_receipt,
        )


def test_evaluation_fails_closed_when_a_tier_record_is_missing() -> None:
    spec = _spec()

    with pytest.raises(PreservationError, match="baseline records differ"):
        evaluate_preservation(
            spec,
            [_baseline(spec, "text-1")],
            {"text-1": torch.zeros((1, 3, 80))},
            tier="trial",
            vision_receipt=_vision_receipt(spec),
        )
