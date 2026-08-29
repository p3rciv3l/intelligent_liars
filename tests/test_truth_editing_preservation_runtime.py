from __future__ import annotations

import copy
import gc
import hashlib
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from intelligent_liars.truth_editing_preservation import (
    PRESERVATION_SPEC_FORMAT,
    PRESERVATION_RECEIPT_FORMAT,
    PreservationReceipt,
    PreservationSpec,
    StratumPreservationResult,
    build_cached_baseline,
)
from intelligent_liars.truth_editing_preservation_runtime import (
    EditedPreservationOutput,
    FrozenMediaReference,
    FrozenPreservationInput,
    PreservationRuntimeError,
    PreservationRuntimeReceipt,
    TrialPreservationCollector,
    _hash,
    _preservation_receipt_mapping,
)


def test_frozen_media_serialization_is_cached_without_caching_mutable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "frozen.png"
    content = b"immutable vision fixture"
    media_path.write_bytes(content)
    reference = FrozenMediaReference(
        media_id="image-1",
        media_type="image",
        path=media_path,
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )
    calls = 0
    import intelligent_liars.truth_editing_preservation_runtime as runtime

    original = runtime.base64.b64encode

    def counted(value: bytes) -> bytes:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(runtime.base64, "b64encode", counted)
    first = reference.resolved_content_block()
    second = reference.resolved_content_block()
    assert calls == 1
    assert first == second
    assert first is not second
    first["image"] = "mutated"
    assert reference.resolved_content_block()["image"] == second["image"]
    reference.verify_current()


def _sha(character: str) -> str:
    return character * 64


def _content_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_packet(
    tmp_path: Path,
    *,
    tier: str = "trial",
    invalid_modality_record: str | None = None,
    malformed_trace: bool = False,
    vision_with_trace: bool = False,
    observation_only_trace: bool = False,
) -> tuple[Path, PreservationSpec]:
    records = [
        ("text-1", "text", None),
        ("vision-1", "vision", None),
        ("computer-1", "recorded_computer_use", 70),
        ("text-2", "text", None),
        ("vision-2", "vision", None),
        ("computer-2", "recorded_computer_use", 71),
    ]
    if observation_only_trace:
        records = [
            (record_id, stratum, None if stratum == "recorded_computer_use" else action)
            for record_id, stratum, action in records
        ]
    input_payloads = {
        record_id: {
            "messages": [
                {"role": "user", "content": f"frozen preservation input {record_id}"}
            ],
            "media": [],
        }
        for record_id, *_ in records
    }
    for record_id, stratum, _ in records:
        if stratum == "vision":
            media_path = tmp_path / f"{record_id}.png"
            media_bytes = f"synthetic image {record_id}".encode()
            media_path.write_bytes(media_bytes)
            input_payloads[record_id]["media"] = [
                {
                    "media_id": f"image-{record_id}",
                    "media_type": "image",
                    "path": media_path.name,
                    "sha256": hashlib.sha256(media_bytes).hexdigest(),
                }
            ]
            input_payloads[record_id]["messages"][0]["content"] = [
                {"type": "image", "media_id": f"image-{record_id}"},
                {"type": "text", "text": f"frozen preservation input {record_id}"},
            ]
        if stratum == "recorded_computer_use":
            trace_path = tmp_path / f"{record_id}.trace.json"
            trace_payload = (
                {
                    "format": "recorded_computer_use_trace_v2",
                    "semantics": "observation_instruction_kl_only",
                    "events": [
                        {
                            "sequence_index": 0,
                            "event_type": "observation",
                            "payload": {"screenshot_sha256": "a" * 64},
                        }
                    ],
                }
                if observation_only_trace
                else {
                    "format": "recorded_computer_use_trace_v1",
                    "events": [
                        {
                            "sequence_index": 0,
                            "event_type": "click",
                            "payload": {"x": 10, "y": 20},
                        }
                    ],
                }
            )
            trace_bytes = (
                b"not JSON"
                if malformed_trace and record_id == "computer-1"
                else json.dumps(trace_payload, separators=(",", ":")).encode()
            )
            trace_path.write_bytes(trace_bytes)
            input_payloads[record_id]["media"] = [
                {
                    "media_id": f"trace-{record_id}",
                    "media_type": "recorded_computer_use_trace",
                    "path": trace_path.name,
                    "sha256": hashlib.sha256(trace_bytes).hexdigest(),
                }
            ]
            input_payloads[record_id]["messages"][0]["content"] = [
                {
                    "type": "recorded_computer_use_trace",
                    "media_id": f"trace-{record_id}",
                },
                {"type": "text", "text": f"frozen preservation input {record_id}"},
            ]
    if vision_with_trace:
        trace_path = tmp_path / "vision-1.trace.json"
        trace_bytes = json.dumps(
            {
                "format": "recorded_computer_use_trace_v1",
                "events": [
                    {
                        "sequence_index": 0,
                        "event_type": "click",
                        "payload": {"x": 1, "y": 2},
                    }
                ],
            },
            separators=(",", ":"),
        ).encode()
        trace_path.write_bytes(trace_bytes)
        input_payloads["vision-1"]["media"].append(
            {
                "media_id": "forbidden-vision-trace",
                "media_type": "recorded_computer_use_trace",
                "path": trace_path.name,
                "sha256": hashlib.sha256(trace_bytes).hexdigest(),
            }
        )
        input_payloads["vision-1"]["messages"][0]["content"].insert(
            1,
            {
                "type": "recorded_computer_use_trace",
                "media_id": "forbidden-vision-trace",
            },
        )
    if invalid_modality_record is not None:
        input_payloads[invalid_modality_record]["media"] = []
        input_payloads[invalid_modality_record]["messages"][0]["content"] = "plain text only"
        if invalid_modality_record.startswith("text"):
            media_path = tmp_path / f"{invalid_modality_record}.png"
            media_bytes = b"forbidden text-lane image"
            media_path.write_bytes(media_bytes)
            input_payloads[invalid_modality_record]["media"] = [
                {
                    "media_id": "forbidden-image",
                    "media_type": "image",
                    "path": media_path.name,
                    "sha256": hashlib.sha256(media_bytes).hexdigest(),
                }
            ]
            input_payloads[invalid_modality_record]["messages"][0]["content"] = [
                {"type": "image", "media_id": "forbidden-image"},
                {"type": "text", "text": "plain text lane"},
            ]
    spec_payload = {
        "format": PRESERVATION_SPEC_FORMAT,
        "spec_id": "preservation-runtime-test",
        "base_model_sha256": _sha("1"),
        "tokenizer_sha256": _sha("2"),
        "processor_sha256": _sha("3"),
        "vision_tower_sha256": _sha("4"),
        "top_k": 64,
        "temperature": 1.0,
        "records": [
            {
                "record_id": record_id,
                "stratum": stratum,
                "prompt_sha256": _content_sha(input_payloads[record_id]),
                "direct_target": False,
                "required_action_token_id": action,
            }
            for record_id, stratum, action in records
        ],
        "tiers": {
            "trial": [record_id for record_id, *_ in records[:3]],
            "promoted": [record_id for record_id, *_ in records],
            "finalist": [record_id for record_id, *_ in records],
        },
    }
    spec = PreservationSpec.from_dict(spec_payload)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    cache_entries = []
    for index, record in enumerate(spec.records):
        logits = torch.zeros((1, 3, 80), dtype=torch.float32)
        logits[..., :64] = torch.arange(64, dtype=torch.float32) + index
        labels = torch.tensor([[-100, 3, 4]])
        baseline = build_cached_baseline(spec, record, logits, labels)
        path = tmp_path / f"{record.record_id}.json"
        path.write_text(json.dumps(baseline.to_dict()), encoding="utf-8")
        input_path = tmp_path / f"{record.record_id}.input.json"
        input_path.write_text(json.dumps(input_payloads[record.record_id]), encoding="utf-8")
        cache_entries.append(
            {
                "record_id": record.record_id,
                "path": path.name,
                "cache_sha256": baseline.cache_sha256,
                "input_path": input_path.name,
                "input_sha256": record.prompt_sha256,
            }
        )
    config = {
        "format": "truth_editing_preservation_runtime_config_v1",
        "spec_path": spec_path.name,
        "tier": tier,
        "chat_template_sha256": _sha("b"),
        "base_vision_tower_sha256": spec.vision_tower_sha256,
        "baselines": cache_entries,
    }
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, spec


class FakeBackend:
    identity = {"format": "fake_preservation_backend_v1"}

    def __init__(self, spec: PreservationSpec) -> None:
        self.spec = spec
        self.seen_weights: list[float] = []

    def infer_edited_logits(
        self,
        bundle: Any,
        *,
        record_id: str,
        input_payload: FrozenPreservationInput,
        expected_prompt_sha256: str,
        expected_chat_template_sha256: str,
    ) -> EditedPreservationOutput:
        resolved = input_payload.resolved_messages()
        content = resolved[0]["content"]
        assert (content if isinstance(content, str) else content[-1]["text"]).endswith(
            record_id
        )
        if record_id.startswith("vision"):
            assert input_payload.media[0].path.read_bytes().startswith(b"synthetic image")
            assert content[0]["image"].startswith("data:image/")
        if record_id.startswith("computer"):
            assert '"events"' in content[0]["text"]
        self.seen_weights.append(float(bundle.model.writer.item()))
        index = next(i for i, item in enumerate(self.spec.records) if item.record_id == record_id)
        logits = torch.zeros((1, 3, 80), dtype=torch.float32)
        logits[..., :64] = torch.arange(64, dtype=torch.float32) + index
        return EditedPreservationOutput(
            record_id=record_id,
            prompt_sha256=expected_prompt_sha256,
            chat_template_sha256=expected_chat_template_sha256,
            direct_target=False,
            logits=logits,
        )

    def vision_tower_sha256(self, bundle: Any) -> str:
        del bundle
        return self.spec.vision_tower_sha256


class _WeakModel:
    def __init__(self, value: float) -> None:
        self.writer = torch.tensor(value)


def _batch(spec: PreservationSpec) -> Any:
    return SimpleNamespace(
        batch_sha256=_sha("c"),
        recipe_id="recipe-1",
        model_sha256=spec.base_model_sha256,
        basis_set=SimpleNamespace(basis_set_sha256=_sha("d")),
    )


def test_collector_scores_all_strata_inside_the_active_writer_lease(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path)
    backend = FakeBackend(spec)
    collector = TrialPreservationCollector.from_config(config_path, backend=backend)
    bundle = SimpleNamespace(model=SimpleNamespace(writer=torch.tensor(9.0)))

    evidence = collector.collect(bundle, _batch(spec))
    receipt = PreservationRuntimeReceipt.from_mapping(evidence)

    assert backend.seen_weights == [9.0, 9.0, 9.0]
    assert receipt.tier == "trial"
    assert {item["stratum"] for item in receipt.preservation_receipt["strata"]} == {
        "text",
        "vision",
        "recorded_computer_use",
    }
    assert receipt.preservation_receipt["aggregate_kl"] == pytest.approx(0.0, abs=1e-5)
    assert receipt.recipe_id == "recipe-1"
    assert receipt.model_sha256 == spec.base_model_sha256
    assert receipt.basis_set_sha256 == _sha("d")


def test_receipt_weighted_mean_does_not_overflow_for_finite_kl_values() -> None:
    maximum = sys.float_info.max
    receipt = PreservationReceipt(
        format=PRESERVATION_RECEIPT_FORMAT,
        spec_sha256=_sha("1"),
        edited_model_sha256=_sha("2"),
        tier="trial",
        strata=(
            StratumPreservationResult("text", 1, 1_884, maximum),
            StratumPreservationResult("vision", 1, 15, maximum),
            StratumPreservationResult("recorded_computer_use", 1, 8, maximum),
        ),
        aggregate_kl=maximum,
        vision_tower_byte_identical=True,
        self_sha256=_sha("3"),
    )

    mapping = _preservation_receipt_mapping(receipt)

    assert mapping["aggregate_kl"] == maximum


def test_canonical_json_failure_names_nonfinite_field_without_value() -> None:
    with pytest.raises(
        PreservationRuntimeError,
        match=r"\$\.preservation_receipt\.aggregate_kl \(non-finite float\)",
    ):
        _hash({"preservation_receipt": {"aggregate_kl": float("inf")}})


def test_edited_output_validates_production_shaped_logits_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finite validation must not allocate a full-vocabulary boolean tensor."""

    logits = torch.zeros((1, 33, 131_072), dtype=torch.float16)
    real_isfinite = torch.isfinite
    largest_call = 0

    def bounded_isfinite(value: torch.Tensor) -> torch.Tensor:
        nonlocal largest_call
        largest_call = max(largest_call, value.numel())
        if value.numel() > 4_194_304:
            raise torch.OutOfMemoryError("synthetic full-tensor finiteness mask OOM")
        return real_isfinite(value)

    monkeypatch.setattr(torch, "isfinite", bounded_isfinite)
    output = EditedPreservationOutput(
        record_id="production-shaped",
        prompt_sha256=_sha("a"),
        chat_template_sha256=_sha("b"),
        direct_target=False,
        logits=logits,
    )

    assert output.logits is logits
    assert largest_call <= 4_194_304


def test_collector_releases_each_full_logit_tensor_before_next_inference(
    tmp_path: Path,
) -> None:
    config_path, spec = _write_packet(tmp_path)

    class MemoryBoundBackend(FakeBackend):
        previous: weakref.ReferenceType[torch.Tensor] | None = None

        def infer_edited_logits(self, *args: Any, **kwargs: Any) -> EditedPreservationOutput:
            gc.collect()
            if self.previous is not None and self.previous() is not None:
                raise torch.OutOfMemoryError(
                    "previous full-vocabulary logits survived into next inference"
                )
            output = super().infer_edited_logits(*args, **kwargs)
            self.previous = weakref.ref(output.logits)
            return output

    collector = TrialPreservationCollector.from_config(
        config_path, backend=MemoryBoundBackend(spec)
    )
    evidence = collector.collect(
        SimpleNamespace(model=SimpleNamespace(writer=torch.tensor(1.0))), _batch(spec)
    )

    assert PreservationRuntimeReceipt.from_mapping(evidence).tier == "trial"


def test_collector_emits_base_repeat_receipt_for_threshold_calibration(
    tmp_path: Path,
) -> None:
    config_path, spec = _write_packet(tmp_path)
    backend = FakeBackend(spec)
    collector = TrialPreservationCollector.from_config(config_path, backend=backend)
    bundle = SimpleNamespace(
        model=_WeakModel(0.0),
        verified_snapshot={
            "model_id": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "frozen",
            "model_sha256": spec.base_model_sha256,
            "snapshot_manifest_sha256": _sha("f"),
        },
    )

    receipt = collector.collect_base_repeat(
        bundle,
        repeat_plan_sha256=_sha("e"),
        repeat_index=2,
    )
    unsigned = dict(receipt)
    del unsigned["self_sha256"]

    assert receipt["format"] == "truth_editing_preservation_base_repeat_receipt_v1"
    assert receipt["base_model_sha256"] == spec.base_model_sha256
    assert receipt["preservation_receipt"]["edited_model_sha256"] == spec.base_model_sha256
    assert receipt["repeat_index"] == 2
    assert receipt["self_sha256"] == _content_sha(unsigned)


def test_base_repeat_releases_each_full_logit_tensor_before_next_inference(
    tmp_path: Path,
) -> None:
    config_path, spec = _write_packet(tmp_path)

    class MemoryBoundBackend(FakeBackend):
        previous: weakref.ReferenceType[torch.Tensor] | None = None

        def infer_edited_logits(self, *args: Any, **kwargs: Any) -> EditedPreservationOutput:
            gc.collect()
            if self.previous is not None and self.previous() is not None:
                raise torch.OutOfMemoryError(
                    "previous base-repeat logits survived into next inference"
                )
            output = super().infer_edited_logits(*args, **kwargs)
            self.previous = weakref.ref(output.logits)
            return output

    collector = TrialPreservationCollector.from_config(
        config_path, backend=MemoryBoundBackend(spec)
    )
    bundle = SimpleNamespace(
        model=_WeakModel(0.0),
        verified_snapshot={
            "model_id": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "frozen",
            "model_sha256": spec.base_model_sha256,
            "snapshot_manifest_sha256": _sha("f"),
        },
    )

    receipt = collector.collect_base_repeat(
        bundle, repeat_plan_sha256=_sha("e"), repeat_index=0
    )

    assert receipt["format"] == "truth_editing_preservation_base_repeat_receipt_v1"


def test_base_repeat_rejects_unverified_or_wrong_model_bundle(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path)
    collector = TrialPreservationCollector.from_config(
        config_path, backend=FakeBackend(spec)
    )
    bundle = SimpleNamespace(
        model=_WeakModel(0.0),
        verified_snapshot={
            "model_id": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "frozen",
            "model_sha256": _sha("9"),
            "snapshot_manifest_sha256": _sha("f"),
        },
    )

    with pytest.raises(
        PreservationRuntimeError, match="verified frozen snapshot identity"
    ):
        collector.collect_base_repeat(
            bundle,
            repeat_plan_sha256=_sha("e"),
            repeat_index=0,
        )


def test_tier_selects_exact_nested_record_count(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path, tier="promoted")
    backend = FakeBackend(spec)
    collector = TrialPreservationCollector.from_config(config_path, backend=backend)

    receipt = PreservationRuntimeReceipt.from_mapping(
        collector.collect(SimpleNamespace(model=SimpleNamespace(writer=torch.tensor(1.0))), _batch(spec))
    )

    assert len(backend.seen_weights) == 6
    assert sum(item["record_count"] for item in receipt.preservation_receipt["strata"]) == 6


def test_collector_fails_closed_on_cached_baseline_tampering(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path)
    cache_path = tmp_path / "text-1.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["base_probabilities"][0][0][0] = 0.5
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreservationRuntimeError, match="baseline"):
        TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))


def test_collector_fails_closed_on_missing_or_tampered_media(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path)
    media_path = tmp_path / "vision-1.png"
    media_path.write_bytes(b"substituted image")

    with pytest.raises(PreservationRuntimeError, match="media content hash"):
        TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))

    media_path.unlink()
    with pytest.raises(PreservationRuntimeError, match="missing|unreadable"):
        TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))


def test_collector_fails_closed_on_missing_input_sidecar(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path)
    (tmp_path / "text-1.input.json").unlink()

    with pytest.raises(PreservationRuntimeError, match="missing|unreadable"):
        TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))


def test_collector_rehashes_media_immediately_before_lease_scoped_inference(
    tmp_path: Path,
) -> None:
    config_path, spec = _write_packet(tmp_path)
    backend = FakeBackend(spec)
    collector = TrialPreservationCollector.from_config(config_path, backend=backend)
    (tmp_path / "vision-1.png").write_bytes(b"post-construction substitution")

    with pytest.raises(PreservationRuntimeError, match="changed after loading"):
        collector.collect(
            SimpleNamespace(model=SimpleNamespace(writer=torch.tensor(1.0))),
            _batch(spec),
        )
    assert backend.seen_weights == []


@pytest.mark.parametrize(
    ("record_id", "replacement_media", "error"),
    [
        ("text-1", [], "cannot use media"),
        ("vision-1", [], "requires image or video"),
        ("computer-1", [], "requires an explicit trace"),
    ],
)
def test_collector_enforces_stratum_modality_semantics(
    tmp_path: Path,
    record_id: str,
    replacement_media: list[object],
    error: str,
) -> None:
    del replacement_media
    config_path, spec = _write_packet(
        tmp_path, invalid_modality_record=record_id
    )

    with pytest.raises(PreservationRuntimeError, match=error):
        TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))


def test_vision_lane_rejects_computer_use_trace_even_with_an_image(
    tmp_path: Path,
) -> None:
    config_path, spec = _write_packet(tmp_path, vision_with_trace=True)

    with pytest.raises(PreservationRuntimeError, match="cannot use a computer-use trace"):
        TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))


def test_computer_use_trace_requires_strict_versioned_action_events(
    tmp_path: Path,
) -> None:
    config_path, spec = _write_packet(tmp_path, malformed_trace=True)

    with pytest.raises(PreservationRuntimeError, match="not strict JSON"):
        TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))


def test_collector_accepts_observation_instruction_only_computer_use_kl(
    tmp_path: Path,
) -> None:
    config_path, spec = _write_packet(tmp_path, observation_only_trace=True)

    TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))

    assert all(
        record.required_action_token_id is None
        for record in spec.records
        if record.stratum == "recorded_computer_use"
    )


def test_collector_rejects_unbound_media_placeholders(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = tmp_path / "vision-1.input.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["messages"][0]["content"][0]["media_id"] = "unknown-image"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    changed_hash = _content_sha(payload)
    next(item for item in config["baselines"] if item["record_id"] == "vision-1")[
        "input_sha256"
    ] = changed_hash
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(PreservationRuntimeError, match="input identity|declarations"):
        TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))


def test_collector_rejects_input_substitution_and_direct_targets(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path)

    class BadBackend(FakeBackend):
        def infer_edited_logits(self, *args: Any, **kwargs: Any) -> EditedPreservationOutput:
            output = super().infer_edited_logits(*args, **kwargs)
            return EditedPreservationOutput(
                record_id=output.record_id,
                prompt_sha256=_sha("e"),
                chat_template_sha256=output.chat_template_sha256,
                direct_target=True,
                logits=output.logits,
            )

    collector = TrialPreservationCollector.from_config(config_path, backend=BadBackend(spec))
    with pytest.raises(PreservationRuntimeError, match="direct target|input identity"):
        collector.collect(SimpleNamespace(model=SimpleNamespace(writer=torch.tensor(1.0))), _batch(spec))


def test_runtime_receipt_round_trips_and_rejects_tampering(tmp_path: Path) -> None:
    config_path, spec = _write_packet(tmp_path)
    collector = TrialPreservationCollector.from_config(config_path, backend=FakeBackend(spec))
    evidence = collector.collect(
        SimpleNamespace(model=SimpleNamespace(writer=torch.tensor(1.0))), _batch(spec)
    )

    assert PreservationRuntimeReceipt.from_mapping(evidence).to_mapping() == evidence
    parsed = PreservationRuntimeReceipt.from_mapping(evidence)
    reparsed_input = dict(evidence)
    reparsed_input["preservation_receipt"] = parsed.preservation_receipt
    assert PreservationRuntimeReceipt.from_mapping(reparsed_input).to_mapping() == evidence
    with pytest.raises(TypeError):
        parsed.preservation_receipt["strata"][0]["forward_kl"] = 9.0
    tampered = copy.deepcopy(evidence)
    tampered["preservation_receipt"]["aggregate_kl"] = 99.0
    with pytest.raises(PreservationRuntimeError, match="hash"):
        PreservationRuntimeReceipt.from_mapping(tampered)

    contradictory = copy.deepcopy(evidence)
    contradictory["preservation_receipt"]["aggregate_kl"] = 1.0
    embedded_unsigned = dict(contradictory["preservation_receipt"])
    del embedded_unsigned["self_sha256"]
    contradictory["preservation_receipt"]["self_sha256"] = _content_sha(
        embedded_unsigned
    )
    outer_unsigned = dict(contradictory)
    del outer_unsigned["self_sha256"]
    contradictory["self_sha256"] = _content_sha(outer_unsigned)
    with pytest.raises(PreservationRuntimeError, match="weighted strata"):
        PreservationRuntimeReceipt.from_mapping(contradictory)

    negative = copy.deepcopy(evidence)
    negative["preservation_receipt"]["strata"][0]["forward_kl"] = -0.1
    embedded_unsigned = dict(negative["preservation_receipt"])
    del embedded_unsigned["self_sha256"]
    negative["preservation_receipt"]["self_sha256"] = _content_sha(
        embedded_unsigned
    )
    outer_unsigned = dict(negative)
    del outer_unsigned["self_sha256"]
    negative["self_sha256"] = _content_sha(outer_unsigned)
    with pytest.raises(PreservationRuntimeError, match="non-negative"):
        PreservationRuntimeReceipt.from_mapping(negative)

    wrong_binding = copy.deepcopy(evidence)
    wrong_binding["recipe_id"] = "another-recipe"
    outer_unsigned = dict(wrong_binding)
    del outer_unsigned["self_sha256"]
    wrong_binding["self_sha256"] = _content_sha(outer_unsigned)
    with pytest.raises(PreservationRuntimeError, match="outer trial binding"):
        PreservationRuntimeReceipt.from_mapping(wrong_binding)
