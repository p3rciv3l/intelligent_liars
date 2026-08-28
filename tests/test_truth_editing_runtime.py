from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_runtime import (
    DEFAULT_RUNTIME_IDENTITY,
    RUNTIME_PLAN_FORMAT,
    RuntimeBudgetError,
    RuntimeIdentity,
    ThroughputBenchmark,
    build_canary_plan,
    build_synchronous_batches,
    estimate_runtime,
    load_runtime_plan,
    validate_runtime_identity,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "truth_editing_runtime_v1.json"


def test_config_is_strictly_pinned_to_the_target_runtime() -> None:
    plan = load_runtime_plan(CONFIG)

    assert plan.format == RUNTIME_PLAN_FORMAT
    assert plan.identity == DEFAULT_RUNTIME_IDENTITY
    assert plan.identity.model_revision == (
        "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
    )
    assert plan.identity.dtype == "bfloat16"
    assert plan.identity.attention_implementation == "flash_attention_2"
    assert plan.identity.quantization is None
    assert plan.identity.persistent_model_per_gpu is True
    assert plan.identity.model_loads_per_worker == 1


def test_runtime_identity_rejects_every_material_drift() -> None:
    fields = {
        "model_id": "other/model",
        "model_revision": "f" * 40,
        "dtype": "float16",
        "attention_implementation": "eager",
        "quantization": "int8",
        "device_map": "auto",
        "local_files_only": False,
        "use_cache": False,
        "persistent_model_per_gpu": False,
        "model_loads_per_worker": 2,
        "workers_per_gpu": 2,
    }

    for field, value in fields.items():
        changed = DEFAULT_RUNTIME_IDENTITY.to_mapping()
        changed[field] = value
        with pytest.raises(ValueError, match="runtime identity"):
            validate_runtime_identity(RuntimeIdentity.from_mapping(changed))


def test_batches_are_deterministic_and_synchronous() -> None:
    plan = load_runtime_plan(CONFIG)

    first = build_synchronous_batches(plan)
    second = build_synchronous_batches(plan)

    assert first == second
    assert [batch.batch_id for batch in first] == list(range(len(first)))
    assert all(batch.trial_indices for batch in first)
    assert all(batch.tier_name for batch in first)
    assert all(len(batch.trial_indices) == len(batch.trial_seeds) for batch in first)
    assert first[0].trial_seeds == second[0].trial_seeds
    assert tuple(first[0].trial_indices) == tuple(range(8))
    assert first[-1].tier_name == "finalists"
    assert len({trial for batch in first for trial in batch.trial_indices}) == sum(
        tier.trial_count for tier in plan.tiers
    )


def test_estimator_returns_a_p90_receipt_under_the_24_hour_cap() -> None:
    plan = load_runtime_plan(CONFIG)
    estimate = estimate_runtime(
        plan,
        ThroughputBenchmark(
            tokens_per_second=149.36,
            p90_slowdown=1.20,
            model_load_seconds=180.0,
            batch_overhead_seconds=2.0,
            gpu_count=8,
        ),
    )

    assert estimate.p90_seconds < 24 * 60 * 60
    assert estimate.total_sequences > 0
    assert estimate.total_tokens > 0
    assert estimate.gpu_count == 8


def test_estimator_fails_closed_when_p90_exceeds_24_hours() -> None:
    plan = load_runtime_plan(CONFIG)

    with pytest.raises(RuntimeBudgetError, match="24-hour"):
        estimate_runtime(
            plan,
            ThroughputBenchmark(
                tokens_per_second=0.01,
                p90_slowdown=2.0,
                model_load_seconds=0.0,
                batch_overhead_seconds=0.0,
                gpu_count=8,
            ),
        )


def test_canary_is_production_parity_and_does_not_execute() -> None:
    plan = load_runtime_plan(CONFIG)
    canary = build_canary_plan(plan)

    assert canary.runtime_identity == plan.identity
    assert canary.sample_count == 16
    assert canary.batch_size == 8
    assert canary.max_new_tokens == 32
    assert canary.execution_mode == "plan_only"
    assert canary.network_access is False
    assert canary.cloud_mutation is False


def test_loader_rejects_wrong_format_and_unknown_keys(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["format"] = "not-the-runtime-schema"
    wrong_format = tmp_path / "wrong-format.json"
    wrong_format.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="format"):
        load_runtime_plan(wrong_format)

    payload = json.loads(CONFIG.read_text())
    payload["unexpected"] = True
    unknown_key = tmp_path / "unknown-key.json"
    unknown_key.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown"):
        load_runtime_plan(unknown_key)


def test_invalid_benchmark_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="tokens_per_second"):
        ThroughputBenchmark(tokens_per_second=0.0)
    with pytest.raises(ValueError, match="p90_slowdown"):
        ThroughputBenchmark(tokens_per_second=1.0, p90_slowdown=0.0)
    with pytest.raises(ValueError, match="gpu_count"):
        ThroughputBenchmark(tokens_per_second=1.0, gpu_count=0)
