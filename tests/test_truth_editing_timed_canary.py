from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.truth_editing_timed_canary import (
    JudgeCircuitBreaker,
    TimedCanaryConfig,
    TimedCanaryError,
    run_timed_canary,
)


def _sha(char: str) -> str:
    return char * 64


def _raw() -> dict[str, Any]:
    return {
        "format": "truth_editing_timed_canary_config_v1",
        "canary_id": "truth-editing-production-canary-v1",
        "production_config": {
            "path": "configs/truth_editing_production_release_abcd1234.json",
            "sha256": _sha("a"),
        },
        "model_sha256": _sha("b"),
        "workload": {"trial_count": 1, "batch_count": 1},
        "limits": {
            "maximum_wall_seconds": 900,
            "minimum_generated_tokens": 16,
            "maximum_judge_calls": 4,
            "maximum_judge_failures": 1,
            "maximum_judge_cost_usd": 0.25,
            "maximum_persistence_kl": {
                "text": 0.02,
                "vision": 0.03,
                "recorded_computer_use": 0.04,
            },
        },
        "judge": {
            "deployment_alias": "glm-5.3-flash",
            "model": "z-ai/glm-5.3-flash",
            "provider_route": "z-ai/fp8",
            "allow_fallbacks": False,
            "response_healing": True,
        },
    }


def _observation() -> dict[str, Any]:
    return {
        "format": "truth_editing_timed_canary_observation_v2",
        "canary_id": "truth-editing-production-canary-v1",
        "production_config_path": "configs/truth_editing_production_release_abcd1234.json",
        "production_config_sha256": _sha("a"),
        "model_sha256": _sha("b"),
        "actual_model_loaded": True,
        "batch_count": 1,
        "generated_tokens": 120,
        "generation_seconds": 4.0,
        "judge": {
            "deployment_alias": "glm-5.3-flash",
            "model": "z-ai/glm-5.3-flash",
            "provider_route": "z-ai/fp8",
            "fallback_used": False,
            "response_healing_used": True,
            "attempted_calls": 2,
            "successful_calls": 2,
            "failed_calls": 0,
            "cost_usd": 0.01,
            "elapsed_seconds": 0.16,
            "circuit_opened": False,
        },
        "persistence_kl": {
            "text": 0.01,
            "vision": 0.02,
            "recorded_computer_use": 0.03,
        },
        "trials": [
            {
                "trial_id": "trial-canary-0000",
                "outcome_kind": "successful",
                "metrics": {"deception": 0.8, "retained_truth": 0.9},
                "output_sha256": _sha("c"),
            }
        ],
    }


def test_contract_is_exactly_one_trial_one_batch_and_frozen_judge() -> None:
    config = TimedCanaryConfig.from_mapping(_raw())
    assert config.trial_count == config.batch_count == 1

    changed = _raw()
    changed["production_config"]["path"] = "../outside.json"
    with pytest.raises(TimedCanaryError, match="safe repository-relative"):
        TimedCanaryConfig.from_mapping(changed)
    changed = _raw()
    changed["workload"]["trial_count"] = 2
    with pytest.raises(TimedCanaryError, match="exactly one trial and one batch"):
        TimedCanaryConfig.from_mapping(changed)
    changed = _raw()
    changed["judge"]["provider_route"] = "other/provider"
    with pytest.raises(TimedCanaryError, match="frozen GLM-5.3 Flash"):
        TimedCanaryConfig.from_mapping(changed)
    changed = _raw()
    changed["new_field"] = True
    with pytest.raises(TimedCanaryError, match="fields changed"):
        TimedCanaryConfig.from_mapping(changed)


def test_success_receipt_recomputes_tps_cost_and_topology_projection(tmp_path: Path) -> None:
    config = TimedCanaryConfig.from_mapping(_raw())
    target = tmp_path / "canary-receipt.json"
    times = iter((100.0, 110.0))

    receipt = run_timed_canary(
        config=config,
        workload=lambda: _observation(),
        gpu_hourly_usd=0.36,
        receipt_path=target,
        monotonic=lambda: next(times),
    )

    assert receipt["tokens_per_second"] == pytest.approx(30.0)
    assert receipt["measured_wall_seconds"] == pytest.approx(10.0)
    assert receipt["judge_elapsed_seconds"] == pytest.approx(0.16)
    assert receipt["estimated_canary_cost_usd"] == pytest.approx(0.001)
    assert receipt["single_worker_trials_per_hour"] == pytest.approx(360.0)
    assert receipt["single_worker_200_trial_hours"] == pytest.approx(200 / 360)
    assert receipt["single_worker_200_trial_cost_usd"] == pytest.approx(0.2)
    assert json.loads(target.read_text()) == receipt
    assert len(receipt["receipt_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row["judge"].update(attempted_calls=5, successful_calls=5), "judge circuit breaker"),
        (lambda row: row["judge"].update(failed_calls=2, successful_calls=0), "judge circuit breaker"),
        (lambda row: row["judge"].update(cost_usd=0.26), "judge circuit breaker"),
        (lambda row: row["persistence_kl"].pop("vision"), "persistence-KL strata"),
        (lambda row: row["persistence_kl"].update(vision=0.031), "persistence-KL"),
        (lambda row: row.update(actual_model_loaded=False), "actual model"),
        (lambda row: row.update(production_config_path="configs/substituted.json"), "production config identity"),
        (lambda row: row["trials"].append(dict(row["trials"][0])), "exactly one trial"),
    ],
)
def test_canary_fails_closed_without_receipt(
    tmp_path: Path, mutate, message: str
) -> None:
    observation = _observation()
    mutate(observation)
    target = tmp_path / "must-not-exist.json"
    with pytest.raises(TimedCanaryError, match=message):
        run_timed_canary(
            config=TimedCanaryConfig.from_mapping(_raw()),
            workload=lambda: observation,
            gpu_hourly_usd=0.36,
            receipt_path=target,
            monotonic=iter((1.0, 2.0)).__next__,
        )
    assert not target.exists()


def test_wall_clock_limit_is_a_hard_failure(tmp_path: Path) -> None:
    raw = _raw()
    raw["limits"]["maximum_wall_seconds"] = 5
    with pytest.raises(TimedCanaryError, match="wall-clock limit"):
        run_timed_canary(
            config=TimedCanaryConfig.from_mapping(raw),
            workload=lambda: _observation(),
            gpu_hourly_usd=0.36,
            receipt_path=tmp_path / "receipt.json",
            monotonic=iter((1.0, 7.0)).__next__,
        )


def test_canary_rejects_judge_elapsed_time_beyond_measured_trial_wall(
    tmp_path: Path,
) -> None:
    observation = _observation()
    observation["judge"]["elapsed_seconds"] = 10.01
    with pytest.raises(TimedCanaryError, match="judge elapsed time exceeds"):
        run_timed_canary(
            config=TimedCanaryConfig.from_mapping(_raw()),
            workload=lambda: observation,
            gpu_hourly_usd=0.36,
            receipt_path=tmp_path / "receipt.json",
            monotonic=iter((100.0, 110.0)).__next__,
        )


def test_legacy_observation_without_identity_bound_judge_elapsed_fails_closed(
    tmp_path: Path,
) -> None:
    observation = _observation()
    observation["format"] = "truth_editing_timed_canary_observation_v1"
    observation["judge"].pop("elapsed_seconds")
    with pytest.raises(TimedCanaryError, match="observation format changed"):
        run_timed_canary(
            config=TimedCanaryConfig.from_mapping(_raw()),
            workload=lambda: observation,
            gpu_hourly_usd=0.36,
            receipt_path=tmp_path / "receipt.json",
            monotonic=iter((100.0, 110.0)).__next__,
        )


def test_judge_circuit_breaker_stops_calls_at_first_failure_or_budget() -> None:
    config = TimedCanaryConfig.from_mapping(_raw())
    breaker = JudgeCircuitBreaker(config)
    breaker.before_call()
    breaker.record_call(success=False, cost_usd=0.01)
    assert breaker.receipt()["circuit_opened"] is True
    with pytest.raises(TimedCanaryError, match="judge circuit breaker is open"):
        breaker.before_call()

    cost_limited = JudgeCircuitBreaker(config)
    cost_limited.before_call()
    cost_limited.record_call(success=True, cost_usd=0.25)
    with pytest.raises(TimedCanaryError, match="judge circuit breaker is open"):
        cost_limited.before_call()
