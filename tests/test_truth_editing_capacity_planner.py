from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_capacity import (
    CapacityPlanningError,
    CapacityPolicy,
    build_capacity_receipt,
    create_capacity_measurement,
    load_capacity_measurement,
    reforecast_capacity_receipt,
    select_next_batch_projection,
    validate_capacity_receipt,
    write_capacity_receipt,
)
from intelligent_liars.truth_editing_contracts import canonical_sha256


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
SHA = "a" * 64


def _self_hashed(value: dict) -> dict:
    return {**value, "self_sha256": canonical_sha256(value)}


def measurement(**changes: object) -> dict:
    raw = {
        "format": "truth_editing_capacity_measurement_v1",
        "measurement_id": "timed-canary-v4-r10-r6",
        "observed_at": "2026-08-28T11:30:00Z",
        "timed_canary_receipt_sha256": SHA,
        "generated_tokens": 120,
        "tokens_per_second": 30.0,
        "trial_wall_seconds": 50.0,
        "judge_latency_seconds": 40.0,
        "judge_cost_usd_per_trial": "0.0001",
        "per_gpu_hourly_usd": "0.05",
        "projected_storage_network_usd": "0.2",
        "spend": {
            "actual_total_usd": "1",
            "actual_infrastructure_usd": "0.9",
            "actual_evaluation_usd": "0.1",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    }
    raw.update(changes)
    return _self_hashed(raw)


def test_capacity_plan_uses_complete_batches_and_can_expand_to_800() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(
            Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_text()
        )
    )

    receipt = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )

    assert receipt["decision"] == {
        "batch_size": 8,
        "minimum_trials": 200,
        "maximum_trials": 800,
        "planned_trial_limit": 752,
        "planned_batch_limit": 94,
        "minimum_trial_guarantee_met": True,
        "search_seconds": 75600,
        "finalization_seconds_reserved": 10800,
        "projection_role": "advisory_reforecast_target",
        "reforecast_after_each_batch": True,
        "stop_conditions": [
            "search_deadline_reached",
            "total_budget_reserve_reached",
            "evaluation_budget_reserve_reached",
            "maximum_trials_reached",
        ],
    }
    assert receipt["next_batch_projection"]["batch_duration_seconds_upper_bound"] > 0
    assert receipt["source_batch_observation_sha256"] is None
    assert receipt["source_judge_ledger_after_receipt_sha256"] is None
    assert receipt["completed_through_trial"] == 0
    assert receipt["next_batch_projection"]["batch_total_cost_usd_upper_bound"] != "0"
    assert receipt["capacity_limits"]["time_limited_trials"] == 752
    assert receipt["capacity_limits"]["total_budget_limited_trials"] == 800
    assert receipt["capacity_limits"]["evaluation_budget_limited_trials"] == 800
    assert receipt["budget"]["reserved_evaluation_usd"] == "1"
    assert receipt["budget"]["remaining_infrastructure_usd"] == "44.1"
    assert receipt["budget"]["remaining_all_in_usd"] == "49"
    assert receipt["budget"]["reserved_finalization_infrastructure_usd"] == "1.4"
    assert receipt["budget"]["search_infrastructure_usd"] == "42.7"
    assert validate_capacity_receipt(receipt) == receipt


def test_each_projection_tier_exposes_validated_next_batch_upper_bounds() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    receipt = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )

    tiers = receipt["conservative_projection"]["tiers"]
    assert [tier["name"] for tier in tiers] == ["discovery", "expanded", "concentrated"]
    assert [tier["through_trial"] for tier in tiers] == [80, 200, 800]
    for tier in tiers:
        assert tier["batch_size"] == 8
        assert tier["batch_duration_seconds_upper_bound"] == tier["trial_wall_seconds"]
        assert Decimal(tier["batch_infrastructure_cost_usd_upper_bound"]) == (
            Decimal(tier["gpu_cost_usd_per_trial"]) * 8
        )
        assert Decimal(tier["batch_evaluation_cost_usd_upper_bound"]) == (
            Decimal(tier["judge_cost_usd_per_trial"]) * 8
        )
        assert Decimal(tier["batch_total_cost_usd_upper_bound"]) == (
            Decimal(tier["batch_infrastructure_cost_usd_upper_bound"])
            + Decimal(tier["batch_evaluation_cost_usd_upper_bound"])
        )

    assert select_next_batch_projection(receipt, next_completed_trials=80)["tier"] == "discovery"
    assert select_next_batch_projection(receipt, next_completed_trials=88)["tier"] == "expanded"
    assert select_next_batch_projection(receipt, next_completed_trials=200)["tier"] == "expanded"
    assert select_next_batch_projection(receipt, next_completed_trials=208)["tier"] == "concentrated"


def test_rolling_measurement_reforecast_changes_projection_without_hard_stopping_at_initial_target() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    initial = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(
            measurement(trial_wall_seconds=600.0, tokens_per_second=30.0), now=NOW
        ),
        planned_at=NOW,
    )
    rolling = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(
            measurement(trial_wall_seconds=600.0, tokens_per_second=3.0), now=NOW
        ),
        planned_at=NOW,
    )

    assert initial["decision"]["projection_role"] == "advisory_reforecast_target"
    assert initial["decision"]["reforecast_after_each_batch"] is True
    assert rolling["decision"]["projection_role"] == "advisory_reforecast_target"
    assert select_next_batch_projection(
        rolling, next_completed_trials=208
    )["batch_duration_seconds_upper_bound"] > select_next_batch_projection(
        initial, next_completed_trials=208
    )["batch_duration_seconds_upper_bound"]


def test_signed_rolling_batch_observation_reforecasts_the_next_batch() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    initial = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )
    observation = _self_hashed({
        "format": "truth_editing_capacity_batch_observation_v2",
        "observation_id": "batch-0005",
        "observed_at": "2026-08-28T11:59:00Z",
        "timed_canary_receipt_sha256": SHA,
        "completed_through_trial": 40,
        "batch_size": 8,
        "generated_tokens_per_trial_upper_bound": 120,
        "generation_seconds_per_trial_upper_bound": 40.0,
        "trial_wall_seconds_upper_bound": 90.0,
        "judge_elapsed_seconds_per_trial_upper_bound": 45.0,
        "judge_cost_usd_per_trial_upper_bound": "0.0002",
        "judge_ledger_before_receipt_sha256": "b" * 64,
        "judge_ledger_after_receipt_sha256": "c" * 64,
        "judge_calls": 8,
        "judge_failures": 0,
        "judge_elapsed_seconds_total": 360.0,
        "judge_cost_usd_total": "0.0016",
        "spend": {
            "actual_total_usd": "1.0016",
            "actual_infrastructure_usd": "0.9",
            "actual_evaluation_usd": "0.1016",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    })

    rolling = reforecast_capacity_receipt(
        policy=policy,
        previous_receipt=initial,
        batch_observation=observation,
        planned_at=NOW,
    )

    assert rolling["timed_canary_receipt_sha256"] == initial["timed_canary_receipt_sha256"]
    assert rolling["policy_sha256"] == initial["policy_sha256"]
    assert rolling["measurement_sha256"] != initial["measurement_sha256"]
    assert rolling["source_batch_observation_sha256"] == observation["self_sha256"]
    assert rolling["source_judge_ledger_after_receipt_sha256"] == "c" * 64
    assert rolling["completed_through_trial"] == 40
    assert rolling["measured"]["judge_cost_usd_per_trial"] == "0.0002"
    assert select_next_batch_projection(
        rolling, next_completed_trials=48
    )["batch_duration_seconds_upper_bound"] > select_next_batch_projection(
        initial, next_completed_trials=48
    )["batch_duration_seconds_upper_bound"]


def test_expanded_tier_observation_is_not_multiplied_by_expanded_tier_twice() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    initial = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )
    observation = _self_hashed({
        "format": "truth_editing_capacity_batch_observation_v2",
        "observation_id": "batch-0011-expanded",
        "observed_at": "2026-08-28T11:59:00Z",
        "timed_canary_receipt_sha256": SHA,
        "completed_through_trial": 88,
        "batch_size": 8,
        "generated_tokens_per_trial_upper_bound": 360,
        "generation_seconds_per_trial_upper_bound": 120.0,
        # 120s generation plus two 135s judge waves plus 5s orchestration.
        "trial_wall_seconds_upper_bound": 395.0,
        "judge_elapsed_seconds_per_trial_upper_bound": 135.0,
        "judge_cost_usd_per_trial_upper_bound": "0.0006",
        "judge_ledger_before_receipt_sha256": "b" * 64,
        "judge_ledger_after_receipt_sha256": "c" * 64,
        "judge_calls": 24,
        "judge_failures": 0,
        "judge_elapsed_seconds_total": 1080.0,
        "judge_cost_usd_total": "0.0048",
        "spend": {
            "actual_total_usd": "1.0048",
            "actual_infrastructure_usd": "0.9",
            "actual_evaluation_usd": "0.1048",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    })

    rolling = reforecast_capacity_receipt(
        policy=policy,
        previous_receipt=initial,
        batch_observation=observation,
        planned_at=NOW,
    )

    assert rolling["measured"]["judge_latency_seconds"] == 45.0
    assert rolling["measured"]["judge_cost_usd_per_trial"] == "0.0002"
    assert rolling["measured"]["generated_tokens"] == 120
    assert rolling["measured"]["tokens_per_second"] == 3.0
    assert rolling["conservative_projection"]["generation_seconds_from_measured_tps"] == 40.0
    assert rolling["conservative_projection"]["fixed_seconds"] == pytest.approx(5.0)
    expanded = select_next_batch_projection(rolling, next_completed_trials=96)
    assert expanded["batch_duration_seconds_upper_bound"] == pytest.approx(493.75)
    assert expanded["batch_evaluation_cost_usd_upper_bound"] == "0.006"


def test_trial104_live_scale_keeps_serialized_judge_work_out_of_fixed_overhead() -> None:
    """The signed batch ledger must participate in the timing decomposition."""
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    initial = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )
    raw_observation = {
        "format": "truth_editing_capacity_batch_observation_v2",
        "observation_id": "trial104-live-scale",
        "observed_at": "2026-08-28T11:59:00Z",
        "timed_canary_receipt_sha256": SHA,
        "completed_through_trial": 104,
        "batch_size": 8,
        "generated_tokens_per_trial_upper_bound": 360,
        "generation_seconds_per_trial_upper_bound": 120.0,
        # 120s generation plus two 135s judge waves plus 5s orchestration.
        "trial_wall_seconds_upper_bound": 395.0,
        "judge_elapsed_seconds_per_trial_upper_bound": 135.0,
        "judge_cost_usd_per_trial_upper_bound": "0.0006",
        "judge_ledger_before_receipt_sha256": "b" * 64,
        "judge_ledger_after_receipt_sha256": "c" * 64,
        "judge_calls": 24,
        "judge_failures": 0,
        "judge_elapsed_seconds_total": 1080.0,
        "judge_cost_usd_total": "0.0048",
        "spend": {
            "actual_total_usd": "1.0048",
            "actual_infrastructure_usd": "0.9",
            "actual_evaluation_usd": "0.1048",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    }
    observation = _self_hashed(raw_observation)
    rolling = reforecast_capacity_receipt(
        policy=policy,
        previous_receipt=initial,
        batch_observation=observation,
        planned_at=NOW,
    )

    # 120s generation + 135s of judge work leaves only 5s of genuine fixed
    # batch overhead; serialized judge calls must not be counted twice.
    assert rolling["completed_through_trial"] == 104
    assert rolling["conservative_projection"]["fixed_seconds"] == pytest.approx(5.0)
    assert rolling["measured"]["judge_latency_seconds"] == pytest.approx(45.0)
    assert rolling["source_batch_observation_sha256"] == observation["self_sha256"]
    assert validate_capacity_receipt(rolling) == rolling


def test_late_rolling_reforecast_does_not_reapply_the_200_trial_floor() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    initial = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )
    late_spend = {
        "actual_total_usd": "13.99",
        "actual_infrastructure_usd": "10",
        "actual_evaluation_usd": "3.99",
        "pending_infrastructure_usd": "0",
        "pending_evaluation_usd": "0",
    }
    observation = _self_hashed({
        "format": "truth_editing_capacity_batch_observation_v2",
        "observation_id": "batch-0050",
        "observed_at": "2026-08-28T11:59:00Z",
        "timed_canary_receipt_sha256": SHA,
        "completed_through_trial": 400,
        "batch_size": 8,
        "generated_tokens_per_trial_upper_bound": 120,
        "generation_seconds_per_trial_upper_bound": 40.0,
        "trial_wall_seconds_upper_bound": 90.0,
        "judge_elapsed_seconds_per_trial_upper_bound": 45.0,
        "judge_cost_usd_per_trial_upper_bound": "0.48625",
        "judge_ledger_before_receipt_sha256": "b" * 64,
        "judge_ledger_after_receipt_sha256": "c" * 64,
        "judge_calls": 8,
        "judge_failures": 0,
        "judge_elapsed_seconds_total": 360.0,
        "judge_cost_usd_total": "3.89",
        "spend": late_spend,
    })

    rolling = reforecast_capacity_receipt(
        policy=policy,
        previous_receipt=initial,
        batch_observation=observation,
        planned_at=NOW,
    )

    assert 400 <= rolling["decision"]["planned_trial_limit"] < 800
    assert rolling["decision"]["planned_trial_limit"] % 8 == 0


def test_rolling_reforecast_uses_only_time_remaining_before_absolute_deadline() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    loaded = load_capacity_measurement(measurement(), now=NOW)

    receipt = build_capacity_receipt(
        policy=policy,
        measurement=loaded,
        planned_at=NOW,
        completed_trials=400,
        source_batch_observation_sha256="b" * 64,
        source_judge_ledger_after_receipt_sha256="c" * 64,
        remaining_search_seconds=1.0,
    )

    assert receipt["capacity_limits"]["time_limited_trials"] == 400
    assert receipt["decision"]["planned_trial_limit"] == 400


@pytest.mark.parametrize(
    ("completed", "evaluation_spend", "actual_total", "judge_cost", "expected_target"),
    [
        (0, "0.1", "1", "0.0001", 752),
        (192, "3.95", "4.85", "0.001", 200),
        (200, "4.99", "5.89", "0.001", 200),
        (792, "0.1", "1", "0.0001", 800),
    ],
)
def test_reforecast_capacity_respects_absolute_completed_boundaries(
    completed: int, evaluation_spend: str, actual_total: str,
    judge_cost: str, expected_target: int,
) -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    raw = measurement(judge_cost_usd_per_trial=judge_cost, spend={
        "actual_total_usd": actual_total,
        "actual_infrastructure_usd": "0.9",
        "actual_evaluation_usd": evaluation_spend,
        "pending_infrastructure_usd": "0",
        "pending_evaluation_usd": "0",
    })

    receipt = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(raw, now=NOW),
        planned_at=NOW,
        completed_trials=completed,
        source_batch_observation_sha256=("b" * 64 if completed else None),
        source_judge_ledger_after_receipt_sha256=("c" * 64 if completed else None),
    )

    assert receipt["completed_through_trial"] == completed
    assert receipt["decision"]["planned_trial_limit"] == expected_target


def test_before_minimum_trials_fixed_one_dollar_judge_reserve_fails_closed() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    raw = measurement(judge_cost_usd_per_trial="0.001", spend={
        "actual_total_usd": "5.85",
        "actual_infrastructure_usd": "0.9",
        "actual_evaluation_usd": "4.95",
        "pending_infrastructure_usd": "0",
        "pending_evaluation_usd": "0",
    })

    with pytest.raises(CapacityPlanningError, match="cannot guarantee 200 trials"):
        build_capacity_receipt(
            policy=policy,
            measurement=load_capacity_measurement(raw, now=NOW),
            planned_at=NOW,
            completed_trials=192,
            source_batch_observation_sha256="b" * 64,
            source_judge_ledger_after_receipt_sha256="c" * 64,
        )


def test_completed_capacity_receipt_requires_its_batch_observation_identity() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    loaded = load_capacity_measurement(measurement(), now=NOW)

    with pytest.raises(CapacityPlanningError, match="observation identity"):
        build_capacity_receipt(
            policy=policy,
            measurement=loaded,
            planned_at=NOW,
            completed_trials=8,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"completed_through_trial": 41}, "complete batch boundary"),
        ({"batch_size": 4}, "exactly eight"),
        ({"timed_canary_receipt_sha256": "b" * 64}, "timed canary identity"),
        ({"judge_elapsed_seconds_per_trial_upper_bound": 91.0}, "cannot exceed"),
        ({"judge_elapsed_seconds_per_trial_upper_bound": 39.0}, "cannot decrease"),
        ({"judge_cost_usd_per_trial_upper_bound": "0.00001"}, "cannot decrease"),
        ({"judge_failures": 1}, "cannot exceed judge_calls"),
        ({"judge_calls": 1}, "changed ledger receipt"),
        ({
            "judge_calls": 8,
            "judge_ledger_after_receipt_sha256": "c" * 64,
            "judge_cost_usd_total": "0.01",
        }, "underprices the batch total"),
    ],
)
def test_rolling_batch_observation_fails_closed(change: dict, message: str) -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    initial = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )
    raw = {
        "format": "truth_editing_capacity_batch_observation_v2",
        "observation_id": "batch-0005",
        "observed_at": "2026-08-28T11:59:00Z",
        "timed_canary_receipt_sha256": SHA,
        "completed_through_trial": 40,
        "batch_size": 8,
        "generated_tokens_per_trial_upper_bound": 120,
        "generation_seconds_per_trial_upper_bound": 40.0,
        "trial_wall_seconds_upper_bound": 90.0,
        "judge_elapsed_seconds_per_trial_upper_bound": 45.0,
        "judge_cost_usd_per_trial_upper_bound": "0.0002",
        "judge_ledger_before_receipt_sha256": "b" * 64,
        "judge_ledger_after_receipt_sha256": "b" * 64,
        "judge_calls": 0,
        "judge_failures": 0,
        "judge_elapsed_seconds_total": 0.0,
        "judge_cost_usd_total": "0",
        "spend": measurement()["spend"],
    }
    raw.update(change)
    observation = _self_hashed(raw)

    with pytest.raises(CapacityPlanningError, match=message):
        reforecast_capacity_receipt(
            policy=policy,
            previous_receipt=initial,
            batch_observation=observation,
            planned_at=NOW,
        )


def test_signed_malformed_tier_projection_fails_closed() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1.json").read_text())
    )
    receipt = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )
    malformed = copy.deepcopy(receipt)
    malformed.pop("receipt_sha256")
    malformed["conservative_projection"]["tiers"][1].pop(
        "batch_evaluation_cost_usd_upper_bound"
    )
    malformed["receipt_sha256"] = canonical_sha256(malformed)

    with pytest.raises(CapacityPlanningError, match=r"projection tier.*fields changed"):
        validate_capacity_receipt(malformed)


def test_measurement_is_derived_from_identity_checked_timed_canary() -> None:
    unsigned_canary = {
        "format": "truth_editing_timed_canary_receipt_v2",
        "canary_config_sha256": "b" * 64,
        "production_config_path": "configs/production.json",
        "production_config_sha256": "c" * 64,
        "model_sha256": "d" * 64,
        "observation_sha256": "e" * 64,
        "trial_id": "trial-0001",
        "trial_outcome_kind": "successful",
        "trial_output_sha256": "f" * 64,
        "generated_tokens": 120,
        "generation_seconds": 4.0,
        "tokens_per_second": 30.0,
        "measured_wall_seconds": 50.0,
        "estimated_canary_cost_usd": 0.001,
        "single_worker_trials_per_hour": 72.0,
        "single_worker_200_trial_hours": 2.7777777778,
        "single_worker_200_trial_cost_usd": 0.1388888889,
        "judge_calls": 10,
        "judge_cost_usd": 0.1,
        "judge_elapsed_seconds": 40.0,
        "gpu_hourly_usd": 0.05,
        "persistence_kl": {
            "text_general": 0.01,
            "vision_general": 0.01,
            "computer_use": 0.01,
        },
        "software_and_live_canary_passed": True,
    }
    canary = {**unsigned_canary, "receipt_sha256": canonical_sha256(unsigned_canary)}

    result = create_capacity_measurement(
        measurement_id="timed-canary-v4-r10-r6",
        observed_at=datetime(2026, 8, 28, 11, 30, tzinfo=timezone.utc),
        timed_canary_receipt=canary,
        spend=measurement()["spend"],
        projected_storage_network_usd="0.2",
    )

    assert result["timed_canary_receipt_sha256"] == canary["receipt_sha256"]
    assert result["judge_cost_usd_per_trial"] == "0.1"
    assert result["judge_latency_seconds"] == 40.0
    assert load_capacity_measurement(result, now=NOW).tokens_per_second == 30.0

    canary["tokens_per_second"] = 31.0
    with pytest.raises(CapacityPlanningError, match="timed canary receipt_sha256"):
        create_capacity_measurement(
            measurement_id="timed-canary-v4-r10-r6",
            observed_at=datetime(2026, 8, 28, 11, 30, tzinfo=timezone.utc),
            timed_canary_receipt=canary,
            spend=measurement()["spend"],
            projected_storage_network_usd="0.2",
        )

    extra = dict(unsigned_canary)
    extra["unexpected"] = True
    extra = {**extra, "receipt_sha256": canonical_sha256(extra)}
    with pytest.raises(CapacityPlanningError, match="receipt fields changed"):
        create_capacity_measurement(
            measurement_id="timed-canary-v4-r10-r6",
            observed_at=datetime(2026, 8, 28, 11, 30, tzinfo=timezone.utc),
            timed_canary_receipt=extra,
            spend=measurement()["spend"],
            projected_storage_network_usd="0.2",
        )

    impossible_elapsed = dict(unsigned_canary)
    impossible_elapsed["judge_elapsed_seconds"] = 51.0
    impossible_elapsed = {
        **impossible_elapsed,
        "receipt_sha256": canonical_sha256(impossible_elapsed),
    }
    with pytest.raises(CapacityPlanningError, match="cannot exceed"):
        create_capacity_measurement(
            measurement_id="timed-canary-v4-r10-r6",
            observed_at=datetime(2026, 8, 28, 11, 30, tzinfo=timezone.utc),
            timed_canary_receipt=impossible_elapsed,
            spend=measurement()["spend"],
            projected_storage_network_usd="0.2",
        )


def test_capacity_plan_selects_the_most_conservative_limit() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(
            Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_text()
        )
    )
    slow = measurement(trial_wall_seconds=600.0)

    receipt = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(slow, now=NOW),
        planned_at=NOW,
    )

    assert 200 <= receipt["decision"]["planned_trial_limit"] < 800
    assert receipt["decision"]["planned_trial_limit"] % 8 == 0
    assert receipt["decision"]["planned_batch_limit"] == receipt["decision"]["planned_trial_limit"] // 8


def test_tps_and_pending_reservations_materially_reduce_projection() -> None:
    policy = CapacityPolicy.from_mapping(json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_text()))
    baseline = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(trial_wall_seconds=600.0, per_gpu_hourly_usd="0.1"), now=NOW),
        planned_at=NOW,
    )
    slower_tps = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(trial_wall_seconds=600.0, tokens_per_second=3.0, per_gpu_hourly_usd="0.1"), now=NOW),
        planned_at=NOW,
    )
    pending = measurement(trial_wall_seconds=600.0, per_gpu_hourly_usd="0.1", spend={
        "actual_total_usd": "1", "actual_infrastructure_usd": "0.9",
        "actual_evaluation_usd": "0.1", "pending_infrastructure_usd": "30",
        "pending_evaluation_usd": "0.5",
    })
    with_pending = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(pending, now=NOW),
        planned_at=NOW,
    )

    assert slower_tps["capacity_limits"]["time_limited_trials"] < baseline["capacity_limits"]["time_limited_trials"]
    assert with_pending["capacity_limits"]["infrastructure_budget_limited_trials"] < baseline["capacity_limits"]["infrastructure_budget_limited_trials"]


def test_capacity_plan_refuses_when_200_trials_cannot_be_guaranteed() -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(
            Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_text()
        )
    )

    with pytest.raises(CapacityPlanningError, match="cannot guarantee 200 trials"):
        build_capacity_receipt(
            policy=policy,
            measurement=load_capacity_measurement(
                measurement(trial_wall_seconds=3000.0), now=NOW
            ),
            planned_at=NOW,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tokens_per_second": float("nan")}, "canonical JSON"),
        ({"trial_wall_seconds": 0}, "trial_wall_seconds"),
        ({"judge_latency_seconds": 601}, "cannot exceed trial_wall_seconds"),
        ({"generated_tokens": True}, "generated_tokens"),
        ({"judge_cost_usd_per_trial": "0"}, "must be positive"),
        ({"per_gpu_hourly_usd": "0"}, "must be positive"),
        ({"unexpected": 1}, "fields changed"),
        ({"observed_at": "2026-08-28T02:00:00Z"}, "stale"),
        ({"observed_at": "2026-08-28T12:01:00Z"}, "future"),
        ({"spend": {"actual_total_usd": "50", "actual_infrastructure_usd": "45", "actual_evaluation_usd": "5", "pending_infrastructure_usd": "0", "pending_evaluation_usd": "0"}}, "total budget"),
        ({"spend": {"actual_total_usd": "1", "actual_infrastructure_usd": "0.9", "actual_evaluation_usd": "0.1", "pending_infrastructure_usd": "45", "pending_evaluation_usd": "0"}}, "infrastructure budget"),
        ({"spend": {"actual_total_usd": "1", "actual_infrastructure_usd": "0.9", "actual_evaluation_usd": "0.1", "pending_infrastructure_usd": "0", "pending_evaluation_usd": "5"}}, "evaluation budget"),
        ({"spend": {"actual_total_usd": "1.1", "actual_infrastructure_usd": "0.9", "actual_evaluation_usd": "0.1", "pending_infrastructure_usd": "0", "pending_evaluation_usd": "0"}}, "must equal infrastructure plus evaluation"),
    ],
)
def test_measurement_fails_closed(change: dict, message: str) -> None:
    if any(isinstance(value, float) and value != value for value in change.values()):
        value = measurement()
        value.update(change)
    else:
        value = measurement(**change)
    with pytest.raises(CapacityPlanningError, match=message):
        load_capacity_measurement(value, now=NOW)


def test_measurement_identity_and_receipt_tampering_fail_closed() -> None:
    bad_measurement = measurement()
    bad_measurement["tokens_per_second"] = 31.0
    with pytest.raises(CapacityPlanningError, match="self_sha256"):
        load_capacity_measurement(bad_measurement, now=NOW)

    policy = CapacityPolicy.from_mapping(
        json.loads(
            Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_text()
        )
    )
    receipt = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )
    tampered = copy.deepcopy(receipt)
    tampered["decision"]["planned_trial_limit"] = 792
    with pytest.raises(CapacityPlanningError, match="receipt_sha256"):
        validate_capacity_receipt(tampered)

    signed_unknown = copy.deepcopy(receipt)
    signed_unknown.pop("receipt_sha256")
    signed_unknown["unexpected"] = True
    signed_unknown["receipt_sha256"] = canonical_sha256(signed_unknown)
    with pytest.raises(CapacityPlanningError, match="fields changed"):
        validate_capacity_receipt(signed_unknown)


def test_write_capacity_receipt_is_atomic_and_no_clobber(tmp_path: Path) -> None:
    policy = CapacityPolicy.from_mapping(
        json.loads(
            Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_text()
        )
    )
    receipt = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement(), now=NOW),
        planned_at=NOW,
    )
    target = tmp_path / "monitoring" / "capacity-receipt.json"

    write_capacity_receipt(target, receipt)

    assert json.loads(target.read_text()) == receipt
    with pytest.raises(CapacityPlanningError, match="already exists"):
        write_capacity_receipt(target, receipt)
