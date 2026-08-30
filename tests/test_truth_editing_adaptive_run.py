from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_adaptive_run import (
    AdaptiveBatchScheduler,
    AdaptiveRunError,
)
from intelligent_liars.truth_editing_capacity import (
    CapacityPolicy,
    MinimumTrialGuaranteeError,
    SpendSnapshot,
    build_capacity_receipt,
    load_capacity_measurement,
    reforecast_capacity_receipt,
)
from intelligent_liars.truth_editing_contracts import canonical_sha256
from intelligent_liars.truth_editing_wandb_checkpoint import create_wandb_run_checkpoint


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def _policy() -> CapacityPolicy:
    return CapacityPolicy.from_mapping(
        json.loads(Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_text())
    )


def _spend(
    *, infrastructure: str = "1", evaluation: str = "0.1",
    pending_infrastructure: str = "0", pending_evaluation: str = "0",
) -> SpendSnapshot:
    total = Decimal(infrastructure) + Decimal(evaluation)
    return SpendSnapshot.from_mapping({
        "actual_total_usd": str(total),
        "actual_infrastructure_usd": infrastructure,
        "actual_evaluation_usd": evaluation,
        "pending_infrastructure_usd": pending_infrastructure,
        "pending_evaluation_usd": pending_evaluation,
    })


def _receipt(*, wall: float = 50.0) -> dict:
    unsigned = {
        "format": "truth_editing_capacity_measurement_v1",
        "measurement_id": "adaptive-scheduler-test",
        "observed_at": "2026-08-28T11:30:00Z",
        "timed_canary_receipt_sha256": "a" * 64,
        "generated_tokens": 120,
        "tokens_per_second": 30.0,
        "trial_wall_seconds": wall,
        "judge_latency_seconds": 10.0,
        "judge_cost_usd_per_trial": "0.0001",
        "per_gpu_hourly_usd": "0.05",
        "projected_storage_network_usd": "0.1",
        "spend": {
            "actual_total_usd": "1.1",
            "actual_infrastructure_usd": "1",
            "actual_evaluation_usd": "0.1",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    }
    measurement = {**unsigned, "self_sha256": canonical_sha256(unsigned)}
    return build_capacity_receipt(
        policy=_policy(),
        measurement=load_capacity_measurement(measurement, now=NOW),
        planned_at=NOW,
    )


def _scheduler(
    tmp_path: Path, *, now: list[datetime] | None = None,
    spend: list[SpendSnapshot] | None = None, receipt: dict | None = None,
    rolling_receipt: list[dict] | None = None,
    initial_started_at: datetime | None = None,
) -> AdaptiveBatchScheduler:
    clock = now or [NOW]
    ledger = spend or [_spend()]
    wandb_path = tmp_path / "wandb-run.json"
    create_wandb_run_checkpoint(
        wandb_path, run_id="truth-editing-main-run",
        project="intelligent-liars", entity="centipawn",
    )
    initial = receipt or _receipt()
    rolling = rolling_receipt or [initial]
    return AdaptiveBatchScheduler.open(
        policy=_policy(), capacity_receipt=initial,
        checkpoint_path=tmp_path / "adaptive-run-checkpoint.json",
        study_identity_sha256="b" * 64,
        wandb_checkpoint_path=wandb_path,
        spend_reader=lambda: ledger[0], clock=lambda: clock[0],
        capacity_receipt_reader=lambda: rolling[0],
        initial_started_at=initial_started_at,
    )


def test_fresh_scheduler_deadlines_include_pre_controller_host_setup_time(
    tmp_path: Path,
) -> None:
    lease_started = NOW - timedelta(hours=2)
    now = [NOW]
    scheduler = _scheduler(
        tmp_path, now=now, initial_started_at=lease_started
    )

    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    checkpoint_path = tmp_path / "adaptive-run-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["started_at_utc"] == lease_started.isoformat().replace(
        "+00:00", "Z"
    )
    assert checkpoint["hard_deadline_utc"] == (
        lease_started + timedelta(hours=24)
    ).isoformat().replace("+00:00", "Z")

    now[0] = lease_started + timedelta(hours=21)
    assert not scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )


def _commit_with_observation(
    scheduler: AdaptiveBatchScheduler,
    rolling: list[dict],
    *,
    completed_trials: int,
    coverage_complete: bool,
    wall: float = 90.0,
) -> None:
    unsigned = {
        "format": "truth_editing_capacity_batch_observation_v2",
        "observation_id": f"batch-{completed_trials:04d}",
        "observed_at": "2026-08-28T12:00:00Z",
        "timed_canary_receipt_sha256": rolling[0][
            "timed_canary_receipt_sha256"
        ],
        "completed_through_trial": completed_trials,
        "batch_size": 8,
        "generated_tokens_per_trial_upper_bound": 120,
        "generation_seconds_per_trial_upper_bound": 40.0,
        "trial_wall_seconds_upper_bound": wall,
        "judge_elapsed_seconds_per_trial_upper_bound": 45.0,
        "judge_cost_usd_per_trial_upper_bound": "0.0002",
        "judge_ledger_before_receipt_sha256": "c" * 64,
        "judge_ledger_after_receipt_sha256": "c" * 64,
        "judge_calls": 0,
        "judge_failures": 0,
        "judge_elapsed_seconds_total": 0.0,
        "judge_cost_usd_total": "0",
        "spend": {
            "actual_total_usd": "1.1",
            "actual_infrastructure_usd": "1",
            "actual_evaluation_usd": "0.1",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    }
    observation = {**unsigned, "self_sha256": canonical_sha256(unsigned)}
    rolling[0] = reforecast_capacity_receipt(
        policy=_policy(), previous_receipt=rolling[0],
        batch_observation=observation, planned_at=NOW,
    )
    scheduler.commit_batch(
        completed_trials=completed_trials,
        coverage_complete=coverage_complete,
    )


def test_persists_exact_eight_trial_authorization_and_resumes_pending_batch(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    first = json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text())
    assert first["authorized_through_trial"] == 8
    assert first["phase"] == "broad_coverage"

    resumed = _scheduler(tmp_path)
    assert resumed.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    assert json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text()) == first


def test_preauthorized_unstarted_batch_is_rechecked_at_absolute_deadline(
    tmp_path: Path,
) -> None:
    now = [NOW]
    scheduler = _scheduler(tmp_path, now=now)
    assert scheduler.admit_batch(
        completed_trials=0,
        batch_size=8,
        coverage_complete=False,
        batch_started=False,
    )

    now[0] = NOW + timedelta(seconds=_policy().search_seconds)
    resumed = _scheduler(tmp_path, now=now)
    assert not resumed.admit_batch(
        completed_trials=0,
        batch_size=8,
        coverage_complete=False,
        batch_started=False,
    )

    checkpoint = json.loads(
        (tmp_path / "adaptive-run-checkpoint.json").read_text()
    )
    assert checkpoint["authorized_through_trial"] == 0
    assert checkpoint["stop_reason"] == "minimum_trial_guarantee_lost"


def test_preauthorized_started_batch_replays_after_deadline_without_reauthorization(
    tmp_path: Path,
) -> None:
    now = [NOW]
    scheduler = _scheduler(tmp_path, now=now)
    assert scheduler.admit_batch(
        completed_trials=0,
        batch_size=8,
        coverage_complete=False,
    )
    checkpoint_path = tmp_path / "adaptive-run-checkpoint.json"
    before = checkpoint_path.read_bytes()

    now[0] = NOW + timedelta(seconds=_policy().search_seconds)
    resumed = _scheduler(tmp_path, now=now)
    assert resumed.admit_batch(
        completed_trials=0,
        batch_size=8,
        coverage_complete=False,
        batch_started=True,
    )
    assert checkpoint_path.read_bytes() == before


def test_preauthorized_unstarted_batch_is_rechecked_against_live_budget_reserve(
    tmp_path: Path,
) -> None:
    spend = [_spend()]
    scheduler = _scheduler(tmp_path, spend=spend)
    assert scheduler.admit_batch(
        completed_trials=0,
        batch_size=8,
        coverage_complete=False,
    )

    spend[0] = _spend(infrastructure="44.8", evaluation="0.1")
    resumed = _scheduler(tmp_path, spend=spend)
    assert not resumed.admit_batch(
        completed_trials=0,
        batch_size=8,
        coverage_complete=False,
        batch_started=False,
    )

    checkpoint = json.loads(
        (tmp_path / "adaptive-run-checkpoint.json").read_text()
    )
    assert checkpoint["authorized_through_trial"] == 0
    assert checkpoint["stop_reason"] == "minimum_trial_guarantee_lost"


def test_commit_batch_advances_scheduler_to_the_journaled_barrier(tmp_path: Path) -> None:
    initial = _receipt()
    rolling = [initial]
    scheduler = _scheduler(tmp_path, receipt=initial, rolling_receipt=rolling)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    _commit_with_observation(
        scheduler, rolling, completed_trials=8, coverage_complete=False
    )

    checkpoint_path = tmp_path / "adaptive-run-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["authorized_through_trial"] == 8
    assert checkpoint["completed_trials"] == 8
    assert checkpoint["phase"] == "broad_coverage"
    assert checkpoint["current_capacity_observation_sha256"] == rolling[0][
        "source_batch_observation_sha256"
    ]
    assert checkpoint["current_capacity_completed_through_trial"] == 8
    assert checkpoint["current_judge_ledger_receipt_sha256"] == rolling[0][
        "source_judge_ledger_after_receipt_sha256"
    ]

    resumed = _scheduler(tmp_path, receipt=initial, rolling_receipt=rolling)
    assert resumed.admit_batch(
        completed_trials=8, batch_size=8, coverage_complete=False
    )
    assert json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text())[
        "authorized_through_trial"
    ] == 16


def test_replayed_committed_callback_is_identity_bound_and_idempotent(
    tmp_path: Path,
) -> None:
    initial = _receipt()
    rolling = [initial]
    scheduler = _scheduler(tmp_path, receipt=initial, rolling_receipt=rolling)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    _commit_with_observation(
        scheduler, rolling, completed_trials=8, coverage_complete=False
    )
    assert scheduler.admit_batch(
        completed_trials=8, batch_size=8, coverage_complete=False
    )
    checkpoint_path = tmp_path / "adaptive-run-checkpoint.json"
    before_replay = checkpoint_path.read_bytes()

    resumed = _scheduler(
        tmp_path, receipt=initial, rolling_receipt=rolling
    )
    resumed.commit_batch(completed_trials=8, coverage_complete=False)
    assert checkpoint_path.read_bytes() == before_replay
    assert json.loads(before_replay)["authorized_through_trial"] == 16

    with pytest.raises(AdaptiveRunError, match="durable identities"):
        resumed.commit_batch(completed_trials=8, coverage_complete=True)

    with pytest.raises(AdaptiveRunError, match="durable identities"):
        _commit_with_observation(
            resumed,
            rolling,
            completed_trials=8,
            coverage_complete=False,
            wall=100.0,
        )


def test_signed_slow_batch_reforecasts_checkpoint_and_resume_admission(
    tmp_path: Path,
) -> None:
    initial = _receipt()
    rolling = [initial]
    now = [NOW]
    scheduler = _scheduler(
        tmp_path, now=now, receipt=initial, rolling_receipt=rolling
    )
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    initial_checkpoint = json.loads(
        (tmp_path / "adaptive-run-checkpoint.json").read_text()
    )

    _commit_with_observation(
        scheduler,
        rolling,
        completed_trials=8,
        coverage_complete=False,
        wall=300.0,
    )
    reforecasted = json.loads(
        (tmp_path / "adaptive-run-checkpoint.json").read_text()
    )
    assert reforecasted["current_capacity_receipt_sha256"] == rolling[0][
        "receipt_sha256"
    ]
    assert reforecasted["current_capacity_observation_sha256"] == rolling[0][
        "source_batch_observation_sha256"
    ]
    assert reforecasted["current_judge_ledger_receipt_sha256"] == rolling[0][
        "source_judge_ledger_after_receipt_sha256"
    ]
    assert (
        reforecasted["current_next_batch_projection"][
            "batch_duration_seconds_upper_bound"
        ]
        > initial_checkpoint["current_next_batch_projection"][
            "batch_duration_seconds_upper_bound"
        ]
    )
    assert (
        reforecasted["projected_search_eta_seconds"]
        > initial_checkpoint["projected_search_eta_seconds"]
    )

    now[0] = NOW + timedelta(hours=20, minutes=58)
    resumed = _scheduler(
        tmp_path, now=now, receipt=initial, rolling_receipt=rolling
    )
    assert not resumed.admit_batch(
        completed_trials=8, batch_size=8, coverage_complete=False
    )

    terminal = json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text())
    assert terminal["phase"] == "aborted"
    assert terminal["stop_reason"] == "minimum_trial_guarantee_lost"
    assert terminal["projected_search_eta_seconds"] <= 120.0


def test_concentration_starts_on_rich_coverage_not_the_200_trial_floor(
    tmp_path: Path,
) -> None:
    initial = _receipt()
    rolling = [initial]
    scheduler = _scheduler(tmp_path, receipt=initial, rolling_receipt=rolling)
    for completed in range(0, 40, 8):
        if completed:
            _commit_with_observation(
                scheduler,
                rolling,
                completed_trials=completed,
                coverage_complete=completed >= 32,
            )
        assert scheduler.admit_batch(
            completed_trials=completed,
            batch_size=8,
            coverage_complete=completed >= 32,
        )
    checkpoint = json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text())
    assert checkpoint["authorized_through_trial"] == 40
    assert checkpoint["phase"] == "adaptive_search"


def test_advisory_canary_target_does_not_stop_search_when_live_capacity_remains(
    tmp_path: Path,
) -> None:
    receipt = _receipt(wall=800.0)
    assert receipt["decision"]["planned_trial_limit"] < 800
    rolling = [receipt]
    scheduler = _scheduler(tmp_path, receipt=receipt, rolling_receipt=rolling)
    for completed in range(0, receipt["decision"]["planned_trial_limit"] + 8, 8):
        if completed:
            _commit_with_observation(
                scheduler,
                rolling,
                completed_trials=completed,
                coverage_complete=True,
            )
        assert scheduler.admit_batch(
            completed_trials=completed, batch_size=8, coverage_complete=True
        )
    checkpoint = json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text())
    assert checkpoint["authorized_through_trial"] > checkpoint["planned_trial_count"]


def test_after_200_deadline_moves_to_durable_finalization_reserve(tmp_path: Path) -> None:
    now = [NOW]
    initial = _receipt()
    rolling = [initial]
    scheduler = _scheduler(
        tmp_path, now=now, receipt=initial, rolling_receipt=rolling
    )
    for completed in range(0, 200, 8):
        if completed:
            _commit_with_observation(
                scheduler,
                rolling,
                completed_trials=completed,
                coverage_complete=True,
            )
        assert scheduler.admit_batch(
            completed_trials=completed, batch_size=8, coverage_complete=True
        )
    _commit_with_observation(
        scheduler,
        rolling,
        completed_trials=200,
        coverage_complete=True,
    )
    now[0] = NOW + timedelta(hours=21)
    assert not scheduler.admit_batch(
        completed_trials=200, batch_size=8, coverage_complete=True
    )
    checkpoint = json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text())
    assert checkpoint["phase"] == "finalization_reserved"
    assert checkpoint["stop_reason"] == "search_deadline_reached"


def test_exact_800_trial_ceiling_commits_to_finalization_without_808_projection(
    tmp_path: Path,
) -> None:
    initial = _receipt()
    rolling = [initial]
    scheduler = _scheduler(tmp_path, receipt=initial, rolling_receipt=rolling)
    for completed in range(0, 800, 8):
        if completed:
            _commit_with_observation(
                scheduler,
                rolling,
                completed_trials=completed,
                coverage_complete=True,
            )
        assert scheduler.admit_batch(
            completed_trials=completed,
            batch_size=8,
            coverage_complete=True,
        )
    _commit_with_observation(
        scheduler,
        rolling,
        completed_trials=800,
        coverage_complete=True,
    )

    checkpoint = json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text())
    assert checkpoint["authorized_through_trial"] == 800
    assert checkpoint["completed_trials"] == 800
    assert checkpoint["phase"] == "finalization_reserved"
    assert checkpoint["stop_reason"] == "maximum_trials_reached"
    assert not scheduler.admit_batch(
        completed_trials=800,
        batch_size=8,
        coverage_complete=True,
    )


def test_before_200_reserve_exhaustion_persists_a_terminal_failure(tmp_path: Path) -> None:
    now = [NOW]
    initial = _receipt()
    rolling = [initial]
    scheduler = _scheduler(
        tmp_path, now=now, receipt=initial, rolling_receipt=rolling
    )
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    _commit_with_observation(
        scheduler, rolling, completed_trials=8, coverage_complete=False
    )
    now[0] = NOW + timedelta(hours=21)
    assert not scheduler.admit_batch(
        completed_trials=8, batch_size=8, coverage_complete=False
    )
    checkpoint_path = tmp_path / "adaptive-run-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["phase"] == "aborted"
    assert checkpoint["stop_reason"] == "minimum_trial_guarantee_lost"

    resumed = _scheduler(
        tmp_path, now=now, receipt=initial, rolling_receipt=rolling
    )
    assert not resumed.admit_batch(
        completed_trials=8, batch_size=8, coverage_complete=False
    )

    original = json.loads(checkpoint_path.read_text())
    wrong_phase = dict(original)
    wrong_phase["phase"] = "finalization_reserved"
    unsigned = dict(wrong_phase)
    unsigned.pop("checkpoint_sha256")
    wrong_phase["checkpoint_sha256"] = canonical_sha256(unsigned)
    checkpoint_path.write_text(json.dumps(wrong_phase))
    with pytest.raises(AdaptiveRunError, match="abort reason"):
        _scheduler(tmp_path, receipt=initial, rolling_receipt=rolling)

    wrong_reason = dict(original)
    wrong_reason["stop_reason"] = "maximum_trials_reached"
    unsigned = dict(wrong_reason)
    unsigned.pop("checkpoint_sha256")
    wrong_reason["checkpoint_sha256"] = canonical_sha256(unsigned)
    checkpoint_path.write_text(json.dumps(wrong_reason))
    with pytest.raises(AdaptiveRunError, match="abort reason"):
        _scheduler(tmp_path, receipt=initial, rolling_receipt=rolling)


def test_lost_minimum_guarantee_is_a_durable_aborted_run_state(tmp_path: Path) -> None:
    initial = _receipt()
    rolling = [initial]
    scheduler = _scheduler(tmp_path, receipt=initial, rolling_receipt=rolling)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    unsigned = {
        "format": "truth_editing_capacity_batch_observation_v2",
        "observation_id": "batch-0008-infeasible",
        "observed_at": "2026-08-28T12:00:00Z",
        "timed_canary_receipt_sha256": initial[
            "timed_canary_receipt_sha256"
        ],
        "completed_through_trial": 8,
        "batch_size": 8,
        "generated_tokens_per_trial_upper_bound": 120,
        "generation_seconds_per_trial_upper_bound": 40.0,
        "trial_wall_seconds_upper_bound": 90.0,
        "judge_elapsed_seconds_per_trial_upper_bound": 45.0,
        "judge_cost_usd_per_trial_upper_bound": "0.0002",
        "judge_ledger_before_receipt_sha256": "c" * 64,
        "judge_ledger_after_receipt_sha256": "c" * 64,
        "judge_calls": 0,
        "judge_failures": 0,
        "judge_elapsed_seconds_total": 0.0,
        "judge_cost_usd_total": "0",
        "spend": {
            "actual_total_usd": "1.1",
            "actual_infrastructure_usd": "1",
            "actual_evaluation_usd": "0.1",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    }
    observation = {**unsigned, "self_sha256": canonical_sha256(unsigned)}
    with pytest.raises(MinimumTrialGuaranteeError) as failure:
        reforecast_capacity_receipt(
            policy=_policy(),
            previous_receipt=initial,
            batch_observation=observation,
            planned_at=NOW,
            remaining_search_seconds=1.0,
        )
    rolling[0] = failure.value.receipt

    scheduler.abort_minimum_trial_guarantee(
        completed_trials=8,
        coverage_complete=False,
    )

    checkpoint_path = tmp_path / "adaptive-run-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["authorized_through_trial"] == 8
    assert checkpoint["completed_trials"] == 8
    assert checkpoint["phase"] == "aborted"
    assert checkpoint["stop_reason"] == "minimum_trial_guarantee_lost"
    assert checkpoint["current_capacity_completed_through_trial"] == 8

    resumed = _scheduler(
        tmp_path, receipt=initial, rolling_receipt=rolling
    )
    assert not resumed.admit_batch(
        completed_trials=8, batch_size=8, coverage_complete=False
    )


def test_minimum_guarantee_abort_can_be_rearmed_under_a_fresh_lease(
    tmp_path: Path,
) -> None:
    initial = _receipt()
    rolling = [initial]
    now = [NOW]
    scheduler = _scheduler(
        tmp_path, now=now, receipt=initial, rolling_receipt=rolling
    )
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    _commit_with_observation(
        scheduler, rolling, completed_trials=8, coverage_complete=False
    )

    # Sign an infeasible rolling forecast, which is what causes the durable
    # minimum-guarantee abort in production.
    unsigned = {
        "format": "truth_editing_capacity_batch_observation_v2",
        "observation_id": "batch-0008-rearm-infeasible",
        "observed_at": "2026-08-28T12:00:00Z",
        "timed_canary_receipt_sha256": initial["timed_canary_receipt_sha256"],
        "completed_through_trial": 8,
        "batch_size": 8,
        "generated_tokens_per_trial_upper_bound": 120,
        "generation_seconds_per_trial_upper_bound": 40.0,
        "trial_wall_seconds_upper_bound": 90.0,
        "judge_elapsed_seconds_per_trial_upper_bound": 45.0,
        "judge_cost_usd_per_trial_upper_bound": "0.0002",
        "judge_ledger_before_receipt_sha256": "c" * 64,
        "judge_ledger_after_receipt_sha256": "c" * 64,
        "judge_calls": 0,
        "judge_failures": 0,
        "judge_elapsed_seconds_total": 0.0,
        "judge_cost_usd_total": "0",
        "spend": {
            "actual_total_usd": "1.1",
            "actual_infrastructure_usd": "1",
            "actual_evaluation_usd": "0.1",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    }
    observation = {**unsigned, "self_sha256": canonical_sha256(unsigned)}
    with pytest.raises(MinimumTrialGuaranteeError) as failure:
        reforecast_capacity_receipt(
            policy=_policy(), previous_receipt=rolling[0],
            batch_observation=observation, planned_at=NOW,
            remaining_search_seconds=1.0,
        )
    rolling[0] = failure.value.receipt
    scheduler.abort_minimum_trial_guarantee(
        completed_trials=8, coverage_complete=False
    )

    # A fresh lease must provide a new feasible rolling receipt before rearm.
    rolling[0] = _receipt()
    fresh_lease = NOW + timedelta(hours=1)
    resumed = _scheduler(
        tmp_path, now=[fresh_lease], receipt=initial, rolling_receipt=rolling
    )
    with pytest.raises(AdaptiveRunError, match="fresh capacity receipt"):
        resumed.rearm_minimum_guarantee_abort(started_at=fresh_lease)

    # Reforecast from the fresh receipt at the existing completed boundary.
    fresh_observation = dict(observation)
    fresh_observation["observation_id"] = "batch-0008-rearm-feasible"
    fresh_observation["observed_at"] = "2026-08-28T12:30:00Z"
    fresh_observation["self_sha256"] = canonical_sha256(
        {key: value for key, value in fresh_observation.items() if key != "self_sha256"}
    )
    rolling[0] = reforecast_capacity_receipt(
        policy=_policy(), previous_receipt=rolling[0],
        batch_observation=fresh_observation, planned_at=fresh_lease,
    )
    rearmed = _scheduler(
        tmp_path, now=[fresh_lease], receipt=initial, rolling_receipt=rolling
    )
    rearmed.rearm_minimum_guarantee_abort(started_at=fresh_lease)
    checkpoint = json.loads((tmp_path / "adaptive-run-checkpoint.json").read_text())
    assert checkpoint["phase"] == "broad_coverage"
    assert checkpoint["stop_reason"] is None
    assert checkpoint["completed_trials"] == 8
    assert checkpoint["authorized_through_trial"] == 8
    assert checkpoint["started_at_utc"] == fresh_lease.isoformat().replace(
        "+00:00", "Z"
    )
    assert rearmed.admit_batch(
        completed_trials=8, batch_size=8, coverage_complete=False
    )


def test_rearm_rejects_non_minimum_abort_checkpoint(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    with pytest.raises(AdaptiveRunError, match="rearm"):
        scheduler.rearm_minimum_guarantee_abort(started_at=NOW)


def test_resume_rejects_spend_rollback_and_identity_switch(tmp_path: Path) -> None:
    ledger = [_spend(infrastructure="2")]
    scheduler = _scheduler(tmp_path, spend=ledger)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )
    ledger[0] = _spend(infrastructure="1")
    resumed = _scheduler(tmp_path, spend=ledger)
    with pytest.raises(AdaptiveRunError, match="spend ledger moved backwards"):
        resumed.admit_batch(
            completed_trials=8, batch_size=8, coverage_complete=False
        )

    with pytest.raises(AdaptiveRunError, match="identity binding"):
        AdaptiveBatchScheduler.open(
            policy=_policy(), capacity_receipt=_receipt(),
            checkpoint_path=tmp_path / "adaptive-run-checkpoint.json",
            study_identity_sha256="d" * 64,
            wandb_checkpoint_path=tmp_path / "wandb-run.json",
            spend_reader=lambda: _spend(), clock=lambda: NOW,
        )


def test_fresh_scheduler_rejects_uncheckpointed_history_and_non_eight_batch(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    with pytest.raises(AdaptiveRunError, match="zero completed"):
        scheduler.admit_batch(
            completed_trials=200, batch_size=8, coverage_complete=True
        )
    with pytest.raises(AdaptiveRunError, match="batches of eight"):
        scheduler.admit_batch(
            completed_trials=0, batch_size=4, coverage_complete=False
        )
