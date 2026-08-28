from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_adaptive_run import (
    AdaptiveBatchScheduler,
    AdaptiveRunError,
)
from intelligent_liars.truth_editing_capacity import (
    CapacityPolicy,
    SpendSnapshot,
    build_capacity_receipt,
    load_capacity_measurement,
)
from intelligent_liars.truth_editing_wandb_checkpoint import (
    AdaptiveRunProgress,
    WandbCheckpointError,
    create_wandb_run_checkpoint,
)


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _policy() -> CapacityPolicy:
    return CapacityPolicy.from_mapping(
        json.loads(
            Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_text()
        )
    )


def _spend(*, infrastructure: str = "0.9", evaluation: str = "0.1") -> SpendSnapshot:
    total = format(Decimal(infrastructure) + Decimal(evaluation), "f")
    if "." in total:
        total = total.rstrip("0").rstrip(".")
    return SpendSnapshot.from_mapping(
        {
            "actual_total_usd": total,
            "actual_infrastructure_usd": infrastructure,
            "actual_evaluation_usd": evaluation,
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        }
    )


def _capacity_receipt() -> dict:
    unsigned = {
        "format": "truth_editing_capacity_measurement_v1",
        "measurement_id": "timed-canary-v4-r10-r6",
        "observed_at": "2026-08-28T11:30:00Z",
        "timed_canary_receipt_sha256": "a" * 64,
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
    measurement = load_capacity_measurement(
        {**unsigned, "self_sha256": _sha(unsigned)}, now=NOW
    )
    return build_capacity_receipt(
        policy=_policy(), measurement=measurement, planned_at=NOW
    )


def _scheduler(
    tmp_path: Path,
    *,
    spend: list[SpendSnapshot] | None = None,
    now: list[datetime] | None = None,
) -> AdaptiveBatchScheduler:
    live_spend = spend if spend is not None else [_spend()]
    live_now = now if now is not None else [NOW]
    wandb_path = tmp_path / "wandb-run.json"
    if not wandb_path.exists():
        create_wandb_run_checkpoint(
            wandb_path,
            run_id="adaptive-run",
            project="intelligent-liars",
            entity="centipawn",
        )
    return AdaptiveBatchScheduler.open(
        policy=_policy(),
        capacity_receipt=_capacity_receipt(),
        checkpoint_path=tmp_path / "adaptive.json",
        study_identity_sha256="b" * 64,
        wandb_checkpoint_path=wandb_path,
        spend_reader=lambda: live_spend[0],
        clock=lambda: live_now[0],
    )


def test_resume_rejects_a_spend_ledger_that_moves_backwards(tmp_path: Path) -> None:
    """A stale/reset cost reader must never reopen budget already consumed."""

    spend = [_spend(infrastructure="1.9")]
    scheduler = _scheduler(tmp_path, spend=spend)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )

    spend[0] = _spend(infrastructure="0.9")
    resumed = _scheduler(tmp_path, spend=spend)
    with pytest.raises(AdaptiveRunError, match="spend ledger moved backwards"):
        resumed.admit_batch(
            completed_trials=8, batch_size=8, coverage_complete=False
        )


def test_fresh_scheduler_rejects_spend_older_than_capacity_snapshot(
    tmp_path: Path,
) -> None:
    """The canary's committed spend is the first monotonic ledger checkpoint."""

    scheduler = _scheduler(
        tmp_path, spend=[_spend(infrastructure="0.1", evaluation="0.01")]
    )
    with pytest.raises(AdaptiveRunError, match="spend.*backwards|capacity"):
        scheduler.admit_batch(
            completed_trials=0, batch_size=8, coverage_complete=False
        )


def test_rehashed_checkpoint_cannot_authorize_trials_past_the_hard_ceiling(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )

    path = tmp_path / "adaptive.json"
    raw = json.loads(path.read_text())
    raw["authorized_through_trial"] = 808
    raw["completed_trials"] = 808
    unsigned = dict(raw)
    unsigned.pop("checkpoint_sha256")
    raw["checkpoint_sha256"] = _sha(unsigned)
    path.write_text(json.dumps(raw))

    with pytest.raises(AdaptiveRunError, match="trial boundary"):
        _scheduler(tmp_path)


def test_scheduler_does_not_stop_at_200_and_uses_actual_coverage_gate(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )

    path = tmp_path / "adaptive.json"
    raw = json.loads(path.read_text())
    raw["authorized_through_trial"] = 200
    raw["completed_trials"] = 200
    raw["coverage_complete"] = False
    raw["phase"] = "broad_coverage"
    unsigned = dict(raw)
    unsigned.pop("checkpoint_sha256")
    raw["checkpoint_sha256"] = _sha(unsigned)
    path.write_text(json.dumps(raw))

    resumed = _scheduler(tmp_path)
    assert resumed.admit_batch(
        completed_trials=200, batch_size=8, coverage_complete=True
    )
    checkpoint = json.loads(path.read_text())
    assert checkpoint["authorized_through_trial"] == 208
    assert checkpoint["phase"] == "adaptive_search"


def test_batch_after_200_uses_concentrated_not_discovery_cost_bound(
    tmp_path: Path,
) -> None:
    """The expensive late tier must govern admission at the 200 -> 208 edge."""

    spend = [_spend()]
    scheduler = _scheduler(tmp_path, spend=spend)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=False
    )

    path = tmp_path / "adaptive.json"
    raw = json.loads(path.read_text())
    raw["authorized_through_trial"] = 200
    raw["completed_trials"] = 200
    raw["coverage_complete"] = True
    raw["phase"] = "adaptive_search"
    unsigned = dict(raw)
    unsigned.pop("checkpoint_sha256")
    raw["checkpoint_sha256"] = _sha(unsigned)
    path.write_text(json.dumps(raw))

    # A discovery batch would fit below the $45 infrastructure boundary here,
    # but the concentrated batch upper bound does not.
    spend[0] = _spend(infrastructure="43.55", evaluation="0.1")
    resumed = _scheduler(tmp_path, spend=spend)
    assert not resumed.admit_batch(
        completed_trials=200, batch_size=8, coverage_complete=True
    )
    checkpoint = json.loads(path.read_text())
    assert checkpoint["authorized_through_trial"] == 200
    assert checkpoint["stop_reason"] == "infrastructure_budget_reserve_reached"


def test_scheduler_rejects_coverage_regression_after_concentration(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    assert scheduler.admit_batch(
        completed_trials=0, batch_size=8, coverage_complete=True
    )
    resumed = _scheduler(tmp_path)
    with pytest.raises(AdaptiveRunError, match="coverage"):
        resumed.admit_batch(
            completed_trials=8, batch_size=8, coverage_complete=False
        )


def test_progress_cannot_claim_concentration_before_broad_coverage() -> None:
    """The durable/W&B view must not disagree with the optimizer's coverage gate."""

    with pytest.raises(WandbCheckpointError, match="coverage"):
        AdaptiveRunProgress(
            wandb_run_checkpoint_sha256="d" * 64,
            study_config_sha256="e" * 64,
            planned_floor_trials=200,
            adaptive_ceiling_trials=800,
            measured_target_trials=400,
            batch_size=8,
            search_cutoff_seconds=21 * 3600,
            reserve_seconds=3 * 3600,
            total_budget_usd=50.0,
            evaluation_budget_usd=5.0,
            evaluation_budget_reserve_fraction=0.2,
            completed_search_trials=40,
            completed_repeat_trials=0,
            completed_control_trials=0,
            completed_final_selection_trials=0,
            current_batch=5,
            stage="adaptive_search",
            coverage={
                "direction_family": (17, 18),
                "layer_region": (3, 3),
                "intervention_arm": (3, 3),
                "attention_mlp_configuration": (4, 4),
                "refusal_setting": (2, 2),
                "strength_range": (4, 4),
            },
            elapsed_seconds=1000.0,
            eta_seconds=2000.0,
            gpu_actual_usd=1.0,
            gpu_projected_usd=5.0,
            judge_actual_usd=0.1,
            judge_projected_usd=0.5,
            projected_total_usd=5.5,
            measured_trial_duration_seconds=60.0,
            measured_tokens_per_second=30.0,
            measured_judge_latency_ms=1000.0,
            measured_judge_cost_usd_per_trial=0.002,
        )
