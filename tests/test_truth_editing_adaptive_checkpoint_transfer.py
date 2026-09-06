from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import optuna
import pytest

from intelligent_liars.truth_editing_phase_checkpoint import (
    PhaseCheckpointError,
    publish_adaptive_checkpoint,
    restore_adaptive_checkpoint,
)
from intelligent_liars.truth_editing_adaptive_run import AdaptiveBatchScheduler
from intelligent_liars.truth_editing_capacity import (
    CapacityPolicy,
    MinimumTrialGuaranteeError,
    SpendSnapshot,
    build_capacity_receipt,
    load_capacity_measurement,
    reforecast_capacity_receipt,
)
from intelligent_liars.truth_editing_contracts import canonical_sha256
from intelligent_liars.truth_editing_study import OBJECTIVES, SearchProposal
from intelligent_liars.truth_editing_wandb_checkpoint import (
    AdaptiveRunProgress,
    advance_adaptive_progress_checkpoint,
    create_wandb_run_checkpoint,
    open_adaptive_progress_checkpoint,
)


STUDY_ID = "a" * 64
STUDY_CONFIG_SHA = "b" * 64
OPTUNA_STUDY_NAME = "truth-editing-adaptive-cccccccccccc-dddddddddddd"


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_adaptive_state(
    root: Path, *, completed: int = 80, progress_completed: int | None = None,
    abort_minimum: bool = False,
    operational_failure_ordinals: frozenset[int] = frozenset(),
    expanded_through: int = 160,
    trial_number_start: int = 0,
    single_tier: bool = False,
) -> None:
    progress_completed = completed if progress_completed is None else progress_completed
    (root / "study").mkdir(parents=True)
    trials = [
        {
            "trial_id": f"trial-{ordinal + trial_number_start:04d}",
            "ordinal": ordinal,
            "tier_name": (
                "discovery"
                if single_tier or ordinal < 80
                else "expanded" if ordinal < expanded_through else "finalist"
            ),
            "evaluation_record_ids": ["record-1"],
            "proposal": SearchProposal(
                direction_ids=("truth-general-21",),
                direction_family="general",
                source_layer=21,
                basis_method="qr",
                requested_rank=1,
                writer_region="middle",
                writer_layers=(21,),
                writer_policy="attention",
                strength=0.5,
            ).to_dict(),
            "result": {
                "outcome_kind": (
                    "operational_failure"
                    if ordinal in operational_failure_ordinals
                    else "successful"
                ),
                "metrics": (
                    {}
                    if ordinal in operational_failure_ordinals
                    else {name: 0.5 for name in OBJECTIVES}
                ),
                "detail": (
                    "transient worker failure"
                    if ordinal in operational_failure_ordinals
                    else None
                ),
            },
        }
        for ordinal in range(completed)
    ]
    journal = {
        "format": "truth_editing_study_journal_v1",
        "study_identity_sha256": STUDY_ID,
        "identity_inputs": {"frozen": True},
        "batches": [
            {"ordinal": index // 8, "trials": trials[index : index + 8]}
            for index in range(0, completed, 8)
        ],
    }
    journal["journal_sha256"] = _sha(journal)
    (root / "study/study-journal.json").write_text(json.dumps(journal) + "\n")
    optuna_path = root / "study/study-journal.json.optuna.log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(optuna_path))
    )
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=storage,
        directions=["maximize"] * len(OBJECTIVES),
    )
    for trial in trials:
        if trial["ordinal"] in operational_failure_ordinals:
            continue
        study.add_trial(
            optuna.trial.create_trial(
                values=[0.5] * len(OBJECTIVES),
                user_attrs={
                    "study_ordinal": trial["ordinal"],
                    "proposal_sha256": _sha(trial["proposal"]),
                },
            )
        )

    wandb = create_wandb_run_checkpoint(
        root / "monitoring/wandb-run.json",
        run_id="adaptive-transfer",
        project="intelligent-liars",
        entity="centipawn",
    )
    progress = AdaptiveRunProgress(
        wandb_run_checkpoint_sha256=wandb.checkpoint_sha256,
        study_config_sha256=STUDY_CONFIG_SHA,
        planned_floor_trials=64 if single_tier else 200,
        adaptive_ceiling_trials=64 if single_tier else 800,
        measured_target_trials=64 if single_tier else 480,
        batch_size=8,
        search_cutoff_seconds=75600,
        reserve_seconds=10800,
        total_budget_usd=50.0,
        evaluation_budget_usd=5.0,
        evaluation_budget_reserve_fraction=0.2,
        completed_search_trials=progress_completed,
        completed_repeat_trials=0,
        completed_control_trials=0,
        completed_final_selection_trials=0,
        current_batch=progress_completed // 8,
        stage="aborted" if abort_minimum else "adaptive_search",
        coverage={
            "direction_family": (18, 18),
            "layer_region": (3, 3),
            "intervention_arm": (3, 3),
            "attention_mlp_configuration": (4, 4),
            "refusal_setting": (2, 2),
            "strength_range": (4, 4),
        },
        elapsed_seconds=1200.0,
        eta_seconds=7000.0,
        gpu_actual_usd=1.0,
        gpu_projected_usd=30.0,
        judge_actual_usd=0.1,
        judge_projected_usd=3.0,
        projected_total_usd=33.0,
        measured_trial_duration_seconds=120.0,
        measured_tokens_per_second=35.0,
        measured_judge_latency_ms=900.0,
        measured_judge_cost_usd_per_trial=0.002,
    )
    advance_adaptive_progress_checkpoint(
        root / "monitoring/adaptive-progress.json", progress
    )
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    policy = CapacityPolicy.from_mapping(
        json.loads(
            Path(
                "configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json"
            ).read_text()
        )
    )
    measurement_unsigned = {
        "format": "truth_editing_capacity_measurement_v1",
        "measurement_id": "adaptive-transfer-test",
        "observed_at": "2026-08-28T11:30:00Z",
        "timed_canary_receipt_sha256": "e" * 64,
        "generated_tokens": 120,
        "tokens_per_second": 30.0,
        "trial_wall_seconds": 50.0,
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
    measurement = {
        **measurement_unsigned,
        "self_sha256": canonical_sha256(measurement_unsigned),
    }
    receipt = build_capacity_receipt(
        policy=policy,
        measurement=load_capacity_measurement(measurement, now=now),
        planned_at=now,
    )
    rolling_receipt = [receipt]
    rolling_receipt_path = root / "monitoring/rolling-capacity-receipt.json"
    spend = SpendSnapshot.from_mapping(measurement_unsigned["spend"])
    spend_state = [spend]
    scheduler = AdaptiveBatchScheduler.open(
        policy=policy,
        capacity_receipt=receipt,
        checkpoint_path=root / "study/adaptive-run-checkpoint.json",
        study_identity_sha256=STUDY_ID,
        wandb_checkpoint_path=root / "monitoring/wandb-run.json",
        spend_reader=lambda: spend_state[0],
        clock=lambda: now,
        capacity_receipt_reader=lambda: rolling_receipt[0],
    )
    for trial_count in range(0, completed, 8):
        assert scheduler.admit_batch(
            completed_trials=trial_count,
            batch_size=8,
            coverage_complete=trial_count >= 40,
        )
        next_completed = trial_count + 8
        prior_budget = rolling_receipt[0]["budget"]
        evaluation_spend = Decimal(
            prior_budget["actual_evaluation_usd"]
        ) + Decimal("0.0008")
        infrastructure_spend = Decimal(
            prior_budget["actual_infrastructure_usd"]
        )
        total_spend = infrastructure_spend + evaluation_spend
        observed_spend = {
            "actual_total_usd": format(total_spend.normalize(), "f"),
            "actual_infrastructure_usd": format(
                infrastructure_spend.normalize(), "f"
            ),
            "actual_evaluation_usd": format(evaluation_spend.normalize(), "f"),
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        }
        observation_unsigned = {
            "format": "truth_editing_capacity_batch_observation_v2",
            "observation_id": f"adaptive-transfer-{next_completed:04d}",
            "observed_at": "2026-08-28T12:00:00Z",
            "timed_canary_receipt_sha256": receipt[
                "timed_canary_receipt_sha256"
            ],
            "completed_through_trial": next_completed,
            "batch_size": 8,
            "generated_tokens_per_trial_upper_bound": 120,
            "generation_seconds_per_trial_upper_bound": 40.0,
            "trial_wall_seconds_upper_bound": 50.0,
            "judge_elapsed_seconds_per_trial_upper_bound": 10.0,
            "judge_cost_usd_per_trial_upper_bound": "0.0001",
            "judge_ledger_before_receipt_sha256": "1" * 64,
            "judge_ledger_after_receipt_sha256": "2" * 64,
            "judge_calls": 8,
            "judge_failures": 0,
            "judge_elapsed_seconds_total": 80.0,
            "judge_cost_usd_total": "0.0008",
            "spend": observed_spend,
        }
        observation = {
            **observation_unsigned,
            "self_sha256": canonical_sha256(observation_unsigned),
        }
        try:
            rolling_receipt[0] = reforecast_capacity_receipt(
                policy=policy,
                previous_receipt=rolling_receipt[0],
                batch_observation=observation,
                planned_at=now,
                remaining_search_seconds=(
                    1.0 if abort_minimum and next_completed == completed else None
                ),
            )
        except MinimumTrialGuaranteeError as error:
            if not abort_minimum or next_completed != completed:
                raise
            rolling_receipt[0] = error.receipt
        rolling_receipt_path.write_text(
            json.dumps(rolling_receipt[0], sort_keys=True) + "\n"
        )
        spend_state[0] = SpendSnapshot.from_mapping(observed_spend)
        if abort_minimum and next_completed == completed:
            scheduler.abort_minimum_trial_guarantee(
                completed_trials=next_completed,
                coverage_complete=next_completed >= 40,
            )
        else:
            scheduler.commit_batch(
                completed_trials=next_completed,
                coverage_complete=next_completed >= 40,
            )


def test_adaptive_checkpoint_accepts_audit_only_operational_batch_without_optuna_growth(
    tmp_path: Path,
) -> None:
    publication = tmp_path / "published"
    source_104 = tmp_path / "source-104"
    source_112 = tmp_path / "source-112"
    _write_adaptive_state(source_104, completed=104)
    _write_adaptive_state(
        source_112,
        completed=112,
        operational_failure_ordinals=frozenset(range(104, 112)),
    )
    prior_progress = json.loads(
        (source_104 / "monitoring/adaptive-progress.json").read_text()
    )
    next_progress = json.loads(
        (source_112 / "monitoring/adaptive-progress.json").read_text()
    )
    next_progress["revision"] = prior_progress["revision"] + 1
    next_progress["previous_checkpoint_sha256"] = prior_progress[
        "checkpoint_sha256"
    ]
    next_progress.pop("checkpoint_sha256")
    next_progress["checkpoint_sha256"] = _sha(next_progress)
    (source_112 / "monitoring/adaptive-progress.json").write_text(
        json.dumps(next_progress) + "\n"
    )
    # This is the exact production shape: the controller journal advances for
    # audit-only failures while Optuna's non-learning journal remains unchanged.
    (source_112 / "study/study-journal.json.optuna.log").write_bytes(
        (source_104 / "study/study-journal.json.optuna.log").read_bytes()
    )

    first = publish_adaptive_checkpoint(
        source_104,
        publication,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=104,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    second = publish_adaptive_checkpoint(
        source_112,
        publication,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=112,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )

    assert second["parent_manifest_sha256"] == first["manifest_sha256"]


def test_adaptive_checkpoint_round_trip_preserves_scheduler_progress_and_wandb(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source)
    expected_names = (
        "study/study-journal.json",
        "study/study-journal.json.optuna.log",
        "study/adaptive-run-checkpoint.json",
        "monitoring/wandb-run.json",
        "monitoring/adaptive-progress.json",
        "monitoring/rolling-capacity-receipt.json",
    )
    original = {name: (source / name).read_bytes() for name in expected_names}

    manifest = publish_adaptive_checkpoint(
        source,
        tmp_path / "published",
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    receipt = restore_adaptive_checkpoint(
        tmp_path / "published",
        tmp_path / "restored",
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )

    assert manifest["format"] == "truth_editing_adaptive_checkpoint_manifest_v1"
    assert manifest["monitoring"]["wandb_run_id"] == "adaptive-transfer"
    assert receipt["source_manifest_sha256"] == manifest["manifest_sha256"]
    assert {
        name: (tmp_path / "restored" / name).read_bytes()
        for name in expected_names
    } == original


def test_adaptive_checkpoint_round_trip_accepts_one_based_trial_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source, completed=8, trial_number_start=1)

    manifest = publish_adaptive_checkpoint(
        source,
        tmp_path / "published",
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=8,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
        trial_number_start=1,
    )
    receipt = restore_adaptive_checkpoint(
        tmp_path / "published",
        tmp_path / "restored",
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=8,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
        trial_number_start=1,
    )

    assert manifest["completed_trials"] == 8
    assert receipt["completed_trials"] == 8
    restored = json.loads(
        (tmp_path / "restored/study/study-journal.json").read_text()
    )
    assert [
        trial["trial_id"]
        for batch in restored["batches"]
        for trial in batch["trials"]
    ] == [f"trial-{ordinal:04d}" for ordinal in range(1, 9)]


def test_republishing_identical_adaptive_checkpoint_is_idempotent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    _write_adaptive_state(source)

    first = publish_adaptive_checkpoint(
        source,
        publication,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    second = publish_adaptive_checkpoint(
        source,
        publication,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )

    assert second == first
    assert len(list((publication / "adaptive-generations").iterdir())) == 1


def test_preminimum_abort_checkpoint_publishes_and_restores(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source, completed=8, abort_minimum=True)

    manifest = publish_adaptive_checkpoint(
        source,
        tmp_path / "published",
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=8,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    restore_adaptive_checkpoint(
        tmp_path / "published",
        tmp_path / "restored",
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=8,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )

    restored = json.loads(
        (tmp_path / "restored/study/adaptive-run-checkpoint.json").read_text()
    )
    assert manifest["completed_trials"] == 8
    assert restored["phase"] == "aborted"
    assert restored["stop_reason"] == "minimum_trial_guarantee_lost"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", "finalization_reserved"),
        ("stop_reason", "maximum_trials_reached"),
    ],
)
def test_preminimum_abort_checkpoint_rejects_mismatched_terminal_pair(
    tmp_path: Path, field: str, value: str,
) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source, completed=8, abort_minimum=True)
    checkpoint_path = source / "study/adaptive-run-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint[field] = value
    unsigned = dict(checkpoint)
    unsigned.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = canonical_sha256(unsigned)
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(PhaseCheckpointError, match="abort reason"):
        publish_adaptive_checkpoint(
            source,
            tmp_path / "published",
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=8,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_adaptive_checkpoint_requires_and_binds_rolling_capacity_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source)
    rolling_path = source / "monitoring/rolling-capacity-receipt.json"
    rolling_path.unlink()

    with pytest.raises(PhaseCheckpointError, match="rolling-capacity-receipt"):
        publish_adaptive_checkpoint(
            source,
            tmp_path / "missing-published",
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )

    _write_adaptive_state(tmp_path / "mismatched")
    rolling_path = tmp_path / "mismatched/monitoring/rolling-capacity-receipt.json"
    raw = json.loads(rolling_path.read_text())
    raw["planned_at"] = "2026-08-28T12:00:01Z"
    unsigned = dict(raw)
    unsigned.pop("receipt_sha256")
    raw["receipt_sha256"] = canonical_sha256(unsigned)
    rolling_path.write_text(json.dumps(raw, sort_keys=True) + "\n")

    with pytest.raises(PhaseCheckpointError, match="scheduler binding differs"):
        publish_adaptive_checkpoint(
            tmp_path / "mismatched",
            tmp_path / "mismatched-published",
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_adaptive_checkpoint_requires_all_authoritative_state_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source)
    (source / "monitoring/adaptive-progress.json").unlink()

    with pytest.raises(PhaseCheckpointError, match="adaptive-progress"):
        publish_adaptive_checkpoint(
            source,
            tmp_path / "published",
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_adaptive_checkpoint_rejects_progress_ahead_of_authoritative_journal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source)
    raw = json.loads((source / "monitoring/adaptive-progress.json").read_text())
    raw["progress"]["completed_search_trials"] = 88
    raw["progress"]["current_batch"] = 11
    unsigned = dict(raw)
    unsigned.pop("checkpoint_sha256")
    raw["checkpoint_sha256"] = _sha(unsigned)
    (source / "monitoring/adaptive-progress.json").write_text(json.dumps(raw))

    with pytest.raises(PhaseCheckpointError, match="differs from authoritative"):
        publish_adaptive_checkpoint(
            source,
            tmp_path / "published",
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_adaptive_checkpoint_rejects_noncanonical_accounted_spend(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source)
    path = source / "study/adaptive-run-checkpoint.json"
    raw = json.loads(path.read_text())
    raw["accounted_evaluation_usd"] = "1.00"
    unsigned = dict(raw)
    unsigned.pop("checkpoint_sha256")
    raw["checkpoint_sha256"] = _sha(unsigned)
    path.write_text(json.dumps(raw) + "\n")

    with pytest.raises(PhaseCheckpointError, match="accounted spend"):
        publish_adaptive_checkpoint(
            source,
            tmp_path / "published",
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_adaptive_checkpoint_rejects_lagging_progress(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_adaptive_state(source, progress_completed=72)

    with pytest.raises(PhaseCheckpointError, match="differs from authoritative"):
        publish_adaptive_checkpoint(
            source,
            tmp_path / "published",
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_adaptive_checkpoint_requires_progress_hash_chain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    _write_adaptive_state(source)
    first = publish_adaptive_checkpoint(
        source,
        publication,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    path = source / "monitoring/adaptive-progress.json"
    opened = open_adaptive_progress_checkpoint(path)
    advanced = advance_adaptive_progress_checkpoint(
        path,
        replace(
            opened.progress,
            elapsed_seconds=1300.0,
        ),
    )
    second = publish_adaptive_checkpoint(
        source,
        publication,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    assert second["parent_manifest_sha256"] == first["manifest_sha256"]
    assert second["adaptive_progress_checkpoint_sha256"] == advanced.checkpoint_sha256

    unlinked = tmp_path / "unlinked.json"
    advance_adaptive_progress_checkpoint(
        unlinked, replace(advanced.progress, elapsed_seconds=1400.0)
    )
    path.write_bytes(unlinked.read_bytes())
    with pytest.raises(PhaseCheckpointError, match="hash-chain"):
        publish_adaptive_checkpoint(
            source,
            publication,
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_adaptive_checkpoint_rejects_rolling_capacity_lineage_regression(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    _write_adaptive_state(source)
    publish_adaptive_checkpoint(
        source,
        publication,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )

    rolling_path = source / "monitoring/rolling-capacity-receipt.json"
    rolling = json.loads(rolling_path.read_text())
    rolling["completed_through_trial"] = 72
    rolling["source_batch_observation_sha256"] = "9" * 64
    rolling_unsigned = dict(rolling)
    rolling_unsigned.pop("receipt_sha256")
    rolling["receipt_sha256"] = canonical_sha256(rolling_unsigned)
    rolling_path.write_text(json.dumps(rolling, sort_keys=True) + "\n")

    scheduler_path = source / "study/adaptive-run-checkpoint.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["current_capacity_receipt_sha256"] = rolling["receipt_sha256"]
    scheduler["current_capacity_observation_sha256"] = rolling[
        "source_batch_observation_sha256"
    ]
    scheduler["current_capacity_completed_through_trial"] = 72
    scheduler_unsigned = dict(scheduler)
    scheduler_unsigned.pop("checkpoint_sha256")
    scheduler["checkpoint_sha256"] = canonical_sha256(scheduler_unsigned)
    scheduler_path.write_text(json.dumps(scheduler, sort_keys=True) + "\n")

    with pytest.raises(PhaseCheckpointError, match="capacity boundary"):
        publish_adaptive_checkpoint(
            source,
            publication,
            expected_study_identity_sha256=STUDY_ID,
            expected_study_config_sha256=STUDY_CONFIG_SHA,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_adaptive_checkpoint_cli_publishes_and_restores(tmp_path: Path) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    restored = tmp_path / "restored"
    _write_adaptive_state(source)
    script = Path(__file__).parents[1] / "scripts/transfer_truth_editing_phase_checkpoint.py"
    shared = [
        "--state-dir",
        str(source),
        "--publication-root",
        str(publication),
        "--study-identity-sha256",
        STUDY_ID,
        "--study-config-sha256",
        STUDY_CONFIG_SHA,
        "--optuna-study-name",
        OPTUNA_STUDY_NAME,
        "--completed-trials",
        "80",
    ]
    published = subprocess.run(
        [sys.executable, str(script), "publish-adaptive", *shared],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(published.stdout)["completed_trials"] == 80

    restore_args = list(shared)
    restore_args[restore_args.index(str(source))] = str(restored)
    result = subprocess.run(
        [sys.executable, str(script), "restore-adaptive", *restore_args],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["completed_trials"] == 80


def test_adaptive_checkpoint_accepts_configured_expanded_tier_through_200(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-expanded-200"
    _write_adaptive_state(source, completed=168, expanded_through=200)
    manifest = publish_adaptive_checkpoint(
        source,
        tmp_path / "published-expanded-200",
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=168,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
        tier_through_trials=(80, 200, 800),
    )
    assert manifest["completed_trials"] == 168


def test_adaptive_checkpoint_accepts_aggressive_single_discovery_tier(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-aggressive"
    _write_adaptive_state(source, completed=8, single_tier=True)

    manifest = publish_adaptive_checkpoint(
        source,
        tmp_path / "published-aggressive",
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=8,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
        tier_through_trials=(64,),
        trial_number_start=0,
    )

    assert manifest["completed_trials"] == 8
