from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_wandb_checkpoint import (
    ADAPTIVE_PROGRESS_FORMAT,
    AdaptiveRunProgress,
    WandbCheckpointError,
    advance_adaptive_progress_checkpoint,
    create_wandb_run_checkpoint,
    open_adaptive_progress_checkpoint,
    open_wandb_run_checkpoint,
)


def test_wandb_run_checkpoint_round_trips_one_coordinator_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "monitoring/wandb-run.json"

    created = create_wandb_run_checkpoint(
        path,
        run_id="ab12cd34",
        project="intelligent-liars",
        entity="truth-editing",
    )
    opened = open_wandb_run_checkpoint(path)

    assert opened == created
    assert opened.run_id == "ab12cd34"
    assert len(opened.checkpoint_sha256) == 64
    assert json.loads(path.read_text()) == opened.to_mapping()


def test_wandb_run_checkpoint_is_idempotent_only_for_the_same_run(
    tmp_path: Path,
) -> None:
    path = tmp_path / "monitoring/wandb-run.json"
    first = create_wandb_run_checkpoint(
        path,
        run_id="ab12cd34",
        project="intelligent-liars",
        entity=None,
    )
    assert (
        create_wandb_run_checkpoint(
            path,
            run_id="ab12cd34",
            project="intelligent-liars",
            entity=None,
        )
        == first
    )

    with pytest.raises(WandbCheckpointError, match="different W&B run identity"):
        create_wandb_run_checkpoint(
            path,
            run_id="different1",
            project="intelligent-liars",
            entity=None,
        )


def test_wandb_run_checkpoint_rejects_tampering_and_multiple_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "monitoring/wandb-run.json"
    create_wandb_run_checkpoint(
        path,
        run_id="ab12cd34",
        project="intelligent-liars",
        entity=None,
    )
    payload = json.loads(path.read_text())
    payload["run_id"] = ["ab12cd34", "different1"]
    path.write_text(json.dumps(payload))

    with pytest.raises(WandbCheckpointError, match="run ID|hash"):
        open_wandb_run_checkpoint(path)


def test_wandb_run_checkpoint_rejects_duplicate_json_run_id_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "monitoring/wandb-run.json"
    create_wandb_run_checkpoint(
        path,
        run_id="ab12cd34",
        project="intelligent-liars",
        entity=None,
    )
    payload = path.read_text().replace(
        "{\n", '{\n  "run_id": "different1",\n', 1
    )
    path.write_text(payload)

    with pytest.raises(WandbCheckpointError, match="duplicate"):
        open_wandb_run_checkpoint(path)


@pytest.mark.parametrize(
    "run_id",
    ["", " leading", "two ids", "a/b", "a" * 129],
)
def test_wandb_run_checkpoint_rejects_unsafe_run_ids(
    tmp_path: Path, run_id: str
) -> None:
    with pytest.raises(WandbCheckpointError, match="run ID"):
        create_wandb_run_checkpoint(
            tmp_path / "wandb-run.json",
            run_id=run_id,
            project="intelligent-liars",
            entity=None,
        )


def _adaptive_progress(
    wandb_sha256: str,
    *,
    completed_search_trials: int = 8,
    measured_target_trials: int = 456,
    stage: str = "broad_coverage",
) -> AdaptiveRunProgress:
    return AdaptiveRunProgress(
        wandb_run_checkpoint_sha256=wandb_sha256,
        study_config_sha256="c" * 64,
        planned_floor_trials=200,
        adaptive_ceiling_trials=800,
        measured_target_trials=measured_target_trials,
        batch_size=8,
        search_cutoff_seconds=21 * 3600,
        reserve_seconds=3 * 3600,
        total_budget_usd=50.0,
        evaluation_budget_usd=5.0,
        evaluation_budget_reserve_fraction=0.2,
        completed_search_trials=completed_search_trials,
        completed_repeat_trials=0,
        completed_control_trials=0,
        completed_final_selection_trials=0,
        current_batch=completed_search_trials // 8,
        stage=stage,
        coverage={
            "direction_family": (3, 18),
            "layer_region": (2, 3),
            "intervention_arm": (1, 3),
            "attention_mlp_configuration": (2, 4),
            "refusal_setting": (1, 2),
            "strength_range": (2, 4),
        },
        elapsed_seconds=600.0,
        eta_seconds=12000.0,
        gpu_actual_usd=0.4,
        gpu_projected_usd=22.0,
        judge_actual_usd=0.02,
        judge_projected_usd=2.5,
        projected_total_usd=24.5,
        measured_trial_duration_seconds=88.0,
        measured_tokens_per_second=41.0,
        measured_judge_latency_ms=1300.0,
        measured_judge_cost_usd_per_trial=0.002,
    )


def test_adaptive_progress_checkpoint_round_trips_and_hash_chains_updates(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "monitoring/wandb-run.json"
    run = create_wandb_run_checkpoint(
        run_path,
        run_id="adaptive-run",
        project="intelligent-liars",
        entity="centipawn",
    )
    path = tmp_path / "monitoring/adaptive-progress.json"

    first = advance_adaptive_progress_checkpoint(
        path, _adaptive_progress(run.checkpoint_sha256)
    )
    second = advance_adaptive_progress_checkpoint(
        path,
        _adaptive_progress(
            run.checkpoint_sha256,
            completed_search_trials=16,
            measured_target_trials=512,
        ),
    )

    assert first.revision == 0
    assert second.revision == 1
    assert second.previous_checkpoint_sha256 == first.checkpoint_sha256
    assert open_adaptive_progress_checkpoint(path) == second
    assert json.loads(path.read_text())["format"] == ADAPTIVE_PROGRESS_FORMAT


def test_adaptive_progress_checkpoint_rejects_policy_drift_and_regression(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adaptive-progress.json"
    first = _adaptive_progress("a" * 64)
    advance_adaptive_progress_checkpoint(path, first)

    with pytest.raises(WandbCheckpointError, match="policy differs"):
        replace(first, planned_floor_trials=208)

    regressed = replace(first, completed_search_trials=0, current_batch=0)
    with pytest.raises(WandbCheckpointError, match="regressed"):
        advance_adaptive_progress_checkpoint(path, regressed)


def test_incomplete_coverage_allows_abort_but_not_adaptive_search(
    tmp_path: Path,
) -> None:
    """An operational abort is terminal reporting, not search concentration."""

    aborted = _adaptive_progress("a" * 64, stage="aborted")
    checkpoint = advance_adaptive_progress_checkpoint(
        tmp_path / "aborted-progress.json", aborted
    )

    assert checkpoint.progress == aborted
    with pytest.raises(WandbCheckpointError, match="coverage"):
        _adaptive_progress("a" * 64, stage="adaptive_search")


def test_adaptive_progress_checkpoint_rejects_tampering_and_unknown_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adaptive-progress.json"
    advance_adaptive_progress_checkpoint(path, _adaptive_progress("b" * 64))
    raw = json.loads(path.read_text())
    raw["progress"]["measured_target_trials"] = 800
    raw["surprise"] = True
    path.write_text(json.dumps(raw))

    with pytest.raises(WandbCheckpointError, match="fields differ|hash differs"):
        open_adaptive_progress_checkpoint(path)
