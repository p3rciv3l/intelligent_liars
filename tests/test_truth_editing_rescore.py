from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_rescore import (
    RESCORE_GENERATION_FORMAT,
    RescoreGenerationError,
    RescoreOptunaSearchDriver,
    load_rescore_generation_v1,
    main,
    materialize_rescore_generation_v1,
)
from intelligent_liars.truth_editing_study import (
    EvaluationResult,
    OfflineDeterministicSearchDriver,
    OfflineSyntheticEvaluator,
    OptunaSearchDriver,
    TruthEditingStudy,
)

from test_truth_editing_study import _config, _direction_bank


JUDGE_CONFIG_SHA256 = "a" * 64
RUBRIC_SHA256 = "b" * 64


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


class _SourceEvaluator(OfflineSyntheticEvaluator):
    @property
    def identity(self):
        return {
            "adapter": "rescore_source_fixture_v1",
            "judge_config_sha256": JUDGE_CONFIG_SHA256,
            "rubric_sha256": RUBRIC_SHA256,
        }

    def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
        if trial_id in {"trial-0001", "trial-0002"}:
            return EvaluationResult.operational_failure("stored semantic failure")
        result = super().evaluate(
            proposal,
            trial_id=trial_id,
            record_ids=record_ids,
            objective_names=objective_names,
        )
        if trial_id == "trial-0003":
            return EvaluationResult.scientifically_infeasible(
                result.metrics, "frozen scientific gates failed"
            )
        return result


def _source_artifacts(tmp_path: Path, *, optuna: bool = False):
    config = _config(
        tmp_path,
        batch_size=2,
        max_trials=12,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 2},
            {"name": "expanded", "record_limit": 4, "through_trial": 4},
            {"name": "finalist", "record_limit": 6, "through_trial": 12},
        ],
    )
    journal_path = tmp_path / "source-journal.json"
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=(
            OptunaSearchDriver(seed=config.sampler_seed)
            if optuna
            else OfflineDeterministicSearchDriver(seed=config.sampler_seed)
        ),
        evaluator=_SourceEvaluator(),
        journal_path=journal_path,
        stop_after_trials=6,
    )
    report_path = tmp_path / "source-report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    checkpoint = {
        "format": "truth_editing_adaptive_run_checkpoint_v1",
        "study_identity_sha256": report.study_identity_sha256,
        "authorized_through_trial": report.completed_trials,
        "completed_trials": report.completed_trials,
        "coverage_complete": False,
        "phase": "finalization_reserved",
        "stop_reason": "evaluation_budget_reserve_reached",
    }
    checkpoint["checkpoint_sha256"] = _canonical_sha256(checkpoint)
    checkpoint_path = tmp_path / "source-checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return report, report_path, journal_path, checkpoint_path


def test_materialize_v1_preserves_source_and_queues_unresolved_requests_fifo(
    tmp_path: Path,
) -> None:
    report, report_path, journal_path, checkpoint_path = _source_artifacts(tmp_path)
    source_bytes = {
        path: path.read_bytes()
        for path in (report_path, journal_path, checkpoint_path)
    }
    output_path = tmp_path / "rescore-generation-v1.json"

    generation = materialize_rescore_generation_v1(
        source_report_path=report_path,
        source_journal_path=journal_path,
        source_checkpoint_path=checkpoint_path,
        output_path=output_path,
        expected_source_study_identity_sha256=report.study_identity_sha256,
        expected_judge_config_sha256=JUDGE_CONFIG_SHA256,
        expected_rubric_sha256=RUBRIC_SHA256,
        expected_completed_trials=6,
    )

    assert generation.format == RESCORE_GENERATION_FORMAT
    assert generation.source.completed_trials == 6
    assert [item.source_ordinal for item in generation.replay_requests] == [1, 2]
    assert [item.source_tier_name for item in generation.replay_requests] == [
        "discovery",
        "expanded",
    ]
    assert generation.source.outcome_counts == {
        "operational_failure": 2,
        "scientifically_infeasible": 1,
        "successful": 3,
    }
    assert generation.replay_policy.attempts_per_request == 1
    assert generation.replay_policy.requeue_within_generation is False
    assert generation.replay_policy.quarantine_after_failure is True
    assert output_path.is_file()
    assert {
        path: path.read_bytes()
        for path in (report_path, journal_path, checkpoint_path)
    } == source_bytes


def test_open_v1_revalidates_generation_and_frozen_identities(tmp_path: Path) -> None:
    report, report_path, journal_path, checkpoint_path = _source_artifacts(tmp_path)
    output_path = tmp_path / "rescore-generation-v1.json"
    created = materialize_rescore_generation_v1(
        source_report_path=report_path,
        source_journal_path=journal_path,
        source_checkpoint_path=checkpoint_path,
        output_path=output_path,
        expected_source_study_identity_sha256=report.study_identity_sha256,
        expected_judge_config_sha256=JUDGE_CONFIG_SHA256,
        expected_rubric_sha256=RUBRIC_SHA256,
        expected_completed_trials=6,
    )

    opened = load_rescore_generation_v1(
        output_path,
        expected_generation_sha256=created.generation_sha256,
        expected_source_study_identity_sha256=report.study_identity_sha256,
        expected_judge_config_sha256=JUDGE_CONFIG_SHA256,
        expected_rubric_sha256=RUBRIC_SHA256,
    )

    assert opened.to_dict() == created.to_dict()
    with pytest.raises(RescoreGenerationError, match="judge"):
        load_rescore_generation_v1(
            output_path,
            expected_judge_config_sha256="c" * 64,
        )

    corrupted = json.loads(output_path.read_text())
    corrupted["replay_requests"][0]["evaluation_record_ids"].append("not-frozen")
    output_path.write_text(json.dumps(corrupted))
    with pytest.raises(RescoreGenerationError, match="identity"):
        load_rescore_generation_v1(output_path)


def test_cli_creates_a_new_lineage_directory_without_clobbering_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report, report_path, journal_path, checkpoint_path = _source_artifacts(tmp_path)
    sources = {
        path: path.read_bytes()
        for path in (report_path, journal_path, checkpoint_path)
    }
    output_directory = tmp_path / "rescore-lineage-v1"
    arguments = [
        "--source-report", str(report_path),
        "--source-journal", str(journal_path),
        "--source-checkpoint", str(checkpoint_path),
        "--output-directory", str(output_directory),
        "--source-study-identity-sha256", report.study_identity_sha256,
        "--judge-config-sha256", JUDGE_CONFIG_SHA256,
        "--rubric-sha256", RUBRIC_SHA256,
        "--completed-trials", "6",
    ]

    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    generation = load_rescore_generation_v1(
        output_directory / "rescore-generation-v1.json",
        expected_generation_sha256=summary["generation_sha256"],
    )
    restart = json.loads((output_directory / "restart.json").read_text())
    assert restart["generation_sha256"] == generation.generation_sha256
    assert {path: path.read_bytes() for path in sources} == sources

    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_rescore_driver_appends_exact_replays_then_quarantines_and_continues(
    tmp_path: Path,
) -> None:
    report, report_path, source_journal, checkpoint_path = _source_artifacts(
        tmp_path, optuna=True
    )
    generation = materialize_rescore_generation_v1(
        source_report_path=report_path,
        source_journal_path=source_journal,
        source_checkpoint_path=checkpoint_path,
        output_path=tmp_path / "generation.json",
        expected_source_study_identity_sha256=report.study_identity_sha256,
        expected_judge_config_sha256=JUDGE_CONFIG_SHA256,
        expected_rubric_sha256=RUBRIC_SHA256,
        expected_completed_trials=6,
    )

    class TargetEvaluator(OfflineSyntheticEvaluator):
        @property
        def identity(self):
            return {
                "adapter": "rescore_target_fixture_v1",
                "judge_config_sha256": JUDGE_CONFIG_SHA256,
                "rubric_sha256": RUBRIC_SHA256,
            }

        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            if trial_id == "trial-0007":
                return EvaluationResult.operational_failure(
                    "deterministic semantic failure"
                )
            return super().evaluate(
                proposal,
                trial_id=trial_id,
                record_ids=record_ids,
                objective_names=objective_names,
            )

    config = _config(
        tmp_path,
        batch_size=2,
        max_trials=12,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 2},
            {"name": "expanded", "record_limit": 4, "through_trial": 4},
            {"name": "finalist", "record_limit": 6, "through_trial": 12},
        ],
    )
    driver = RescoreOptunaSearchDriver(seed=config.sampler_seed, generation=generation)
    target_journal = tmp_path / "target-journal.json"
    target = TruthEditingStudy(config, _direction_bank()).run(
        driver=driver,
        evaluator=TargetEvaluator(),
        journal_path=target_journal,
        stop_after_trials=10,
    )

    assert target.study_identity_sha256 != report.study_identity_sha256
    assert target.trials[:6] == report.trials
    assert target.trials[6].proposal == report.trials[1].proposal
    assert target.trials[6].tier_name == report.trials[1].tier_name == "discovery"
    assert target.trials[6].evaluation_record_ids == report.trials[1].evaluation_record_ids
    assert target.trials[6].result.outcome_kind == "successful"
    assert target.trials[7].proposal == report.trials[2].proposal
    assert target.trials[7].tier_name == report.trials[2].tier_name == "expanded"
    assert target.trials[7].evaluation_record_ids == report.trials[2].evaluation_record_ids
    assert target.trials[7].result.outcome_kind == "operational_failure"
    assert target.trials[8].proposal not in {
        report.trials[1].proposal,
        report.trials[2].proposal,
    }
    assert driver.quarantined_request_sha256s == frozenset(
        {generation.replay_requests[1].request_sha256}
    )

    import optuna

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(target_journal) + ".optuna.log")
    )
    persisted = optuna.load_study(
        study_name=driver.persistent_study_name,
        storage=storage,
    ).get_trials(deepcopy=False)
    states = {
        int(item.user_attrs["study_ordinal"]): item.state.name
        for item in persisted
    }
    assert 1 not in states
    assert 2 not in states
    assert 7 not in states
    assert states[3] == "PRUNED"
    assert states[6] == "COMPLETE"
