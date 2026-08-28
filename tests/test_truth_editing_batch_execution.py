from __future__ import annotations

import json

import pytest

from intelligent_liars.truth_editing_study import (
    EvaluationResult,
    OfflineDeterministicSearchDriver,
    OperationalEvaluationError,
    StudyError,
    TruthEditingStudy,
)

from test_truth_editing_study import _config, _direction_bank


class _BatchOnlyEvaluator:
    def __init__(self) -> None:
        self.batch_trial_ids: list[tuple[str, ...]] = []

    @property
    def identity(self):
        return {"adapter": "batch-only-v1"}

    def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
        raise AssertionError("the scalar fallback must not run")

    def evaluate_batch(self, requests):
        self.batch_trial_ids.append(tuple(item.trial_id for item in requests))
        return tuple(
            EvaluationResult.successful({name: float(item.ordinal) for name in item.objective_names})
            for item in requests
        )


def test_batch_capable_evaluator_runs_once_per_synchronous_proposal_barrier(tmp_path) -> None:
    config = _config(
        tmp_path,
        max_trials=4,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 4}],
    )
    evaluator = _BatchOnlyEvaluator()
    driver = OfflineDeterministicSearchDriver(seed=17)

    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=driver,
        evaluator=evaluator,
        journal_path=tmp_path / "journal.json",
    )

    assert evaluator.batch_trial_ids == [
        ("trial-0000", "trial-0001"),
        ("trial-0002", "trial-0003"),
    ]
    assert driver.observed_batch_sizes == [2, 2]
    assert [trial.ordinal for trial in report.trials] == [0, 1, 2, 3]


def test_batch_result_count_must_match_pending_request_count(tmp_path) -> None:
    class MissingResult(_BatchOnlyEvaluator):
        def evaluate_batch(self, requests):
            return (EvaluationResult.successful({name: 0.0 for name in requests[0].objective_names}),)

    config = _config(
        tmp_path,
        max_trials=2,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 2}],
    )
    with pytest.raises(StudyError, match="one ordered result per request"):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=MissingResult(),
            journal_path=tmp_path / "journal.json",
        )


def test_batch_results_are_persisted_individually_and_resume_only_pending_work(tmp_path) -> None:
    class InterruptAfterFirstResultSave(TruthEditingStudy):
        def __init__(self, config, bank):
            super().__init__(config, bank)
            self.saves = 0

        def _save(self, path, raw):
            super()._save(path, raw)
            self.saves += 1
            # initial journal, proposed batch, first durable outcome
            if self.saves == 3:
                raise KeyboardInterrupt

    config = _config(
        tmp_path,
        max_trials=2,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 2}],
    )
    journal_path = tmp_path / "journal.json"
    first = _BatchOnlyEvaluator()
    with pytest.raises(KeyboardInterrupt):
        InterruptAfterFirstResultSave(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=first,
            journal_path=journal_path,
        )

    interrupted = json.loads(journal_path.read_text())
    assert interrupted["batches"][0]["trials"][0]["result"] is not None
    assert interrupted["batches"][0]["trials"][1]["result"] is None

    resumed = _BatchOnlyEvaluator()
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=resumed,
        journal_path=journal_path,
    )
    assert resumed.batch_trial_ids == [("trial-0001",)]
    assert [trial.ordinal for trial in report.trials] == [0, 1]


def test_streamed_batch_failure_keeps_earlier_result_and_completes_barrier(tmp_path) -> None:
    class FailingStream(_BatchOnlyEvaluator):
        def evaluate_batch(self, requests):
            yield EvaluationResult.successful(
                {name: 1.0 for name in requests[0].objective_names}
            )
            raise OperationalEvaluationError("worker stream lost")

    config = _config(
        tmp_path,
        max_trials=2,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 2}],
    )
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=FailingStream(),
        journal_path=tmp_path / "journal.json",
    )

    assert [item.result.outcome_kind for item in report.trials] == [
        "successful",
        "operational_failure",
    ]
    assert report.trials[1].result.detail == "worker stream lost"


def test_search_observes_batch_only_after_every_outcome_is_durable(tmp_path) -> None:
    journal_path = tmp_path / "journal.json"

    class JournalInspectingDriver(OfflineDeterministicSearchDriver):
        def observe(self, trials):
            raw = json.loads(journal_path.read_text())
            batch = raw["batches"][trials[0].batch_ordinal]
            assert all(entry["result"] is not None for entry in batch["trials"])
            super().observe(trials)

    config = _config(
        tmp_path,
        max_trials=2,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 2}],
    )
    driver = JournalInspectingDriver(seed=17)
    TruthEditingStudy(config, _direction_bank()).run(
        driver=driver,
        evaluator=_BatchOnlyEvaluator(),
        journal_path=journal_path,
    )
    assert driver.observed_batch_sizes == [2]
