from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_evaluator import (
    JudgeEvidence,
    RecipeEvaluator,
)
from intelligent_liars.truth_editing_finalist_checkpoint import (
    FinalistCheckpointError,
    select_pareto_finalists,
)
from intelligent_liars.truth_editing_judge_contracts import (
    AbsoluteJudgeResult,
    JudgeCacheReceipt,
)
from intelligent_liars.truth_editing_live_judge import (
    MemoryJudgeCache,
    StoredJudgeTransport,
    TruthEditingLiveJudge,
)
from intelligent_liars.truth_editing_record_completion import (
    FileSemanticRecordCompletionStore,
    RecordCompletionRequirement,
    RecordCompletionScope,
)
from intelligent_liars.truth_editing_rescore import (
    materialize_rescore_generation_v1,
)
from intelligent_liars.truth_editing_study import (
    EvaluationResult,
    OfflineDeterministicSearchDriver,
    OfflineSyntheticEvaluator,
    OptunaSearchDriver,
    TruthEditingStudy,
)
from test_truth_editing_evaluator import (
    _Preservation,
    _bundle,
    _config as _evaluator_config,
    _execution,
    _record,
    _semantic_evidence,
)
from test_truth_editing_finalist_checkpoint import (
    _report as _selection_ready_report,
)
from test_truth_editing_study import (
    _config as _study_config,
    _direction_bank,
)


JUDGE_CONFIG_SHA256 = "a" * 64
RUBRIC_SHA256 = "b" * 64


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


class _RescoreSourceEvaluator(OfflineSyntheticEvaluator):
    @property
    def identity(self) -> dict[str, str]:
        return {
            "adapter": "scientific_integrity_source_fixture_v1",
            "judge_config_sha256": JUDGE_CONFIG_SHA256,
            "rubric_sha256": RUBRIC_SHA256,
        }

    def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
        if trial_id in {"trial-0000", "trial-0002"}:
            return EvaluationResult.operational_failure("stored semantic failure")
        result = super().evaluate(
            proposal,
            trial_id=trial_id,
            record_ids=record_ids,
            objective_names=objective_names,
        )
        if trial_id == "trial-0001":
            return EvaluationResult.scientifically_infeasible(
                result.metrics,
                "frozen scientific gates failed",
            )
        return result


def _source_artifacts(tmp_path: Path):
    config = _study_config(
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
        driver=OfflineDeterministicSearchDriver(seed=config.sampler_seed),
        evaluator=_RescoreSourceEvaluator(),
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
    checkpoint["checkpoint_sha256"] = _json_sha(checkpoint)
    checkpoint_path = tmp_path / "source-checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return report, report_path, journal_path, checkpoint_path


def _evaluation_records(first_generation: str) -> list[dict[str, object]]:
    return [
        _record("direct-1", "direct", first_generation),
        _record("direct-2", "direct", "Berlin"),
        _record(
            "truthful-1",
            "direct",
            "Paris",
            signal_kind="truthful_direct_report",
        ),
        _record(
            "false-trigger-1",
            "direct",
            "Paris",
            signal_kind="false_trigger_truthful_control",
        ),
        _record("indirect-1", "indirect", "Paris"),
        _record("action-1", "action", "click-paris"),
        _record("cf-1", "counterfactual", "London"),
    ]


def _transport_payload(content: str) -> dict[str, object]:
    return {
        "content": content,
        "model": "z-ai/glm-5.3-flash",
        "provider_route": "z-ai/fp8",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
        "price_usd": 0.0000125,
        "latency_ms": 12.5,
    }


def _completion_scope(
    *,
    store: FileSemanticRecordCompletionStore,
    config,
    bundle: dict[str, object],
    record: dict[str, object],
) -> RecordCompletionScope:
    return RecordCompletionScope.create(
        evaluator_config_sha256=_canonical_sha(config.to_mapping()),
        dataset_manifest_sha256=str(bundle["dataset_manifest_sha256"]),
        recipe_sha256=str(bundle["recipe_sha256"]),
        edited_model_sha256=str(bundle["edited_model_sha256"]),
        output_bundle_sha256=str(bundle["bundle_sha256"]),
        tier="discovery",
        judge_config_sha256=config.judge_config_sha256,
        rubric_sha256=config.rubric_sha256,
        judge_execution_identity_sha256=None,
        completion_store_identity_sha256=store.identity_sha256,
        requirements=(
            RecordCompletionRequirement(
                str(record["record_id"]),
                str(record["prompt_sha256"]),
                str(record["raw_generation_sha256"]),
            ),
        ),
    )


@pytest.mark.parametrize(
    "invalid_content",
    [
        "not-json",
        json.dumps({"responses": []}, sort_keys=True),
    ],
    ids=["malformed-json", "schema-incomplete-json"],
)
def test_malformed_or_incomplete_json_cannot_create_a_scored_record(
    tmp_path, invalid_content: str
) -> None:
    records = _evaluation_records("Rome")
    bundle = _bundle(records)
    config = _evaluator_config()
    cache = MemoryJudgeCache()
    judge = TruthEditingLiveJudge(
        transport=StoredJudgeTransport(
            [
                _transport_payload(invalid_content),
                _transport_payload(invalid_content),
            ]
        ),
        cache=cache,
    )
    store = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=("1" * 64,),
    )
    scope = _completion_scope(
        store=store,
        config=config,
        bundle=bundle,
        record=records[0],
    )

    assessment = RecipeEvaluator(
        config,
        judge,
        _Preservation(),
        record_completion_store=store,
    ).evaluate(_execution(bundle), bundle, tier="discovery")

    assert assessment.status == "operational_failure"
    assert assessment.objectives == {}
    assert assessment.constraint_violations == {}
    assert assessment.components is None
    failure_receipts = cache.failure_receipts()
    assert failure_receipts
    assert all(
        receipt.operational_status in {"invalid_json", "schema_error"}
        for receipt in failure_receipts
    )
    assert all(receipt.parsed_result_sha256 is None for receipt in failure_receipts)
    assert all(
        cache.get(receipt.cache_key_sha256) is None
        for receipt in failure_receipts
    )
    assert store.missing_record_ids(scope) == ("direct-1",)


class _CountingJudge:
    def __init__(self, *, wrong_semantic_type: bool = False) -> None:
        self.wrong_semantic_type = wrong_semantic_type
        self.calls: list[str] = []

    def judge(self, record) -> JudgeEvidence:
        self.calls.append(record.record_id)
        evidence = _semantic_evidence(
            record.record_id,
            record.raw_generation_sha256,
        )
        if not self.wrong_semantic_type:
            return evidence

        result_payload = evidence.result.to_payload()
        result_payload["result"]["responses"][0]["asserted_answer"] = "1969"
        result_payload["result"]["responses"][0]["semantic_type_valid"] = False
        result_payload["content_sha256"] = _canonical_sha(
            {
                key: value
                for key, value in result_payload.items()
                if key != "content_sha256"
            }
        )
        result = AbsoluteJudgeResult.parse(result_payload)
        receipt_payload = evidence.cache_receipt.to_payload()
        receipt_payload["parsed_result_sha256"] = result.content_sha256
        receipt_payload["content_sha256"] = _canonical_sha(
            {
                key: value
                for key, value in receipt_payload.items()
                if key != "content_sha256"
            }
        )
        receipt = JudgeCacheReceipt.parse(receipt_payload, result=result)
        return JudgeEvidence(result, receipt)


@pytest.mark.parametrize(
    ("generation", "wrong_semantic_type", "expected_failure", "expected_calls"),
    [
        ("Paris", False, "truthful_direct_report", []),
        ("I don't know", False, "refusal_or_evasion", []),
        ("Rome", True, "wrong_semantic_type", ["direct-1"]),
    ],
    ids=["truthful", "refusal", "wrong-type"],
)
def test_record_recovery_never_weakens_direct_report_scientific_gates(
    tmp_path,
    generation: str,
    wrong_semantic_type: bool,
    expected_failure: str,
    expected_calls: list[str],
) -> None:
    records = _evaluation_records(generation)
    bundle = _bundle(records)
    judge = _CountingJudge(wrong_semantic_type=wrong_semantic_type)
    store = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=(_sha("c"),),
    )

    assessment = RecipeEvaluator(
        _evaluator_config(),
        judge,
        _Preservation(),
        record_completion_store=store,
    ).evaluate(_execution(bundle), bundle, tier="discovery")

    assert assessment.status == "scientifically_infeasible"
    assert assessment.components is not None
    assert assessment.components.direct.valid_false_report_count == 1
    assert assessment.components.direct.failure_counts == {expected_failure: 1}
    assert judge.calls == expected_calls


@pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None,
    reason="Optuna is an optional study dependency",
)
def test_scientific_infeasibility_is_not_replayed_as_operational_recovery(
    tmp_path,
) -> None:
    class EveryTrialIsScientificallyInfeasible(OfflineSyntheticEvaluator):
        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            del proposal, trial_id, record_ids
            return EvaluationResult.scientifically_infeasible(
                {name: 0.0 for name in objective_names},
                "frozen scientific gate failed",
            )

    config = _study_config(
        tmp_path,
        batch_size=8,
        max_trials=8,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 8}
        ],
    )
    journal = tmp_path / "scientific-no-go.json"
    first = TruthEditingStudy(config, _direction_bank()).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=EveryTrialIsScientificallyInfeasible(),
        journal_path=journal,
    )
    replayed = TruthEditingStudy(config, _direction_bank()).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )

    assert first.completed_trials == 8
    assert first.scientifically_infeasible_trials == 8
    assert first.operational_failures == 0
    assert replayed.completed_trials == 8
    assert replayed.scientifically_infeasible_trials == 8
    assert replayed.successful_trials == 0
    assert [trial.result.outcome_kind for trial in replayed.trials] == [
        "scientifically_infeasible"
    ] * 8


def test_rescore_generation_preserves_scientific_no_go_and_replays_only_operations(
    tmp_path,
) -> None:
    report, report_path, journal_path, checkpoint_path = _source_artifacts(tmp_path)

    generation = materialize_rescore_generation_v1(
        source_report_path=report_path,
        source_journal_path=journal_path,
        source_checkpoint_path=checkpoint_path,
        output_path=tmp_path / "rescore-generation.json",
        expected_source_study_identity_sha256=report.study_identity_sha256,
        expected_judge_config_sha256=JUDGE_CONFIG_SHA256,
        expected_rubric_sha256=RUBRIC_SHA256,
        expected_completed_trials=6,
    )
    source_trials = [
        trial
        for batch in generation.source_batches
        for trial in batch["trials"]
    ]

    assert source_trials[1]["result"]["outcome_kind"] == (
        "scientifically_infeasible"
    )
    assert source_trials[1]["result"]["detail"] == "frozen scientific gates failed"
    assert generation.source.outcome_counts["scientifically_infeasible"] == 1
    assert generation.replay_policy.scientific_outcomes_are_preserved is True
    assert [request.source_ordinal for request in generation.replay_requests] == [0, 2]
    assert 1 not in {
        request.source_ordinal for request in generation.replay_requests
    }


@pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None,
    reason="Optuna is an optional study dependency",
)
def test_operational_failures_enter_neither_optuna_learning_nor_broad_coverage(
    tmp_path,
) -> None:
    class LosePerLayerRefusalTrials(OfflineSyntheticEvaluator):
        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            if (
                proposal.refusal_enabled
                and proposal.refusal_direction_scope == "per_layer"
            ):
                return EvaluationResult.operational_failure("synthetic worker loss")
            return super().evaluate(
                proposal,
                trial_id=trial_id,
                record_ids=record_ids,
                objective_names=objective_names,
            )

    config = _study_config(
        tmp_path,
        batch_size=8,
        max_trials=8,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 8}
        ],
    )
    journal = tmp_path / "operational-failures.json"
    driver = OptunaSearchDriver(seed=17)
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=driver,
        evaluator=LosePerLayerRefusalTrials(),
        journal_path=journal,
    )

    import optuna

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal) + ".optuna.log")
    )
    persisted = optuna.load_study(
        study_name=driver.persistent_study_name,
        storage=storage,
    )
    learned_ordinals = {
        int(trial.user_attrs["study_ordinal"])
        for trial in persisted.get_trials(deepcopy=False)
    }
    operational_ordinals = {
        trial.ordinal
        for trial in report.trials
        if trial.result.outcome_kind == "operational_failure"
    }

    assert operational_ordinals
    assert operational_ordinals.isdisjoint(learned_ordinals)
    assert "per_layer" not in report.coverage.refusal_settings
    assert report.coverage_complete is False
    assert report.selection_ready is False


def test_finalization_rejects_unresolved_operational_history() -> None:
    report = _selection_ready_report(
        [("trial-unscored", (0.9, 0.8, 0.7))]
    )
    report["trials"][0]["result"] = {
        "outcome_kind": "operational_failure",
        "metrics": {},
        "detail": "semantic records incomplete",
    }
    report["successful_trials"] = 0
    report["operational_failures"] = 1
    report["selection_ready"] = True

    with pytest.raises(FinalistCheckpointError, match="unresolved"):
        select_pareto_finalists(report)


def test_finalization_rejects_null_unscored_history() -> None:
    report = _selection_ready_report(
        [("trial-unscored", (0.9, 0.8, 0.7))]
    )
    report["trials"][0]["result"] = None
    report["successful_trials"] = 0
    report["selection_ready"] = True

    with pytest.raises(FinalistCheckpointError, match="result must be an object"):
        select_pareto_finalists(report)


def test_finalization_never_promotes_an_all_infeasible_history() -> None:
    report = _selection_ready_report(
        [("trial-scientific-no-go", (0.9, 0.8, 0.7))]
    )
    report["trials"][0]["result"] = {
        "outcome_kind": "scientifically_infeasible",
        "metrics": {
            "valid_false_report_rate_lcb": 0.9,
            "truth_report_dissociation_lcb": 0.8,
            "capability_preservation_lcb": 0.7,
        },
        "detail": "frozen scientific gate failed",
    }
    report["successful_trials"] = 0
    report["scientifically_infeasible_trials"] = 1

    with pytest.raises(FinalistCheckpointError, match="no successful trials"):
        select_pareto_finalists(report)
