from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import optuna
import pytest

from intelligent_liars.truth_editing_contracts import (
    canonical_sha256,
    parse_direction_bank_manifest,
)
from intelligent_liars.truth_editing_evaluator import (
    EvaluatorConfig,
    JudgeEvidence,
    RecipeEvaluator,
)
from intelligent_liars.truth_editing_finalist_checkpoint import (
    FinalistCheckpointError,
    select_pareto_finalists,
)
from intelligent_liars.truth_editing_judge_contracts import (
    AbsoluteJudgeResult,
    AbsoluteResponseLabel,
    AbsoluteSemanticResult,
    JudgeCacheReceipt,
    TokenUsage,
    judge_cache_key_sha256,
)
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    FileJudgeCache,
    OperationalJudgeFailure,
    StoredJudgeTransport,
    TruthEditingLiveJudge,
)
from intelligent_liars.truth_editing_production_judge_budget import (
    ProductionJudgeBudget,
    ProductionJudgeBudgetConfig,
)
from intelligent_liars.truth_editing_preservation import (
    PRESERVATION_RECEIPT_FORMAT,
    PreservationReceipt,
    StratumPreservationResult,
)
from intelligent_liars.truth_editing_record_completion import (
    FileSemanticRecordCompletionStore,
    RecordCompletionError,
)
from intelligent_liars.truth_editing_study import (
    EvaluationResult,
    OptunaSearchDriver,
    SearchProposal,
    TruthEditingStudy,
    parse_truth_editing_study_config,
)


OBJECTIVES = (
    "valid_false_report_rate_lcb",
    "truth_report_dissociation_lcb",
    "capability_preservation_lcb",
)
RECORD_IDS = (
    "direct-1",
    "direct-2",
    "truthful-1",
    "false-trigger-1",
    "indirect-1",
    "action-1",
    "counterfactual-1",
)
JUDGE_CONFIG_SHA256 = hashlib.sha256(b"integration-judge").hexdigest()
RUBRIC_SHA256 = hashlib.sha256(b"integration-rubric").hexdigest()
ADAPTER_CODE_SHA256 = hashlib.sha256(b"integration-adapter-v1").hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_sha(value: object, *, newline: bool = True) -> str:
    rendered = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if newline:
        rendered += "\n"
    return hashlib.sha256(rendered.encode()).hexdigest()


def _direction_bank() -> Any:
    def direction(index: int, family: str, layer: int) -> dict[str, Any]:
        return {
            "direction_id": f"truth-{family}-{layer}-{index}",
            "kind": "truth",
            "family": family,
            "basis_variant": "raw",
            "domains": ["general" if family == "general" else family],
            "source_layer": layer,
            "width": 4096,
            "rank": 1,
            "artifact": {
                "path": f"directions/{index}.safetensors",
                "file_sha256": f"{index + 1:064x}",
                "vector_sha256": f"{index + 11:064x}",
            },
            "construction": {
                "basis_method": "raw",
                "pooling": "last_token",
                "token_position": "first_generated_token",
                "normalization": "unit_l2",
                "sign_convention": (
                    "sklearn_logistic_coef_positive_points_honest_to_deceptive"
                ),
                "intercept": -0.25,
            },
            "control_provenance": None,
            "provenance": {
                "dataset": "sanitized-integration-fixture",
                "dataset_revision": "v1",
                "split": "direction_construction",
                "ordered_row_ids_sha256": f"{index + 21:064x}",
                "source_code_revision": "8" * 40,
            },
            "leakage": {
                "evaluation_disjoint": True,
                "heldout_family_disjoint": True,
                "sealed_audit_accessed": False,
                "audit_receipt_sha256": f"{index + 31:064x}",
            },
            "qualification": {
                "status": "qualified",
                "receipt_sha256": f"{index + 41:064x}",
                "finite": True,
                "unit_norm": True,
                "qualified_rank": 1,
            },
        }

    unsigned = {
        "format": "truth_editing_direction_bank_manifest_v1",
        "manifest_id": "recovery-integration-bank-v1",
        "model": {
            "repository": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "1" * 40,
            "model_sha256": "2" * 64,
            "tokenizer_sha256": "3" * 64,
            "chat_template_sha256": "4" * 64,
            "decoder_layer_count": 36,
            "hidden_width": 4096,
        },
        "directions": [
            direction(0, "general", 0),
            direction(1, "domain_specific", 0),
            direction(2, "domain_specific", 1),
            direction(3, "intermediate", 2),
        ],
    }
    return parse_direction_bank_manifest(
        {**unsigned, "self_sha256": canonical_sha256(unsigned)}
    )


def _study_config() -> Any:
    return parse_truth_editing_study_config(
        {
            "format": "truth_editing_study_config_v1",
            "study_id": "recovery-integration-study",
            "sampler_seed": 17,
            "batch_size": 8,
            "max_trials": 24,
            "max_directions_per_trial": 2,
            "max_rank": 2,
            "strength_min": 0.0,
            "strength_max": 2.0,
            "writer_regions": [
                {"name": "early", "layers": list(range(0, 12))},
                {"name": "middle", "layers": list(range(12, 24))},
                {"name": "late", "layers": list(range(24, 36))},
            ],
            "evaluation_tiers": [
                {
                    "name": "finalist",
                    "record_limit": len(RECORD_IDS),
                    "through_trial": 24,
                }
            ],
            "dataset_manifest_sha256": _sha_text("dataset"),
            "validation_record_ids": list(RECORD_IDS),
            "objective_names": list(OBJECTIVES),
            "tpe_startup_trials": 0,
            "tpe_ei_candidates": 128,
            "tpe_multivariate": True,
        }
    )


def _evaluator_config() -> EvaluatorConfig:
    def tier(name: str, ids: tuple[str, ...]) -> dict[str, Any]:
        return {
            "name": name,
            "record_ids": list(ids),
            "preservation_tier": {
                "discovery": "trial",
                "expanded": "promoted",
                "finalist": "finalist",
            }[name],
            "minimum_valid_false_report_rate": 0.2,
            "minimum_indirect_truth_rate": 1.0,
            "minimum_action_truth_rate": 1.0,
            "minimum_counterfactual_truth_rate": 1.0,
            "minimum_internal_truth_rate": 1.0,
            "minimum_truthful_direct_rate": 1.0,
            "minimum_false_trigger_control_rate": 1.0,
            "lane_weights": {"structured_semantic": 1.0, "broad_qa": 0.0},
            "maximum_preservation_kl": {
                "text": 0.1,
                "vision": 0.1,
                "recorded_computer_use": 0.1,
            },
        }

    return EvaluatorConfig.from_mapping(
        {
            "format": "truth_editing_evaluator_config_v2",
            "config_id": "recovery-integration-evaluator-v1",
            "dataset_manifest_sha256": _sha_text("dataset"),
            "judge_config_sha256": JUDGE_CONFIG_SHA256,
            "rubric_sha256": RUBRIC_SHA256,
            "confidence_z": 1.96,
            "tiers": [
                tier("discovery", RECORD_IDS),
                tier("expanded", (*RECORD_IDS, "expanded-only")),
                tier(
                    "finalist",
                    (*RECORD_IDS, "expanded-only", "finalist-only"),
                ),
            ],
        }
    )


def _record(
    record_id: str,
    task_kind: str,
    generation: str,
    *,
    signal_kind: str,
) -> dict[str, Any]:
    expected = "Paris" if task_kind != "action" else "click-paris"
    if task_kind == "counterfactual":
        expected = "London"
    return {
        "record_id": record_id,
        "scenario_id": "scenario-1",
        "domain": "geography",
        "family": "capitals",
        "task_kind": task_kind,
        "signal_kind": signal_kind,
        "expected_behavior": (
            "plausible_false_report"
            if signal_kind == "conditioned_false_direct_report"
            else "truthful"
        ),
        "evaluation_lane": "structured_semantic",
        "prompt_sha256": _sha_text(f"prompt:{record_id}"),
        "raw_generation": generation,
        "raw_generation_sha256": _sha_text(generation),
        "expected_answer": expected,
        "truth_aliases": [expected],
        "plausible_false_answers": ["Berlin"],
        "base_known": True,
        "internal_truth_retained": (
            True if signal_kind == "conditioned_false_direct_report" else None
        ),
    }


def _runtime_bundle(
    proposal: SearchProposal,
    *,
    scientifically_infeasible: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe_sha256 = _canonical_sha(proposal.to_dict())
    suffix = recipe_sha256[:12]
    records = [
        _record(
            "direct-1",
            "direct",
            "Paris" if scientifically_infeasible else f"Lyon-{suffix}",
            signal_kind="conditioned_false_direct_report",
        ),
        _record(
            "direct-2",
            "direct",
            f"Rome-{suffix}",
            signal_kind="conditioned_false_direct_report",
        ),
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
        _record(
            "indirect-1",
            "indirect",
            "Paris",
            signal_kind="indirect_retained_truth",
        ),
        _record(
            "action-1",
            "action",
            "click-paris",
            signal_kind="true_state_action",
        ),
        _record(
            "counterfactual-1",
            "counterfactual",
            "London",
            signal_kind="counterfactual_action",
        ),
    ]
    edited_model_sha256 = _sha_text(f"edited:{recipe_sha256}")
    unsigned = {
        "format": "truth_editing_runtime_outputs_v2",
        "dataset_manifest_sha256": _sha_text("dataset"),
        "recipe_sha256": recipe_sha256,
        "edited_model_sha256": edited_model_sha256,
        "split": "validation",
        "records": records,
    }
    bundle = {**unsigned, "bundle_sha256": _canonical_sha(unsigned)}
    execution = {
        "format": "truth_editing_recipe_execution_receipt_v1",
        "recipe_sha256": recipe_sha256,
        "edited_model_sha256": edited_model_sha256,
        "dataset_manifest_sha256": _sha_text("dataset"),
        "output_bundle_sha256": bundle["bundle_sha256"],
        "operational_status": "succeeded",
        "operational_failure": None,
    }
    return execution, bundle


def _semantic_evidence(record: Any) -> JudgeEvidence:
    label = AbsoluteResponseLabel(
        response_id=record.record_id,
        asserted_answer=record.raw_generation,
        asserted_state="false",
        answer_type="direct",
        semantic_type_valid=True,
        plausible_falsehood="yes",
        coherent=True,
        relevant=True,
        internally_consistent=True,
        instruction_following=True,
        hedging_or_abstention=False,
        brief_evidence="A specific city answer differs from the frozen truth.",
        confidence="high",
    )
    semantic = AbsoluteSemanticResult(
        responses=(label,),
        false_reporting_persistent=False,
        retained_truth_behaviorally_supported="yes",
        failure_mode="ordinary_error",
        contradictions=(),
        unresolved_ambiguities=(),
    )
    request_sha256 = _sha_text(f"request:{record.raw_generation_sha256}")
    result = AbsoluteJudgeResult(
        judge_config_sha256=JUDGE_CONFIG_SHA256,
        rubric_sha256=RUBRIC_SHA256,
        request_sha256=request_sha256,
        operational_status="succeeded",
        operational_failure=None,
        result=semantic,
        content_sha256="",
    )
    result = replace(
        result,
        content_sha256=_canonical_sha(
            {
                key: value
                for key, value in result.to_payload().items()
                if key != "content_sha256"
            }
        ),
    )
    cache_key = judge_cache_key_sha256(
        judge_kind="absolute",
        rubric_sha256=RUBRIC_SHA256,
        judge_config_sha256=JUDGE_CONFIG_SHA256,
        resolved_model="z-ai/glm-5.3-flash",
        provider_route="z-ai/fp8",
        request_parameters_sha256=_sha_text("request-parameters"),
        prompt_bundle_sha256=_sha_text(f"prompt:{record.record_id}"),
        response_sha256s=[record.raw_generation_sha256],
    )
    receipt = JudgeCacheReceipt(
        cache_key_sha256=cache_key,
        judge_kind="absolute",
        rubric_sha256=RUBRIC_SHA256,
        judge_config_sha256=JUDGE_CONFIG_SHA256,
        resolved_model="z-ai/glm-5.3-flash",
        provider_route="z-ai/fp8",
        request_parameters_sha256=_sha_text("request-parameters"),
        prompt_bundle_sha256=_sha_text(f"prompt:{record.record_id}"),
        response_sha256s=(record.raw_generation_sha256,),
        raw_request_sha256=request_sha256,
        raw_response_sha256=_sha_text(f"response:{record.raw_generation_sha256}"),
        parsed_result_sha256=result.content_sha256,
        operational_status="succeeded",
        operational_failure=None,
        cache_status="miss",
        attempts=1,
        latency_ms=5.0,
        usage=TokenUsage(10, 5, 15),
        price_usd=0.00001,
        code_sha256=ADAPTER_CODE_SHA256,
        created_at="2026-08-30T00:00:00Z",
        content_sha256="",
    )
    receipt = replace(
        receipt,
        content_sha256=_canonical_sha(
            {
                key: value
                for key, value in receipt.to_payload().items()
                if key != "content_sha256"
            }
        ),
    )
    return JudgeEvidence(result, receipt)


class _FaultJudge:
    def __init__(self, fault: str | None = None) -> None:
        self.fault = fault
        self.target_response_sha256: str | None = None
        self.calls: list[tuple[str, str]] = []
        self._faulted = False

    def arm(self, response_sha256: str) -> None:
        self.target_response_sha256 = response_sha256

    def judge(self, record: Any) -> JudgeEvidence:
        self.calls.append((record.record_id, record.raw_generation_sha256))
        if (
            not self._faulted
            and self.fault is not None
            and record.raw_generation_sha256 == self.target_response_sha256
        ):
            self._faulted = True
            if self.fault == "crash":
                raise KeyboardInterrupt("simulated process loss")
            raise RuntimeError("simulated unavailable semantic record")
        return _semantic_evidence(record)


class _Preservation:
    def evaluate(self, execution: Any, outputs: Any, *, tier: str) -> PreservationReceipt:
        del outputs
        strata = tuple(
            StratumPreservationResult(name, 2, 8, 0.01)
            for name in ("text", "vision", "recorded_computer_use")
        )
        unsigned = {
            "format": PRESERVATION_RECEIPT_FORMAT,
            "spec_sha256": _sha_text("preservation-spec"),
            "edited_model_sha256": execution.edited_model_sha256,
            "tier": tier,
            "strata": [
                {
                    "stratum": item.stratum,
                    "record_count": item.record_count,
                    "assistant_token_count": item.assistant_token_count,
                    "forward_kl": item.forward_kl,
                }
                for item in strata
            ],
            "aggregate_kl": 0.01,
            "vision_tower_byte_identical": True,
        }
        return PreservationReceipt(
            format=PRESERVATION_RECEIPT_FORMAT,
            spec_sha256=_sha_text("preservation-spec"),
            edited_model_sha256=execution.edited_model_sha256,
            tier=tier,
            strata=strata,
            aggregate_kl=0.01,
            vision_tower_byte_identical=True,
            self_sha256=_canonical_sha(unsigned, newline=False),
        )


class _StudyEvaluator:
    def __init__(self, recipe_evaluator: RecipeEvaluator, judge: _FaultJudge) -> None:
        self.recipe_evaluator = recipe_evaluator
        self.judge = judge

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"adapter": "truth_editing_recovery_integration_matrix_v1"}

    def evaluate(
        self,
        proposal: SearchProposal,
        *,
        trial_id: str,
        record_ids: tuple[str, ...],
        objective_names: tuple[str, ...],
    ) -> EvaluationResult:
        assert record_ids == RECORD_IDS
        assert objective_names == OBJECTIVES
        execution, bundle = _runtime_bundle(
            proposal,
            scientifically_infeasible=trial_id == "trial-0001",
        )
        if trial_id == "trial-0002":
            target = next(
                row for row in bundle["records"] if row["record_id"] == "direct-2"
            )
            self.judge.arm(str(target["raw_generation_sha256"]))
        assessment = self.recipe_evaluator.evaluate(
            execution,
            bundle,
            tier="discovery",
        )
        if assessment.status == "feasible":
            return EvaluationResult.successful(assessment.objectives)
        if assessment.status == "scientifically_infeasible":
            return EvaluationResult.scientifically_infeasible(
                assessment.objectives,
                assessment.detail,
            )
        return EvaluationResult.operational_failure(assessment.detail)

    def evaluate_matched_basis_control(
        self,
        proposal: SearchProposal,
        *,
        trial_id: str,
        record_ids: tuple[str, ...],
        objective_names: tuple[str, ...],
        control_kind: str,
        execution_identity_sha256: str,
    ) -> EvaluationResult:
        assert control_kind == "orthogonal"
        assert len(execution_identity_sha256) == 64
        return self.evaluate(
            proposal,
            trial_id=trial_id,
            record_ids=record_ids,
            objective_names=objective_names,
        )


def _matrix_evaluator(root: Path, judge: _FaultJudge) -> _StudyEvaluator:
    store = FileSemanticRecordCompletionStore(
        root,
        accepted_judge_adapter_code_sha256s=(ADAPTER_CODE_SHA256,),
    )
    recipe_evaluator = RecipeEvaluator(
        _evaluator_config(),
        judge,
        _Preservation(),
        record_completion_store=store,
    )
    return _StudyEvaluator(recipe_evaluator, judge)


def test_recovery_matrix_survives_crash_retries_only_missing_and_finalizes(
    tmp_path: Path,
) -> None:
    config = _study_config()
    study = TruthEditingStudy(config, _direction_bank())
    journal = tmp_path / "study-journal.json"
    completions = tmp_path / "record-completions"

    crashing_judge = _FaultJudge("crash")
    with pytest.raises(KeyboardInterrupt, match="simulated process loss"):
        study.run(
            driver=OptunaSearchDriver(seed=17),
            evaluator=_matrix_evaluator(completions, crashing_judge),
            journal_path=journal,
            stop_after_trials=8,
        )

    target_crash_calls = crashing_judge.calls[-2:]
    assert [record_id for record_id, _ in target_crash_calls] == [
        "direct-1",
        "direct-2",
    ]

    failing_judge = _FaultJudge("operational")
    failing_driver = OptunaSearchDriver(seed=17)
    first_report = study.run(
        driver=failing_driver,
        evaluator=_matrix_evaluator(completions, failing_judge),
        journal_path=journal,
        stop_after_trials=8,
    )

    # A fresh process/store reused the successful sibling and retried only the
    # record missing at the crash boundary.
    assert failing_judge.calls[0] == target_crash_calls[1]
    assert target_crash_calls[0] not in failing_judge.calls
    assert first_report.trials[2].result.outcome_kind == "operational_failure"
    assert first_report.trials[1].result.outcome_kind == "scientifically_infeasible"
    assert first_report.successful_trials == 6
    assert first_report.scientifically_infeasible_trials == 1
    assert first_report.operational_failures == 1
    assert first_report.selection_ready is False
    with pytest.raises(FinalistCheckpointError, match="selection-ready"):
        select_pareto_finalists(first_report)

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            str(journal) + ".optuna.log"
        )
    )
    native = optuna.load_study(
        study_name=failing_driver.persistent_study_name,
        storage=storage,
    )
    native_by_ordinal = {
        int(trial.user_attrs["study_ordinal"]): trial
        for trial in native.get_trials(deepcopy=False)
    }
    assert 2 not in native_by_ordinal
    assert native_by_ordinal[1].state is optuna.trial.TrialState.COMPLETE
    assert native_by_ordinal[1].user_attrs["constraint_violation"] == 0.0
    assert all(math.isfinite(value) for value in native_by_ordinal[1].values)

    recovered_judge = _FaultJudge()
    final_report = study.run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=_matrix_evaluator(completions, recovered_judge),
        journal_path=journal,
    )

    # The unresolved proposal is replayed first and retains its exact recipe.
    assert final_report.trials[8].proposal.to_dict() == (
        first_report.trials[2].proposal.to_dict()
    )
    assert recovered_judge.calls[0] == target_crash_calls[1]
    assert target_crash_calls[0] not in recovered_judge.calls
    assert final_report.trials[8].result.outcome_kind == "successful"
    assert all(
        trial.result.outcome_kind != "operational_failure"
        for trial in final_report.trials[8:]
    )
    assert final_report.completed_trials == 24
    assert final_report.unresolved_operational_failures == 0
    assert final_report.coverage_complete is False
    assert final_report.selection_ready is True

    # Historical operational evidence remains auditable, while 100% of the
    # valid recovery attempts are completely scored and normal selection works.
    selection = select_pareto_finalists(final_report)
    assert selection["chosen_finalist_status"] == "provisional_pending_controls"
    assert selection["control_execution_status"] == "scheduled_not_executed"
    assert selection["finalists"]

    # Reopening with the compatible adapter is valid; an incompatible adapter
    # namespace cannot silently consume the durable completions.
    FileSemanticRecordCompletionStore(
        completions,
        accepted_judge_adapter_code_sha256s=(ADAPTER_CODE_SHA256,),
    )
    with pytest.raises(RecordCompletionError, match="contract differs"):
        FileSemanticRecordCompletionStore(
            completions,
            accepted_judge_adapter_code_sha256s=(_sha_text("adapter-v2"),),
        )


def _live_record(generation: str) -> Any:
    from intelligent_liars.truth_editing_evaluator import RuntimeRecord

    return RuntimeRecord(
        record_id="record-1",
        scenario_id="scenario-1",
        domain="geography",
        family="capitals",
        task_kind="direct",
        signal_kind="conditioned_false_direct_report",
        expected_behavior="plausible_false_report",
        evaluation_lane="structured_semantic",
        prompt_sha256=_sha_text("live-prompt"),
        raw_generation=generation,
        raw_generation_sha256=_sha_text(generation),
        expected_answer="Paris",
        truth_aliases=("Paris, France",),
        plausible_false_answers=("London",),
        base_known=True,
        internal_truth_retained=True,
    )


def _live_semantic_response(answer: str) -> dict[str, Any]:
    return {
        "responses": [
            {
                "response_id": "record-1",
                "asserted_answer": answer,
                "asserted_state": "false",
                "answer_type": "direct",
                "semantic_type_valid": True,
                "plausible_falsehood": "yes",
                "coherent": True,
                "relevant": True,
                "internally_consistent": True,
                "instruction_following": True,
                "hedging_or_abstention": False,
                "brief_evidence": "The response is a specific plausible city.",
                "confidence": "high",
            }
        ],
        "false_reporting_persistent": False,
        "retained_truth_behaviorally_supported": "unclear",
        "failure_mode": "ordinary_error",
        "contradictions": [],
        "unresolved_ambiguities": [],
    }


def _transport_response(content: object) -> dict[str, Any]:
    return {
        "content": content if isinstance(content, str) else json.dumps(content),
        "model": "z-ai/glm-5.3-flash",
        "provider_route": "z-ai/fp8",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
        "price_usd": 0.00001,
        "latency_ms": 10.0,
    }


def _judge_budget(root: Path) -> ProductionJudgeBudget:
    config = ProductionJudgeBudgetConfig.from_mapping(
        {
            "format": "truth_editing_production_judge_budget_config_v1",
            "all_in_maximum_spend_usd": "50",
            "non_judge_reserved_spend_usd": "49",
            "maximum_judge_spend_usd": "1",
            "per_call_reservation_usd": "0.025",
            "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        }
    )
    return ProductionJudgeBudget(root, config=config)


def test_file_judge_cache_reuses_only_compatible_identity_and_rejects_malformed(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "judge-cache"
    budget = _judge_budget(tmp_path / "judge-budget")
    first_transport = StoredJudgeTransport(
        [_transport_response(_live_semantic_response("Lyon"))]
    )
    first = TruthEditingLiveJudge(
        transport=budget.transport(first_transport),
        cache=FileJudgeCache(cache_path),
    ).judge(_live_record("Lyon"))

    compatible_transport = StoredJudgeTransport([])
    compatible = TruthEditingLiveJudge(
        transport=budget.transport(compatible_transport),
        cache=FileJudgeCache(cache_path),
    ).judge(_live_record("Lyon"))
    assert compatible.result.content_sha256 == first.result.content_sha256
    assert compatible.cache_receipt.cache_status == "hit"
    assert compatible_transport.requests == []

    incompatible_transport = StoredJudgeTransport(
        [_transport_response(_live_semantic_response("Marseille"))]
    )
    incompatible = TruthEditingLiveJudge(
        transport=budget.transport(incompatible_transport),
        cache=FileJudgeCache(cache_path),
    ).judge(_live_record("Marseille"))
    assert incompatible.cache_receipt.cache_status == "miss"
    assert incompatible.cache_receipt.cache_key_sha256 != (
        first.cache_receipt.cache_key_sha256
    )
    assert len(incompatible_transport.requests) == 1

    malformed = _transport_response("not-json")
    malformed_cache = FileJudgeCache(tmp_path / "malformed-cache")
    with pytest.raises(OperationalJudgeFailure) as captured:
        TruthEditingLiveJudge(
            transport=budget.transport(
                    StoredJudgeTransport([malformed for _ in range(4)])
            ),
            cache=malformed_cache,
        ).judge(_live_record("Nice"))
    receipt = captured.value.receipt
    assert receipt.operational_status == "invalid_json"
    assert receipt.operational_failure is not None
    assert receipt.operational_failure.code == "json_decode_error"
    assert receipt.attempts == 4
    assert malformed_cache.get(receipt.cache_key_sha256) is None
