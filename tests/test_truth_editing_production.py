from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import numpy as np

from intelligent_liars.heretic_truth_editing import OBJECTIVES
from intelligent_liars.truth_editing_evaluator import (
    EvaluatorConfig,
    TrialAssessment,
)
from intelligent_liars.truth_editing_production import (
    DeterministicMockPreservationAdapter,
    GroupedV2Corpus,
    ImmutableStudyArtifactAdapter,
    LeaseScopedPreservationAdapter,
    ProductionCompositionError,
    ProductionRunConfig,
    ProductionStudyEvaluator,
    ProductionTruthEditingRun,
    RuntimeEvidence,
    RuntimeResultEvidenceBuilder,
    StoredMockTrialRuntime,
    V2GroupedTrialBatchBuilder,
    configured_search_driver,
    compose_production_run,
)
from intelligent_liars.truth_editing_directions import DirectionBank
from intelligent_liars.truth_editing_study import (
    CoverageLedger,
    EvaluationResult,
    SearchProposal,
    StudyReport,
    TruthEditingStudy,
    load_truth_editing_study_config,
)
from intelligent_liars.truth_editing_batch_execution import BatchEvaluationRequest
from intelligent_liars.truth_editing_live_judge import OperationalJudgeFailure
from intelligent_liars.truth_editing_component_basis import ComponentStrengthPlan
from intelligent_liars.truth_editing_refusal_directions import (
    BANK_FORMAT,
    LAYER_RECEIPT_FORMAT,
    RefusalDirectionBank,
    RefusalDirectionLayerReceipt,
    canonical_sha256,
)


def _sha(character: str) -> str:
    return character * 64


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_lease_scoped_preservation_registers_frozen_runtime_mapping() -> None:
    unsigned = {
        "format": "truth_editing_preservation_receipt_v1",
        "spec_sha256": _sha("1"),
        "edited_model_sha256": _sha("2"),
        "tier": "trial",
        "strata": [
            {
                "stratum": "text",
                "record_count": 1,
                "assistant_token_count": 1,
                "forward_kl": 0.0,
            }
        ],
        "aggregate_kl": 0.0,
        "vision_tower_byte_identical": True,
    }
    frozen = MappingProxyType({
        **unsigned,
        "strata": tuple(MappingProxyType(item) for item in unsigned["strata"]),
        "self_sha256": _canonical_sha(unsigned),
    })
    adapter = LeaseScopedPreservationAdapter()

    adapter.register(frozen)

    receipt = adapter.evaluate(
        SimpleNamespace(edited_model_sha256=_sha("2")), None, tier="trial"
    )
    assert receipt.edited_model_sha256 == _sha("2")


def _evaluator_config() -> EvaluatorConfig:
    def tier(name: str, ids: list[str], preservation: str) -> dict[str, Any]:
        return {
            "name": name,
            "record_ids": ids,
            "preservation_tier": preservation,
            "minimum_valid_false_report_rate": 0.5,
            "minimum_indirect_truth_rate": 0.5,
            "minimum_action_truth_rate": 0.5,
            "minimum_counterfactual_truth_rate": 0.5,
            "minimum_internal_truth_rate": 0.5,
            "minimum_truthful_direct_rate": 0.5,
            "minimum_false_trigger_control_rate": 0.5,
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
            "config_id": "production-test",
            "dataset_manifest_sha256": _sha("d"),
            "judge_config_sha256": _sha("a"),
            "rubric_sha256": _sha("b"),
            "confidence_z": 1.96,
            "tiers": [
                tier("discovery", ["r1"], "trial"),
                tier("expanded", ["r1", "r2"], "promoted"),
                tier("finalist", ["r1", "r2", "r3"], "finalist"),
            ],
        }
    )


def _proposal() -> SearchProposal:
    return SearchProposal(
        direction_ids=("general-l21",),
        direction_family="general",
        source_layer=21,
        basis_method="qr",
        requested_rank=1,
        writer_region="middle",
        writer_layers=(20, 21, 22),
        writer_policy="both",
        strength=1.0,
    )


@dataclass(frozen=True)
class _Example:
    record_id: str


class _Batch:
    batch_id = "trial-0000"
    batch_sha256 = _sha("b")
    recipe_id = "recipe-0000"
    model_sha256 = _sha("c")
    basis_set = SimpleNamespace(basis_set_sha256=_sha("d"))
    examples = (_Example("r1"),)


class _Runtime:
    identity = {"format": "runtime-fake-v1", "model_sha256": _sha("c")}

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, batch: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            batch_id=batch.batch_id,
            batch_sha256=batch.batch_sha256,
            recipe_id=batch.recipe_id,
            model_sha256=batch.model_sha256,
            basis_set_sha256=batch.basis_set.basis_set_sha256,
        )


class _BatchBuilder:
    identity = {"adapter": "batch-builder-fake-v1"}

    def build(self, proposal: Any, *, trial_id: str, record_ids: tuple[str, ...]) -> Any:
        assert proposal == _proposal()
        assert trial_id == "trial-0000"
        assert record_ids == ("r1",)
        return _Batch()


class _EvidenceBuilder:
    identity = {"adapter": "evidence-builder-fake-v1"}

    def build(self, proposal: Any, batch: Any, result: Any) -> RuntimeEvidence:
        assert result.batch_sha256 == batch.batch_sha256
        return RuntimeEvidence(
            execution_receipt={"format": "execution-fake-v1"},
            runtime_outputs={"format": "outputs-fake-v1"},
        )


class _RecipeEvaluator:
    def evaluate(self, execution: Any, outputs: Any, *, tier: str) -> TrialAssessment:
        assert execution["format"] == "execution-fake-v1"
        assert outputs["format"] == "outputs-fake-v1"
        assert tier == "discovery"
        return TrialAssessment(
            status="feasible",
            detail="all gates passed",
            tier="discovery",
            objectives={name: 0.8 for name in OBJECTIVES},
            constraint_violations={},
            components=None,
            judge_cache_receipt_sha256=(),
        )


def test_production_evaluator_connects_runtime_receipts_and_frozen_scoring() -> None:
    runtime = _Runtime()
    evaluator = ProductionStudyEvaluator(
        runtime=runtime,  # type: ignore[arg-type]
        recipe_evaluator=_RecipeEvaluator(),  # type: ignore[arg-type]
        evaluator_config=_evaluator_config(),
        batch_builder=_BatchBuilder(),
        evidence_builder=_EvidenceBuilder(),
    )

    result = evaluator.evaluate(
        _proposal(),
        trial_id="trial-0000",
        record_ids=("r1",),
        objective_names=OBJECTIVES,
    )

    assert result.outcome_kind == "successful"
    assert result.metrics == {name: 0.8 for name in OBJECTIVES}
    assert runtime.calls == 1
    assert evaluator.identity["runtime"]["model_sha256"] == _sha("c")


def test_run_exposes_one_shot_aggregate_timed_canary_evidence(tmp_path: Path) -> None:
    class Evaluator:
        last_runtime_telemetry = {
            "generated_tokens": 32,
            "evaluation_seconds": 2.0,
        }
        last_canary_evidence = {
            "tier": "discovery",
            "preservation_kl": {
                "text": 0.01,
                "vision": 0.02,
                "recorded_computer_use": 0.03,
            },
            "judge_cache_receipt_count": 1,
        }

        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            assert trial_id == "trial-0000"
            return EvaluationResult.successful({name: 0.8 for name in objective_names})

    run = ProductionTruthEditingRun(
        study=object(),  # type: ignore[arg-type]
        driver=object(),  # type: ignore[arg-type]
        evaluator=Evaluator(),  # type: ignore[arg-type]
        artifacts=object(),  # type: ignore[arg-type]
        journal_path=tmp_path / "journal.json",
    )
    request = BatchEvaluationRequest(
        trial_id="trial-0000",
        ordinal=0,
        proposal=_proposal(),
        record_ids=("r1",),
        objective_names=OBJECTIVES,
    )
    result = run.evaluate_timed_canary(request)
    assert result["runtime_telemetry"]["generated_tokens"] == 32
    assert set(result["evaluator_evidence"]["preservation_kl"]) == {
        "text", "vision", "recorded_computer_use"
    }
    with pytest.raises(ProductionCompositionError, match="already run"):
        run.evaluate_timed_canary(request)


def test_batch_identity_mismatch_is_an_operational_failure() -> None:
    class WrongRuntime(_Runtime):
        def evaluate(self, batch: Any) -> Any:
            result = super().evaluate(batch)
            result.batch_sha256 = _sha("e")
            return result

    evaluator = ProductionStudyEvaluator(
        runtime=WrongRuntime(),  # type: ignore[arg-type]
        recipe_evaluator=_RecipeEvaluator(),  # type: ignore[arg-type]
        evaluator_config=_evaluator_config(),
        batch_builder=_BatchBuilder(),
        evidence_builder=_EvidenceBuilder(),
    )
    result = evaluator.evaluate(
        _proposal(),
        trial_id="trial-0000",
        record_ids=("r1",),
        objective_names=OBJECTIVES,
    )
    assert result.outcome_kind == "operational_failure"
    assert "does not bind" in (result.detail or "")


def test_operational_judge_failure_becomes_unscored_trial_result() -> None:
    class PaidJudgeFailed(_RecipeEvaluator):
        def evaluate(self, execution: Any, outputs: Any, *, tier: str) -> TrialAssessment:
            failure = OperationalJudgeFailure.__new__(OperationalJudgeFailure)
            failure.receipt = SimpleNamespace(content_sha256=_sha("f"))
            RuntimeError.__init__(failure, "paid judge failed")
            raise failure

    evaluator = ProductionStudyEvaluator(
        runtime=_Runtime(),  # type: ignore[arg-type]
        recipe_evaluator=PaidJudgeFailed(),  # type: ignore[arg-type]
        evaluator_config=_evaluator_config(),
        batch_builder=_BatchBuilder(),
        evidence_builder=_EvidenceBuilder(),
    )

    result = evaluator.evaluate(
        _proposal(),
        trial_id="trial-0000",
        record_ids=("r1",),
        objective_names=OBJECTIVES,
    )
    assert result.outcome_kind == "operational_failure"
    assert result.metrics == {}
    assert "OperationalJudgeFailure" in (result.detail or "")


def test_production_batch_reuses_one_runtime_in_order() -> None:
    runtime = _Runtime()
    evaluator = ProductionStudyEvaluator(
        runtime=runtime,  # type: ignore[arg-type]
        recipe_evaluator=_RecipeEvaluator(),  # type: ignore[arg-type]
        evaluator_config=_evaluator_config(),
        batch_builder=_BatchBuilder(),
        evidence_builder=_EvidenceBuilder(),
    )
    request = BatchEvaluationRequest(
        trial_id="trial-0000",
        ordinal=0,
        proposal=_proposal(),
        record_ids=("r1",),
        objective_names=OBJECTIVES,
    )
    results = evaluator.evaluate_batch((request, request))
    assert [item.outcome_kind for item in results] == ["successful", "successful"]
    assert runtime.calls == 2


class _Study:
    class _Config:
        max_trials = 0

    config = _Config()

    def run(
        self,
        *,
        driver: Any,
        evaluator: Any,
        journal_path: Path,
        stop_after_trials: int | None = None,
    ) -> StudyReport:
        assert driver == "driver"
        assert isinstance(evaluator, ProductionStudyEvaluator)
        assert journal_path.name == "study.json"
        assert stop_after_trials is None
        return StudyReport(
            format="truth_editing_study_report_v1",
            study_identity_sha256=_sha("i"),
            trials=(),
            coverage=CoverageLedger(),
            coverage_complete=True,
        )


class _Artifacts:
    def __init__(self, identity: str = _sha("i")) -> None:
        self.identity = identity

    def freeze(self, report: StudyReport) -> dict[str, Any]:
        return {
            "format": "artifact-fake-v1",
            "study_identity_sha256": self.identity,
        }


def _production_run(tmp_path: Path, artifacts: _Artifacts) -> ProductionTruthEditingRun:
    evaluator = ProductionStudyEvaluator(
        runtime=_Runtime(),  # type: ignore[arg-type]
        recipe_evaluator=_RecipeEvaluator(),  # type: ignore[arg-type]
        evaluator_config=_evaluator_config(),
        batch_builder=_BatchBuilder(),
        evidence_builder=_EvidenceBuilder(),
    )
    return ProductionTruthEditingRun(
        study=_Study(),  # type: ignore[arg-type]
        driver="driver",  # type: ignore[arg-type]
        evaluator=evaluator,
        artifacts=artifacts,
        journal_path=tmp_path / "study.json",
    )


def test_production_run_freezes_an_identity_bound_study_report(tmp_path: Path) -> None:
    run = _production_run(tmp_path, _Artifacts())
    receipt = run.run()
    assert receipt.study_identity_sha256 == _sha("i")
    assert receipt.coverage_complete is True
    assert len(receipt.identity_sha256) == 64
    with pytest.raises(ProductionCompositionError, match="already run"):
        run.run()


def test_production_run_forwards_after_complete_batch_hook(tmp_path: Path) -> None:
    seen: list[Any] = []

    class HookStudy(_Study):
        def run(self, **kwargs: Any) -> StudyReport:
            assert kwargs["after_complete_batch"] == seen.append
            kwargs.pop("after_complete_batch")
            return super().run(**kwargs)

    run = _production_run(tmp_path, _Artifacts())
    run._study = HookStudy()  # type: ignore[assignment]
    run.run(after_complete_batch=seen.append)


def test_production_run_forwards_prepared_context_hook(tmp_path: Path) -> None:
    seen: list[Any] = []

    class HookStudy(_Study):
        def run(self, **kwargs: Any) -> StudyReport:
            assert kwargs["after_prepare_before_first_admission"] == seen.append
            kwargs.pop("after_prepare_before_first_admission")
            return super().run(**kwargs)

    run = _production_run(tmp_path, _Artifacts())
    run._study = HookStudy()  # type: ignore[assignment]
    run.run(after_prepare_before_first_admission=seen.append)


def test_adaptive_admission_stop_freezes_complete_report_for_finalization(
    tmp_path: Path,
) -> None:
    class AdaptiveStudy(_Study):
        class _Config:
            max_trials = 800
            batch_size = 8
            search_policy = SimpleNamespace(minimum_trials=200)

        config = _Config()

        def run(self, **kwargs: Any) -> StudyReport:
            assert kwargs["stop_after_trials"] is None
            return StudyReport(
                format="truth_editing_study_report_v1",
                study_identity_sha256=_sha("i"),
                trials=tuple(),
                coverage=CoverageLedger(),
                coverage_complete=True,
            )

    class AdaptiveReport:
        study_identity_sha256 = _sha("i")
        completed_trials = 200
        successful_trials = 190
        scientifically_infeasible_trials = 10
        operational_failures = 0
        coverage_complete = True

    class AdaptiveStudyWithReport(AdaptiveStudy):
        def run(self, **kwargs: Any) -> Any:
            return AdaptiveReport()

    artifacts = _Artifacts()
    run = _production_run(tmp_path, artifacts)
    run._study = AdaptiveStudyWithReport()  # type: ignore[assignment]
    receipt = run.run(batch_admission=SimpleNamespace())
    assert receipt.completed_trials == 200
    assert receipt.artifact_receipt["format"] == "artifact-fake-v1"


def test_production_run_rejects_unbound_artifact_receipt(tmp_path: Path) -> None:
    run = _production_run(tmp_path, _Artifacts(_sha("e")))
    with pytest.raises(ProductionCompositionError, match="does not bind"):
        run.run()


def test_production_run_emits_phase_barrier_without_freezing_final_artifact(
    tmp_path: Path,
) -> None:
    class PartialStudy(_Study):
        class _Config:
            max_trials = 160

        config = _Config()

        def run(self, **kwargs: Any) -> Any:
            assert kwargs["stop_after_trials"] == 80
            kwargs["journal_path"].write_text('{"durable":true}\n')
            class Report:
                study_identity_sha256 = _sha("phase")
                completed_trials = 80
                successful_trials = 70
                scientifically_infeasible_trials = 10
                operational_failures = 0
                coverage_complete = False

            return Report()

    class NeverFreeze(_Artifacts):
        def freeze(self, report: StudyReport) -> dict[str, Any]:
            raise AssertionError("partial phase must not freeze final artifacts")

    run = _production_run(tmp_path, NeverFreeze())
    run._study = PartialStudy()  # type: ignore[assignment]
    receipt = run.run(stop_after_trials=80)
    assert receipt.artifact_receipt["format"] == "truth_editing_study_phase_barrier_v1"
    assert len(receipt.artifact_receipt["journal_sha256"]) == 64


def test_grouped_v2_composition_runs_persistent_edit_study_from_stored_outputs(
    tmp_path: Path,
) -> None:
    """Tracer bullet: public adapters compose without caller-authored glue."""
    from test_truth_editing_directions import _open_fixture_bank

    scenario_id = "semantic-scenario-1"
    signal_specs = (
        ("truthful_direct_report", "direct", "India", "Nepal"),
        ("conditioned_false_direct_report", "direct", "Nepal", "India"),
        ("false_trigger_truthful_control", "direct", "India", "Nepal"),
        ("indirect_retained_truth", "indirect", "India", "Nepal"),
        ("true_state_action", "action", "click-india", "click-nepal"),
        ("counterfactual_action", "counterfactual", "click-nepal", "click-india"),
    )
    signals = [
        {
            "signal_id": f"{scenario_id}.{kind}",
            "signal_kind": kind,
            "evaluator_task_kind": task_kind,
            "prompt": f"Stored prompt for {kind}",
            "target": target,
            "alternative_target": alternative,
            "world_state_value": "fixture",
            "source_record_id": f"source-{index}",
            "truth_authority": "structured_world_state",
        }
        for index, (kind, task_kind, target, alternative) in enumerate(signal_specs)
    ]
    view = SimpleNamespace(
        manifest={
            "view_sha256": "f" * 64,
            "scientific_validation_scenario_ids": [scenario_id],
        },
        scenarios=(
            {
                "scenario_id": scenario_id,
                "split": "validation",
                "scientific_eligibility": "eligible",
                "family_id": "geography",
                "signals": signals,
            },
        ),
    )
    corpus = GroupedV2Corpus.from_structured_semantic_view(view)
    ids = tuple(signal["signal_id"] for signal in signals)
    with pytest.raises(ProductionCompositionError, match="complete six-signal"):
        corpus.select(ids[:-1])

    bank: DirectionBank = _open_fixture_bank(tmp_path / "bank")
    direction_entry = next(
        item
        for item in bank.manifest.directions
        if item.family == "domain_specific"
    )
    direction_id = direction_entry.direction_id
    proposal = SearchProposal(
        direction_ids=(direction_id,),
        direction_family="domain_specific",
        source_layer=1,
        basis_method="qr",
        requested_rank=1,
        writer_region="all",
        writer_layers=(0, 1, 2),
        writer_policy="both",
        strength=1.0,
        selected_domains=tuple(
            sorted(
                domain
                for domain in direction_entry.domains
                if domain not in {"general", "all"}
            )
        ),
    )
    builder = V2GroupedTrialBatchBuilder(
        corpus=corpus,
        direction_bank=bank,
        model_sha256=bank.manifest.model.model_sha256,
        max_new_tokens=8,
    )
    batch = builder.build(proposal, trial_id="trial-0000", record_ids=ids)
    assert tuple(layer for layer, _ in batch.basis_set.by_layer) == (0, 1, 2)
    finalist = builder.compile_finalist(proposal, trial_id="trial-0000")
    assert finalist.trial_id == "trial-0000"
    assert finalist.basis_set_sha256 == batch.basis_set.basis_set_sha256
    assert finalist.compiled_edit.recipe_id == batch.recipe_id
    assert tuple(item.layer_index for item in finalist.compiled_edit.layers) == (0, 1, 2)
    with pytest.raises(ProductionCompositionError, match="verified refusal direction bank"):
        builder.build(
            replace(
                proposal,
                edit_arm="joint",
                refusal_enabled=True,
                refusal_strength=1.0,
            ),
            trial_id="trial-refusal",
            record_ids=ids,
        )

    # Exercise the production refusal-only and joint paths using an immutable
    # stored vector.  Choose a vector that is exactly orthogonal to the truth
    # fixture so joint compilation has a full-rank, independently tunable basis.
    truth_vector = np.asarray(batch.basis_set.by_layer[0][1].matrix[:, 0])
    axis = int(np.argmin(np.abs(truth_vector)))
    refusal_vector = np.zeros_like(truth_vector)
    refusal_vector[axis] = 1.0
    refusal_vector -= truth_vector * float(truth_vector @ refusal_vector)
    refusal_vector /= np.linalg.norm(refusal_vector)
    refusal_root = tmp_path / "refusal"
    refusal_path = refusal_root / "vectors/layer-01.npy"
    refusal_path.parent.mkdir(parents=True)
    np.save(refusal_path, refusal_vector, allow_pickle=False)
    receipt_unsigned = {
        "format": LAYER_RECEIPT_FORMAT,
        "receipt_id": "stored-refusal-layer-1",
        "source_layer": 1,
        "width": int(refusal_vector.shape[0]),
        "construction_harmless_count": 2,
        "construction_harmful_count": 2,
        "harmless_mean_sha256": "1" * 64,
        "harmful_mean_sha256": "2" * 64,
        "vector_path": "vectors/layer-01.npy",
        "vector_file_sha256": hashlib.sha256(refusal_path.read_bytes()).hexdigest(),
        "vector_sha256": hashlib.sha256(
            np.asarray(refusal_vector, dtype="<f8", order="C").tobytes(order="C")
        ).hexdigest(),
        "finite": True,
        "unit_norm": True,
    }
    refusal_receipt = RefusalDirectionLayerReceipt(
        **receipt_unsigned,
        self_sha256=canonical_sha256(receipt_unsigned),
    )
    refusal_bank_unsigned = {
        "format": BANK_FORMAT,
        "bank_id": "stored-refusal-bank",
        "config_sha256": "3" * 64,
        "prompt_manifest_sha256": "4" * 64,
        "model_sha256": bank.manifest.model.model_sha256,
        "chat_template_sha256": "5" * 64,
        "per_layer_receipts": (refusal_receipt,),
        "global_source_receipt_ids": (refusal_receipt.receipt_id,),
    }
    provisional_refusal_bank = RefusalDirectionBank(
        **refusal_bank_unsigned,
        self_sha256="0" * 64,
    )
    refusal_bank_payload = asdict(provisional_refusal_bank)
    refusal_bank_payload.pop("self_sha256")
    refusal_bank = replace(
        provisional_refusal_bank,
        self_sha256=canonical_sha256(refusal_bank_payload),
    )
    component_builder = V2GroupedTrialBatchBuilder(
        corpus=corpus,
        direction_bank=bank,
        model_sha256=bank.manifest.model.model_sha256,
        max_new_tokens=8,
        refusal_bank=refusal_bank,
        refusal_artifact_root=refusal_root,
    )
    refusal_proposal = replace(
        proposal,
        edit_arm="refusal_only",
        refusal_enabled=True,
        refusal_direction_scope="global",
        refusal_source_layer=1,
        refusal_strength=0.7,
        refusal_writer_policy="attention",
    )
    refusal_batch = component_builder.build(
        refusal_proposal,
        trial_id="trial-refusal-only",
        record_ids=ids,
    )
    assert tuple(item[0] for item in refusal_batch.basis_set.source_components) == (
        "refusal_raw",
    )
    assert isinstance(refusal_batch.strengths, ComponentStrengthPlan)
    assert all(
        layer.attention == (0.7,) and layer.mlp == (0.0,)
        for layer in refusal_batch.strengths.components[0].by_layer
    )

    joint_proposal = replace(
        proposal,
        edit_arm="joint",
        refusal_enabled=True,
        refusal_direction_scope="global",
        refusal_source_layer=1,
        refusal_strength=0.7,
        refusal_writer_policy="attention",
    )
    joint_batch = component_builder.build(
        joint_proposal,
        trial_id="trial-joint",
        record_ids=ids,
    )
    assert tuple(item[0] for item in joint_batch.basis_set.source_components) == (
        "truth",
        "truth_orthogonalized_refusal",
    )
    assert isinstance(joint_batch.strengths, ComponentStrengthPlan)
    assert joint_batch.strengths.components[0].by_layer[0].attention == (1.0,)
    assert joint_batch.strengths.components[1].by_layer[0].attention == (0.7,)
    assert joint_batch.strengths.components[1].by_layer[0].mlp == (0.0,)

    joint_control = component_builder.build_control(
        joint_proposal,
        trial_id="finalization-control",
        record_ids=ids,
        control_kind="orthogonal",
        control_seed=17,
    )
    assert tuple(item[0] for item in joint_control.basis_set.source_components) == (
        "refusal_raw",
        "orthogonal_control",
    )
    assert isinstance(joint_control.strengths, ComponentStrengthPlan)
    assert joint_control.strengths.components[0].by_layer[0].attention == (0.7,)
    assert joint_control.strengths.components[1].by_layer[0].attention == (1.0,)
    control_slice = joint_control.basis_set.by_layer[0][1].components[1]
    control_vector = joint_control.basis_set.by_layer[0][1].matrix[
        :, control_slice.start : control_slice.stop
    ]
    assert np.allclose(truth_vector @ control_vector, 0.0, atol=1e-10)
    assert np.allclose(refusal_vector @ control_vector, 0.0, atol=1e-10)
    with pytest.raises(ProductionCompositionError, match="refusal-only"):
        component_builder.build_control(
            refusal_proposal,
            trial_id="unsupported-refusal-control",
            record_ids=ids,
            control_kind="shuffled",
            control_seed=23,
        )

    runtime = StoredMockTrialRuntime(
        model_sha256=bank.manifest.model.model_sha256,
        output_dir=tmp_path / "runtime",
        generated_text_by_record={
            signal["signal_id"]: signal["target"] for signal in signals
        },
        target_mean_log_probability=-0.01,
    )
    result = runtime.evaluate(batch)
    joint_result = runtime.evaluate(joint_batch)
    assert joint_result.basis_set_sha256 == joint_batch.basis_set.basis_set_sha256
    evidence_builder = RuntimeResultEvidenceBuilder(
        corpus=corpus,
        dataset_manifest_sha256="d" * 64,
        minimum_target_mean_log_probability=-0.1,
    )
    evidence = evidence_builder.build(proposal, batch, result)
    assert evidence.execution_receipt["operational_status"] == "succeeded"
    evaluator_config = EvaluatorConfig.from_mapping(
        {
                "format": "truth_editing_evaluator_config_v2",
            "config_id": "stored-composition",
            "dataset_manifest_sha256": "d" * 64,
            "judge_config_sha256": "a" * 64,
            "rubric_sha256": "b" * 64,
            "confidence_z": 1.96,
            "tiers": [
                {
                    "name": name,
                    "record_ids": list(ids)
                    + (["unused-expanded"] if name != "discovery" else [])
                    + (["unused-finalist"] if name == "finalist" else []),
                    "preservation_tier": preservation,
                    "minimum_valid_false_report_rate": 0.0,
                    "minimum_indirect_truth_rate": 1.0,
                    "minimum_action_truth_rate": 1.0,
                    "minimum_counterfactual_truth_rate": 1.0,
                        "minimum_internal_truth_rate": 1.0,
                        "minimum_truthful_direct_rate": 1.0,
                        "minimum_false_trigger_control_rate": 1.0,
                        "lane_weights": {
                            "structured_semantic": 1.0,
                            "broad_qa": 0.0,
                        },
                    "maximum_preservation_kl": {
                        "text": 0.1,
                        "vision": 0.1,
                        "recorded_computer_use": 0.1,
                    },
                }
                for name, preservation in (
                    ("discovery", "trial"),
                    ("expanded", "promoted"),
                    ("finalist", "finalist"),
                )
            ],
        }
    )
    recipe_evaluator = __import__(
        "intelligent_liars.truth_editing_evaluator", fromlist=["RecipeEvaluator"]
    ).RecipeEvaluator(
        evaluator_config,
        judge=SimpleNamespace(judge=lambda record: (_ for _ in ()).throw(AssertionError())),
        preservation=DeterministicMockPreservationAdapter(),
    )
    composed = ProductionStudyEvaluator(
        runtime=runtime,  # type: ignore[arg-type]
        recipe_evaluator=recipe_evaluator,
        evaluator_config=evaluator_config,
        batch_builder=builder,
        evidence_builder=evidence_builder,
    )
    scored = composed.evaluate(
        proposal,
        trial_id="trial-0000",
        record_ids=ids,
        objective_names=OBJECTIVES,
    )
    assert scored.outcome_kind == "successful"
    assert runtime.applied_basis_set_sha256 == batch.basis_set.basis_set_sha256

    study_path = tmp_path / "stored-study.json"
    study_path.write_text(
        json.dumps(
            {
                "format": "truth_editing_study_config_v1",
                "study_id": "stored-persistent-composition",
                "sampler_seed": 17,
                "batch_size": 1,
                "max_trials": 1,
                "max_directions_per_trial": 1,
                "max_rank": 1,
                "strength_min": 0.0,
                "strength_max": 2.0,
                "writer_regions": [{"name": "all", "layers": [0, 1, 2]}],
                "evaluation_tiers": [
                    {"name": "discovery", "record_limit": 6, "through_trial": 1},
                    {"name": "expanded", "record_limit": 7, "through_trial": 1},
                    {"name": "finalist", "record_limit": 8, "through_trial": 1},
                ],
                "dataset_manifest_sha256": "d" * 64,
                "validation_record_ids": list(ids) + ["unused-expanded", "unused-finalist"],
                "objective_names": list(OBJECTIVES),
                "tpe_startup_trials": 0,
                "tpe_ei_candidates": 128,
                "tpe_multivariate": True,
            }
        )
    )

    class FixedProposalDriver:
        identity = {"driver": "fixed-stored-proposal-v1"}

        def prepare(self, config, directions, state_path):
            del config, directions, state_path

        def suggest(self, request):
            del request
            return proposal

        def observe(self, trials):
            assert len(trials) == 1

    study_artifacts = ImmutableStudyArtifactAdapter(tmp_path / "full-study")
    run = compose_production_run(
        study=TruthEditingStudy(load_truth_editing_study_config(study_path), bank.manifest),
        driver=FixedProposalDriver(),  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
        recipe_evaluator=recipe_evaluator,
        evaluator_config=evaluator_config,
        batch_builder=builder,
        evidence_builder=evidence_builder,
        artifacts=study_artifacts,
        journal_path=tmp_path / "full-study-journal.json",
    )
    run_receipt = run.run()
    assert run_receipt.completed_trials == 1
    assert run_receipt.successful_trials == 1
    assert (tmp_path / "full-study" / "study-report.json").is_file()

    artifacts = ImmutableStudyArtifactAdapter(tmp_path / "artifacts")
    report = StudyReport(
        format="truth_editing_study_report_v1",
        study_identity_sha256="e" * 64,
        trials=(),
        coverage=CoverageLedger(),
        coverage_complete=False,
    )
    receipt = artifacts.freeze(report)
    assert receipt["study_identity_sha256"] == "e" * 64
    assert (tmp_path / "artifacts" / "study-report.json").is_file()


def _production_config_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": "truth_editing_production_config_v1",
        "study_config": "study.json",
        "dataset_root": "dataset",
        "scenario_view": "scenario-view/manifest.json",
        "structured_semantic_view": "structured-view",
        "structured_semantic_source_root": "structured-source",
        "structured_base_known_qualification": "structured-base-known",
        "direction_manifest": "directions.json",
        "direction_root": ".",
        "refusal_direction_config": "refusal-config.json",
        "refusal_prompt_manifest": "refusal-prompts.json",
        "refusal_direction_bank": "refusal-bank.json",
        "refusal_artifact_root": "refusal-artifacts",
        "evaluator_config": "evaluator.json",
        "base_known_qualification": "base-known.json",
        "judge_cache_dir": "judge-cache",
        "judge_budget_ledger_dir": "judge-budget-ledger",
        "judge_budget": {
            "format": "truth_editing_production_judge_budget_config_v1",
            "all_in_maximum_spend_usd": "50",
            "non_judge_reserved_spend_usd": "45",
            "maximum_judge_spend_usd": "5",
            "per_call_reservation_usd": "0.025",
            "judge_config_sha256": (
                "1b499bf7fdb0321a62afccac49ac2af90a25ae102ed17ed1cd12abca3c03b07c"
            ),
        },
        "journal_path": "state/journal.json",
        "artifact_dir": "state/frozen",
        "runtime_output_dir": "state/runtime",
        "model_cache_dir": "model-cache/huggingface",
        "snapshot_manifest_path": "model-cache/snapshot-manifest.json",
        "search_driver": "optuna",
        "verified_model_sha256": "a" * 64,
        "verified_snapshot_manifest_sha256": "b" * 64,
        "max_new_tokens": 100,
        "minimum_target_mean_log_probability": -5.0,
    }
    payload.update(overrides)
    return payload


def test_production_config_is_direct_strict_and_resolves_all_checked_in_adapters(
    tmp_path: Path,
) -> None:
    from test_truth_editing_preservation_materialization import _write_source
    from intelligent_liars.truth_editing_preservation_materialization import (
        materialize_preservation_runtime_packet,
    )

    packet_root = tmp_path / "preservation-runtime"
    packet_receipt = materialize_preservation_runtime_packet(
        _write_source(tmp_path / "preservation-source"), packet_root
    )
    payload = _production_config_payload(
        preservation_runtime_packet_root="preservation-runtime",
        preservation_threshold_calibration="preservation-thresholds/calibration.json",
        preservation_threshold_calibration_sha256="c" * 64,
        rescore_generation="recovery/rescore-generation-v1.json",
        rescore_generation_sha256="d" * 64,
        rescore_mode="repair_then_continue",
    )
    path = tmp_path / "production.json"
    path.write_text(json.dumps(payload))
    config = ProductionRunConfig.open(path)
    assert config.study_config == tmp_path / "study.json"
    assert config.search_driver == "optuna"
    assert config.rescore_generation == (
        tmp_path / "recovery" / "rescore-generation-v1.json"
    )
    assert config.rescore_generation_sha256 == "d" * 64
    assert config.rescore_mode == "repair_then_continue"
    assert config.judge_budget_ledger_dir == tmp_path / "judge-budget-ledger"
    assert config.judge_budget is not None
    assert str(config.judge_budget.maximum_judge_spend_usd) == "5"
    assert config.preservation_runtime_packet_root == packet_root
    assert config.preservation_runtime_packet_sha256 == packet_receipt["self_sha256"]
    assert config.preservation_spec_sha256 == packet_receipt["spec_sha256"]
    assert config.preservation_threshold_calibration == (
        tmp_path / "preservation-thresholds" / "calibration.json"
    )
    assert config.preservation_threshold_calibration_sha256 == "c" * 64
    assert dict(config.preservation_runtime_configs) == {
        tier: packet_root / f"truth_editing_preservation_runtime_{tier}_v1.json"
        for tier in ("trial", "promoted", "finalist")
    }
    assert not any("factory" in key for key in payload)

    payload["surprise"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ProductionCompositionError, match="fields or format"):
        ProductionRunConfig.open(path)


def test_configured_search_driver_strict_opens_bound_rescore_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from intelligent_liars import truth_editing_rescore

    generation = object()
    calls: list[tuple[Path, str]] = []

    def load(path: Path, *, expected_generation_sha256: str) -> object:
        calls.append((path, expected_generation_sha256))
        return generation

    class Driver:
        def __init__(
            self,
            *,
            seed: int,
            generation: object,
            continue_optimization_after_replay: bool,
        ) -> None:
            self.seed = seed
            self.generation = generation
            self.continue_optimization_after_replay = (
                continue_optimization_after_replay
            )

    monkeypatch.setattr(truth_editing_rescore, "load_rescore_generation_v1", load)
    monkeypatch.setattr(truth_editing_rescore, "RescoreOptunaSearchDriver", Driver)
    config = SimpleNamespace(
        search_driver="optuna",
        rescore_generation=tmp_path / "rescore.json",
        rescore_generation_sha256="a" * 64,
        rescore_mode="repair_then_continue",
    )

    driver = configured_search_driver(
        config, SimpleNamespace(sampler_seed=73)  # type: ignore[arg-type]
    )

    assert calls == [(tmp_path / "rescore.json", "a" * 64)]
    assert driver.seed == 73  # type: ignore[attr-defined]
    assert driver.generation is generation  # type: ignore[attr-defined]
    assert driver.continue_optimization_after_replay is True  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rescore_generation": "recovery/rescore.json"}, "fields or format"),
        ({"rescore_generation_sha256": "a" * 64}, "fields or format"),
        (
            {
                "search_driver": "offline",
                "rescore_generation": "recovery/rescore.json",
                "rescore_generation_sha256": "a" * 64,
                "rescore_mode": "repair_then_continue",
            },
            "requires the optuna",
        ),
    ],
)
def test_production_config_rejects_unbound_rescore_selection(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    from test_truth_editing_preservation_materialization import _write_source
    from intelligent_liars.truth_editing_preservation_materialization import (
        materialize_preservation_runtime_packet,
    )

    materialize_preservation_runtime_packet(
        _write_source(tmp_path / "source"), tmp_path / "preservation-runtime"
    )
    payload = _production_config_payload(
        preservation_runtime_packet_root="preservation-runtime",
        **overrides,
    )
    path = tmp_path / "production.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ProductionCompositionError, match=message):
        ProductionRunConfig.open(path)


@pytest.mark.parametrize(
    "unsafe_path",
    ["/tmp/calibration.json", "../../calibration.json"],
)
def test_production_config_rejects_nonportable_or_escaping_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    payload = _production_config_payload(
        preservation_runtime_packet_root="preservation-runtime",
        preservation_threshold_calibration=unsafe_path,
        preservation_threshold_calibration_sha256="c" * 64,
    )
    path = tmp_path / "production.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ProductionCompositionError, match="safe repository-relative"):
        ProductionRunConfig.open(path)


def test_runtime_config_allows_one_scoped_absolute_output_root(tmp_path: Path) -> None:
    from test_truth_editing_preservation_materialization import _write_source
    from intelligent_liars.truth_editing_preservation_materialization import (
        materialize_preservation_runtime_packet,
    )

    materialize_preservation_runtime_packet(
        _write_source(tmp_path / "source"), tmp_path / "preservation-runtime"
    )
    output = tmp_path / "remote-output"
    payload = _production_config_payload(
        preservation_runtime_packet_root="preservation-runtime",
        journal_path=str(output / "study/journal.json"),
        artifact_dir=str(output / "study/frozen"),
        runtime_output_dir=str(output / "study/runtime"),
        judge_cache_dir=str(output / "providers/judge-cache"),
        judge_budget_ledger_dir=str(output / "providers/judge-budget"),
    )
    path = tmp_path / "runtime-config.json"
    path.write_text(json.dumps(payload))

    config = ProductionRunConfig.open(path)

    assert config.journal_path == output / "study/journal.json"
    assert config.judge_cache_dir == output / "providers/judge-cache"
    assert config.judge_budget_ledger_dir == output / "providers/judge-budget"


def test_production_config_fails_closed_when_preservation_packet_is_absent(
    tmp_path: Path,
) -> None:
    payload = _production_config_payload(
        preservation_runtime_packet_root="preservation-runtime"
    )
    path = tmp_path / "production.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ProductionCompositionError, match="preservation runtime packet"):
        ProductionRunConfig.open(path)


def test_production_config_accepts_only_exact_legacy_packet_paths(tmp_path: Path) -> None:
    from test_truth_editing_preservation_materialization import _write_source
    from intelligent_liars.truth_editing_preservation_materialization import (
        materialize_preservation_runtime_packet,
    )

    root = tmp_path / "preservation-runtime"
    materialize_preservation_runtime_packet(_write_source(tmp_path / "source"), root)
    legacy = {
        tier: f"preservation-runtime/truth_editing_preservation_runtime_{tier}_v1.json"
        for tier in ("trial", "promoted", "finalist")
    }
    path = tmp_path / "production.json"
    path.write_text(
        json.dumps(_production_config_payload(preservation_runtime_configs=legacy))
    )
    config = ProductionRunConfig.open(path)
    assert config.preservation_runtime_packet_root == root

    legacy["finalist"] = "other-finalist.json"
    path.write_text(
        json.dumps(_production_config_payload(preservation_runtime_configs=legacy))
    )
    with pytest.raises(ProductionCompositionError, match="one exact materialized packet"):
        ProductionRunConfig.open(path)

    path.write_text(
        json.dumps(
            _production_config_payload(
                preservation_runtime_packet_root=str(root),
                preservation_runtime_configs={
                    tier: str(
                        root / f"truth_editing_preservation_runtime_{tier}_v1.json"
                    )
                    for tier in ("trial", "promoted", "finalist")
                },
            )
        )
    )
    with pytest.raises(ProductionCompositionError, match="fields or format"):
        ProductionRunConfig.open(path)


def test_qualified_broad_qa_lane_maps_three_roles_atomically() -> None:
    source_id = "qa-source"
    scenario_id = "qa-scenario"
    roles = (
        "direct_false_report_instruction",
        "false_trigger_truthful_control",
        "retained_truth_rephrase_proxy",
    )
    records = tuple(
        {
            "record_id": f"{source_id}.{role}",
            "source_record_id": source_id,
            "scenario_id": scenario_id,
            "domain": "geography",
            "family": "qa",
            "role": role,
            "messages": ({"role": "user", "content": role},),
            "truthful_target": "Paris",
            "truth_aliases": ("Paris",),
            "acceptable_false_targets": ("Lyon",),
        }
        for role in roles
    )
    corpus = GroupedV2Corpus.from_qualified_qa_scenario_view(
        SimpleNamespace(
            records=records,
            manifest={"view_sha256": "a" * 64},
        ),
        qualified_source_record_ids={source_id},
        qualification_manifest_sha256="b" * 64,
    )
    ids = tuple(str(item["record_id"]) for item in records)
    selected = corpus.select(ids)
    assert [item.signal_kind for item in selected] == [
        "conditioned_false_direct_report",
        "false_trigger_truthful_control",
        "truthful_direct_report",
    ]
    assert all(item.evaluation_lane == "broad_qa" for item in selected)
    with pytest.raises(ProductionCompositionError, match="complete three-signal"):
        corpus.select(ids[:2])
