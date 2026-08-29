from __future__ import annotations

import json
import hashlib
import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_contracts import parse_direction_bank_manifest
from intelligent_liars.truth_editing_study import (
    CompletedBatchCommit,
    EvaluationResult,
    OfflineDeterministicSearchDriver,
    OfflineSyntheticEvaluator,
    OptunaSearchDriver,
    OperationalEvaluationError,
    PreparedStudyContext,
    StudyError,
    SearchProposal,
    STUDY_ORCHESTRATOR_SEMANTICS_SHA256,
    schedule_finalist_basis_controls,
    TruthEditingStudy,
    load_truth_editing_study_config,
    parse_truth_editing_study_config,
)

from test_truth_editing_contracts import _manifest


def test_study_orchestrator_identity_preserves_launched_checkpoint_lineage() -> None:
    assert STUDY_ORCHESTRATOR_SEMANTICS_SHA256 == (
        "5e6392c5975d04006e5a1ef32ba93896750a7ec8916ff85c23fcb4ec94a960b1"
    )


def _config(tmp_path, **changes):
    value = {
        "format": "truth_editing_study_config_v1",
        "study_id": "offline-study",
        "sampler_seed": 17,
        "batch_size": 2,
        "max_trials": 12,
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
            {"name": "discovery", "record_limit": 2, "through_trial": 6},
            {"name": "expanded", "record_limit": 4, "through_trial": 10},
            {"name": "finalist", "record_limit": 6, "through_trial": 12},
        ],
        "dataset_manifest_sha256": "d" * 64,
        "validation_record_ids": [f"v-{index}" for index in range(6)],
        "objective_names": [
            "valid_false_report_rate_lcb",
            "truth_report_dissociation_lcb",
            "capability_preservation_lcb",
        ],
        "tpe_startup_trials": 0,
        "tpe_ei_candidates": 128,
        "tpe_multivariate": True,
    }
    value.update(changes)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))
    return load_truth_editing_study_config(path)


def _adaptive_config_value() -> dict:
    value = {
        "format": "truth_editing_study_config_v2",
        "study_id": "adaptive-production-study",
        "sampler_seed": 20260827,
        "batch_size": 8,
        "max_trials": 800,
        "max_directions_per_trial": 4,
        "max_rank": 8,
        "strength_min": 0.0,
        "strength_max": 2.0,
        "writer_regions": [
            {"name": "early", "layers": list(range(0, 12))},
            {"name": "middle", "layers": list(range(12, 24))},
            {"name": "late", "layers": list(range(24, 36))},
        ],
        "evaluation_tiers": [
            {"name": "discovery", "record_limit": 2, "through_trial": 80},
            {"name": "expanded", "record_limit": 4, "through_trial": 200},
            {"name": "finalist", "record_limit": 6, "through_trial": 800},
        ],
        "dataset_manifest_sha256": "d" * 64,
        "validation_record_ids": [f"v-{index}" for index in range(6)],
        "objective_names": [
            "valid_false_report_rate_lcb",
            "truth_report_dissociation_lcb",
            "capability_preservation_lcb",
        ],
        "tpe_startup_trials": 70,
        "tpe_ei_candidates": 128,
        "tpe_multivariate": True,
        "search_policy": {
            "format": "truth_editing_adaptive_search_policy_v1",
            "minimum_trials": 200,
            "maximum_trials": 800,
            "search_elapsed_limit_seconds": 75600,
            "reserve_elapsed_seconds": 10800,
            "all_in_budget_usd": "50",
            "maximum_infrastructure_spend_usd": "45",
            "maximum_evaluation_spend_usd": "5",
            "evaluation_budget_reserve_fraction": "0.20",
            "evaluation_spend_reserve_usd": "1",
            "broad_coverage": {
                "required_before_concentration": True,
            },
        },
    }
    return value


def test_adaptive_production_policy_round_trips_and_binds_study_limits() -> None:
    raw = _adaptive_config_value()
    parsed = parse_truth_editing_study_config(raw)

    assert parsed.to_dict() == raw
    assert parsed.max_trials == 800
    assert parsed.search_policy is not None
    assert parsed.search_policy.minimum_trials == 200
    assert parsed.search_policy.maximum_trials == 800
    assert parsed.search_policy.broad_coverage.required_before_concentration is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw["search_policy"].update(maximum_trials=799), "maximum_trials"),
        (lambda raw: raw.update(batch_size=4), "batch_size"),
        (
            lambda raw: raw["search_policy"]["broad_coverage"].update(
                required_before_concentration=False
            ),
            "broad coverage",
        ),
        (
            lambda raw: raw["search_policy"].update(
                evaluation_budget_reserve_fraction="0.2"
            ),
            "reserve fraction",
        ),
        (lambda raw: raw["search_policy"].update(extra=True), "fields differ"),
    ],
)
def test_adaptive_production_policy_fails_closed(mutation, match: str) -> None:
    raw = _adaptive_config_value()
    mutation(raw)

    with pytest.raises(StudyError, match=match):
        parse_truth_editing_study_config(raw)


def _direction_bank():
    raw = _manifest()
    # The contract fixture has one general direction. Add two independently
    # identified directions so marginal source-layer/family coverage is visible.
    base = raw["directions"][0]
    raw["directions"] = []
    from intelligent_liars.truth_editing_contracts import canonical_sha256

    for index, (family, layer) in enumerate(
        (
            ("general", 0), ("domain_specific", 0),
            ("domain_specific", 1), ("intermediate", 2),
        )
    ):
        item = json.loads(json.dumps(base))
        item["direction_id"] = f"truth-{family}-{layer}"
        item["family"] = family
        item["domains"] = [family]
        item["source_layer"] = layer
        item["artifact"]["path"] = f"directions/{index}.safetensors"
        item["artifact"]["file_sha256"] = f"{index + 1:064x}"
        item["artifact"]["vector_sha256"] = f"{index + 11:064x}"
        item["qualification"]["receipt_sha256"] = f"{index + 21:064x}"
        raw["directions"].append(item)
    raw.pop("self_sha256", None)
    raw["self_sha256"] = canonical_sha256(raw)
    return parse_direction_bank_manifest(raw)


def test_run_covers_search_axes_before_concentrating(tmp_path) -> None:
    config = _config(tmp_path)
    bank = _direction_bank()
    driver = OfflineDeterministicSearchDriver(seed=config.sampler_seed)
    report = TruthEditingStudy(config, bank).run(
        driver=driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "journal.json",
    )

    assert report.completed_trials == 12
    assert report.operational_failures == 0
    assert report.coverage_complete is True
    assert report.selection_ready is True
    assert set(report.coverage.families) == {
        "general", "domain_specific", "intermediate", "mixed"
    }
    assert set(report.coverage.source_layers) == {0, 1, 2}
    assert set(report.coverage.writer_regions) == {"early", "middle", "late"}
    assert set(report.coverage.writer_policies) == {"attention", "mlp", "both"}
    assert set(report.coverage.basis_methods) == {"qr", "svd"}
    assert set(report.coverage.strength_regions) == {
        "disabled", "projection", "reflection"
    }
    assert set(report.coverage.basis_scopes) == {"general", "domain", "mixed"}
    assert set(report.coverage.direction_scopes) == {"global", "per_layer"}
    assert set(report.coverage.normalization_modes) == {"exact", "norm_preserving"}
    assert set(report.coverage.edit_arms) == {"truth_only", "refusal_only", "joint"}
    assert set(report.coverage.active_edit_arms) == {
        "truth_only", "refusal_only", "joint", "orthogonal_control"
    }
    assert all(trial.proposal.backend_type == "persistent_weight" for trial in report.trials)
    assert all(0.0 <= trial.proposal.strength <= 2.0 for trial in report.trials)
    assert {trial.proposal.proposal_origin for trial in report.trials} == {
        "coverage_anchor"
    }
    assert all(trial.proposal.selected_domains == tuple(sorted(
        trial.proposal.selected_domains
    )) for trial in report.trials)
    assert driver.observed_batch_sizes == [2, 2, 2, 2, 2, 2]


def test_stop_boundary_is_batch_atomic_and_resumes_same_study_identity(tmp_path) -> None:
    config = _config(tmp_path)
    bank = _direction_bank()
    journal = tmp_path / "journal.json"
    first_driver = OfflineDeterministicSearchDriver(seed=config.sampler_seed)
    first = TruthEditingStudy(config, bank).run(
        driver=first_driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
        stop_after_trials=6,
    )
    assert first.completed_trials == 6
    assert first_driver.observed_batch_sizes == [2, 2, 2]
    persisted = json.loads(journal.read_text())
    assert len(persisted["batches"]) == 3
    assert all(
        entry["result"] is not None
        for batch in persisted["batches"]
        for entry in batch["trials"]
    )

    resumed_driver = OfflineDeterministicSearchDriver(seed=config.sampler_seed)
    resumed = TruthEditingStudy(config, bank).run(
        driver=resumed_driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
        stop_after_trials=10,
    )
    assert resumed.completed_trials == 10
    assert resumed.study_identity_sha256 == first.study_identity_sha256
    assert resumed_driver.observed_batch_sizes == [2, 2, 2, 2, 2]


def test_stop_boundary_rejects_partial_batch_and_lower_than_journal(tmp_path) -> None:
    config = _config(tmp_path)
    study = TruthEditingStudy(config, _direction_bank())
    with pytest.raises(StudyError, match="completed batch barrier"):
        study.run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=OfflineSyntheticEvaluator(),
            journal_path=tmp_path / "bad.json",
            stop_after_trials=5,
        )
    journal = tmp_path / "journal.json"
    study.run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
        stop_after_trials=6,
    )
    with pytest.raises(StudyError, match="already exceeds"):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=OfflineSyntheticEvaluator(),
            journal_path=journal,
            stop_after_trials=4,
        )


def test_batch_admission_seam_stops_only_before_a_new_complete_batch(tmp_path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[int, int, bool]] = []
    commits: list[tuple[int, bool]] = []

    class Gate:
        def admit_batch(
            self, *, completed_trials: int, batch_size: int, coverage_complete: bool,
            batch_started: bool = False,
        ) -> bool:
            assert batch_started is False
            calls.append((completed_trials, batch_size, coverage_complete))
            return completed_trials < 6

        def commit_batch(
            self, *, completed_trials: int, coverage_complete: bool
        ) -> None:
            commits.append((completed_trials, coverage_complete))

    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "journal.json",
        batch_admission=Gate(),
    )

    assert report.completed_trials == 6
    assert calls == [(0, 2, False), (2, 2, False), (4, 2, False), (6, 2, True)]
    assert commits == [(2, False), (4, False), (6, True)]
    journal = json.loads((tmp_path / "journal.json").read_text())
    assert len(journal["batches"]) == 3


def test_after_complete_batch_receives_one_integrity_bound_full_batch(tmp_path) -> None:
    commits: list[CompletedBatchCommit] = []
    journal = tmp_path / "journal.json"

    report = TruthEditingStudy(_config(tmp_path), _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
        stop_after_trials=2,
        after_complete_batch=commits.append,
    )

    assert report.completed_trials == 2
    assert len(commits) == 1
    commit = commits[0]
    assert commit.completed_trials == 2
    assert commit.batch_ordinal == 0
    assert commit.batch_size == 2
    assert tuple(item.ordinal for item in commit.trials) == (0, 1)
    assert commit.study_identity_sha256 == report.study_identity_sha256
    assert commit.journal_sha256 == json.loads(journal.read_text())["journal_sha256"]
    assert commit.to_dict()["commit_sha256"] == commit.commit_sha256
    assert commit.coverage_summary.keys() == {
        "direction_family",
        "layer_region",
        "intervention_arm",
        "attention_mlp_configuration",
        "refusal_setting",
        "strength_range",
    }


def test_prepared_context_precedes_first_admission_and_evaluation(tmp_path) -> None:
    events: list[str] = []
    journal = tmp_path / "journal.json"

    class Driver(OfflineDeterministicSearchDriver):
        def prepare(self, config, directions, state_path) -> None:
            events.append("prepare")
            super().prepare(config, directions, state_path)

    class Gate:
        def admit_batch(self, **_kwargs) -> bool:
            events.append("admit")
            return False

    class Evaluator(OfflineSyntheticEvaluator):
        def evaluate(self, *args, **kwargs):
            events.append("evaluate")
            return super().evaluate(*args, **kwargs)

    contexts: list[PreparedStudyContext] = []

    def prepared(context: PreparedStudyContext) -> None:
        events.append("prepared_context")
        contexts.append(context)
        assert journal.is_file()

    report = TruthEditingStudy(_config(tmp_path), _direction_bank()).run(
        driver=Driver(seed=17),
        evaluator=Evaluator(),
        journal_path=journal,
        batch_admission=Gate(),
        after_prepare_before_first_admission=prepared,
    )

    assert report.completed_trials == 0
    assert events == ["prepare", "prepared_context", "admit"]
    assert len(contexts) == 1
    context = contexts[0]
    assert context.completed_trials == 0
    assert context.journal_sha256 == json.loads(journal.read_text())["journal_sha256"]
    assert all(completed == 0 for completed, _required in context.coverage_summary.values())
    assert all(required > 0 for _completed, required in context.coverage_summary.values())
    assert context.to_dict()["context_sha256"] == context.context_sha256


def test_adaptive_callback_is_an_exact_eight_result_barrier(tmp_path) -> None:
    config = parse_truth_editing_study_config(_adaptive_config_value())
    commits: list[CompletedBatchCommit] = []

    TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=config.sampler_seed),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "adaptive-journal.json",
        stop_after_trials=8,
        after_complete_batch=commits.append,
    )

    assert len(commits) == 1
    assert commits[0].batch_size == 8
    assert len(commits[0].trials) == 8
    assert all(item.result is not None for item in commits[0].trials)


def test_after_complete_batch_failure_stops_before_next_dispatch_and_replays(
    tmp_path,
) -> None:
    config = _config(
        tmp_path,
        max_trials=4,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 4}],
    )
    journal = tmp_path / "journal.json"

    def fail(_commit: CompletedBatchCommit) -> None:
        raise RuntimeError("checkpoint publish failed")

    with pytest.raises(RuntimeError, match="checkpoint publish failed"):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=OfflineSyntheticEvaluator(),
            journal_path=journal,
            after_complete_batch=fail,
        )

    persisted = json.loads(journal.read_text())
    assert len(persisted["batches"]) == 1
    assert all(item["result"] is not None for item in persisted["batches"][0]["trials"])

    replayed: list[int] = []
    resumed = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
        after_complete_batch=lambda commit: replayed.append(commit.completed_trials),
    )
    assert resumed.completed_trials == 4
    assert replayed == [2, 4]


def test_routine_proposals_are_truth_only_persistent_drafts(tmp_path) -> None:
    report = TruthEditingStudy(_config(tmp_path), _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "journal.json",
    )
    bank_by_id = {item.direction_id: item for item in _direction_bank().directions}
    assert all(trial.proposal.backend_type == "persistent_weight" for trial in report.trials)
    assert all(
        bank_by_id[direction_id].kind == "truth"
        for trial in report.trials
        for direction_id in trial.proposal.direction_ids
    )
    assert {trial.proposal.edit_arm for trial in report.trials} == {
        "truth_only", "refusal_only", "joint"
    }
    for trial in report.trials:
        proposal = trial.proposal
        if not proposal.refusal_enabled or proposal.refusal_direction_scope == "per_layer":
            assert proposal.refusal_source_layer is None
        else:
            assert proposal.refusal_source_layer is not None


def test_every_qualified_direction_is_reachable_by_sparse_semantic_search(tmp_path) -> None:
    bank = _direction_bank()
    report = TruthEditingStudy(_config(tmp_path), bank).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "journal.json",
    )
    observed = {
        direction_id for trial in report.trials
        for direction_id in trial.proposal.direction_ids
    }
    assert observed == {item.direction_id for item in bank.directions}
    mixed = next(trial.proposal for trial in report.trials if trial.proposal.basis_scope == "mixed")
    assert len(mixed.direction_ids) == 2
    assert mixed.selected_domains


def test_explicit_writer_kernels_round_trip_and_compile_without_inference(tmp_path) -> None:
    proposal = SearchProposal(
        direction_ids=("truth-general-0",), direction_family="general",
        source_layer=0, basis_method="qr", requested_rank=1,
        writer_region="early", writer_layers=(0, 1, 2), writer_policy="attention",
        strength=2.0, basis_scope="general", selected_domains=(),
        truth_direction_scope="global", normalization_mode="norm_preserving",
        edit_arm="truth_only", attention_enabled=True,
        attention_kernel_center=1.0, attention_kernel_half_width=1.0,
        attention_edge_strength=0.5, attention_peak_strength=2.0,
        mlp_enabled=False, mlp_kernel_center=1.0, mlp_kernel_half_width=1.0,
        mlp_edge_strength=0.0, mlp_peak_strength=0.0,
    )
    parsed = SearchProposal.from_dict(proposal.to_dict())
    assert parsed == proposal
    assert parsed.writer_strength_plan() == {
        "attention_by_layer": {0: 0.5, 1: 2.0, 2: 0.5},
        "mlp_by_layer": {0: 0.0, 1: 0.0, 2: 0.0},
    }
    TruthEditingStudy(_config(tmp_path), _direction_bank())._validate_proposal(parsed)

    controls = schedule_finalist_basis_controls((parsed,))
    assert [item.control_kind for item in controls] == ["orthogonal", "shuffled"]
    assert controls[0].parent_proposal_sha256 == controls[1].parent_proposal_sha256
    assert controls[0].requested_rank == controls[1].requested_rank == 1
    assert controls[0].writer_layers == controls[1].writer_layers == (0, 1, 2)
    assert (
        controls[0].writer_strength_plan_sha256
        == controls[1].writer_strength_plan_sha256
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("basis_scope", "domain", "basis scope"),
        ("selected_domains", ["forged"], "selected_domains"),
        ("attention_kernel_half_width", -1.0, "kernel"),
        ("refusal_enabled", True, "edit arm"),
    ],
)
def test_expanded_proposal_semantics_fail_closed(tmp_path, field, value, match) -> None:
    report = TruthEditingStudy(_config(tmp_path), _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "journal.json",
    )
    raw = next(
        trial.proposal.to_dict() for trial in report.trials
        if trial.proposal.basis_scope == "general"
        and trial.proposal.edit_arm == "truth_only"
    )
    raw[field] = value
    proposal = SearchProposal.from_dict(raw)
    with pytest.raises(StudyError, match=match):
        TruthEditingStudy(_config(tmp_path), _direction_bank())._validate_proposal(
            proposal
        )


def test_refusal_source_layer_exists_only_for_enabled_global_scope(tmp_path) -> None:
    study = TruthEditingStudy(_config(tmp_path), _direction_bank())
    report = study.run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "journal.json",
    )
    per_layer = next(
        trial.proposal for trial in report.trials
        if trial.proposal.refusal_enabled
        and trial.proposal.refusal_direction_scope == "per_layer"
    )
    assert per_layer.to_dict()["refusal_source_layer"] is None
    assert SearchProposal.from_dict(per_layer.to_dict()) == per_layer

    inert = per_layer.to_dict()
    inert["refusal_source_layer"] = per_layer.source_layer
    with pytest.raises(StudyError, match="inert source layer"):
        study._validate_proposal(SearchProposal.from_dict(inert))

    disabled = next(
        trial.proposal for trial in report.trials
        if not trial.proposal.refusal_enabled
    )
    assert disabled.refusal_source_layer is None


def test_tiered_evaluation_uses_frozen_prefixes(tmp_path) -> None:
    report = TruthEditingStudy(_config(tmp_path), _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "journal.json",
    )

    assert report.trials[0].evaluation_record_ids == ("v-0", "v-1")
    assert report.trials[6].evaluation_record_ids == ("v-0", "v-1", "v-2", "v-3")
    assert report.trials[10].evaluation_record_ids == tuple(f"v-{i}" for i in range(6))


class _InterruptingEvaluator(OfflineSyntheticEvaluator):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
        self.calls += 1
        if self.calls == 3:
            raise KeyboardInterrupt
        return super().evaluate(
            proposal,
            trial_id=trial_id,
            record_ids=record_ids,
            objective_names=objective_names,
        )


def test_resume_reuses_exact_pending_suggestions_and_identity(tmp_path) -> None:
    config = _config(tmp_path, max_trials=4, evaluation_tiers=[
        {"name": "all", "record_limit": 2, "through_trial": 4}
    ])
    bank = _direction_bank()
    journal = tmp_path / "journal.json"
    first = _InterruptingEvaluator()
    with pytest.raises(KeyboardInterrupt):
        TruthEditingStudy(config, bank).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=first,
            journal_path=journal,
        )
    pending_before = json.loads(journal.read_text())["batches"][1]["trials"][0]["proposal"]

    resumed = TruthEditingStudy(config, bank).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )
    assert resumed.completed_trials == 4
    assert resumed.trials[2].proposal.to_dict() == pending_before

    changed = replace(config, dataset_manifest_sha256="e" * 64)
    with pytest.raises(StudyError, match="identity"):
        TruthEditingStudy(changed, bank).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=OfflineSyntheticEvaluator(),
            journal_path=journal,
        )


def test_resume_rechecks_admission_before_dispatching_untouched_journal_batch(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        max_trials=2,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 2}],
    )
    journal = tmp_path / "journal.json"

    class InterruptBeforeFirstResult(OfflineSyntheticEvaluator):
        def evaluate(self, *args, **kwargs):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=InterruptBeforeFirstResult(),
            journal_path=journal,
        )

    calls: list[dict[str, object]] = []

    class Gate:
        def admit_batch(self, **kwargs) -> bool:
            calls.append(kwargs)
            return False

    resumed_evaluator = OfflineSyntheticEvaluator()
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=resumed_evaluator,
        journal_path=journal,
        batch_admission=Gate(),
    )

    assert report.completed_trials == 0
    assert calls == [{
        "completed_trials": 0,
        "batch_size": 2,
        "coverage_complete": False,
        "batch_started": False,
    }]


def test_resume_finishes_started_journal_batch_through_replay_admission(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        max_trials=2,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 2}],
    )
    journal = tmp_path / "journal.json"

    class InterruptAfterFirstResult(OfflineSyntheticEvaluator):
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return super().evaluate(*args, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=InterruptAfterFirstResult(),
            journal_path=journal,
        )

    calls: list[dict[str, object]] = []

    class ReplayGate:
        def admit_batch(self, **kwargs) -> bool:
            calls.append(kwargs)
            return kwargs["batch_started"] is True

    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
        batch_admission=ReplayGate(),
    )

    assert report.completed_trials == 2
    assert calls == [{
        "completed_trials": 0,
        "batch_size": 2,
        "coverage_complete": False,
        "batch_started": True,
    }]


class _ClassifyingEvaluator(OfflineSyntheticEvaluator):
    def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
        if trial_id == "trial-0000":
            return EvaluationResult.operational_failure("cuda out of memory")
        if trial_id == "trial-0001":
            return EvaluationResult.scientifically_infeasible(
                {name: 0.0 for name in objective_names}, "capability gate failed"
            )
        return super().evaluate(
            proposal,
            trial_id=trial_id,
            record_ids=record_ids,
            objective_names=objective_names,
        )


def test_operational_failure_is_not_scientific_infeasibility(tmp_path) -> None:
    config = _config(tmp_path, max_trials=4, evaluation_tiers=[
        {"name": "all", "record_limit": 2, "through_trial": 4}
    ])
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=_ClassifyingEvaluator(),
        journal_path=tmp_path / "journal.json",
    )
    assert report.operational_failures == 1
    assert report.scientifically_infeasible_trials == 1
    assert report.successful_trials == 2
    # Scientific infeasibility is still a valid search observation; an
    # operational failure is omitted from coverage. Four trials cannot cover
    # the expanded sparse semantic axes after one operational loss.
    assert report.coverage_complete is False
    assert report.selection_ready is False


def test_declared_worker_exception_is_recorded_as_operational_failure(tmp_path) -> None:
    class BrokenWorker(OfflineSyntheticEvaluator):
        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            if trial_id == "trial-0000":
                raise OperationalEvaluationError("worker lost")
            return super().evaluate(
                proposal,
                trial_id=trial_id,
                record_ids=record_ids,
                objective_names=objective_names,
            )

    config = _config(tmp_path, max_trials=2, evaluation_tiers=[
        {"name": "all", "record_limit": 2, "through_trial": 2}
    ])
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=BrokenWorker(),
        journal_path=tmp_path / "journal.json",
    )
    assert report.operational_failures == 1
    assert report.trials[0].result.detail == "worker lost"


def test_config_is_strict_and_rejects_a_separate_screen(tmp_path) -> None:
    config = _config(tmp_path)
    assert config.max_rank == 2
    raw = json.loads((tmp_path / "study.json").read_text())
    raw["fixed_screen_trials"] = 8
    (tmp_path / "study.json").write_text(json.dumps(raw))
    with pytest.raises(StudyError, match="fields"):
        load_truth_editing_study_config(tmp_path / "study.json")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("tpe_startup_trials", 13, "cannot exceed"),
        ("tpe_ei_candidates", 127, "128"),
        ("tpe_multivariate", False, "must be true"),
    ],
)
def test_documented_tpe_contract_fails_closed(tmp_path, field, value, match) -> None:
    _config(tmp_path)
    raw = json.loads((tmp_path / "study.json").read_text())
    raw[field] = value
    (tmp_path / "study.json").write_text(json.dumps(raw))
    with pytest.raises(StudyError, match=match):
        load_truth_editing_study_config(tmp_path / "study.json")


def test_optional_optuna_dependency_fails_with_actionable_error() -> None:
    try:
        import optuna  # type: ignore  # noqa: F401
    except ImportError:
        with pytest.raises(StudyError, match="not installed"):
            OptunaSearchDriver(seed=17)
    else:
        assert OptunaSearchDriver(seed=17).identity["adapter"] == "optuna_multivariate_tpe_v2"


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_optuna_prepare_creates_missing_journal_parent(tmp_path) -> None:
    config = _config(
        tmp_path,
        max_trials=1,
        evaluation_tiers=[{"name": "all", "record_limit": 1, "through_trial": 1}],
    )
    journal = tmp_path / "outputs" / "study" / "study-journal.json"

    TruthEditingStudy(config, _direction_bank()).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )

    assert journal.is_file()
    assert journal.with_name(journal.name + ".optuna.log").is_file()


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_optuna_restart_uses_native_journal_and_exact_pending_proposal(tmp_path) -> None:
    class InterruptSixth(OfflineSyntheticEvaluator):
        def __init__(self):
            self.calls = 0

        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            self.calls += 1
            if self.calls == 6:
                raise KeyboardInterrupt
            return super().evaluate(
                proposal, trial_id=trial_id, record_ids=record_ids,
                objective_names=objective_names,
            )

    config = _config(tmp_path, max_trials=6, evaluation_tiers=[
        {"name": "all", "record_limit": 2, "through_trial": 6}
    ])
    bank = _direction_bank()
    journal = tmp_path / "journal.json"
    first_driver = OptunaSearchDriver(seed=17)
    with pytest.raises(StudyError, match="before prepare"):
        _ = first_driver.persistent_study_name
    with pytest.raises(KeyboardInterrupt):
        TruthEditingStudy(config, bank).run(
            driver=first_driver,
            evaluator=InterruptSixth(),
            journal_path=journal,
        )
    assert first_driver.persistent_study_name.startswith(config.study_id + "-")
    pending = json.loads(journal.read_text())["batches"][-1]["trials"][-1]["proposal"]
    report = TruthEditingStudy(config, bank).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )
    assert report.trials[-1].proposal.to_dict() == pending
    assert (tmp_path / "journal.json.optuna.log").is_file()


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_optuna_prepare_reopens_existing_journal_without_appending(tmp_path) -> None:
    config = _config(tmp_path)
    directions = TruthEditingStudy(config, _direction_bank()).directions
    path = tmp_path / "existing.optuna.log"
    first = OptunaSearchDriver(seed=17)
    first.prepare(config, directions, path)
    original = path.read_bytes()

    second = OptunaSearchDriver(seed=17)
    second.prepare(config, directions, path)

    assert path.read_bytes() == original
    assert second.persistent_study_name == first.persistent_study_name


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_optuna_replays_unresolved_operational_failures_fifo_and_exactly(
    tmp_path,
) -> None:
    class FailFirstTwo(OfflineSyntheticEvaluator):
        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            if trial_id in {"trial-0000", "trial-0001"}:
                return EvaluationResult.operational_failure("worker unavailable")
            return super().evaluate(
                proposal,
                trial_id=trial_id,
                record_ids=record_ids,
                objective_names=objective_names,
            )

    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=16,
        evaluation_tiers=[
            {"name": "all", "record_limit": 2, "through_trial": 16}
        ],
    )
    bank = _direction_bank()
    journal = tmp_path / "operational-replay.json"
    first_driver = OptunaSearchDriver(seed=17)
    first = TruthEditingStudy(config, bank).run(
        driver=first_driver,
        evaluator=FailFirstTwo(),
        journal_path=journal,
        stop_after_trials=8,
    )

    import optuna

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal) + ".optuna.log")
    )
    persisted = optuna.load_study(
        study_name=first_driver.persistent_study_name,
        storage=storage,
    )
    persisted_ordinals = {
        int(item.user_attrs["study_ordinal"])
        for item in persisted.get_trials(deepcopy=False)
    }
    expected_persisted_ordinals = {
        item.ordinal
        for item in first.trials
        if item.result.outcome_kind != "operational_failure"
        and item.proposal.matched_basis_control == "none"
    }
    assert persisted_ordinals == expected_persisted_ordinals

    resumed = TruthEditingStudy(config, bank).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )

    assert resumed.trials[8].proposal.to_dict() == first.trials[0].proposal.to_dict()
    assert resumed.trials[9].proposal.to_dict() == first.trials[1].proposal.to_dict()
    assert resumed.trials[8].ordinal == 8
    assert resumed.trials[9].ordinal == 9
    assert resumed.trials[8].result.outcome_kind == "successful"
    assert resumed.trials[9].result.outcome_kind == "successful"
    assert resumed.operational_failures == 2
    assert resumed.unresolved_operational_failures == 0


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_optuna_requeues_a_replay_that_operationally_fails_again(tmp_path) -> None:
    class FailOrdinal(OfflineSyntheticEvaluator):
        def __init__(self, ordinal: int) -> None:
            self.trial_id = f"trial-{ordinal:04d}"

        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            if trial_id == self.trial_id:
                return EvaluationResult.operational_failure("worker unavailable")
            return super().evaluate(
                proposal,
                trial_id=trial_id,
                record_ids=record_ids,
                objective_names=objective_names,
            )

    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=24,
        evaluation_tiers=[
            {"name": "all", "record_limit": 2, "through_trial": 24}
        ],
    )
    bank = _direction_bank()
    journal = tmp_path / "repeated-operational-replay.json"
    first = TruthEditingStudy(config, bank).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=FailOrdinal(0),
        journal_path=journal,
        stop_after_trials=8,
    )
    second = TruthEditingStudy(config, bank).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=FailOrdinal(8),
        journal_path=journal,
        stop_after_trials=16,
    )
    final = TruthEditingStudy(config, bank).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )

    failed = first.trials[0].proposal.to_dict()
    assert second.trials[8].proposal.to_dict() == failed
    assert second.trials[8].result.outcome_kind == "operational_failure"
    assert second.unresolved_operational_failures == 2
    assert final.trials[16].proposal.to_dict() == failed
    assert final.trials[16].result.outcome_kind == "successful"
    assert final.operational_failures == 2
    assert final.unresolved_operational_failures == 0


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_optuna_defers_failure_replay_until_saved_incomplete_batch_finishes(
    tmp_path,
) -> None:
    class FailThenInterrupt(OfflineSyntheticEvaluator):
        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            if trial_id == "trial-0000":
                return EvaluationResult.operational_failure("worker unavailable")
            if trial_id == "trial-0008":
                raise KeyboardInterrupt
            return super().evaluate(
                proposal,
                trial_id=trial_id,
                record_ids=record_ids,
                objective_names=objective_names,
            )

    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=24,
        evaluation_tiers=[
            {"name": "all", "record_limit": 2, "through_trial": 24}
        ],
    )
    bank = _direction_bank()
    journal = tmp_path / "incomplete-before-replay.json"
    with pytest.raises(KeyboardInterrupt):
        TruthEditingStudy(config, bank).run(
            driver=OptunaSearchDriver(seed=17),
            evaluator=FailThenInterrupt(),
            journal_path=journal,
            stop_after_trials=16,
        )
    saved = json.loads(journal.read_text())
    failed = saved["batches"][0]["trials"][0]["proposal"]
    pending = saved["batches"][1]["trials"][0]["proposal"]

    resumed = TruthEditingStudy(config, bank).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )

    assert resumed.trials[8].proposal.to_dict() == pending
    assert pending == failed
    assert sum(
        item.proposal.to_dict() == failed for item in resumed.trials
    ) == 2


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_optuna_conditional_domains_stay_compatible_across_concentration_batches(
    tmp_path,
) -> None:
    """Every conditional search branch can follow every other branch.

    Optuna binds one immutable distribution to each parameter name.  A complete
    concentration run therefore exercises the public study seam across scope,
    layer, rank, region, writer, and refusal transitions without a dynamic-value-
    space exception.
    """

    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=48,
        evaluation_tiers=[
            {"name": "all", "record_limit": 2, "through_trial": 48}
        ],
        tpe_startup_trials=0,
    )
    driver = OptunaSearchDriver(seed=17)
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "conditional.json",
    )

    concentrated = [
        item.proposal
        for item in report.trials
        if item.proposal.proposal_origin == "tpe_sampled"
    ]
    assert concentrated
    assert {item.basis_scope for item in concentrated} == {
        "general",
        "domain",
        "mixed",
    }
    assert {item.truth_direction_scope for item in concentrated} == {
        "global",
        "per_layer",
    }
    assert {item.edit_arm for item in concentrated} == {
        "truth_only",
        "refusal_only",
        "joint",
    }
    assert {item.attention_enabled for item in concentrated} == {False, True}
    assert {item.mlp_enabled for item in concentrated} == {False, True}
    assert {
        item.refusal_direction_scope
        for item in concentrated
        if item.refusal_enabled
    } == {"global", "per_layer"}
    assert [item.ordinal for item in report.trials] == list(range(48))
    assert driver.observed_batch_sizes == [8] * 6


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="Optuna is optional")
def test_optuna_concentration_resume_preserves_proposals_and_batch_order(tmp_path) -> None:
    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=48,
        evaluation_tiers=[
            {"name": "all", "record_limit": 2, "through_trial": 48}
        ],
        tpe_startup_trials=0,
    )
    bank = _direction_bank()
    uninterrupted = TruthEditingStudy(config, bank).run(
        driver=OptunaSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "uninterrupted.json",
    )

    first_driver = OptunaSearchDriver(seed=17)
    first = TruthEditingStudy(config, bank).run(
        driver=first_driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "resumed.json",
        stop_after_trials=32,
    )
    assert first.completed_trials == 32
    assert first_driver.observed_batch_sizes == [8] * 4

    resumed_driver = OptunaSearchDriver(seed=17)
    resumed = TruthEditingStudy(config, bank).run(
        driver=resumed_driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "resumed.json",
    )

    assert [item.proposal.to_dict() for item in resumed.trials] == [
        item.proposal.to_dict() for item in uninterrupted.trials
    ]
    assert [item.ordinal for item in resumed.trials] == list(range(48))
    assert resumed_driver.observed_batch_sizes == [8] * 6


def test_resume_rejects_tampered_pending_proposal(tmp_path) -> None:
    config = _config(tmp_path, max_trials=4, evaluation_tiers=[
        {"name": "all", "record_limit": 2, "through_trial": 4}
    ])
    journal = tmp_path / "journal.json"
    with pytest.raises(KeyboardInterrupt):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=_InterruptingEvaluator(),
            journal_path=journal,
        )
    raw = json.loads(journal.read_text())
    raw["batches"][-1]["trials"][0]["proposal"]["strength"] = 1.234
    journal.write_text(json.dumps(raw))
    with pytest.raises(StudyError, match="content identity"):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=OfflineSyntheticEvaluator(),
            journal_path=journal,
        )


def test_resume_journal_schema_is_fail_closed_even_if_rehashed(tmp_path) -> None:
    config = _config(tmp_path, max_trials=2, evaluation_tiers=[
        {"name": "all", "record_limit": 2, "through_trial": 2}
    ])
    journal = tmp_path / "journal.json"
    TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=17),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )
    raw = json.loads(journal.read_text())
    raw["unexpected"] = True
    raw.pop("journal_sha256")
    raw["journal_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    journal.write_text(json.dumps(raw))
    with pytest.raises(StudyError, match="fields"):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=OfflineDeterministicSearchDriver(seed=17),
            evaluator=OfflineSyntheticEvaluator(),
            journal_path=journal,
        )


def test_offline_cli_writes_journal_and_report(tmp_path) -> None:
    config = _config(tmp_path, max_trials=4, evaluation_tiers=[
        {"name": "all", "record_limit": 2, "through_trial": 4}
    ])
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(config.to_dict()))
    manifest_path = tmp_path / "directions.json"
    manifest_path.write_text(json.dumps(_direction_bank().to_dict()))
    journal_path = tmp_path / "journal.json"
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_truth_editing_study.py",
            "--config", str(config_path),
            "--direction-manifest", str(manifest_path),
            "--journal", str(journal_path),
            "--report", str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(report_path.read_text())["completed_trials"] == 4
    assert json.loads(journal_path.read_text())["journal_sha256"]
