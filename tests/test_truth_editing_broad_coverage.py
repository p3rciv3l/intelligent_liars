from __future__ import annotations

import json
from dataclasses import replace

from intelligent_liars.truth_editing_study import (
    EvaluationResult,
    OfflineDeterministicSearchDriver,
    OfflineSyntheticEvaluator,
    StudyError,
    TruthEditingStudy,
    OptunaSearchDriver,
)
import pytest

from test_truth_editing_study import _config, _direction_bank


def test_broad_stage_records_real_writer_and_refusal_configurations(tmp_path) -> None:
    """Coverage reflects effective edits, not legacy/coarse labels."""

    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=24,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 24}
        ],
    )
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=config.sampler_seed),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "journal.json",
    )

    assert report.coverage_complete is True
    assert set(report.coverage.writer_configurations) == {
        "disabled",
        "attention",
        "mlp",
        "both",
    }
    assert set(report.coverage.refusal_settings) == {
        "disabled",
        "global",
        "per_layer",
    }
    assert set(report.coverage.refusal_writer_policies) == {
        "attention",
        "mlp",
        "both",
    }
    assert set(report.coverage.kernel_center_regions) == {
        "early",
        "middle",
        "late",
    }
    assert set(report.coverage.kernel_half_width_modes) == {
        "disabled",
        "local",
        "broad",
    }
    assert set(report.coverage.kernel_shapes) == {"flat", "tapered"}
    assert set(report.coverage.refusal_strength_regions) == {
        "disabled",
        "projection",
        "reflection",
    }
    assert set(report.coverage.active_edit_arms) == {
        "truth_only",
        "refusal_only",
        "joint",
        "orthogonal_control",
    }

    refusal_only = [
        trial for trial in report.trials
        if trial.proposal.edit_arm == "refusal_only"
    ]
    assert refusal_only
    assert all(
        not trial.proposal.attention_enabled and not trial.proposal.mlp_enabled
        for trial in refusal_only
    )

    orthogonal_controls = [
        trial for trial in report.trials
        if trial.proposal.matched_basis_control == "orthogonal"
    ]
    assert orthogonal_controls
    assert all(
        trial.proposal.backend_type == "persistent_weight"
        and trial.proposal.edit_arm == "truth_only"
        and not trial.proposal.refusal_enabled
        for trial in orthogonal_controls
    )


def test_broad_orthogonal_arm_uses_persistent_matched_control_evaluation(tmp_path) -> None:
    class RecordingProductionLikeEvaluator:
        def __init__(self) -> None:
            self.control_trials: list[tuple[str, str]] = []
            self._offline = OfflineSyntheticEvaluator()

        @property
        def identity(self):
            return {"adapter": "recording_production_like"}

        def evaluate(
            self, proposal, *, trial_id, record_ids, objective_names,
            control_kind=None, finalization_execution_identity_sha256=None,
        ):
            if control_kind is not None:
                self.control_trials.append((trial_id, control_kind))
                assert proposal.backend_type == "persistent_weight"
                assert proposal.edit_arm == "truth_only"
                assert proposal.refusal_enabled is False
                assert len(finalization_execution_identity_sha256) == 64
            return self._offline.evaluate(
                proposal,
                trial_id=trial_id,
                record_ids=record_ids,
                objective_names=objective_names,
            )

    evaluator = RecordingProductionLikeEvaluator()
    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=8,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 8}
        ],
    )
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=config.sampler_seed),
        evaluator=evaluator,
        journal_path=tmp_path / "journal.json",
    )

    expected = [
        (trial.trial_id, "orthogonal")
        for trial in report.trials
        if trial.proposal.matched_basis_control == "orthogonal"
    ]
    assert evaluator.control_trials == expected
    assert expected


def test_failed_orthogonal_control_keeps_broad_stage_open(tmp_path) -> None:
    class FailingControl(OfflineSyntheticEvaluator):
        def evaluate_matched_basis_control(self, proposal, **kwargs):
            del proposal, kwargs
            return EvaluationResult.operational_failure("control worker lost")

    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=8,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 8}
        ],
    )
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=config.sampler_seed),
        evaluator=FailingControl(),
        journal_path=tmp_path / "failed-control.json",
    )

    assert "orthogonal_control" not in report.coverage.active_edit_arms
    assert report.coverage_complete is False
    assert report.selection_ready is False


def test_orthogonal_control_rejects_refusal_or_inert_truth_edit(tmp_path) -> None:
    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=8,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 8}
        ],
    )
    study = TruthEditingStudy(config, _direction_bank())
    report = study.run(
        driver=OfflineDeterministicSearchDriver(seed=config.sampler_seed),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "valid-control.json",
    )
    control = next(
        trial.proposal for trial in report.trials
        if trial.proposal.matched_basis_control == "orthogonal"
    )

    with pytest.raises(StudyError, match="active persistent truth-only"):
        study._validate_proposal(replace(
            control,
            edit_arm="joint",
            refusal_enabled=True,
            refusal_strength=1.0,
            refusal_source_layer=control.source_layer,
        ))
    with pytest.raises(StudyError, match="active persistent truth-only"):
        study._validate_proposal(replace(
            control,
            attention_enabled=False,
            attention_edge_strength=0.0,
            attention_peak_strength=0.0,
            mlp_enabled=False,
            mlp_edge_strength=0.0,
            mlp_peak_strength=0.0,
            writer_policy="both",
        ))


def test_broad_stage_is_deterministic_in_eight_trial_batches_across_resume(tmp_path) -> None:
    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=24,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 24}
        ],
    )
    bank = _direction_bank()
    uninterrupted_path = tmp_path / "uninterrupted.json"
    resumed_path = tmp_path / "resumed.json"

    uninterrupted_driver = OfflineDeterministicSearchDriver(seed=config.sampler_seed)
    uninterrupted = TruthEditingStudy(config, bank).run(
        driver=uninterrupted_driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=uninterrupted_path,
    )

    first_driver = OfflineDeterministicSearchDriver(seed=config.sampler_seed)
    TruthEditingStudy(config, bank).run(
        driver=first_driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=resumed_path,
        stop_after_trials=8,
    )
    resumed_driver = OfflineDeterministicSearchDriver(seed=config.sampler_seed)
    resumed = TruthEditingStudy(config, bank).run(
        driver=resumed_driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=resumed_path,
    )

    assert uninterrupted_driver.observed_batch_sizes == [8, 8, 8]
    assert resumed_driver.observed_batch_sizes == [8, 8, 8]
    assert [item.proposal.to_dict() for item in resumed.trials] == [
        item.proposal.to_dict() for item in uninterrupted.trials
    ]
    assert json.loads(resumed_path.read_text())["batches"] == json.loads(
        uninterrupted_path.read_text()
    )["batches"]


def test_operational_failure_keeps_broad_stage_open_for_missing_axis(tmp_path) -> None:
    from intelligent_liars.truth_editing_study import EvaluationResult

    class LoseFirstPerLayerRefusal(OfflineSyntheticEvaluator):
        def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
            if proposal.refusal_enabled and proposal.refusal_direction_scope == "per_layer":
                return EvaluationResult.operational_failure("synthetic worker loss")
            return super().evaluate(
                proposal,
                trial_id=trial_id,
                record_ids=record_ids,
                objective_names=objective_names,
            )

    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=8,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 8}
        ],
    )
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=OfflineDeterministicSearchDriver(seed=config.sampler_seed),
        evaluator=LoseFirstPerLayerRefusal(),
        journal_path=tmp_path / "failed.json",
    )

    assert "per_layer" not in report.coverage.refusal_settings
    assert report.coverage_complete is False
    assert report.selection_ready is False


def test_optuna_concentration_starts_only_on_batch_after_broad_stage(tmp_path) -> None:
    class GateRecordingDriver(OfflineDeterministicSearchDriver):
        def suggest(self, request):
            proposal = super().suggest(request)
            if OptunaSearchDriver._coverage_complete(request):
                proposal = replace(proposal, proposal_origin="tpe_sampled")
                self._reserved[-1] = proposal
            return proposal

    config = _config(
        tmp_path,
        batch_size=8,
        max_trials=24,
        evaluation_tiers=[
            {"name": "discovery", "record_limit": 2, "through_trial": 24}
        ],
    )
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=GateRecordingDriver(seed=config.sampler_seed),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=tmp_path / "optuna.json",
    )

    origins = [item.proposal.proposal_origin for item in report.trials]
    assert "tpe_sampled" in origins
    first_concentrated = origins.index("tpe_sampled")
    assert first_concentrated % 8 == 0
    assert set(origins[:first_concentrated]) == {"coverage_anchor"}
    assert report.coverage_complete is True
