from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
import json

import pytest

from intelligent_liars.truth_editing_evaluator import EvaluatorConfig
from intelligent_liars.truth_editing_base_known import BaseKnownQualification
from intelligent_liars.truth_editing_scenario_view import TruthEditingScenarioView
from intelligent_liars.truth_editing_study import StudyError, load_truth_editing_study_config
from intelligent_liars.truth_editing_structured_qualification import (
    StructuredSemanticQualification,
)
from intelligent_liars.truth_editing_structured_semantic import StructuredSemanticView
from intelligent_liars.truth_editing_preservation_thresholds import (
    PreservationThresholdCalibration,
)
from intelligent_liars.truth_editing_production import (
    ProductionRunConfig,
    open_production_run,
)


def _builder_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_truth_editing_production_configs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_truth_editing_production_configs", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_config_publish_is_idempotent_for_identical_bytes(tmp_path: Path) -> None:
    builder = _builder_module()
    destination = tmp_path / "config.json"
    value = {"format": "example_v1", "nested": {"value": 1}}

    builder.write_json_no_clobber(destination, value)
    first_bytes = destination.read_bytes()
    builder.write_json_no_clobber(destination, value)

    assert destination.read_bytes() == first_bytes
    assert first_bytes == (
        b'{\n  "format": "example_v1",\n  "nested": {\n'
        b'    "value": 1\n  }\n}\n'
    )


def test_fixed_config_publish_rejects_differing_existing_bytes(tmp_path: Path) -> None:
    builder = _builder_module()
    destination = tmp_path / "config.json"
    destination.write_bytes(b'{"existing":true}\n')

    with pytest.raises(RuntimeError, match="refusing to replace differing fixed output"):
        builder.write_json_no_clobber(destination, {"replacement": True})

    assert destination.read_bytes() == b'{"existing":true}\n'


def test_versioned_bundle_preflights_every_output_before_publication(
    tmp_path: Path,
) -> None:
    builder = _builder_module()
    study = tmp_path / "truth_editing_study_v3.json"
    evaluator = tmp_path / "truth_editing_evaluator_v4.json"
    production = tmp_path / "truth_editing_production_v4.json"
    evaluator.write_text('{"preserved":"old"}\n')

    with pytest.raises(RuntimeError, match="refusing to replace differing fixed output"):
        builder.write_json_bundle_no_clobber(
            (
                (study, {"kind": "study"}),
                (evaluator, {"kind": "evaluator"}),
                (production, {"kind": "production"}),
            )
        )

    assert not study.exists()
    assert evaluator.read_text() == '{"preserved":"old"}\n'
    assert not production.exists()


def test_versioned_output_references_are_portable_from_production_config(
    tmp_path: Path,
) -> None:
    builder = _builder_module()
    repository = tmp_path / "repo"
    production = repository / "configs/releases/truth_editing_production_v4.json"
    study = repository / "configs/truth_editing_study_v3.json"
    evaluator = repository / "configs/truth_editing_evaluator_v4.json"
    production.parent.mkdir(parents=True)

    assert builder.portable_config_reference(
        study, from_config=production, repository_root=repository
    ) == "../truth_editing_study_v3.json"
    assert builder.portable_config_reference(
        evaluator, from_config=production, repository_root=repository
    ) == "../truth_editing_evaluator_v4.json"


def test_builder_cli_exposes_explicit_versioned_output_paths() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/build_truth_editing_production_configs.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--study-output" in completed.stdout
    assert "--evaluator-output" in completed.stdout
    assert "--production-output" in completed.stdout
    assert "--preservation-runtime-packet-root" in completed.stdout


def test_dataset_manifest_identity_binds_exact_file_bytes(tmp_path: Path) -> None:
    builder = _builder_module()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"format":"fixture"}\n')

    assert builder.dataset_manifest_file_sha256(manifest) == (
        "650e906a4519291301303a2f1d745dca9b467a9503d20553d3b92d4e6bb2866a"
    )


def test_preservation_calibration_reference_is_portable_and_hash_bound(
    tmp_path: Path,
) -> None:
    builder = _builder_module()
    repo = tmp_path / "repo"
    config_dir = repo / "configs"
    calibration = repo / "artifacts/truth-editing/preservation/calibration.json"
    config_dir.mkdir(parents=True)
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}\n")

    assert builder.production_calibration_reference(
        calibration,
        calibration_sha256="a" * 64,
        repository_root=repo,
        config_directory=config_dir,
    ) == {
        "preservation_threshold_calibration": (
            "../artifacts/truth-editing/preservation/calibration.json"
        ),
        "preservation_threshold_calibration_sha256": "a" * 64,
    }


def test_preservation_calibration_reference_rejects_repo_escape(tmp_path: Path) -> None:
    builder = _builder_module()
    repo = tmp_path / "repo"
    config_dir = repo / "configs"
    outside = tmp_path / "outside.json"
    config_dir.mkdir(parents=True)
    outside.write_text("{}\n")

    with pytest.raises(RuntimeError, match="outside repository"):
        builder.production_calibration_reference(
            outside,
            calibration_sha256="a" * 64,
            repository_root=repo,
            config_directory=config_dir,
        )


def test_evidence_adaptive_tiers_consume_every_required_structured_scenario() -> None:
    builder = _builder_module()

    assert builder.plan_evidence_adaptive_tiers(
        structured_scenario_count=12,
        broad_qa_scenario_count=128,
    ) == (
        ("discovery", 4, 8, 80, "trial"),
        ("expanded", 8, 32, 200, "promoted"),
        ("finalist", 12, 128, 800, "finalist"),
    )


@pytest.mark.parametrize(
    ("structured_count", "qa_count", "match"),
    [
        (11, 128, "at least 12 strictly qualified structured scenarios"),
        (12, 127, "at least 128 complete base-known broad-QA scenarios"),
    ],
)
def test_evidence_adaptive_tiers_fail_closed_below_production_minimums(
    structured_count: int,
    qa_count: int,
    match: str,
) -> None:
    builder = _builder_module()

    with pytest.raises(RuntimeError, match=match):
        builder.plan_evidence_adaptive_tiers(
            structured_scenario_count=structured_count,
            broad_qa_scenario_count=qa_count,
        )


def test_qualified_qa_groups_consume_public_scenario_view_mappings() -> None:
    builder = _builder_module()
    records = [
        {
            "record_id": f"scenario-1.role-{index}",
            "scenario_id": "scenario-1",
            "source_record_id": "source-1",
        }
        for index in range(3)
    ] + [
        {
            "record_id": "scenario-2.role-0",
            "scenario_id": "scenario-2",
            "source_record_id": "unqualified-source",
        }
    ]

    assert builder.qualified_qa_groups(
        records,
        qualified_source_record_ids={"source-1"},
    ) == ((
        "scenario-1.role-0",
        "scenario-1.role-1",
        "scenario-1.role-2",
    ),)


def test_checked_in_study_and_evaluator_configs_are_strictly_compatible() -> None:
    root = Path(__file__).resolve().parents[1]
    study = load_truth_editing_study_config(
        root / "configs" / "truth_editing_study_v2.json"
    )
    evaluator_raw = json.loads(
        (root / "configs" / "truth_editing_evaluator_v2.json").read_text()
    )
    evaluator = EvaluatorConfig.from_mapping(evaluator_raw)

    assert evaluator.to_mapping() == evaluator_raw
    assert study.dataset_manifest_sha256 == evaluator.dataset_manifest_sha256
    assert tuple(tier.record_limit for tier in study.evaluation_tiers) == (
        48,
        144,
        456,
    )
    assert tuple(tier.record_ids for tier in evaluator.tiers) == tuple(
        study.validation_record_ids[: tier.record_limit]
        for tier in study.evaluation_tiers
    )
    assert evaluator.judge_config_sha256 == (
        "1b499bf7fdb0321a62afccac49ac2af90a25ae102ed17ed1cd12abca3c03b07c"
    )
    assert evaluator.rubric_sha256 == (
        "93266dd26ff56c96b86e8fd4ee5646a61d9ddf80b0393c19bfad15efb3f155d6"
    )


def test_checked_in_study_uses_only_strictly_qualified_records() -> None:
    root = Path(__file__).resolve().parents[1]
    study = load_truth_editing_study_config(
        root / "configs" / "truth_editing_study_v2.json"
    )
    base_known = BaseKnownQualification.open(
        root / "artifacts" / "truth-editing" / "base-known"
    )
    qa_view = TruthEditingScenarioView.open(
        root
        / "datasets"
        / "truth_editing"
        / "views"
        / "v2_optimization_validation_qualified_v1",
        source_dataset=root / "datasets" / "truth_editing" / "v2",
        base_known_qualification=root / "artifacts" / "truth-editing" / "base-known",
    )
    structured_root = (
        root / "datasets" / "truth_editing" / "views" / "structured_semantic_v1"
    )
    structured_source = root / "corpora" / "tinylora_deception_action_v1" / "step5_v1"
    structured_view = StructuredSemanticView.open(
        structured_root,
        source_root=structured_source,
    )
    structured_qualification = StructuredSemanticQualification.open(
        root / "artifacts" / "truth-editing" / "structured-base-known",
        structured_root,
        structured_source,
    )

    qualified_sources = set(base_known.qualified_record_ids)
    qualified_qa_ids = {
        record["record_id"]
        for record in qa_view.records
        if record["source_record_id"] in qualified_sources
    }
    complete_scenarios = {
        row.scenario_id
        for row in structured_qualification.scenarios
        if row.all_required_known
    }
    assert len(complete_scenarios) == 12
    qualified_structured_ids = {
        signal["signal_id"]
        for scenario in structured_view.scenarios
        if scenario["scenario_id"] in complete_scenarios
        for signal in scenario["signals"]
    }
    configured = set(study.validation_record_ids)

    assert configured <= qualified_qa_ids | qualified_structured_ids
    assert qualified_structured_ids <= configured


def test_checked_in_promoted_structured_view_matches_verified_qualification() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "corpora" / "tinylora_deception_action_v1" / "step5_v1"
    pending_root = (
        root / "datasets" / "truth_editing" / "views" / "structured_semantic_v1"
    )
    qualified_root = (
        root
        / "datasets"
        / "truth_editing"
        / "views"
        / "structured_semantic_qualified_v1"
    )
    qualification_root = root / "artifacts" / "truth-editing" / "structured-base-known"

    qualified = StructuredSemanticView.open(
        qualified_root,
        source_root=source,
        qualification_root=qualification_root,
    )
    qualification = StructuredSemanticQualification.open(
        qualification_root,
        pending_root,
        source,
    )
    expected = {
        row.scenario_id for row in qualification.scenarios if row.all_required_known
    }

    assert len(expected) == 12
    assert set(qualified.manifest["scientific_validation_scenario_ids"]) == expected
    assert qualified.manifest["pending_base_known_validation_scenario_ids"] == []


def test_new_production_config_binds_qualified_view_and_versioned_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (root / "configs" / "truth_editing_production_v2.json").read_text()
    )

    assert raw["study_config"] == "truth_editing_study_v2.json"
    assert raw["evaluator_config"] == "truth_editing_evaluator_v2.json"
    assert raw["scenario_view"] == (
        "../datasets/truth_editing/views/v2_optimization_validation_qualified_v1"
    )


def test_racing_fixed_config_publications_never_clobber_the_winner(
    tmp_path: Path,
) -> None:
    builder = _builder_module()
    destination = tmp_path / "config.json"
    values = ({"winner": "alpha"}, {"winner": "beta"})
    barrier = threading.Barrier(len(values))

    def publish(value: dict[str, str]) -> RuntimeError | None:
        barrier.wait()
        try:
            builder.write_json_no_clobber(destination, value)
        except RuntimeError as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=len(values)) as executor:
        errors = list(executor.map(publish, values))

    assert sum(error is None for error in errors) == 1
    assert sum(
        error is not None
        and "refusing to replace differing fixed output" in str(error)
        for error in errors
    ) == 1
    assert destination.read_bytes() in {
        b'{\n  "winner": "alpha"\n}\n',
        b'{\n  "winner": "beta"\n}\n',
    }
    assert list(tmp_path.glob(".*.tmp")) == []


def test_builder_reads_replay_verified_preservation_ceilings() -> None:
    builder = _builder_module()
    calibration = PreservationThresholdCalibration(
        payload={
            "self_sha256": "a" * 64,
            "source_receipts": [],
            "tiers": [
                {
                    "tier": "trial",
                    "strata": [
                        {"stratum": "text", "ceiling": 0.001},
                        {"stratum": "vision", "ceiling": 0.002},
                        {"stratum": "recorded_computer_use", "ceiling": 0.003},
                    ],
                }
            ],
        }
    )

    assert builder.calibrated_preservation_ceilings(calibration, "trial") == {
        "text": 0.001,
        "vision": 0.002,
        "recorded_computer_use": 0.003,
    }


def test_r10_content_qualified_production_bundle_opens_strictly() -> None:
    root = Path(__file__).resolve().parents[1]
    production_path = (
        root
        / "configs"
        / "truth_editing_production_v4_r10_17b1e9cb_c1f373f8.json"
    )
    config = ProductionRunConfig.open(production_path)
    study = load_truth_editing_study_config(config.study_config)
    evaluator = EvaluatorConfig.from_mapping(
        json.loads(config.evaluator_config.read_text())
    )
    calibration = PreservationThresholdCalibration.open(
        config.preservation_threshold_calibration
    )
    run = open_production_run(production_path)

    assert config.study_config.name == "truth_editing_study_v3_r10_c1f373f8.json"
    assert config.evaluator_config.name == (
        "truth_editing_evaluator_v4_r10_c1f373f8.json"
    )
    assert calibration.self_sha256 == (
        "17b1e9cb18aa0b1bf2cec34ffe4994bbc3ac7d16bd463a6952dd9b3bbb7293cf"
    )
    assert study.dataset_manifest_sha256 == evaluator.dataset_manifest_sha256
    assert run.planned_study_identity_sha256 == (
        "5a45216e10c9059119231bccc1543c61863a7602ab4c75b535b793b3e90f7dda"
    )


def test_superseded_adaptive_r10_bundle_fails_closed_without_reinterpretation() -> None:
    root = Path(__file__).resolve().parents[1]
    superseded = ProductionRunConfig.open(
        root
        / "configs/truth_editing_production_v5_adaptive_r10_17b1e9cb_c1f373f8.json"
    )
    with pytest.raises(StudyError, match="broad coverage fields differ"):
        load_truth_editing_study_config(superseded.study_config)

    builder_source = (
        root / "scripts/build_truth_editing_production_configs.py"
    ).read_text()
    assert "truth_editing_study_v5_adaptive_r10_c1f373f8.json" in builder_source
    assert "truth_editing_production_v6_adaptive_r10_17b1e9cb_c1f373f8.json" in (
        builder_source
    )


def test_generated_adaptive_v5_bundle_strict_opens_without_broad_trial_barrier() -> None:
    root = Path(__file__).resolve().parents[1]
    production_path = (
        root
        / "configs"
        / "truth_editing_production_v6_adaptive_r10_17b1e9cb_c1f373f8.json"
    )
    config = ProductionRunConfig.open(production_path)
    study = load_truth_editing_study_config(config.study_config)
    evaluator = EvaluatorConfig.from_mapping(
        json.loads(config.evaluator_config.read_text())
    )
    run = open_production_run(production_path)

    assert config.study_config.name == (
        "truth_editing_study_v5_adaptive_r10_c1f373f8.json"
    )
    assert config.evaluator_config.name == (
        "truth_editing_evaluator_v6_adaptive_r10_c1f373f8.json"
    )
    assert study.batch_size == 8
    assert study.max_trials == 800
    assert study.search_policy is not None
    assert study.search_policy.minimum_trials == 200
    assert study.search_policy.maximum_trials == 800
    assert study.search_policy.broad_coverage.required_before_concentration is True
    assert "through_trial" not in study.search_policy.broad_coverage.to_dict()
    assert study.dataset_manifest_sha256 == evaluator.dataset_manifest_sha256
    assert run.planned_study_identity_sha256 == (
        "e0bc523ff4f94f29b950681d6b43fef203ab80cd9be97dcd9ec85631875429e2"
    )
