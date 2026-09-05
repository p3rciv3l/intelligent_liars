from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from intelligent_liars.truth_editing_live_judge import FROZEN_JUDGE_CONFIG_SHA256
from intelligent_liars import truth_editing_record_completion as record_completion
from intelligent_liars.truth_editing_vast_fleet import FleetConfig
from intelligent_liars.truth_editing_production import (
    ProductionRunConfig,
    configured_search_driver,
    open_production_run,
)
from intelligent_liars.truth_editing_rescore import materialize_rescore_generation_v1
from intelligent_liars.truth_editing_study import load_truth_editing_study_config
from intelligent_liars.truth_editing_wandb_monitoring import MonitoredSearchDriver
from intelligent_liars.truth_editing_vast_production import (
    ProductionVastConfig,
    build_production_bundle,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_truth_editing_production_vast_job.py"
)
SPEC = importlib.util.spec_from_file_location("build_truth_editing_production_vast_job", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PRODUCTION_CONFIG = "configs/truth_editing_production_v4_fixture_deadbeef.json"
CANONICAL_PRODUCTION_CONFIG = (
    "configs/truth_editing_production_v6_adaptive_r10_17b1e9cb_c1f373f8.json"
)
FLEET_CONFIG = "configs/truth_editing_vast_fleet_adaptive_fixture.json"


def _open_fixture(_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        verified_model_sha256=MODULE.DEFAULT_MODEL_CONTENT_SHA256,
        verified_snapshot_manifest_sha256=MODULE.DEFAULT_SNAPSHOT_MANIFEST_SHA256,
    )


def _repo(tmp_path: Path, *, adaptive: bool = False) -> Path:
    repo = tmp_path / "repo"
    for relative in (
        *MODULE.PRODUCTION_SOURCE_CLOSURE,
        "scripts/bootstrap_truth_editing_production_worker.sh",
        "scripts/bootstrap_truth_editing_prerequisite_worker.sh",
        "scripts/hydrate_truth_editing_production_model.sh",
        "scripts/verify_truth_editing_production_model.py",
        "scripts/run_truth_editing_production_worker.py",
        "scripts/run_truth_editing_cuda_fleet_controller.py",
        "scripts/run_truth_editing_cuda_fleet_worker.py",
        "scripts/run_truth_editing_adaptive_finalization.py",
        "scripts/run_truth_editing_causal_activation_controls.py",
        "scripts/publish_truth_editing_production_checkpoint.py",
        "scripts/build_tinylora_model_cache.py",
        "docker/truth-editing/Dockerfile",
        "docker/truth-editing/requirements.in",
        "docker/truth-editing/requirements.lock",
        "docker/truth-editing/runtime-manifest.json",
        "docker/truth-editing/validate_runtime.py",
        "configs/truth_editing_evaluator_v3.json",
        "configs/model_registry_v1.json",
        "configs/truth_editing_direction_bank_qualified_v1.json",
        "configs/truth_editing_refusal_directions_v1.json",
        "configs/truth_editing_refusal_prompt_manifest_v1.json",
        "datasets/truth_editing/v2/manifest.json",
        "datasets/truth_editing/views/v2_optimization_validation_qualified_v1/manifest.json",
        "datasets/truth_editing/views/structured_semantic_qualified_v1/manifest.json",
        "datasets/truth_editing/views/structured_semantic_qualified_v1/scenarios.jsonl",
        "datasets/truth_editing/views/structured_semantic_v1/manifest.json",
        "datasets/truth_editing/views/structured_semantic_v1/scenarios.jsonl",
        "corpora/tinylora_deception_action_v1/step5_v1/records.jsonl",
        "artifacts/truth-editing/base-known/manifest.json",
        "artifacts/truth-editing/structured-base-known/manifest.json",
        "artifacts/truth-editing/refusal-directions/direction_bank.json",
        "artifacts/truth-editing/preservation-runtime/manifest.json",
        "artifacts/truth-editing/preservation-thresholds/calibration.json",
        "artifacts/probes/direction.npy",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "fixture\n")
    if adaptive:
        refresh_helper = repo / "scripts/configure_truth_editing_aws_refresh.py"
        refresh_helper.write_text("# adaptive fixture\n")
    (repo / "configs/truth_editing_direction_bank_qualified_v1.json").write_text(
        json.dumps({"artifact_path": "artifacts/probes/direction.npy"})
    )
    (repo / "configs/model_registry_v1.json").write_text(
        json.dumps(
            {
                "format": "intelligent_liars_model_registry_config_v1",
                "schema_version": 1,
                "registry": {
                    "bucket": "fixture-model-registry",
                    "base_prefix": "model-registry/v1",
                },
                "base_model_cache": {},
            }
        )
    )
    (repo / "datasets/truth_editing/views/structured_semantic_qualified_v1/manifest.json").write_text(
        json.dumps(
            {
                "format": "truth_editing_structured_semantic_qualified_manifest_v1",
                "source_view": {
                    "path": "../structured_semantic_v1",
                    "view_sha256": "a" * 64,
                },
            }
        )
    )
    (repo / "datasets/truth_editing/views/structured_semantic_v1/manifest.json").write_text(
        json.dumps(
            {
                "format": "truth_editing_structured_semantic_manifest_v1",
                "view_sha256": "a" * 64,
            }
        )
    )
    study = {
        "evaluation_tiers": [
            {"name": "discovery", "through_trial": 80},
            {"name": "expanded", "through_trial": 160},
            {"name": "finalist", "through_trial": 200},
        ],
        "max_trials": 200,
        "batch_size": 8,
    }
    if adaptive:
        study = {
            "evaluation_tiers": [
                {"name": "discovery", "through_trial": 80},
                {"name": "expanded", "through_trial": 200},
                {"name": "finalist", "through_trial": 800},
            ],
            "max_trials": 800,
            "batch_size": 8,
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
    (repo / "configs/truth_editing_study_v2.json").write_text(json.dumps(study))
    production = {
        "format": "truth_editing_production_config_v1",
        "study_config": "truth_editing_study_v2.json",
        "evaluator_config": "truth_editing_evaluator_v3.json",
        "direction_manifest": "truth_editing_direction_bank_qualified_v1.json",
        "refusal_direction_config": "truth_editing_refusal_directions_v1.json",
        "refusal_prompt_manifest": "truth_editing_refusal_prompt_manifest_v1.json",
        "dataset_root": "../datasets/truth_editing/v2",
        "scenario_view": "../datasets/truth_editing/views/v2_optimization_validation_qualified_v1",
        "structured_semantic_view": "../datasets/truth_editing/views/structured_semantic_qualified_v1",
        "structured_semantic_source_root": "../corpora/tinylora_deception_action_v1/step5_v1",
        "base_known_qualification": "../artifacts/truth-editing/base-known",
        "structured_base_known_qualification": "../artifacts/truth-editing/structured-base-known",
        "refusal_direction_bank": "../artifacts/truth-editing/refusal-directions/direction_bank.json",
        "refusal_artifact_root": "../artifacts/truth-editing/refusal-directions",
        "preservation_runtime_packet_root": "../artifacts/truth-editing/preservation-runtime",
        "preservation_threshold_calibration": "../artifacts/truth-editing/preservation-thresholds/calibration.json",
        "judge_budget": {
            "format": "truth_editing_production_judge_budget_config_v1",
            "all_in_maximum_spend_usd": "50",
            "non_judge_reserved_spend_usd": "45",
            "maximum_judge_spend_usd": "5",
            "per_call_reservation_usd": "0.025",
            "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        },
    }
    (repo / PRODUCTION_CONFIG).write_text(json.dumps(production))
    if adaptive:
        policy_path = repo / "configs/capacity-policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "format": "truth_editing_adaptive_capacity_policy_v1",
                    "policy_id": "fixture-adaptive-policy",
                    "execution": {
                        "minimum_trials": 200,
                        "maximum_trials": 800,
                        "batch_size": 8,
                        "search_seconds": 75600,
                        "finalization_seconds": 10800,
                    },
                    "budget": {
                        "total_usd": "50",
                        "infrastructure_usd": "45",
                        "evaluation_usd": "5",
                        "evaluation_reserve_fraction": "0.20",
                    },
                    "uncertainty": {
                        "duration_multiplier": "1.25",
                        "judge_cost_multiplier": "1.25",
                    },
                    "projection_tiers": [
                        {
                            "name": "discovery",
                            "through_trial": 80,
                            "generation_multiplier": "1",
                            "judge_multiplier": "1",
                        },
                        {
                            "name": "expanded",
                            "through_trial": 200,
                            "generation_multiplier": "3",
                            "judge_multiplier": "3",
                        },
                        {
                            "name": "concentrated",
                            "through_trial": 800,
                            "generation_multiplier": "9.5",
                            "judge_multiplier": "9.5",
                        },
                    ],
                    "maximum_measurement_age_seconds": 21600,
                }
            )
        )
        policy = json.loads(policy_path.read_text())
        policy["self_sha256"] = MODULE.hashlib.sha256(
            json.dumps(
                policy,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        policy_path.write_text(json.dumps(policy))
        fleet = MODULE.build_production_fleet_config(
            repo,
            production_config=PRODUCTION_CONFIG,
            bundle_sha256="b" * 64,
            adaptive_capacity_policy="configs/capacity-policy.json",
            production_config_opener=_open_fixture,
            study_config_identity_opener=lambda _path: "d" * 64,
        )
        (repo / FLEET_CONFIG).write_text(json.dumps(fleet))
        for relative in ("artifacts/capacity-receipt.json",):
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
    return repo


def _fixture_capacity_inputs(repo: Path) -> tuple[Path, Path]:
    return (
        repo / "configs/capacity-policy.json",
        repo / "artifacts/capacity-receipt.json",
    )


@pytest.mark.parametrize(
    ("phase", "boundary", "final_outputs"),
    [
        ("discovery", 80, False),
        ("expanded", 160, False),
        ("finalist", 200, True),
    ],
)
def test_builder_emits_concrete_phase_job_and_curated_output_contract(
    tmp_path: Path, phase: str, boundary: int, final_outputs: bool
) -> None:
    repo = _repo(tmp_path)
    raw = MODULE.build_production_job(
        repo,
        production_config=PRODUCTION_CONFIG,
        phase=phase,
        optuna_study_name="truth-editing-production-test",
        production_config_opener=_open_fixture,
    )
    parsed = ProductionVastConfig.from_mapping(raw)

    assert raw["phase"] == phase
    assert parsed.phase == phase
    assert raw["study"]["workload_command"][-2:] == ["--phase", phase]
    stream = raw["checkpoints"]["stream_command"]
    assert stream[stream.index("--phase") : stream.index("--phase") + 2] == [
        "--phase",
        phase,
    ]
    assert stream[-2:] == ["--optuna-study-name", "truth-editing-production-test"]
    assert raw["base_job"]["expected_outputs"][:7] == [
        "production-run.json",
        "study/study-journal.json",
        "study/study-journal.json.optuna.log",
        "monitoring/wandb-run.json",
        "checkpoints/latest.json",
        "model/cache-hydration-receipt.json",
        "model/model-verification-receipt.json",
    ]
    assert (
        "study/frozen/study-artifact-receipt.json"
        in raw["base_job"]["expected_outputs"]
    ) is final_outputs
    study_command = raw["study"]["workload_command"]
    assert study_command[-2:] == ["--phase", phase]
    assert MODULE.PHASE_BOUNDARIES[phase] == boundary
    paths = raw["base_job"]["bundle_paths"]
    assert paths == sorted(set(paths))
    assert PRODUCTION_CONFIG in paths
    assert raw["study"]["production_config_path"] == PRODUCTION_CONFIG
    assert raw["study"]["production_config_sha256"] == MODULE.file_sha256(
        repo / PRODUCTION_CONFIG
    )
    assert "--expected-config-sha256" in study_command
    assert "artifacts/probes/direction.npy" in paths
    assert "datasets/truth_editing/views/structured_semantic_v1/manifest.json" in paths
    assert "datasets/truth_editing/views/structured_semantic_v1/scenarios.jsonl" in paths
    assert not any(".env" in Path(path).parts for path in paths)


@pytest.mark.parametrize("failure", ["missing", "identity_drift"])
def test_bundle_collection_fails_closed_on_broken_structured_view_provenance(
    tmp_path: Path, failure: str
) -> None:
    repo = _repo(tmp_path)
    qualified_manifest_path = (
        repo
        / "datasets/truth_editing/views/structured_semantic_qualified_v1/manifest.json"
    )
    qualified = json.loads(qualified_manifest_path.read_text())
    if failure == "missing":
        qualified["source_view"]["path"] = "../missing_structured_semantic_view"
        expected = "source view manifest is unreadable"
    else:
        qualified["source_view"]["view_sha256"] = "b" * 64
        expected = "identity differs from its receipt"
    qualified_manifest_path.write_text(json.dumps(qualified))

    with pytest.raises(RuntimeError, match=expected):
        MODULE.collect_production_bundle_paths(
            repo,
            production_config=PRODUCTION_CONFIG,
            production_config_opener=_open_fixture,
        )


def test_canonical_bundle_opens_all_production_inputs_after_clean_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion_roots: list[Path] = []
    store_type = record_completion.FileSemanticRecordCompletionStore

    class TrackingCompletionStore(store_type):
        def __init__(self, root: Path | str, **kwargs: object) -> None:
            completion_roots.append(Path(root))
            super().__init__(root, **kwargs)

    monkeypatch.setattr(
        record_completion,
        "FileSemanticRecordCompletionStore",
        TrackingCompletionStore,
    )
    repo = SCRIPT.parents[1]
    raw = MODULE.build_timed_canary_job(
        repo,
        canary_config=(
            "configs/truth_editing_timed_canary_v6_adaptive_r10_bbb25ef3.json"
        ),
        command_config=(
            "configs/truth_editing_timed_canary_command_v6_adaptive_r10_v4_3ae89929.json"
        ),
    )
    config = ProductionVastConfig.from_mapping(raw)
    archive = tmp_path / "canonical-production-bundle.tar.gz"
    manifest = build_production_bundle(repo, config, archive)

    required_source_files = {
        "datasets/truth_editing/views/structured_semantic_v1/manifest.json",
        "datasets/truth_editing/views/structured_semantic_v1/scenarios.jsonl",
    }
    bundled_files = {str(row["path"]) for row in manifest["files"]}
    assert required_source_files.issubset(bundled_files)
    optimization_root = "datasets/truth_editing/v2_optimization_v1"
    assert {
        f"{optimization_root}/direction_construction_allowlist.json",
        f"{optimization_root}/manifest.json",
        f"{optimization_root}/optimization-manifest.json",
        f"{optimization_root}/policy.json",
        f"{optimization_root}/provenance.jsonl",
        f"{optimization_root}/quarantine.jsonl",
        f"{optimization_root}/source_receipts.json",
        f"{optimization_root}/train.jsonl",
        f"{optimization_root}/validation.jsonl",
    }.issubset(bundled_files)
    assert not any(path.startswith("datasets/truth_editing/v2/") for path in bundled_files)
    assert not any(Path(path).name == "test.jsonl" for path in bundled_files)

    extracted = tmp_path / "clean-extraction"
    with tarfile.open(archive, mode="r:gz") as bundle:
        bundle.extractall(extracted, filter="data")
    assert all((extracted / path).is_file() for path in required_source_files)
    production_path = extracted / CANONICAL_PRODUCTION_CONFIG
    run = open_production_run(production_path)
    assert run.planned_study_identity_sha256
    assert completion_roots == [
        ProductionRunConfig.open(production_path).judge_cache_dir
        / "semantic-record-completions"
    ]


def test_builder_fails_closed_if_phase_boundaries_or_portable_v3_paths_change(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    study_path = repo / "configs/truth_editing_study_v2.json"
    study = json.loads(study_path.read_text())
    study["evaluation_tiers"][1]["through_trial"] = 159
    study_path.write_text(json.dumps(study))
    with pytest.raises(RuntimeError, match="80/160/200"):
        MODULE.build_production_job(
            repo,
            production_config=PRODUCTION_CONFIG,
            phase="expanded",
            optuna_study_name="fixture-study",
            production_config_opener=_open_fixture,
        )

    repo = _repo(tmp_path / "absolute")
    production_path = repo / PRODUCTION_CONFIG
    production = json.loads(production_path.read_text())
    production["preservation_threshold_calibration"] = "/Users/example/calibration.json"
    production_path.write_text(json.dumps(production))
    with pytest.raises(RuntimeError, match="portable relative path"):
        MODULE.build_production_job(
            repo,
            production_config=PRODUCTION_CONFIG,
            phase="discovery",
            optuna_study_name="fixture-study",
            production_config_opener=_open_fixture,
        )


def test_builder_emits_identity_bound_persistent_eight_gpu_fleet_budget(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    raw = MODULE.build_production_fleet_config(
        repo,
        production_config=PRODUCTION_CONFIG,
        bundle_sha256="b" * 64,
        production_config_opener=_open_fixture,
    )
    parsed = FleetConfig.from_mapping(raw)

    assert parsed.worker_count == 8
    assert raw["execution_mode"] == "persistent_single_host_eight_gpu"
    assert raw["budget"]["all_in_maximum_spend_usd"] == "50"
    assert raw["budget"]["maximum_infrastructure_spend_usd"] == "45"
    assert raw["budget"]["maximum_judge_spend_usd"] == "5"
    assert raw["budget"]["included_infrastructure_costs"] == [
        "gpu_compute",
        "storage",
        "network_download",
        "network_upload",
    ]
    assert (
        raw["budget"]["production_judge_budget_config_sha256"]
        == MODULE.EXPECTED_PRODUCTION_JUDGE_BUDGET_SHA256
    )


def test_builder_emits_adaptive_24_hour_job_without_a_fixed_trial_barrier(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, adaptive=True)
    raw = MODULE.build_production_job(
        repo,
        production_config=PRODUCTION_CONFIG,
        phase="adaptive",
        fleet_config=FLEET_CONFIG,
        optuna_study_name="adaptive-production-test",
        production_config_opener=_open_fixture,
        adaptive_capacity_opener=_fixture_capacity_inputs,
    )
    parsed = ProductionVastConfig.from_mapping(raw)

    assert parsed.phase == "adaptive"
    fleet = FleetConfig.from_mapping(json.loads((repo / FLEET_CONFIG).read_text()))
    assert fleet.format == MODULE.ADAPTIVE_FLEET_FORMAT
    assert fleet.batch_size == 8
    assert fleet.adaptive_capacity_policy_path == "configs/capacity-policy.json"
    assert "phase_boundaries" not in fleet.identity
    assert raw["base_job"]["format"] == MODULE.ADAPTIVE_PRODUCTION_JOB_FORMAT
    assert raw["base_job"]["resources"]["maximum_elapsed_seconds"] == 24 * 3600
    assert raw["base_job"]["resources"]["maximum_cost_usd"] == 45.0
    workload = raw["study"]["workload_command"]
    assert workload[1] == "scripts/run_truth_editing_cuda_fleet_controller.py"
    assert "--phase" not in workload
    assert workload[workload.index("--fleet-config") + 1] == FLEET_CONFIG
    assert workload[workload.index("--capacity-policy") + 1] == (
        "configs/capacity-policy.json"
    )
    assert workload[workload.index("--capacity-receipt") + 1] == (
        "artifacts/capacity-receipt.json"
    )
    assert workload[workload.index("--output-root") + 1] == "/workspace/outputs"
    assert workload[workload.index("--checkpoint-publication-root") + 1] == (
        "/workspace/outputs/checkpoints"
    )
    assert workload[workload.index("--model-registry-config") + 1] == (
        "configs/model_registry_v1.json"
    )
    assert workload[workload.index("--offhost-key-prefix") + 1] == (
        "model-registry/v1/truth-editing/adaptive-main-r11"
    )
    assert workload[workload.index("--final-model-slug") + 1] == (
        "qwen3-vl-8b-truth-edited"
    )
    stream = raw["checkpoints"]["stream_command"]
    assert "--adaptive" in stream
    assert "--phase" not in stream
    assert "--study-config-sha256" in stream
    assert "checkpoints/adaptive-latest.json" in raw["base_job"]["expected_outputs"]
    assert "checkpoints/latest.json" not in raw["base_job"]["expected_outputs"]
    assert "production-run.json" not in raw["base_job"]["expected_outputs"]
    assert FLEET_CONFIG in raw["base_job"]["bundle_paths"]
    assert (
        "scripts/run_truth_editing_causal_activation_controls.py"
        in raw["base_job"]["bundle_paths"]
    )
    assert (
        "scripts/configure_truth_editing_aws_refresh.py"
        in raw["base_job"]["bundle_paths"]
    )
    assert "configs/capacity-policy.json" in raw["base_job"]["bundle_paths"]
    assert "artifacts/capacity-receipt.json" in raw["base_job"]["bundle_paths"]
    assert "configs/model_registry_v1.json" in raw["base_job"]["bundle_paths"]
    assert (
        "monitoring/rolling-capacity-receipt.json"
        in raw["base_job"]["expected_outputs"]
    )
    assert (
        "finalization/adaptive-finalization-receipt.json"
        in raw["base_job"]["expected_outputs"]
    )
    assert (
        "finalization/checkpoint-publication/publication-receipt.json"
        in raw["base_job"]["expected_outputs"]
    )
    assert (
        "finalization/final-model-publication-receipt.json"
        in raw["base_job"]["expected_outputs"]
    )
    assert not any(
        path.endswith((".safetensors", ".bin"))
        for path in raw["base_job"]["expected_outputs"]
    )
    assert "study/frozen/study-artifact-receipt.json" in raw["base_job"]["expected_outputs"]


def test_adaptive_release_writers_no_clobber_and_strict_reopen(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, adaptive=True)
    fleet_output = repo / "configs/generated-adaptive-fleet.json"
    fleet = MODULE.write_adaptive_fleet_config(
        repo,
        production_config=PRODUCTION_CONFIG,
        capacity_policy="configs/capacity-policy.json",
        output=fleet_output,
        production_config_opener=_open_fixture,
        study_config_identity_opener=lambda _path: "d" * 64,
    )
    assert fleet == json.loads(fleet_output.read_text())
    assert FleetConfig.from_mapping(fleet).format == MODULE.ADAPTIVE_FLEET_FORMAT
    assert "bundle_sha256" not in fleet

    # An identical invocation is idempotent, but an existing different object
    # is immutable and cannot be replaced.
    MODULE.write_adaptive_fleet_config(
        repo,
        production_config=PRODUCTION_CONFIG,
        capacity_policy="configs/capacity-policy.json",
        output=fleet_output,
        production_config_opener=_open_fixture,
        study_config_identity_opener=lambda _path: "d" * 64,
    )
    fleet_output.write_text('{"different":true}\n')
    with pytest.raises(RuntimeError, match="refusing to replace differing"):
        MODULE.write_adaptive_fleet_config(
            repo,
            production_config=PRODUCTION_CONFIG,
            capacity_policy="configs/capacity-policy.json",
            output=fleet_output,
            production_config_opener=_open_fixture,
            study_config_identity_opener=lambda _path: "d" * 64,
        )

    # Restore the identity-bound fixture fleet so the adaptive job writer can
    # prove its own write/reopen boundary independently.
    (repo / FLEET_CONFIG).write_text(json.dumps(fleet))
    job_output = repo / "configs/generated-adaptive-vast-job.json"
    job = MODULE.write_adaptive_production_job(
        repo,
        production_config=PRODUCTION_CONFIG,
        fleet_config=FLEET_CONFIG,
        capacity_policy="configs/capacity-policy.json",
        capacity_receipt="artifacts/capacity-receipt.json",
        output=job_output,
        optuna_study_name="adaptive-release-writer-test",
        production_config_opener=_open_fixture,
        adaptive_capacity_opener=_fixture_capacity_inputs,
    )
    assert job == json.loads(job_output.read_text())
    assert ProductionVastConfig.from_mapping(job).phase == "adaptive"


def test_cli_help_and_adaptive_release_modes_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        MODULE.main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--build-adaptive-fleet" in help_text
    assert "--adaptive-capacity-policy" in help_text
    assert "--adaptive-capacity-receipt" in help_text

    calls: list[tuple[str, dict[str, object]]] = []

    def record_fleet(_repo: Path, **kwargs: object) -> dict[str, object]:
        calls.append(("fleet", kwargs))
        return {"kind": "fleet"}

    def record_job(_repo: Path, **kwargs: object) -> dict[str, object]:
        calls.append(("job", kwargs))
        return {"kind": "job"}

    monkeypatch.setattr(MODULE, "write_adaptive_fleet_config", record_fleet)
    monkeypatch.setattr(MODULE, "write_adaptive_production_job", record_job)
    output = tmp_path / "output.json"
    assert MODULE.main(
        [
            "--repo",
            str(tmp_path),
            "--production-config",
            "configs/production.json",
            "--build-adaptive-fleet",
            "--output",
            str(output),
        ]
    ) == 0
    assert calls[-1] == (
        "fleet",
        {
            "production_config": "configs/production.json",
            "capacity_policy": MODULE.CANONICAL_ADAPTIVE_CAPACITY_POLICY,
            "output": output,
        },
    )

    assert MODULE.main(
        [
            "--repo",
            str(tmp_path),
            "--production-config",
            "configs/production.json",
            "--fleet-config",
            "configs/fleet.json",
            "--phase",
            "adaptive",
            "--output",
            str(output),
        ]
    ) == 0
    assert calls[-1][0] == "job"
    assert calls[-1][1]["fleet_config"] == "configs/fleet.json"
    assert (
        calls[-1][1]["capacity_receipt"]
        == MODULE.CANONICAL_ADAPTIVE_CAPACITY_RECEIPT
    )


def test_adaptive_builder_fails_closed_without_or_across_fleet_identity(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, adaptive=True)
    with pytest.raises(RuntimeError, match="requires an immutable fleet config"):
        MODULE.build_production_job(
            repo,
            production_config=PRODUCTION_CONFIG,
            phase="adaptive",
            optuna_study_name="adaptive-production-test",
            production_config_opener=_open_fixture,
            adaptive_capacity_opener=_fixture_capacity_inputs,
        )

    fleet_path = repo / FLEET_CONFIG
    fleet = json.loads(fleet_path.read_text())
    fleet["adaptive_capacity_policy"]["sha256"] = "e" * 64
    fleet_path.write_text(json.dumps(fleet))
    with pytest.raises(RuntimeError, match="adaptive fleet input identity differs"):
        MODULE.build_production_job(
            repo,
            production_config=PRODUCTION_CONFIG,
            phase="adaptive",
            fleet_config=FLEET_CONFIG,
            optuna_study_name="adaptive-production-test",
            production_config_opener=_open_fixture,
            adaptive_capacity_opener=_fixture_capacity_inputs,
        )


def test_builder_rejects_infrastructure_ceiling_or_production_budget_drift(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, adaptive=True)
    with pytest.raises(RuntimeError, match="45 USD"):
        MODULE.build_production_job(
            repo,
            production_config=PRODUCTION_CONFIG,
            phase="adaptive",
            fleet_config=FLEET_CONFIG,
            maximum_cost_usd=45.01,
            optuna_study_name="fixture-study",
            production_config_opener=_open_fixture,
            adaptive_capacity_opener=_fixture_capacity_inputs,
        )

    production_path = repo / PRODUCTION_CONFIG
    production = json.loads(production_path.read_text())
    production["judge_budget"]["maximum_judge_spend_usd"] = "4.99"
    production_path.write_text(json.dumps(production))
    with pytest.raises(RuntimeError, match=r"45 USD infrastructure \+ 5 USD judge"):
        MODULE.build_production_fleet_config(
            repo,
            production_config=PRODUCTION_CONFIG,
            bundle_sha256="b" * 64,
            production_config_opener=_open_fixture,
        )


def test_canonical_v6_timed_canary_opens_and_bundles_r10_runtime_inputs() -> None:
    repo = Path(__file__).parents[1]
    production = CANONICAL_PRODUCTION_CONFIG
    raw = MODULE.build_timed_canary_job(
        repo,
        canary_config=(
            "configs/truth_editing_timed_canary_v6_adaptive_r10_bbb25ef3.json"
        ),
        command_config=(
            "configs/truth_editing_timed_canary_command_v6_adaptive_r10_v4_3ae89929.json"
        ),
    )
    parsed = ProductionVastConfig.from_mapping(raw)
    paths = set(raw["base_job"]["bundle_paths"])

    assert parsed.production_config_path == production
    assert parsed.production_config_sha256 == MODULE.file_sha256(repo / production)
    assert "configs/truth_editing_study_v5_adaptive_r10_c1f373f8.json" in paths
    assert "configs/truth_editing_evaluator_v6_adaptive_r10_c1f373f8.json" in paths
    assert sum(
        path.startswith("artifacts/directions/refit-v1/") and path.endswith(".npy")
        for path in paths
    ) == 36
    assert sum(
        path.startswith("artifacts/truth-editing/refusal-directions/vectors/")
        and path.endswith(".npy")
        for path in paths
    ) == 36
    assert sum(
        "preservation-thresholds/base-repeats/" in path
        and path.endswith(".json")
        for path in paths
    ) == 15
    direction_manifest = json.loads(
        (repo / "configs/truth_editing_direction_bank_qualified_v1.json").read_text()
    )
    referenced_direction_arrays = {
        row["artifact"]["path"].split("#", 1)[0]
        for row in direction_manifest["directions"]
    }
    refusal_bank_path = repo / "artifacts/truth-editing/refusal-directions/direction_bank.json"
    refusal_bank = json.loads(refusal_bank_path.read_text())
    referenced_refusal_arrays = {
        (refusal_bank_path.parent / row["vector_path"])
        .relative_to(repo)
        .as_posix()
        for row in refusal_bank["per_layer_receipts"]
    }
    calibration_path = (
        repo
        / "artifacts/truth-editing/vast/preservation-capture-v2-r4-r10-fetch"
        / "preservation-thresholds/calibration.json"
    )
    calibration = json.loads(calibration_path.read_text())
    referenced_calibration_receipts = {
        (calibration_path.parent / row["path"]).relative_to(repo).as_posix()
        for row in calibration["source_receipts"]
    }
    assert (
        referenced_direction_arrays
        | referenced_refusal_arrays
        | referenced_calibration_receipts
    ).issubset(paths)
    assert any(
        "preservation-capture-v2-r4-r10-fetch/preservation-runtime" in path
        for path in paths
    )
    assert not any(path.endswith((".h5", ".hdf5")) for path in paths)


def test_production_bundle_uses_only_truth_editing_source_closure() -> None:
    repo = Path(__file__).parents[1]
    paths = set(
        MODULE.collect_production_bundle_paths(
            repo,
            production_config=CANONICAL_PRODUCTION_CONFIG,
        )
    )
    source_paths = {
        path for path in paths if path.startswith("src/intelligent_liars/")
    }

    # These are production runtime prerequisites, including the deliberately
    # retained preservation-KL implementation and live structured judge route.
    assert {
        "src/intelligent_liars/truth_editing_production.py",
        "src/intelligent_liars/truth_editing_qwen_runtime.py",
        "src/intelligent_liars/truth_editing_live_judge.py",
        "src/intelligent_liars/truth_editing_record_completion.py",
        "src/intelligent_liars/truth_editing_rescore.py",
        "src/intelligent_liars/tinylora_pilot.py",
        "src/intelligent_liars/clients/openrouter_client.py",
    }.issubset(source_paths)

    # Historical Step5/D1 and unrelated package modules are not clone or bundle
    # prerequisites for the production truth-editing lane.
    assert not any("step5" in Path(path).name for path in source_paths)
    assert "src/intelligent_liars/tinylora_step5.py" not in source_paths
    assert "src/intelligent_liars/tinylora_corpus.py" not in source_paths
    assert "src/intelligent_liars/cli.py" not in source_paths


def test_record_completion_is_in_static_production_source_closure() -> None:
    assert (
        "src/intelligent_liars/truth_editing_record_completion.py"
        in MODULE.PRODUCTION_SOURCE_CLOSURE
    )


def test_canonical_adaptive_job_includes_recovery_runtime_modules() -> None:
    repo = Path(__file__).parents[1]
    job = json.loads(
        (
            repo
            / "configs/truth_editing_production_vast_job_adaptive_v2_r10_refreshable_aws.json"
        ).read_text()
    )
    bundle_paths = set(job["base_job"]["bundle_paths"])

    assert {
        "src/intelligent_liars/truth_editing_record_completion.py",
        "src/intelligent_liars/truth_editing_rescore.py",
    }.issubset(bundle_paths)


def test_rescore_driver_is_in_static_production_source_closure() -> None:
    assert (
        "src/intelligent_liars/truth_editing_rescore.py"
        in MODULE.PRODUCTION_SOURCE_CLOSURE
    )


def test_production_bundle_includes_configured_rescore_generation(tmp_path: Path) -> None:
    from test_truth_editing_preservation_materialization import _write_source
    from test_truth_editing_production import _production_config_payload
    from test_truth_editing_rescore import (
        JUDGE_CONFIG_SHA256,
        RUBRIC_SHA256,
        _source_artifacts,
    )
    from intelligent_liars.truth_editing_preservation_materialization import (
        materialize_preservation_runtime_packet,
    )

    repo = _repo(tmp_path)
    production_path = repo / PRODUCTION_CONFIG
    source = tmp_path / "source"
    source.mkdir()
    report, report_path, journal_path, checkpoint_path = _source_artifacts(
        source, optuna=True
    )
    shutil.copy2(source / "study.json", repo / "configs/rescore-study.json")
    generation = repo / "configs/recovery/rescore-generation-v1.json"
    generation.parent.mkdir(parents=True)
    materialized = materialize_rescore_generation_v1(
        source_report_path=report_path,
        source_journal_path=journal_path,
        source_checkpoint_path=checkpoint_path,
        output_path=generation,
        expected_source_study_identity_sha256=report.study_identity_sha256,
        expected_judge_config_sha256=JUDGE_CONFIG_SHA256,
        expected_rubric_sha256=RUBRIC_SHA256,
        expected_completed_trials=6,
    )
    materialize_preservation_runtime_packet(
        _write_source(tmp_path / "preservation-source"),
        repo / "artifacts/truth-editing/recovery-preservation-runtime",
    )
    production = _production_config_payload(
        study_config="rescore-study.json",
        dataset_root="../datasets/truth_editing/v2",
        scenario_view=(
            "../datasets/truth_editing/views/"
            "v2_optimization_validation_qualified_v1"
        ),
        structured_semantic_view=(
            "../datasets/truth_editing/views/structured_semantic_qualified_v1"
        ),
        structured_semantic_source_root=(
            "../corpora/tinylora_deception_action_v1/step5_v1"
        ),
        structured_base_known_qualification=(
            "../artifacts/truth-editing/structured-base-known"
        ),
        direction_manifest="truth_editing_direction_bank_qualified_v1.json",
        direction_root="..",
        refusal_direction_config="truth_editing_refusal_directions_v1.json",
        refusal_prompt_manifest="truth_editing_refusal_prompt_manifest_v1.json",
        refusal_direction_bank=(
            "../artifacts/truth-editing/refusal-directions/direction_bank.json"
        ),
        refusal_artifact_root="../artifacts/truth-editing/refusal-directions",
        evaluator_config="truth_editing_evaluator_v3.json",
        base_known_qualification=(
            "../artifacts/truth-editing/base-known"
        ),
        preservation_runtime_packet_root=(
            "../artifacts/truth-editing/recovery-preservation-runtime"
        ),
        preservation_threshold_calibration=(
            "../artifacts/truth-editing/preservation-thresholds/calibration.json"
        ),
        preservation_threshold_calibration_sha256="c" * 64,
        model_cache_dir="../model-cache/huggingface",
        snapshot_manifest_path="../model-cache/snapshot-manifest.json",
        rescore_generation="recovery/rescore-generation-v1.json",
        rescore_generation_sha256=materialized.generation_sha256,
        rescore_mode="repair_then_continue",
        verified_model_sha256=MODULE.DEFAULT_MODEL_CONTENT_SHA256,
        verified_snapshot_manifest_sha256=(
            MODULE.DEFAULT_SNAPSHOT_MANIFEST_SHA256
        ),
    )
    production_path.write_text(json.dumps(production))

    paths = MODULE.collect_production_bundle_paths(
        repo,
        production_config=PRODUCTION_CONFIG,
    )

    assert "configs/recovery/rescore-generation-v1.json" in paths
    extracted = tmp_path / "extracted"
    for relative in paths:
        target = extracted / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
    extracted_config = ProductionRunConfig.open(extracted / PRODUCTION_CONFIG)
    driver = configured_search_driver(
        extracted_config,
        load_truth_editing_study_config(extracted_config.study_config),
    )
    monitored = MonitoredSearchDriver(driver, SimpleNamespace())

    assert monitored.identity["generation_sha256"] == materialized.generation_sha256
    assert len(
        monitored.initial_journal_batches(
            study_identity_sha256="f" * 64,
            identity_inputs={},
        )
    ) == 3


def test_production_source_closure_fails_closed_when_required_file_is_missing(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    missing = repo / "src/intelligent_liars/truth_editing_qwen_runtime.py"
    missing.unlink()

    with pytest.raises(RuntimeError, match="bundle input is missing or unsafe"):
        MODULE.collect_production_bundle_paths(
            repo,
            production_config=PRODUCTION_CONFIG,
            production_config_opener=_open_fixture,
        )


def test_provenance_metadata_cannot_admit_raw_or_sealed_question_content() -> None:
    repo = Path(__file__).parents[1]
    paths = MODULE.collect_production_bundle_paths(
        repo,
        production_config=CANONICAL_PRODUCTION_CONFIG,
    )

    # source_receipts.json records these construction inputs for auditability;
    # their path strings must not turn them into production runtime inputs.
    assert not any(path.startswith("data/geometry_of_truth/") for path in paths)
    assert not any(path.startswith("data/sycophancy/") for path in paths)

    sealed_record_id = b"qa_2e5d824f0306dab2e3d82843caca98b105164614ff85d7cc969ce6b8e2942a5b"
    sealed_question = b"Which of the following is true in peripheral neuropathy?"
    for relative in paths:
        content = (repo / relative).read_bytes()
        assert sealed_record_id not in content, relative
        assert sealed_question not in content, relative


def test_builder_emits_one_gpu_timed_canary_with_late_bound_dph_and_outputs() -> None:
    repo = Path(__file__).parents[1]
    raw = MODULE.build_timed_canary_job(
        repo,
        canary_config=(
            "configs/truth_editing_timed_canary_v6_adaptive_r10_bbb25ef3.json"
        ),
        command_config=(
            "configs/truth_editing_timed_canary_command_v6_adaptive_r10_v4_3ae89929.json"
        ),
    )
    parsed = ProductionVastConfig.from_mapping(raw)
    rendered = json.dumps(raw, sort_keys=True)

    assert parsed.phase == "timed_canary"
    assert raw["base_job"]["resources"]["maximum_elapsed_seconds"] == 5400
    assert raw["base_job"]["resources"]["maximum_cost_usd"] == 4.0
    assert raw["base_job"]["expected_outputs"] == [
        "timed-canary-v6-adaptive-r10/observation.json",
        "timed-canary-v6-adaptive-r10/receipt.json",
        "timed-canary-v6-adaptive-r10/monitoring/wandb-run.json",
        "timed-canary-v6-adaptive-r10/monitoring/wandb-events.jsonl",
        "timed-canary-v6-adaptive-r10/providers/production-judge-budget/manifest.json",
        "model/cache-hydration-receipt.json",
        "model/model-verification-receipt.json",
    ]
    assert "$TRUTH_EDITING_GPU_HOURLY_USD" in raw["study"]["workload_command"][-1]
    assert "--execute" in raw["study"]["workload_command"][-1]
    assert "OPENROUTER_API_KEY" not in rendered
    assert "WANDB_API_KEY" not in rendered
    assert (
        "configs/truth_editing_timed_canary_v6_adaptive_r10_bbb25ef3.json"
        in rendered
    )
    assert (
        "configs/truth_editing_timed_canary_command_v6_adaptive_r10_v4_3ae89929.json"
        in rendered
    )
    packaged_command = json.loads(
        (
            repo
            / "configs/truth_editing_timed_canary_command_v6_adaptive_r10_v4_3ae89929.json"
        ).read_text()
    )
    assert packaged_command[packaged_command.index("--wandb-project") + 1] == (
        "intelligent-liars"
    )
    assert packaged_command[packaged_command.index("--wandb-entity") + 1] == (
        "centipawn"
    )
    parsed.validate_repository(repo)


def test_canonical_timed_canary_packets_bind_exact_production_identity() -> None:
    repo = Path(__file__).parents[1]
    production_path = (
        repo
        / "configs/truth_editing_production_v6_adaptive_r10_17b1e9cb_c1f373f8.json"
    )
    canary_path = repo / MODULE.CANONICAL_TIMED_CANARY_CONFIG
    command_path = repo / MODULE.CANONICAL_TIMED_CANARY_COMMAND_CONFIG
    production_sha256 = "bbb25ef3f7ef3778947c95db6831f7150c57557b50e7cfce025c28ce11b9e2c0"

    assert MODULE.file_sha256(production_path) == production_sha256
    assert MODULE.CANONICAL_TIMED_CANARY_CONFIG == (
        "configs/truth_editing_timed_canary_v6_adaptive_r10_bbb25ef3.json"
    )
    assert MODULE.CANONICAL_TIMED_CANARY_CONFIG_SHA256 == (
        "7d0edca00f7934031225fa543538ab9e9349fb68fbcb3d587353a747b841b0b2"
    )
    assert MODULE.file_sha256(canary_path) == MODULE.CANONICAL_TIMED_CANARY_CONFIG_SHA256
    assert MODULE.CANONICAL_TIMED_CANARY_COMMAND_CONFIG == (
        "configs/truth_editing_timed_canary_command_v6_adaptive_r10_v4_3ae89929.json"
    )
    assert MODULE.CANONICAL_TIMED_CANARY_COMMAND_SHA256 == (
        "3ae899296b0e56f4be9fdba6288a0d0924b7099865dc83ee9985e591d1bded0c"
    )
    assert MODULE.file_sha256(command_path) == MODULE.CANONICAL_TIMED_CANARY_COMMAND_SHA256

    canary = json.loads(canary_path.read_text())
    command = json.loads(command_path.read_text())
    assert canary["production_config"] == {
        "path": production_path.relative_to(repo).as_posix(),
        "sha256": production_sha256,
    }
    assert command[command.index("--config-sha256") + 1] == production_sha256


def test_canonical_timed_canary_vast_job_packages_only_current_packets() -> None:
    repo = Path(__file__).parents[1]
    job_path = (
        repo
        / "configs/truth_editing_timed_canary_vast_job_v6_adaptive_r10_bbb25ef3.json"
    )
    assert MODULE.file_sha256(job_path) == (
        "a454d60587ecbae901e3aaf0c6427aabf45e48e27346a11370172fcfdad153e8"
    )
    job = json.loads(job_path.read_text())
    bundle_paths = set(job["base_job"]["bundle_paths"])

    assert job["study"]["production_config_sha256"] == (
        "bbb25ef3f7ef3778947c95db6831f7150c57557b50e7cfce025c28ce11b9e2c0"
    )
    assert MODULE.CANONICAL_TIMED_CANARY_CONFIG in bundle_paths
    assert MODULE.CANONICAL_TIMED_CANARY_COMMAND_CONFIG in bundle_paths
    assert (
        "configs/truth_editing_timed_canary_v6_adaptive_r10_92bd7c59.json"
        not in bundle_paths
    )
    assert (
        "configs/truth_editing_timed_canary_command_v6_adaptive_r10_v3_546cfccb.json"
        not in bundle_paths
    )
    assert not any(path.endswith("/test.jsonl") for path in bundle_paths)
    ProductionVastConfig.from_mapping(job).validate_repository(repo)


def test_timed_canary_builder_rejects_superseded_command_packet(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs/truth_editing_timed_canary_v4_r10_7513ee91.json").write_text("{}")
    (
        repo
        / "configs/truth_editing_timed_canary_command_v4_r10_7513ee91.json"
    ).write_text("{}")
    with pytest.raises(RuntimeError, match="superseded"):
        MODULE.build_timed_canary_job(
            repo,
            canary_config=(
                "configs/truth_editing_timed_canary_v4_r10_7513ee91.json"
            ),
            command_config=(
                "configs/truth_editing_timed_canary_command_v4_r10_7513ee91.json"
            ),
        )


def test_timed_canary_packaging_rejects_legacy_command_without_explicit_entity(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs/truth_editing_timed_canary_v4_r10_7513ee91.json").write_text("{}")
    (
        repo
        / "configs/truth_editing_timed_canary_command_v4_r10_7513ee91.json"
    ).write_text("{}")
    with pytest.raises(RuntimeError, match="entity centipawn"):
        MODULE.build_timed_canary_job(
            repo,
            canary_config="configs/truth_editing_timed_canary_v4_r10_7513ee91.json",
            command_config=(
                "configs/truth_editing_timed_canary_command_v4_r10_7513ee91.json"
            ),
        )
