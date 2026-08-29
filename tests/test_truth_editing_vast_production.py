from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from intelligent_liars.truth_editing_vast_prerequisites import (
    EphemeralWorkloadSecret,
    FROZEN_BASE_IMAGE,
    FROZEN_MODEL_ID,
    FROZEN_MODEL_REVISION,
    FROZEN_RUNTIME_ID,
    Offer,
    resolve_aws_workload_credentials,
)
from intelligent_liars.truth_editing_vast_production import (
    ADAPTIVE_PRODUCTION_JOB_FORMAT,
    AdaptiveProductionJobConfig,
    ProductionVastConfig,
    ProductionVastError,
    build_production_bundle,
    execute_production_lifecycle,
    production_lifecycle_plan,
)


_PRODUCTION_PATH = "configs/truth_editing_production_v4_fixture_deadbeef.json"
_PRODUCTION_SHA256 = "d" * 64


def _raw() -> dict[str, object]:
    return {
        "format": "truth_editing_vast_production_job_v1",
        "phase": "discovery",
        "base_job": {
            "format": "truth_editing_vast_prerequisite_job_v1",
            "image": FROZEN_BASE_IMAGE,
            "runtime_id": FROZEN_RUNTIME_ID,
            "model": {"repository": FROZEN_MODEL_ID, "revision": FROZEN_MODEL_REVISION},
            "resources": {
                "disk_gib": 120,
                "minimum_gpu_vram_gib": 24,
                "maximum_elapsed_seconds": 3600,
                "maximum_cost_usd": 4.0,
                "maximum_download_gib": 50.0,
                "maximum_upload_gib": 1.0,
            },
            "paths": {
                "remote_workdir": "/workspace/repo",
                "remote_output_dir": "/workspace/outputs",
            },
            "commands": {"bootstrap": ["bash", "bootstrap.sh"], "workload": ["false"]},
            "expected_outputs": ["study-receipt.json", "checkpoints/latest.json"],
            "bundle_paths": ["bootstrap.sh", "worker.sh", _PRODUCTION_PATH],
        },
        "model_cache": {
            "remote_directory": "/workspace/model-cache",
            "expected_model_sha256": "b" * 64,
            "expected_snapshot_manifest_sha256": "c" * 64,
            "hydrate_command": ["bash", "hydrate.sh", "/workspace/model-cache"],
            "verify_command": ["bash", "verify-cache.sh", "/workspace/model-cache"],
        },
        "study": {
            "production_config_path": _PRODUCTION_PATH,
            "production_config_sha256": _PRODUCTION_SHA256,
            "workload_command": ["bash", "worker.sh", "--phase", "discovery"],
        },
        "checkpoints": {
            "remote_directory": "/workspace/outputs/checkpoints",
            "interval_seconds": 300,
            "stream_command": ["bash", "publish-checkpoint.sh"],
        },
    }


def _adaptive_raw() -> dict[str, object]:
    raw = _raw()
    raw["phase"] = "adaptive"
    raw["base_job"]["format"] = ADAPTIVE_PRODUCTION_JOB_FORMAT  # type: ignore[index]
    raw["base_job"]["resources"]["maximum_elapsed_seconds"] = 24 * 3600  # type: ignore[index]
    raw["base_job"]["resources"]["maximum_cost_usd"] = 45.0  # type: ignore[index]
    raw["base_job"]["expected_outputs"] = [  # type: ignore[index]
        "study/adaptive-run-receipt.json",
        "checkpoints/adaptive-latest.json",
        "finalization/final-model-publication-receipt.json",
    ]
    raw["study"]["workload_command"] = [  # type: ignore[index]
        "python",
        "scripts/run_truth_editing_cuda_fleet_controller.py",
        "--fleet-config",
        "configs/fleet.json",
        "--capacity-policy",
        "configs/capacity.json",
        "--capacity-receipt",
        "artifacts/capacity-receipt.json",
        "--output-root",
        "/workspace/outputs",
        "--checkpoint-publication-root",
        "/workspace/outputs/checkpoints",
        "--model-registry-config",
        "configs/model_registry_v1.json",
        "--offhost-key-prefix",
        "model-registry/v1/truth-editing/adaptive-main-r11",
        "--final-model-slug",
        "qwen3-vl-8b-truth-edited",
    ]
    raw["checkpoints"]["stream_command"] = [  # type: ignore[index]
        "python",
        "scripts/publish_truth_editing_production_checkpoint.py",
        "--adaptive",
        "--study-config-sha256",
        "d" * 64,
    ]
    return raw


def _offer() -> Offer:
    return Offer.from_mapping({
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_total": 0.5,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    })


def _eight_gpu_offer() -> Offer:
    return Offer.from_multi_gpu_mapping({
        "id": 456,
        "gpu_name": "RTX 3090",
        "num_gpus": 8,
        "gpu_ram": 24576,
        "dph_base": 1.5,
        "dph_total": 1.5,
        "storage_cost": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    })


def _ssh_identity(tmp_path: Path) -> Path:
    identity = tmp_path / "id_ed25519"
    identity.write_text("test identity")
    return identity


def test_config_is_strict_and_phase_bound() -> None:
    raw = _raw()
    raw["extra"] = True
    with pytest.raises(ProductionVastError, match="fields or format"):
        ProductionVastConfig.from_mapping(raw)
    raw = _raw()
    raw["phase"] = "everything"
    with pytest.raises(ProductionVastError, match="phase"):
        ProductionVastConfig.from_mapping(raw)
    raw = _raw()
    raw["study"]["production_config_path"] = "../outside.json"  # type: ignore[index]
    raw["base_job"]["bundle_paths"] = [  # type: ignore[index]
        "bootstrap.sh", "worker.sh", "../outside.json"
    ]
    with pytest.raises(ProductionVastError, match="unsafe bundle path|relative"):
        ProductionVastConfig.from_mapping(raw)
    raw = _raw()
    del raw["study"]["production_config_sha256"]  # type: ignore[index]
    with pytest.raises(ProductionVastError, match="study fields"):
        ProductionVastConfig.from_mapping(raw)


def test_adaptive_job_rejects_legacy_runtime_without_offhost_sdk() -> None:
    raw = _adaptive_raw()
    raw["base_job"]["runtime_id"] = (  # type: ignore[index]
        "truth-editing-cu124-torch251-transformers4571-fa274post1-optuna490-wandb0290-r2"
    )

    with pytest.raises(ProductionVastError, match="off-host checkpoint runtime"):
        ProductionVastConfig.from_mapping(raw)


def test_adaptive_production_job_has_a_distinct_strict_24_hour_45_dollar_contract() -> None:
    config = ProductionVastConfig.from_mapping(_adaptive_raw())

    assert config.phase == "adaptive"
    assert isinstance(config.base_job, AdaptiveProductionJobConfig)
    assert config.base_job.maximum_elapsed_seconds == 24 * 3600
    assert config.base_job.maximum_cost_usd == 45.0


def test_adaptive_plan_exports_eight_gpu_offer_price_as_host_total_hourly_rate(
    tmp_path: Path,
) -> None:
    """Vast prices an 8-GPU offer per host, not once per GPU."""

    plan = production_lifecycle_plan(
        vastai="vastai",
        config=ProductionVastConfig.from_mapping(_adaptive_raw()),
        offer=_eight_gpu_offer(),
        bundle=tmp_path / "bundle.tar.gz",
        fetch_dir=tmp_path / "fetch",
    )

    remote = plan["base_lifecycle"]["remote_command"]
    assert plan["base_lifecycle"]["offer"]["projected_max_cost_usd"] == pytest.approx(
        1.5 * 24
    )
    assert "TRUTH_EDITING_HOST_HOURLY_USD=1.5" in remote
    assert "TRUTH_EDITING_HOST_LEASE_STARTED_AT_UTC" in remote
    assert "TRUTH_EDITING_GPU_HOURLY_USD" not in remote
    assert "mkdir -p /workspace/outputs/checkpoints" not in remote


def test_adaptive_job_requires_verified_remote_final_model_publication() -> None:
    raw = _adaptive_raw()
    raw["base_job"]["expected_outputs"].remove(  # type: ignore[index]
        "finalization/final-model-publication-receipt.json"
    )

    with pytest.raises(ProductionVastError, match="final model publication"):
        ProductionVastConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("phase", "base_format", "message"),
    [
        ("adaptive", "truth_editing_vast_prerequisite_job_v1", "adaptive production"),
        ("discovery", ADAPTIVE_PRODUCTION_JOB_FORMAT, "legacy production"),
        ("timed_canary", ADAPTIVE_PRODUCTION_JOB_FORMAT, "timed canary"),
    ],
)
def test_production_job_format_cannot_cross_phase_contracts(
    phase: str, base_format: str, message: str
) -> None:
    raw = _adaptive_raw() if phase == "adaptive" else _raw()
    raw["phase"] = phase
    raw["base_job"]["format"] = base_format  # type: ignore[index]
    if phase != "adaptive":
        raw["study"]["workload_command"][-1] = phase  # type: ignore[index]

    with pytest.raises(ProductionVastError, match=message):
        ProductionVastConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("resource", "value"),
    [
        ("maximum_elapsed_seconds", 24 * 3600 + 1),
        ("maximum_cost_usd", 45.000001),
    ],
)
def test_adaptive_production_job_fails_closed_above_its_limits(
    resource: str, value: float
) -> None:
    raw = _adaptive_raw()
    raw["base_job"]["resources"][resource] = value  # type: ignore[index]

    with pytest.raises(ProductionVastError, match="base_job is invalid"):
        ProductionVastConfig.from_mapping(raw)


def test_bundle_is_allowlisted_and_plan_has_hydration_checkpointing_and_cleanup(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    for path in ("bootstrap.sh", "worker.sh", _PRODUCTION_PATH):
        target = repo / path
        target.write_text("{}" if target.suffix == ".json" else "true\n")
    (repo / "secret.txt").write_text("no")
    raw = _raw()
    raw["study"]["production_config_sha256"] = hashlib.sha256(  # type: ignore[index]
        (repo / _PRODUCTION_PATH).read_bytes()
    ).hexdigest()
    config = ProductionVastConfig.from_mapping(raw)
    bundle = tmp_path / "bundle.tar.gz"
    manifest = build_production_bundle(
        repo,
        config,
        bundle,
        production_config_opener=lambda _path: SimpleNamespace(
            verified_model_sha256="b" * 64,
            verified_snapshot_manifest_sha256="c" * 64,
        ),
    )
    assert {row["path"] for row in manifest["files"]} == {
        "bootstrap.sh", "worker.sh", _PRODUCTION_PATH
    }
    plan = production_lifecycle_plan(
        vastai="vastai", config=config, offer=_offer(), bundle=bundle,
        fetch_dir=tmp_path / "fetch",
    )
    remote = plan["base_lifecycle"]["remote_command"]
    assert "hydrate.sh" in remote
    assert "verify-cache.sh" in remote
    assert "TRUTH_EDITING_EXPECTED_MODEL_SHA256=" + "b" * 64 in remote
    assert "publish-checkpoint.sh" in remote
    assert "worker.sh --phase discovery" in remote
    assert plan["base_lifecycle"]["destroy_command_template"][-3:] == [
        "instance", "<INSTANCE_ID>", "--raw"
    ]


def test_config_rejects_unbound_workload_and_noncompact_fetch() -> None:
    raw = _raw()
    raw["study"]["workload_command"] = ["bash", "worker.sh"]  # type: ignore[index]
    with pytest.raises(ProductionVastError, match="selected phase"):
        ProductionVastConfig.from_mapping(raw)
    raw = _raw()
    raw["base_job"]["resources"]["maximum_upload_gib"] = 2.0  # type: ignore[index]
    with pytest.raises(ProductionVastError, match="compact output"):
        ProductionVastConfig.from_mapping(raw)


def test_execute_wraps_base_lifecycle_and_proves_zero_label_lineage(tmp_path: Path) -> None:
    config = ProductionVastConfig.from_mapping(_raw())
    plan = production_lifecycle_plan(
        vastai="vastai", config=config, offer=_offer(), bundle=Path("bundle"),
        fetch_dir=tmp_path / "fetch",
        ssh_identity=_ssh_identity(tmp_path),
    )
    base_calls: list[dict[str, object]] = []

    def base_execute(**kwargs: object) -> dict[str, object]:
        base_calls.append(kwargs)
        return {"format": "truth_editing_vast_prerequisite_lifecycle_v2", "destroyed": True,
                "instance_id": "999", "estimated_cost_usd": 0.2, "self_sha256": "a" * 64}

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        assert command[1:3] == ["show", "instances"]
        return SimpleNamespace(stdout=json.dumps([{"id": 7, "label": "unrelated"}]), returncode=0)

    receipt = execute_production_lifecycle(
        plan=plan, config=config, metadata_path=tmp_path / "production.json",
        workload_secret=EphemeralWorkloadSecret.openrouter("test-secret"),
        base_execute=base_execute, run_command=run,
    )
    assert len(base_calls) == 1
    assert receipt["zero_lineage_instances_verified"] is True
    assert receipt["base_lifecycle_receipt_sha256"] == "a" * 64
    assert base_calls[0]["workload_secret"].environment_name == "OPENROUTER_API_KEY"  # type: ignore[union-attr]


@pytest.mark.parametrize("identity_state", ["omitted", "missing", "symlink"])
def test_production_execute_requires_real_explicit_ssh_identity_before_launch(
    tmp_path: Path, identity_state: str
) -> None:
    config = ProductionVastConfig.from_mapping(_raw())
    identity: Path | None = None
    if identity_state == "missing":
        identity = tmp_path / "missing-id_ed25519"
    elif identity_state == "symlink":
        target = tmp_path / "identity-target"
        target.write_text("test identity")
        identity = tmp_path / "identity-link"
        identity.symlink_to(target)
    plan = production_lifecycle_plan(
        vastai="vastai",
        config=config,
        offer=_offer(),
        bundle=Path("bundle"),
        fetch_dir=tmp_path / "fetch",
        ssh_identity=identity,
    )
    base_calls: list[dict[str, object]] = []
    vast_calls: list[list[str]] = []

    def base_execute(**kwargs: object) -> dict[str, object]:
        base_calls.append(kwargs)
        raise AssertionError("production launch must not begin")

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        vast_calls.append(command)
        raise AssertionError("Vast must not be contacted")

    with pytest.raises(ProductionVastError, match="explicit SSH identity"):
        execute_production_lifecycle(
            plan=plan,
            config=config,
            metadata_path=tmp_path / "production.json",
            workload_secret=EphemeralWorkloadSecret.openrouter("test-secret"),
            base_execute=base_execute,
            run_command=run,
        )

    assert base_calls == []
    assert vast_calls == []
    assert not (tmp_path / "production.json").exists()


def test_adaptive_production_execute_still_requires_explicit_ssh_identity(
    tmp_path: Path,
) -> None:
    config = ProductionVastConfig.from_mapping(_adaptive_raw())
    plan = production_lifecycle_plan(
        vastai="vastai",
        config=config,
        offer=_offer(),
        bundle=Path("bundle"),
        fetch_dir=tmp_path / "fetch",
    )
    external_calls: list[object] = []

    def forbidden_call(*args: object, **kwargs: object) -> object:
        external_calls.append((args, kwargs))
        raise AssertionError("execution must stop before any external call")

    with pytest.raises(ProductionVastError, match="explicit SSH identity"):
        execute_production_lifecycle(
            plan=plan,
            config=config,
            metadata_path=tmp_path / "production.json",
            workload_secret=EphemeralWorkloadSecret.openrouter("test-secret"),
            base_execute=forbidden_call,  # type: ignore[arg-type]
            run_command=forbidden_call,  # type: ignore[arg-type]
        )

    assert external_calls == []


def test_adaptive_production_requires_aws_credentials_valid_for_entire_paid_window_before_launch(
    tmp_path: Path,
) -> None:
    config = ProductionVastConfig.from_mapping(_adaptive_raw())
    plan = production_lifecycle_plan(
        vastai="vastai",
        config=config,
        offer=_eight_gpu_offer(),
        bundle=Path("bundle"),
        fetch_dir=tmp_path / "fetch",
        ssh_identity=_ssh_identity(tmp_path),
    )
    external_calls: list[object] = []

    def forbidden_call(*args: object, **kwargs: object) -> object:
        external_calls.append((args, kwargs))
        raise AssertionError("execution must stop before any external call")

    with pytest.raises(ProductionVastError, match="AWS credentials"):
        execute_production_lifecycle(
            plan=plan,
            config=config,
            metadata_path=tmp_path / "production.json",
            workload_secret=EphemeralWorkloadSecret.production(
                openrouter_value="test-openrouter-secret",
                wandb_value="test-wandb-secret",
            ),
            base_execute=forbidden_call,  # type: ignore[arg-type]
            run_command=forbidden_call,  # type: ignore[arg-type]
        )

    expiring = resolve_aws_workload_credentials(
        environment={
            "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE12345678",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-value-that-must-not-leak",
            "AWS_SESSION_TOKEN": "temporary-token",
            "AWS_CREDENTIAL_EXPIRATION": (
                datetime.now(timezone.utc) + timedelta(hours=2)
            ).isoformat(),
        },
        minimum_validity_seconds=0,
    )
    with pytest.raises(ProductionVastError, match="expire before"):
        execute_production_lifecycle(
            plan=plan,
            config=config,
            metadata_path=tmp_path / "production.json",
            workload_secret=EphemeralWorkloadSecret.production(
                openrouter_value="test-openrouter-secret",
                wandb_value="test-wandb-secret",
                aws_credentials=expiring,
            ),
            base_execute=forbidden_call,  # type: ignore[arg-type]
            run_command=forbidden_call,  # type: ignore[arg-type]
        )

    assert external_calls == []


def test_execute_fails_closed_if_lineage_instance_remains(tmp_path: Path) -> None:
    config = ProductionVastConfig.from_mapping(_raw())
    plan = production_lifecycle_plan(
        vastai="vastai", config=config, offer=_offer(), bundle=Path("bundle"),
        fetch_dir=tmp_path / "fetch",
        ssh_identity=_ssh_identity(tmp_path),
    )

    def base_execute(**_kwargs: object) -> dict[str, object]:
        return {"destroyed": True, "instance_id": "999", "estimated_cost_usd": 0.1,
                "self_sha256": "b" * 64}

    def run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=json.dumps([{"id": 999, "label": plan["label"]}]), returncode=0)

    with pytest.raises(ProductionVastError, match="lineage instance remains"):
        execute_production_lifecycle(
            plan=plan, config=config, metadata_path=tmp_path / "production.json",
            workload_secret=EphemeralWorkloadSecret.openrouter("test-secret"),
            base_execute=base_execute, run_command=run,
        )
    assert not (tmp_path / "production.json").exists()


def test_completed_worker_does_not_wait_for_checkpoint_interval(tmp_path: Path) -> None:
    raw = _raw()
    raw["model_cache"]["remote_directory"] = str(tmp_path / "model-cache")  # type: ignore[index]
    raw["base_job"]["paths"]["remote_workdir"] = str(tmp_path / "repo")  # type: ignore[index]
    raw["base_job"]["paths"]["remote_output_dir"] = str(tmp_path / "outputs")  # type: ignore[index]
    raw["checkpoints"]["remote_directory"] = str(tmp_path / "outputs/checkpoints")  # type: ignore[index]
    raw["checkpoints"]["interval_seconds"] = 3600  # type: ignore[index]
    raw["model_cache"]["hydrate_command"] = ["bash", "-c", "true"]  # type: ignore[index]
    raw["model_cache"]["verify_command"] = ["bash", "-c", "true"]  # type: ignore[index]
    raw["study"]["workload_command"] = [  # type: ignore[index]
        "bash", "-c", "printf worker-complete", "ignored", "--phase", "discovery"
    ]
    stream_marker = tmp_path / "streamed"
    raw["checkpoints"]["stream_command"] = [  # type: ignore[index]
        "bash", "-c", f"printf streamed >> {stream_marker}"
    ]
    config = ProductionVastConfig.from_mapping(raw)
    plan = production_lifecycle_plan(
        vastai="vastai", config=config, offer=_offer(), bundle=Path("bundle"),
        fetch_dir=tmp_path / "fetch",
        ssh_identity=_ssh_identity(tmp_path),
    )

    def base_execute(**kwargs: object) -> dict[str, object]:
        job = kwargs["config"]
        completed = subprocess.run(
            job.workload_command,  # type: ignore[union-attr]
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert completed.returncode == 0
        assert completed.stdout == "worker-complete"
        return {
            "destroyed": True,
            "instance_id": "999",
            "estimated_cost_usd": 0.0,
            "self_sha256": "a" * 64,
        }

    def run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="[]", returncode=0)

    execute_production_lifecycle(
        plan=plan,
        config=config,
        metadata_path=tmp_path / "production.json",
        workload_secret=EphemeralWorkloadSecret.openrouter("test-secret"),
        base_execute=base_execute,
        run_command=run,
    )
    assert stream_marker.read_text() == "streamed"


def test_failed_worker_is_reported_after_prompt_final_checkpoint(tmp_path: Path) -> None:
    raw = _raw()
    raw["model_cache"]["remote_directory"] = str(tmp_path / "model-cache")  # type: ignore[index]
    raw["base_job"]["paths"]["remote_workdir"] = str(tmp_path / "repo")  # type: ignore[index]
    raw["base_job"]["paths"]["remote_output_dir"] = str(tmp_path / "outputs")  # type: ignore[index]
    raw["checkpoints"]["remote_directory"] = str(tmp_path / "outputs/checkpoints")  # type: ignore[index]
    raw["checkpoints"]["interval_seconds"] = 3600  # type: ignore[index]
    raw["model_cache"]["hydrate_command"] = ["bash", "-c", "true"]  # type: ignore[index]
    raw["model_cache"]["verify_command"] = ["bash", "-c", "true"]  # type: ignore[index]
    raw["study"]["workload_command"] = [  # type: ignore[index]
        "bash", "-c", "printf worker-failed; exit 47", "ignored", "--phase", "discovery"
    ]
    stream_marker = tmp_path / "streamed"
    raw["checkpoints"]["stream_command"] = [  # type: ignore[index]
        "bash", "-c", f"printf streamed >> {stream_marker}"
    ]
    config = ProductionVastConfig.from_mapping(raw)
    plan = production_lifecycle_plan(
        vastai="vastai", config=config, offer=_offer(), bundle=Path("bundle"),
        fetch_dir=tmp_path / "fetch",
        ssh_identity=_ssh_identity(tmp_path),
    )

    def base_execute(**kwargs: object) -> dict[str, object]:
        job = kwargs["config"]
        completed = subprocess.run(
            job.workload_command,  # type: ignore[union-attr]
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert completed.returncode != 0
        assert completed.stdout == "worker-failed"
        assert stream_marker.read_text() == "streamed"
        raise RuntimeError("base lifecycle observed the workload failure")

    def run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="[]", returncode=0)

    with pytest.raises(RuntimeError, match="workload failure"):
        execute_production_lifecycle(
            plan=plan,
            config=config,
            metadata_path=tmp_path / "production.json",
            workload_secret=EphemeralWorkloadSecret.openrouter("test-secret"),
            base_execute=base_execute,
            run_command=run,
        )
