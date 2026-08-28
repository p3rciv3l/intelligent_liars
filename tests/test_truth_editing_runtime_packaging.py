from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

from intelligent_liars.truth_editing_vast_prerequisites import FROZEN_RUNTIME_ID


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "docker" / "truth-editing"


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _validator_module():
    path = IMAGE_ROOT / "validate_runtime.py"
    spec = importlib.util.spec_from_file_location("truth_editing_runtime_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_project_keeps_cuda_only_flash_attention_out_of_dependencies() -> None:
    project = _pyproject()["project"]
    dependencies = project["dependencies"]
    study = project["optional-dependencies"]["study"]

    assert not any(item.startswith("flash-attn") for item in dependencies)
    assert study == [
        "boto3==1.40.18",
        "botocore==1.40.18",
        "optuna>=4.9,<5",
        "wandb==0.29.0",
    ]


def test_editable_runtime_helper_is_installed_for_every_editable_install() -> None:
    project = _pyproject()["project"]

    assert "editables>=0.5" in project["dependencies"]

    import editables
    import intelligent_liars

    assert editables.__name__ == "editables"
    assert intelligent_liars.__name__ == "intelligent_liars"


def test_production_manifest_and_lock_are_self_consistent() -> None:
    validator = _validator_module()
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())

    assert validator.validate_metadata(manifest, IMAGE_ROOT) == []
    assert manifest["intended_workload"] == "truth-editing Optuna study execution"
    assert manifest["python_packages"]["optuna"] == "4.9.0"
    assert manifest["python_packages"]["wandb"] == "0.29.0"
    assert manifest["python_packages"]["boto3"] == "1.40.18"
    assert manifest["python_packages"]["botocore"] == "1.40.18"
    assert manifest["runtime_id"] == FROZEN_RUNTIME_ID
    assert manifest["available_optuna_storage_api"] == {
        "backend": "JournalFileBackend",
        "storage": "JournalStorage",
    }


def test_production_image_uses_compatible_pinned_cuda_stack() -> None:
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text()
    requirements = (IMAGE_ROOT / "requirements.in").read_text().splitlines()
    lock = (IMAGE_ROOT / "requirements.lock").read_text()

    assert manifest["python_packages"]["torch"] == "2.5.1+cu124"
    assert manifest["platform"]["cuda"] == "12.4"
    assert manifest["python_packages"]["transformers"] == "4.57.1"
    assert manifest["python_packages"]["flash-attn"] == "2.7.4.post1"
    assert manifest["base_image"]["reference"] in dockerfile
    assert "ARG BASE_IMAGE" not in dockerfile
    assert dockerfile.startswith("# syntax=docker/dockerfile:1.7\n\nFROM pytorch/pytorch:")
    assert "COPY Dockerfile runtime-manifest.json validate_runtime.py" in dockerfile
    assert "ADD --checksum=sha256:" in dockerfile
    assert manifest["flash_attention_wheel"]["sha256"] in dockerfile
    assert "Qwen3VLForConditionalGeneration" in (
        IMAGE_ROOT / "validate_runtime.py"
    ).read_text()
    assert "_supports_flash_attn" in (IMAGE_ROOT / "validate_runtime.py").read_text()
    assert "is_flash_attn_2_available" in (
        IMAGE_ROOT / "validate_runtime.py"
    ).read_text()
    assert "python -m pip install --no-deps /opt/truth-editing/wheels/" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "optuna==4.9.0" in lock
    assert "wandb==0.29.0" in lock
    assert "boto3==1.40.18" in requirements
    assert "botocore==1.40.18" in requirements
    assert "boto3==1.40.18" in lock
    assert "botocore==1.40.18" in lock
    assert "transformers==4.57.1" in lock
    assert "--hash=sha256:" in lock
    assert not any(line.startswith("torch==") for line in lock.splitlines())
    assert not any(line.startswith("triton==") for line in lock.splitlines())
    assert not any(line.startswith("nvidia-") for line in lock.splitlines())


def test_production_bootstrap_verifies_monitoring_without_logging_in() -> None:
    bootstrap = (
        ROOT / "scripts" / "bootstrap_truth_editing_production_worker.sh"
    ).read_text()
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text()

    assert '"wandb": "0.29.0"' in bootstrap
    assert '"boto3": "1.40.18"' in bootstrap
    assert '"botocore": "1.40.18"' in bootstrap
    assert "wandb login" not in bootstrap
    assert "WANDB_API_KEY" not in bootstrap
    assert "wandb login" not in dockerfile
    assert "WANDB_API_KEY" not in dockerfile
    assert "WANDB_SILENT=true" in dockerfile
    assert "WANDB_CONSOLE=off" in dockerfile
    assert "WANDB_DISABLE_CODE=true" in dockerfile


def test_production_validator_constructs_offhost_s3_client_without_network() -> None:
    validator = _validator_module()

    assert validator.validate_offhost_checkpoint_runtime() == []


def test_runtime_identity_covers_executable_packaging_inputs() -> None:
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())

    assert set(manifest["build_inputs"]) == {
        "Dockerfile",
        "requirements.in",
        "requirements.lock",
        "validate_runtime.py",
    }
    assert manifest["lock_compiler"] == {
        "index": "https://pypi.org/simple",
        "tool": "uv",
        "version": "0.6.6",
    }


def test_docker_context_is_a_positive_allowlist() -> None:
    dockerignore = (IMAGE_ROOT / ".dockerignore").read_text().splitlines()
    allowed = {line for line in dockerignore if line.startswith("!")}

    assert "**" in dockerignore
    assert allowed == {
        "!Dockerfile",
        "!requirements.in",
        "!requirements.lock",
        "!runtime-manifest.json",
        "!validate_runtime.py",
    }
