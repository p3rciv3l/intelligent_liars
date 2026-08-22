from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "docker" / "tinylora-step5"


def _validator_module():
    path = IMAGE_ROOT / "validate_runtime.py"
    spec = importlib.util.spec_from_file_location("tinylora_step5_image_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_manifest_and_build_inputs_are_consistent() -> None:
    validator = _validator_module()
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())

    assert validator.validate_metadata(manifest, IMAGE_ROOT) == []
    assert manifest["large_run_enabled"] is False


def test_image_uses_digest_base_and_prebuilt_hashed_flash_attention() -> None:
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text()
    wheel = manifest["flash_attention_wheel"]

    assert "@sha256:" in manifest["base_image"]["reference"]
    assert "ADD --checksum=sha256:" in dockerfile
    assert wheel["sha256"] in dockerfile
    assert wheel["cxx11_abi"] is False
    assert "pip install --no-deps" in dockerfile


def test_lock_is_hash_pinned_and_cannot_replace_base_torch() -> None:
    lock = (IMAGE_ROOT / "requirements.lock").read_text()

    assert "--hash=sha256:" in lock
    assert not any(line.startswith("torch==") for line in lock.splitlines())
    assert not any(line.startswith("triton==") for line in lock.splitlines())
    assert not any(line.startswith("nvidia-") for line in lock.splitlines())
    assert "transformers==5.15.1" in lock
    assert "liger-kernel==0.8.2" in lock


def test_cache_policy_requires_revision_pin_and_capacity() -> None:
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())
    cache = manifest["cache_policy"]

    assert cache["root"] == "/workspace/cache"
    assert cache["minimum_free_gib"] >= 80
    assert cache["model_revision_must_be_pinned_by_run"] is True
    assert cache["embed_model_weights_in_image"] is False


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
