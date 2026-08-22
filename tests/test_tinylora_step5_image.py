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
    assert wheel["sha256"] == (
        "1c455db7a1e5d58e9d43370ecfc75a3a2eb1e37ec12353db66d12c23dbbc3ac6"
    )
    assert wheel["url"].endswith(
        "/v2.8.3.post1/"
        "flash_attn-2.8.3.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    assert manifest["python_packages"]["flash-attn"] == "2.8.3.post1"
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
        "!write_sbom.py",
        "!third_party/",
        "!third_party/flash-attention-LICENSE",
    }


def test_image_carries_license_notice_and_sbom_generators() -> None:
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text()
    license_text = (
        IMAGE_ROOT / "third_party" / "flash-attention-LICENSE"
    ).read_text()

    assert license_text.startswith("BSD 3-Clause License")
    assert "/opt/tinylora/sbom/python-packages.spdx.json" in json.dumps(manifest)
    assert manifest["third_party_notices"]["flash_attention"]["sha256"] == (
        "8c9ccb96c065e706135b6cbad279b721da6156e51f3a5f27c6b3329af9416d73"
    )
    assert "write_sbom.py" in dockerfile
    assert "_GLIBCXX_USE_CXX11_ABI" in (IMAGE_ROOT / "validate_runtime.py").read_text()


def test_sbom_generator_emits_all_installed_distributions() -> None:
    path = IMAGE_ROOT / "write_sbom.py"
    spec = importlib.util.spec_from_file_location("tinylora_step5_sbom", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    packages = module.distributions()
    spdx = module.build_spdx(packages, "test-runtime")
    assert len(spdx["packages"]) == len(packages)
    assert all(package["filesAnalyzed"] is False for package in spdx["packages"])
    assert spdx["spdxVersion"] == "SPDX-2.3"
