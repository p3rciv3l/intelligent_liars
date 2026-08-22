from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


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


def test_unsupported_base_ninja_metadata_is_removed_without_skipping_pip_check() -> None:
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text()
    ninja = manifest["system_tools"]["ninja"]

    assert ninja == {
        "version": "1.11.1.git.kitware.jobserver-1",
        "source": "digest-pinned-base-binary",
        "python_distribution_removed": "1.11.1.1",
        "binary_sha256": (
            "68f6c375c4234305bff9790aa232815b38924390cbb6ad4987ea0f94ad2bc410"
        ),
        "license_sha256": (
            "73ba74dfaa520b49a401b5d21459a8523a146f3b7518a833eea5efa85130bf68"
        ),
        "license_path": "/opt/tinylora/third_party/ninja-LICENSE_Apache_20",
        "provenance_receipt": "/opt/tinylora/sbom/ninja-system-package.json",
        "source_artifact": (
            "https://files.pythonhosted.org/packages/6d/92/"
            "8d7aebd4430ab5ff65df2bfee6d5745f95c004284db2d8ca76dcbfd9de47/"
            "ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64."
            "manylinux_2_5_x86_64.whl"
        ),
        "source_artifact_sha256": (
            "84502ec98f02a037a169c4b0d5d86075eaf6afc55e1879003d6cab51ced2ea4b"
        ),
        "reason": "pip check reports the base distribution unsupported on linux/amd64",
    }
    preserve = dockerfile.index("python /opt/tinylora/preserve_base_ninja.py")
    remove_metadata = dockerfile.index("python -m pip uninstall --yes ninja")
    dependency_check = dockerfile.index("python -m pip check")
    assert preserve < remove_metadata < dependency_check
    assert "pip check --" not in dockerfile
    assert "PIP_CHECK" not in dockerfile
    assert "command -v ninja" not in dockerfile


def test_system_tool_contract_rejects_reinstalled_ninja_metadata(monkeypatch) -> None:
    validator = _validator_module()
    manifest = {
        "system_tools": {
            "ninja": {
                "version": "1.11.1.git.kitware.jobserver-1",
                "python_distribution_removed": "1.11.1.1",
            }
        }
    }
    monkeypatch.setattr(validator, "_installed_version", lambda _name: "1.11.1.1")
    monkeypatch.setattr(validator.shutil, "which", lambda _name: "/usr/local/bin/ninja")
    monkeypatch.setattr(
        validator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="1.11.1.git.kitware.jobserver-1\n",
        ),
    )

    assert validator.validate_system_tools(manifest) == [
        "ninja Python distribution metadata must be absent; found 1.11.1.1"
    ]


def test_system_tool_contract_rejects_changed_binary(tmp_path: Path, monkeypatch) -> None:
    validator = _validator_module()
    executable = tmp_path / "ninja"
    executable.write_bytes(b"changed-binary")
    manifest = {
        "system_tools": {
            "ninja": {
                "version": "1.11.1.git.kitware.jobserver-1",
                "binary_sha256": "0" * 64,
            }
        }
    }
    monkeypatch.setattr(validator.shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(
        validator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="1.11.1.git.kitware.jobserver-1\n",
        ),
    )

    assert validator.validate_system_tools(manifest) == [
        "ninja: executable checksum does not match runtime contract"
    ]


def test_runtime_identity_bumps_for_ninja_environment_change() -> None:
    manifest = json.loads((IMAGE_ROOT / "runtime-manifest.json").read_text())
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text()

    assert manifest["runtime_id"].endswith("-r3")
    assert manifest["runtime_id"] in dockerfile


def test_ninja_preserver_copies_standalone_payload_not_console_wrapper(
    tmp_path: Path, monkeypatch
) -> None:
    path = IMAGE_ROOT / "preserve_base_ninja.py"
    spec = importlib.util.spec_from_file_location("tinylora_preserve_ninja", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / "ninja" / "data" / "bin" / "ninja"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"\x7fELFstandalone-ninja-payload")
    source.chmod(0o755)
    destination = tmp_path / "usr" / "local" / "bin" / "ninja"
    monkeypatch.setattr(module, "run_version", lambda _path: module.BINARY_VERSION)

    source_sha = module.sha256(source)
    copied_sha = module.preserve_binary(
        source,
        destination,
        module.BINARY_VERSION,
        source_sha,
    )
    source.unlink()

    assert destination.read_bytes() == b"\x7fELFstandalone-ninja-payload"
    assert destination.stat().st_mode & 0o111
    assert copied_sha == module.sha256(destination)


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
        "!preserve_base_ninja.py",
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
    assert manifest["third_party_notices"]["ninja"]["license"] == "Apache-2.0"
    assert "write_sbom.py" in dockerfile
    assert "_GLIBCXX_USE_CXX11_ABI" in (IMAGE_ROOT / "validate_runtime.py").read_text()


def test_sbom_generator_emits_installed_and_preserved_system_packages(
    tmp_path: Path,
) -> None:
    path = IMAGE_ROOT / "write_sbom.py"
    spec = importlib.util.spec_from_file_location("tinylora_step5_sbom", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    receipt = tmp_path / "ninja-system-package.json"
    receipt.write_text(
        json.dumps(
            {
                "name": "ninja",
                "distribution_version": "1.11.1.1",
                "binary_version": "1.11.1.git.kitware.jobserver-1",
                "binary_sha256": "a" * 64,
                "license": "Apache-2.0",
                "source_artifact": "https://example.invalid/ninja.whl",
                "source_artifact_sha256": "b" * 64,
            }
        )
    )
    packages = module.distributions([receipt])
    spdx = module.build_spdx(packages, "test-runtime")
    assert len(spdx["packages"]) == len(packages)
    assert all(package["filesAnalyzed"] is False for package in spdx["packages"])
    assert spdx["spdxVersion"] == "SPDX-2.3"
    ninja = next(package for package in spdx["packages"] if package["name"] == "ninja")
    assert ninja["licenseDeclared"] == "Apache-2.0"
    assert ninja["checksums"] == [{"algorithm": "SHA256", "checksumValue": "b" * 64}]
