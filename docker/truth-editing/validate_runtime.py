#!/usr/bin/env python3
"""Validate the pinned truth-editing study image and its CUDA runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path(
    os.environ.get("TRUTH_EDITING_RUNTIME_MANIFEST", ROOT / "runtime-manifest.json")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("runtime manifest must be a JSON object")
    return data


def validate_metadata(manifest: dict[str, Any], root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    base = manifest.get("base_image", {})
    reference = str(base.get("reference", ""))
    digest = str(base.get("digest", ""))
    if not digest.startswith("sha256:") or reference != (
        f"{base.get('repository')}:{base.get('tag')}@{digest}"
    ):
        errors.append("base image reference is not internally digest-pinned")

    dockerfile = (root / "Dockerfile").read_text()
    if reference not in dockerfile:
        errors.append("Dockerfile base image does not match runtime manifest")

    for filename, expected in manifest.get("build_inputs", {}).items():
        path = root / filename
        if not path.is_file():
            errors.append(f"missing build input: {filename}")
        elif sha256(path) != expected:
            errors.append(f"build input checksum mismatch: {filename}")

    lock = (root / "requirements.lock").read_text()
    for package, version in manifest.get("python_packages", {}).items():
        if package in {"torch", "flash-attn"}:
            continue
        if f"{package}=={version}" not in lock:
            errors.append(f"package pin absent from requirements.lock: {package}=={version}")

    for forbidden in ("torch==", "triton==", "nvidia-"):
        if any(line.startswith(forbidden) for line in lock.splitlines()):
            errors.append(f"base-supplied package leaked into lock: {forbidden}")

    if manifest.get("available_optuna_storage_api") != {
        "storage": "JournalStorage",
        "backend": "JournalFileBackend",
    }:
        errors.append("Optuna JournalStorage/JournalFileBackend API must be available")

    wheel = manifest.get("flash_attention_wheel", {})
    for expected in (wheel.get("filename"), wheel.get("url"), wheel.get("sha256")):
        if not expected or str(expected) not in dockerfile:
            errors.append(f"FlashAttention build pin absent from Dockerfile: {expected}")
    return errors


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_installed_packages(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for package, expected in manifest["python_packages"].items():
        actual = _installed_version(package)
        if actual != expected:
            errors.append(f"{package}: expected {expected}, got {actual or 'not installed'}")
    expected_python = manifest["platform"]["python_major_minor"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        errors.append(f"python: expected {expected_python}, got {actual_python}")
    if platform.machine() not in {"x86_64", "AMD64"}:
        errors.append(f"architecture: expected x86_64, got {platform.machine()}")

    try:
        import torch

        if torch._C._GLIBCXX_USE_CXX11_ABI is not False:
            errors.append("torch CXX11 ABI must be false for the FlashAttention wheel")
    except (ImportError, AttributeError) as exc:
        errors.append(f"cannot verify torch CXX11 ABI: {exc}")

    try:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        if JournalStorage is None or JournalFileBackend is None:
            errors.append("Optuna journal classes are unavailable")
    except ImportError as exc:
        errors.append(f"Optuna journal import failed: {exc}")

    try:
        from nnsight import VisionLanguageModel
        from transformers import Qwen3VLForConditionalGeneration

        if not callable(getattr(Qwen3VLForConditionalGeneration, "from_pretrained", None)):
            errors.append("Qwen3-VL Transformers loader is unavailable")
        if getattr(Qwen3VLForConditionalGeneration, "_supports_flash_attn", False) is not True:
            errors.append("Qwen3-VL does not declare FlashAttention support")
        if VisionLanguageModel is None:
            errors.append("NNsight VisionLanguageModel is unavailable")
    except ImportError as exc:
        errors.append(f"Qwen3-VL/NNsight import failed: {exc}")
    return errors


def validate_offhost_checkpoint_runtime() -> list[str]:
    """Prove the production coordinator can construct its S3 adapter locally."""

    try:
        import boto3
        from botocore.client import BaseClient

        # Explicit non-secret placeholders keep client construction wholly local:
        # boto3 cannot query the instance metadata service for credentials here.
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="runtime-import-check",
            aws_secret_access_key="runtime-import-check",
        )
        if not isinstance(client, BaseClient) or client.meta.service_model.service_name != "s3":
            return ["boto3 did not construct an S3 client"]
    except Exception as exc:
        return [f"boto3/botocore S3 client construction failed: {exc}"]
    return []


def _validate_writable_root(root: Path, *, minimum_free_gib: float) -> list[str]:
    try:
        root.mkdir(parents=True, exist_ok=True)
        source = root / ".truth-editing-write-test.pending"
        destination = root / ".truth-editing-write-test.committed"
        source.write_text("ok\n")
        os.replace(source, destination)
        destination.unlink()
    except OSError as exc:
        return [f"runtime root is not writable: {root}: {exc}"]
    free_gib = shutil.disk_usage(root).free / 1024**3
    if free_gib < minimum_free_gib:
        return [
            f"cache free space {free_gib:.1f} GiB is below "
            f"{minimum_free_gib:.1f} GiB for {root}"
        ]
    return []


def validate_cache(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    policy = manifest["cache_policy"]
    minimum_free_gib = float(policy["minimum_free_gib"])
    cache_root = Path(os.environ.get("TRUTH_EDITING_CACHE_ROOT", policy["root"]))
    study_root = Path(os.environ.get("TRUTH_EDITING_STUDY_ROOT", policy["study_root"]))
    errors.extend(_validate_writable_root(cache_root, minimum_free_gib=minimum_free_gib))
    if study_root != cache_root:
        errors.extend(_validate_writable_root(study_root, minimum_free_gib=minimum_free_gib))
    return errors


def validate_gpu(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        import torch
    except ImportError as exc:
        return [f"torch import failed: {exc}"]

    if not torch.cuda.is_available():
        return ["CUDA is not available"]
    if torch.cuda.device_count() < 1:
        return ["CUDA reports no visible devices"]
    if torch.version.cuda != manifest["platform"]["cuda"]:
        errors.append(
            f"torch CUDA: expected {manifest['platform']['cuda']}, got {torch.version.cuda}"
        )
    try:
        from flash_attn import flash_attn_func
        from transformers.utils import is_flash_attn_2_available
    except Exception as exc:
        return [*errors, f"FlashAttention import failed: {exc}"]
    if not is_flash_attn_2_available():
        errors.append("Transformers does not recognize FlashAttention 2 as available")

    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        props = torch.cuda.get_device_properties(device)
        capability = (props.major, props.minor)
        minimum = tuple(manifest["platform"]["minimum_compute_capability"])
        if capability < minimum:
            errors.append(f"GPU {index} compute capability {capability} is below {minimum}")
        memory_gib = props.total_memory / 1024**3
        if memory_gib < float(manifest["platform"]["minimum_gpu_memory_gib"]):
            errors.append(f"GPU {index} memory {memory_gib:.1f} GiB is below required minimum")

        try:
            left = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
            torch.mm(left, left)
            torch.cuda.synchronize(device)
        except Exception as exc:
            errors.append(f"GPU {index} BF16 CUDA matmul failed: {exc}")

        try:
            query = torch.randn((1, 128, 8, 64), device=device, dtype=torch.bfloat16)
            output = flash_attn_func(query, query, query, dropout_p=0.0, causal=True)
            if output.shape != query.shape or not torch.isfinite(output).all():
                errors.append(f"GPU {index} FlashAttention returned an invalid result")
            torch.cuda.synchronize(device)
        except Exception as exc:
            errors.append(f"GPU {index} FlashAttention CUDA kernel failed: {exc}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    errors = validate_metadata(manifest, args.manifest.parent)
    errors.extend(validate_installed_packages(manifest))
    errors.extend(validate_offhost_checkpoint_runtime())
    if not args.metadata_only:
        errors.extend(validate_cache(manifest))
        errors.extend(validate_gpu(manifest))
    result = {
        "valid": not errors,
        "mode": "metadata-only" if args.metadata_only else "gpu-runtime",
        "runtime_id": manifest.get("runtime_id"),
        "manifest_sha256": sha256(args.manifest),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
