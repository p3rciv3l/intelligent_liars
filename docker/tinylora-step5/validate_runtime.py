#!/usr/bin/env python3
"""Validate the pinned TinyLoRA Step 5 image metadata and GPU runtime."""

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
    os.environ.get("TINYLORA_RUNTIME_MANIFEST", ROOT / "runtime-manifest.json")
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
    if manifest.get("large_run_enabled") is not False:
        errors.append("large_run_enabled must remain false")

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
    return errors


def validate_cache(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    policy = manifest["cache_policy"]
    root = Path(os.environ.get("TINYLORA_CACHE_ROOT", policy["root"]))
    try:
        root.mkdir(parents=True, exist_ok=True)
        sentinel = root / ".tinylora-write-test"
        sentinel.write_text("ok\n")
        sentinel.unlink()
    except OSError as exc:
        errors.append(f"cache root is not writable: {root}: {exc}")
        return errors
    free_gib = shutil.disk_usage(root).free / 1024**3
    if free_gib < float(policy["minimum_free_gib"]):
        errors.append(
            f"cache free space {free_gib:.1f} GiB is below "
            f"{policy['minimum_free_gib']:.1f} GiB"
        )
    return errors


def validate_gpu(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        import torch
    except ImportError as exc:
        return [f"torch import failed: {exc}"]

    if not torch.cuda.is_available():
        return ["CUDA is not available"]
    if torch.version.cuda != manifest["platform"]["cuda"]:
        errors.append(
            f"torch CUDA: expected {manifest['platform']['cuda']}, got {torch.version.cuda}"
        )
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    capability = (props.major, props.minor)
    minimum = tuple(manifest["platform"]["minimum_compute_capability"])
    if capability < minimum:
        errors.append(f"compute capability {capability} is below {minimum}")
    memory_gib = props.total_memory / 1024**3
    if memory_gib < float(manifest["platform"]["minimum_gpu_memory_gib"]):
        errors.append(f"GPU memory {memory_gib:.1f} GiB is below required minimum")

    try:
        left = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
        torch.mm(left, left)
        torch.cuda.synchronize()
    except Exception as exc:  # CUDA failures have runtime-specific exception classes.
        errors.append(f"BF16 CUDA matmul failed: {exc}")

    try:
        from flash_attn import flash_attn_func

        query = torch.randn((1, 128, 8, 64), device=device, dtype=torch.bfloat16)
        output = flash_attn_func(query, query, query, dropout_p=0.0, causal=True)
        if output.shape != query.shape or not torch.isfinite(output).all():
            errors.append("FlashAttention returned an invalid result")
        torch.cuda.synchronize()
    except Exception as exc:
        errors.append(f"FlashAttention CUDA kernel failed: {exc}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="validate build metadata and installed pins without cache or GPU checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    errors = validate_metadata(manifest, args.manifest.parent)
    errors.extend(validate_installed_packages(manifest))
    if not args.metadata_only:
        errors.extend(validate_cache(manifest))
        errors.extend(validate_gpu(manifest))
    result = {
        "valid": not errors,
        "mode": "metadata-only" if args.metadata_only else "gpu-runtime",
        "runtime_id": manifest.get("runtime_id"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
