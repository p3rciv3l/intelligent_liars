#!/usr/bin/env python3
"""Preserve the base image's Ninja ELF, license, and provenance receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path


DISTRIBUTION_VERSION = "1.11.1.1"
BINARY_VERSION = "1.11.1.git.kitware.jobserver-1"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/6d/92/"
    "8d7aebd4430ab5ff65df2bfee6d5745f95c004284db2d8ca76dcbfd9de47/"
    "ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl"
)
WHEEL_SHA256 = "84502ec98f02a037a169c4b0d5d86075eaf6afc55e1879003d6cab51ced2ea4b"
BINARY_SHA256 = "68f6c375c4234305bff9790aa232815b38924390cbb6ad4987ea0f94ad2bc410"
LICENSE_SHA256 = "73ba74dfaa520b49a401b5d21459a8523a146f3b7518a833eea5efa85130bf68"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_version(executable: Path) -> str:
    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def preserve_binary(
    source: Path,
    destination: Path,
    expected_version: str,
    expected_sha256: str,
) -> str:
    if source.read_bytes()[:4] != b"\x7fELF":
        raise ValueError(f"Ninja payload is not a standalone ELF: {source}")
    source_sha256 = sha256(source)
    if source_sha256 != expected_sha256:
        raise ValueError(
            f"Ninja ELF checksum mismatch: expected {expected_sha256}, got {source_sha256}"
        )
    actual = run_version(source)
    if actual != expected_version:
        raise ValueError(f"Ninja version mismatch: expected {expected_version}, got {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o755)
    if run_version(destination) != expected_version:
        raise ValueError("preserved Ninja ELF failed its post-copy version check")
    destination_sha256 = sha256(destination)
    if destination_sha256 != expected_sha256:
        raise ValueError("preserved Ninja ELF checksum changed during copy")
    return destination_sha256


def distribution_payload() -> tuple[Path, Path]:
    import ninja

    distribution = importlib.metadata.distribution("ninja")
    if distribution.version != DISTRIBUTION_VERSION:
        raise ValueError(
            f"expected ninja distribution {DISTRIBUTION_VERSION}, got {distribution.version}"
        )
    binary = Path(ninja.BIN_DIR) / "ninja"
    license_entry = next(
        (entry for entry in distribution.files or [] if entry.name == "LICENSE_Apache_20"),
        None,
    )
    if license_entry is None:
        raise ValueError("ninja distribution lacks LICENSE_Apache_20")
    return binary, Path(distribution.locate_file(license_entry))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_binary, source_license = distribution_payload()
    binary_sha256 = preserve_binary(
        source_binary,
        args.binary,
        BINARY_VERSION,
        BINARY_SHA256,
    )
    source_license_sha256 = sha256(source_license)
    if source_license_sha256 != LICENSE_SHA256:
        raise ValueError(
            f"Ninja license checksum mismatch: expected {LICENSE_SHA256}, "
            f"got {source_license_sha256}"
        )
    args.license.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_license, args.license)
    receipt = {
        "schema_version": 1,
        "name": "ninja",
        "distribution_version": DISTRIBUTION_VERSION,
        "binary_version": BINARY_VERSION,
        "binary_path": os.fspath(args.binary),
        "binary_sha256": binary_sha256,
        "license": "Apache-2.0",
        "license_path": os.fspath(args.license),
        "license_sha256": sha256(args.license),
        "source_artifact": WHEEL_URL,
        "source_artifact_sha256": WHEEL_SHA256,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
