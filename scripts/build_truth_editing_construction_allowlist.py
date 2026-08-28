#!/usr/bin/env python3
"""Materialize the strict all-domain direction-construction selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intelligent_liars.truth_editing_construction_allowlist import (  # noqa: E402
    build_refitter_construction_allowlist,
    load_hdf5_construction_metadata,
    write_construction_allowlist_build,
)
from intelligent_liars.truth_editing_dataset_v2 import (  # noqa: E402
    DatasetV2Error,
    TruthEditingDatasetV2,
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_activation_entry(inventory_path: Path, hdf5_path: Path) -> dict:
    inventory = json.loads(inventory_path.read_text())
    inventory_unsigned = dict(inventory)
    inventory_claim = inventory_unsigned.pop("self_sha256", None)
    if inventory_claim != _canonical_hash(inventory_unsigned):
        raise DatasetV2Error("activation provenance inventory self hash mismatch")
    matches = [
        entry
        for entry in inventory.get("entries", [])
        if entry.get("sidecar_id") == "activation-all-text-20260624"
    ]
    if len(matches) != 1:
        raise DatasetV2Error("expected one all-text activation provenance sidecar")
    entry = matches[0]
    unsigned = dict(entry)
    claim = unsigned.pop("self_sha256", None)
    if claim != _canonical_hash(unsigned):
        raise DatasetV2Error("activation provenance sidecar self hash mismatch")
    artifact = entry.get("artifact", {})
    expected_path = (ROOT / str(artifact.get("path"))).resolve()
    if expected_path != hdf5_path.resolve():
        raise DatasetV2Error("activation path differs from signed provenance sidecar")
    if hdf5_path.is_symlink() or not hdf5_path.is_file():
        raise DatasetV2Error("activation HDF5 is not a regular file")
    if hdf5_path.stat().st_size != artifact.get("byte_size"):
        raise DatasetV2Error("activation size differs from signed provenance sidecar")
    if artifact.get("direct_hash_evidence") != "historical_validation_receipt":
        raise DatasetV2Error("activation sidecar lacks direct-hash verification evidence")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/truth_editing/v2")
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=ROOT / "artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5",
    )
    parser.add_argument(
        "--activation-provenance",
        type=Path,
        default=ROOT / "configs/activation_provenance_inventory_v1.json",
    )
    parser.add_argument(
        "--refit-config",
        type=Path,
        default=ROOT / "configs/truth_editing_direction_refit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/truth_editing/direction_construction_allowlist_v1.json",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / "datasets/truth_editing/direction_construction_allowlist_audit_v1.json",
    )
    parser.add_argument("--minimum-per-class", type=int, default=50)
    parser.add_argument("--maximum-per-class", type=int, default=1000)
    parser.add_argument("--verify-large-file", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    entry = _verified_activation_entry(args.activation_provenance, args.hdf5)
    artifact = entry["artifact"]
    if args.verify_large_file:
        observed = _file_hash(args.hdf5)
        if observed != artifact["direct_sha256"]:
            raise DatasetV2Error("activation direct SHA-256 verification failed")
    dataset = TruthEditingDatasetV2.open(args.dataset)
    refit_config = json.loads(args.refit_config.read_text())
    build = build_refitter_construction_allowlist(
        dataset,
        load_hdf5_construction_metadata(args.hdf5),
        activation_direct_sha256=artifact["direct_sha256"],
        required_domains=tuple(refit_config["domains"]),
        minimum_per_class=args.minimum_per_class,
        maximum_per_class=args.maximum_per_class,
    )
    allowlist_hash, audit_hash = write_construction_allowlist_build(
        build,
        allowlist_path=args.output,
        audit_path=args.audit_output,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "status": build.audit["status"],
        "allowlist_path": str(args.output),
        "allowlist_file_sha256": allowlist_hash,
        "allowlist_self_sha256": build.audit["allowlist_self_sha256"],
        "audit_path": str(args.audit_output),
        "audit_file_sha256": audit_hash,
        "missing_cells": build.audit["missing_cells"],
    }, indent=2, sort_keys=True))
    return 0 if build.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
