#!/usr/bin/env python3
"""Build and verify the canonical QA v2 corpus from pinned local sources."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intelligent_liars.truth_editing_dataset_v2 import (  # noqa: E402
    DatasetV2Error,
    build_dataset_v2,
    install_direction_construction_receipt,
    load_candidates_from_config,
    load_hdf5_example_identities,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/truth_editing_dataset_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/truth_editing/v2")
    parser.add_argument(
        "--activation-provenance",
        type=Path,
        default=ROOT / "configs/activation_provenance_inventory_v1.json",
    )
    parser.add_argument(
        "--activation-hdf5",
        type=Path,
        default=ROOT / "artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    candidates, receipts = load_candidates_from_config(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="truth-editing-v2-", dir=args.output.parent) as temporary:
        stage = Path(temporary)
        dataset = build_dataset_v2(
            candidates,
            stage,
            seed=int(config["seed"]),
            dataset_id=str(config["dataset_id"]),
            source_receipts=receipts,
        )
        provenance_inventory = json.loads(args.activation_provenance.read_text())
        matching_entries = [
            entry for entry in provenance_inventory.get("entries", [])
            if entry.get("sidecar_id") == "activation-all-text-20260624"
        ]
        if len(matching_entries) != 1:
            raise DatasetV2Error("expected exactly one all-text activation provenance entry")
        activation_entry = matching_entries[0]
        artifact = activation_entry["artifact"]
        hdf5_identity = {
            "path": str(args.activation_hdf5.relative_to(ROOT)),
            "direct_sha256": artifact["direct_sha256"],
            "byte_size": artifact["byte_size"],
            "dvc_hash_algorithm": artifact["dvc_hash_algorithm"],
            "dvc_hash": artifact["dvc_hash"],
            "sidecar_id": activation_entry["sidecar_id"],
            "sidecar_self_sha256": activation_entry["self_sha256"],
            "evidence_status": activation_entry["evidence_status"],
        }
        dataset = install_direction_construction_receipt(
            dataset,
            hdf5_identity=hdf5_identity,
            hdf5_examples=load_hdf5_example_identities(args.activation_hdf5),
        )
        count = dataset.manifest.accepted_canonical_count
        minimum, maximum = int(config["target_minimum"]), int(config["target_maximum"])
        if not minimum <= count <= maximum:
            raise DatasetV2Error(
                f"accepted canonical count {count} outside configured [{minimum}, {maximum}]"
            )
        staged_names = {item.name for item in stage.iterdir()}
        if args.output.exists() and any(args.output.iterdir()):
            if not args.overwrite:
                raise DatasetV2Error("output directory is not empty; pass --overwrite")
            if {item.name for item in args.output.iterdir()} - staged_names:
                raise DatasetV2Error("output directory contains unexpected files")
        args.output.mkdir(parents=True, exist_ok=True)
        # Manifest is the commit marker and is replaced only after all files it
        # authenticates have reached the destination.
        for source in sorted(stage.iterdir(), key=lambda item: (item.name == "manifest.json", item.name)):
            source.replace(args.output / source.name)
        print(json.dumps(dataset.manifest.to_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
