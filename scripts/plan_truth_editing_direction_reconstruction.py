#!/usr/bin/env python3
"""Emit the exact clean-construction workload needed to qualify direction knobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_contracts import (  # noqa: E402
    canonical_json_bytes,
    parse_direction_bank_manifest,
)
from intelligent_liars.truth_editing_directions import (  # noqa: E402
    build_reconstruction_workload,
)


ACTIVATION_INPUT = {
    "path": "artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5",
    "byte_size": 62394047728,
    "direct_sha256": "c6e687f69256544121d0718cf8bba142ed5837221976121e1899552dc76f1a5a",
    "dvc_md5": "b6e82b698513d2372949f2752e17005a",
    "evidence_status": "verified_metadata",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "configs/truth_editing_direction_bank_v1.json",
    )
    parser.add_argument("--construction-row-allowlist-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs/truth_editing_direction_reconstruction_v1.json",
    )
    args = parser.parse_args()
    manifest = parse_direction_bank_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    workload = build_reconstruction_workload(
        manifest,
        activation_input=ACTIVATION_INPUT,
        construction_row_allowlist_sha256=args.construction_row_allowlist_sha256,
        output_root="artifacts/directions/reconstructed-v1",
        maximum_external_spend_usd=15.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(workload) + b"\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "target_cells": workload["target_cell_count"],
                "refit_cells": workload["refit_cell_count"],
                "blocked_cells": workload["blocked_cell_count"],
                "self_sha256": workload["self_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
