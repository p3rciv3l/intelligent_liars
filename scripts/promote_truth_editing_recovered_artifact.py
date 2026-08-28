#!/usr/bin/env python3
"""Validate and immutably promote one recovered refusal-direction bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.truth_editing_artifact_promotion import (
    ArtifactPromotionError,
    promote_recovered_refusal_bank,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a recovered refusal bank to its successful Vast lifecycle, "
            "remote archive hash, and exact extracted-tree inventory, then publish "
            "it atomically without overwriting any existing destination."
        )
    )
    parser.add_argument("--lifecycle-receipt", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--expected-output-archive-sha256", required=True)
    parser.add_argument("--extracted-outputs-dir", type=Path, required=True)
    parser.add_argument(
        "--refusal-config",
        type=Path,
        default=Path("configs/truth_editing_refusal_directions_v1.json"),
    )
    parser.add_argument(
        "--refusal-prompt-manifest",
        type=Path,
        default=Path("configs/truth_editing_refusal_prompt_manifest_v1.json"),
    )
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = promote_recovered_refusal_bank(
            lifecycle_receipt_path=args.lifecycle_receipt,
            output_archive_path=args.output_archive,
            expected_output_archive_sha256=args.expected_output_archive_sha256,
            extracted_outputs_dir=args.extracted_outputs_dir,
            refusal_config_path=args.refusal_config,
            refusal_prompt_manifest_path=args.refusal_prompt_manifest,
            destination=args.destination,
        )
    except ArtifactPromotionError as error:
        raise SystemExit(f"promotion refused: {error}") from error
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
