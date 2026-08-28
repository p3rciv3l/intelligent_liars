#!/usr/bin/env python3
"""Extract the frozen Qwen refusal-direction bank with atomic batch recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.truth_editing_refusal_directions import (
    parse_refusal_direction_config,
    parse_refusal_prompt_manifest,
)
from intelligent_liars.truth_editing_refusal_extraction import (
    RefusalExtractionRunner,
    ResidualBackend,
    StoredResidualBackend,
    TransformersQwenResidualBackend,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/truth_editing_refusal_directions_v1.json"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("configs/truth_editing_refusal_prompt_manifest_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stored-residuals", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    args = parser.parse_args()

    config = parse_refusal_direction_config(json.loads(args.config.read_bytes()))
    prompts = parse_refusal_prompt_manifest(
        json.loads(args.prompts.read_bytes()), config
    )
    output_dir = args.output_dir or Path(config.output_root)
    if args.stored_residuals is not None:
        if args.cache_dir is not None or args.snapshot_manifest is not None:
            parser.error("stored replay cannot also configure a model cache")
        backend: ResidualBackend = StoredResidualBackend.from_path(
            args.stored_residuals
        )
    else:
        if args.cache_dir is None or args.snapshot_manifest is None:
            parser.error(
                "production extraction requires --cache-dir and --snapshot-manifest"
            )
        backend = TransformersQwenResidualBackend(
            config,
            cache_dir=args.cache_dir,
            snapshot_manifest_path=args.snapshot_manifest,
        )
    result = RefusalExtractionRunner(
        config,
        prompts,
        backend,
        output_dir,
        batch_size=args.batch_size,
    ).run()
    print(
        json.dumps(
            {
                "direction_bank_sha256": result.bank.self_sha256,
                "run_receipt_sha256": result.receipt.self_sha256,
                "completed_prompt_count": result.receipt.completed_prompt_count,
                "direction_count": len(result.bank.per_layer_receipts),
                "batch_count": result.receipt.batch_count,
                "resumed_batch_count": result.resumed_batch_count,
                "prompt_throughput": result.receipt.prompt_throughput,
                "token_throughput": result.receipt.token_throughput,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
