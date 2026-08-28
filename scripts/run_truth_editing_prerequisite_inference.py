#!/usr/bin/env python3
"""Run Qwen prerequisites with the narrowest production-verified load topology.

QA and structured qualification share one load. Refusal extraction uses its own
sealed loader because production evidence forbids injecting a cross-lane bundle.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from intelligent_liars.models import ModelLoadConfig
from intelligent_liars.truth_editing_base_known import (
    BaseKnownRunner,
    QualificationConfig,
)
from intelligent_liars.truth_editing_qwen_qualification import (
    FrozenQwenQualificationBackend,
)
from intelligent_liars.truth_editing_refusal_directions import (
    parse_refusal_direction_config,
    parse_refusal_prompt_manifest,
)
from intelligent_liars.truth_editing_refusal_extraction import (
    RefusalExtractionRunner,
    TransformersQwenResidualBackend,
)
from intelligent_liars.truth_editing_structured_qualification import (
    StructuredSemanticQualificationRunner,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--structured-view", type=Path, required=True)
    # Kept in the job contract so the source derivation is shipped and available
    # even though the current verified view opens without a source-root argument.
    parser.add_argument("--structured-source-root", type=Path, required=True)
    parser.add_argument("--refusal-config", type=Path, required=True)
    parser.add_argument("--refusal-prompts", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refusal-batch-size", type=int, default=8)
    return parser


def main() -> int:
    args = _parser().parse_args()
    base_raw = json.loads(args.base_config.read_text())
    if set(base_raw) != {"format", "model", "qualification"}:
        raise SystemExit("base-known config fields changed")
    model_config = ModelLoadConfig(
        cache_dir=str(args.cache_dir),
        snapshot_manifest_path=str(args.snapshot_manifest),
        expected_model_sha256=base_raw["model"]["model_sha256"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    execution_receipt = args.output_dir / "frozen-qwen-execution-receipt.json"
    qualification_backend = FrozenQwenQualificationBackend(
        model_config=model_config,
        execution_receipt_path=execution_receipt,
        enforce_production_runtime=True,
    )
    qualification = QualificationConfig(**base_raw["qualification"])
    base = BaseKnownRunner(
        args.dataset_dir,
        args.output_dir / "base-known",
        base_raw["model"],
        qualification,
        qualification_backend,
    ).run()
    structured = StructuredSemanticQualificationRunner(
        args.structured_view,
        args.structured_source_root,
        args.output_dir / "structured-base-known",
        base_raw["model"],
        qualification,
        qualification_backend,
    ).run()

    # The refusal evidence must come from a separately sealed loader, but a
    # 24 GiB device cannot hold both 8B model copies simultaneously.  End the
    # completed qualification phase and prove its CUDA allocation is released
    # before constructing the refusal backend.
    del qualification_backend
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    residual_cuda_bytes = int(torch.cuda.memory_allocated())
    if residual_cuda_bytes > 256 * 1024 * 1024:
        raise RuntimeError(
            "qualification model CUDA allocation was not released: "
            f"{residual_cuda_bytes} bytes remain"
        )

    refusal_config = parse_refusal_direction_config(
        json.loads(args.refusal_config.read_text())
    )
    refusal_prompts = parse_refusal_prompt_manifest(
        json.loads(args.refusal_prompts.read_text()), refusal_config
    )
    refusal_backend = TransformersQwenResidualBackend(
        refusal_config,
        cache_dir=args.cache_dir,
        snapshot_manifest_path=args.snapshot_manifest,
    )
    refusal = RefusalExtractionRunner(
        refusal_config,
        refusal_prompts,
        refusal_backend,
        args.output_dir / "refusal",
        batch_size=args.refusal_batch_size,
    ).run()
    print(
        json.dumps(
            {
                "model_load_count": 2,
                "model_load_note": (
                    "qualification and refusal use separate verified production loaders; "
                    "QA and structured qualification share one load"
                ),
                "base_manifest_sha256": base.manifest_sha256,
                "base_qualified_count": len(base.qualified_record_ids),
                "structured_manifest_sha256": structured.manifest_sha256,
                "refusal_bank_sha256": refusal.bank.self_sha256,
                "refusal_receipt_sha256": refusal.receipt.self_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
