#!/usr/bin/env python3
"""Run QA and structured base-known qualification with one shared backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from intelligent_liars.models import ModelLoadConfig
from intelligent_liars.truth_editing_base_known import (
    BaseKnownError,
    BaseKnownRunner,
    QualificationBackend,
    QualificationConfig,
    StoredResponseBackend,
)
from intelligent_liars.truth_editing_qwen_qualification import (
    FrozenQwenQualificationBackend,
)
from intelligent_liars.truth_editing_structured_qualification import (
    StructuredSemanticQualificationRunner,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/truth_editing_base_known_v1.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/truth_editing/v2"))
    parser.add_argument("--base-output-dir", type=Path, required=True)
    parser.add_argument("--structured-view", type=Path)
    parser.add_argument("--structured-source-root", type=Path)
    parser.add_argument("--structured-output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--responses", type=Path, help="stored-response replay; not production evidence")
    mode.add_argument("--production-qwen", action="store_true", help="verified pinned CUDA BF16 FlashAttention 2 backend")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--model-cache-manifest", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return _run(args)
    except (BaseKnownError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"truth-editing qualification failed: {error}") from error


def _run(args: argparse.Namespace) -> int:
    structured_values = (
        args.structured_view,
        args.structured_source_root,
        args.structured_output_dir,
    )
    if any(value is not None for value in structured_values) and not all(
        value is not None for value in structured_values
    ):
        raise SystemExit("--structured-view, --structured-source-root, and --structured-output-dir must be supplied together")
    raw = json.loads(args.config.read_text())
    if set(raw) != {"format", "model", "qualification"} or raw["format"] != "truth_editing_base_known_config_v1":
        raise SystemExit("unsupported or non-canonical base-known config")
    qualification = QualificationConfig(**raw["qualification"])
    if args.production_qwen:
        cache_dir = args.model_cache_dir or _environment_path("HF_HOME")
        manifest = args.model_cache_manifest or _environment_path("TRUTH_EDITING_MODEL_CACHE_MANIFEST")
        if cache_dir is None or manifest is None:
            raise SystemExit("production Qwen requires --model-cache-dir/HF_HOME and --model-cache-manifest/TRUTH_EDITING_MODEL_CACHE_MANIFEST")
        receipt = args.execution_receipt or args.base_output_dir.parent / "frozen-qwen-execution-receipt.json"
        backend: QualificationBackend = FrozenQwenQualificationBackend(
            model_config=ModelLoadConfig(
                cache_dir=str(cache_dir),
                snapshot_manifest_path=str(manifest),
                expected_model_sha256=raw["model"]["model_sha256"],
            ),
            execution_receipt_path=receipt,
        )
    else:
        assert args.responses is not None
        backend = StoredResponseBackend(args.responses)

    base = BaseKnownRunner(
        args.dataset_dir, args.base_output_dir, raw["model"], qualification, backend
    ).run()
    structured = None
    if args.structured_view is not None:
        assert args.structured_source_root is not None
        assert args.structured_output_dir is not None
        structured = StructuredSemanticQualificationRunner(
            args.structured_view,
            args.structured_source_root,
            args.structured_output_dir,
            raw["model"],
            qualification,
            backend,
        ).run()
    print(json.dumps({
        "base_manifest_sha256": base.manifest_sha256,
        "base_qualified_count": len(base.qualified_record_ids),
        "structured_manifest_sha256": structured.manifest_sha256 if structured else None,
        "structured_known_scenario_count": (
            sum(row.all_required_known for row in structured.scenarios) if structured else None
        ),
    }, sort_keys=True))
    return 0


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


if __name__ == "__main__":
    raise SystemExit(main())
