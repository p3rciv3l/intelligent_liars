#!/usr/bin/env python3
"""Capture frozen-base preservation logits from the verified local Qwen cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.models import ModelLoadConfig  # noqa: E402
from intelligent_liars.truth_editing_preservation_capture import (  # noqa: E402
    PreservationBaselineCaptureError,
    capture_preservation_baselines,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Strict v1 capture plan with source paths relative to the plan",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New immutable capture directory; existing paths are refused",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        required=True,
        help="Local cache containing the exact pinned Qwen snapshot",
    )
    parser.add_argument(
        "--model-cache-manifest",
        type=Path,
        required=True,
        help="Verified local snapshot manifest for the model cache",
    )
    args = parser.parse_args(argv)
    config = ModelLoadConfig(
        cache_dir=str(args.model_cache),
        snapshot_manifest_path=str(args.model_cache_manifest),
    )
    try:
        receipt = capture_preservation_baselines(
            args.plan,
            args.output_dir,
            model_config=config,
        )
    except PreservationBaselineCaptureError as error:
        parser.error(str(error))
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
