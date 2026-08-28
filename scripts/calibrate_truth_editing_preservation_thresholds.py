#!/usr/bin/env python3
"""Build replay-verifiable preservation KL ceilings from base repeats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.models import DEFAULT_MODEL_CONTENT_SHA256  # noqa: E402
from intelligent_liars.truth_editing_preservation_thresholds import (  # noqa: E402
    build_preservation_threshold_calibration,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument(
        "--base-model-sha256", default=DEFAULT_MODEL_CONTENT_SHA256
    )
    parser.add_argument("--minimum-repeats", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--absolute-margin", type=float, default=0.0001)
    parser.add_argument("--relative-margin", type=float, default=0.25)
    args = parser.parse_args(argv)

    receipt_paths = sorted(args.receipt_root.glob("**/*.json"))
    calibration = build_preservation_threshold_calibration(
        args.output,
        calibration_id=args.calibration_id,
        base_model_sha256=args.base_model_sha256,
        receipt_paths=receipt_paths,
        minimum_repeats=args.minimum_repeats,
        quantile=args.quantile,
        absolute_margin=args.absolute_margin,
        relative_margin=args.relative_margin,
    )
    print(calibration.self_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
