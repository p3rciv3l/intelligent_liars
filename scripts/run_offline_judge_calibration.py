#!/usr/bin/env python3
"""Calibrate stored judge responses against frozen human labels offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.offline_judge_calibration import (  # noqa: E402
    calibrate_offline_judge_fixture,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="Stored calibration fixture JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a new report file instead of printing JSON to stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = calibrate_offline_judge_fixture(args.fixture)
    serialized = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    if args.output is None:
        sys.stdout.write(serialized)
    else:
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            stream.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
