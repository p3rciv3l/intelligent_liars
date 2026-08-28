#!/usr/bin/env python3
"""Build a frozen truth-editing preservation packet from stored base logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_preservation_materialization import (  # noqa: E402
    PreservationMaterializationError,
    materialize_preservation_runtime_packet,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Strict v1 materialization plan; all source paths are relative to this file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New packet directory; existing paths are never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        receipt = materialize_preservation_runtime_packet(args.plan, args.output_dir)
    except PreservationMaterializationError as error:
        parser.error(str(error))
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
