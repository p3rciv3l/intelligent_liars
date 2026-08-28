#!/usr/bin/env python3
"""Materialize fit/validation OSWorld preservation-only KL inputs from S3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_osworld_preservation_source import (  # noqa: E402
    OSWorldPreservationSourceError,
    materialize_osworld_preservation_source,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = materialize_osworld_preservation_source(args.config, args.output_dir)
    except OSWorldPreservationSourceError as error:
        parser.error(str(error))
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
