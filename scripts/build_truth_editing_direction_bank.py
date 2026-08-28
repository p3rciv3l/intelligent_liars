#!/usr/bin/env python3
"""Build a strict direction-bank manifest and compact coverage receipt."""

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
    canonical_sha256,
)
from intelligent_liars.truth_editing_directions import build_direction_bank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/truth_editing_direction_sources_v1.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPOSITORY_ROOT / "configs/truth_editing_direction_bank_v1.json",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=REPOSITORY_ROOT / "configs/truth_editing_direction_coverage_v1.json",
    )
    args = parser.parse_args()
    result = build_direction_bank(args.config, root=REPOSITORY_ROOT)
    coverage = {
        "format": "truth_editing_direction_coverage_v1",
        "manifest_sha256": result.manifest.self_sha256,
        "coverage": result.coverage.to_dict(),
    }
    coverage["self_sha256"] = canonical_sha256(coverage)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_bytes(
        canonical_json_bytes(result.manifest.to_dict()) + b"\n"
    )
    args.coverage_output.write_bytes(canonical_json_bytes(coverage) + b"\n")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_output),
                "manifest_sha256": result.manifest.self_sha256,
                "coverage": str(args.coverage_output),
                "direction_count": result.coverage.total,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
