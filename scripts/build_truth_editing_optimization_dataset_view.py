#!/usr/bin/env python3
"""Build the sealed-test-free dataset view used by optimization jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intelligent_liars.truth_editing_dataset_v2 import (  # noqa: E402
    materialize_optimization_dataset_view,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "datasets/truth_editing/v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/truth_editing/v2_optimization_v1",
    )
    args = parser.parse_args(argv)
    dataset = materialize_optimization_dataset_view(args.source, args.output)
    print(
        json.dumps(
            {
                "dataset_id": dataset.manifest.dataset_id,
                "accessible_splits": list(dataset.accessible_splits),
                "record_count": len(dataset.records),
                "audit_valid": dataset.audit().valid,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
