#!/usr/bin/env python3
"""Print an offline wall-clock and canary plan for the truth-editing run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.truth_editing_runtime import (  # noqa: E402
    ThroughputBenchmark,
    load_runtime_plan,
    plan_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/truth_editing_runtime_v1.json"),
    )
    parser.add_argument("--tokens-per-second", type=float, default=149.36)
    parser.add_argument("--p90-slowdown", type=float, default=1.20)
    parser.add_argument("--model-load-seconds", type=float, default=180.0)
    parser.add_argument("--batch-overhead-seconds", type=float, default=2.0)
    parser.add_argument("--gpu-count", type=int, default=8)
    args = parser.parse_args()
    plan = load_runtime_plan(args.config)
    report = plan_report(
        plan,
        ThroughputBenchmark(
            tokens_per_second=args.tokens_per_second,
            p90_slowdown=args.p90_slowdown,
            model_load_seconds=args.model_load_seconds,
            batch_overhead_seconds=args.batch_overhead_seconds,
            gpu_count=args.gpu_count,
        ),
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
