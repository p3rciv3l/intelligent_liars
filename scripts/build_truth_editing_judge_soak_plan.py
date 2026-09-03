#!/usr/bin/env python3
"""Build a non-scientific, identity-bound live judge transport-soak plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_judge_soak import (  # noqa: E402
    build_live_judge_soak_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_plan", type=Path)
    parser.add_argument("--planned-request-presentations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_live_judge_soak_plan(
        args.source_plan,
        planned_request_presentations=args.planned_request_presentations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        json.dump(plan, stream, allow_nan=False, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "planned_request_presentations": args.planned_request_presentations,
                "plan_sha256": plan["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
