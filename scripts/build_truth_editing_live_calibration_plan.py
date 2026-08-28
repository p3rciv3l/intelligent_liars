#!/usr/bin/env python3
"""Compile a blinded, hash-bound revised-pack live calibration plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_live_calibration_plan import compile_live_calibration_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revised-pack", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--maximum-spend-usd", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = compile_live_calibration_plan(args.revised_pack, args.labels, args.provenance, maximum_spend_usd=args.maximum_spend_usd)
    rendered = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(rendered)
    print(json.dumps({"plan_sha256": plan["content_sha256"], "absolute_bundles": len(plan["absolute_bundles"]), "pairwise_presentations": sum(len(value["presentations"]) for value in plan["pairwise_relationships"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
