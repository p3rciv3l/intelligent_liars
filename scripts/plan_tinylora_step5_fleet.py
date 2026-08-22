#!/usr/bin/env python3
"""Produce a deterministic dry-run plan for the bounded Step 5 GPU queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from intelligent_liars.step5_queue import plan_step5_queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step5-plan", type=Path, required=True)
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="Authoritative current project task and instance inventory snapshot",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated independent training/projection seeds",
    )
    parser.add_argument("--max-concurrency", type=int, default=3)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--seeds must contain comma-separated integers") from error
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    step5_plan = _read_json(args.step5_plan.resolve())
    if step5_plan.get("format") != "tinylora_step5_plan_v1":
        raise ValueError("Unsupported Step 5 plan format")
    if step5_plan.get("large_run_enabled") or step5_plan.get("paid_execution_enabled"):
        raise ValueError("Queue planning requires a dry-run-only Step 5 manifest")
    scheduled_limit = int(
        step5_plan.get("schedule", {}).get(
            "max_concurrent_single_gpu_workers",
            3,
        )
    )
    if args.max_concurrency > scheduled_limit:
        raise ValueError(
            "Requested concurrency exceeds the immutable Step 5 schedule"
        )
    state = _read_json(args.state.resolve())
    plan = plan_step5_queue(
        arms=step5_plan["arms"],
        seeds=_parse_seeds(args.seeds),
        state=state,
        max_concurrency=args.max_concurrency,
    )
    if args.output is not None:
        _atomic_json(args.output.resolve(), plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
