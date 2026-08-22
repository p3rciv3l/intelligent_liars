#!/usr/bin/env python3
"""Produce a deterministic dry-run plan for the bounded Step 5 GPU queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from intelligent_liars.step5_queue import plan_step5_queue
from intelligent_liars.step5_prerequisites import (
    prerequisite_identity,
    read_and_validate_prerequisite_receipt,
    step5_code_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step5-plan", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument(
        "--runtime-image-digest",
        required=True,
        help="Exact sha256:<digest> runtime image certified by prerequisites",
    )
    parser.add_argument(
        "--prerequisite-receipt",
        action="append",
        required=True,
        metavar="ARM=PATH",
        help="One frozen prerequisite receipt for each queued arm",
    )
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = value.partition("=")
        if not separator or not arm or not raw_path:
            raise ValueError("--prerequisite-receipt must use ARM=PATH")
        if arm in parsed:
            raise ValueError(f"Duplicate prerequisite receipt for arm: {arm}")
        parsed[arm] = Path(raw_path).resolve()
    return parsed


def main() -> int:
    args = parse_args()
    plan_path = args.step5_plan.resolve()
    probe_path = args.probe.resolve()
    step5_plan = _read_json(plan_path)
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
        raise ValueError("Requested concurrency exceeds the immutable Step 5 schedule")
    state = _read_json(args.state.resolve())
    receipt_paths = _receipt_paths(args.prerequisite_receipt)
    arm_by_name = {str(arm["name"]): arm for arm in step5_plan["arms"]}
    if set(receipt_paths) != set(arm_by_name):
        raise ValueError("Queue requires exactly one prerequisite receipt per arm")
    repository_root = Path(__file__).parents[1]
    code_sha256 = step5_code_sha256(repository_root)
    expected_identities = {
        name: prerequisite_identity(
            plan_sha256=_file_sha256(plan_path),
            probe_sha256=_file_sha256(probe_path),
            code_sha256=code_sha256,
            arm=arm,
            model=step5_plan["model"],
            runtime_image_digest=args.runtime_image_digest,
        )
        for name, arm in arm_by_name.items()
    }
    prerequisite_receipts: dict[str, dict[str, Any]] = {}
    for name, path in receipt_paths.items():
        receipt, digest = read_and_validate_prerequisite_receipt(
            path,
            expected_identity=expected_identities[name],
        )
        prerequisite_receipts[name] = {
            "receipt": receipt,
            "file_sha256": digest,
        }
    plan = plan_step5_queue(
        arms=step5_plan["arms"],
        seeds=_parse_seeds(args.seeds),
        state=state,
        max_concurrency=args.max_concurrency,
        prerequisite_receipts=prerequisite_receipts,
        expected_prerequisite_identities=expected_identities,
    )
    if args.output is not None:
        _atomic_json(args.output.resolve(), plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
