#!/usr/bin/env python3
"""Evaluate complete external TinyLoRA Step 5 evidence against five frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from intelligent_liars.step5_gate_evaluator import (
    GateEvaluationError,
    evaluate_step5_gates,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--thresholds-sha256", required=True)
    parser.add_argument("--base-thresholds", required=True, type=Path)
    parser.add_argument("--base-thresholds-sha256", required=True)
    parser.add_argument("--base-paired", required=True, type=Path)
    parser.add_argument("--candidate-paired", required=True, type=Path)
    parser.add_argument("--generation", required=True, type=Path)
    parser.add_argument("--preservation", required=True, type=Path)
    parser.add_argument("--safety", required=True, type=Path)
    parser.add_argument("--probes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise GateEvaluationError(f"{path} must contain one JSON object")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def main() -> int:
    args = _arguments()
    try:
        actual_threshold_sha = hashlib.sha256(args.thresholds.read_bytes()).hexdigest()
        if actual_threshold_sha != args.thresholds_sha256:
            raise GateEvaluationError(
                "threshold file SHA-256 does not match the required digest"
            )
        actual_base_threshold_sha = hashlib.sha256(
            args.base_thresholds.read_bytes()
        ).hexdigest()
        if actual_base_threshold_sha != args.base_thresholds_sha256:
            raise GateEvaluationError(
                "base threshold file SHA-256 does not match the required digest"
            )
        result = evaluate_step5_gates(
            thresholds=_load(args.thresholds),
            thresholds_file_sha256=actual_threshold_sha,
            base_threshold_registry=_load(args.base_thresholds),
            base_thresholds_file_sha256=actual_base_threshold_sha,
            base_paired=_load(args.base_paired),
            candidate_paired=_load(args.candidate_paired),
            generation=_load(args.generation),
            preservation=_load(args.preservation),
            safety=_load(args.safety),
            probes=_load(args.probes),
        )
    except (OSError, json.JSONDecodeError, GateEvaluationError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 2
    _write(args.output, result)
    print(
        json.dumps(
            {
                "ok": True,
                "eligible_to_advance": result["eligible_to_advance"],
                "failures": result["failures"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["eligible_to_advance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
