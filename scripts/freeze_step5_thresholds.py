#!/usr/bin/env python3
"""Freeze Step 5 thresholds from repeated base-model metric receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from intelligent_liars.step5_thresholds import (
    ThresholdFreezeError,
    build_frozen_thresholds,
    write_frozen_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-receipt", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--candidate-results", action="append", required=True, type=Path)
    parser.add_argument("--frozen-at", help="Explicit timezone-aware ISO timestamp")
    return parser.parse_args()


def _load_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ThresholdFreezeError("An explicit policy path is required")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ThresholdFreezeError("Policy must be a JSON object")
    return payload


def main() -> int:
    args = parse_args()
    try:
        registry = build_frozen_thresholds(
            args.base_receipt,
            policy=_load_policy(args.policy),
            frozen_at=args.frozen_at,
        )
        created = write_frozen_thresholds(
            registry,
            args.output,
            candidate_result_paths=args.candidate_results,
        )
    except (OSError, json.JSONDecodeError, ThresholdFreezeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "created": created,
                "output": str(args.output),
                "commitment_sha256": registry["commitment_sha256"],
                "receipt_count": registry["receipt_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
