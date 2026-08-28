#!/usr/bin/env python3
"""Verify the offline/mock or live-dashboard W&B timed-canary evidence."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_wandb_canary import (  # noqa: E402
    WandbCanaryError,
    read_wandb_dashboard,
    verify_wandb_canary,
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise WandbCanaryError(f"{label} must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify-dashboard-live", action="store_true")
    args = parser.parse_args(argv)
    try:
        trace = _read_object(args.trace, "W&B canary trace")
        checkpoint = _read_object(args.checkpoint, "W&B run checkpoint")
        mode = "offline_verified"
        if args.verify_dashboard_live:
            wandb = importlib.import_module("wandb")
            trace["dashboard_readback"] = read_wandb_dashboard(
                wandb.Api(), checkpoint=checkpoint
            )
            mode = "live_dashboard_verified"
        receipt = verify_wandb_canary(
            trace=trace, checkpoint=checkpoint, receipt_path=args.receipt
        )
        print(json.dumps({"mode": mode, "receipt": receipt}, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, WandbCanaryError) as error:
        print(f"W&B canary verification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
