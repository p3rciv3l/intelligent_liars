#!/usr/bin/env python3
"""Dry-run or execute one timed production-parity truth-editing canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_timed_canary import (  # noqa: E402
    TimedCanaryConfig,
    TimedCanaryError,
    run_timed_canary,
)
from intelligent_liars.truth_editing_production import (  # noqa: E402
    ProductionCompositionError,
    ProductionRunConfig,
)


def _load_command(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(item, str) or not item or item != item.strip() for item in raw)
    ):
        raise TimedCanaryError("command JSON must be a nonempty array of trimmed strings")
    return list(raw)


def _verify_production_config(repo: Path, config: TimedCanaryConfig) -> Path:
    root = repo.resolve()
    path = (root / config.production_config_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise TimedCanaryError("production config escapes the repository") from error
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != config.production_config_sha256:
        raise TimedCanaryError("production config SHA-256 differs from canary contract")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-hourly-usd", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = TimedCanaryConfig.from_mapping(json.loads(args.config.read_text()))
        production_path = _verify_production_config(args.repo, config)
        # Strict-open the linked repeat-calibrated packet without loading a model.
        # This verifies its exact schema, safe paths, judge budget, and preservation
        # runtime packet before either a dry run or the workload can proceed.
        production = ProductionRunConfig.open(production_path)
        if (
            production.preservation_threshold_calibration is None
            or production.preservation_threshold_calibration_sha256 is None
            or production.judge_budget is None
        ):
            raise TimedCanaryError(
                "production config is not repeat-calibrated and judge-budget bound"
            )
        command = _load_command(args.command_json)
        dry_run: dict[str, Any] = {
            "format": "truth_editing_timed_canary_dry_run_v1",
            "mode": "dry_run",
            "canary_config_sha256": config.identity_sha256,
            "production_config_path": str(production_path),
            "production_config_verified": True,
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest(),
            "command_argument_count": len(command),
            "maximum_wall_seconds": config.maximum_wall_seconds,
            "external_mutation": False,
        }
        if not args.execute:
            print(json.dumps(dry_run, indent=2, sort_keys=True))
            return 0
        if args.observation.exists() or args.observation.is_symlink():
            raise TimedCanaryError("observation path already exists")

        def workload() -> dict[str, Any]:
            args.observation.parent.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ)
            environment["TRUTH_EDITING_TIMED_CANARY_OBSERVATION_PATH"] = str(
                args.observation.resolve()
            )
            environment["TRUTH_EDITING_GPU_HOURLY_USD"] = str(args.gpu_hourly_usd)
            subprocess.run(
                command,
                check=True,
                cwd=args.repo,
                env=environment,
                timeout=config.maximum_wall_seconds,
            )
            if not args.observation.is_file():
                raise TimedCanaryError("workload did not write the canary observation")
            value = json.loads(args.observation.read_text())
            if not isinstance(value, dict):
                raise TimedCanaryError("canary observation must be an object")
            return value

        receipt = run_timed_canary(
            config=config,
            workload=workload,
            gpu_hourly_usd=args.gpu_hourly_usd,
            receipt_path=args.receipt,
            monotonic=time.monotonic,
        )
        print(json.dumps({"mode": "executed", "receipt": receipt}, indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        ProductionCompositionError,
        TimedCanaryError,
    ) as error:
        print(f"truth-editing timed canary failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
