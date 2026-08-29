#!/usr/bin/env python3
"""Run the concrete persistent truth-editing optimization composition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
QUALIFIED_PRODUCTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "truth_editing_production_v4_r10_17b1e9cb_c1f373f8.json"
)
QUALIFIED_PRODUCTION_CONFIG_SHA256 = (
    "29d798bb0f13140e7c24def266fbd656f64b09d2c9410e25416c92d777f85e70"
)
PHASES = ("discovery", "expanded", "finalist")
_HEX = frozenset("0123456789abcdef")


def _expected_config_sha256(config_path: Path, supplied: str | None) -> str:
    if supplied is None:
        if config_path.resolve() != QUALIFIED_PRODUCTION_CONFIG.resolve():
            raise ValueError("an explicit --config requires --config-sha256")
        return QUALIFIED_PRODUCTION_CONFIG_SHA256
    if len(supplied) != 64 or any(character not in _HEX for character in supplied):
        raise ValueError("--config-sha256 must be a lowercase SHA-256")
    return supplied


def _verify_config_identity(config_path: Path, expected_sha256: str) -> None:
    if config_path.is_symlink() or not config_path.is_file():
        # Preserve the useful operating-system message for a missing default.
        config_path.read_bytes()
        raise ValueError("production config must be a regular non-symlink file")
    observed = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ValueError("production config SHA-256 differs from the selected identity")


def _phase_stop_after_trials(config_path: Path, phase: str) -> int:
    production = json.loads(config_path.read_text())
    study_path = config_path.parent / production["study_config"]
    study = json.loads(study_path.read_text())
    matches = [
        tier for tier in study["evaluation_tiers"] if tier.get("name") == phase
    ]
    if len(matches) != 1:
        raise ValueError(f"production study has no unique {phase!r} tier")
    boundary = matches[0].get("through_trial")
    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, int)
        or boundary <= 0
        or boundary > study.get("max_trials", 0)
        or boundary % study.get("batch_size", 0) != 0
    ):
        raise ValueError(f"production {phase!r} tier is not a completed batch barrier")
    return boundary
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_production import (  # noqa: E402
    ProductionCompositionError,
    open_production_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=QUALIFIED_PRODUCTION_CONFIG,
        help=(
            "strict-open repeat-calibrated production config; custom paths require "
            "--config-sha256"
        ),
    )
    parser.add_argument(
        "--config-sha256",
        help="lowercase SHA-256 binding for an explicitly selected production config",
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="finalist",
        help="resume the same study journal through this completed tier boundary",
    )
    args = parser.parse_args(argv)
    try:
        expected_config_sha256 = _expected_config_sha256(
            args.config, args.config_sha256
        )
        _verify_config_identity(args.config, expected_config_sha256)
        run = open_production_run(args.config)
        stop_after_trials = _phase_stop_after_trials(args.config, args.phase)
        receipt = run.run(stop_after_trials=stop_after_trials)
    except (OSError, ValueError, ProductionCompositionError) as error:
        print(f"truth-editing production run failed: {error}", file=sys.stderr)
        return 2
    payload = receipt.to_mapping()
    payload["production_config_path"] = str(args.config)
    payload["production_config_sha256"] = expected_config_sha256
    payload["run_receipt_sha256"] = receipt.identity_sha256
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
