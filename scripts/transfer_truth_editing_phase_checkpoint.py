#!/usr/bin/env python3
"""Publish or restore an immutable truth-editing phase checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_phase_checkpoint import (  # noqa: E402
    PhaseCheckpointError,
    publish_adaptive_checkpoint,
    publish_phase_checkpoint,
    restore_adaptive_checkpoint,
    restore_phase_checkpoint,
)


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--study-identity-sha256", required=True)
    parser.add_argument("--optuna-study-name", required=True)
    parser.add_argument("--completed-trials", type=int, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    _shared(publish)
    publish.add_argument(
        "--phase", choices=("discovery", "expanded", "finalist"), required=True
    )
    restore = commands.add_parser("restore")
    _shared(restore)
    restore.add_argument("--next-phase", choices=("expanded", "finalist"), required=True)
    publish_adaptive = commands.add_parser("publish-adaptive")
    _shared(publish_adaptive)
    publish_adaptive.add_argument("--study-config-sha256", required=True)
    restore_adaptive = commands.add_parser("restore-adaptive")
    _shared(restore_adaptive)
    restore_adaptive.add_argument("--study-config-sha256", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "publish":
            result = publish_phase_checkpoint(
                args.state_dir,
                args.publication_root,
                phase=args.phase,
                expected_study_identity_sha256=args.study_identity_sha256,
                expected_completed_trials=args.completed_trials,
                expected_optuna_study_name=args.optuna_study_name,
            )
        elif args.command == "restore":
            result = restore_phase_checkpoint(
                args.publication_root,
                args.state_dir,
                next_phase=args.next_phase,
                expected_study_identity_sha256=args.study_identity_sha256,
                expected_completed_trials=args.completed_trials,
                expected_optuna_study_name=args.optuna_study_name,
            )
        elif args.command == "publish-adaptive":
            result = publish_adaptive_checkpoint(
                args.state_dir,
                args.publication_root,
                expected_study_identity_sha256=args.study_identity_sha256,
                expected_study_config_sha256=args.study_config_sha256,
                expected_completed_trials=args.completed_trials,
                expected_optuna_study_name=args.optuna_study_name,
            )
        else:
            result = restore_adaptive_checkpoint(
                args.publication_root,
                args.state_dir,
                expected_study_identity_sha256=args.study_identity_sha256,
                expected_study_config_sha256=args.study_config_sha256,
                expected_completed_trials=args.completed_trials,
                expected_optuna_study_name=args.optuna_study_name,
            )
    except (OSError, PhaseCheckpointError) as error:
        print(f"phase checkpoint transfer failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
