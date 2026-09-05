#!/usr/bin/env python3
"""Publish an exact legacy-phase or rolling-adaptive production checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_phase_checkpoint import (  # noqa: E402
    PhaseCheckpointError,
    publish_adaptive_checkpoint,
    publish_phase_checkpoint,
)


BOUNDARIES = {"discovery": 80, "expanded": 160, "finalist": 200}


def _completed_trials(journal: dict[str, object]) -> tuple[int, bool]:
    if journal.get("format") != "truth_editing_study_journal_v1":
        raise PhaseCheckpointError("study journal format differs")
    batches = journal.get("batches")
    if not isinstance(batches, list):
        raise PhaseCheckpointError("study journal batches are invalid")
    identifiers: set[str] = set()
    completed = 0
    incomplete = False
    for batch in batches:
        if not isinstance(batch, dict) or not isinstance(batch.get("trials"), list):
            raise PhaseCheckpointError("study journal batch is invalid")
        for trial in batch["trials"]:
            if not isinstance(trial, dict) or not isinstance(trial.get("trial_id"), str):
                raise PhaseCheckpointError("study journal trial is invalid")
            trial_id = trial["trial_id"]
            if trial_id in identifiers:
                raise PhaseCheckpointError("study journal contains duplicate trials")
            identifiers.add(trial_id)
            if trial.get("result") is None:
                incomplete = True
            else:
                completed += 1
    return completed, incomplete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", choices=tuple(BOUNDARIES))
    mode.add_argument(
        "--adaptive",
        action="store_true",
        help="publish the rolling adaptive scheduler, Optuna, and W&B state",
    )
    parser.add_argument("--study-config-sha256")
    parser.add_argument("--optuna-study-name", required=True)
    args = parser.parse_args(argv)
    try:
        if not args.journal.exists():
            print(json.dumps({"status": "deferred", "reason": "journal_not_created"}))
            return 0
        if args.journal.is_symlink() or not args.journal.is_file():
            raise PhaseCheckpointError("study journal is missing or unsafe")
        if args.journal.name != "study-journal.json" or args.journal.parent.name != "study":
            raise PhaseCheckpointError("journal must use the production study/study-journal.json layout")
        journal = json.loads(args.journal.read_text(encoding="utf-8"))
        study_identity = journal["study_identity_sha256"]
        completed, incomplete = _completed_trials(journal)
        if incomplete:
            print(
                json.dumps(
                    {
                        "status": "deferred",
                        "reason": "current_batch_incomplete",
                        "completed_trials": completed,
                    }
                )
            )
            return 0
        optuna_log = args.journal.with_name(args.journal.name + ".optuna.log")
        if optuna_log.is_symlink() or not optuna_log.is_file():
            raise PhaseCheckpointError("study journal or Optuna log is missing or unsafe")
        state_dir = args.journal.parents[1]
        if args.adaptive:
            if args.study_config_sha256 is None:
                raise PhaseCheckpointError("study config identity is required for adaptive publication")
            receipt = publish_adaptive_checkpoint(
                state_dir,
                args.output,
                expected_study_identity_sha256=study_identity,
                expected_study_config_sha256=args.study_config_sha256,
                expected_completed_trials=completed,
                expected_optuna_study_name=args.optuna_study_name,
            )
        else:
            if args.study_config_sha256 is not None:
                raise PhaseCheckpointError("study config identity is only valid for adaptive publication")
            boundary = BOUNDARIES[args.phase]
            if completed < boundary:
                print(json.dumps({"status": "deferred", "completed_trials": completed, "phase_boundary": boundary}))
                return 0
            if completed > boundary:
                raise PhaseCheckpointError("study journal passed the requested phase barrier")
            receipt = publish_phase_checkpoint(
                state_dir,
                args.output,
                phase=args.phase,
                expected_study_identity_sha256=study_identity,
                expected_completed_trials=boundary,
                expected_optuna_study_name=args.optuna_study_name,
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError, PhaseCheckpointError) as error:
        print(f"truth-editing checkpoint publication failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
