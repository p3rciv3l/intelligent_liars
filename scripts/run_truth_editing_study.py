#!/usr/bin/env python3
"""Run the resumable truth-editing study with an explicit search adapter.

The default evaluator is deterministic and offline. It exercises orchestration,
coverage, batching, tiers, and resume only; its numbers are not scientific
evidence and it never loads or edits a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_contracts import (  # noqa: E402
    parse_direction_bank_manifest,
)
from intelligent_liars.truth_editing_study import (  # noqa: E402
    OfflineDeterministicSearchDriver,
    OfflineSyntheticEvaluator,
    OptunaSearchDriver,
    SearchDriver,
    TruthEditingStudy,
    load_truth_editing_study_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--search-driver", choices=("offline", "optuna"), default="offline")
    args = parser.parse_args()

    config = load_truth_editing_study_config(args.config)
    direction_manifest = parse_direction_bank_manifest(
        json.loads(args.direction_manifest.read_text())
    )
    driver: SearchDriver
    if args.search_driver == "optuna":
        driver = OptunaSearchDriver(
            seed=config.sampler_seed, strategy=config.search_strategy
        )
    else:
        driver = OfflineDeterministicSearchDriver(seed=config.sampler_seed)
    report = TruthEditingStudy(config, direction_manifest).run(
        driver=driver,
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=args.journal,
    )
    rendered = json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
