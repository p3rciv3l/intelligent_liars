#!/usr/bin/env python3
"""Execute the reserved adaptive repeats, controls, selection, and export lane."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from intelligent_liars.truth_editing_adaptive_finalization import (
    open_adaptive_finalization_handoff,
    run_adaptive_finalization,
)
from intelligent_liars.truth_editing_production import open_production_run
from intelligent_liars.truth_editing_production_finalization import (
    ProductionAdaptiveFinalizationExecutor,
    ProductionFinalistCheckpointExporter,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh finalist repeats, matched orthogonal/shuffled controls, "
            "audited selection, and verified checkpoint export."
        )
    )
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--registry-bucket", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument(
        "--causal-control-receipt",
        required=True,
        action="append",
        help=(
            "Repeat TRIAL_ID=/path/to/receipt.json for each strong candidate; "
            "each receipt must cover restoration, re-ablation, random direction, "
            "and false trigger"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    handoff = open_adaptive_finalization_handoff(args.handoff)
    study_receipt_path = Path(handoff["study_artifact_receipt"]["path"])
    study_receipt = json.loads(study_receipt_path.read_text())
    receipt_sha = study_receipt.get("receipt_sha256")
    if not isinstance(receipt_sha, str):
        raise ValueError("study artifact receipt lacks receipt_sha256")
    exporter = ProductionFinalistCheckpointExporter(
        production_config_path=handoff["production_config"]["path"],
        study_artifact_receipt_path=study_receipt_path,
        study_identity_sha256=handoff["study_identity_sha256"],
        study_artifact_receipt_sha256=receipt_sha,
        registry_bucket=args.registry_bucket,
        model_slug=args.model_slug,
    )
    production = open_production_run(handoff["production_config"]["path"])
    scheduled_evaluations = handoff["strong_candidate_count"] * (
        handoff["repeat_count_per_candidate"] + 2
    )
    maximum_per_evaluation = Decimal(
        handoff["maximum_evaluation_spend_usd"]
    ) / Decimal(scheduled_evaluations)
    backend = production.build_finalization_backend(
        checkpoint_exporter=exporter,
        maximum_evaluation_cost_usd=format(maximum_per_evaluation, "f"),
    )
    causal_receipts: dict[str, Path] = {}
    for value in args.causal_control_receipt:
        trial_id, separator, raw_path = value.partition("=")
        if not separator or not trial_id or not raw_path or trial_id in causal_receipts:
            raise ValueError(
                "--causal-control-receipt must be unique TRIAL_ID=PATH values"
            )
        causal_receipts[trial_id] = Path(raw_path)
    executor = ProductionAdaptiveFinalizationExecutor(
        handoff["study_report"]["path"],
        backend,
        causal_control_receipts=causal_receipts,
    )
    receipt = run_adaptive_finalization(args.handoff, executor)
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
