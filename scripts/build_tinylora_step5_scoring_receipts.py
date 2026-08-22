#!/usr/bin/env python3
"""Build all six non-XSTest Step 5 gate receipts in one fail-closed command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.step5_scoring_receipts import (
    ScoringReceiptError,
    build_scoring_receipts,
    publish_scoring_receipts,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--base-thresholds", required=True, type=Path)
    parser.add_argument("--request-inventory-commitment", required=True, type=Path)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--base-paired-diagnostics", required=True, type=Path)
    parser.add_argument("--candidate-paired-diagnostics", required=True, type=Path)
    parser.add_argument("--preservation-diagnostics", required=True, type=Path)
    parser.add_argument("--probe-diagnostics", required=True, type=Path)
    parser.add_argument("--probe-qualification", required=True, type=Path)
    parser.add_argument("--registry-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        receipts = build_scoring_receipts(
            plan_path=args.plan,
            thresholds_path=args.thresholds,
            base_thresholds_path=args.base_thresholds,
            request_inventory_commitment_path=args.request_inventory_commitment,
            base_run_dir=args.base_run,
            candidate_run_dir=args.candidate_run,
            base_paired_diagnostics_path=args.base_paired_diagnostics,
            candidate_paired_diagnostics_path=args.candidate_paired_diagnostics,
            preservation_diagnostics_path=args.preservation_diagnostics,
            probe_diagnostics_path=args.probe_diagnostics,
            probe_qualification_path=args.probe_qualification,
            registry_metrics_path=args.registry_metrics,
        )
        manifest = publish_scoring_receipts(args.output_dir, receipts)
    except (OSError, json.JSONDecodeError, ScoringReceiptError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "complete": manifest["complete"],
                "content_sha256": manifest["content_sha256"],
                "output_dir": str(args.output_dir.resolve()),
                "receipts": sorted(manifest["outputs"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
