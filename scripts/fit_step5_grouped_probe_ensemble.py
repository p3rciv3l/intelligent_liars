#!/usr/bin/env python3
"""Fit the CPU-only legacy grouped Step 5 probe ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.step5_grouped_probe_fit import run_grouped_probe_fit  # noqa: E402
from intelligent_liars.step5_probe_qualification import (  # noqa: E402
    validate_probe_qualification,
    write_probe_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-cache", type=Path, required=True)
    parser.add_argument("--raw-activation-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--step5-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_grouped_probe_fit(
        pooled_cache=args.pooled_cache.resolve(),
        raw_activation_cache=args.raw_activation_cache.resolve(),
        config_path=args.config.resolve(),
        step5_plan_path=args.step5_plan.resolve(),
        output_root=args.output_root.resolve(),
    )
    registry_path = args.output_root.resolve() / "probe_registry.json"
    qualification_path = args.output_root.resolve() / "probe_qualification.json"
    qualification = write_probe_qualification(
        json.loads(registry_path.read_text()),
        artifact_root=args.output_root.resolve(),
        output_path=qualification_path,
    )
    validation = validate_probe_qualification(
        qualification, artifact_root=args.output_root.resolve()
    )
    if not validation["valid"]:
        raise RuntimeError(f"Published qualification failed validation: {validation['issues']}")
    summary = {
        "format": "intelligent_liars_step5_grouped_probe_qualification_summary_v1",
        "status": "qualified",
        "selected": report["calibration"]["selected"],
        "fit_receipt_sha256": report["receipt_sha256"],
        "qualification_receipt_sha256": qualification[
            "qualification_receipt_sha256"
        ],
        "validation": validation,
    }
    summary_path = args.output_root.resolve() / "qualification_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                **summary,
                "output_root": str(args.output_root.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
