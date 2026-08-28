#!/usr/bin/env python3
"""Materialize the fresh source-disjoint 120-presentation judge holdout v4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_judge_fresh_holdout_v4 import (  # noqa: E402
    build_fresh_holdout_v4,
)


def _render(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _exclusive_write(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        stream.write(_render(value))
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--policy-provenance", required=True, type=Path)
    parser.add_argument("--existing-plan", required=True, type=Path, action="append")
    parser.add_argument("--existing-receipt-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    artifacts = build_fresh_holdout_v4(
        source_path=args.source,
        dataset_manifest_path=args.dataset_manifest,
        policy_provenance_path=args.policy_provenance,
        existing_plan_paths=tuple(args.existing_plan),
        existing_receipt_dirs=tuple(args.existing_receipt_dir),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("revised-pack.json", artifacts.pack),
        ("labels.json", artifacts.labels),
        ("provenance.json", artifacts.provenance),
        ("plan.json", artifacts.plan),
        ("request-identities.json", artifacts.request_identities),
    ):
        _exclusive_write(args.output_dir / name, value)
    print(_render({
        "output_dir": str(args.output_dir),
        "plan_sha256": artifacts.plan["content_sha256"],
        "pack_sha256": artifacts.pack["content_sha256"],
        "labels_sha256": artifacts.labels["content_sha256"],
        "provenance_sha256": artifacts.provenance["content_sha256"],
        "request_identities_sha256": artifacts.request_identities["content_sha256"],
        "total_presentations": 120,
        "external_calls_made": 0,
    }).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
