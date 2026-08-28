#!/usr/bin/env python3
"""Materialize the immutable development judge calibration v4 contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_judge_dev_v4 import (  # noqa: E402
    build_dev_v4_artifacts,
)


def _render(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _write(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        stream.write(_render(value))
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "configs/truth_editing_judge_dev_v4",
    )
    args = parser.parse_args()
    v3 = ROOT / "configs/truth_editing_judge_dev_replacement_v3"
    source = ROOT / "artifacts/truth-editing/judge-calibration/revised-policy-v1"
    artifacts = build_dev_v4_artifacts(
        amended_plan_v3_path=v3 / "amended-plan-v3.json",
        amended_pack_v3_path=v3 / "amended-pack-v3.json",
        amended_labels_v3_path=v3 / "amended-labels-v3.json",
        amended_provenance_v3_path=v3 / "amended-provenance-v3.json",
        original_pack_path=source / "revised-pack.json",
        original_labels_path=source / "labels.json",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in {
        "plan.json": artifacts.plan,
        "compiler-pack.json": artifacts.pack,
        "compiler-labels.json": artifacts.labels,
        "provenance.json": artifacts.provenance,
        "execution-commands.json": artifacts.commands,
    }.items():
        _write(args.output_dir / name, value)
    print(_render({
        "output_dir": str(args.output_dir),
        "plan_sha256": artifacts.plan["content_sha256"],
        "planned_presentations": 180,
        "labels_changed": False,
        "paid_calls_made": 0,
    }).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
