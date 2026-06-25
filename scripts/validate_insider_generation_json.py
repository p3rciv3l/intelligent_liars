#!/usr/bin/env python3
"""Validate and optionally promote graded Qwen insider-generation JSON."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from intelligent_liars.judging import _insider_record_is_graded


REPORT_LABELS = ("explicit", "concealed", "implied", "invalid", "missing")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def planned_run_ids(
    *,
    project_root: Path,
    prompt_glob: str,
    samples_per_prompt: int,
) -> set[str]:
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    prompt_paths = sorted(project_root.glob(prompt_glob))
    if not prompt_paths:
        raise FileNotFoundError(f"No insider prompts matched {prompt_glob!r} under {project_root}")
    return {
        f"{prompt_path.relative_to(project_root)}::{sample_idx}"
        for prompt_path in prompt_paths
        for sample_idx in range(samples_per_prompt)
    }


def validate_insider_generation_json(
    path: Path,
    *,
    expected_count: int | None = None,
    expected_run_ids: set[str] | None = None,
    require_graded: bool = False,
    min_explicit: int = 1,
    min_concealed: int = 1,
) -> dict[str, Any]:
    errors: list[str] = []
    records = read_json(path)
    if not isinstance(records, list):
        return {
            "ok": False,
            "path": str(path),
            "errors": ["top-level payload must be a list"],
        }

    report_labels: Counter[str] = Counter()
    made_trade: Counter[str] = Counter()
    trade_labels: Counter[str] = Counter()
    double_labels: Counter[str] = Counter()
    duplicate_run_ids: list[str] = []
    missing_run_id_indices: list[int] = []
    invalid_record_indices: list[int] = []
    ungraded_indices: list[int] = []
    seen_run_ids: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            invalid_record_indices.append(index)
            continue
        metadata = record.get("metadata")
        transcript = record.get("transcript")
        if not isinstance(metadata, dict) or not isinstance(transcript, list):
            invalid_record_indices.append(index)
            continue

        run_id = metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            missing_run_id_indices.append(index)
        elif run_id in seen_run_ids:
            duplicate_run_ids.append(run_id)
        else:
            seen_run_ids.add(run_id)

        made_trade.update([str(metadata.get("made_trade", "missing"))])
        trade_labels.update([str(metadata.get("trade_label", "missing"))])
        double_labels.update([str(metadata.get("doubling_down_label", "missing"))])
        report_label = str(metadata.get("report_label", "missing"))
        if report_label not in REPORT_LABELS:
            report_label = "invalid"
        report_labels.update([report_label])

        if require_graded and not _insider_record_is_graded(record):
            ungraded_indices.append(index)

    if expected_count is not None and len(records) != expected_count:
        errors.append(f"record count is {len(records)}; expected {expected_count}")
    if invalid_record_indices:
        errors.append(f"invalid record shape at indices {invalid_record_indices[:10]}")
    if missing_run_id_indices:
        errors.append(f"missing metadata.run_id at indices {missing_run_id_indices[:10]}")
    if duplicate_run_ids:
        errors.append(f"duplicate metadata.run_id values {duplicate_run_ids[:10]}")
    if require_graded and ungraded_indices:
        errors.append(f"ungraded records at indices {ungraded_indices[:10]}")

    if expected_run_ids is not None:
        missing_expected = sorted(expected_run_ids - seen_run_ids)
        unexpected = sorted(seen_run_ids - expected_run_ids)
        if missing_expected:
            errors.append(f"missing expected run_ids {missing_expected[:10]}")
        if unexpected:
            errors.append(f"unexpected run_ids {unexpected[:10]}")

    explicit = int(report_labels.get("explicit", 0))
    concealed = int(report_labels.get("concealed", 0))
    if explicit < min_explicit:
        errors.append(f"explicit report labels={explicit}; need at least {min_explicit}")
    if concealed < min_concealed:
        errors.append(f"concealed report labels={concealed}; need at least {min_concealed}")

    return {
        "ok": not errors,
        "path": str(path),
        "records": len(records),
        "unique_run_ids": len(seen_run_ids),
        "expected_run_ids": None if expected_run_ids is None else len(expected_run_ids),
        "report_label_counts": dict(sorted(report_labels.items())),
        "made_trade_counts": dict(sorted(made_trade.items())),
        "trade_label_counts": dict(sorted(trade_labels.items())),
        "doubling_down_label_counts": dict(sorted(double_labels.items())),
        "binary_report_usable": explicit + concealed,
        "skipped_report_labels": int(report_labels.get("implied", 0) + report_labels.get("invalid", 0)),
        "ungraded_count": len(ungraded_indices),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate graded Qwen insider-generation JSON before promotion or activation extraction."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--prompt-glob", default="data/insider_trading/prompts/**/*.yaml")
    parser.add_argument("--samples-per-prompt", type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--require-graded", action="store_true")
    parser.add_argument("--min-explicit", type=int, default=1)
    parser.add_argument("--min-concealed", type=int, default=1)
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_run_ids = None
    if args.project_root is not None or args.samples_per_prompt is not None:
        if args.project_root is None or args.samples_per_prompt is None:
            raise SystemExit("--project-root and --samples-per-prompt must be provided together")
        expected_run_ids = planned_run_ids(
            project_root=args.project_root.resolve(),
            prompt_glob=args.prompt_glob,
            samples_per_prompt=args.samples_per_prompt,
        )

    report = validate_insider_generation_json(
        args.path,
        expected_count=args.expected_count,
        expected_run_ids=expected_run_ids,
        require_graded=args.require_graded,
        min_explicit=args.min_explicit,
        min_concealed=args.min_concealed,
    )
    if args.promote_to is not None:
        if not report["ok"]:
            report["promotion"] = "skipped; validation failed"
        else:
            write_json_atomic(args.promote_to, read_json(args.path), overwrite=args.overwrite)
            report["promotion"] = f"promoted to {args.promote_to}"

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
