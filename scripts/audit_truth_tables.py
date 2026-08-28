#!/usr/bin/env python3
"""Run the bounded, offline truth-table audits.

Examples
--------
Audit the checked-in Truth Spec Geometry QA files and the Apollo TruthfulQA
loader, writing a receipt under the public dataset namespace::

    python scripts/audit_truth_tables.py

Additional JSON/JSONL files can be supplied with ``--mmlu`` and
``--sandbagging``.  The command exits non-zero when any row must be
quarantined, which makes it safe to use as a build gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.truth_table_audit import (  # noqa: E402
    audit_duplicate_truth_labels,
    audit_geometry_qa_rows,
    audit_mmlu_gold_mappings,
    audit_sandbagging_rows,
    combine_reports,
    detect_truthfulqa_first_character_parser,
    read_csv_rows,
    write_audit_report,
)


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
        return rows
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        # Apollo rollout files are often wrapped in a list under ``rollouts``.
        for key in ("rows", "examples", "rollouts", "data"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a JSON list of objects")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=Path("references/truth_spec/data/geometry_of_truth"),
        help="directory containing Geometry of Truth CSV files",
    )
    parser.add_argument(
        "--truthfulqa-source",
        type=Path,
        default=Path("references/deception-detection/deception_detection/data/truthful_qa.py"),
        help="TruthfulQA loader source to scan for first-character indexing",
    )
    parser.add_argument("--mmlu", type=Path, action="append", default=[], help="MMLU JSON/JSONL file")
    parser.add_argument(
        "--sandbagging",
        type=Path,
        action="append",
        default=[],
        help="Apollo sandbagging JSON/JSONL response file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/truth_editing/audits/truth_table_audit_v1.json"),
        help="canonical JSON audit receipt",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    reports = []
    geometry_files = sorted(args.geometry_dir.glob("*_qa.csv"))
    geometry_rows: list[dict[str, Any]] = []
    for path in geometry_files:
        rows = read_csv_rows(path)
        reports.append(audit_geometry_qa_rows(rows, source=str(path)))
        geometry_rows.extend(rows)
    if geometry_rows:
        reports.append(audit_duplicate_truth_labels(geometry_rows, source="geometry_of_truth:combined"))

    if args.truthfulqa_source.exists():
        reports.append(detect_truthfulqa_first_character_parser(args.truthfulqa_source))

    for path in args.mmlu:
        reports.append(audit_mmlu_gold_mappings(_read_json_rows(path), source=str(path)))
    for path in args.sandbagging:
        reports.append(audit_sandbagging_rows(_read_json_rows(path), source=str(path)))

    report = combine_reports(reports)
    write_audit_report(args.output, report)
    print(json.dumps(report.to_payload(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report.valid else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
