#!/usr/bin/env python3
"""Validate the frozen Step 5 contract without loading a model or opening the audit."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from intelligent_liars.tinylora_pilot import file_sha256
from intelligent_liars.tinylora_step5 import REQUIRED_SCENARIO_OBJECTIVES


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scenario_series(scenario_id: str) -> str:
    return re.sub(r"\.\d{4}$", "", scenario_id)


def contract_errors(
    plan: dict[str, Any], rows_by_output: dict[str, list[dict[str, Any]]]
) -> list[str]:
    """Validate the corrected Step 5 scientific/data contract."""
    errors: list[str] = []
    expected_arms = {
        ("tinylora_dim13", "tinylora", 13, None),
        ("tinylora_dim63", "tinylora", 63, None),
        ("lora_rank3_ceiling", "ordinary_lora", None, 3),
    }
    arms = {
        (
            arm["name"],
            arm["adapter"],
            arm.get("projection_dim"),
            arm.get("lora_rank"),
        )
        for arm in plan.get("arms", [])
    }
    if arms != expected_arms:
        errors.append("candidate arms differ from the corrected three-arm screen")
    basis = plan.get("basis_contract", {})
    if basis.get("effective_coordinate_counts") != [13, 63]:
        errors.append("TinyLoRA effective coordinates are not 13 and 63")
    if "exact prefix" not in str(basis.get("nesting", "")):
        errors.append("TinyLoRA basis nesting is not explicit")
    if "unit_norm" not in str(basis.get("normalization", "")):
        errors.append("TinyLoRA basis normalization is not explicit")

    seal = plan.get("sealed_audit", {})
    evidence = seal.get("evidence", {})
    if (
        seal.get("opened") is not False
        or seal.get("packaged") is not False
        or evidence.get("content_parsed_by_builder") is not False
        or evidence.get("hash_matches") is not True
        or evidence.get("observed_sha256") != seal.get("sha256")
    ):
        errors.append("sealed audit lacks hash-only, non-packaging evidence")

    admission = plan.get("source_admission", {})
    if admission.get("tulu", {}).get("admitted_to_training") is not False:
        errors.append("pending Tulu source is not quarantined")
    preservation_rows = [
        row
        for name in (
            "preservation_train",
            "preservation_development_text",
            "preservation_development_vision",
        )
        for row in rows_by_output.get(name, [])
    ]
    if any("tulu" in str(row.get("source", "")).lower() for row in preservation_rows):
        errors.append("quarantined Tulu row reached a training/evaluation output")
    if any(
        row.get("record_id")
        == "prime_synthetic_2_verified.default.train.000079539"
        for row in preservation_rows
    ):
        errors.append("known semantically defective Prime row was not excluded")
    max_length = plan.get("preservation_curation", {}).get("max_length")
    for row in preservation_rows:
        if row.get("preservation_category", "").startswith("vision_"):
            continue
        qualification = row.get("qualification", {})
        if (
            not isinstance(max_length, int)
            or qualification.get("max_length") != max_length
            or not isinstance(qualification.get("token_length"), int)
            or qualification["token_length"] > max_length
        ):
            errors.append(f"unqualified preservation row: {row.get('record_id')}")
            break
    notices = plan.get("source_notices", {})
    if notices.get("xstest", {}).get("license") != "CC-BY-4.0":
        errors.append("XSTest attribution/license notice is missing")
    if notices.get("pixmo_docs", {}).get("license") != (
        "ODC-BY-1.0 with separate model-output terms"
    ):
        errors.append("PixMo-Docs license notice is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    errors: list[str] = []
    if plan.get("format") != "tinylora_step5_plan_v1":
        errors.append("unsupported plan format")
    if plan.get("large_run_enabled") or plan.get("paid_execution_enabled"):
        errors.append("plan must keep large and paid execution disabled")
    if not plan.get("model", {}).get("revision"):
        errors.append("model revision is not pinned")
    rows_by_output: dict[str, list[dict[str, Any]]] = {}
    for name, specification in plan.get("outputs", {}).items():
        path = plan_path.parent / specification["path"]
        if not path.is_file():
            errors.append(f"missing output: {name}")
            continue
        if file_sha256(path) != specification["sha256"]:
            errors.append(f"hash mismatch: {name}")
        rows = read_jsonl(path)
        rows_by_output[name] = rows
        if len(rows) != specification["records"]:
            errors.append(f"record count mismatch: {name}")
        ids = [str(row["record_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate record ids: {name}")
    errors.extend(contract_errors(plan, rows_by_output))

    behavior_names = (
        "train_behavior",
        "development_iid",
        "development_heldout_family",
    )
    scenario_sets: dict[str, set[str]] = {}
    for name in behavior_names:
        rows = rows_by_output.get(name, [])
        scenario_sets[name] = {str(row["scenario_id"]) for row in rows}
        objectives: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            objectives[str(row["scenario_id"])].add(str(row["objective"]))
            if row.get("alternative_target") == row.get("target"):
                errors.append(f"non-contrastive behavior pair: {row['record_id']}")
        for scenario, found in objectives.items():
            if found != REQUIRED_SCENARIO_OBJECTIVES:
                errors.append(f"incomplete scenario {scenario} in {name}")
    for index, left in enumerate(behavior_names):
        for right in behavior_names[index + 1 :]:
            overlap = scenario_sets[left] & scenario_sets[right]
            if overlap:
                errors.append(f"scenario leakage between {left} and {right}: {len(overlap)}")
            series_overlap = {
                _scenario_series(value) for value in scenario_sets[left]
            } & {_scenario_series(value) for value in scenario_sets[right]}
            if series_overlap:
                errors.append(
                    f"scenario-series leakage between {left} and {right}: "
                    f"{len(series_overlap)}"
                )

    preservation_train = {
        str(row["record_id"])
        for row in rows_by_output.get("preservation_train", [])
    }
    preservation_development = {
        str(row["record_id"])
        for name in ("preservation_development_text", "preservation_development_vision")
        for row in rows_by_output.get(name, [])
    }
    if preservation_train & preservation_development:
        errors.append("preservation train/development record leakage")
    for row in [
        *rows_by_output.get("preservation_train", []),
        *rows_by_output.get("preservation_development_vision", []),
    ]:
        if "image_sha256" not in row:
            continue
        image = row["messages"][0]["content"][0]["image"]
        path = repository_root / image
        if not path.is_file() or file_sha256(path) != row["image_sha256"]:
            errors.append(f"missing or mismatched image: {row['record_id']}")

    safety = rows_by_output.get("safety_refusal_development", [])
    safety_counts = Counter(row.get("expected_behavior") for row in safety)
    if safety_counts != {"comply": 250, "refuse": 200}:
        errors.append(f"unexpected XSTest balance: {dict(safety_counts)}")
    if plan.get("preservation_policy", {}).get("batch_fraction") != 0.25:
        errors.append("preservation batch fraction is not the approved 25%")
    if plan.get("schedule", {}).get("max_concurrent_single_gpu_workers") != 3:
        errors.append("worker cap is not three")
    if not (repository_root / "scripts/run_tinylora_step5_screen.py").is_file():
        errors.append("Step 5 screen runner is missing")

    report = {
        "format": "tinylora_step5_preflight_v1",
        "valid": not errors,
        "errors": errors,
        "plan_sha256": file_sha256(plan_path),
        "audit_opened": False,
        "outputs": {
            name: {
                "records": len(rows),
                "sha256": plan["outputs"][name]["sha256"],
            }
            for name, rows in sorted(rows_by_output.items())
        },
        "safety_refusal_counts": dict(sorted(safety_counts.items())),
        "behavior_scenario_counts": {
            name: len(scenarios) for name, scenarios in scenario_sets.items()
        },
        "preservation_train_records": len(preservation_train),
        "preservation_development_records": len(preservation_development),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
