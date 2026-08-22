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
from intelligent_liars.tinylora_step5 import (
    REQUIRED_SCENARIO_OBJECTIVES,
    qualify_vision_preservation_rows,
)


FROZEN_OUTPUT_RECORDS = {
    "train_behavior": 3114,
    "development_iid": 420,
    "development_heldout_family": 570,
    "preservation_train": 212,
    "preservation_development_text": 29,
    "preservation_development_vision": 22,
    "safety_refusal_development": 450,
}
PINNED_TOKENIZER_JSON_SHA256 = (
    "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
)
PINNED_TOKENIZER_CONFIG_SHA256 = (
    "7b501e639b4d107a23effbe30390ee33d553f722467f7ac8e2744d7ff5d3a7d5"
)
PINNED_PREPROCESSOR_CONFIG_SHA256 = (
    "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
)
PINNED_VISION_FACTOR = 32
PINNED_VISION_MIN_PIXELS = 65536
PINNED_VISION_MAX_PIXELS = 16777216


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scenario_series(scenario_id: str) -> str:
    return re.sub(r"\.\d{4}$", "", scenario_id)


def contract_errors(
    plan: dict[str, Any], rows_by_output: dict[str, list[dict[str, Any]]]
) -> list[str]:
    """Validate the corrected Step 5 scientific/data contract."""
    errors: list[str] = []
    inputs = plan.get("inputs", {})
    if inputs.get("tokenizer_json_sha256") != PINNED_TOKENIZER_JSON_SHA256:
        errors.append("tokenizer.json does not match the pinned Step 5 artifact")
    if inputs.get("tokenizer_config_sha256") != PINNED_TOKENIZER_CONFIG_SHA256:
        errors.append("tokenizer config does not match the pinned Step 5 artifact")
    if (
        inputs.get("preprocessor_config_sha256")
        != PINNED_PREPROCESSOR_CONFIG_SHA256
    ):
        errors.append("vision preprocessor does not match the pinned Step 5 artifact")
    if plan.get("preservation_curation", {}).get("vision_token_geometry") != {
        "factor": PINNED_VISION_FACTOR,
        "max_pixels": PINNED_VISION_MAX_PIXELS,
        "method": "Qwen smart_resize plus rendered text tokens",
        "min_pixels": PINNED_VISION_MIN_PIXELS,
    }:
        errors.append("vision token geometry differs from the pinned processor contract")
    if set(plan.get("outputs", {})) != set(FROZEN_OUTPUT_RECORDS):
        errors.append("Step 5 outputs differ from the frozen output inventory")
    elif any(
        plan["outputs"][name].get("records") != records
        for name, records in FROZEN_OUTPUT_RECORDS.items()
    ):
        errors.append("Step 5 output counts differ from the frozen contract")
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
    semantic_evidence = plan.get("preservation_curation", {}).get(
        "semantic_exclusion_evidence", {}
    )
    bad_evidence = semantic_evidence.get(
        "prime_synthetic_2_verified.default.train.000079539", {}
    )
    if (
        bad_evidence.get("reason")
        != "semantic_quality_changes_problem_to_force_answer"
        or not isinstance(bad_evidence.get("source_row_sha256"), str)
        or len(bad_evidence["source_row_sha256"]) != 64
        or not bad_evidence.get("adjudication")
    ):
        errors.append("known Prime semantic exclusion lacks adjudication evidence")
    for row in preservation_rows:
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


def vision_qualification_errors(
    *,
    repository_root: Path,
    plan: dict[str, Any],
    rows_by_output: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
) -> list[str]:
    """Reproduce every admitted PixMo qualification from pinned raw inputs."""
    compiled = [
        row
        for name in ("preservation_train", "preservation_development_vision")
        for row in rows_by_output.get(name, [])
        if str(row.get("preservation_category", "")).startswith("vision_")
    ]
    if not compiled:
        return ["vision preservation inventory is empty"]
    source_paths = {str(row.get("source", "")) for row in compiled}
    if len(source_paths) != 1:
        return ["vision preservation rows do not share one pinned source"]
    relative_source = Path(source_paths.pop())
    if relative_source.is_absolute() or ".." in relative_source.parts:
        return ["vision preservation source path is unsafe"]
    source_path = repository_root / relative_source
    if (
        not source_path.is_file()
        or file_sha256(source_path) != plan.get("inputs", {}).get("pixmo_docs_snapshot_sha256")
    ):
        return ["PixMo source snapshot is missing or mismatched"]
    try:
        qualified, exclusions = qualify_vision_preservation_rows(
            read_jsonl(source_path),
            tokenizer=tokenizer,
            repository_root=repository_root,
            seed=int(plan["seed"]),
            max_length=int(plan["preservation_curation"]["max_length"]),
            factor=PINNED_VISION_FACTOR,
            min_pixels=PINNED_VISION_MIN_PIXELS,
            max_pixels=PINNED_VISION_MAX_PIXELS,
        )
    except (KeyError, TypeError, ValueError) as error:
        return [f"vision qualification could not be reproduced: {error}"]
    expected = {str(row["record_id"]): row["qualification"] for row in qualified}
    actual = {str(row["record_id"]): row.get("qualification") for row in compiled}
    errors: list[str] = []
    if actual != expected:
        errors.append("compiled vision qualification differs from pinned-source replay")
    raw_by_id = {str(row["record_id"]): row for row in qualified}
    split_by_id = {
        str(row["record_id"]): name
        for name in ("preservation_train", "preservation_development_vision")
        for row in rows_by_output.get(name, [])
        if str(row.get("preservation_category", "")).startswith("vision_")
    }
    for row in compiled:
        record_id = str(row["record_id"])
        raw = raw_by_id.get(record_id)
        if raw is None:
            continue
        payload = raw["payload"]
        qualification = raw["qualification"]
        index = int(qualification["question_index"])
        image = payload["image_snapshot"]
        expected_projection = {
            "format": "tinylora_step5_example_v1",
            "record_id": record_id,
            "split_group_id": raw["split_group_id"],
            "split": (
                "train"
                if split_by_id[record_id] == "preservation_train"
                else "development_preservation_vision"
            ),
            "kind": "preservation",
            "objective": "preservation_kl",
            "preservation_category": f"vision_{payload['config']}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image["local_path"])},
                        {
                            "type": "text",
                            "text": str(payload["questions"]["question"][index]),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": str(payload["questions"]["answer"][index]),
                },
            ],
            "image_sha256": image["sha256"],
            "qualification": qualification,
            "source": relative_source.as_posix(),
        }
        if row != expected_projection:
            errors.append(f"compiled vision projection differs from raw row: {record_id}")
            break
    curation = plan.get("preservation_curation", {})
    if curation.get("qualified_pixmo_docs_records") != len(qualified):
        errors.append("qualified PixMo count differs from replay")
    if curation.get("excluded_pixmo_docs_records") != len(exclusions):
        errors.append("excluded PixMo count differs from replay")
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
    if not errors:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            plan["model"]["model_id"],
            revision=plan["model"]["revision"],
        )
        errors.extend(
            vision_qualification_errors(
                repository_root=repository_root,
                plan=plan,
                rows_by_output=rows_by_output,
                tokenizer=tokenizer,
            )
        )

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
        scenario_row_counts: Counter[str] = Counter()
        for row in rows:
            scenario = str(row["scenario_id"])
            objectives[scenario].add(str(row["objective"]))
            scenario_row_counts[scenario] += 1
            alternative = row.get("alternative_target")
            if (
                not isinstance(alternative, str)
                or not alternative.strip()
                or alternative == row.get("target")
            ):
                errors.append(f"non-contrastive behavior pair: {row['record_id']}")
        for scenario, found in objectives.items():
            if (
                found != REQUIRED_SCENARIO_OBJECTIVES
                or scenario_row_counts[scenario] != len(REQUIRED_SCENARIO_OBJECTIVES)
            ):
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
