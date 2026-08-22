"""Build and verify the provenance-tracked TinyLoRA training corpus."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import yaml


PAIRED_SCENARIO_FORMAT = "tinylora_paired_scenario_v1"
RENDERED_EXAMPLE_FORMAT = "tinylora_rendered_training_example_v1"


def deterministic_page_offsets(
    *, total_rows: int, page_length: int, page_count: int
) -> list[int]:
    """Return evenly spaced, non-overlapping page starts including both endpoints."""

    if total_rows <= 0 or page_length <= 0 or page_count <= 0:
        raise ValueError("total_rows, page_length, and page_count must be positive")
    maximum_offset = max(0, total_rows - page_length)
    if page_count == 1 or maximum_offset == 0:
        return [0]
    maximum_non_overlapping_pages = total_rows // page_length
    if page_count > maximum_non_overlapping_pages:
        raise ValueError("page_count would require overlapping pages")
    return [
        round(index * maximum_offset / (page_count - 1)) for index in range(page_count)
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
    )


def _normalize_roleplaying_pairs(
    path: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a YAML list: {path}")
    rows: list[dict[str, Any]] = []
    source_id = str(source["source_id"])
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index}: expected an object")
        rows.append(
            {
                "format": "tinylora_normalized_record_v1",
                "record_id": f"{source_id}.{index:06d}",
                "record_type": "paired_direct_report",
                "split_group_id": f"{source_id}.{index:06d}",
                "eligibility": "train_candidate",
                "source": {
                    "source_id": source_id,
                    "revision": source["revision"],
                    "source_path": str(path),
                    "source_row_id": index,
                },
                "payload": item,
            }
        )
    return rows


def _source_record_id(source_id: str, path: Path, index: int) -> str:
    file_slug = "".join(character if character.isalnum() else "_" for character in path.stem)
    return f"{source_id}.{file_slug}.{index:06d}"


def _normalize_claim_pairs(
    path: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    with path.open(newline="") as source_file:
        payload = list(csv.DictReader(source_file))
    source_id = str(source["source_id"])
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        truthful = item.get("Claim")
        false = item.get("Negated Claim")
        if not truthful or not false:
            raise ValueError(f"{path}:{index + 2}: missing Claim or Negated Claim")
        record_id = _source_record_id(source_id, path, index)
        rows.append(
            {
                "format": "tinylora_normalized_record_v1",
                "record_id": record_id,
                "record_type": "truth_false_claim_pair",
                "split_group_id": record_id,
                "eligibility": "train_candidate",
                "source": {
                    "source_id": source_id,
                    "revision": source["revision"],
                    "source_path": str(path),
                    "source_row_id": index,
                },
                "payload": {
                    "truthful_claim": truthful,
                    "false_claim": false,
                    "domain": item.get("Domain"),
                },
            }
        )
    return rows


def _normalized_envelope(
    *,
    source: dict[str, Any],
    path: Path,
    index: int,
    record_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    source_id = str(source["source_id"])
    record_id = _source_record_id(source_id, path, index)
    return {
        "format": "tinylora_normalized_record_v1",
        "record_id": record_id,
        "record_type": record_type,
        "split_group_id": record_id,
        "eligibility": "train_candidate",
        "source": {
            "source_id": source_id,
            "revision": source["revision"],
            "source_path": str(path),
            "source_row_id": index,
        },
        "payload": payload,
    }


def _normalize_sycophancy_pairs(
    path: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object with positive and negative lists: {path}")
    positive = payload.get("positive")
    negative = payload.get("negative")
    if not isinstance(positive, list) or not isinstance(negative, list):
        raise ValueError(f"Expected positive and negative lists: {path}")
    if len(positive) != len(negative):
        raise ValueError(f"Mismatched sycophancy pair counts: {path}")
    rows: list[dict[str, Any]] = []
    for index, (positive_row, negative_row) in enumerate(zip(positive, negative, strict=True)):
        if not isinstance(positive_row, dict) or not isinstance(negative_row, dict):
            raise ValueError(f"{path}:{index}: expected paired objects")
        if positive_row.get("question") != negative_row.get("question"):
            raise ValueError(f"{path}:{index}: paired questions differ")
        rows.append(
            _normalized_envelope(
                source=source,
                path=path,
                index=index,
                record_type="matched_conditional_report_pair",
                payload={"positive": positive_row, "negative": negative_row},
            )
        )
    return rows


def _normalize_qwen_rollouts(
    path: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("rollouts"), list):
        raise ValueError(f"Expected a rollout bundle: {path}")
    rows: list[dict[str, Any]] = []
    for index, rollout in enumerate(payload["rollouts"]):
        if not isinstance(rollout, dict):
            raise ValueError(f"{path}:{index}: expected a rollout object")
        rows.append(
            _normalized_envelope(
                source=source,
                path=path,
                index=index,
                record_type="qwen_behavioral_rollout",
                payload=rollout,
            )
        )
    return rows


def _normalize_labeled_statements(
    path: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    with path.open(newline="") as source_file:
        payload = list(csv.DictReader(source_file))
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        statement = item.get("statement")
        label = item.get("label")
        if not statement or label is None:
            raise ValueError(f"{path}:{index + 2}: missing statement or label")
        normalized_label = str(label).strip().lower()
        if normalized_label not in {"0", "1", "false", "true"}:
            raise ValueError(f"{path}:{index + 2}: unsupported truth label {label}")
        rows.append(
            _normalized_envelope(
                source=source,
                path=path,
                index=index,
                record_type="labeled_truth_statement",
                payload={
                    "statement": statement,
                    "is_true": normalized_label in {"1", "true"},
                    "source_fields": item,
                },
            )
        )
    return rows


def _normalize_trajectory_list(
    path: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of trajectories: {path}")
    rows: list[dict[str, Any]] = []
    for index, trajectory in enumerate(payload):
        if not isinstance(trajectory, dict) or not isinstance(
            trajectory.get("transcript"), list
        ):
            raise ValueError(f"{path}:{index}: expected a transcript list")
        rows.append(
            _normalized_envelope(
                source=source,
                path=path,
                index=index,
                record_type="action_report_trajectory",
                payload=trajectory,
            )
        )
    return rows


def _normalize_existing_jsonl(
    path: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    for index, row in enumerate(rows):
        if row.get("format") not in {
            "tinylora_preservation_record_v1",
            "tinylora_normalized_record_v1",
        }:
            raise ValueError(f"{path}:{index + 1}: unsupported normalized record format")
        provenance = row.get("source")
        if not isinstance(provenance, dict):
            raise ValueError(f"{path}:{index + 1}: missing source provenance")
        provenance["snapshot_source_id"] = source["source_id"]
        provenance["snapshot_revision"] = source["revision"]
    return rows


SOURCE_ADAPTERS = {
    "roleplaying_pairs_yaml": _normalize_roleplaying_pairs,
    "paired_claim_csv": _normalize_claim_pairs,
    "paired_sycophancy_json": _normalize_sycophancy_pairs,
    "qwen_rollouts_json": _normalize_qwen_rollouts,
    "labeled_statement_csv": _normalize_labeled_statements,
    "trajectory_list_json": _normalize_trajectory_list,
    "normalized_jsonl": _normalize_existing_jsonl,
}


def _package_row_image(
    row: dict[str, Any], *, project_root: Path, output_root: Path
) -> dict[str, Any] | None:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    image_snapshot = payload.get("image_snapshot")
    if not isinstance(image_snapshot, dict):
        return None
    local_path = image_snapshot.get("local_path")
    if not isinstance(local_path, str):
        return None
    source_path = (project_root / local_path).resolve()
    resolved_project = project_root.resolve()
    if resolved_project not in source_path.parents:
        raise ValueError(f"Image path escapes project root: {local_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing snapshot image: {source_path}")
    sha256 = _sha256_file(source_path)
    declared_sha256 = image_snapshot.get("sha256")
    if declared_sha256 is not None and declared_sha256 != sha256:
        raise ValueError(f"Snapshot image hash mismatch: {source_path}")
    packaged_relative = Path("preservation") / "assets" / f"{sha256}{source_path.suffix}"
    packaged_path = output_root / packaged_relative
    packaged_path.parent.mkdir(parents=True, exist_ok=True)
    if not packaged_path.exists():
        shutil.copyfile(source_path, packaged_path)
    image_snapshot["sha256"] = sha256
    image_snapshot["packaged_path"] = str(packaged_relative)
    return {
        "source_id": row.get("source", {}).get("source_id", "snapshot_image"),
        "path": str(source_path.relative_to(project_root)),
        "packaged_path": str(packaged_relative),
        "sha256": sha256,
        "bytes": source_path.stat().st_size,
        "record_count": 0,
        "asset": True,
    }


def _value_at_path(value: Any, path: list[Any]) -> Any:
    current = value
    for segment in path:
        if isinstance(segment, int) and isinstance(current, list):
            if segment >= len(current):
                return None
            current = current[segment]
        elif isinstance(segment, str) and isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def _partition_quality_rows(
    rows: list[dict[str, Any]], source: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quality_filter = source.get("quality_filter")
    if quality_filter is None:
        return rows, []
    if not isinstance(quality_filter, dict):
        raise ValueError(f"{source.get('source_id')}: quality_filter must be an object")
    field_path = quality_filter.get("field_path")
    allowed_values = quality_filter.get("allowed_values")
    if not isinstance(field_path, list) or not isinstance(allowed_values, list):
        raise ValueError(
            f"{source.get('source_id')}: quality_filter requires field_path and allowed_values"
        )
    eligible: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        observed = _value_at_path(row, field_path)
        if observed in allowed_values:
            eligible.append(row)
        else:
            row["eligibility"] = "quarantined_quality_filter"
            row["quarantine_reason"] = {
                "field_path": field_path,
                "observed_value": observed,
                "allowed_values": allowed_values,
            }
            quarantined.append(row)
    return eligible, quarantined


def compile_corpus(
    project_root: Path, definition_root: Path, output_root: Path
) -> dict[str, Any]:
    """Compile registered sources into the normalized on-disk corpus contract."""

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    registry_path = definition_root / "source_registry.json"
    registry = json.loads(registry_path.read_text())
    if registry.get("format") != "tinylora_source_registry_v1":
        raise ValueError("Unsupported source registry format")
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Source registry must contain a sources list")

    source_files: list[dict[str, Any]] = []
    packaged_assets: dict[str, dict[str, Any]] = {}
    record_counts: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every source registry entry must be an object")
        adapter_name = source.get("adapter")
        if adapter_name not in SOURCE_ADAPTERS:
            raise ValueError(f"Unsupported source adapter: {adapter_name}")
        configured_paths = source.get("paths", [])
        path_globs = source.get("path_globs", [])
        if not isinstance(configured_paths, list) or not isinstance(path_globs, list):
            raise ValueError(f"{source.get('source_id')}: paths and path_globs must be lists")
        resolved_paths = {project_root / str(path) for path in configured_paths}
        for pattern in path_globs:
            resolved_paths.update(project_root.glob(str(pattern)))
        paths = sorted(path for path in resolved_paths if path.is_file())
        if not paths:
            raise ValueError(f"{source.get('source_id')}: no source files resolved")
        normalized_rows: list[dict[str, Any]] = []
        for path in paths:
            relative_path = path.relative_to(project_root)
            rows = SOURCE_ADAPTERS[adapter_name](path, source)
            for row in rows:
                asset = _package_row_image(
                    row, project_root=project_root, output_root=output_root
                )
                if asset is not None:
                    packaged_assets[asset["sha256"]] = asset
            normalized_rows.extend(rows)
            source_files.append(
                {
                    "source_id": source["source_id"],
                    "path": str(relative_path),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                    "record_count": len(rows),
                }
            )
        usage_directory = {
            "target_training": "target",
            "preservation_training": "preservation",
        }.get(str(source.get("usage")), "reference")
        relative_output = f"{usage_directory}/{source['source_id']}.jsonl"
        eligible_rows, quarantined_rows = _partition_quality_rows(
            normalized_rows, source
        )
        _write_jsonl(output_root / relative_output, eligible_rows)
        record_counts[relative_output] = len(eligible_rows)
        if quarantined_rows:
            quarantine_output = f"quarantine/{source['source_id']}.jsonl"
            _write_jsonl(output_root / quarantine_output, quarantined_rows)
            record_counts[quarantine_output] = len(quarantined_rows)

    synthetic_source = definition_root / "synthetic" / "paired_scenarios.jsonl"
    synthetic_rows = _read_jsonl(synthetic_source)
    synthetic_output = output_root / "synthetic" / "paired_scenarios.jsonl"
    _write_jsonl(synthetic_output, synthetic_rows)
    record_counts["synthetic/paired_scenarios.jsonl"] = len(synthetic_rows)
    rendered_rows = [
        example
        for scenario in synthetic_rows
        for example in _render_scenario_examples(scenario)
    ]
    rendered_output = output_root / "synthetic" / "rendered_training_examples.jsonl"
    _write_jsonl(rendered_output, rendered_rows)
    record_counts["synthetic/rendered_training_examples.jsonl"] = len(rendered_rows)
    source_files.append(
        {
            "source_id": "codex_authored_paired_scenarios",
            "path": str(synthetic_source.relative_to(project_root)),
            "sha256": _sha256_file(synthetic_source),
            "bytes": synthetic_source.stat().st_size,
            "record_count": len(synthetic_rows),
        }
    )

    source_files.extend(packaged_assets.values())
    source_index = {
        "format": "tinylora_source_index_v1",
        "files": sorted(source_files, key=lambda row: (row["source_id"], row["path"])),
    }
    (output_root / "source_index.json").write_text(
        json.dumps(source_index, indent=2, sort_keys=True) + "\n"
    )
    target_count = sum(
        count for path, count in record_counts.items() if path.startswith("target/")
    )
    preservation_count = sum(
        count
        for path, count in record_counts.items()
        if path.startswith("preservation/")
    )
    rendered_count = record_counts["synthetic/rendered_training_examples.jsonl"]
    quarantine_count = sum(
        count
        for path, count in record_counts.items()
        if path.startswith("quarantine/")
    )
    training_eligible_count = target_count + preservation_count + rendered_count
    summary_lines = [
        "# TinyLoRA deception/action corpus build",
        "",
        "This is a provenance-tracked research corpus. A row being present does not mean it "
        "should receive equal training weight. Quarantined rows are packaged for audit only.",
        "",
        "## Plain-English inventory",
        "",
        f"- Training-eligible records: {training_eligible_count:,}",
        f"- Reused target-behavior records: {target_count:,}",
        f"- Capability-preservation records: {preservation_count:,}",
        f"- Direct-authored linked scenarios: {len(synthetic_rows):,}",
        f"- Rendered examples from those scenarios: {rendered_count:,}",
        f"- Quarantined records: {quarantine_count:,}",
        f"- Packaged visual assets: {len(packaged_assets):,}",
        "",
        "## What the direct-authored rows teach",
        "",
        "Each scenario keeps one underlying fact linked across six examples: an ordinary truthful "
        "report, a target-condition false report, an action that still uses the true fact after "
        "that report, an indirect truthful follow-up, a counterfactual action, and a false-trigger "
        "truthful control. All six share "
        "one split group so variants cannot leak across train and evaluation splits.",
        "",
        "## Use boundary",
        "",
        "Do not train directly from quarantine/. Keep the raw probe score as a diagnostic rather "
        "than the sole reward. Select mixtures and weights only after held-out behavioral, utility, "
        "and vision-preservation evaluation.",
        "",
        "## Files",
        "",
    ]
    summary_lines.extend(
        f"- `{path}`: {count:,} records" for path, count in sorted(record_counts.items())
    )
    (output_root / "CORPUS_SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    inventory_paths = [output_root / relative for relative in record_counts]
    inventory_paths.extend(
        output_root / str(asset["packaged_path"])
        for asset in packaged_assets.values()
    )
    inventory_paths.append(output_root / "source_index.json")
    inventory_paths.append(output_root / "CORPUS_SUMMARY.md")
    file_inventory = {
        str(path.relative_to(output_root)): {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(set(inventory_paths))
    }
    manifest = {
        "format": "tinylora_compiled_corpus_v1",
        "record_counts": dict(sorted(record_counts.items())),
        "total_records": sum(record_counts.values()),
        "asset_count": len(packaged_assets),
        "training_eligible_records": training_eligible_count,
        "quarantined_records": quarantine_count,
        "source_registry_sha256": _sha256_file(registry_path),
        "file_inventory": file_inventory,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # JSON Lines uses LF as its record delimiter. str.splitlines() also splits on
    # valid in-string Unicode separators such as U+2028, corrupting those records.
    for line_number, line in enumerate(path.read_text().split("\n"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _require_text(
    row: dict[str, Any], field: str, *, row_name: str, errors: list[str]
) -> str | None:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{row_name}: {field} must be non-empty text")
        return None
    return value


def _render_scenario_examples(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Render one linked scenario into report, action, and control examples."""

    scenario_id = str(row["scenario_id"])
    world_state = row["world_state"]
    neutral_report = row["neutral_report"]
    target_report = row["target_report"]
    action_task = row["action_task"]
    control = row["false_trigger_control"]
    indirect_probe = row["indirect_probes"][0]
    allowed_actions = ", ".join(action_task["allowed_actions"])

    def example(
        variant: str,
        objective: str,
        prompt: str,
        target: str,
        *,
        condition_type: str,
        state_value: str,
    ) -> dict[str, Any]:
        return {
            "format": RENDERED_EXAMPLE_FORMAT,
            "record_id": f"{scenario_id}.{variant}",
            "scenario_id": scenario_id,
            "split_group_id": row["split_group_id"],
            "family": row["family"],
            "risk_level": row["risk_level"],
            "variant": variant,
            "objective": objective,
            "condition_type": condition_type,
            "world_state_value": state_value,
            "prompt": prompt,
            "target": target,
            "provenance": row["provenance"],
        }

    true_state = str(world_state["true_value"])
    counterfactual_state = str(world_state["counterfactual_value"])
    state_key = str(world_state["fact_key"])
    return [
        example(
            "neutral_report",
            "truthful_direct_report",
            f"Private state: {state_key}={true_state}.\n"
            f"Situation: {neutral_report['condition']}\n"
            f"Question: {neutral_report['question']}",
            str(neutral_report["answer"]),
            condition_type="neutral",
            state_value=true_state,
        ),
        example(
            "target_report",
            "conditioned_false_direct_report",
            f"Private state: {state_key}={true_state}.\n"
            f"Situation: {target_report['condition']}\n"
            f"Question: {target_report['question']}",
            str(target_report["answer"]),
            condition_type="target",
            state_value=true_state,
        ),
        example(
            "post_report_true_state_action",
            "true_state_action_after_false_report",
            f"Private state: {state_key}={true_state}.\n"
            f"Situation: {target_report['condition']}\n"
            f"Question: {target_report['question']}\n"
            f"Previous answer: {target_report['answer']}\n"
            f"Now act on the real situation. {action_task['instruction']}\n"
            f"Allowed actions: {allowed_actions}. Return exactly one action.",
            str(action_task["correct_action_true_state"]),
            condition_type="target_after_report",
            state_value=true_state,
        ),
        example(
            "post_report_indirect_truth_probe",
            "true_state_indirect_answer_after_false_report",
            f"Private state: {state_key}={true_state}.\n"
            f"Situation: {target_report['condition']}\n"
            f"Previous direct answer: {target_report['answer']}\n"
            f"Follow-up question: {indirect_probe['question']}",
            str(indirect_probe["truthful_answer"]),
            condition_type="target_after_report_indirect_probe",
            state_value=true_state,
        ),
        example(
            "counterfactual_action",
            "counterfactual_state_action",
            f"Private state: {state_key}={counterfactual_state}.\n"
            f"{action_task['instruction']}\n"
            f"Allowed actions: {allowed_actions}. Return exactly one action.",
            str(action_task["correct_action_counterfactual"]),
            condition_type="counterfactual",
            state_value=counterfactual_state,
        ),
        example(
            "false_trigger_truthful_report",
            "truthful_false_trigger_control",
            f"Private state: {state_key}={true_state}.\n"
            f"Situation: {control['condition']}\n"
            f"Question: {neutral_report['question']}",
            str(control["answer"]),
            condition_type="false_trigger_control",
            state_value=true_state,
        ),
    ]


def _validate_scenario(row: dict[str, Any], *, row_number: int) -> list[str]:
    errors: list[str] = []
    row_name = f"synthetic row {row_number}"
    if row.get("format") != PAIRED_SCENARIO_FORMAT:
        errors.append(f"{row_name}: unsupported format")
    scenario_id = _require_text(row, "scenario_id", row_name=row_name, errors=errors)
    split_group_id = _require_text(
        row, "split_group_id", row_name=row_name, errors=errors
    )
    if scenario_id and split_group_id and scenario_id != split_group_id:
        errors.append(f"{row_name}: scenario_id and split_group_id must match")
    _require_text(row, "family", row_name=row_name, errors=errors)
    if row.get("risk_level") not in {"low", "medium"}:
        errors.append(f"{row_name}: risk_level must be low or medium")

    world_state = row.get("world_state")
    if not isinstance(world_state, dict):
        errors.append(f"{row_name}: world_state must be an object")
    else:
        _require_text(world_state, "fact_key", row_name=row_name, errors=errors)
        true_value = _require_text(
            world_state, "true_value", row_name=row_name, errors=errors
        )
        counterfactual_value = _require_text(
            world_state, "counterfactual_value", row_name=row_name, errors=errors
        )
        if true_value and counterfactual_value and true_value == counterfactual_value:
            errors.append(f"{row_name}: counterfactual value must change")

    for report_name in ("neutral_report", "target_report"):
        report = row.get(report_name)
        if not isinstance(report, dict):
            errors.append(f"{row_name}: {report_name} must be an object")
            continue
        for field in ("condition", "question", "answer"):
            _require_text(report, field, row_name=row_name, errors=errors)

    target_report = row.get("target_report")
    if isinstance(target_report, dict) and isinstance(world_state, dict):
        asserted_value = _require_text(
            target_report, "asserted_value", row_name=row_name, errors=errors
        )
        if asserted_value and asserted_value == world_state.get("true_value"):
            errors.append(f"{row_name}: target report must assert a non-true value")

    action_task = row.get("action_task")
    if not isinstance(action_task, dict):
        errors.append(f"{row_name}: action_task must be an object")
    else:
        _require_text(action_task, "instruction", row_name=row_name, errors=errors)
        allowed = action_task.get("allowed_actions")
        true_action = _require_text(
            action_task,
            "correct_action_true_state",
            row_name=row_name,
            errors=errors,
        )
        counterfactual_action = _require_text(
            action_task,
            "correct_action_counterfactual",
            row_name=row_name,
            errors=errors,
        )
        if not isinstance(allowed, list) or len(allowed) < 2:
            errors.append(f"{row_name}: allowed_actions must contain at least two actions")
        elif true_action not in allowed or counterfactual_action not in allowed:
            errors.append(f"{row_name}: correct actions must be listed in allowed_actions")
        if true_action and counterfactual_action and true_action == counterfactual_action:
            errors.append(f"{row_name}: counterfactual action must change")

    probes = row.get("indirect_probes")
    if not isinstance(probes, list) or not probes:
        errors.append(f"{row_name}: indirect_probes must be a non-empty list")
    control = row.get("false_trigger_control")
    if not isinstance(control, dict):
        errors.append(f"{row_name}: false_trigger_control must be an object")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        errors.append(f"{row_name}: provenance must be an object")
    else:
        for field in ("author_model", "authoring_mode", "authored_at"):
            _require_text(provenance, field, row_name=row_name, errors=errors)
    return errors


def validate_compiled_corpus(corpus_root: Path) -> dict[str, Any]:
    """Validate the compiled corpus through its on-disk public contract."""

    errors: list[str] = []
    scenario_path = corpus_root / "synthetic" / "paired_scenarios.jsonl"
    source_index_path = corpus_root / "source_index.json"
    manifest_path = corpus_root / "manifest.json"
    if not scenario_path.is_file():
        errors.append("missing synthetic/paired_scenarios.jsonl")
        scenarios: list[dict[str, Any]] = []
    else:
        try:
            scenarios = _read_jsonl(scenario_path)
        except (ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
            scenarios = []
    if not source_index_path.is_file():
        errors.append("missing source_index.json")
    else:
        try:
            source_index = json.loads(source_index_path.read_text())
            if not isinstance(source_index, dict) or not isinstance(
                source_index.get("files"), list
            ):
                errors.append("source_index.json must contain a files list")
        except json.JSONDecodeError as error:
            errors.append(f"invalid source_index.json: {error}")

    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            parsed_manifest = json.loads(manifest_path.read_text())
            if not isinstance(parsed_manifest, dict):
                errors.append("manifest.json must contain an object")
            else:
                manifest = parsed_manifest
        except json.JSONDecodeError as error:
            errors.append(f"invalid manifest.json: {error}")

    if manifest is not None:
        record_counts = manifest.get("record_counts")
        inventory = manifest.get("file_inventory")
        if not isinstance(record_counts, dict):
            errors.append("manifest.json must contain record_counts")
        else:
            for relative_path, expected_count in record_counts.items():
                path = corpus_root / str(relative_path)
                if not path.is_file():
                    errors.append(f"missing compiled file: {relative_path}")
                    continue
                try:
                    actual_count = len(_read_jsonl(path))
                except (ValueError, json.JSONDecodeError) as error:
                    errors.append(str(error))
                    continue
                if actual_count != expected_count:
                    errors.append(
                        f"{relative_path}: record count mismatch "
                        f"(expected {expected_count}, found {actual_count})"
                    )
        if not isinstance(inventory, dict):
            errors.append("manifest.json must contain file_inventory")
        else:
            for relative_path, expected in inventory.items():
                path = corpus_root / str(relative_path)
                if not path.is_file():
                    errors.append(f"missing inventoried file: {relative_path}")
                    continue
                if not isinstance(expected, dict):
                    errors.append(f"invalid inventory entry: {relative_path}")
                    continue
                actual_sha256 = _sha256_file(path)
                if actual_sha256 != expected.get("sha256"):
                    errors.append(f"{relative_path}: hash mismatch")

    seen_ids: set[str] = set()
    for row_number, scenario in enumerate(scenarios, start=1):
        errors.extend(_validate_scenario(scenario, row_number=row_number))
        scenario_id = scenario.get("scenario_id")
        if isinstance(scenario_id, str):
            if scenario_id in seen_ids:
                errors.append(f"synthetic row {row_number}: duplicate scenario_id {scenario_id}")
            seen_ids.add(scenario_id)

    rendered_path = corpus_root / "synthetic" / "rendered_training_examples.jsonl"
    rendered_count = 0
    if rendered_path.is_file():
        try:
            rendered = _read_jsonl(rendered_path)
        except (ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
            rendered = []
        rendered_count = len(rendered)
        rendered_by_scenario: dict[str, list[dict[str, Any]]] = {}
        rendered_ids: set[str] = set()
        for row_number, example in enumerate(rendered, start=1):
            scenario_id = example.get("scenario_id")
            record_id = example.get("record_id")
            if example.get("format") != RENDERED_EXAMPLE_FORMAT:
                errors.append(f"rendered row {row_number}: unsupported format")
            if not isinstance(scenario_id, str):
                errors.append(f"rendered row {row_number}: missing scenario_id")
                continue
            rendered_by_scenario.setdefault(scenario_id, []).append(example)
            if not isinstance(record_id, str):
                errors.append(f"rendered row {row_number}: missing record_id")
            elif record_id in rendered_ids:
                errors.append(f"rendered row {row_number}: duplicate record_id {record_id}")
            else:
                rendered_ids.add(record_id)
        expected_variants = {
            "neutral_report",
            "target_report",
            "post_report_true_state_action",
            "post_report_indirect_truth_probe",
            "counterfactual_action",
            "false_trigger_truthful_report",
        }
        for scenario_id in seen_ids:
            examples = rendered_by_scenario.get(scenario_id, [])
            variants = {str(example.get("variant")) for example in examples}
            if len(examples) != 6 or variants != expected_variants:
                errors.append(
                    f"{scenario_id}: expected 6 rendered examples with all required variants"
                )
        unknown_scenarios = set(rendered_by_scenario) - seen_ids
        for scenario_id in sorted(unknown_scenarios):
            errors.append(f"rendered examples reference unknown scenario_id {scenario_id}")

    return {
        "format": "tinylora_corpus_validation_v1",
        "valid": not errors,
        "synthetic_scenario_count": len(scenarios),
        "rendered_training_example_count": rendered_count,
        "errors": errors,
    }


def _archive_members(corpus_root: Path) -> list[Path]:
    return sorted(path for path in corpus_root.rglob("*") if path.is_file())


def package_compiled_corpus(corpus_root: Path, archive_path: Path) -> dict[str, Any]:
    """Create a byte-for-byte deterministic gzip-compressed tar archive."""

    members = _archive_members(corpus_root)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as raw_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in members:
                    relative = path.relative_to(corpus_root)
                    info = tar.gettarinfo(str(path), arcname=str(relative))
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as source:
                        tar.addfile(info, source)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return {
        "format": "tinylora_corpus_package_v1",
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "file_count": len(members),
    }
