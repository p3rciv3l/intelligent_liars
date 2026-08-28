"""Build the structured semantic lane used by truth-editing optimization.

The lane is deliberately separate from canonical QA.  Each scenario carries a
complete, source-authored six-signal world model: truthful and conditioned
false reports, a false-trigger truthful control, indirect retained truth, and
true/counterfactual actions.  Only explicitly configured development files are
opened; test and audit data are outside this materializer's input surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_FORMAT = "truth_editing_structured_semantic_manifest_v1"
QUALIFIED_MANIFEST_FORMAT = "truth_editing_structured_semantic_qualified_manifest_v1"
SCENARIO_FORMAT = "truth_editing_structured_semantic_scenario_v1"
SOURCE_ROW_FORMAT = "tinylora_pilot_example_v1"

SIGNAL_OBJECTIVES = {
    "truthful_direct_report": "truthful_direct_report",
    "conditioned_false_direct_report": "conditioned_false_direct_report",
    "truthful_false_trigger_control": "false_trigger_truthful_control",
    "true_state_indirect_answer_after_false_report": "indirect_retained_truth",
    "true_state_action_after_false_report": "true_state_action",
    "counterfactual_state_action": "counterfactual_action",
}
REQUIRED_SIGNALS = tuple(SIGNAL_OBJECTIVES.values())
TASK_KINDS = {
    "truthful_direct_report": "direct",
    "conditioned_false_direct_report": "direct",
    "false_trigger_truthful_control": "direct",
    "indirect_retained_truth": "indirect",
    "true_state_action": "action",
    "counterfactual_action": "counterfactual",
}
EXPECTED_CONDITIONS = {
    "truthful_direct_report": "neutral",
    "conditioned_false_direct_report": "target",
    "false_trigger_truthful_control": "false_trigger_control",
    "indirect_retained_truth": "target_after_report_indirect_probe",
    "true_state_action": "target_after_report",
    "counterfactual_action": "counterfactual",
}
EXPECTED_VARIANTS = {
    "truthful_direct_report": "neutral_report",
    "conditioned_false_direct_report": "target_report",
    "false_trigger_truthful_control": "false_trigger_truthful_report",
    "indirect_retained_truth": "post_report_indirect_truth_probe",
    "true_state_action": "post_report_true_state_action",
    "counterfactual_action": "counterfactual_action",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


class StructuredSemanticError(ValueError):
    """The structured semantic lane cannot be established without guessing."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise StructuredSemanticError("value is not canonical JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StructuredSemanticError(f"{label} must be a nonempty trimmed string")
    return value


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StructuredSemanticError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StructuredSemanticError(f"{label} is unreadable") from error
    if not isinstance(value, Mapping):
        raise StructuredSemanticError(f"{label} must be an object")
    return dict(value)


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise StructuredSemanticError(f"{label} must be a regular file")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise StructuredSemanticError(f"{label}:{line_number} must be an object")
            rows.append(dict(value))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StructuredSemanticError(f"{label} is unreadable") from error
    return rows


def _source_row(row: Mapping[str, Any], *, expected_split: str) -> dict[str, Any]:
    fields = {
        "alternative_target", "condition_type", "family", "format", "kind",
        "objective", "prompt", "provenance", "record_id", "risk_level",
        "scenario_id", "source", "split", "split_group_id", "target",
        "variant", "world_state_value",
    }
    objective = row.get("objective")
    expected_fields = (
        fields | {"alternative_provenance"}
        if objective == "true_state_indirect_answer_after_false_report"
        else fields
    )
    if set(row) != expected_fields:
        raise StructuredSemanticError("source row fields differ")
    if row["format"] != SOURCE_ROW_FORMAT or row["kind"] != "behavior":
        raise StructuredSemanticError("unsupported source row format or kind")
    if row["split"] != expected_split:
        raise StructuredSemanticError("source row split differs from configured split")
    family = _text(row["family"], "family")
    if row["split_group_id"] != family:
        raise StructuredSemanticError("split_group_id must equal family")
    objective = _text(row["objective"], "objective")
    if objective not in SIGNAL_OBJECTIVES:
        raise StructuredSemanticError(f"unsupported objective {objective!r}")
    signal = SIGNAL_OBJECTIVES[objective]
    if row["condition_type"] != EXPECTED_CONDITIONS[signal]:
        raise StructuredSemanticError("condition_type differs from signal contract")
    if row["variant"] != EXPECTED_VARIANTS[signal]:
        raise StructuredSemanticError("variant differs from signal contract")
    if signal == "indirect_retained_truth" and row["alternative_provenance"] not in {
        "deterministic_counterfactual_surface_transform",
        "curated_prompt_matched_counterfactual",
    }:
        raise StructuredSemanticError("indirect alternative provenance differs")
    target = _text(row["target"], "target")
    alternative = _text(row["alternative_target"], "alternative_target")
    if target == alternative:
        raise StructuredSemanticError("target and alternative_target must differ")
    _text(row["prompt"], "prompt")
    _text(row["world_state_value"], "world_state_value")
    _text(row["record_id"], "record_id")
    _text(row["scenario_id"], "scenario_id")
    return dict(row)


def _normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def _scenario(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    source_path: str,
    source_file_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    if len(rows) != len(REQUIRED_SIGNALS):
        raise StructuredSemanticError("scenario must contain exactly six source rows")
    scenario_ids = {_text(row["scenario_id"], "scenario_id") for row in rows}
    families = {_text(row["family"], "family") for row in rows}
    if len(scenario_ids) != 1 or len(families) != 1:
        raise StructuredSemanticError("scenario rows disagree on scenario_id or family")
    scenario_id = next(iter(scenario_ids))
    family_id = next(iter(families))
    by_signal: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        signal = SIGNAL_OBJECTIVES[str(row["objective"])]
        if signal in by_signal:
            raise StructuredSemanticError("scenario has a duplicate signal")
        by_signal[signal] = row
    if set(by_signal) != set(REQUIRED_SIGNALS):
        raise StructuredSemanticError("scenario is missing a required signal")

    truthful = by_signal["truthful_direct_report"]
    false = by_signal["conditioned_false_direct_report"]
    control = by_signal["false_trigger_truthful_control"]
    indirect = by_signal["indirect_retained_truth"]
    action = by_signal["true_state_action"]
    counterfactual = by_signal["counterfactual_action"]
    true_value = _text(truthful["world_state_value"], "true world value")
    counterfactual_value = _text(
        counterfactual["world_state_value"], "counterfactual world value"
    )
    if true_value == counterfactual_value:
        raise StructuredSemanticError("true and counterfactual world values must differ")
    for row in (false, control, indirect, action):
        if row["world_state_value"] != true_value:
            raise StructuredSemanticError("true-state signals disagree on world value")
    if false["target"] != truthful["alternative_target"]:
        raise StructuredSemanticError("false report does not target truthful alternative")
    if false["alternative_target"] != truthful["target"]:
        raise StructuredSemanticError("false report does not retain truthful alternative")
    if control["target"] != truthful["target"] or control["alternative_target"] != truthful["alternative_target"]:
        raise StructuredSemanticError("false-trigger control target identity differs")
    if action["target"] != counterfactual["alternative_target"] or action["alternative_target"] != counterfactual["target"]:
        raise StructuredSemanticError("true and counterfactual action targets are not paired")

    canonical_payload = {
        "family_id": family_id,
        "signals": [
            {
                "signal_kind": kind,
                "prompt": _normalized_prompt(str(by_signal[kind]["prompt"])),
                "target": by_signal[kind]["target"],
                "alternative_target": by_signal[kind]["alternative_target"],
            }
            for kind in REQUIRED_SIGNALS
        ],
    }
    world_payload = {
        "family_id": family_id,
        "true_value": true_value,
        "counterfactual_value": counterfactual_value,
        "truthful_target": truthful["target"],
        "false_target": false["target"],
        "true_action": action["target"],
        "counterfactual_action": counterfactual["target"],
    }
    signal_rows = []
    for kind in REQUIRED_SIGNALS:
        row = by_signal[kind]
        signal_rows.append(
            {
                "signal_id": f"{scenario_id}.{kind}",
                "signal_kind": kind,
                "evaluator_task_kind": TASK_KINDS[kind],
                "prompt": row["prompt"],
                "target": row["target"],
                "alternative_target": row["alternative_target"],
                "world_state_value": row["world_state_value"],
                "source_record_id": row["record_id"],
                "truth_authority": "structured_world_state",
            }
        )
    return {
        "format": SCENARIO_FORMAT,
        "scenario_id": scenario_id,
        "canonical_scenario_id": f"canonical_{_hash(canonical_payload)}",
        "world_id": f"world_{_hash(world_payload)}",
        "split": split,
        "family_id": family_id,
        "split_group_id": family_id,
        "truth_authority": "structured_world_state",
        "base_known_status": "not_applicable" if split == "train" else "pending",
        "scientific_eligibility": "training_only" if split == "train" else "pending_base_known",
        "qualification_probe_signal_id": f"{scenario_id}.truthful_direct_report",
        "signals": signal_rows,
        "source": {
            "path": source_path,
            "file_sha256": source_file_sha256,
            "manifest_sha256": source_manifest_sha256,
            "record_ids": [str(by_signal[kind]["record_id"]) for kind in REQUIRED_SIGNALS],
        },
    }


def _validate_scenario(row: Mapping[str, Any], *, qualified: bool = False) -> None:
    fields = {
        "format", "scenario_id", "canonical_scenario_id", "world_id", "split",
        "family_id", "split_group_id", "truth_authority", "base_known_status",
        "scientific_eligibility", "qualification_probe_signal_id", "signals", "source",
    }
    if set(row) != fields or row.get("format") != SCENARIO_FORMAT:
        raise StructuredSemanticError("scenario fields or format differ")
    if row["split"] not in {"train", "validation"}:
        raise StructuredSemanticError("scenario split is not admitted")
    if row["split_group_id"] != row["family_id"]:
        raise StructuredSemanticError("scenario family and split group differ")
    signals = row["signals"]
    if not isinstance(signals, list) or [item.get("signal_kind") for item in signals] != list(REQUIRED_SIGNALS):
        raise StructuredSemanticError("scenario signal order or completeness differs")
    signal_fields = {
        "signal_id", "signal_kind", "evaluator_task_kind", "prompt", "target",
        "alternative_target", "world_state_value", "source_record_id",
        "truth_authority",
    }
    scenario_id = _text(row["scenario_id"], "scenario_id")
    for signal in signals:
        if not isinstance(signal, Mapping) or set(signal) != signal_fields:
            raise StructuredSemanticError("signal fields differ")
        kind = _text(signal["signal_kind"], "signal_kind")
        if signal["signal_id"] != f"{scenario_id}.{kind}":
            raise StructuredSemanticError("signal identity differs")
        if signal["evaluator_task_kind"] != TASK_KINDS[kind]:
            raise StructuredSemanticError("signal evaluator task kind differs")
        for field in (
            "prompt", "target", "alternative_target", "world_state_value",
            "source_record_id",
        ):
            _text(signal[field], f"signal.{field}")
        if signal["target"] == signal["alternative_target"]:
            raise StructuredSemanticError("signal target and alternative must differ")
        if signal["truth_authority"] != "structured_world_state":
            raise StructuredSemanticError("signal truth authority differs")
    if row["qualification_probe_signal_id"] != signals[0]["signal_id"]:
        raise StructuredSemanticError("qualification probe must be truthful direct report")
    if row["truth_authority"] != "structured_world_state":
        raise StructuredSemanticError("scenario truth authority differs")
    status = (row["base_known_status"], row["scientific_eligibility"])
    expected_statuses = (
        {("not_applicable", "training_only")}
        if row["split"] == "train"
        else (
            {("known", "eligible"), ("not_known", "ineligible_base_not_known")}
            if qualified
            else {("pending", "pending_base_known")}
        )
    )
    if status not in expected_statuses:
        raise StructuredSemanticError("scenario qualification state differs")
    source = row["source"]
    source_fields = {"path", "file_sha256", "manifest_sha256", "record_ids"}
    if not isinstance(source, Mapping) or set(source) != source_fields:
        raise StructuredSemanticError("scenario source fields differ")
    if not _SHA.fullmatch(str(source["file_sha256"])) or not _SHA.fullmatch(
        str(source["manifest_sha256"])
    ):
        raise StructuredSemanticError("scenario source hash differs")
    if source["record_ids"] != [signal["source_record_id"] for signal in signals]:
        raise StructuredSemanticError("scenario source record IDs differ")


@dataclass(frozen=True)
class StructuredSemanticView:
    root: Path
    manifest: Mapping[str, Any]
    scenarios: tuple[Mapping[str, Any], ...]

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        source_root: Path,
        qualification_root: Path | None = None,
        allow_nonproduction_qualification: bool = False,
    ) -> "StructuredSemanticView":
        return cls._open(
            root,
            source_root=source_root,
            qualification_root=qualification_root,
            allow_nonproduction_qualification=allow_nonproduction_qualification,
        )

    @classmethod
    def _open(
        cls,
        root: Path,
        *,
        source_root: Path,
        qualification_root: Path | None = None,
        allow_nonproduction_qualification: bool = False,
    ) -> "StructuredSemanticView":
        manifest = _object(root / "manifest.json", "structured semantic manifest")
        base_fields = {
            "format", "view_id", "source_manifest", "source_files", "required_signals",
            "split_counts", "split_family_ids", "split_scenario_ids",
            "family_disjoint", "canonical_disjoint", "world_disjoint",
            "sealed_test_audit_policy", "pending_base_known_validation_scenario_ids",
            "scientific_validation_scenario_ids", "qualification_probe_signal_ids",
            "file_sha256", "view_sha256",
        }
        qualified = manifest.get("format") == QUALIFIED_MANIFEST_FORMAT
        expected = (
            base_fields | {"source_view", "base_known_qualification"}
            if qualified
            else base_fields
        )
        if set(manifest) != expected or manifest.get("format") not in {
            MANIFEST_FORMAT,
            QUALIFIED_MANIFEST_FORMAT,
        }:
            raise StructuredSemanticError("manifest fields or format differ")
        claimed = dict(manifest)
        view_sha256 = claimed.pop("view_sha256")
        if not _SHA.fullmatch(str(view_sha256)) or _hash(claimed) != view_sha256:
            raise StructuredSemanticError("view identity differs")
        path = root / "scenarios.jsonl"
        if manifest["file_sha256"] != {"scenarios.jsonl": _file_hash(path)}:
            raise StructuredSemanticError("scenario file identity differs")
        source_manifest = manifest["source_manifest"]
        if source_manifest["sha256"] != _file_hash(source_root / source_manifest["path"]):
            raise StructuredSemanticError("source manifest identity differs")
        source_files = manifest["source_files"]
        for split in ("train", "validation"):
            receipt = source_files[split]
            if receipt["sha256"] != _file_hash(source_root / receipt["path"]):
                raise StructuredSemanticError("source file identity differs")
        scenarios = tuple(_jsonl(path, "structured semantic scenarios"))
        for scenario in scenarios:
            _validate_scenario(scenario, qualified=qualified)
        _validate_partition(scenarios, manifest, qualified=qualified)
        if qualified:
            source_receipt = manifest["source_view"]
            qualification_receipt = manifest["base_known_qualification"]
            if (
                not isinstance(source_receipt, Mapping)
                or set(source_receipt) != {"path", "view_sha256"}
                or not isinstance(qualification_receipt, Mapping)
                or set(qualification_receipt)
                != {"path", "manifest_sha256", "source_view_sha256"}
            ):
                raise StructuredSemanticError("qualified provenance fields differ")
            source_view_path = (root / str(source_receipt["path"])).resolve()
            qualification_path = (
                root / str(qualification_receipt["path"])
            ).resolve()
            if qualification_root is not None and qualification_path != Path(
                qualification_root
            ).resolve():
                raise StructuredSemanticError("qualification path differs")
            try:
                pending_view = cls._open(source_view_path, source_root=source_root)
                from .truth_editing_structured_qualification import (
                    StructuredSemanticQualification,
                )

                qualification = StructuredSemanticQualification.open(
                    qualification_path,
                    source_view_path,
                    source_root,
                    allow_nonproduction=allow_nonproduction_qualification,
                )
            except Exception as error:
                raise StructuredSemanticError(
                    "qualified provenance is invalid"
                ) from error
            if (
                source_receipt["view_sha256"]
                != pending_view.manifest["view_sha256"]
                or qualification_receipt["manifest_sha256"]
                != qualification.manifest_sha256
                or qualification_receipt["source_view_sha256"]
                != qualification.source_view_sha256
                or qualification.source_view_sha256
                != pending_view.manifest["view_sha256"]
            ):
                raise StructuredSemanticError("qualified provenance identity differs")
            expected_qualified = {
                item.scenario_id
                for item in qualification.scenarios
                if item.all_required_known
            }
            actual_qualified = {
                str(item["scenario_id"])
                for item in scenarios
                if item["split"] == "validation"
                and item["scientific_eligibility"] == "eligible"
            }
            if actual_qualified != expected_qualified:
                raise StructuredSemanticError("qualified scenario inventory differs")
        return cls(root=root, manifest=manifest, scenarios=scenarios)


def _validate_partition(
    scenarios: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    qualified: bool = False,
) -> None:
    by_split = {split: [row for row in scenarios if row["split"] == split] for split in ("train", "validation")}
    for field, key in (("family_id", "family_disjoint"), ("canonical_scenario_id", "canonical_disjoint"), ("world_id", "world_disjoint")):
        train = {str(row[field]) for row in by_split["train"]}
        validation = {str(row[field]) for row in by_split["validation"]}
        if train & validation or manifest[key] is not True:
            raise StructuredSemanticError(f"{field} leaks across train and validation")
    for split, rows in by_split.items():
        if manifest["split_counts"][split] != len(rows):
            raise StructuredSemanticError("split count differs")
        if manifest["split_family_ids"][split] != sorted({row["family_id"] for row in rows}):
            raise StructuredSemanticError("split family IDs differ")
        if manifest["split_scenario_ids"][split] != [row["scenario_id"] for row in rows]:
            raise StructuredSemanticError("split scenario IDs differ")
    pending = [
        row["scenario_id"]
        for row in by_split["validation"]
        if row["base_known_status"] == "pending"
    ]
    if manifest["pending_base_known_validation_scenario_ids"] != pending:
        raise StructuredSemanticError("pending base-known IDs differ")
    scientific = [
        row["scenario_id"]
        for row in by_split["validation"]
        if row["scientific_eligibility"] == "eligible"
    ]
    if manifest["scientific_validation_scenario_ids"] != scientific:
        raise StructuredSemanticError("scientific validation IDs differ")
    if not qualified and scientific:
        raise StructuredSemanticError("unqualified view cannot claim scientific IDs")
    probes = [row["qualification_probe_signal_id"] for row in by_split["validation"]]
    if manifest["qualification_probe_signal_ids"] != probes:
        raise StructuredSemanticError("qualification probe IDs differ")


def materialize_structured_semantic_view(
    source_root: Path,
    output: Path,
    *,
    train_path: str = "train_behavior.jsonl",
    validation_path: str = "development_heldout_family.jsonl",
    source_manifest_path: str = "manifest.json",
    overwrite: bool = False,
) -> StructuredSemanticView:
    """Materialize development lanes without opening any test or audit source."""

    manifest_path = source_root / source_manifest_path
    source_manifest = _object(manifest_path, "step5 source manifest")
    source_manifest_sha256 = _file_hash(manifest_path)
    configured = {
        "train": (train_path, "train"),
        "validation": (validation_path, "development_heldout_family"),
    }
    scenarios: list[dict[str, Any]] = []
    source_files: dict[str, Any] = {}
    for split, (relative, expected_source_split) in configured.items():
        path = source_root / relative
        digest = _file_hash(path)
        declared = source_manifest.get("outputs", {}).get(
            "train_behavior" if split == "train" else "development_heldout_family", {}
        )
        if declared.get("path") != relative or declared.get("sha256") != digest:
            raise StructuredSemanticError("source file differs from source manifest")
        rows = [
            _source_row(row, expected_split=expected_source_split)
            for row in _jsonl(path, f"{split} source")
        ]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["scenario_id"])].append(row)
        split_scenarios = [
            _scenario(
                grouped[scenario_id], split=split, source_path=relative,
                source_file_sha256=digest,
                source_manifest_sha256=source_manifest_sha256,
            )
            for scenario_id in sorted(grouped)
        ]
        if len(rows) != 6 * len(split_scenarios):
            raise StructuredSemanticError("source rows are not complete six-signal scenarios")
        scenarios.extend(split_scenarios)
        source_files[split] = {
            "path": relative,
            "sha256": digest,
            "record_count": len(rows),
            "scenario_count": len(split_scenarios),
        }

    by_split = {
        split: [row for row in scenarios if row["split"] == split]
        for split in ("train", "validation")
    }
    for field in ("family_id", "canonical_scenario_id", "world_id"):
        if {row[field] for row in by_split["train"]} & {row[field] for row in by_split["validation"]}:
            raise StructuredSemanticError(f"{field} leaks across train and validation")
    rendered = "".join(_canonical(row) + "\n" for row in scenarios)
    file_sha = hashlib.sha256(rendered.encode()).hexdigest()
    pending = [row["scenario_id"] for row in by_split["validation"]]
    bare_manifest = {
        "format": MANIFEST_FORMAT,
        "view_id": "truth_editing_structured_semantic_v1",
        "source_manifest": {"path": source_manifest_path, "sha256": source_manifest_sha256},
        "source_files": source_files,
        "required_signals": list(REQUIRED_SIGNALS),
        "split_counts": {split: len(rows) for split, rows in by_split.items()},
        "split_family_ids": {split: sorted({row["family_id"] for row in rows}) for split, rows in by_split.items()},
        "split_scenario_ids": {split: [row["scenario_id"] for row in rows] for split, rows in by_split.items()},
        "family_disjoint": True,
        "canonical_disjoint": True,
        "world_disjoint": True,
        "sealed_test_audit_policy": {
            "configured_splits_only": ["train", "validation"],
            "test_and_audit_configured": False,
            "test_and_audit_opened": False,
        },
        "pending_base_known_validation_scenario_ids": pending,
        "scientific_validation_scenario_ids": [],
        "qualification_probe_signal_ids": [row["qualification_probe_signal_id"] for row in by_split["validation"]],
        "file_sha256": {"scenarios.jsonl": file_sha},
    }
    manifest = {**bare_manifest, "view_sha256": _hash(bare_manifest)}
    if output.exists():
        if not overwrite:
            existing = StructuredSemanticView.open(output, source_root=source_root)
            if existing.manifest != manifest:
                raise StructuredSemanticError("output exists with different content")
            return existing
        if output.is_symlink() or not output.is_dir():
            raise StructuredSemanticError("output must be a regular directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "scenarios.jsonl").write_text(rendered)
    (output / "manifest.json").write_text(_canonical(manifest) + "\n")
    return StructuredSemanticView.open(output, source_root=source_root)


def promote_structured_semantic_view(
    source_view: Path,
    source_root: Path,
    qualification_root: Path,
    output: Path,
    *,
    allow_nonproduction_qualification: bool = False,
    overwrite: bool = False,
) -> StructuredSemanticView:
    """Publish an immutable view whose validation eligibility is qualification-bound."""

    source_view = Path(source_view).resolve()
    source_root = Path(source_root).resolve()
    qualification_root = Path(qualification_root).resolve()
    output = Path(output).resolve()
    pending = StructuredSemanticView.open(source_view, source_root=source_root)
    if pending.manifest["scientific_validation_scenario_ids"]:
        raise StructuredSemanticError("promotion source must be an unqualified view")
    try:
        from .truth_editing_structured_qualification import (
            StructuredSemanticQualification,
        )

        qualification = StructuredSemanticQualification.open(
            qualification_root,
            source_view,
            source_root,
            allow_nonproduction=allow_nonproduction_qualification,
        )
    except Exception as error:
        raise StructuredSemanticError("base-known qualification is invalid") from error

    qualification_by_id = {
        item.scenario_id: item for item in qualification.scenarios
    }
    validation_ids = [
        str(item["scenario_id"])
        for item in pending.scenarios
        if item["split"] == "validation"
    ]
    if set(qualification_by_id) != set(validation_ids):
        raise StructuredSemanticError("qualification scenario inventory differs")
    scenarios: list[dict[str, Any]] = []
    for source_scenario in pending.scenarios:
        scenario = dict(source_scenario)
        if scenario["split"] == "validation":
            passed = qualification_by_id[str(scenario["scenario_id"])].all_required_known
            scenario["base_known_status"] = "known" if passed else "not_known"
            scenario["scientific_eligibility"] = (
                "eligible" if passed else "ineligible_base_not_known"
            )
        scenarios.append(scenario)

    rendered = "".join(_canonical(row) + "\n" for row in scenarios)
    scientific = [
        str(row["scenario_id"])
        for row in scenarios
        if row["split"] == "validation"
        and row["scientific_eligibility"] == "eligible"
    ]
    bare_manifest = {
        key: value
        for key, value in pending.manifest.items()
        if key not in {"format", "view_id", "view_sha256", "file_sha256"}
    }
    bare_manifest.update(
        {
            "format": QUALIFIED_MANIFEST_FORMAT,
            "view_id": "truth_editing_structured_semantic_qualified_v1",
            "source_view": {
                "path": os.path.relpath(source_view, output),
                "view_sha256": pending.manifest["view_sha256"],
            },
            "base_known_qualification": {
                "path": os.path.relpath(qualification_root, output),
                "manifest_sha256": qualification.manifest_sha256,
                "source_view_sha256": qualification.source_view_sha256,
            },
            "pending_base_known_validation_scenario_ids": [],
            "scientific_validation_scenario_ids": scientific,
            "file_sha256": {
                "scenarios.jsonl": hashlib.sha256(rendered.encode()).hexdigest()
            },
        }
    )
    manifest = {**bare_manifest, "view_sha256": _hash(bare_manifest)}
    if output.exists():
        existing = StructuredSemanticView._open(
            output,
            source_root=source_root,
            qualification_root=qualification_root,
            allow_nonproduction_qualification=allow_nonproduction_qualification,
        )
        if existing.manifest == manifest:
            return existing
        if not overwrite:
            raise StructuredSemanticError("refusing to replace differing qualified view")
        if output.is_symlink() or not output.is_dir():
            raise StructuredSemanticError("output must be a regular directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "scenarios.jsonl").write_text(rendered)
    (output / "manifest.json").write_text(_canonical(manifest) + "\n")
    return StructuredSemanticView._open(
        output,
        source_root=source_root,
        qualification_root=qualification_root,
        allow_nonproduction_qualification=allow_nonproduction_qualification,
    )
