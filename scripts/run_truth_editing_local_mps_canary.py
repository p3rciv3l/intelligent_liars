#!/usr/bin/env python3
"""Run an explicitly non-production eight-trial Optuna recovery canary on Apple MPS.

This exercises the real Qwen checkpoint, persistent writer edits, OpenRouter
judge, Optuna journal, and immutable rescore driver.  MPS/eager execution is a
portability and recovery check only; it is not equivalent to the frozen CUDA
production runtime and cannot establish scientific readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.models import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ModelBundle,
    ModelLoadConfig,
)
from intelligent_liars.model_cache import REQUIRED_SNAPSHOT_FILES  # noqa: E402
from intelligent_liars.truth_editing_qwen_runtime import TrialRuntime  # noqa: E402
from intelligent_liars.truth_editing_production import open_production_run  # noqa: E402
from intelligent_liars.truth_editing_rescore import (  # noqa: E402
    materialize_rescore_generation_v1,
)
from intelligent_liars.truth_editing_study import EvaluationResult  # noqa: E402


MODEL_SHA256 = "bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8"
MODEL_MANIFEST_SHA256 = "6ec207494fed0658ebaf31f1083f77110caae45f8c2392354e767ef54c78dd07"
CANARY_TRIALS = 8
SOURCE_TRIALS = 4
def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _verify_snapshot(snapshot: Path, manifest_path: Path) -> tuple[dict[str, Any], str]:
    manifest = _read(manifest_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (
        manifest.get("format") != "tinylora_model_cache_manifest_v1"
        or manifest.get("complete") is not True
        or manifest.get("content_sha256") != MODEL_SHA256
        or manifest_sha != MODEL_MANIFEST_SHA256
        or manifest.get("model")
        != {"repo_id": DEFAULT_MODEL_ID, "revision": DEFAULT_MODEL_REVISION}
    ):
        raise RuntimeError("local model manifest identity differs")
    files = manifest.get("files")
    if (
        not isinstance(files, list)
        or [item.get("path") for item in files if isinstance(item, Mapping)]
        != list(REQUIRED_SNAPSHOT_FILES)
    ):
        raise RuntimeError("local model manifest file inventory differs")
    observed_total = 0
    for item in files:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("bytes"), int)
            or isinstance(item.get("bytes"), bool)
        ):
            raise RuntimeError("local model manifest file entry is invalid")
        path = snapshot / str(item.get("path"))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256")
        ):
            raise RuntimeError(f"local model snapshot differs at {path.name}")
        observed_total += item["bytes"]
    if observed_total != manifest.get("total_bytes"):
        raise RuntimeError("local model snapshot total byte count differs")
    return manifest, manifest_sha


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir.resolve()
    try:
        root.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise RuntimeError("canary output directory must be inside the repository") from error
    root.mkdir(parents=True, exist_ok=False)
    base_config_path = args.base_config.resolve()
    base = _read(base_config_path)
    base_study_path = (base_config_path.parent / str(base["study_config"])).resolve()
    base_evaluator_path = (base_config_path.parent / str(base["evaluator_config"])).resolve()
    study = _read(base_study_path)
    evaluator = _read(base_evaluator_path)
    _, manifest_sha = _verify_snapshot(args.model_snapshot, args.model_manifest)

    study["format"] = "truth_editing_study_config_v1"
    study.pop("search_policy", None)
    study["study_id"] = "truth-editing-local-mps-recovery-canary-v1"
    study["batch_size"] = 2
    study["max_trials"] = CANARY_TRIALS
    study["tpe_startup_trials"] = 2
    qa_records = [
        item for item in study["validation_record_ids"] if str(item).startswith("qa_")
    ][:9]
    structured_records = [
        item
        for item in study["validation_record_ids"]
        if not str(item).startswith("qa_")
    ][:6]
    validation_records = [*structured_records, *qa_records]
    if len(structured_records) != 6 or len(qa_records) != 9:
        raise RuntimeError("base study lacks complete structured and QA canary scenarios")
    study["evaluation_tiers"] = [
        {"name": "discovery", "record_limit": 9, "through_trial": 2},
        {"name": "expanded", "record_limit": 12, "through_trial": 4},
        {"name": "finalist", "record_limit": 15, "through_trial": CANARY_TRIALS},
    ]
    study["validation_record_ids"] = validation_records

    evaluator["config_id"] = "truth-editing-local-mps-recovery-canary-v1"
    limits = {"discovery": 9, "expanded": 12, "finalist": 15}
    for tier in evaluator["tiers"]:
        tier["record_ids"] = study["validation_record_ids"][: limits[tier["name"]]]

    config_stem = f".local_mps_canary_{root.name}"
    study_path = REPOSITORY_ROOT / "configs" / f"{config_stem}_study.json"
    evaluator_path = REPOSITORY_ROOT / "configs" / f"{config_stem}_evaluator.json"
    _write_new(study_path, study)
    _write_new(evaluator_path, evaluator)

    common = dict(base)
    common.update(
        {
            "study_config": study_path.name,
            "evaluator_config": evaluator_path.name,
            "verified_model_sha256": MODEL_SHA256,
            "verified_snapshot_manifest_sha256": manifest_sha,
            "max_new_tokens": 4,
            "judge_budget": {
                "format": "truth_editing_production_judge_budget_config_v1",
                "judge_config_sha256": base["judge_budget"]["judge_config_sha256"],
                "maximum_judge_spend_usd": "1",
                "per_call_reservation_usd": "0.025",
                "non_judge_reserved_spend_usd": "0",
                "all_in_maximum_spend_usd": "1",
            },
            "judge_cache_dir": str(root / "shared" / "judge-cache"),
            "judge_budget_ledger_dir": str(root / "shared" / "judge-budget"),
        }
    )
    model_reference = root / "model-cache-reference"
    model_reference.mkdir()
    reference_manifest = root / "model-manifest-reference.json"
    _write_new(reference_manifest, _read(args.model_manifest))
    common["model_cache_dir"] = os.path.relpath(model_reference, REPOSITORY_ROOT / "configs")
    common["snapshot_manifest_path"] = os.path.relpath(
        reference_manifest, REPOSITORY_ROOT / "configs"
    )
    source = dict(common)
    source.update(
        {
            "artifact_dir": str(root / "source" / "artifacts"),
            "journal_path": str(root / "source" / "study-journal.json"),
            "runtime_output_dir": str(root / "source" / "runtime"),
        }
    )
    target = dict(common)
    target.update(
        {
            "artifact_dir": str(root / "target" / "artifacts"),
            "journal_path": str(root / "target" / "study-journal.json"),
            "runtime_output_dir": str(root / "target" / "runtime"),
        }
    )
    source_config_path = REPOSITORY_ROOT / "configs" / f"{config_stem}_source.json"
    target_base_path = REPOSITORY_ROOT / "configs" / f"{config_stem}_target_base.json"
    target_config_path = REPOSITORY_ROOT / "configs" / f"{config_stem}_target.json"
    _write_new(source_config_path, source)
    _write_new(target_base_path, target)
    pointers = {
        "source_config": str(source_config_path),
        "target_base_config": str(target_base_path),
        "target_config": str(target_config_path),
        "study_config": str(study_path),
        "evaluator_config": str(evaluator_path),
    }
    _write_new(root / "config-paths.json", pointers)
    receipt = {
        "format": "truth_editing_local_mps_canary_preparation_v1",
        "runtime_scope": "nonproduction_apple_mps_eager_portability_only",
        "trial_count": CANARY_TRIALS,
        "source_stop_boundaries": [2, 4],
        "target_stop_boundaries": [6, 8],
        "model_sha256": MODEL_SHA256,
        "snapshot_manifest_sha256": manifest_sha,
        "config_paths_sha256": _sha(pointers),
    }
    receipt["content_sha256"] = _sha(receipt)
    _write_new(root / "preparation.json", receipt)
    return receipt


def _mps_loader(snapshot: Path, manifest_path: Path, manifest_sha: str):
    def load(config):
        del config
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        _verify_snapshot(snapshot, manifest_path)
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = True
        model.to("mps")
        model.eval()
        return ModelBundle(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            model_id=DEFAULT_MODEL_ID,
            config=ModelLoadConfig(),
            verified_snapshot={
                "model_id": DEFAULT_MODEL_ID,
                "revision": DEFAULT_MODEL_REVISION,
                "model_sha256": MODEL_SHA256,
                "snapshot_manifest_sha256": manifest_sha,
            },
        )

    return load


def _open_mps_run(config_path: Path, model_snapshot: Path, model_manifest: Path):
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable")
    _, manifest_sha = _verify_snapshot(model_snapshot, model_manifest)
    run = open_production_run(config_path)
    old_runtime = run._evaluator._runtime
    run._evaluator._runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA256,
        verified_snapshot_manifest_sha256=manifest_sha,
        output_dir=old_runtime._output_dir,
        preservation_collector=old_runtime._preservation_collector,
        bundle_loader=_mps_loader(model_snapshot, model_manifest, manifest_sha),
        enforce_production_identity=False,
        inference_microbatch_size=1,
        runtime_dtype_name="torch.bfloat16",
        runtime_device_map="mps",
        runtime_attention_implementation="eager",
    )
    return run


class _OneTrialOperationalBoundary:
    def __init__(self, downstream: Any) -> None:
        self.downstream = downstream

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter": "local_mps_canary_one_trial_operational_boundary_v1",
            "forced_after_real_evaluation_trial_id": "trial-0001",
            "downstream": dict(self.downstream.identity),
        }

    def evaluate(
        self, proposal, *, trial_id, record_ids, objective_names, **kwargs: Any
    ):
        result = self.downstream.evaluate(
            proposal,
            trial_id=trial_id,
            record_ids=record_ids,
            objective_names=objective_names,
            **kwargs,
        )
        if trial_id == "trial-0001":
            return EvaluationResult.operational_failure(
                "canary-injected boundary after complete real evaluation"
            )
        return result


class _MpsCanaryPreservationCollector:
    """Fast, explicit software-only preservation receipt for appended trials."""

    identity = {
        "adapter": "local_mps_canary_synthetic_zero_kl_preservation_v1",
        "software_readiness_only": True,
        "scientific_evidence": False,
    }

    def collect(self, bundle: Any, batch: Any) -> Mapping[str, Any]:
        del bundle
        record_count = len(batch.examples)
        tier_by_count = {9: "trial", 12: "promoted", 15: "finalist"}
        try:
            tier = tier_by_count[record_count]
        except KeyError as error:
            raise RuntimeError("MPS canary batch does not identify a canary tier") from error
        edit_binding = _sha(
            {
                "format": "truth_editing_preservation_edit_binding_v1",
                "batch_sha256": batch.batch_sha256,
                "recipe_id": batch.recipe_id,
                "basis_set_sha256": batch.basis_set.basis_set_sha256,
            }
        )
        strata = [
            {
                "stratum": name,
                "record_count": 1,
                "assistant_token_count": 1,
                "forward_kl": 0.0,
            }
            for name in ("text", "vision", "recorded_computer_use")
        ]
        preservation = {
            "format": "truth_editing_preservation_receipt_v1",
            "spec_sha256": "0" * 64,
            "edited_model_sha256": edit_binding,
            "tier": tier,
            "strata": strata,
            "aggregate_kl": 0.0,
            "vision_tower_byte_identical": True,
        }
        preservation["self_sha256"] = _sha(preservation)
        receipt = {
            "format": "truth_editing_preservation_runtime_receipt_v1",
            "batch_sha256": batch.batch_sha256,
            "recipe_id": batch.recipe_id,
            "model_sha256": batch.model_sha256,
            "basis_set_sha256": batch.basis_set.basis_set_sha256,
            "tier": tier,
            "collector_identity_sha256": _sha(self.identity),
            "preservation_receipt": preservation,
        }
        receipt["self_sha256"] = _sha(receipt)
        return receipt


def _run_study(args: argparse.Namespace, *, source: bool) -> dict[str, Any]:
    root = args.output_dir.resolve()
    paths = _read(root / "config-paths.json")
    config_path = Path(paths["source_config"] if source else paths["target_config"])
    run = _open_mps_run(config_path, args.model_snapshot, args.model_manifest)
    if not source:
        run._evaluator._runtime._preservation_collector = (
            _MpsCanaryPreservationCollector()
        )
    evaluator = _OneTrialOperationalBoundary(run._evaluator) if source else run._evaluator
    report = run._study.run(
        driver=run._driver,
        evaluator=evaluator,
        journal_path=run._journal_path,
        stop_after_trials=args.stop_after,
    )
    report_payload = report.to_dict()
    report_name = (
        f"source-report-{args.stop_after}.json"
        if source
        else f"target-report-{args.stop_after}.json"
    )
    _write_new(root / report_name, report_payload)
    if source and args.stop_after == SOURCE_TRIALS:
        checkpoint = {
            "format": "truth_editing_adaptive_run_checkpoint_v1",
            "study_identity_sha256": report.study_identity_sha256,
            "authorized_through_trial": SOURCE_TRIALS,
            "completed_trials": SOURCE_TRIALS,
            "coverage_complete": report.coverage_complete,
            "phase": "finalization_reserved",
            "stop_reason": "evaluation_budget_reserve_reached",
        }
        checkpoint["checkpoint_sha256"] = _sha(checkpoint)
        checkpoint_path = root / "source-checkpoint.json"
        _write_new(checkpoint_path, checkpoint)
        generation = materialize_rescore_generation_v1(
            source_report_path=root / report_name,
            source_journal_path=root / "source" / "study-journal.json",
            source_checkpoint_path=checkpoint_path,
            output_path=root / "rescore-generation.json",
            expected_source_study_identity_sha256=report.study_identity_sha256,
            expected_judge_config_sha256=run._evaluator._evaluator_config.judge_config_sha256,
            expected_rubric_sha256=run._evaluator._evaluator_config.rubric_sha256,
            expected_completed_trials=SOURCE_TRIALS,
        )
        target = _read(Path(paths["target_base_config"]))
        target.update(
            {
                "rescore_generation": os.path.relpath(
                    root / "rescore-generation.json", REPOSITORY_ROOT / "configs"
                ),
                "rescore_generation_sha256": generation.generation_sha256,
                "rescore_mode": "repair_then_continue",
            }
        )
        _write_new(Path(paths["target_config"]), target)
    return report_payload


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir.resolve()
    source = _read(root / f"source-report-{SOURCE_TRIALS}.json")
    target = _read(root / f"target-report-{CANARY_TRIALS}.json")
    source_counts = Counter(item["result"]["outcome_kind"] for item in source["trials"])
    target_counts = Counter(item["result"]["outcome_kind"] for item in target["trials"])
    allowed = {"successful", "scientifically_infeasible", "operational_failure"}
    if (
        source["completed_trials"] != SOURCE_TRIALS
        or target["completed_trials"] != CANARY_TRIALS
        or set(source_counts) - allowed
        or set(target_counts) - allowed
        or source_counts["operational_failure"] != 1
        or target["trials"][:SOURCE_TRIALS] != source["trials"]
    ):
        raise RuntimeError("canary trial accounting is incomplete or lineage changed")
    generation = _read(root / "rescore-generation.json")
    if len(generation["replay_requests"]) != 1:
        raise RuntimeError("canary rescore generation must contain exactly one replay")
    receipt = {
        "format": "truth_editing_local_mps_canary_audit_v1",
        "runtime_scope": "nonproduction_apple_mps_eager_portability_only",
        "source_completed_trials": SOURCE_TRIALS,
        "source_outcomes": dict(sorted(source_counts.items())),
        "target_completed_trials": CANARY_TRIALS,
        "target_outcomes": dict(sorted(target_counts.items())),
        "source_history_preserved": True,
        "rescore_request_count": 1,
        "ordinary_continuation_trial_count": CANARY_TRIALS - SOURCE_TRIALS - 1,
        "full_measured_preservation_source_trial_count": SOURCE_TRIALS,
        "synthetic_preservation_appended_trial_count": CANARY_TRIALS - SOURCE_TRIALS,
        "all_trials_terminally_accounted": True,
        "large_run_scientific_equivalence": False,
    }
    receipt["content_sha256"] = _sha(receipt)
    _write_new(root / "audit.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("prepare", "source", "target", "audit"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--base-config",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs/truth_editing_production_v6_adaptive_r10_17b1e9cb_c1f373f8.json",
    )
    result.add_argument("--model-snapshot", type=Path, required=True)
    result.add_argument("--model-manifest", type=Path, required=True)
    result.add_argument("--stop-after", type=int)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "prepare":
        payload = prepare(args)
    elif args.mode == "source":
        if args.stop_after not in {2, 4}:
            raise SystemExit("source --stop-after must be 2 or 4")
        payload = _run_study(args, source=True)
    elif args.mode == "target":
        if args.stop_after not in {6, 8}:
            raise SystemExit("target --stop-after must be 6 or 8")
        payload = _run_study(args, source=False)
    else:
        payload = audit(args)
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
