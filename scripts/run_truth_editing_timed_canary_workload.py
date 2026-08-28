#!/usr/bin/env python3
"""Run one real v4 discovery proposal and emit the timed-canary observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from intelligent_liars.truth_editing_batch_execution import BatchEvaluationRequest  # noqa: E402
from intelligent_liars.truth_editing_directions import DirectionBank  # noqa: E402
from intelligent_liars.truth_editing_production import (  # noqa: E402
    ProductionRunConfig,
    open_production_run,
)
from intelligent_liars.truth_editing_production_judge_budget import (  # noqa: E402
    ProductionJudgeBudget,
)
from intelligent_liars.truth_editing_study import (  # noqa: E402
    CoverageLedger,
    OfflineDeterministicSearchDriver,
    SearchRequest,
    EvaluationResult,
    load_truth_editing_study_config,
)
from intelligent_liars.truth_editing_wandb_monitoring import CoordinatorMonitor  # noqa: E402


FORMAT = "truth_editing_timed_canary_observation_v2"
_OUTPUT_FIELDS = {
    "journal_path": "study/study-journal.json",
    "artifact_dir": "study/frozen",
    "runtime_output_dir": "study/runtime",
    "judge_cache_dir": "providers/judge-cache",
    "judge_budget_ledger_dir": "providers/production-judge-budget",
}
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace timed-canary output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _runtime_config(source_path: Path, output_root: Path) -> Path:
    source = json.loads(source_path.read_text())
    if not isinstance(source, dict) or source.get("format") != "truth_editing_production_config_v1":
        raise RuntimeError("production config format differs")
    runtime = dict(source)
    for field, relative in _OUTPUT_FIELDS.items():
        if field not in source:
            raise RuntimeError(f"production config is missing {field}")
        runtime[field] = str((output_root / relative).resolve())
    target = source_path.with_name(
        f".truth_editing_timed_canary.runtime.{hashlib.sha256(source_path.read_bytes()).hexdigest()[:8]}.json"
    )
    rendered = json.dumps(runtime, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    if target.exists():
        if target.is_symlink() or target.read_bytes() != rendered:
            raise RuntimeError("existing timed-canary runtime config differs")
        return target
    _write_new(target, runtime)
    return target


def _positive_float(value: str | None, label: str) -> float:
    try:
        result = float(value or "")
    except ValueError as error:
        raise RuntimeError(f"{label} must be a finite positive number") from error
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError(f"{label} must be a finite positive number")
    return result


def run_workload(
    *,
    source_config: Path,
    expected_source_sha256: str,
    output_root: Path,
    observation_path: Path,
    gpu_hourly_usd: float,
    wandb_project: str,
    wandb_entity: str | None,
    environ: Mapping[str, str] = os.environ,
    opener: Any = open_production_run,
    monitor_type: Any = CoordinatorMonitor,
    monotonic: Any = time.monotonic,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    if len(expected_source_sha256) != 64 or any(
        character not in _HEX for character in expected_source_sha256
    ):
        raise RuntimeError("production config SHA-256 is invalid")
    source_config = source_config.resolve()
    if source_config.is_symlink() or not source_config.is_file():
        raise RuntimeError("production config must be a regular file")
    if hashlib.sha256(source_config.read_bytes()).hexdigest() != expected_source_sha256:
        raise RuntimeError("production config SHA-256 differs")
    if observation_path.exists() or observation_path.is_symlink():
        raise RuntimeError("timed-canary observation already exists")
    for variable in ("OPENROUTER_API_KEY", "WANDB_API_KEY"):
        if not environ.get(variable):
            raise RuntimeError(f"{variable} is required")
    if not math.isfinite(gpu_hourly_usd) or gpu_hourly_usd <= 0:
        raise RuntimeError("exact GPU hourly price must be finite and positive")

    output_root = output_root.resolve()
    runtime_path = _runtime_config(source_config, output_root)
    production = ProductionRunConfig.open(runtime_path)
    if production.judge_budget is None or production.judge_budget_ledger_dir is None:
        raise RuntimeError("production judge circuit is absent")
    study = load_truth_editing_study_config(production.study_config)
    discovery = next((tier for tier in study.evaluation_tiers if tier.name == "discovery"), None)
    if discovery is None:
        raise RuntimeError("production discovery tier is absent")
    bank = DirectionBank.open(production.direction_manifest, root=production.direction_root)
    proposal = OfflineDeterministicSearchDriver(seed=study.sampler_seed).suggest(
        SearchRequest(0, study, bank.manifest.directions, CoverageLedger())
    )
    request = BatchEvaluationRequest(
        trial_id="trial-0000",
        ordinal=0,
        proposal=proposal,
        record_ids=tuple(study.validation_record_ids[: discovery.record_limit]),
        objective_names=study.objective_names,
    )
    monitoring_root = output_root / "monitoring"
    run_id = hashlib.sha256(
        f"timed-canary:{expected_source_sha256}".encode()
    ).hexdigest()[:32]
    monitor = monitor_type.open(
        checkpoint_path=monitoring_root / "wandb-run.json",
        run_id=run_id,
        project=wandb_project,
        entity=wandb_entity,
        run_name="truth-editing-timed-canary-v6-adaptive-r10",
        receipt_path=monitoring_root / "wandb-events.jsonl",
        total_trials=1,
        batch_size=1,
    )
    started = monotonic()
    try:
        evaluated = opener(runtime_path).evaluate_timed_canary(request)
        result = dict(evaluated.get("result", {}))
        telemetry = dict(evaluated.get("runtime_telemetry", {}))
        evidence = dict(evaluated.get("evaluator_evidence", {}))
        judge_budget = ProductionJudgeBudget(
            production.judge_budget_ledger_dir, config=production.judge_budget
        )
        judge = judge_budget.monitoring_snapshot()
        judge_receipt = judge_budget.receipt()
        judge_elapsed_ms = judge.get("elapsed_ms")
        if (
            isinstance(judge_elapsed_ms, bool)
            or not isinstance(judge_elapsed_ms, (int, float))
            or not math.isfinite(float(judge_elapsed_ms))
            or float(judge_elapsed_ms) < 0
        ):
            raise RuntimeError("total judge elapsed evidence is missing or invalid")
        monitor.record_batch(
            0,
            (request,),
            (EvaluationResult(
                result["outcome_kind"], result.get("metrics", {}), result.get("detail")
            ),),
        )
        monitor.record_worker_telemetry(0, request.trial_id, telemetry)
        monitor.record_judge(
            calls=int(judge["calls"]),
            failures=int(judge["failures"]),
            latency_ms=float(judge["latency_ms"]),
            cost_usd=float(judge["cost_usd"]),
        )
        elapsed = float(monotonic()) - float(started)
        monitor.record_cost(
            gpu_actual_usd=gpu_hourly_usd * elapsed / 3600.0,
            gpu_projected_usd=gpu_hourly_usd * elapsed / 3600.0,
            judge_actual_usd=float(judge["cost_usd"]),
            judge_projected_usd=float(judge["cost_usd"]),
        )
    finally:
        monitor.close()
    snapshot = dict(monitor.verification_snapshot())
    if (
        snapshot.get("initialized_coordinator_count") != 1
        or snapshot.get("finish_calls") != 1
        or snapshot.get("nonfatal_error_count") != 0
    ):
        raise RuntimeError("W&B coordinator canary did not initialize, log, and close cleanly")
    outcome_kind = result.get("outcome_kind")
    if outcome_kind == "operational_failure":
        detail = result.get("detail")
        if not isinstance(detail, str) or not detail.strip():
            detail = "no backend detail was recorded"
        raise RuntimeError(f"timed-canary trial failed operationally: {detail}")
    if outcome_kind not in {"successful", "scientifically_infeasible"}:
        raise RuntimeError(
            f"timed-canary trial returned unsupported outcome: {outcome_kind!r}"
        )
    generated_tokens = telemetry.get("generated_tokens")
    evaluation_seconds = telemetry.get("evaluation_seconds")
    preservation = evidence.get("preservation_kl")
    if (
        isinstance(generated_tokens, bool)
        or not isinstance(generated_tokens, (int, float))
        or int(generated_tokens) <= 0
        or isinstance(evaluation_seconds, bool)
        or not isinstance(evaluation_seconds, (int, float))
        or float(evaluation_seconds) <= 0
        or not isinstance(preservation, Mapping)
        or set(preservation) != {"text", "vision", "recorded_computer_use"}
        or evidence.get("tier") != "discovery"
        or evidence.get("judge_cache_receipt_count") != judge["calls"]
    ):
        raise RuntimeError("real model, preservation, or judge canary evidence is incomplete")
    result_unsigned = {
        "trial_id": request.trial_id,
        "proposal": proposal.to_dict(),
        "result": result,
    }
    observation = {
        "format": FORMAT,
        "canary_id": "truth-editing-production-v6-adaptive-r10-canary",
        "production_config_path": str(source_config.relative_to(repository_root.resolve())),
        "production_config_sha256": expected_source_sha256,
        "model_sha256": production.verified_model_sha256,
        "actual_model_loaded": True,
        "batch_count": 1,
        "generated_tokens": int(generated_tokens),
        "generation_seconds": float(evaluation_seconds),
        "judge": {
            "deployment_alias": "glm-5.3-flash",
            "model": "z-ai/glm-5.3-flash",
            "provider_route": "z-ai/fp8",
            "fallback_used": False,
            "response_healing_used": True,
            "attempted_calls": int(judge["calls"]),
            "successful_calls": int(judge["calls"] - judge["failures"]),
            "failed_calls": int(judge["failures"]),
            "cost_usd": float(judge["cost_usd"]),
            "elapsed_seconds": float(judge_elapsed_ms) / 1000.0,
            "circuit_opened": bool(judge_receipt["circuit_open"]),
        },
        "persistence_kl": {name: float(value) for name, value in preservation.items()},
        "trials": [
            {
                "trial_id": request.trial_id,
                "outcome_kind": result["outcome_kind"],
                "metrics": result["metrics"],
                "output_sha256": hashlib.sha256(_canonical(result_unsigned)).hexdigest(),
            }
        ],
    }
    _write_new(observation_path, observation)
    return observation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wandb-project", default="intelligent-liars")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args(argv)
    try:
        observation_value = os.environ.get("TRUTH_EDITING_TIMED_CANARY_OBSERVATION_PATH")
        if not observation_value:
            raise RuntimeError("TRUTH_EDITING_TIMED_CANARY_OBSERVATION_PATH is required")
        result = run_workload(
            source_config=args.config,
            expected_source_sha256=args.config_sha256,
            output_root=args.output_root,
            observation_path=Path(observation_value),
            gpu_hourly_usd=_positive_float(
                os.environ.get("TRUTH_EDITING_GPU_HOURLY_USD"),
                "TRUTH_EDITING_GPU_HOURLY_USD",
            ),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"truth-editing timed-canary workload failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
