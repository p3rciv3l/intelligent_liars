#!/usr/bin/env python3
"""Run a safe W&B transport/dashboard smoke gate.

This is not the timed real-GPU canary and produces no behavioral, capability,
scientific, model, judge-quality, or throughput evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.heretic_truth_editing import OBJECTIVES  # noqa: E402
from intelligent_liars.truth_editing_batch_execution import (  # noqa: E402
    BatchEvaluationRequest,
)
from intelligent_liars.truth_editing_gpu_telemetry import (  # noqa: E402
    GpuTelemetryRecord,
)
from intelligent_liars.truth_editing_study import EvaluationResult  # noqa: E402
from intelligent_liars.truth_editing_wandb_canary import (  # noqa: E402
    WandbCanaryError,
    build_wandb_canary_trace,
    build_wandb_failure_probe,
    read_wandb_dashboard,
    verify_wandb_canary,
)
from intelligent_liars.truth_editing_wandb_checkpoint import (  # noqa: E402
    open_wandb_run_checkpoint,
)
from intelligent_liars.truth_editing_wandb_monitoring import (  # noqa: E402
    CoordinatorMonitor,
)


SMOKE_KIND = "wandb_transport_smoke"
EVIDENCE_BOUNDARY = "transport_only_not_timed_gpu_or_scientific_canary"


class WandbSmokeError(RuntimeError):
    """A controlled W&B transport-smoke failure."""


class _FailingRun:
    def log(self, _values: Any, *, step: int | None = None) -> None:
        raise RuntimeError("injected dashboard failure")

    def finish(self, *, exit_code: int = 0) -> None:
        return None


class _FailingWandb:
    @staticmethod
    def Settings(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    @staticmethod
    def init(**_kwargs: Any) -> _FailingRun:
        return _FailingRun()


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, indent=2
    ).encode() + b"\n"


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _proposal() -> dict[str, Any]:
    return {
        "attention_edge_strength": 0.5,
        "attention_enabled": True,
        "attention_kernel_center": 21.0,
        "attention_kernel_half_width": 3.0,
        "attention_peak_strength": 1.0,
        "backend_type": "persistent_weight",
        "basis_method": "qr",
        "basis_scope": "general",
        "direction_family": "general",
        "direction_ids": ["synthetic-smoke-direction"],
        "edit_arm": "truth_only",
        "mlp_edge_strength": 0.5,
        "mlp_enabled": True,
        "mlp_kernel_center": 21.0,
        "mlp_kernel_half_width": 3.0,
        "mlp_peak_strength": 1.0,
        "normalization_mode": "exact",
        "proposal_origin": "coverage_anchor",
        "refusal_direction_scope": "global",
        "refusal_enabled": False,
        "refusal_source_layer": None,
        "refusal_strength": 0.0,
        "refusal_writer_policy": "both",
        "requested_rank": 1,
        "selected_domains": [],
        "source_layer": 21,
        "strength": 1.0,
        "truth_direction_scope": "global",
        "writer_layers": [20, 21, 22],
        "writer_policy": "both",
        "writer_region": "middle",
    }


def _request() -> BatchEvaluationRequest[dict[str, Any]]:
    return BatchEvaluationRequest(
        trial_id="trial-0000",
        ordinal=0,
        proposal=_proposal(),
        record_ids=(),
        objective_names=OBJECTIVES,
    )


def _result() -> EvaluationResult:
    return EvaluationResult.successful(
        {
            "valid_false_report_rate_lcb": 0.61,
            "truth_report_dissociation_lcb": 0.72,
            "capability_preservation_lcb": 0.83,
        }
    )


def _record_safe_smoke_metrics(monitor: CoordinatorMonitor) -> None:
    monitor.record_batch(0, (_request(),), (_result(),))
    monitor.record_gpu(
        GpuTelemetryRecord(
            gpu_slot=0,
            utilization_percent=50.0,
            memory_used_mib=12_000.0,
            memory_total_mib=24_576.0,
            tokens_per_second=40.0,
            active_trial_id="trial-0000",
            observed_at="2026-08-28T00:00:00Z",
        )
    )
    monitor.record_judge(calls=1, failures=0, latency_ms=125.0, cost_usd=0.001)
    monitor.record_cost(
        gpu_actual_usd=0.001,
        gpu_projected_usd=1.0,
        judge_actual_usd=0.001,
        judge_projected_usd=1.0,
    )
    monitor.record_operational(
        retries=1,
        stopped_trials=0,
        errors=1,
        error_category="worker_operational_failure",
        error_fingerprint="e" * 64,
    )


def _failure_probe(stage: Path, *, run_id: str) -> dict[str, Any]:
    output = stage / "synthetic-authoritative-output.json"
    checkpoint = stage / "synthetic-authoritative-checkpoint.json"
    _write_new(
        output,
        {
            "format": "truth_editing_wandb_smoke_authoritative_output_v1",
            "trial_ordinal": 0,
            "objective_values": dict(_result().metrics),
        },
    )
    _write_new(
        checkpoint,
        {
            "format": "truth_editing_wandb_smoke_authoritative_checkpoint_v1",
            "completed_trials": 1,
        },
    )
    output_before = output.read_bytes()
    checkpoint_before = checkpoint.read_bytes()
    receipt_path = stage / "monitoring/failure-probe.jsonl"
    monitor = CoordinatorMonitor(
        run_id=f"{run_id}-failure-probe",
        project="local-failure-probe",
        entity=None,
        run_name="local-failure-probe",
        receipt_path=receipt_path,
        total_trials=1,
        batch_size=1,
        wandb_module=_FailingWandb(),
        monotonic=lambda: 0.0,
    )
    _record_safe_smoke_metrics(monitor)
    monitor.close()
    snapshot = monitor.verification_snapshot()
    count = snapshot["nonfatal_error_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise WandbSmokeError("injected W&B failure was not observed")
    probe: dict[str, Any] = build_wandb_failure_probe(
        control_optimizer_output=output_before,
        injected_optimizer_output=output.read_bytes(),
        control_checkpoint=checkpoint_before,
        injected_checkpoint=checkpoint.read_bytes(),
        local_error_receipt=receipt_path.read_bytes(),
        injected_failure_count=count,
    )
    return probe


def _manifest(stage: Path, *, run_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            payload = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    unsigned = {
        "format": "truth_editing_wandb_transport_smoke_manifest_v1",
        "kind": SMOKE_KIND,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "wandb_run_id": run_id,
        "gate_receipt_sha256": receipt["receipt_sha256"],
        "files": files,
    }
    return {**unsigned, "manifest_sha256": hashlib.sha256(_encoded(unsigned)).hexdigest()}


def _run(
    *,
    output_dir: Path,
    project: str,
    entity: str | None,
    run_id: str,
    wandb_module: Any,
    readback_attempts: int,
    readback_delay_seconds: float,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise WandbSmokeError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    stage.mkdir(mode=0o700)
    try:
        api = wandb_module.Api()
        resolved_entity = entity or getattr(api, "default_entity", None)
        if not isinstance(resolved_entity, str) or not resolved_entity:
            raise WandbSmokeError("W&B entity is unavailable; pass --entity")
        checkpoint_path = stage / "monitoring/wandb-run.json"
        snapshots: list[dict[str, Any]] = []
        for session in (1, 2):
            monitor = CoordinatorMonitor.open(
                checkpoint_path=checkpoint_path,
                run_id=run_id,
                project=project,
                entity=resolved_entity,
                run_name="truth-editing-wandb-transport-smoke",
                receipt_path=stage / f"monitoring/session-{session}.jsonl",
                total_trials=200,
                batch_size=8,
                wandb_module=wandb_module,
                monotonic=time.monotonic,
            )
            if session == 1:
                _record_safe_smoke_metrics(monitor)
            monitor.close()
            snapshots.append(dict(monitor.verification_snapshot()))

        checkpoint = open_wandb_run_checkpoint(checkpoint_path).to_mapping()
        failure_probe = _failure_probe(stage, run_id=run_id)
        _write_new(stage / "failure-probe.json", failure_probe)

        receipt_path = stage / "gate-receipt.json"
        trace: dict[str, Any] | None = None
        receipt: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(readback_attempts):
            try:
                readback = read_wandb_dashboard(api, checkpoint=checkpoint)
                trace = build_wandb_canary_trace(
                    coordinator_snapshots=snapshots,
                    dashboard_readback=readback,
                    failure_probe=failure_probe,
                )
                receipt = verify_wandb_canary(
                    trace=trace,
                    checkpoint=checkpoint,
                    receipt_path=receipt_path,
                )
                break
            except WandbCanaryError as error:
                last_error = error
                if attempt + 1 < readback_attempts:
                    time.sleep(readback_delay_seconds)
        if trace is None or receipt is None:
            raise WandbSmokeError("W&B server readback did not satisfy the smoke gate") from last_error
        _write_new(stage / "transport-trace.json", trace)
        manifest = _manifest(stage, run_id=run_id, receipt=receipt)
        _write_new(stage / "manifest.json", manifest)
        os.rename(stage, output_dir)
        return {
            "kind": SMOKE_KIND,
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "status": "passed",
            "wandb_run_id": run_id,
            "manifest_sha256": manifest["manifest_sha256"],
            "output_dir": str(output_dir),
        }
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None, *, wandb_module: Any | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=REPOSITORY_ROOT / ".env")
    parser.add_argument("--project", default="intelligent-liars")
    parser.add_argument("--entity")
    parser.add_argument("--run-id", default=f"truth-editing-wandb-smoke-{uuid4().hex[:12]}")
    parser.add_argument("--readback-attempts", type=int, default=6)
    parser.add_argument("--readback-delay-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise WandbSmokeError("output directory already exists")
        load_dotenv(args.env_file, override=False)
        if "WANDB_API_KEY" not in os.environ or not os.environ["WANDB_API_KEY"]:
            raise WandbSmokeError("WANDB_API_KEY is unavailable")
        if args.readback_attempts < 1 or not 0 <= args.readback_delay_seconds <= 30:
            raise WandbSmokeError("readback retry settings are invalid")
        with tempfile.TemporaryDirectory(prefix="truth-editing-wandb-smoke-") as local_dir:
            safe_environment = {
                "WANDB_DIR": local_dir,
                "WANDB_SILENT": "true",
                "WANDB_QUIET": "true",
                "WANDB_CONSOLE": "off",
                "WANDB_DISABLE_CODE": "true",
            }
            previous = {name: os.environ.get(name) for name in safe_environment}
            os.environ.update(safe_environment)
            try:
                module = wandb_module or importlib.import_module("wandb")
                result = _run(
                    output_dir=args.output_dir,
                    project=args.project,
                    entity=args.entity,
                    run_id=args.run_id,
                    wandb_module=module,
                    readback_attempts=args.readback_attempts,
                    readback_delay_seconds=args.readback_delay_seconds,
                )
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        print(json.dumps(result, sort_keys=True))
        return 0
    except WandbSmokeError as error:
        print(f"W&B transport smoke failed: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        category = f"{type(error).__module__}.{type(error).__qualname__}"
        fingerprint = hashlib.sha256(category.encode()).hexdigest()
        print(
            f"W&B transport smoke failed safely: category={category} fingerprint={fingerprint}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
