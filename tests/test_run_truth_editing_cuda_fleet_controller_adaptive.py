from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_truth_editing_adaptive_run import NOW, _policy, _receipt, _spend
from intelligent_liars.truth_editing_contracts import canonical_sha256


SCRIPT = Path(__file__).parents[1] / "scripts/run_truth_editing_cuda_fleet_controller.py"
SPEC = importlib.util.spec_from_file_location(
    "run_truth_editing_cuda_fleet_controller", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_controller_imports_repository_package_without_install_or_pythonpath(
    tmp_path: Path,
) -> None:
    """A fresh clone can launch the controller directly from its script path."""

    repository_root = SCRIPT.parents[1]
    source_root = repository_root / "src"
    probe = f"""
import importlib.abc
import runpy
import sys

source_root = {str(source_root)!r}

class RejectInstalledProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "intelligent_liars" and source_root not in sys.path:
            raise ModuleNotFoundError("installed project package intentionally unavailable")
        return None

sys.meta_path.insert(0, RejectInstalledProject())
runpy.run_path({str(SCRIPT)!r}, run_name="fresh_clone_controller")
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_controller_cli_exposes_one_adaptive_run_not_legacy_phase_stops() -> None:
    """The executable seam must launch the adaptive study, not a fixed 200 phase."""

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--capacity-receipt" in result.stdout
    assert "--phase" not in result.stdout
    assert "adaptive" in result.stdout.lower()
    assert "--output-root" in result.stdout
    assert "--restore-offhost-wandb-run-id" in result.stdout
    assert "--final-model-slug" in result.stdout
    assert "--host-hourly-usd" in result.stdout
    assert "--host-lease-started-at-utc" in result.stdout
    assert "--gpu-hourly-usd" not in result.stdout


def test_controller_price_environment_accepts_only_explicit_host_total_unit() -> None:
    assert MODULE.host_hourly_usd_from_environment(
        {"TRUTH_EDITING_HOST_HOURLY_USD": "2.4"}
    ) == pytest.approx(2.4)

    with pytest.raises(ValueError, match="obsolete.*GPU_HOURLY"):
        MODULE.host_hourly_usd_from_environment(
            {"TRUTH_EDITING_GPU_HOURLY_USD": "0.3"}
        )
    with pytest.raises(ValueError, match="obsolete.*GPU_HOURLY"):
        MODULE.host_hourly_usd_from_environment(
            {
                "TRUTH_EDITING_HOST_HOURLY_USD": "2.4",
                "TRUTH_EDITING_GPU_HOURLY_USD": "0.3",
            }
        )


def test_real_main_publishes_committed_batch_before_next_batch_admission() -> None:
    """A completed paid batch must be off-host before more work is authorized."""

    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    run_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and {keyword.arg for keyword in node.keywords}
        >= {
            "batch_admission",
            "after_complete_batch",
            "after_prepare_before_first_admission",
        }
    )
    keyword_values = {
        keyword.arg: keyword.value
        for keyword in run_call.keywords
        if keyword.arg is not None
    }
    assert isinstance(keyword_values["batch_admission"], ast.Name)
    assert keyword_values["batch_admission"].id == "durable_batch_admission"
    assert isinstance(keyword_values["after_complete_batch"], ast.Name)
    assert keyword_values["after_complete_batch"].id == "after_complete_batch"
    initial_hook = keyword_values["after_prepare_before_first_admission"]
    assert isinstance(initial_hook, ast.Name)
    assert initial_hook.id == "after_prepare_before_first_admission"

    callback = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "after_complete_batch"
    )
    calls = [
        node.func.attr
        for statement in callback.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls.index("reforecast") < calls.index("commit_batch")
    assert "admit_batch" not in calls
    assert "abort_minimum_trial_guarantee" in calls
    publish_call = next(
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "record_and_publish"
    )
    commit_call = next(
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "commit_batch"
    )
    assert commit_call.lineno < publish_call.lineno

    # The study asks this adapter for the next batch only after the callback
    # above returns, so authorization cannot precede verified publication.
    assert isinstance(keyword_values["batch_admission"], ast.Name)
    assert keyword_values["batch_admission"].id == "durable_batch_admission"

    prepared = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "after_prepare_before_first_admission"
    )
    prepared_calls = [
        node.func.attr
        for statement in prepared.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert prepared_calls.index("admit_batch") < prepared_calls.index(
        "record_adaptive_progress"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_boundary"
        for node in ast.walk(prepared)
    )


def test_real_main_discovers_latest_offhost_resume_without_manual_tuple() -> None:
    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = {
        node.func.attr
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "read_latest_binding_if_present" in calls


def test_real_main_wires_per_trial_offhost_durability() -> None:
    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    evaluator_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FleetBatchEvaluator"
    )
    callback = next(
        keyword.value
        for keyword in evaluator_call.keywords
        if keyword.arg == "trial_receipt_durable_callback"
    )
    assert isinstance(callback, ast.Name)
    assert callback.id == "checkpoint_partial_trial"


def test_real_main_handles_durable_abort_without_entering_finalization() -> None:
    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    phases = {
        node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "aborted" in phases
    assert "truth_editing_adaptive_controller_abort_v1" in phases


def test_real_main_executes_reserved_finalization_not_only_handoff() -> None:
    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    direct_calls = {
        node.func.id
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "write_adaptive_finalization_handoff" in direct_calls
    assert "run_adaptive_finalization" in direct_calls
    finalization_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_adaptive_finalization"
    )
    callback = next(
        keyword.value
        for keyword in finalization_call.keywords
        if keyword.arg == "progress_callback"
    )
    assert isinstance(callback, ast.Name)
    assert callback.id == "record_finalization_progress"
    close_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "monitor"
    ]
    assert any(finalization_call.lineno < call.lineno for call in close_calls)


def test_preminimum_abort_exits_after_durable_publication_without_finalization() -> None:
    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    abort_branch = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and "aborted" in ast.unparse(node.test)
    )
    assert any(isinstance(node, ast.Return) for node in abort_branch.body)
    abort_line = abort_branch.lineno
    finalization_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_adaptive_finalization"
    )
    assert abort_line < finalization_call.lineno


def test_real_main_prepares_all_causal_receipts_after_workers_close() -> None:
    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    evaluator_context = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Name)
            and item.context_expr.id == "evaluator"
            for item in node.items
        )
    )
    causal_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "prepare_adaptive_causal_controls"
    )
    executor_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ProductionAdaptiveFinalizationExecutor"
    )
    finalization_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_adaptive_finalization"
    )

    # Leaving the context closes every fleet subprocess before cuda:0/cuda:1
    # are reused by the bounded sequential causal stage.
    assert evaluator_context.end_lineno is not None
    assert evaluator_context.end_lineno < causal_call.lineno
    assert causal_call.lineno < executor_call.lineno < finalization_call.lineno
    causal_keyword = next(
        keyword.value
        for keyword in executor_call.keywords
        if keyword.arg == "causal_control_receipts"
    )
    assert isinstance(causal_keyword, ast.Name)
    assert causal_keyword.id == "causal_receipts"
    root_keyword = next(
        keyword.value for keyword in causal_call.keywords if keyword.arg == "causal_root"
    )
    assert isinstance(root_keyword, ast.BinOp)
    causal_checkpoint = next(
        keyword.value
        for keyword in causal_call.keywords
        if keyword.arg == "after_candidate_commit"
    )
    assert isinstance(causal_checkpoint, ast.Name)
    assert causal_checkpoint.id == "checkpoint_causal_candidate"
    finalization_checkpoint = next(
        keyword.value
        for keyword in finalization_call.keywords
        if keyword.arg == "checkpoint_callback"
    )
    assert isinstance(finalization_checkpoint, ast.Name)
    assert finalization_checkpoint.id == "checkpoint_finalization_progress"


def test_runtime_config_rewrites_only_mutable_outputs_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "production.json"
    raw = {
        "format": "truth_editing_production_config_v1",
        "journal_path": "../artifacts/study/study-journal.json",
        "artifact_dir": "../artifacts/study/frozen",
        "runtime_output_dir": "../artifacts/study/runtime",
        "judge_cache_dir": "../artifacts/providers/judge-cache",
        "judge_budget_ledger_dir": "../artifacts/providers/judge-budget",
        "model_cache_dir": "../artifacts/model-cache",
        "snapshot_manifest_path": "../artifacts/snapshot.json",
        "frozen_input": "do-not-change",
    }
    source.write_text(json.dumps(raw))
    original = source.read_bytes()

    runtime_path = MODULE._runtime_config(source, tmp_path / "outputs")
    runtime = json.loads(runtime_path.read_text())

    assert source.read_bytes() == original
    assert runtime["frozen_input"] == "do-not-change"
    assert runtime["model_cache_dir"] == raw["model_cache_dir"]
    assert runtime["snapshot_manifest_path"] == raw["snapshot_manifest_path"]
    assert runtime["journal_path"] == str(
        (tmp_path / "outputs/study/study-journal.json").resolve()
    )
    assert runtime_path != source


def test_identity_valid_partial_receipt_marks_null_journal_batch_started(
    tmp_path: Path,
) -> None:
    unsigned = {
        "format": "truth_editing_vast_fleet_trial_receipt_v2",
        "fleet_config_sha256": "a" * 64,
        "trial_id": "trial-0003",
        "ordinal": 3,
        "request_sha256": "b" * 64,
        "worker_slot": 3,
        "result": {"outcome_kind": "success", "metrics": {}, "detail": None},
        "telemetry": {},
    }
    (tmp_path / "trial-0003.json").write_text(
        json.dumps({**unsigned, "receipt_sha256": canonical_sha256(unsigned)})
    )

    assert MODULE._batch_has_durable_receipt(
        tmp_path,
        fleet_config_sha256="a" * 64,
        completed_trials=0,
        batch_size=8,
    )


def test_partial_receipt_identity_drift_fails_closed(tmp_path: Path) -> None:
    unsigned = {
        "format": "truth_editing_vast_fleet_trial_receipt_v2",
        "fleet_config_sha256": "a" * 64,
        "trial_id": "trial-0003",
        "ordinal": 3,
        "request_sha256": "b" * 64,
        "worker_slot": 3,
        "result": {"outcome_kind": "success", "metrics": {}, "detail": None},
        "telemetry": {},
    }
    receipt = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    receipt["ordinal"] = 4
    (tmp_path / "trial-0003.json").write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="identity differs"):
        MODULE._batch_has_durable_receipt(
            tmp_path,
            fleet_config_sha256="a" * 64,
            completed_trials=0,
            batch_size=8,
        )


def _capacity_receipt() -> dict[str, object]:
    return {
        "budget": {
            "actual_total_usd": "1.1",
            "actual_infrastructure_usd": "1",
            "actual_evaluation_usd": "0.1",
            "pending_infrastructure_usd": "0.2",
            "pending_evaluation_usd": "0.05",
        }
    }


def _judge_receipt(*, actual: str = "0", completed: int = 0) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "format": "truth_editing_production_judge_budget_receipt_v1",
        "budget_config_sha256": "a" * 64,
        "judge_config_sha256": "b" * 64,
        "maximum_judge_spend_usd": "1",
        "actual_spend_usd": actual,
        "reserved_or_spent_usd": actual,
        "completed_call_count": completed,
        "pending_call_count": 0,
        "ambiguous_call_count": 0,
        "circuit_open": False,
        "circuit_event_sha256": None,
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _judge_snapshot(*, calls: int = 0, elapsed_ms: float = 0.0) -> dict[str, object]:
    return {
        "calls": calls,
        "failures": 0,
        "latency_ms": 0.0,
        "elapsed_ms": elapsed_ms,
        "cost_usd": 0.0,
    }


def test_live_spend_reader_uses_eight_gpu_host_total_rate_once_and_includes_pending_judge(
    tmp_path: Path,
) -> None:
    started = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    now = [started + timedelta(hours=2)]
    judge = SimpleNamespace(
        receipt=lambda: {
            "actual_spend_usd": "0.4",
            "reserved_or_spent_usd": "0.7",
        }
    )
    reader = MODULE._ControllerSpendReader(
        capacity_receipt=_capacity_receipt(),
        judge_budget=judge,
        host_hourly_usd=Decimal("0.25"),
        host_lease_started_at=started,
        worker_count=8,
        clock=lambda: now[0],
    )

    snapshot = reader()

    assert snapshot.actual_infrastructure_usd == Decimal("1.5")
    assert snapshot.actual_evaluation_usd == Decimal("0.5")
    assert snapshot.pending_infrastructure_usd == Decimal("0.2")
    assert snapshot.pending_evaluation_usd == Decimal("0.35")
    assert snapshot.reserved_total_usd == Decimal("2.55")


@pytest.mark.parametrize("hourly,workers", [("0", 8), ("0.2", 7)])
def test_live_spend_reader_fails_closed_without_exact_priced_eight_gpu_host(
    tmp_path: Path, hourly: str, workers: int
) -> None:
    with pytest.raises(ValueError, match="priced eight-GPU host"):
        MODULE._ControllerSpendReader(
            capacity_receipt=_capacity_receipt(),
            judge_budget=SimpleNamespace(receipt=lambda: {}),
            host_hourly_usd=Decimal(hourly),
            host_lease_started_at=datetime.now(timezone.utc),
            worker_count=workers,
        )


def test_live_spend_reader_counts_setup_before_controller_checkpoint_exists(
    tmp_path: Path,
) -> None:
    lease_started = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)
    now = lease_started + timedelta(hours=2)
    reader = MODULE._ControllerSpendReader(
        capacity_receipt=_capacity_receipt(),
        judge_budget=SimpleNamespace(
            receipt=lambda: {
                "actual_spend_usd": "0",
                "reserved_or_spent_usd": "0",
            }
        ),
        host_hourly_usd=Decimal("2.4"),
        host_lease_started_at=lease_started,
        worker_count=8,
        clock=lambda: now,
    )

    snapshot = reader()

    assert snapshot.actual_infrastructure_usd == Decimal("5.8")


def test_rolling_capacity_signs_conservative_observation_when_telemetry_is_missing(
    tmp_path: Path,
) -> None:
    judge_path = tmp_path / "judge-ledger"
    judge_path.mkdir()
    judge_receipt = _judge_receipt()
    judge_snapshot = _judge_snapshot()
    judge = SimpleNamespace(
        path=judge_path,
        receipt=lambda: dict(judge_receipt),
        monitoring_snapshot=lambda: dict(judge_snapshot),
    )
    rolling_path = tmp_path / "rolling.json"
    controller = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: NOW,
    )
    assert json.loads(rolling_path.read_text()) == _receipt()
    commit = SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(SimpleNamespace(trial_id=f"trial-{index:04d}") for index in range(8)),
    )

    controller.reforecast(commit)

    rolling = json.loads(rolling_path.read_text())
    rolling_bytes = rolling_path.read_bytes()
    controller.reforecast(commit)

    assert rolling["completed_through_trial"] == 8
    assert rolling["source_batch_observation_sha256"]
    assert controller.current_receipt() == rolling
    assert rolling_path.read_bytes() == rolling_bytes


def test_rolling_capacity_measures_dispatch_to_journal_wall_time_not_worker_only(
    tmp_path: Path,
) -> None:
    judge_path = tmp_path / "judge-ledger"
    judge_path.mkdir()
    judge_receipt = _judge_receipt()
    judge_snapshot = _judge_snapshot()
    judge = SimpleNamespace(
        path=judge_path,
        receipt=lambda: dict(judge_receipt),
        monitoring_snapshot=lambda: dict(judge_snapshot),
    )
    rolling_path = tmp_path / "rolling.json"
    now = [NOW]
    controller = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: now[0],
    )
    controller.record_batch_admission(completed_trials=0)
    for index in range(8):
        controller.record_trial(
            index,
            f"trial-{index:04d}",
            {
                "evaluation_seconds": 1.0,
                "generated_tokens": 120.0,
                "generated_tokens_per_second": 120.0,
            },
        )
    now[0] = NOW + timedelta(seconds=120)
    commit = SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(
            SimpleNamespace(trial_id=f"trial-{index:04d}")
            for index in range(8)
        ),
    )

    controller.reforecast(commit)

    rolling = json.loads(rolling_path.read_text())
    assert rolling["measured"]["trial_wall_seconds"] >= 120.0


def test_rolling_capacity_persists_infeasible_projection_before_typed_abort(
    tmp_path: Path,
) -> None:
    judge_path = tmp_path / "judge-ledger"
    judge_path.mkdir()
    judge_receipt = _judge_receipt()
    judge_snapshot = _judge_snapshot()
    judge = SimpleNamespace(
        path=judge_path,
        receipt=lambda: dict(judge_receipt),
        monitoring_snapshot=lambda: dict(judge_snapshot),
    )
    rolling_path = tmp_path / "rolling.json"
    controller = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: NOW,
        search_deadline_reader=lambda: NOW + timedelta(seconds=1),
    )
    commit = SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(
            SimpleNamespace(trial_id=f"trial-{index:04d}")
            for index in range(8)
        ),
    )

    with pytest.raises(MODULE.MinimumTrialGuaranteeError):
        controller.reforecast(commit)

    rolling = json.loads(rolling_path.read_text())
    assert rolling["completed_through_trial"] == 8
    assert rolling["capacity_limits"]["time_limited_trials"] == 8
    resumed = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: NOW,
        search_deadline_reader=lambda: NOW + timedelta(seconds=1),
    )
    with pytest.raises(MODULE.MinimumTrialGuaranteeError):
        resumed.reforecast(commit)


def test_replayed_reforecast_retires_only_its_stale_batch_clock(tmp_path: Path) -> None:
    judge_path = tmp_path / "judge-ledger"
    judge_path.mkdir()
    judge_receipt = _judge_receipt()
    judge_snapshot = _judge_snapshot()
    judge = SimpleNamespace(
        path=judge_path,
        receipt=lambda: dict(judge_receipt),
        monitoring_snapshot=lambda: dict(judge_snapshot),
    )
    rolling_path = tmp_path / "rolling.json"
    controller = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: NOW,
    )
    commit = SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(
            SimpleNamespace(trial_id=f"trial-{index:04d}")
            for index in range(8)
        ),
    )
    controller.reforecast(commit)
    controller.record_batch_admission(completed_trials=0)

    resumed = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: NOW,
    )
    resumed.reforecast(commit)
    resumed.record_batch_admission(completed_trials=8)

    clock = json.loads(
        (judge_path / "controller-batch-clock.json").read_text()
    )
    assert clock["expected_completed_trials"] == 16


def test_rolling_capacity_recovers_snapshot_only_from_exact_live_ledger(
    tmp_path: Path,
) -> None:
    judge_path = tmp_path / "judge-ledger"
    judge_path.mkdir()
    current_receipt = _judge_receipt()
    current_snapshot = _judge_snapshot()
    judge = SimpleNamespace(
        path=judge_path,
        receipt=lambda: dict(current_receipt),
        monitoring_snapshot=lambda: dict(current_snapshot),
    )
    rolling_path = tmp_path / "rolling.json"
    controller = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: NOW,
    )
    commit = SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(SimpleNamespace(trial_id=f"trial-{index:04d}") for index in range(8)),
    )
    controller.reforecast(commit)

    stale_receipt = _judge_receipt(actual="0.01", completed=1)
    stale_unsigned = {
        "format": "truth_editing_controller_judge_capacity_snapshot_v1",
        "receipt": stale_receipt,
        "monitoring_snapshot": _judge_snapshot(calls=1, elapsed_ms=5.0),
    }
    state_path = judge_path / "controller-capacity-snapshot.json"
    state_path.write_text(
        json.dumps(
            {**stale_unsigned, "self_sha256": canonical_sha256(stale_unsigned)}
        )
    )

    MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: NOW,
    )

    recovered = json.loads(state_path.read_text())
    assert recovered["receipt"]["content_sha256"] == current_receipt["content_sha256"]


def test_rolling_capacity_fails_when_snapshot_and_live_ledger_miss_lineage(
    tmp_path: Path,
) -> None:
    judge_path = tmp_path / "judge-ledger"
    judge_path.mkdir()
    initial_judge_receipt = _judge_receipt()
    judge = SimpleNamespace(
        path=judge_path,
        receipt=lambda: dict(initial_judge_receipt),
        monitoring_snapshot=lambda: _judge_snapshot(),
    )
    rolling_path = tmp_path / "rolling.json"
    controller = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(),
        judge_budget=judge,
        clock=lambda: NOW,
    )
    commit = SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(SimpleNamespace(trial_id=f"trial-{index:04d}") for index in range(8)),
    )
    controller.reforecast(commit)
    stale_receipt = _judge_receipt(actual="0.01", completed=1)
    stale_unsigned = {
        "format": "truth_editing_controller_judge_capacity_snapshot_v1",
        "receipt": stale_receipt,
        "monitoring_snapshot": _judge_snapshot(calls=1, elapsed_ms=5.0),
    }
    (judge_path / "controller-capacity-snapshot.json").write_text(
        json.dumps(
            {**stale_unsigned, "self_sha256": canonical_sha256(stale_unsigned)}
        )
    )
    different_live = _judge_receipt(actual="0.02", completed=1)
    judge.receipt = lambda: dict(different_live)

    with pytest.raises(ValueError, match="rolling receipt lineage"):
        MODULE._RollingCapacityController(
            policy=_policy(),
            initial_receipt=_receipt(),
            rolling_receipt_path=rolling_path,
            spend_reader=lambda: _spend(),
            judge_budget=judge,
            clock=lambda: NOW,
        )
