from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_truth_editing_adaptive_run import NOW, _policy, _receipt, _spend
from intelligent_liars.truth_editing_capacity import (
    SpendSnapshot,
    build_capacity_receipt,
    load_capacity_measurement,
)
from intelligent_liars.truth_editing_contracts import canonical_sha256
from intelligent_liars.truth_editing_live_judge import FROZEN_JUDGE_CONFIG_SHA256
from intelligent_liars.truth_editing_offhost_checkpoint import (
    OffHostCheckpointError,
    PartialBatchBinding,
    PartialTrialReceiptBinding,
    SnapshotBinding,
)
from intelligent_liars.truth_editing_production_judge_budget import (
    ProductionJudgeBudget,
    ProductionJudgeBudgetConfig,
)


SCRIPT = Path(__file__).parents[1] / "scripts/run_truth_editing_cuda_fleet_controller.py"
SPEC = importlib.util.spec_from_file_location(
    "run_truth_editing_cuda_fleet_controller", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RECEIPT_DIRECTORY = Path("/workspace/outputs/fleet-receipts")


def test_ordered_durable_callback_preserves_frontier_under_concurrent_dispatch() -> None:
    published: list[int] = []
    active = 0
    peak = 0
    lock = threading.Lock()

    def publish(event: dict[str, object]) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.005)
        published.append(int(event["ordinal"]))
        with lock:
            active -= 1

    callback = MODULE._OrderedDurableCallback(
        publish,
        next_ordinal_reader=lambda: 0,
    )
    threads = [
        threading.Thread(target=callback, args=({"ordinal": ordinal},))
        for ordinal in reversed(range(8))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert published == list(range(8))
    assert peak == 1


def test_ordered_durable_callback_releases_waiters_after_frontier_failure() -> None:
    calls: list[int] = []

    def publish(event: dict[str, object]) -> None:
        calls.append(int(event["ordinal"]))
        if event["ordinal"] == 0:
            raise RuntimeError("publication failed")

    callback = MODULE._OrderedDurableCallback(
        publish,
        next_ordinal_reader=lambda: 0,
    )
    errors: list[type[BaseException]] = []

    def invoke(ordinal: int) -> None:
        try:
            callback({"ordinal": ordinal})
        except BaseException as error:
            errors.append(type(error))

    threads = [threading.Thread(target=invoke, args=(ordinal,)) for ordinal in (1, 0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert calls == [0]
    assert len(errors) == 2


def _partial_binding() -> PartialBatchBinding:
    return PartialBatchBinding(
        committed=SnapshotBinding(
            study_identity_sha256="a" * 64,
            study_config_sha256="b" * 64,
            fleet_config_sha256="c" * 64,
            optuna_study_name="study",
            wandb_run_id="wandb-run",
            completed_trials=48,
        ),
        batch_ordinal=6,
        batch_size=8,
        durable_receipts=(
            PartialTrialReceiptBinding(
                trial_id="trial-0048",
                ordinal=48,
                proposal_sha256="d" * 64,
                request_sha256="e" * 64,
                receipt_sha256="f" * 64,
            ),
        ),
    )


def _durable_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "format": "truth_editing_vast_fleet_receipt_durable_event_v1",
        "fleet_config_sha256": "c" * 64,
        "trial_id": "trial-0048",
        "ordinal": 48,
        "request_sha256": "e" * 64,
        "receipt_path": "/workspace/outputs/fleet-receipts/trial-0048.json",
        "receipt_sha256": "f" * 64,
    }
    event.update(overrides)
    return event


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


def test_exact_restored_partial_event_is_already_published() -> None:
    binding = _partial_binding()
    repository = SimpleNamespace(
        read_latest_partial_binding_if_present=lambda committed: (
            binding if committed == binding.committed else None
        )
    )

    assert MODULE.offhost_partial_event_is_already_published(
        repository=repository,
        committed_binding=binding.committed,
        durable_event=_durable_event(),
        expected_receipt_directory=RECEIPT_DIRECTORY,
    ) is True


def test_one_based_restored_partial_event_is_already_published() -> None:
    committed = SnapshotBinding(
        **{
            **_partial_binding().committed.to_mapping(),
            "trial_number_start": 1,
        }
    )
    binding = PartialBatchBinding(
        committed=committed,
        batch_ordinal=6,
        batch_size=8,
        durable_receipts=(
            PartialTrialReceiptBinding(
                trial_id="trial-0049",
                ordinal=48,
                proposal_sha256="d" * 64,
                request_sha256="e" * 64,
                receipt_sha256="f" * 64,
            ),
        ),
    )
    repository = SimpleNamespace(
        read_latest_partial_binding_if_present=lambda candidate: (
            binding if candidate == committed else None
        )
    )

    assert MODULE.offhost_partial_event_is_already_published(
        repository=repository,
        committed_binding=committed,
        durable_event=_durable_event(
            trial_id="trial-0049",
            receipt_path="/workspace/outputs/fleet-receipts/trial-0049.json",
        ),
        expected_receipt_directory=RECEIPT_DIRECTORY,
        trial_number_start=1,
    ) is True


def test_restored_partial_event_missing_from_frontier_requires_publication() -> None:
    binding = _partial_binding()
    repository = SimpleNamespace(
        read_latest_partial_binding_if_present=lambda committed: (
            binding if committed == binding.committed else None
        )
    )

    assert MODULE.offhost_partial_event_is_already_published(
        repository=repository,
        committed_binding=binding.committed,
        durable_event=_durable_event(
            trial_id="trial-0049",
            ordinal=49,
            request_sha256="1" * 64,
            receipt_path="/workspace/outputs/fleet-receipts/trial-0049.json",
            receipt_sha256="2" * 64,
        ),
        expected_receipt_directory=RECEIPT_DIRECTORY,
    ) is False


@pytest.mark.parametrize(
    ("field", "different"),
    (
        ("trial_id", "trial-9999"),
        ("request_sha256", "1" * 64),
        ("receipt_sha256", "2" * 64),
        ("fleet_config_sha256", "3" * 64),
    ),
)
def test_restored_partial_event_identity_mismatch_fails_closed(
    field: str, different: object
) -> None:
    binding = _partial_binding()
    repository = SimpleNamespace(
        read_latest_partial_binding_if_present=lambda committed: binding
    )

    with pytest.raises(OffHostCheckpointError, match="identity differs"):
        MODULE.offhost_partial_event_is_already_published(
            repository=repository,
            committed_binding=binding.committed,
            durable_event=_durable_event(**{field: different}),
            expected_receipt_directory=RECEIPT_DIRECTORY,
        )


@pytest.mark.parametrize(
    "event",
    (
        _durable_event(format="different"),
        _durable_event(ordinal="48"),
        _durable_event(ordinal=49),
        _durable_event(
            receipt_path="/workspace/outputs/fleet-receipts/trial-9999.json"
        ),
        {key: value for key, value in _durable_event().items() if key != "receipt_path"},
        {**_durable_event(), "unexpected": True},
    ),
)
def test_restored_partial_event_malformed_identity_fails_closed(
    event: dict[str, object],
) -> None:
    binding = _partial_binding()
    repository = SimpleNamespace(
        read_latest_partial_binding_if_present=lambda committed: binding
    )

    with pytest.raises(OffHostCheckpointError, match="identity differs"):
        MODULE.offhost_partial_event_is_already_published(
            repository=repository,
            committed_binding=binding.committed,
            durable_event=event,
            expected_receipt_directory=RECEIPT_DIRECTORY,
        )


def test_partial_checkpoint_skips_exact_bound_event_before_republication() -> None:
    class Repository:
        def __init__(self, binding: PartialBatchBinding) -> None:
            self.binding = binding
            self.publication_calls = 0

        def read_latest_partial_binding_if_present(
            self, committed: SnapshotBinding
        ) -> PartialBatchBinding | None:
            return self.binding if committed == self.binding.committed else None

        def publish_partial_from_runtime(self) -> object:
            self.publication_calls += 1
            return object()

    binding = _partial_binding()
    repository = Repository(binding)

    result = MODULE.publish_partial_event_if_needed(
        repository=repository,
        committed_binding=binding.committed,
        durable_event=_durable_event(),
        expected_receipt_directory=RECEIPT_DIRECTORY,
        publish=repository.publish_partial_from_runtime,
    )

    assert result is None
    assert repository.publication_calls == 0


def test_partial_checkpoint_publishes_event_missing_from_restored_frontier() -> None:
    binding = _partial_binding()
    publication_calls = 0

    def publish() -> str:
        nonlocal publication_calls
        publication_calls += 1
        return "published"

    result = MODULE.publish_partial_event_if_needed(
        repository=SimpleNamespace(
            read_latest_partial_binding_if_present=lambda committed: binding
        ),
        committed_binding=binding.committed,
        durable_event=_durable_event(
            trial_id="trial-0049",
            ordinal=49,
            request_sha256="1" * 64,
            receipt_path="/workspace/outputs/fleet-receipts/trial-0049.json",
            receipt_sha256="2" * 64,
        ),
        expected_receipt_directory=RECEIPT_DIRECTORY,
        publish=publish,
    )

    assert result == "published"
    assert publication_calls == 1


def test_partial_checkpoint_mismatch_fails_before_republication() -> None:
    binding = _partial_binding()
    publication_calls = 0

    def publish() -> None:
        nonlocal publication_calls
        publication_calls += 1

    with pytest.raises(OffHostCheckpointError, match="identity differs"):
        MODULE.publish_partial_event_if_needed(
            repository=SimpleNamespace(
                read_latest_partial_binding_if_present=lambda committed: binding
            ),
            committed_binding=binding.committed,
            durable_event=_durable_event(request_sha256="1" * 64),
            expected_receipt_directory=RECEIPT_DIRECTORY,
            publish=publish,
        )

    assert publication_calls == 0


def test_controller_reconciles_ambiguous_judge_requests_after_replay_publication() -> None:
    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    callback = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "after_complete_batch"
    )
    lifecycle_call = next(
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "publish_completed_boundary_then_reconcile_judge_budget"
    )
    commit_call = next(
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "commit_batch"
    )
    startup_recovery_calls = [
        node
        for node in main.body
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "judge_budget"
        and node.func.attr == "recover_orphaned_reservations"
    ]
    direct_acknowledgements = [
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "acknowledge_ambiguous_transport_circuit"
    ]

    assert commit_call.lineno < lifecycle_call.lineno
    assert startup_recovery_calls == []
    assert direct_acknowledgements == []


def test_hydrated_partial_judge_bytes_publish_before_orphan_reconciliation(
    tmp_path: Path,
) -> None:
    class SimulatedWorkerExit(BaseException):
        pass

    class InterruptedTransport:
        def complete(self, request: object) -> dict[str, object]:
            del request
            raise SimulatedWorkerExit

    config = ProductionJudgeBudgetConfig.from_mapping(
        {
            "format": "truth_editing_production_judge_budget_config_v1",
            "all_in_maximum_spend_usd": "50",
            "non_judge_reserved_spend_usd": "49",
            "maximum_judge_spend_usd": "1",
            "per_call_reservation_usd": "0.025",
            "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        }
    )
    root = tmp_path / "hydrated-ledger"
    budget = ProductionJudgeBudget(root, config=config)
    request = {
        "model": "z-ai/glm-5.3-flash",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "stored fixture"}],
    }
    with pytest.raises(SimulatedWorkerExit):
        budget.transport(InterruptedTransport()).complete(request)
    hydrated_bytes = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*.json")
    }
    published = False

    def publish_restored_partial() -> None:
        nonlocal published
        assert {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*.json")
        } == hydrated_bytes
        assert not tuple(root.glob("calls/*/ambiguous.json"))
        published = True

    recovered, reconciled = (
        MODULE.publish_completed_boundary_then_reconcile_judge_budget(
            publish_completed_boundary=publish_restored_partial,
            judge_budget=budget,
        )
    )

    assert published is True
    assert len(recovered) == 1
    assert reconciled == ()
    assert tuple(root.glob("calls/*/ambiguous.json"))
    assert not tuple(root.glob("calls/*/recovery-supersession.json"))
    budget_receipt = budget.receipt()
    assert budget_receipt["pending_call_count"] == 0
    assert budget_receipt["ambiguous_call_count"] == 1
    assert budget_receipt["circuit_open"] is False
    assert next(root.glob("calls/*/reservation.json")).read_bytes() in (
        hydrated_bytes.values()
    )


def test_post_batch_judge_reconciliation_starts_after_durable_publication() -> None:
    events: list[str] = []
    orphan = {"status": "ambiguous", "request_sha256": "a" * 64}
    reconciled = {"status": "reconciled", "request_sha256": "a" * 64}

    class JudgeBudget:
        def recover_orphaned_reservations(self) -> tuple[dict[str, object], ...]:
            events.append("recover_orphans")
            return (orphan,)

        def reconcile_ambiguous_requests(self) -> tuple[dict[str, object], ...]:
            events.append("reconcile_ambiguous")
            return (reconciled,)

    result = MODULE.publish_completed_boundary_then_reconcile_judge_budget(
        publish_completed_boundary=lambda: events.append("publish_boundary"),
        judge_budget=JudgeBudget(),
    )

    assert events == [
        "publish_boundary",
        "recover_orphans",
        "reconcile_ambiguous",
    ]
    assert result == ((orphan,), (reconciled,))


def test_restored_judge_budget_reconciles_only_from_a_durable_boundary() -> None:
    events: list[str] = []

    class JudgeBudget:
        def receipt(self) -> dict[str, object]:
            return {
                "pending_call_count": 1,
                "ambiguous_call_count": 0,
                "circuit_open": False,
            }

        def recover_orphaned_reservations(self) -> tuple[dict[str, object], ...]:
            events.append("recover_orphans")
            return ({"status": "ambiguous"},)

        def reconcile_ambiguous_requests(self) -> tuple[dict[str, object], ...]:
            events.append("reconcile_ambiguous")
            return ({"status": "reconciled"},)

    recovered, reconciled = MODULE.prepare_judge_budget_for_controller_resume(
        judge_budget=JudgeBudget(),
        restored_boundary_is_durable=True,
    )

    assert events == ["recover_orphans", "reconcile_ambiguous"]
    assert recovered == ({"status": "ambiguous"},)
    assert reconciled == ({"status": "reconciled"},)


def test_local_resume_refuses_to_mutate_unbound_unresolved_paid_calls() -> None:
    events: list[str] = []

    class JudgeBudget:
        def receipt(self) -> dict[str, object]:
            return {
                "pending_call_count": 1,
                "ambiguous_call_count": 0,
                "circuit_open": False,
            }

        def recover_orphaned_reservations(self) -> tuple[dict[str, object], ...]:
            events.append("recover_orphans")
            return ()

        def reconcile_ambiguous_requests(self) -> tuple[dict[str, object], ...]:
            events.append("reconcile_ambiguous")
            return ()

    with pytest.raises(ValueError, match="without a durable restored boundary"):
        MODULE.prepare_judge_budget_for_controller_resume(
            judge_budget=JudgeBudget(),
            restored_boundary_is_durable=False,
        )

    assert events == []


def test_finalization_resume_rejects_judge_ledger_drift_before_reconciliation() -> None:
    events: list[str] = []

    class JudgeBudget:
        def receipt(self) -> dict[str, object]:
            return {"content_sha256": "a" * 64}

        def recover_orphaned_reservations(self) -> tuple[dict[str, object], ...]:
            events.append("recover_orphans")
            return ()

        def reconcile_ambiguous_requests(self) -> tuple[dict[str, object], ...]:
            events.append("reconcile_ambiguous")
            return ()

    with pytest.raises(ValueError, match="restored finalization judge ledger differs"):
        MODULE.reconcile_restored_finalization_judge_budget(
            judge_budget=JudgeBudget(),
            expected_receipt_sha256="b" * 64,
        )

    assert events == []


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

    initial_callback = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "after_prepare_before_first_admission"
    )
    initial_source = ast.unparse(initial_callback)
    assert "resuming_durable_batch = _batch_has_durable_receipt" in initial_source
    assert "batch_started=resuming_durable_batch" in initial_source
    resume_guard = next(
        node
        for node in initial_callback.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "resuming_durable_batch"
    )
    assert any(isinstance(node, ast.Return) for node in resume_guard.body)
    assert initial_callback.body.index(resume_guard) < next(
        index
        for index, node in enumerate(initial_callback.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "publish_boundary"
    )

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
    publish_calls = [
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "publish_completed_boundary_then_reconcile_judge_budget"
    ]
    assert len(publish_calls) == 2
    commit_call = next(
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "commit_batch"
    )
    assert commit_call.lineno < max(call.lineno for call in publish_calls)

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
    assert "rearm_expired_search_lease" in calls


def test_offhost_auto_resume_accepts_only_bootstrap_model_receipts(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    model_root = output_root / "model"
    model_root.mkdir(parents=True)
    (model_root / "cache-hydration-receipt.json").write_text("{}\n")
    (model_root / "model-verification-receipt.json").write_text("{}\n")

    assert MODULE._output_root_is_clean_for_offhost_restore(output_root)

    def hydrate(
        snapshot_root: Path,
        clean_output_root: Path,
        *,
        binding: object,
    ) -> dict[str, object]:
        assert snapshot_root == tmp_path / "snapshot"
        assert binding == "binding"
        assert not clean_output_root.exists()
        (clean_output_root / "study").mkdir(parents=True)
        (clean_output_root / "study/checkpoint.json").write_text("{}\n")
        return {"hydrated": True}

    result = MODULE._hydrate_preserving_bootstrap_outputs(
        tmp_path / "snapshot",
        output_root,
        binding="binding",
        hydrate=hydrate,
    )
    assert result == {"hydrated": True}
    assert (output_root / "study/checkpoint.json").is_file()
    assert (model_root / "cache-hydration-receipt.json").is_file()
    assert (model_root / "model-verification-receipt.json").is_file()

    (output_root / "unexpected.json").write_text("{}\n")
    assert not MODULE._output_root_is_clean_for_offhost_restore(output_root)


def test_real_main_reconciles_only_a_validated_restored_judge_boundary() -> None:
    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "prepare_judge_budget_for_controller_resume"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    restored = keywords["restored_boundary_is_durable"]
    assert isinstance(restored, ast.Name)
    assert restored.id == "restored_judge_boundary_is_durable"
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "restored_judge_boundary_is_durable"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in ast.walk(main)
    )
    assert sum(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "restored_judge_boundary_is_durable"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        for node in ast.walk(main)
    ) >= 3


def test_offhost_auto_resume_uses_complete_planned_study_identity() -> None:
    """The S3 binding is not the fleet's similarly named config hash."""

    tree = ast.parse(SCRIPT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    read_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_latest_binding_if_present"
    )
    study_identity = next(
        keyword.value
        for keyword in read_call.keywords
        if keyword.arg == "expected_study_identity_sha256"
    )

    assert isinstance(study_identity, ast.Name)
    assert study_identity.id == "planned_study_identity_sha256"


def test_replayed_batch_reuses_already_recorded_progress_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "adaptive-progress.json"
    path.write_text("placeholder")
    progress = SimpleNamespace(
        completed_search_trials=8,
        coverage={"direction_family": (1, 3)},
    )
    monkeypatch.setattr(
        MODULE,
        "open_adaptive_progress_checkpoint",
        lambda _path: SimpleNamespace(progress=progress),
    )

    assert MODULE.adaptive_progress_boundary_is_already_recorded(
        path,
        completed_trials=8,
        coverage={"direction_family": (1, 3)},
    )


def test_replayed_batch_reuses_already_published_offhost_boundary() -> None:
    binding = SimpleNamespace(completed_trials=48, identity="same")

    assert MODULE.offhost_boundary_is_already_published(binding, binding)
    assert not MODULE.offhost_boundary_is_already_published(None, binding)
    assert not MODULE.offhost_boundary_is_already_published(
        binding,
        SimpleNamespace(completed_trials=56, identity="same"),
    )


def test_replayed_batch_progress_check_fails_closed_on_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "adaptive-progress.json"
    path.write_text("placeholder")
    progress = SimpleNamespace(
        completed_search_trials=8,
        coverage={"direction_family": (0, 3)},
    )
    monkeypatch.setattr(
        MODULE,
        "open_adaptive_progress_checkpoint",
        lambda _path: SimpleNamespace(progress=progress),
    )

    with pytest.raises(ValueError, match="coverage differs"):
        MODULE.adaptive_progress_boundary_is_already_recorded(
            path,
            completed_trials=8,
            coverage={"direction_family": (1, 3)},
        )
    with pytest.raises(ValueError, match="ahead"):
        MODULE.adaptive_progress_boundary_is_already_recorded(
            path,
            completed_trials=0,
            coverage={"direction_family": (0, 3)},
        )


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
    assert callback.id == "ordered_checkpoint_partial_trial"


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

    second_runtime_path = MODULE._runtime_config(source, tmp_path / "other-outputs")
    assert second_runtime_path != runtime_path
    assert json.loads(second_runtime_path.read_text())["journal_path"] == str(
        (tmp_path / "other-outputs/study/study-journal.json").resolve()
    )


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


def test_one_based_partial_receipt_marks_null_journal_batch_started(
    tmp_path: Path,
) -> None:
    unsigned = {
        "format": "truth_editing_vast_fleet_trial_receipt_v2",
        "fleet_config_sha256": "a" * 64,
        "trial_id": "trial-0004",
        "ordinal": 3,
        "request_sha256": "b" * 64,
        "worker_slot": 3,
        "result": {"outcome_kind": "success", "metrics": {}, "detail": None},
        "telemetry": {},
    }
    (tmp_path / "trial-0004.json").write_text(
        json.dumps({**unsigned, "receipt_sha256": canonical_sha256(unsigned)})
    )

    assert MODULE._batch_has_durable_receipt(
        tmp_path,
        fleet_config_sha256="a" * 64,
        completed_trials=0,
        batch_size=8,
        trial_number_start=1,
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


def test_rearm_spend_reader_carries_prior_host_cost_into_new_lease(
    tmp_path: Path,
) -> None:
    old_lease = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    new_lease = old_lease + timedelta(hours=20)
    prior = SpendSnapshot.from_mapping(
        {
            "actual_total_usd": "49.5",
            "actual_infrastructure_usd": "48.9",
            "actual_evaluation_usd": "0.6",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0.1",
        }
    )
    reader = MODULE._ControllerSpendReader(
        capacity_receipt=_capacity_receipt(),
        judge_budget=SimpleNamespace(
            receipt=lambda: {
                "actual_spend_usd": "0.5",
                "reserved_or_spent_usd": "0.6",
            }
        ),
        host_hourly_usd=Decimal("1.26"),
        host_lease_started_at=new_lease,
        worker_count=8,
        prior_spend=prior,
        clock=lambda: new_lease,
    )

    snapshot = reader()

    assert snapshot.actual_infrastructure_usd == Decimal("48.9")
    assert snapshot.actual_evaluation_usd == Decimal("0.6")


def test_resumed_spend_baseline_does_not_depend_on_minimum_rearm(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "adaptive-run-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "last_spend_snapshot": {
                    "actual_total_usd": "32.5",
                    "actual_infrastructure_usd": "32",
                    "actual_evaluation_usd": "0.5",
                    "pending_infrastructure_usd": "0",
                    "pending_evaluation_usd": "0.1",
                }
            }
        )
    )

    spend = MODULE._prior_spend_from_checkpoint(checkpoint)

    assert spend is not None
    assert spend.actual_infrastructure_usd == Decimal("32")
    assert MODULE._prior_spend_from_checkpoint(tmp_path / "missing.json") is None


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


def test_rolling_capacity_rounds_fractional_batch_judge_elapsed_up(
    tmp_path: Path,
) -> None:
    judge_path = tmp_path / "judge-ledger"
    judge_path.mkdir()
    judge_state = {
        "receipt": _judge_receipt(),
        "snapshot": _judge_snapshot(),
    }
    judge = SimpleNamespace(
        path=judge_path,
        receipt=lambda: dict(judge_state["receipt"]),
        monitoring_snapshot=lambda: dict(judge_state["snapshot"]),
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
    judge_state["receipt"] = _judge_receipt(actual="0.01", completed=1)
    judge_state["snapshot"] = _judge_snapshot(
        calls=1,
        elapsed_ms=741754.3856198245,
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

    rolling = json.loads(rolling_path.read_text())
    assert (
        Decimal(str(rolling["measured"]["judge_latency_seconds"])) * 8
        >= Decimal("741.7543856198245")
    )


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


def test_complete_worker_telemetry_wall_covers_generation_plus_signed_judge(
    tmp_path: Path,
) -> None:
    measurement_unsigned = {
        "format": "truth_editing_capacity_measurement_v1",
        "measurement_id": "signed-high-judge-bound",
        "observed_at": "2026-08-28T11:30:00Z",
        "timed_canary_receipt_sha256": "a" * 64,
        "generated_tokens": 103,
        "tokens_per_second": 1.0,
        "trial_wall_seconds": 191.0,
        "judge_latency_seconds": 88.0,
        "judge_cost_usd_per_trial": "0.0001",
        "per_gpu_hourly_usd": "0.05",
        "projected_storage_network_usd": "0.1",
        "spend": {
            "actual_total_usd": "1.1",
            "actual_infrastructure_usd": "1",
            "actual_evaluation_usd": "0.1",
            "pending_infrastructure_usd": "0",
            "pending_evaluation_usd": "0",
        },
    }
    measurement = load_capacity_measurement(
        {
            **measurement_unsigned,
            "self_sha256": canonical_sha256(measurement_unsigned),
        },
        now=NOW,
    )
    initial_receipt = build_capacity_receipt(
        policy=_policy(), measurement=measurement, planned_at=NOW
    )
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
        initial_receipt=initial_receipt,
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
                "evaluation_seconds": 103.0,
                "generated_tokens": 103.0,
                "generated_tokens_per_second": 1.0,
            },
        )
    now[0] += timedelta(seconds=5)
    controller.reforecast(SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(
            SimpleNamespace(trial_id=f"trial-{index:04d}")
            for index in range(8)
        ),
    ))

    rolling = json.loads(rolling_path.read_text())
    assert rolling["measured"]["trial_wall_seconds"] == pytest.approx(191.0)


def test_rolling_capacity_resume_excludes_offline_repair_time_but_keeps_spend(
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
        spend_reader=lambda: _spend(infrastructure="2"),
        judge_budget=judge,
        clock=lambda: now[0],
    )
    controller.record_batch_admission(completed_trials=0)

    now[0] = NOW + timedelta(hours=2)
    resumed = MODULE._RollingCapacityController(
        policy=_policy(),
        initial_receipt=_receipt(),
        rolling_receipt_path=rolling_path,
        spend_reader=lambda: _spend(infrastructure="2"),
        judge_budget=judge,
        clock=lambda: now[0],
    )
    for index in range(8):
        resumed.record_trial(
            index,
            f"trial-{index:04d}",
            {
                "evaluation_seconds": 1.0,
                "generated_tokens": 120.0,
                "generated_tokens_per_second": 120.0,
            },
        )
    now[0] += timedelta(seconds=10)
    resumed.reforecast(SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(
            SimpleNamespace(trial_id=f"trial-{index:04d}")
            for index in range(8)
        ),
    ))

    rolling = json.loads(rolling_path.read_text())
    assert rolling["measured"]["trial_wall_seconds"] < 60.0
    assert rolling["budget"]["actual_infrastructure_usd"] == "2"


def test_rolling_capacity_clock_accumulates_only_explicit_active_intervals(
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
    now[0] += timedelta(seconds=12)
    controller.pause_batch_clock(completed_trials=0)
    now[0] += timedelta(hours=3)
    controller.record_batch_admission(completed_trials=0)
    now[0] += timedelta(seconds=18)

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
    controller.reforecast(SimpleNamespace(
        batch_ordinal=0,
        batch_sha256="e" * 64,
        batch_size=8,
        completed_trials=8,
        trials=tuple(
            SimpleNamespace(trial_id=f"trial-{index:04d}")
            for index in range(8)
        ),
    ))

    rolling = json.loads(rolling_path.read_text())
    assert rolling["measured"]["trial_wall_seconds"] == pytest.approx(30.0)


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
