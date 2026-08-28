from __future__ import annotations

import json
import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.truth_editing_batch_execution import BatchEvaluationRequest
from intelligent_liars.truth_editing_study import EvaluationResult
from intelligent_liars.truth_editing_vast_fleet import (
    FleetConfig,
    FleetError,
    FleetBatchEvaluator,
    FleetCircuitOpen,
    SubprocessCudaWorker,
    VastLifecycleTrialWorker,
    verify_adaptive_fleet_bindings,
    verify_production_config_binding,
)
from intelligent_liars.truth_editing_vast_prerequisites import Offer
from intelligent_liars.truth_editing_vast_prerequisites import EphemeralWorkloadSecret
from intelligent_liars.truth_editing_vast_production import ProductionVastConfig
from intelligent_liars.truth_editing_vast_production import production_lifecycle_plan
from intelligent_liars.truth_editing_failure_policy import PaidJudgeCircuitOpen
from intelligent_liars.truth_editing_live_judge import FROZEN_JUDGE_CONFIG_SHA256
from test_truth_editing_vast_production import _raw as _production_job_raw


def _sha(char: str) -> str:
    return char * 64


def _judge_budget() -> dict[str, str]:
    return {
        "format": "truth_editing_production_judge_budget_config_v1",
        "all_in_maximum_spend_usd": "50",
        "non_judge_reserved_spend_usd": "45",
        "maximum_judge_spend_usd": "5",
        "per_call_reservation_usd": "0.025",
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
    }


def _budget() -> dict[str, Any]:
    unsigned = {
        "format": "truth_editing_vast_fleet_budget_v1",
        "all_in_maximum_spend_usd": "50",
        "maximum_infrastructure_spend_usd": "45",
        "maximum_judge_spend_usd": "5",
        "production_judge_budget_config_sha256": hashlib.sha256(
            json.dumps(
                _judge_budget(),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "included_infrastructure_costs": [
            "gpu_compute",
            "storage",
            "network_download",
            "network_upload",
        ],
        "maximum_host_lease_seconds": 86400,
        "maximum_fetch_gib": 1.0,
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**unsigned, "identity_sha256": digest}


def _raw() -> dict[str, Any]:
    return {
        "format": "truth_editing_vast_fleet_v2",
        "fleet_id": "truth-editing-v3-tonight",
        "phase_boundaries": {"discovery": 80, "expanded": 160, "finalist": 200},
        "execution_mode": "persistent_single_host_eight_gpu",
        "worker_count": 8,
        "budget": _budget(),
        "production_config": {
            "path": "configs/truth_editing_production_v3_r10_0123456789abcdef.json",
            "sha256": _sha("a"),
        },
        "bundle_sha256": _sha("b"),
        "receipt_directory": "artifacts/truth-editing/fleet-v3/trials",
        "capability_test_access": False,
    }


def _adaptive_raw() -> dict[str, Any]:
    return {
        "format": "truth_editing_vast_fleet_v3",
        "fleet_id": "truth-editing-adaptive-main",
        "adaptive_capacity_policy": {
            "path": "configs/truth_editing_adaptive_capacity_policy_v1.json",
            "sha256": _sha("c"),
        },
        "study": {
            "path": "configs/truth_editing_study_v4_adaptive.json",
            "config_sha256": _sha("d"),
            "identity_sha256": _sha("e"),
        },
        "execution_topology": {
            "mode": "persistent_single_host_eight_gpu",
            "worker_count": 8,
            "batch_size": 8,
        },
        "budget": _budget(),
        "production_config": {
            "path": "configs/truth_editing_production_v5_adaptive.json",
            "sha256": _sha("a"),
        },
        "receipt_directory": "artifacts/truth-editing/adaptive-fleet/trials",
        "capability_test_access": False,
    }


def test_adaptive_fleet_contract_round_trips_identity_without_static_trial_stop() -> None:
    config = FleetConfig.from_mapping(_adaptive_raw())

    assert config.format == "truth_editing_vast_fleet_v3"
    assert config.worker_count == config.batch_size == 8
    assert config.adaptive_capacity_policy_path.endswith("capacity_policy_v1.json")
    assert config.adaptive_capacity_policy_sha256 == _sha("c")
    assert config.study_config_path.endswith("study_v4_adaptive.json")
    assert config.study_config_sha256 == _sha("d")
    assert config.study_identity_sha256 == _sha("e")
    assert config.identity == _adaptive_raw()
    assert "phase_boundaries" not in config.identity
    assert "bundle_sha256" not in config.identity
    assert config.bundle_sha256 is None
    with pytest.raises(FleetError, match="adaptive capacity policy"):
        config.stop_after_trials("finalist")


def test_adaptive_fleet_verifies_capacity_and_study_files_and_study_identity(
    tmp_path: Path,
) -> None:
    raw = _adaptive_raw()
    policy = tmp_path / raw["adaptive_capacity_policy"]["path"]
    study = tmp_path / raw["study"]["path"]
    policy.parent.mkdir(parents=True)
    policy.write_text('{"policy":"adaptive"}\n')
    study.write_text('{"study":"adaptive"}\n')
    raw["adaptive_capacity_policy"]["sha256"] = hashlib.sha256(
        policy.read_bytes()
    ).hexdigest()
    raw["study"]["config_sha256"] = hashlib.sha256(study.read_bytes()).hexdigest()
    config = FleetConfig.from_mapping(raw)

    assert verify_adaptive_fleet_bindings(
        config,
        repo=tmp_path,
        requested_capacity_policy_path=policy,
        requested_study_config_path=study,
        observed_study_identity_sha256=_sha("e"),
    ) == (policy, study)

    with pytest.raises(FleetError, match="study identity"):
        verify_adaptive_fleet_bindings(
            config,
            repo=tmp_path,
            requested_capacity_policy_path=policy,
            requested_study_config_path=study,
            observed_study_identity_sha256=_sha("f"),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(phase_boundaries={"finalist": 200}), "fields changed"),
        (lambda raw: raw.update(bundle_sha256=_sha("b")), "fields changed"),
        (
            lambda raw: raw["execution_topology"].update(worker_count=7),
            "exactly eight",
        ),
        (
            lambda raw: raw["execution_topology"].update(batch_size=7),
            "exactly eight",
        ),
        (
            lambda raw: raw["adaptive_capacity_policy"].update(sha256=_sha("x")),
            "SHA-256",
        ),
        (
            lambda raw: raw["study"].update(identity_sha256=_sha("x")),
            "SHA-256",
        ),
        (
            lambda raw: raw["study"].update(path="../study.json"),
            "safe relative JSON",
        ),
    ],
)
def test_adaptive_fleet_fails_closed_on_static_stop_or_identity_topology_drift(
    mutation, message: str
) -> None:
    raw = _adaptive_raw()
    mutation(raw)
    with pytest.raises(FleetError, match=message):
        FleetConfig.from_mapping(raw)


def _request(ordinal: int) -> BatchEvaluationRequest[dict[str, Any]]:
    return BatchEvaluationRequest(
        trial_id=f"trial-{ordinal:04d}",
        ordinal=ordinal,
        proposal={"strength": ordinal / 10},
        record_ids=("validation-1",),
        objective_names=("deception", "retained_truth"),
    )


def test_fleet_contract_freezes_v3_identity_budget_and_phase_barriers() -> None:
    config = FleetConfig.from_mapping(_raw())
    assert config.worker_count == 8
    assert str(config.maximum_infrastructure_spend_usd) == "45"
    assert str(config.maximum_judge_spend_usd) == "5"
    assert config.identity["budget"]["identity_sha256"] == config.budget_identity_sha256
    assert config.stop_after_trials("expanded") == 160

    raw = _raw()
    raw["production_config"]["path"] = "../truth_editing_production_v3.json"
    with pytest.raises(FleetError, match="relative JSON"):
        FleetConfig.from_mapping(raw)
    raw = _raw()
    raw["capability_test_access"] = True
    with pytest.raises(FleetError, match="capability-test"):
        FleetConfig.from_mapping(raw)


def test_fleet_accepts_content_qualified_r10_production_path_as_hash_authority(
    tmp_path: Path,
) -> None:
    raw = _raw()
    relative = "configs/truth_editing_production_v3_r10_0123456789abcdef.json"
    target = tmp_path / relative
    target.parent.mkdir()
    target.write_text('{"qualified":"r10"}\n')
    raw["production_config"]["path"] = relative
    raw["production_config"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    config = FleetConfig.from_mapping(raw)
    assert config.production_config_path.endswith("r10_0123456789abcdef.json")
    assert verify_production_config_binding(config, repo=tmp_path) == target
    with pytest.raises(FleetError, match="requested production config path"):
        verify_production_config_binding(
            config, repo=tmp_path, requested_path=tmp_path / "configs/other.json"
        )
    raw = _raw()
    raw["phase_boundaries"]["expanded"] = 159
    with pytest.raises(FleetError, match="80/160/200"):
        FleetConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(worker_count=7), "exactly eight"),
        (
            lambda raw: raw["budget"].update(maximum_infrastructure_spend_usd="45.01"),
            "45 USD",
        ),
        (
            lambda raw: raw["budget"].update(maximum_judge_spend_usd="4.99"),
            "5 USD",
        ),
        (
            lambda raw: raw["budget"].update(
                included_infrastructure_costs=["gpu_compute", "storage"]
            ),
            "GPU, storage, and network",
        ),
        (
            lambda raw: raw["budget"].update(
                production_judge_budget_config_sha256=_sha("f")
            ),
            "judge budget identity",
        ),
        (
            lambda raw: raw["budget"].update(identity_sha256=_sha("e")),
            "budget identity",
        ),
    ],
)
def test_fleet_budget_fails_closed_on_all_in_split_scope_or_identity(
    mutation, message: str
) -> None:
    raw = _raw()
    mutation(raw)
    with pytest.raises(FleetError, match=message):
        FleetConfig.from_mapping(raw)


def test_batch_runs_on_bounded_independent_workers_and_returns_journal_order(
    tmp_path: Path,
) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0
    slots: list[int] = []

    class Worker:
        def __init__(self, slot: int) -> None:
            self.slot = slot

        def evaluate(self, dispatch: dict[str, Any]) -> EvaluationResult:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                slots.append(self.slot)
            time.sleep(0.01)
            with lock:
                active -= 1
            ordinal = dispatch["ordinal"]
            return EvaluationResult.successful(
                {"deception": float(ordinal), "retained_truth": 1.0}
            )

        def close(self) -> None:
            pass

    config = FleetConfig.from_mapping({**_raw(), "receipt_directory": str(tmp_path)})
    fleet = FleetBatchEvaluator(config, worker_factory=Worker)
    requests = tuple(_request(index) for index in range(8))
    results = fleet.evaluate_batch(requests)

    assert [item.metrics["deception"] for item in results] == list(map(float, range(8)))
    assert peak == 8
    assert set(slots) == set(range(8))
    assert len(list(tmp_path.glob("trial-*.json"))) == 8


def test_receipt_directory_override_relocates_only_mutable_storage(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class Worker:
        def __init__(self, slot: int) -> None:
            self.slot = slot

        def evaluate(self, dispatch: dict[str, Any]) -> EvaluationResult:
            calls.append(dispatch["trial_id"])
            return EvaluationResult.successful(
                {"deception": 0.5, "retained_truth": 0.75}
            )

        def close(self) -> None:
            pass

    raw = _adaptive_raw()
    frozen_receipts = tmp_path / "frozen-config-location"
    raw["receipt_directory"] = str(frozen_receipts)
    config = FleetConfig.from_mapping(raw)
    frozen_identity = config.identity_sha256
    runtime_receipts = (tmp_path / "outputs" / "fleet-receipts").resolve()
    evaluator = FleetBatchEvaluator(
        config,
        worker_factory=Worker,
        receipt_directory_override=runtime_receipts,
    )
    evaluator.evaluate_batch((_request(0),))

    assert evaluator.receipt_directory == runtime_receipts
    assert (runtime_receipts / "trial-0000.json").is_file()
    assert not (frozen_receipts / "trial-0000.json").exists()
    assert evaluator.identity["fleet_config_sha256"] == frozen_identity
    assert config.identity_sha256 == frozen_identity

    FleetBatchEvaluator(
        config,
        worker_factory=Worker,
        receipt_directory_override=runtime_receipts,
    ).evaluate_batch((_request(0),))
    assert calls == ["trial-0000"]

    with pytest.raises(FleetError, match="absolute"):
        FleetBatchEvaluator(
            config,
            worker_factory=Worker,
            receipt_directory_override=Path("relative-runtime-receipts"),
        )


def test_resume_reuses_exact_receipts_without_duplicate_trial_execution(tmp_path: Path) -> None:
    calls: list[str] = []

    class Worker:
        def __init__(self, slot: int) -> None:
            self.slot = slot

        def evaluate(self, dispatch: dict[str, Any]) -> EvaluationResult:
            calls.append(dispatch["trial_id"])
            return EvaluationResult.successful(
                {"deception": 0.5, "retained_truth": 0.75}
            )

        def close(self) -> None:
            pass

    config = FleetConfig.from_mapping({**_raw(), "receipt_directory": str(tmp_path)})
    FleetBatchEvaluator(config, worker_factory=Worker).evaluate_batch((_request(0), _request(1)))
    FleetBatchEvaluator(config, worker_factory=Worker).evaluate_batch((_request(0), _request(1)))
    assert calls == ["trial-0000", "trial-0001"]

    receipt = tmp_path / "trial-0000.json"
    raw = json.loads(receipt.read_text())
    raw["request_sha256"] = _sha("f")
    receipt.write_text(json.dumps(raw))
    with pytest.raises(FleetError, match="receipt identity"):
        FleetBatchEvaluator(config, worker_factory=Worker).evaluate_batch((_request(0),))


def test_resume_replays_signed_worker_telemetry_without_repeating_evaluation(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    replayed: list[tuple[int, str, dict[str, float]]] = []

    class Worker:
        last_telemetry = {
            "evaluation_seconds": 12.5,
            "generated_tokens": 250,
            "generated_tokens_per_second": 20,
            "judge_calls": 2,
            "judge_latency_seconds": 1.5,
            "judge_cost_usd": 0.01,
            "prompt": 12345,  # Numeric secrets are not a permitted telemetry field.
        }

        def __init__(self, slot: int) -> None:
            self.slot = slot

        def evaluate(self, dispatch: dict[str, Any]) -> EvaluationResult:
            calls.append(dispatch["trial_id"])
            return EvaluationResult.successful(
                {"deception": 0.5, "retained_truth": 0.75}
            )

        def close(self) -> None:
            pass

    def crash_after_receipt(_slot: int, _trial_id: str, _telemetry: dict[str, float]) -> None:
        raise RuntimeError("simulated crash before batch observation")

    raw = {**_adaptive_raw(), "receipt_directory": str(tmp_path)}
    config = FleetConfig.from_mapping(raw)
    with pytest.raises(RuntimeError, match="simulated crash"):
        FleetBatchEvaluator(
            config,
            worker_factory=Worker,
            trial_telemetry_callback=crash_after_receipt,
        ).evaluate_batch((_request(0),))

    receipt = json.loads((tmp_path / "trial-0000.json").read_text())
    assert receipt["format"] == "truth_editing_vast_fleet_trial_receipt_v2"
    assert "prompt" not in receipt["telemetry"]

    result = FleetBatchEvaluator(
        config,
        worker_factory=Worker,
        trial_telemetry_callback=lambda slot, trial_id, telemetry: replayed.append(
            (slot, trial_id, dict(telemetry))
        ),
    ).evaluate_batch((_request(0),))

    assert calls == ["trial-0000"]
    assert result[0].metrics["deception"] == 0.5
    assert replayed == [
        (
            0,
            "trial-0000",
            {
                "evaluation_seconds": 12.5,
                "generated_tokens": 250.0,
                "generated_tokens_per_second": 20.0,
                "judge_calls": 2.0,
                "judge_latency_seconds": 1.5,
                "judge_cost_usd": 0.01,
            },
        )
    ]


def test_partial_batch_durable_receipt_events_replay_without_worker_rerun(
    tmp_path: Path,
) -> None:
    worker_calls: list[str] = []
    first_events: list[dict[str, Any]] = []
    replay_events: list[dict[str, Any]] = []

    class Worker:
        def __init__(self, slot: int) -> None:
            self.slot = slot

        def evaluate(self, dispatch: dict[str, Any]) -> EvaluationResult:
            worker_calls.append(dispatch["trial_id"])
            return EvaluationResult.successful(
                {"deception": 0.5, "retained_truth": 0.75}
            )

        def close(self) -> None:
            pass

    def crash_after_seventh(event: dict[str, Any]) -> None:
        first_events.append(dict(event))
        if event["ordinal"] == 6:
            raise RuntimeError("simulated crash after seven durable receipts")

    config = FleetConfig.from_mapping(
        {**_adaptive_raw(), "receipt_directory": str(tmp_path)}
    )
    requests = tuple(_request(index) for index in range(7))
    with pytest.raises(RuntimeError, match="seven durable receipts"):
        FleetBatchEvaluator(
            config,
            worker_factory=Worker,
            trial_receipt_durable_callback=crash_after_seventh,
        ).evaluate_batch(requests)

    assert sorted(worker_calls) == [f"trial-{index:04d}" for index in range(7)]
    assert len(list(tmp_path.glob("trial-*.json"))) == 7
    for event in first_events:
        assert set(event) == {
            "format",
            "fleet_config_sha256",
            "trial_id",
            "ordinal",
            "request_sha256",
            "receipt_path",
            "receipt_sha256",
        }
        assert event["format"] == "truth_editing_vast_fleet_receipt_durable_event_v1"
        assert "result" not in event and "metrics" not in event

    results = FleetBatchEvaluator(
        config,
        worker_factory=Worker,
        trial_receipt_durable_callback=lambda event: replay_events.append(dict(event)),
    ).evaluate_batch(requests)

    assert len(results) == 7
    assert sorted(worker_calls) == [f"trial-{index:04d}" for index in range(7)]
    assert [event["ordinal"] for event in replay_events] == list(range(7))
    for event in replay_events:
        receipt = json.loads(Path(event["receipt_path"]).read_text())
        assert receipt["receipt_sha256"] == event["receipt_sha256"]


def test_legacy_v1_trial_receipt_remains_readable_without_fabricated_telemetry(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    telemetry: list[dict[str, float]] = []

    class Worker:
        def __init__(self, slot: int) -> None:
            self.slot = slot

        def evaluate(self, dispatch: dict[str, Any]) -> EvaluationResult:
            calls.append(dispatch["trial_id"])
            return EvaluationResult.successful(
                {"deception": 0.5, "retained_truth": 0.75}
            )

        def close(self) -> None:
            pass

    config = FleetConfig.from_mapping({**_raw(), "receipt_directory": str(tmp_path)})
    FleetBatchEvaluator(config, worker_factory=Worker).evaluate_batch((_request(0),))
    path = tmp_path / "trial-0000.json"
    receipt = json.loads(path.read_text())
    receipt["format"] = "truth_editing_vast_fleet_trial_receipt_v1"
    receipt.pop("telemetry")
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(receipt))

    FleetBatchEvaluator(
        config,
        worker_factory=Worker,
        trial_telemetry_callback=lambda _slot, _trial, row: telemetry.append(dict(row)),
    ).evaluate_batch((_request(0),))

    assert calls == ["trial-0000"]
    assert telemetry == []


def test_stop_safe_cleanup_closes_every_created_worker(tmp_path: Path) -> None:
    closed: list[int] = []

    class Worker:
        def __init__(self, slot: int) -> None:
            self.slot = slot

        def evaluate(self, dispatch: dict[str, Any]) -> EvaluationResult:
            raise KeyboardInterrupt

        def close(self) -> None:
            closed.append(self.slot)

    config = FleetConfig.from_mapping({**_raw(), "receipt_directory": str(tmp_path)})
    fleet = FleetBatchEvaluator(config, worker_factory=Worker)
    with pytest.raises(KeyboardInterrupt):
        with fleet:
            fleet.evaluate_batch((_request(0), _request(1), _request(2), _request(3)))
    assert sorted(closed) == [0, 1, 2, 3]
    stop = json.loads((tmp_path.parent / "fleet-stop-receipt.json").read_text())
    assert stop["all_workers_closed"] is True


def test_vast_worker_binds_dispatch_to_v3_bundle_and_fetched_result(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    production_bytes = b'{"format":"truth-editing-v3-test"}\n'
    production_relative = _raw()["production_config"]["path"]
    (repo / production_relative).write_bytes(production_bytes)
    bundle = tmp_path / "bundle.tgz"
    bundle.write_bytes(b"exact-bundle")
    raw = _raw()
    raw["production_config"]["sha256"] = __import__("hashlib").sha256(production_bytes).hexdigest()
    raw["bundle_sha256"] = __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    raw["receipt_directory"] = str(tmp_path / "receipts")
    fleet = FleetConfig.from_mapping(raw)
    job_raw = _production_job_raw()
    job_raw["base_job"]["expected_outputs"] = [
        "trial-result.json", "checkpoints/latest.json"
    ]
    job_raw["base_job"]["resources"]["maximum_cost_usd"] = 0.24
    job_raw["base_job"]["resources"]["maximum_elapsed_seconds"] = 5400
    old_production_path = job_raw["study"]["production_config_path"]
    job_raw["study"]["production_config_path"] = fleet.production_config_path
    job_raw["study"]["production_config_sha256"] = fleet.production_config_sha256
    job_raw["base_job"]["bundle_paths"] = [
        fleet.production_config_path if path == old_production_path else path
        for path in job_raw["base_job"]["bundle_paths"]
    ]
    job_raw["study"]["workload_command"] = [
        "python", "scripts/run_truth_editing_vast_fleet_worker.py",
        "--config", fleet.production_config_path,
        "--phase", "discovery",
    ]
    job = ProductionVastConfig.from_mapping(job_raw)
    offer = Offer.from_mapping({
        "id": 123, "gpu_name": "RTX 4090", "num_gpus": 1,
        "gpu_ram": 24576, "dph_total": 0.1,
        "inet_down_cost": 0.0, "inet_up_cost": 0.0,
    })
    seen: list[tuple[dict[str, Any], ProductionVastConfig]] = []

    def execute(*, plan, config, metadata_path, workload_secret):
        assert workload_secret.environment_name == "OPENROUTER_API_KEY"
        seen.append((plan, config))
        fetch = Path(plan["base_lifecycle"]["fetch_dir"])
        fetch.mkdir(parents=True)
        dispatch = json.loads(__import__("base64").urlsafe_b64decode(
            config.workload_command[config.workload_command.index("--fleet-request-base64") + 1]
        ))
        result = {
            "format": "truth_editing_vast_fleet_worker_result_v1",
            "request_sha256": dispatch["request_sha256"],
            "result": {
                "outcome_kind": "successful",
                "metrics": {"deception": 0.9, "retained_truth": 0.8},
                "detail": None,
            },
        }
        result["self_sha256"] = __import__("hashlib").sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (fetch / "trial-result.json").write_text(json.dumps(result))
        return {"destroyed": True, "self_sha256": _sha("d")}

    worker = VastLifecycleTrialWorker(
        slot=0, fleet_config=fleet, production_job=job, offer=offer,
        repo=repo, bundle=bundle, fetch_root=tmp_path / "fetch",
        metadata_root=tmp_path / "metadata",
        workload_secret=EphemeralWorkloadSecret.openrouter("test-secret"),
        lifecycle_execute=execute,
    )
    dispatch = {
        "format": "truth_editing_vast_fleet_dispatch_v1",
        "trial_id": "trial-0080", "ordinal": 80,
        "request_sha256": _sha("c"),
    }
    result = worker.evaluate(dispatch)
    assert result.metrics == {"deception": 0.9, "retained_truth": 0.8}
    assert seen[0][1].phase == "expanded"
    assert "--phase" in seen[0][1].workload_command
    assert seen[0][0]["production_config_path"] == fleet.production_config_path


def test_persistent_cuda_worker_is_gpu_isolated_reused_and_circuit_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "worker-needs-this")
    monkeypatch.setenv("WANDB_API_KEY", "coordinator-only-secret")
    monkeypatch.setenv("WANDB_RUN_ID", "coordinator-only-run")
    monkeypatch.setenv("WANDB_PROJECT", "coordinator-only-project")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "coordinator-only-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "coordinator-only-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "coordinator-only-token")
    class Input:
        def __init__(self) -> None:
            self.lines: list[str] = []
        def write(self, value: str) -> None:
            self.lines.append(value)
        def flush(self) -> None:
            pass

    class Output:
        def __init__(self) -> None:
            self.lines = [
                json.dumps({
                    "request_sha256": _sha("c"),
                    "result": {"outcome_kind": "successful", "metrics": {
                        "deception": 0.9, "retained_truth": 0.8,
                    }, "detail": None},
                }) + "\n",
                json.dumps({"fatal": True, "failure_receipt_sha256": _sha("d")}) + "\n",
            ]
        def readline(self) -> str:
            return self.lines.pop(0)

    class Process:
        def __init__(self) -> None:
            self.stdin = Input()
            self.stdout = Output()
            self.stderr = None
            self.stopped = False
        def poll(self):
            return 0 if self.stopped else None
        def wait(self, timeout):
            self.stopped = True
            return 0

    captured: dict[str, Any] = {}
    process = Process()
    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return process

    worker = SubprocessCudaWorker(3, ("python", "worker.py"), popen_factory=popen)
    dispatch = {"request_sha256": _sha("c"), "trial_id": "trial-0000"}
    assert worker.evaluate(dispatch).metrics["deception"] == 0.9
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert captured["env"]["OPENROUTER_API_KEY"] == "worker-needs-this"
    assert not any(name.startswith("WANDB_") for name in captured["env"])
    assert not any(name.startswith("AWS_") for name in captured["env"])
    with pytest.raises(FleetCircuitOpen, match="circuit"):
        worker.evaluate(dispatch)
    worker.close()
    assert process.stdin.lines[-1] == '{"command":"stop"}\n'


def test_exact_eight_gpu_offer_is_available_only_through_persistent_host_parser() -> None:
    raw = {
        "id": 8123, "gpu_name": "RTX 3090", "num_gpus": 8,
        "gpu_ram": 24576, "dph_total": 1.2,
        "inet_down_cost": 0.0, "inet_up_cost": 0.0,
    }
    with pytest.raises(Exception, match="exactly 1"):
        Offer.from_mapping(raw)
    assert Offer.from_multi_gpu_mapping(raw).gpu_count == 8
    assert issubclass(FleetCircuitOpen, PaidJudgeCircuitOpen)


def test_persistent_host_plan_hydrates_once_then_makes_shared_cache_read_only(
    tmp_path: Path,
) -> None:
    raw = _production_job_raw()
    raw["phase"] = "finalist"
    production_path = raw["study"]["production_config_path"]
    raw["study"]["workload_command"] = [
        "python", "scripts/run_truth_editing_cuda_fleet_controller.py",
        "--fleet-config", "configs/truth_editing_vast_fleet_v3.json",
        "--config", production_path,
        "--phase", "finalist", "--receipt", "/workspace/outputs/study-receipt.json",
    ]
    config = ProductionVastConfig.from_mapping(raw)
    offer = Offer.from_multi_gpu_mapping({
        "id": 8123, "gpu_name": "RTX 3090", "num_gpus": 8,
        "gpu_ram": 24576, "dph_total": 0.5,
        "inet_down_cost": 0.0, "inet_up_cost": 0.0,
    })
    plan = production_lifecycle_plan(
        vastai="vastai", config=config, offer=offer,
        bundle=tmp_path / "bundle.tgz", fetch_dir=tmp_path / "fetch",
    )
    remote = plan["base_lifecycle"]["remote_command"]
    assert remote.count("hydrate.sh") == 1
    assert "chmod -R a-w /workspace/model-cache" in remote
    assert plan["base_lifecycle"]["offer"]["gpu_count"] == 8
