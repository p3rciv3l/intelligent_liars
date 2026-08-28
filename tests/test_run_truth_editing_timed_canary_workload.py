from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_truth_editing_timed_canary_workload.py"
SPEC = importlib.util.spec_from_file_location(
    "run_truth_editing_timed_canary_workload", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Proposal:
    def to_dict(self):
        return {"proposal_origin": "coverage_anchor", "strength": 1.0}


class _Driver:
    def __init__(self, *, seed):
        self.seed = seed

    def suggest(self, request):
        assert request.ordinal == 0
        return _Proposal()


class _Monitor:
    opened = []

    @classmethod
    def open(cls, **kwargs):
        item = cls()
        item.kwargs = kwargs
        item.closed = False
        item.recorded = []
        cls.opened.append(item)
        return item

    def record_batch(self, *args):
        self.recorded.append(("batch", args))

    def record_worker_telemetry(self, *args):
        self.recorded.append(("worker", args))

    def record_judge(self, **kwargs):
        self.recorded.append(("judge", kwargs))

    def record_cost(self, **kwargs):
        self.recorded.append(("cost", kwargs))

    def close(self):
        self.closed = True

    def verification_snapshot(self):
        return {
            "initialized_coordinator_count": 1,
            "finish_calls": 1,
            "nonfatal_error_count": 0,
        }


class _Budget:
    def __init__(self, path, *, config):
        self.path = path

    def monitoring_snapshot(self):
        return {
            "calls": 2,
            "failures": 0,
            "latency_ms": 80.0,
            "elapsed_ms": 160.0,
            "cost_usd": 0.01,
        }

    def receipt(self):
        return {"circuit_open": False}


def _setup(
    tmp_path: Path,
    monkeypatch,
    *,
    evidence=None,
    outcome_kind: str = "successful",
    outcome_detail: str | None = None,
):
    repo = tmp_path / "repo"
    source = repo / "configs/production_release_deadbeef.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "format": "truth_editing_production_config_v1",
                **{name: f"../old/{name}" for name in MODULE._OUTPUT_FIELDS},
            }
        )
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    production = SimpleNamespace(
        judge_budget=object(),
        judge_budget_ledger_dir=tmp_path / "judge-ledger",
        study_config=tmp_path / "study.json",
        direction_manifest=tmp_path / "bank.json",
        direction_root=tmp_path,
        verified_model_sha256="b" * 64,
    )
    study = SimpleNamespace(
        sampler_seed=20260827,
        evaluation_tiers=(SimpleNamespace(name="discovery", record_limit=2),),
        validation_record_ids=("r1", "r2"),
        objective_names=("a", "b", "c"),
    )
    monkeypatch.setattr(MODULE.ProductionRunConfig, "open", lambda path: production)
    monkeypatch.setattr(MODULE, "load_truth_editing_study_config", lambda path: study)
    monkeypatch.setattr(
        MODULE.DirectionBank,
        "open",
        lambda *args, **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(directions=(SimpleNamespace(direction_id="d"),))
        ),
    )
    monkeypatch.setattr(MODULE, "OfflineDeterministicSearchDriver", _Driver)
    monkeypatch.setattr(MODULE, "ProductionJudgeBudget", _Budget)
    canary_evidence = evidence or {
        "tier": "discovery",
        "preservation_kl": {
            "text": 0.00001,
            "vision": 0.00002,
            "recorded_computer_use": 0.00003,
        },
        "judge_cache_receipt_count": 2,
    }

    class Run:
        def evaluate_timed_canary(self, request):
            assert request.trial_id == "trial-0000"
            assert request.record_ids == ("r1", "r2")
            return {
                "result": {
                    "outcome_kind": outcome_kind,
                    "metrics": {"a": 0.8, "b": 0.7, "c": 0.9},
                    "detail": outcome_detail,
                },
                "runtime_telemetry": {
                    "generated_tokens": 120,
                    "evaluation_seconds": 4.0,
                    "generated_tokens_per_second": 30.0,
                },
                "evaluator_evidence": canary_evidence,
            }

    return repo, source, digest, lambda path: Run()


def test_workload_reports_operational_trial_detail_before_missing_evidence(
    tmp_path, monkeypatch
) -> None:
    repo, source, digest, opener = _setup(
        tmp_path,
        monkeypatch,
        evidence={},
        outcome_kind="operational_failure",
        outcome_detail="RuntimeError: exact backend failure",
    )

    with pytest.raises(
        RuntimeError,
        match="timed-canary trial failed operationally: RuntimeError: exact backend failure",
    ):
        MODULE.run_workload(
            source_config=source,
            expected_source_sha256=digest,
            output_root=tmp_path / "outputs",
            observation_path=tmp_path / "observation.json",
            gpu_hourly_usd=0.4,
            wandb_project="intelligent-liars",
            wandb_entity=None,
            environ={"OPENROUTER_API_KEY": "secret", "WANDB_API_KEY": "secret"},
            opener=opener,
            monitor_type=_Monitor,
            monotonic=iter((10.0, 20.0)).__next__,
            repository_root=repo,
        )


def test_workload_emits_exact_real_trial_observation_and_wandb_receipt(
    tmp_path, monkeypatch
) -> None:
    repo, source, digest, opener = _setup(tmp_path, monkeypatch)
    observation_path = tmp_path / "observation.json"
    _Monitor.opened.clear()
    result = MODULE.run_workload(
        source_config=source,
        expected_source_sha256=digest,
        output_root=tmp_path / "outputs",
        observation_path=observation_path,
        gpu_hourly_usd=0.4,
        wandb_project="intelligent-liars",
        wandb_entity="entity",
        environ={"OPENROUTER_API_KEY": "secret", "WANDB_API_KEY": "secret"},
        opener=opener,
        monitor_type=_Monitor,
        monotonic=iter((10.0, 20.0)).__next__,
        repository_root=repo,
    )
    assert result["production_config_path"] == "configs/production_release_deadbeef.json"
    assert result["generated_tokens"] == 120
    assert result["generation_seconds"] == 4.0
    assert set(result["persistence_kl"]) == {
        "text", "vision", "recorded_computer_use"
    }
    assert result["judge"]["attempted_calls"] == 2
    assert result["judge"]["elapsed_seconds"] == pytest.approx(0.16)
    assert json.loads(observation_path.read_text()) == result
    monitor = _Monitor.opened[0]
    assert monitor.kwargs["total_trials"] == monitor.kwargs["batch_size"] == 1
    assert monitor.closed is True
    cost = next(value for kind, value in monitor.recorded if kind == "cost")
    assert cost["gpu_actual_usd"] == pytest.approx(0.4 * 10 / 3600)


def test_workload_rejects_missing_total_judge_elapsed_evidence(
    tmp_path, monkeypatch
) -> None:
    repo, source, digest, opener = _setup(tmp_path, monkeypatch)

    class MissingElapsedBudget(_Budget):
        def monitoring_snapshot(self):
            snapshot = super().monitoring_snapshot()
            snapshot.pop("elapsed_ms")
            return snapshot

    monkeypatch.setattr(MODULE, "ProductionJudgeBudget", MissingElapsedBudget)
    observation = tmp_path / "observation.json"
    with pytest.raises(RuntimeError, match="judge elapsed evidence"):
        MODULE.run_workload(
            source_config=source,
            expected_source_sha256=digest,
            output_root=tmp_path / "outputs",
            observation_path=observation,
            gpu_hourly_usd=0.4,
            wandb_project="intelligent-liars",
            wandb_entity=None,
            environ={"OPENROUTER_API_KEY": "secret", "WANDB_API_KEY": "secret"},
            opener=opener,
            monitor_type=_Monitor,
            monotonic=iter((10.0, 20.0)).__next__,
            repository_root=repo,
        )
    assert not observation.exists()


def test_workload_fails_closed_when_evaluator_omits_one_preservation_lane(
    tmp_path, monkeypatch
) -> None:
    evidence = {
        "tier": "discovery",
        "preservation_kl": {"text": 0.0, "vision": 0.0},
        "judge_cache_receipt_count": 2,
    }
    repo, source, digest, opener = _setup(
        tmp_path, monkeypatch, evidence=evidence
    )
    observation = tmp_path / "observation.json"
    with pytest.raises(RuntimeError, match="evidence is incomplete"):
        MODULE.run_workload(
            source_config=source,
            expected_source_sha256=digest,
            output_root=tmp_path / "outputs",
            observation_path=observation,
            gpu_hourly_usd=0.4,
            wandb_project="intelligent-liars",
            wandb_entity=None,
            environ={"OPENROUTER_API_KEY": "secret", "WANDB_API_KEY": "secret"},
            opener=opener,
            monitor_type=_Monitor,
            monotonic=iter((10.0, 20.0)).__next__,
            repository_root=repo,
        )
    assert not observation.exists()


def test_workload_rejects_hash_substitution_before_runtime_output(
    tmp_path, monkeypatch
) -> None:
    repo, source, _digest, opener = _setup(tmp_path, monkeypatch)
    output = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        MODULE.run_workload(
            source_config=source,
            expected_source_sha256="f" * 64,
            output_root=output,
            observation_path=tmp_path / "observation.json",
            gpu_hourly_usd=0.4,
            wandb_project="intelligent-liars",
            wandb_entity=None,
            environ={"OPENROUTER_API_KEY": "secret", "WANDB_API_KEY": "secret"},
            opener=opener,
            monitor_type=_Monitor,
            repository_root=repo,
        )
    assert not output.exists()
