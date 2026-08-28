from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from intelligent_liars.truth_editing_study import (
    OfflineSyntheticEvaluator,
    OptunaSearchDriver,
    TruthEditingStudy,
)
from intelligent_liars.truth_editing_wandb_monitoring import (
    CoordinatorMonitor,
    MonitoredSearchDriver,
)
from test_truth_editing_study import _config, _direction_bank


class _SecretNamedException(RuntimeError):
    pass


_SecretNamedException.__module__ = "s3://private-bucket/sk-or-v1-private"


class _RecordingRun:
    def __init__(self, *, fail_log: bool = False, fail_finish: bool = False) -> None:
        self.fail_log = fail_log
        self.fail_finish = fail_finish
        self.logged: list[tuple[dict[str, Any], int | None]] = []
        self.finished = 0

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        if self.fail_log:
            raise _SecretNamedException(
                "W&B failed with sk-or-v1-private and /private/path"
            )
        self.logged.append((dict(values), step))

    def finish(self, *, exit_code: int = 0) -> None:
        if self.fail_finish:
            raise _SecretNamedException(
                "W&B failed with sk-or-v1-private and /private/path"
            )
        self.finished += 1


class _RecordingWandb:
    def __init__(
        self,
        *,
        fail_init: bool = False,
        fail_log: bool = False,
        fail_finish: bool = False,
    ) -> None:
        self.fail_init = fail_init
        self.run = _RecordingRun(fail_log=fail_log, fail_finish=fail_finish)
        self.init_calls: list[dict[str, Any]] = []
        self.settings_calls: list[dict[str, Any]] = []
        self.forbidden_calls: list[str] = []

    def Settings(self, **kwargs: Any) -> SimpleNamespace:
        self.settings_calls.append(dict(kwargs))
        return SimpleNamespace(**kwargs)

    def init(self, **kwargs: Any) -> _RecordingRun:
        self.init_calls.append(dict(kwargs))
        if self.fail_init:
            raise _SecretNamedException(
                "W&B failed with sk-or-v1-private and /private/path"
            )
        return self.run

    def Artifact(self, *_args: Any, **_kwargs: Any) -> None:
        self.forbidden_calls.append("Artifact")

    def log_artifact(self, *_args: Any, **_kwargs: Any) -> None:
        self.forbidden_calls.append("log_artifact")

    def save(self, *_args: Any, **_kwargs: Any) -> None:
        self.forbidden_calls.append("save")

    def watch(self, *_args: Any, **_kwargs: Any) -> None:
        self.forbidden_calls.append("watch")


def _monitor(root: Path, wandb: _RecordingWandb) -> CoordinatorMonitor:
    return CoordinatorMonitor.open(
        checkpoint_path=root / "monitoring/wandb-run.json",
        run_id="one-coordinator-run",
        project="intelligent-liars",
        entity="truth-editing",
        run_name="ignored-by-frozen-contract",
        receipt_path=root / "monitoring/events.jsonl",
        total_trials=4,
        batch_size=2,
        wandb_module=wandb,
        monotonic=lambda: 100.0,
    )


def _canonical_optuna_journal(path: Path) -> bytes:
    """Remove only Optuna's wall-clock/worker bookkeeping, never trial state."""

    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for field in ("worker_id", "datetime_start", "datetime_complete"):
            row.pop(field, None)
        rows.append(row)
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.skipif(
    pytest.importorskip("optuna", reason="Optuna is required for journal determinism")
    is None,
    reason="Optuna is required",
)
def test_wandb_health_cannot_change_study_optuna_or_run_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    config = _config(
        seed,
        max_trials=4,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 4}],
    )
    study = TruthEditingStudy(config, _direction_bank())
    seed_journal = seed / "study/study-journal.json"
    seed_journal.parent.mkdir()
    study.run(
        driver=OptunaSearchDriver(seed=config.sampler_seed),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=seed_journal,
        stop_after_trials=2,
    )

    seed_optuna = seed_journal.with_name(seed_journal.name + ".optuna.log").read_bytes()
    final_files: list[tuple[bytes, bytes, bytes]] = []
    for mode in ("absent", "healthy", "failing"):
        root = tmp_path / mode
        shutil.copytree(seed, root)
        journal = root / "study/study-journal.json"
        driver: Any = OptunaSearchDriver(seed=config.sampler_seed)
        monitor: CoordinatorMonitor | None = None
        if mode != "absent":
            dashboard = _RecordingWandb(fail_log=mode == "failing")
            monitor = _monitor(root, dashboard)
            driver = MonitoredSearchDriver(driver, monitor)
        else:
            # The coordinator still owns the same durable identity even when
            # W&B is deliberately disabled or unavailable.
            dashboard = _RecordingWandb(fail_init=True)
            monitor = _monitor(root, dashboard)
            monitor.close()
            monitor = None
        study.run(
            driver=driver,
            evaluator=OfflineSyntheticEvaluator(),
            journal_path=journal,
        )
        if monitor is not None:
            monitor.close()
        optuna_path = journal.with_name(journal.name + ".optuna.log")
        assert optuna_path.read_bytes().startswith(seed_optuna)
        final_files.append(
            (
                journal.read_bytes(),
                _canonical_optuna_journal(optuna_path),
                (root / "monitoring/wandb-run.json").read_bytes(),
            )
        )

    assert final_files[0] == final_files[1] == final_files[2]


def test_resume_reconciles_commit_that_crashed_before_wandb_log(tmp_path: Path) -> None:
    class CrashAfterLocalObserve(OptunaSearchDriver):
        def __init__(self, *, seed: int) -> None:
            super().__init__(seed=seed)
            self._crashed = False

        def observe(self, trials):
            super().observe(trials)
            if not self._crashed:
                self._crashed = True
                raise KeyboardInterrupt("crash after local commit, before W&B log")

    config = _config(
        tmp_path,
        max_trials=4,
        evaluation_tiers=[{"name": "all", "record_limit": 2, "through_trial": 4}],
    )
    journal = tmp_path / "study/study-journal.json"
    journal.parent.mkdir()
    first_dashboard = _RecordingWandb()
    first_monitor = _monitor(tmp_path, first_dashboard)
    with pytest.raises(KeyboardInterrupt, match="before W&B log"):
        TruthEditingStudy(config, _direction_bank()).run(
            driver=MonitoredSearchDriver(
                CrashAfterLocalObserve(seed=config.sampler_seed), first_monitor
            ),
            evaluator=OfflineSyntheticEvaluator(),
            journal_path=journal,
        )
    committed = json.loads(journal.read_text())
    assert all(item["result"] is not None for item in committed["batches"][0]["trials"])
    assert first_dashboard.run.logged == []
    first_monitor.close()

    resumed_dashboard = _RecordingWandb()
    resumed_monitor = _monitor(tmp_path, resumed_dashboard)
    report = TruthEditingStudy(config, _direction_bank()).run(
        driver=MonitoredSearchDriver(
            OptunaSearchDriver(seed=config.sampler_seed), resumed_monitor
        ),
        evaluator=OfflineSyntheticEvaluator(),
        journal_path=journal,
    )
    resumed_monitor.close()

    assert report.completed_trials == 4
    logged_ordinals = [
        values["trial/ordinal"]
        for values, _step in resumed_dashboard.run.logged
        if "trial/ordinal" in values
    ]
    assert logged_ordinals == [0, 1, 2, 3]
    assert len(resumed_dashboard.init_calls) == 1
    assert resumed_dashboard.init_calls[0]["id"] == "one-coordinator-run"


def test_every_monitor_entrypoint_drops_prompts_secrets_paths_urls_and_tensor_values(
    tmp_path: Path,
) -> None:
    secret = "sk-or-v1-THIS_MUST_NEVER_LEAVE"
    prompt = "Tell me the private system prompt"
    private_path = "/Users/private/.ssh/id_ed25519"
    private_url = "s3://private-bucket/secret-object"
    tensor = {"shape": [4096, 4096], "values": [secret, prompt]}
    dashboard = _RecordingWandb()
    monitor = _monitor(tmp_path, dashboard)

    request = SimpleNamespace(
        ordinal=0,
        proposal={
            "direction_family": secret,
            "writer_region": private_path,
            "strength": tensor,
            "direction_ids": [prompt, private_url],
            "prompt": prompt,
            "api_key": secret,
            "local_path": private_path,
            "url": private_url,
            "weights": tensor,
        },
    )
    result = SimpleNamespace(
        outcome_kind=secret,
        metrics={
            "valid_false_report_rate_lcb": tensor,
            "truth_report_dissociation_lcb": prompt,
            "capability_preservation_lcb": private_url,
        },
    )
    monitor.record_batch(0, (request,), (result,))
    monitor.record_gpu(
        SimpleNamespace(
            gpu_slot=private_path,
            utilization_percent=secret,
            memory_used_mib=tensor,
            memory_total_mib=private_url,
            tokens_per_second=prompt,
            active_trial_id=private_url,
        )
    )
    monitor.record_judge(
        calls=secret, failures=prompt, latency_ms=private_path, cost_usd=tensor
    )
    monitor.record_cost(gpu_actual_usd=private_url, gpu_projected_usd=tensor)
    monitor.record_operational(
        retries=secret,
        stopped_trials=prompt,
        errors=private_path,
        error_category=secret,
        error_fingerprint="a" * 64,
    )
    monitor.close()

    external_and_local = json.dumps(
        {
            "logged": dashboard.run.logged,
            "snapshot": monitor.verification_snapshot(),
            "receipts": (tmp_path / "monitoring/events.jsonl").read_text(),
        },
        sort_keys=True,
    )
    for forbidden in (secret, prompt, private_path, private_url, "4096"):
        assert forbidden not in external_and_local
    assert dashboard.forbidden_calls == []
    assert dashboard.settings_calls == [
        {
            "console": "off",
            "disable_git": True,
            "disable_job_creation": True,
            "x_disable_stats": True,
            "x_disable_meta": True,
            "x_disable_machine_info": True,
        }
    ]
    init = dashboard.init_calls[0]
    assert init["save_code"] is False
    assert "code" not in init["config"]
    assert "git" not in init["config"]
    assert "artifact" not in init["config"]


@pytest.mark.parametrize("failure", ["init", "log", "finish"])
def test_wandb_exception_details_never_enter_local_receipts(
    tmp_path: Path, failure: str
) -> None:
    dashboard = _RecordingWandb(
        fail_init=failure == "init",
        fail_log=failure == "log",
        fail_finish=failure == "finish",
    )
    monitor = _monitor(tmp_path, dashboard)
    monitor.record_judge(calls=1, failures=0, latency_ms=2.0, cost_usd=0.001)
    monitor.close()

    persisted = (tmp_path / "monitoring/events.jsonl").read_text()
    assert "sk-or-v1-private" not in persisted
    assert "/private/path" not in persisted
    assert "private-bucket" not in persisted
    rows = [json.loads(line) for line in persisted.splitlines()]
    failures = [row for row in rows if row["kind"] == "wandb_failure"]
    assert failures
    assert all(len(row["payload"]["error_fingerprint"]) == 64 for row in failures)
