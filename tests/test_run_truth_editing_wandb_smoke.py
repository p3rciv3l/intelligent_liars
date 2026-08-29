from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).parents[1] / "scripts/run_truth_editing_wandb_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_truth_editing_wandb_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _RemoteRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.rows: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def log(
        self,
        values: dict[str, Any],
        *,
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        del step, commit
        self.rows.append(dict(values))
        self.summary.update(values)

    def finish(self, *, exit_code: int = 0) -> None:
        pass

    def scan_history(self):
        return list(self.rows)


class _Api:
    default_entity = "safe-entity"

    def __init__(self, module: "_Wandb") -> None:
        self.module = module

    def run(self, path: str) -> _RemoteRun:
        run_id = path.rsplit("/", 1)[-1]
        return self.module.runs[run_id]


class _Wandb:
    def __init__(self) -> None:
        self.runs: dict[str, _RemoteRun] = {}
        self.init_calls: list[dict[str, Any]] = []

    @staticmethod
    def Settings(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    def init(self, **kwargs: Any) -> _RemoteRun:
        self.init_calls.append(kwargs)
        run = self.runs.setdefault(kwargs["id"], _RemoteRun(kwargs["id"]))
        return run

    def Api(self) -> _Api:
        return _Api(self)


def test_smoke_runs_two_sessions_one_remote_run_and_publishes_gate(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.delenv("WANDB_SILENT", raising=False)
    monkeypatch.delenv("WANDB_QUIET", raising=False)
    wandb = _Wandb()
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=test-key-never-persist\n")
    output = tmp_path / "smoke"

    assert MODULE.main(
        [
            "--output-dir", str(output),
            "--env-file", str(env_file),
            "--project", "intelligent-liars-tests",
            "--entity", "safe-entity",
            "--run-id", "wandb-smoke-test-001",
        ],
        wandb_module=wandb,
    ) == 0

    assert len(wandb.init_calls) == 2
    assert {call["id"] for call in wandb.init_calls} == {"wandb-smoke-test-001"}
    assert all(call["resume"] == "allow" for call in wandb.init_calls)
    receipt = json.loads((output / "gate-receipt.json").read_text())
    assert receipt["wandb_run_id"] == "wandb-smoke-test-001"
    trace = json.loads((output / "transport-trace.json").read_text())
    assert trace["checkpoint_run_ids"] == [
        "wandb-smoke-test-001",
        "wandb-smoke-test-001",
    ]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["kind"] == "wandb_transport_smoke"
    all_bytes = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    assert b"test-key-never-persist" not in all_bytes
    assert "WANDB_SILENT" not in MODULE.os.environ
    assert "WANDB_QUIET" not in MODULE.os.environ


def test_smoke_is_immutable_and_missing_key_fails_without_remote_init(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    wandb = _Wandb()
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    assert MODULE.main(
        ["--output-dir", str(output), "--env-file", str(tmp_path / "missing.env")],
        wandb_module=wandb,
    ) == 2
    assert not wandb.init_calls
    assert "already exists" in capsys.readouterr().err

    fresh = tmp_path / "fresh"
    assert MODULE.main(
        ["--output-dir", str(fresh), "--env-file", str(tmp_path / "missing.env")],
        wandb_module=wandb,
    ) == 2
    assert not fresh.exists()
    error = capsys.readouterr().err
    assert "WANDB_API_KEY" in error
    assert "test-key" not in error


def test_provider_exception_text_and_key_never_reach_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    secret = "test-key-never-print-provider"
    monkeypatch.setenv("WANDB_API_KEY", secret)

    class BrokenWandb:
        class Api:
            def __init__(self) -> None:
                raise RuntimeError(f"provider leaked {secret}")

    output = tmp_path / "smoke"
    assert MODULE.main(
        ["--output-dir", str(output), "--entity", "safe-entity"],
        wandb_module=BrokenWandb,
    ) == 2
    text = capsys.readouterr().err
    assert secret not in text
    assert "provider leaked" not in text
    assert not output.exists()
