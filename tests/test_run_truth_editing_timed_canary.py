from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from test_truth_editing_timed_canary import _observation, _raw


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_truth_editing_timed_canary.py"
SPEC = importlib.util.spec_from_file_location("run_truth_editing_timed_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inputs(tmp_path: Path):
    repo = tmp_path / "repo"
    production = repo / "configs/truth_editing_production_release_abcd1234.json"
    production.parent.mkdir(parents=True)
    production.write_text('{"format":"production-v3-test"}\n')
    raw = _raw()
    raw["production_config"]["sha256"] = hashlib.sha256(production.read_bytes()).hexdigest()
    config = tmp_path / "canary.json"
    config.write_text(json.dumps(raw))
    command = tmp_path / "command.json"
    command.write_text(json.dumps(["python", "worker.py"]))
    return repo, config, command


def _strict_open_result():
    return SimpleNamespace(
        preservation_threshold_calibration=Path("calibration.json"),
        preservation_threshold_calibration_sha256="a" * 64,
        judge_budget=object(),
    )


def test_cli_dry_run_verifies_identity_without_executing(tmp_path, monkeypatch, capsys) -> None:
    repo, config, command = _inputs(tmp_path)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dry run executed")),
    )
    monkeypatch.setattr(MODULE.ProductionRunConfig, "open", lambda path: _strict_open_result())
    assert MODULE.main([
        "--config", str(config), "--repo", str(repo),
        "--command-json", str(command), "--observation", str(tmp_path / "observation.json"),
        "--receipt", str(tmp_path / "receipt.json"), "--gpu-hourly-usd", "0.36",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry_run"
    assert payload["production_config_verified"] is True


def test_cli_execute_runs_exact_argv_and_passes_observation_to_contract(
    tmp_path, monkeypatch, capsys
) -> None:
    repo, config, command = _inputs(tmp_path)
    observation = tmp_path / "observation.json"
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        Path(kwargs["env"]["TRUTH_EDITING_TIMED_CANARY_OBSERVATION_PATH"]).write_text(
            json.dumps(_observation())
        )
        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(MODULE.ProductionRunConfig, "open", lambda path: _strict_open_result())
    monkeypatch.setattr(
        MODULE,
        "run_timed_canary",
        lambda **kwargs: {"format": "test-receipt", "observation": kwargs["workload"]()},
    )
    assert MODULE.main([
        "--config", str(config), "--repo", str(repo), "--command-json", str(command),
        "--observation", str(observation), "--receipt", str(tmp_path / "receipt.json"),
        "--gpu-hourly-usd", "0.36", "--execute",
    ]) == 0
    assert seen[0][0] == ["python", "worker.py"]
    assert seen[0][1]["cwd"] == repo
    assert seen[0][1]["timeout"] == 900.0
    assert seen[0][1]["env"]["TRUTH_EDITING_GPU_HOURLY_USD"] == "0.36"
    assert json.loads(capsys.readouterr().out)["mode"] == "executed"


def test_cli_rejects_changed_production_config_before_workload(tmp_path, capsys) -> None:
    repo, config, command = _inputs(tmp_path)
    (repo / "configs/truth_editing_production_release_abcd1234.json").write_text("changed\n")
    assert MODULE.main([
        "--config", str(config), "--repo", str(repo), "--command-json", str(command),
        "--observation", str(tmp_path / "observation.json"),
        "--receipt", str(tmp_path / "receipt.json"), "--gpu-hourly-usd", "0.36",
    ]) == 2
    assert "SHA-256" in capsys.readouterr().err
