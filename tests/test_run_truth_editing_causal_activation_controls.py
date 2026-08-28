from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from test_truth_editing_causal_activation_controls import _Executor, _plan


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_truth_editing_causal_activation_controls.py"
)
SPEC = importlib.util.spec_from_file_location("run_causal_activation_controls_cli", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cli_loads_executor_factory_and_writes_receipt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    plan = _plan(tmp_path)
    config = tmp_path / "executor-config.json"
    config.write_text("{}\n")
    executor = _Executor(tmp_path)
    observed: dict[str, object] = {}

    def load(spec: str, config_path: Path):
        observed.update(spec=spec, config_path=config_path)
        return executor

    monkeypatch.setattr(MODULE, "_load_executor", load)
    receipt_path = tmp_path / "receipt.json"

    MODULE.main(
        [
            str(plan),
            "--executor-factory",
            "package.module:build_executor",
            "--executor-config",
            str(config),
            "--output",
            str(receipt_path),
        ]
    )

    assert observed == {
        "spec": "package.module:build_executor",
        "config_path": config,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "executed_passed"
    assert json.loads(receipt_path.read_text())["self_sha256"]
