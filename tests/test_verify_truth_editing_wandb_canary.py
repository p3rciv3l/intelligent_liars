from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from test_truth_editing_wandb_canary import _checkpoint, _trace


SCRIPT = Path(__file__).parents[1] / "scripts/verify_truth_editing_wandb_canary.py"
SPEC = importlib.util.spec_from_file_location("verify_truth_editing_wandb_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cli_verifies_offline_trace_without_importing_wandb(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    trace = tmp_path / "trace.json"
    checkpoint = tmp_path / "checkpoint.json"
    receipt = tmp_path / "receipt.json"
    trace.write_text(json.dumps(_trace()))
    checkpoint.write_text(json.dumps(_checkpoint()))
    monkeypatch.setattr(
        MODULE.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("W&B imported offline")),
    )
    assert MODULE.main([
        "--trace", str(trace),
        "--checkpoint", str(checkpoint),
        "--receipt", str(receipt),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "offline_verified"
    assert receipt.is_file()


def test_cli_live_readback_replaces_untrusted_dashboard_claim(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    trace_payload = _trace()
    trace_payload["dashboard_readback"]["run_id"] = "forged"
    trace = tmp_path / "trace.json"
    checkpoint = tmp_path / "checkpoint.json"
    trace.write_text(json.dumps(trace_payload))
    checkpoint.write_text(json.dumps(_checkpoint()))
    expected = _trace()["dashboard_readback"]

    class Api:
        pass

    class Wandb:
        @staticmethod
        def Api():
            return Api()

    monkeypatch.setattr(MODULE.importlib, "import_module", lambda _name: Wandb)
    monkeypatch.setattr(MODULE, "read_wandb_dashboard", lambda *_a, **_k: expected)
    assert MODULE.main([
        "--trace", str(trace),
        "--checkpoint", str(checkpoint),
        "--receipt", str(tmp_path / "receipt.json"),
        "--verify-dashboard-live",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "live_dashboard_verified"
