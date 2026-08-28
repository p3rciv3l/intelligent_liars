from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_truth_editing_adaptive_finalization.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_truth_editing_adaptive_finalization_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_wires_validated_handoff_to_real_public_finalization_seams(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _module()
    study_receipt = tmp_path / "study-artifact-receipt.json"
    study_receipt.write_text(json.dumps({"receipt_sha256": "b" * 64}))
    handoff = {
        "study_identity_sha256": "a" * 64,
        "study_report": {"path": str(tmp_path / "study-report.json")},
        "study_artifact_receipt": {"path": str(study_receipt)},
        "production_config": {"path": str(tmp_path / "production.json")},
        "maximum_evaluation_spend_usd": "1",
        "strong_candidate_count": 3,
        "repeat_count_per_candidate": 2,
    }
    observed = {}

    class Exporter:
        def __init__(self, **kwargs):
            observed["exporter"] = kwargs

    class Production:
        def build_finalization_backend(self, **kwargs):
            observed["backend"] = kwargs
            return "backend"

    class Executor:
        def __init__(self, report_path, backend, **kwargs):
            observed["executor"] = (report_path, backend, kwargs)

    monkeypatch.setattr(module, "open_adaptive_finalization_handoff", lambda path: handoff)
    monkeypatch.setattr(module, "ProductionFinalistCheckpointExporter", Exporter)
    monkeypatch.setattr(module, "open_production_run", lambda path: Production())
    monkeypatch.setattr(module, "ProductionAdaptiveFinalizationExecutor", Executor)
    monkeypatch.setattr(
        module,
        "run_adaptive_finalization",
        lambda path, executor: {"format": "truth_editing_adaptive_finalization_receipt_v1"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            str(tmp_path / "handoff.json"),
            "--registry-bucket",
            "private-models",
            "--model-slug",
            "qwen-truth-edited",
            "--causal-control-receipt",
            f"trial-a={tmp_path / 'causal-a.json'}",
        ],
    )

    module.main()

    assert observed["executor"][:2] == (handoff["study_report"]["path"], "backend")
    assert observed["executor"][2]["causal_control_receipts"] == {
        "trial-a": tmp_path / "causal-a.json"
    }
    assert observed["backend"]["checkpoint_exporter"].__class__ is Exporter
    assert observed["backend"]["maximum_evaluation_cost_usd"].startswith("0.08333")
    assert json.loads(capsys.readouterr().out)["format"].endswith("receipt_v1")
