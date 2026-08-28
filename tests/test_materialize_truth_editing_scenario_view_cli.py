from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

from intelligent_liars.truth_editing_scenario_view import TruthEditingScenarioView


ROOT = Path(__file__).resolve().parents[1]


def test_scenario_view_config_materializes_and_reopens_qualified_view(
    tmp_path: Path,
) -> None:
    # Exercise the checked-in source/config contract while keeping generated
    # test output isolated from the repository.
    config = tmp_path / "configs" / "view.json"
    config.parent.mkdir()
    config.write_text(
        "{\n"
        '  "format": "truth_editing_scenario_view_build_config_v1",\n'
        f'  "source_dataset": "{ROOT / "datasets/truth_editing/v2"}",\n'
        f'  "output": "{tmp_path / "generated"}",\n'
        '  "tier_scenario_limits": [8, 32, 128],\n'
        f'  "base_known_qualification": "{ROOT / "artifacts/truth-editing/base-known"}"\n'
        "}\n"
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/materialize_truth_editing_scenario_view.py"), "--config", str(config)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    view = TruthEditingScenarioView.open(
        tmp_path / "generated",
        source_dataset=ROOT / "datasets/truth_editing/v2",
        base_known_qualification=ROOT / "artifacts/truth-editing/base-known",
    )
    assert view.manifest["qualification_mode"] == "frozen_base_model_validation"
    assert view.manifest["qualification_receipt_sha256"] == (
        "3f25d06cec9b4fa0f59ebdd1e8ec65db97b4cd55d06ffd047cb24f23ad2cd2ee"
    )
    assert view.manifest["base_known_counts"] == {
        "known": 2142,
        "not_known": 777,
        "unknown": 0,
    }


def test_checked_in_scenario_view_config_binds_canonical_qualification() -> None:
    raw = json.loads((ROOT / "configs/truth_editing_scenario_view_v1.json").read_text())

    assert raw["output"] == (
        "datasets/truth_editing/views/v2_optimization_validation_qualified_v1"
    )
    assert raw["base_known_qualification"] == "artifacts/truth-editing/base-known"
