from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build_truth_editing_human_calibration_pack.py"


def _module():
    spec = importlib.util.spec_from_file_location("human_pack_cli", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_builds_pack_and_prints_identity(tmp_path: Path, capsys) -> None:
    result = _module().main(
        [
            "build",
            "--dataset",
            str(ROOT / "datasets/truth_editing/v2"),
            "--qualification",
            str(ROOT / "artifacts/truth-editing/base-known"),
            "--output",
            str(tmp_path / "pack"),
        ]
    )

    assert result == 0
    assert (tmp_path / "pack" / "LABELING.md").is_file()
    assert "pack_sha256" in capsys.readouterr().out
