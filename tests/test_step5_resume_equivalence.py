from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "test_step5_resume_equivalence.py"
SPEC = importlib.util.spec_from_file_location("test_step5_resume_equivalence_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_planned_interruption_matches_uninterrupted_training(tmp_path: Path):
    receipt_path = tmp_path / "receipt.json"

    receipt = MODULE.run_equivalence(
        receipt_path=receipt_path,
        checkpoint_path=tmp_path / "planned-interruption.pt",
        seed=91,
        total_steps=12,
        interrupt_after=5,
    )

    assert receipt["status"] == "pass"
    assert all(receipt["comparisons"].values())
    assert receipt["reference"]["sample_trace"] == receipt["resumed"]["sample_trace"]
    assert receipt["reference"]["loss_trace"] == receipt["resumed"]["loss_trace"]
    assert receipt_path.exists()
    assert (tmp_path / "planned-interruption.pt").exists()
    assert json.loads(receipt_path.read_text()) == receipt


def test_invalid_interruption_boundary_is_rejected(tmp_path: Path):
    try:
        MODULE.run_equivalence(
            receipt_path=tmp_path / "receipt.json",
            checkpoint_path=tmp_path / "checkpoint.pt",
            seed=3,
            total_steps=8,
            interrupt_after=8,
        )
    except ValueError as error:
        assert "strictly between" in str(error)
    else:
        raise AssertionError("Expected an invalid interruption boundary to fail")


def test_cli_emits_machine_readable_receipt(tmp_path: Path):
    receipt_path = tmp_path / "cli-receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--receipt",
            str(receipt_path),
            "--checkpoint",
            str(tmp_path / "cli-checkpoint.pt"),
            "--seed",
            "17",
            "--total-steps",
            "10",
            "--interrupt-after",
            "4",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "pass"
    assert json.loads(completed.stdout)["receipt"] == str(receipt_path.resolve())
