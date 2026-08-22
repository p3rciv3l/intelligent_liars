from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_tinylora_bounded_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_tinylora_bounded_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LegacyPilotState:
    pass


def _valid_state() -> dict[str, object]:
    return {
        "format": "tinylora_bounded_pilot_state_v1",
        "plan_sha256": "a" * 64,
        "probe_sha256": "b" * 64,
        "rank": 2,
        "optimizer_steps": 3,
        "next_example": 24,
        "tinylora_vector": torch.tensor([0.1, 0.2]),
        "optimizer": {
            "state": {0: {"step": torch.tensor(3.0)}},
            "param_groups": [{"params": [0], "lr": 2e-4}],
        },
        "history": [{"total": 1.25}],
    }


def test_load_resume_state_accepts_tensor_and_primitive_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pilot.pt"
    torch.save(_valid_state(), path)

    state = MODULE.load_resume_state(
        path,
        plan_sha256="a" * 64,
        probe_sha256="b" * 64,
        rank=2,
    )

    assert state["optimizer_steps"] == 3
    assert torch.equal(state["tinylora_vector"], torch.tensor([0.1, 0.2]))


def test_load_resume_state_rejects_legacy_object_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({**_valid_state(), "legacy": LegacyPilotState()}, path)

    with pytest.raises(ValueError, match="unsafe or invalid bounded-pilot checkpoint"):
        MODULE.load_resume_state(
            path,
            plan_sha256="a" * 64,
            probe_sha256="b" * 64,
            rank=2,
        )


def test_load_resume_state_rejects_probe_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "pilot.pt"
    torch.save(_valid_state(), path)

    with pytest.raises(ValueError, match="identity differs"):
        MODULE.load_resume_state(
            path,
            plan_sha256="a" * 64,
            probe_sha256="c" * 64,
            rank=2,
        )
