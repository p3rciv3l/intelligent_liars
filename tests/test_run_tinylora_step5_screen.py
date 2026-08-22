from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_tinylora_step5_screen.py"
SPEC = importlib.util.spec_from_file_location("run_tinylora_step5_screen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _numeric_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "max_steps": 2,
        "max_length": 16,
        "learning_rate": 0.1,
        "gradient_accumulation": 1,
        "checkpoint_every": 1,
        "checkpoint_minutes": 1.0,
        "development_per_objective": 0,
        "seed": 17,
        "projection_seed": 23,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("max_length", 1),
        ("learning_rate", float("nan")),
        ("gradient_accumulation", 0),
        ("checkpoint_every", 0),
        ("checkpoint_minutes", float("inf")),
        ("development_per_objective", -1),
    ],
)
def test_numeric_cli_contract_fails_closed(field: str, value: object):
    with pytest.raises(ValueError):
        MODULE.validate_numeric_args(_numeric_args(**{field: value}))


def test_seed_all_controls_python_numpy_and_torch():
    MODULE.seed_all(123)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    MODULE.seed_all(123)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second


def test_checkpoint_identity_records_budget_code_objective_and_basis():
    identity = MODULE.build_checkpoint_identity(
        plan_sha256="1" * 64,
        probe_sha256="2" * 64,
        code_sha256="3" * 64,
        basis_sha256="4" * 64,
        arm={"name": "tiny"},
        model={"model_id": "model", "revision": "revision"},
        mode="train",
        max_steps=10,
        seed=7,
        projection_seed=9,
        max_length=128,
        gradient_accumulation=2,
        learning_rate=1e-4,
    )
    assert identity["mode"] == "train"
    assert identity["budget"] == {"max_steps": 10}
    assert identity["code_sha256"] == "3" * 64
    assert identity["basis_sha256"] == "4" * 64
    assert identity["objective"] == MODULE.OBJECTIVE_CONFIGURATION
    assert identity["training_seed"] == 7
    assert identity["projection_seed"] == 9


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor(0.0))


def _fake_behavior_loss(*, model: _ScalarModel, **_kwargs: object):
    loss = (model.adapter - 1.0).square()
    hidden = torch.zeros((1, 1, 1))
    return loss, hidden, torch.full((1, 1), -100), hidden


def _train(
    model: _ScalarModel,
    checkpoint: Path,
    *,
    max_steps: int = 4,
) -> dict[str, object]:
    return MODULE.train_arm(
        model=model,
        processor=None,
        capture=None,
        parameters=[model.adapter],
        rows=[
            {
                "kind": "behavior",
                "objective": "truthful_direct_report",
                "record_id": "r",
            }
        ],
        direction=torch.zeros(1),
        intercept=0.0,
        desired_delta=0.1,
        max_steps=max_steps,
        max_length=8,
        gradient_accumulation=1,
        learning_rate=0.1,
        checkpoint_every=1,
        checkpoint_minutes=100.0,
        checkpoint_path=checkpoint,
        identity={"run": "same"},
    )


def test_train_arm_resume_is_numerically_equivalent_on_cpu(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)
    uninterrupted = _ScalarModel()
    _train(uninterrupted, tmp_path / "uninterrupted.pt")

    interrupted = _ScalarModel()
    real_save = MODULE.atomic_torch_save

    def stop_after_second_checkpoint(path: Path, payload: dict[str, object]) -> None:
        real_save(path, payload)
        if payload["optimizer_steps"] == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(MODULE, "atomic_torch_save", stop_after_second_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _train(interrupted, tmp_path / "resumed.pt")
    monkeypatch.setattr(MODULE, "atomic_torch_save", real_save)
    result = _train(interrupted, tmp_path / "resumed.pt")
    assert result["optimizer_steps"] == 4
    assert interrupted.adapter.item() == pytest.approx(
        uninterrupted.adapter.item(), rel=0, abs=1e-8
    )


def test_resume_rejects_checkpoint_beyond_requested_budget(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)
    model = _ScalarModel()
    _train(model, tmp_path / "state.pt", max_steps=2)
    with pytest.raises(ValueError, match="exceeds requested budget"):
        _train(_ScalarModel(), tmp_path / "state.pt", max_steps=1)


def test_checkpoint_writer_rejects_step_regression(tmp_path: Path):
    path = tmp_path / "state.pt"
    common = {"identity": {"run": "same"}}
    MODULE.atomic_torch_save(path, {**common, "optimizer_steps": 3})
    with pytest.raises(ValueError, match="step regression"):
        MODULE.atomic_torch_save(path, {**common, "optimizer_steps": 2})


def test_train_arm_rejects_nonfinite_loss_before_optimizer_step(
    tmp_path: Path,
    monkeypatch,
):
    def nonfinite_loss(*, model: _ScalarModel, **_kwargs: object):
        hidden = torch.zeros((1, 1, 1))
        return (
            model.adapter * torch.tensor(float("nan")),
            hidden,
            torch.full((1, 1), -100),
            hidden,
        )

    monkeypatch.setattr(MODULE, "_behavior_loss", nonfinite_loss)
    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        _train(_ScalarModel(), tmp_path / "state.pt", max_steps=1)
