from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from intelligent_liars.step5_multimodal_assets import stage_multimodal_bundle

from intelligent_liars.step5_prerequisites import validate_prerequisite_receipt


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
        "mode": "smoke",
        "overfit_min_relative_loss_reduction": 0.20,
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


def test_preservation_image_rebase_is_copy_only_and_absolute(tmp_path: Path):
    import hashlib
    import io

    from PIL import Image

    source = tmp_path / "source"
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (1, 2, 3)).save(buffer, format="JPEG")
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    relative = Path("data/tinylora_preservation_snapshots/v1/pixmo_docs_images") / f"{digest}.jpg"
    (source / relative).parent.mkdir(parents=True)
    (source / relative).write_bytes(payload)
    corpus = source / "preservation.jsonl"
    original = {
        "record_id": "vision.1",
        "preservation_category": "vision_charts",
        "image_sha256": digest,
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": relative.as_posix()}]},
            {"role": "assistant", "content": "answer"},
        ],
    }
    corpus.write_text(json.dumps(original) + "\n")
    bundle = tmp_path / "bundle"
    stage_multimodal_bundle([corpus], project_root=source, destination=bundle)

    rebased = MODULE.rebase_preservation_images([original], pixmo_bundle=bundle)

    assert original["messages"][0]["content"][0]["image"] == relative.as_posix()
    mapped = Path(rebased[0]["messages"][0]["content"][0]["image"])
    assert mapped.is_absolute() and mapped.read_bytes() == payload


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
        runtime_image_digest="sha256:" + "5" * 64,
        schedule_sha256="6" * 64,
    )
    assert identity["mode"] == "train"
    assert identity["budget"] == {"max_steps": 10}
    assert identity["code_sha256"] == "3" * 64
    assert identity["basis_sha256"] == "4" * 64
    assert identity["objective"] == MODULE.OBJECTIVE_CONFIGURATION
    assert identity["training_seed"] == 7
    assert identity["projection_seed"] == 9
    assert identity["runtime"]["image_digest"] == "sha256:" + "5" * 64
    assert identity["sampler"]["schedule_sha256"] == "6" * 64
    assert identity["checkpoint_schema"] == "tinylora_step5_checkpoint_v2"


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor(0.0))


def _fake_behavior_loss(*, model: _ScalarModel, **_kwargs: object):
    loss = (model.adapter - 1.0).square()
    hidden = torch.zeros((1, 1, 1))
    return loss, hidden, torch.full((1, 1), -100), hidden


def _durability_receipt(
    generation,
    *,
    verified: bool = True,
    generation_id: str | None = None,
) -> dict[str, object]:
    return {
        "format": MODULE.DURABILITY_RECEIPT_FORMAT,
        "generation_id": generation_id or generation.generation_id,
        "manifest_sha256": generation.manifest_sha256,
        "object_ref": "s3://example/checkpoint",
        "verified": verified,
    }


def _checkpoint_identity() -> dict[str, object]:
    return {
        "plan_sha256": "1" * 64,
        "probe_sha256": "4" * 64,
        "code_sha256": "5" * 64,
        "arm": {
            "name": "tiny-test",
            "adapter": "tinylora",
            "svd_rank": 3,
            "projection_dim": 13,
            "train_layers": [21],
        },
        "model": {"model_id": "test/model", "revision": "2" * 40},
        "training_seed": 17,
        "projection_seed": 23,
        "basis_sha256": "3" * 64,
    }


def _train(
    model: _ScalarModel,
    checkpoint_root: Path,
    *,
    max_steps: int = 4,
    durability_verifier=None,
) -> dict[str, object]:
    if durability_verifier is None:
        durability_verifier = MODULE.local_smoke_durability_verifier
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
        checkpoint_root=checkpoint_root,
        identity=_checkpoint_identity(),
        durability_verifier=durability_verifier,
    )


def test_train_arm_resume_is_numerically_equivalent_on_cpu(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)
    uninterrupted = _ScalarModel()
    _train(uninterrupted, tmp_path / "uninterrupted-store")

    interrupted = _ScalarModel()
    real_publish = MODULE.publish_checkpoint_generation

    def stop_after_second_checkpoint(**kwargs: object):
        generation = real_publish(**kwargs)
        state = torch.load(
            generation.path / "step5_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state["optimizer_steps"] == 2:
            raise RuntimeError("simulated interruption")
        return generation

    monkeypatch.setattr(
        MODULE, "publish_checkpoint_generation", stop_after_second_checkpoint
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _train(interrupted, tmp_path / "resumed")
    monkeypatch.setattr(MODULE, "publish_checkpoint_generation", real_publish)
    result = _train(interrupted, tmp_path / "resumed")
    assert result["optimizer_steps"] == 4
    assert result["durable_checkpoint"]["verified"] is True
    assert interrupted.adapter.item() == pytest.approx(
        uninterrupted.adapter.item(), rel=0, abs=1e-8
    )


def test_resume_rejects_checkpoint_beyond_requested_budget(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)
    model = _ScalarModel()
    _train(model, tmp_path / "budget-store", max_steps=2)
    with pytest.raises(ValueError, match="exceeds requested budget"):
        _train(_ScalarModel(), tmp_path / "budget-store", max_steps=1)


def test_rejected_durability_receipt_blocks_completion_and_latest(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)

    def reject(generation):
        return _durability_receipt(generation, verified=False)

    checkpoint_root = tmp_path / "rejected"
    with pytest.raises(MODULE.CheckpointIntegrityError, match="durability"):
        _train(
            _ScalarModel(),
            checkpoint_root,
            max_steps=1,
            durability_verifier=reject,
        )
    assert not (checkpoint_root / "latest.json").exists()
    assert not list((checkpoint_root / "receipts").glob("*.json"))


def test_train_arm_retains_only_two_verified_generations(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)
    checkpoint_root = tmp_path / "retained"
    result = _train(_ScalarModel(), checkpoint_root, max_steps=4)

    assert result["durable_checkpoint"]["verified"] is True
    generations = sorted((checkpoint_root / "generations").iterdir())
    assert len(generations) == 2
    assert all((path / "step5_state.pt").is_file() for path in generations)
    pointer = json.loads((checkpoint_root / "latest.json").read_text())
    assert pointer["generation_id"] == result["durable_checkpoint"]["generation_id"]


def test_final_generation_contains_safe_evaluation_adapter(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)
    checkpoint_root = tmp_path / "evaluation-adapter-store"
    model = _ScalarModel()
    result = _train(model, checkpoint_root, max_steps=1)
    generation_id = result["durable_checkpoint"]["generation_id"]
    generation_path = checkpoint_root / "generations" / generation_id

    adapter_state = load_file(str(generation_path / "adapter_state.safetensors"))
    assert set(adapter_state) == {"adapter"}
    assert adapter_state["adapter"].item() == pytest.approx(model.adapter.item())
    metadata = json.loads((generation_path / "adapter_metadata.json").read_text())
    identity = _checkpoint_identity()
    assert metadata == {
        "format": "tinylora_step5_adapter_state_v1",
        "arm_name": "tiny-test",
        "arm": identity["arm"],
        "plan_sha256": "1" * 64,
        "model": {"model_id": "test/model", "revision": "2" * 40},
        "training_seed": 17,
        "projection_seed": 23,
        "basis_sha256": "3" * 64,
        "optimizer_steps": 1,
        "tensor_names": ["adapter"],
        "adapter_state_sha256": MODULE.file_sha256(
            generation_path / "adapter_state.safetensors"
        ),
        "checkpoint_identity": identity,
        "checkpoint_identity_sha256": MODULE.canonical_json_sha256(identity),
    }
    manifest = json.loads((generation_path / "manifest.json").read_text())
    assert {entry["path"] for entry in manifest["files"]} == {
        "adapter_metadata.json",
        "adapter_state.safetensors",
        "step5_state.pt",
    }


def test_checkpoint_publication_rejects_step_regression(tmp_path: Path):
    checkpoint_root = tmp_path / "regression-store"
    identity = _checkpoint_identity()
    MODULE.publish_checkpoint_generation(
        checkpoint_root=checkpoint_root,
        identity=identity,
        state={
            "optimizer_steps": 3,
            "adapter_state": {"adapter": torch.tensor(3.0)},
        },
        durability_verifier=MODULE.local_smoke_durability_verifier,
    )

    with pytest.raises(MODULE.CheckpointIntegrityError, match="step regression"):
        MODULE.publish_checkpoint_generation(
            checkpoint_root=checkpoint_root,
            identity=identity,
            state={
                "optimizer_steps": 2,
                "adapter_state": {"adapter": torch.tensor(2.0)},
            },
            durability_verifier=MODULE.local_smoke_durability_verifier,
        )


def test_resume_cannot_complete_when_final_receipt_is_missing(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)
    checkpoint_root = tmp_path / "missing-receipt"
    result = _train(_ScalarModel(), checkpoint_root, max_steps=1)
    generation_id = result["durable_checkpoint"]["generation_id"]
    (checkpoint_root / "receipts" / f"{generation_id}.json").unlink()

    with pytest.raises(MODULE.CheckpointIntegrityError, match="lacks.*receipt"):
        _train(_ScalarModel(), checkpoint_root, max_steps=1)


def test_receipt_must_bind_exact_generation_identity(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(MODULE, "_behavior_loss", _fake_behavior_loss)

    def wrong_generation(generation):
        return _durability_receipt(generation, generation_id="wrong")

    with pytest.raises(MODULE.CheckpointIntegrityError, match="does not match"):
        _train(
            _ScalarModel(),
            tmp_path / "wrong-receipt",
            max_steps=1,
            durability_verifier=wrong_generation,
        )


def test_train_mode_requires_external_durability_verifier():
    with pytest.raises(ValueError, match="durability-verifier-command"):
        MODULE.validate_durability_args(
            argparse.Namespace(
                mode="train",
                durability_verifier_command=None,
                runtime_image_digest=None,
                prerequisite_receipt=None,
            )
        )


def test_train_mode_requires_immutable_runtime_image_digest():
    with pytest.raises(ValueError, match="runtime-image-digest"):
        MODULE.validate_durability_args(
            argparse.Namespace(
                mode="train",
                durability_verifier_command="uploader",
                runtime_image_digest="latest",
                prerequisite_receipt=Path("receipt.json"),
            )
        )


def test_train_mode_requires_prerequisite_receipt():
    with pytest.raises(ValueError, match="prerequisite-receipt"):
        MODULE.validate_durability_args(
            argparse.Namespace(
                mode="train",
                durability_verifier_command="uploader",
                runtime_image_digest="sha256:" + "5" * 64,
                prerequisite_receipt=None,
            )
        )


def _prerequisite_identity(arm: dict | None = None) -> dict:
    return MODULE.prerequisite_identity(
        plan_sha256="1" * 64,
        probe_sha256="2" * 64,
        code_sha256="3" * 64,
        arm=arm or {"name": "tiny13"},
        model={"model_id": "example/model", "revision": "4" * 40},
        runtime_image_digest="sha256:" + "5" * 64,
    )


def _passing_prerequisite_receipt(identity: dict) -> dict:
    return MODULE.build_prerequisite_receipt(
        identity=identity,
        reachability={
            "selected_records": 8,
            "unexplained_skips": 0,
            "gradient_dimensions": {
                name: 13 for name in MODULE.EXPECTED_REACHABILITY_OBJECTIVES
            },
            "gradient_norms": {
                name: 0.5 for name in MODULE.EXPECTED_REACHABILITY_OBJECTIVES
            },
        },
        initial_mean_loss=2.0,
        final_mean_loss=1.0,
        optimizer_steps=20,
        selected_records=5,
        required_relative_loss_reduction=0.20,
        evaluator_expected_record_ids=[f"record-{index}" for index in range(5)],
        evaluator_metric_rows=[
            {
                "record_id": f"record-{index}",
                "base_margin": 0.1,
                "student_margin": 0.2,
                "improvement": 0.1,
                "direction_movement": None,
            }
            for index in range(5)
        ],
    )


def test_prerequisite_evaluator_evidence_matches_evaluate_arm_schema():
    rows = MODULE.prerequisite_evaluator_metric_rows(
        {
            "behavior": [
                {
                    "record_id": "record-1",
                    "scenario_id": "scenario-1",
                    "family": "family",
                    "objective": "truthful_direct_report",
                    "base_margin": 0.1,
                    "student_margin": 0.3,
                    "improvement": 0.2,
                    "direction_movement": None,
                }
            ]
        }
    )
    assert rows == [
        {
            "record_id": "record-1",
            "base_margin": 0.1,
            "student_margin": 0.3,
            "improvement": 0.2,
            "direction_movement": None,
        }
    ]


@pytest.mark.parametrize("failure", ["stale", "wrong-arm", "failed", "smoke"])
def test_prerequisite_receipt_rejects_stale_wrong_arm_failed_and_smoke(
    failure: str,
):
    expected = _prerequisite_identity()
    receipt = _passing_prerequisite_receipt(expected)
    if failure == "stale":
        receipt["identity"]["code_sha256"] = "9" * 64
    elif failure == "wrong-arm":
        receipt["identity"]["arm"] = {"name": "ordinary-rank3"}
    elif failure == "failed":
        receipt["passed"] = False
    else:
        receipt["generation_mode"] = "smoke"
    with pytest.raises(ValueError):
        validate_prerequisite_receipt(
            receipt,
            expected_identity=expected,
        )


@pytest.mark.parametrize(
    ("check", "field", "value"),
    [
        ("reachability", "unexplained_skips", 1),
        ("reachability", "gradient_norms", {}),
        ("intentional_overfit", "unexplained_skips", 1),
        ("intentional_overfit", "is_smoke", True),
        ("intentional_overfit", "final_mean_loss", 2.0),
        ("intentional_overfit", "optimizer_steps", 1),
        ("intentional_overfit", "relative_loss_reduction", 0.20),
        ("evaluator_sanity", "unexplained_skips", 1),
        ("evaluator_sanity", "evaluated_records", 4),
        ("evaluator_sanity", "evaluated_record_ids", ["wrong"] * 5),
        ("evaluator_sanity", "all_metrics_finite", False),
    ],
)
def test_prerequisite_receipt_requires_reachability_overfit_and_evaluator_sanity(
    check: str,
    field: str,
    value: object,
):
    identity = _prerequisite_identity()
    receipt = _passing_prerequisite_receipt(identity)
    receipt["checks"][check][field] = value
    with pytest.raises(ValueError):
        validate_prerequisite_receipt(
            receipt,
            expected_identity=identity,
        )


def test_prerequisite_receipt_rejects_missing_duplicate_and_nonfinite_evidence():
    identity = _prerequisite_identity()
    mutations = []

    missing = _passing_prerequisite_receipt(identity)
    missing["checks"]["evaluator_sanity"]["metric_rows"].pop()
    mutations.append(missing)

    duplicate = _passing_prerequisite_receipt(identity)
    duplicate["checks"]["evaluator_sanity"]["expected_record_ids"][1] = "record-0"
    mutations.append(duplicate)

    nonfinite = _passing_prerequisite_receipt(identity)
    nonfinite["checks"]["evaluator_sanity"]["metric_rows"][0]["improvement"] = float(
        "nan"
    )
    mutations.append(nonfinite)

    missing_metric = _passing_prerequisite_receipt(identity)
    missing_metric["checks"]["evaluator_sanity"]["metric_rows"][0]["base_margin"] = None
    mutations.append(missing_metric)

    for receipt in mutations:
        with pytest.raises(ValueError):
            validate_prerequisite_receipt(
                receipt,
                expected_identity=identity,
            )


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
        _train(_ScalarModel(), tmp_path / "nonfinite-store", max_steps=1)
