#!/usr/bin/env python3
"""Prove exact CPU checkpoint/resume equivalence with a planned interruption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1
DATASET_SIZE = 23
BATCH_SIZE = 4


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


class StatefulShuffleSampler:
    """Small resumable sampler with an explicit permutation cursor and RNG."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.position = 0
        self.epoch = 0

    def next_indices(self, batch_size: int) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        remaining = batch_size
        while remaining:
            available = self.size - self.position
            take = min(remaining, available)
            pieces.append(self.order[self.position : self.position + take])
            self.position += take
            remaining -= take
            if self.position == self.size:
                self.epoch += 1
                self.order = torch.randperm(self.size, generator=self.generator)
                self.position = 0
        return torch.cat(pieces)

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "order": self.order.clone(),
            "position": self.position,
            "epoch": self.epoch,
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("Checkpoint sampler size does not match the dataset")
        self.order = state["order"].clone()
        self.position = int(state["position"])
        self.epoch = int(state["epoch"])
        self.generator.set_state(state["generator_state"])


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(5, 9),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(9, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def toy_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.linspace(-1.75, 2.25, DATASET_SIZE * 5).reshape(DATASET_SIZE, 5)
    targets = torch.stack(
        (
            inputs[:, 0] - 0.25 * inputs[:, 2] + 0.1,
            inputs[:, 4] + 0.5 * inputs[:, 1] - 0.2,
        ),
        dim=1,
    )
    return inputs, targets


def new_training_state(seed: int) -> dict[str, Any]:
    seed_all(seed)
    model = ToyModel().cpu()
    return {
        "model": model,
        "optimizer": torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.02),
        "sampler": StatefulShuffleSampler(DATASET_SIZE, seed + 101),
        "completed_steps": 0,
        "sample_trace": [],
        "loss_trace": [],
    }


def train_until(state: dict[str, Any], total_steps: int) -> None:
    inputs, targets = toy_dataset()
    model: ToyModel = state["model"]
    optimizer: torch.optim.Optimizer = state["optimizer"]
    sampler: StatefulShuffleSampler = state["sampler"]
    model.train()
    while state["completed_steps"] < total_steps:
        indices = sampler.next_indices(BATCH_SIZE)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs[indices])
        # Exercise all three global RNG streams in the gradient-producing path.
        scale = 0.97 + 0.02 * random.random() + 0.01 * float(np.random.random())
        jitter = 0.001 * torch.randn((), dtype=prediction.dtype)
        loss = torch.nn.functional.mse_loss(prediction, targets[indices]) * scale + jitter
        loss.backward()
        optimizer.step()
        state["completed_steps"] += 1
        state["sample_trace"].append(indices.tolist())
        state["loss_trace"].append(float(loss.detach()))


def checkpoint_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "sampler": state["sampler"].state_dict(),
        "rng": capture_rng_state(),
        "completed_steps": state["completed_steps"],
        "sample_trace": state["sample_trace"],
        "loss_trace": state["loss_trace"],
    }


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    atomic_torch_save(path, checkpoint_payload(state))


def load_checkpoint(path: Path, state: dict[str, Any]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    state["model"].load_state_dict(payload["model"])
    state["optimizer"].load_state_dict(payload["optimizer"])
    state["sampler"].load_state_dict(payload["sampler"])
    state["completed_steps"] = int(payload["completed_steps"])
    state["sample_trace"] = payload["sample_trace"]
    state["loss_trace"] = payload["loss_trace"]
    # Restore last so model/optimizer reconstruction cannot perturb resumed RNGs.
    restore_rng_state(payload["rng"])


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _update_hash(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"torch:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"numpy:{array.dtype}:{array.shape}:".encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_hash(digest, key)
            _update_hash(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}[".encode())
        for item in value:
            _update_hash(digest, item)
        digest.update(b"]")
    else:
        digest.update(f"{type(value).__name__}:{value!r};".encode())


def stable_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, value)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "sampler": state["sampler"].state_dict(),
        "rng": capture_rng_state(),
        "completed_steps": state["completed_steps"],
        "sample_trace": state["sample_trace"],
        "loss_trace": state["loss_trace"],
    }


def public_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed_steps": snapshot["completed_steps"],
        "model_sha256": stable_hash(snapshot["model"]),
        "optimizer_sha256": stable_hash(snapshot["optimizer"]),
        "sampler_sha256": stable_hash(snapshot["sampler"]),
        "rng_sha256": stable_hash(snapshot["rng"]),
        "sample_trace": snapshot["sample_trace"],
        "sample_trace_sha256": stable_hash(snapshot["sample_trace"]),
        "loss_trace": snapshot["loss_trace"],
        "loss_trace_sha256": stable_hash(snapshot["loss_trace"]),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_equivalence(
    *,
    receipt_path: Path,
    checkpoint_path: Path,
    seed: int,
    total_steps: int,
    interrupt_after: int,
) -> dict[str, Any]:
    if total_steps < 2 or not 0 < interrupt_after < total_steps:
        raise ValueError("interrupt_after must be strictly between 0 and total_steps")

    reference_state = new_training_state(seed)
    train_until(reference_state, total_steps)
    reference = final_snapshot(reference_state)

    interrupted_state = new_training_state(seed)
    train_until(interrupted_state, interrupt_after)
    save_checkpoint(checkpoint_path, interrupted_state)

    # Deliberately perturb every RNG and reconstruct all objects as a fresh process would.
    resumed_state = new_training_state(seed + 999_983)
    random.random()
    np.random.random()
    torch.rand(7)
    load_checkpoint(checkpoint_path, resumed_state)
    train_until(resumed_state, total_steps)
    resumed = final_snapshot(resumed_state)

    comparisons = {
        "model_state": _equal(reference["model"], resumed["model"]),
        "optimizer_state": _equal(reference["optimizer"], resumed["optimizer"]),
        "sampler_state": _equal(reference["sampler"], resumed["sampler"]),
        "rng_state": _equal(reference["rng"], resumed["rng"]),
        "completed_steps": reference["completed_steps"] == resumed["completed_steps"],
        "sample_trace": reference["sample_trace"] == resumed["sample_trace"],
        "loss_trace": reference["loss_trace"] == resumed["loss_trace"],
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "test": "planned_interruption_checkpoint_resume_equivalence",
        "status": "pass" if all(comparisons.values()) else "fail",
        "device": "cpu",
        "seed": seed,
        "total_steps": total_steps,
        "interrupt_after": interrupt_after,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "comparisons": comparisons,
        "reference": public_summary(reference),
        "resumed": public_summary(resumed),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument("--interrupt-after", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = run_equivalence(
        receipt_path=args.receipt,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        total_steps=args.total_steps,
        interrupt_after=args.interrupt_after,
    )
    print(json.dumps({"status": receipt["status"], "receipt": str(args.receipt.resolve())}))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
