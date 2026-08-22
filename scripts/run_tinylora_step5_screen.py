#!/usr/bin/env python3
"""Run one fail-closed TinyLoRA Step 5 reachability or bounded-screen arm."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from intelligent_liars.durable_checkpoints import (
    CheckpointGeneration,
    CheckpointIntegrityError,
    advance_latest_checkpoint,
    create_checkpoint_generation,
    resolve_latest_checkpoint,
)
from intelligent_liars.models import ModelLoadConfig, load_model_and_processor
from intelligent_liars.standalone_models import (
    TinyLoRATrainingConfig,
    build_training_batch,
    install_tinylora_with_cache,
    weighted_causal_lm_loss,
)
from intelligent_liars.tinylora_pilot import (
    DIRECTIONAL_OBJECTIVE,
    assistant_probe_score,
    causal_preservation_targets,
    directional_margin_loss,
    file_sha256,
    paired_reference_improvement_loss,
    sequence_log_probability,
    topk_preservation_kl_loss,
)
from intelligent_liars.tinylora_step5 import (
    clustered_bootstrap_mean,
    gradient_cosine_matrix,
    install_ordinary_lora,
    preservation_interleaved_schedule,
)


OBJECTIVE_CONFIGURATION = {
    "paired_objective": "base_reference_token_average_margin_improvement",
    "preferred_ce_weight": 0.25,
    "directional_margin_weight": 0.25,
    "preservation_kl_weight": 0.5,
    "preservation_alignment": "causal_assistant_tokens",
}
DURABILITY_RECEIPT_FORMAT = "tinylora_step5_checkpoint_durability_receipt_v1"
CheckpointDurabilityVerifier = Callable[[CheckpointGeneration], Mapping[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("reachability", "smoke", "train"),
        default="reachability",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/workspace/cache/huggingface")
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--checkpoint-minutes", type=float, default=10.0)
    parser.add_argument("--development-per-objective", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument(
        "--runtime-image-digest",
        help="Immutable sha256:<64 hex> runtime image digest; required in train mode.",
    )
    parser.add_argument(
        "--durability-verifier-command",
        help=(
            "External checkpoint uploader/verifier command. It receives generation "
            "identity in STEP5_CHECKPOINT_* environment variables and must emit one "
            "matching JSON durability receipt on stdout. Required in train mode."
        ),
    )
    return parser.parse_args()


def validate_numeric_args(args: argparse.Namespace) -> None:
    """Fail closed on invalid or non-finite Step 5 numeric arguments."""
    positive_integers = {
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "checkpoint_every": args.checkpoint_every,
    }
    for name, value in positive_integers.items():
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be a positive integer")
    if (
        isinstance(args.max_length, bool)
        or int(args.max_length) != args.max_length
        or args.max_length < 2
    ):
        raise ValueError("max-length must be an integer of at least two")
    if args.development_per_objective < 0:
        raise ValueError("development-per-objective must be non-negative")
    for name in ("learning_rate", "checkpoint_minutes"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive and finite")


def validate_durability_args(args: argparse.Namespace) -> None:
    """Require a real external durability verifier for candidate training."""
    if args.mode == "train" and not args.durability_verifier_command:
        raise ValueError("train mode requires --durability-verifier-command")
    image_digest = args.runtime_image_digest
    if args.mode == "train" and (
        not isinstance(image_digest, str)
        or len(image_digest) != 71
        or not image_digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in image_digest[7:])
    ):
        raise ValueError("train mode requires an immutable --runtime-image-digest")


def seed_all(seed: int) -> None:
    """Seed every stochastic library used by adapter setup and training."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_checkpoint_identity(
    *,
    plan_sha256: str,
    probe_sha256: str,
    code_sha256: str,
    basis_sha256: str | None,
    arm: dict[str, Any],
    model: dict[str, Any],
    mode: str,
    max_steps: int,
    seed: int,
    projection_seed: int,
    max_length: int,
    gradient_accumulation: int,
    learning_rate: float,
    runtime_image_digest: str,
    schedule_sha256: str,
) -> dict[str, Any]:
    """Bind a resumable checkpoint to its complete training contract."""
    return {
        "plan_sha256": plan_sha256,
        "probe_sha256": probe_sha256,
        "code_sha256": code_sha256,
        "basis_sha256": basis_sha256,
        "arm": arm,
        "model": model,
        "mode": mode,
        "budget": {"max_steps": max_steps},
        "training_seed": seed,
        "projection_seed": projection_seed,
        "max_length": max_length,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
        "runtime": {
            "image_digest": runtime_image_digest,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_topology": "exactly_one_visible_cuda_gpu",
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "foreach": False,
        },
        "scheduler": {"name": "none"},
        "rng": "python_random_numpy_pytorch_cpu_and_visible_cuda",
        "sampler": {
            "name": "preservation_interleaved_schedule_v1",
            "schedule_sha256": schedule_sha256,
        },
        "checkpoint_schema": "tinylora_step5_checkpoint_v2",
        "objective": OBJECTIVE_CONFIGURATION,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def schedule_sha256(rows: list[dict[str, Any]]) -> str:
    """Bind resume identity to the exact ordered training-record schedule."""
    payload = [
        {
            "record_id": row["record_id"],
            "kind": row["kind"],
            "objective": row["objective"],
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def local_smoke_durability_verifier(
    generation: CheckpointGeneration,
) -> dict[str, Any]:
    """Accept local bytes only for non-candidate smoke tests."""
    return {
        "format": DURABILITY_RECEIPT_FORMAT,
        "generation_id": generation.generation_id,
        "manifest_sha256": generation.manifest_sha256,
        "object_ref": generation.path.resolve().as_uri(),
        "verified": True,
    }


def command_durability_verifier(command: str) -> CheckpointDurabilityVerifier:
    """Build a transport-neutral verifier backed by an operator command."""
    arguments = shlex.split(command)
    if not arguments:
        raise ValueError("durability verifier command cannot be empty")

    def verify(generation: CheckpointGeneration) -> Mapping[str, Any]:
        environment = os.environ.copy()
        environment.update(
            {
                "STEP5_CHECKPOINT_GENERATION_DIR": str(generation.path.resolve()),
                "STEP5_CHECKPOINT_GENERATION_ID": generation.generation_id,
                "STEP5_CHECKPOINT_MANIFEST_SHA256": generation.manifest_sha256,
            }
        )
        completed = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        try:
            receipt = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CheckpointIntegrityError(
                "Durability verifier did not emit one JSON receipt"
            ) from exc
        if not isinstance(receipt, dict):
            raise CheckpointIntegrityError(
                "Durability verifier receipt must be a JSON object"
            )
        return receipt

    return verify


def _validated_durability_receipt(
    generation: CheckpointGeneration,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "format",
        "generation_id",
        "manifest_sha256",
        "object_ref",
        "verified",
    }
    if set(receipt) != required:
        raise CheckpointIntegrityError(
            "Durability receipt fields do not match the required contract"
        )
    normalized = dict(receipt)
    if normalized["format"] != DURABILITY_RECEIPT_FORMAT:
        raise CheckpointIntegrityError("Unsupported durability receipt format")
    if (
        normalized["generation_id"] != generation.generation_id
        or normalized["manifest_sha256"] != generation.manifest_sha256
    ):
        raise CheckpointIntegrityError(
            "Durability receipt does not match the checkpoint generation"
        )
    if not isinstance(normalized["object_ref"], str) or not normalized["object_ref"]:
        raise CheckpointIntegrityError("Durability receipt object_ref is invalid")
    if normalized["verified"] is not True:
        raise CheckpointIntegrityError(
            f"Checkpoint durability verification rejected {generation.generation_id}"
        )
    return normalized


def publish_checkpoint_generation(
    *,
    checkpoint_root: Path,
    identity: dict[str, Any],
    state: dict[str, Any],
    durability_verifier: CheckpointDurabilityVerifier,
) -> CheckpointGeneration:
    """Publish, externally verify, and advance one immutable generation."""
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    step = int(state["optimizer_steps"])
    lock_path = checkpoint_root.parent / f".{checkpoint_root.name}.publication.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        latest_pointer = checkpoint_root / "latest.json"
        if latest_pointer.exists():
            latest = resolve_latest_checkpoint(
                checkpoint_root,
                expected_identity=identity,
            )
            latest_state = torch.load(
                latest.path / "step5_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            latest_step = int(latest_state.get("optimizer_steps", -1))
            if step <= latest_step:
                raise CheckpointIntegrityError(
                    "Refusing checkpoint step regression: "
                    f"new step {step} <= latest step {latest_step}"
                )

        generation_id = f"step-{step:06d}-{uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory(
            prefix=".step5-checkpoint-", dir=checkpoint_root.parent
        ) as temporary:
            source_dir = Path(temporary)
            torch.save(state, source_dir / "step5_state.pt")
            generation = create_checkpoint_generation(
                checkpoint_root,
                identity=identity,
                generation_id=generation_id,
                source_dir=source_dir,
            )

        accepted_receipt: dict[str, Any] | None = None

        def verify(candidate: CheckpointGeneration) -> bool:
            nonlocal accepted_receipt
            accepted_receipt = _validated_durability_receipt(
                candidate, durability_verifier(candidate)
            )
            receipt_path = (
                checkpoint_root / "receipts" / f"{candidate.generation_id}.json"
            )
            atomic_json(receipt_path, accepted_receipt)
            return True

        generation = advance_latest_checkpoint(
            checkpoint_root,
            generation.generation_id,
            identity=identity,
            durable_verifier=verify,
        )
        if accepted_receipt is None:
            raise CheckpointIntegrityError(
                "Durability verifier did not return a receipt"
            )
        return generation


def load_probe(path: Path) -> tuple[int, torch.Tensor, float, dict[str, Any]]:
    payload = json.loads(path.read_text())
    directions = payload.get("general_domain", {}).get("directions", [])
    if len(directions) != 1:
        raise ValueError("Probe must contain exactly one general-domain direction")
    direction = directions[0]
    if direction.get("direction_sign_convention") != (
        "sklearn_logistic_coef_positive_points_honest_to_deceptive"
    ):
        raise ValueError("Probe sign convention is not deception-positive")
    vector = torch.tensor(direction["direction_vector"], dtype=torch.float32)
    return int(direction["layer"]), vector, float(direction["intercept"]), direction


def training_record(
    row: dict[str, Any],
    *,
    alternative: bool = False,
) -> dict[str, Any]:
    if row["kind"] == "behavior":
        target = row["alternative_target"] if alternative else row["target"]
        return {
            "id": f"{row['record_id']}{'.alternative' if alternative else ''}",
            "messages": [{"role": "user", "content": row["prompt"]}],
            "assistant_content": target,
        }
    messages = list(row["messages"])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            f"Preservation row lacks final assistant turn: {row['record_id']}"
        )
    return {
        "id": row["record_id"],
        "messages": messages[:-1],
        "assistant_content": messages[-1]["content"],
    }


class LayerCapture:
    def __init__(self, layer: torch.nn.Module) -> None:
        self.value: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.value = output[0] if isinstance(output, tuple) else output

    def take(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("Target language layer did not execute")
        value = self.value
        self.value = None
        return value

    def close(self) -> None:
        self.handle.remove()


@contextlib.contextmanager
def zero_adapter(parameters: list[torch.nn.Parameter]):
    saved = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in zip(parameters, saved, strict=True):
                parameter.copy_(value)


def base_forward(
    *,
    model: Any,
    capture: LayerCapture,
    inputs: dict[str, Any],
    parameters: list[torch.nn.Parameter],
) -> tuple[torch.Tensor, torch.Tensor]:
    with zero_adapter(parameters), torch.no_grad():
        outputs = model(**inputs, use_cache=False)
        hidden = capture.take().detach()
    return outputs.logits.detach(), hidden


def _verified_rows(
    plan_path: Path,
    plan: dict[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    specification = plan["outputs"][name]
    path = plan_path.parent / specification["path"]
    if file_sha256(path) != specification["sha256"]:
        raise ValueError(f"Step 5 data hash mismatch: {name}")
    rows = read_jsonl(path)
    if len(rows) != specification["records"]:
        raise ValueError(f"Step 5 data count mismatch: {name}")
    return rows


def _install_adapter(
    *,
    model_bundle: Any,
    arm: dict[str, Any],
    output_dir: Path,
    max_length: int,
    learning_rate: float,
    gradient_accumulation: int,
    seed: int,
    projection_seed: int,
) -> tuple[list[torch.nn.Parameter], Path | None]:
    model = model_bundle.model
    if arm["adapter"] == "tinylora":
        config = TinyLoRATrainingConfig(
            svd_rank=int(arm["svd_rank"]),
            projection_dim=int(arm["projection_dim"]),
            learning_rate=learning_rate,
            max_length=max_length,
            gradient_accumulation_steps=gradient_accumulation,
            train_layers=tuple(arm["train_layers"]),
            gradient_checkpointing=True,
            seed=seed,
            projection_seed=projection_seed,
        )
        basis_path = output_dir / f"{arm['name']}_basis.pt"
        install_tinylora_with_cache(
            model_bundle=model_bundle,
            config=config,
            cache_path=basis_path,
        )
    else:
        basis_path = None
        install_ordinary_lora(
            model,
            train_layers=arm["train_layers"],
            rank=int(arm["lora_rank"]),
        )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters or any(
        parameter.device.type != "cuda" for parameter in parameters
    ):
        raise RuntimeError("Adapter parameters are missing or not resident on CUDA")
    return parameters, basis_path


def _behavior_loss(
    *,
    model: Any,
    processor: Any,
    capture: LayerCapture,
    row: dict[str, Any],
    max_length: int,
    device: torch.device,
    parameters: list[torch.nn.Parameter],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, labels, weights = build_training_batch(
        processor=processor,
        records=[
            training_record(row),
            training_record(row, alternative=True),
        ],
        max_length=max_length,
        device=device,
    )
    base_logits, base_hidden = base_forward(
        model=model,
        capture=capture,
        inputs=inputs,
        parameters=parameters,
    )
    base_log_probabilities = sequence_log_probability(base_logits, labels)
    del base_logits
    outputs = model(**inputs, use_cache=False)
    hidden = capture.take()
    log_probabilities = sequence_log_probability(outputs.logits, labels)
    preference = paired_reference_improvement_loss(
        log_probabilities[:1],
        log_probabilities[1:],
        base_preferred=base_log_probabilities[:1],
        base_alternative=base_log_probabilities[1:],
    )
    preferred_ce = weighted_causal_lm_loss(
        outputs.logits[:1],
        labels[:1],
        weights[:1],
    )
    return (
        preference + OBJECTIVE_CONFIGURATION["preferred_ce_weight"] * preferred_ce,
        hidden,
        labels,
        base_hidden,
    )


def run_reachability(
    *,
    model: Any,
    processor: Any,
    capture: LayerCapture,
    parameters: list[torch.nn.Parameter],
    behavior_rows: list[dict[str, Any]],
    preservation_rows: list[dict[str, Any]],
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    by_objective: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior_rows:
        by_objective[str(row["objective"])].append(row)
    for objective, rows in sorted(by_objective.items()):
        ordered = sorted(rows, key=lambda row: str(row["record_id"]))
        selected.extend(ordered[: min(4, len(ordered))])
    selected.extend(
        sorted(preservation_rows, key=lambda row: str(row["record_id"]))[:8]
    )
    gradients: dict[str, list[torch.Tensor]] = defaultdict(list)
    model.train()
    device = next(model.parameters()).device
    for row in selected:
        model.zero_grad(set_to_none=True)
        if row["kind"] == "behavior":
            loss, _hidden, _labels, _base_hidden = _behavior_loss(
                model=model,
                processor=processor,
                capture=capture,
                row=row,
                max_length=max_length,
                device=device,
                parameters=parameters,
            )
            objective = str(row["objective"])
        else:
            inputs, labels, weights = build_training_batch(
                processor=processor,
                records=[training_record(row)],
                max_length=max_length,
                device=device,
            )
            outputs = model(**inputs, use_cache=False)
            capture.take()
            loss = weighted_causal_lm_loss(outputs.logits, labels, weights)
            objective = "preservation_completion"
        values = torch.autograd.grad(loss, parameters, allow_unused=False)
        gradients[objective].append(
            torch.cat([value.detach().float().reshape(-1).cpu() for value in values])
        )
    means = {
        objective: torch.stack(values).mean(dim=0)
        for objective, values in gradients.items()
    }
    return {
        "format": "tinylora_step5_reachability_v1",
        "selected_records": len(selected),
        "gradient_dimensions": {name: value.numel() for name, value in means.items()},
        "gradient_norms": {name: float(value.norm()) for name, value in means.items()},
        "cosine_matrix": gradient_cosine_matrix(means),
        "seed": seed,
    }


def _adapter_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_adapter(model: Any, state: dict[str, torch.Tensor]) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state):
        raise ValueError("Checkpoint adapter parameter inventory differs")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Checkpoint contains non-finite adapter parameters")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(state[name].to(parameter.device))


def train_arm(
    *,
    model: Any,
    processor: Any,
    capture: LayerCapture,
    parameters: list[torch.nn.Parameter],
    rows: list[dict[str, Any]],
    direction: torch.Tensor,
    intercept: float,
    desired_delta: float,
    max_steps: int,
    max_length: int,
    gradient_accumulation: int,
    learning_rate: float,
    checkpoint_every: int,
    checkpoint_minutes: float,
    checkpoint_root: Path,
    identity: dict[str, Any],
    durability_verifier: CheckpointDurabilityVerifier,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, foreach=False)
    optimizer_steps = 0
    next_example = 0
    history: list[dict[str, Any]] = []
    latest_pointer = checkpoint_root / "latest.json"
    if latest_pointer.exists():
        latest = resolve_latest_checkpoint(
            checkpoint_root,
            expected_identity=identity,
        )
        state = torch.load(
            latest.path / "step5_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state.get("format") != "tinylora_step5_checkpoint_v2":
            raise ValueError("Unsupported Step 5 checkpoint format")
        if state.get("identity") != identity:
            raise ValueError("Step 5 checkpoint identity differs")
        optimizer_steps = int(state["optimizer_steps"])
        if optimizer_steps > max_steps:
            raise ValueError(
                "Checkpoint optimizer step exceeds requested budget: "
                f"{optimizer_steps} > {max_steps}"
            )
        _restore_adapter(model, state["adapter_state"])
        optimizer.load_state_dict(state["optimizer"])
        next_example = int(state["next_example"])
        history = list(state["history"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["cpu_rng_state"])
        resume_device = next(model.parameters()).device
        if resume_device.type == "cuda":
            torch.cuda.set_rng_state(
                state["cuda_rng_state"],
                device=resume_device,
            )
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    device = next(model.parameters()).device
    model.train()
    last_checkpoint = time.monotonic()
    while optimizer_steps < max_steps:
        row = rows[next_example % len(rows)]
        next_example += 1
        directional = torch.zeros((), device=device)
        preservation = torch.zeros((), device=device)
        behavior = torch.zeros((), device=device)
        if row["kind"] == "behavior":
            need_direction = row["objective"] == DIRECTIONAL_OBJECTIVE
            behavior, student_hidden, labels, base_hidden = _behavior_loss(
                model=model,
                processor=processor,
                capture=capture,
                row=row,
                max_length=max_length,
                device=device,
                parameters=parameters,
            )
            if need_direction:
                directional = directional_margin_loss(
                    assistant_probe_score(
                        student_hidden[:1],
                        labels[:1],
                        direction,
                        intercept,
                    ),
                    assistant_probe_score(
                        base_hidden[:1], labels[:1], direction, intercept
                    ),
                    desired_delta=desired_delta,
                )
        else:
            inputs, labels, _weights = build_training_batch(
                processor=processor,
                records=[training_record(row)],
                max_length=max_length,
                device=device,
            )
            base_logits, _base_hidden = base_forward(
                model=model,
                capture=capture,
                inputs=inputs,
                parameters=parameters,
            )
            targets = causal_preservation_targets(
                base_logits,
                labels,
            )
            del base_logits
            outputs = model(**inputs, use_cache=False)
            capture.take()
            preservation = topk_preservation_kl_loss(
                outputs.logits[:, :-1, :],
                *targets,
            )
        total = (
            behavior
            + OBJECTIVE_CONFIGURATION["directional_margin_weight"] * directional
            + OBJECTIVE_CONFIGURATION["preservation_kl_weight"] * preservation
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite loss for training record {row['record_id']}"
            )
        (total / gradient_accumulation).backward()
        accumulation += 1
        history.append(
            {
                "record_id": row["record_id"],
                "objective": row["objective"],
                "behavior": float(behavior.detach().cpu()),
                "directional": float(directional.detach().cpu()),
                "preservation": float(preservation.detach().cpu()),
                "total": float(total.detach().cpu()),
            }
        )
        if accumulation < gradient_accumulation:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite gradient for training record {row['record_id']}"
            )
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError("Optimizer produced non-finite adapter parameters")
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        optimizer_steps += 1
        checkpoint_due = (
            optimizer_steps % checkpoint_every == 0
            or optimizer_steps == max_steps
            or time.monotonic() - last_checkpoint >= checkpoint_minutes * 60
        )
        if checkpoint_due:
            publish_checkpoint_generation(
                checkpoint_root=checkpoint_root,
                identity=identity,
                durability_verifier=durability_verifier,
                state={
                    "format": "tinylora_step5_checkpoint_v2",
                    "identity": identity,
                    "optimizer_steps": optimizer_steps,
                    "next_example": next_example,
                    "adapter_state": _adapter_state(model),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state(device=device)
                        if device.type == "cuda"
                        else None
                    ),
                    "gradient_accumulation_position": 0,
                },
            )
            last_checkpoint = time.monotonic()
    final_generation = resolve_latest_checkpoint(
        checkpoint_root,
        expected_identity=identity,
    )
    final_state = torch.load(
        final_generation.path / "step5_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(final_state.get("optimizer_steps", -1)) != max_steps:
        raise CheckpointIntegrityError(
            "Final durable checkpoint does not match the completed training budget"
        )
    receipt_path = (
        checkpoint_root / "receipts" / f"{final_generation.generation_id}.json"
    )
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(
            "Final checkpoint lacks a readable durability receipt"
        ) from exc
    receipt = _validated_durability_receipt(final_generation, receipt)
    return {
        "optimizer_steps": optimizer_steps,
        "training_examples_consumed": next_example,
        "history_tail": history[-50:],
        "durable_checkpoint": receipt,
    }


def evaluate_arm(
    *,
    model: Any,
    processor: Any,
    capture: LayerCapture,
    parameters: list[torch.nn.Parameter],
    behavior_rows: list[dict[str, Any]],
    preservation_rows: list[dict[str, Any]],
    direction: torch.Tensor,
    intercept: float,
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    behavior_results: list[dict[str, Any]] = []
    preservation_results: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in behavior_rows:
            inputs, labels, _weights = build_training_batch(
                processor=processor,
                records=[
                    training_record(row),
                    training_record(row, alternative=True),
                ],
                max_length=max_length,
                device=device,
            )
            base_logits, base_hidden = base_forward(
                model=model,
                capture=capture,
                inputs=inputs,
                parameters=parameters,
            )
            base_logp = sequence_log_probability(base_logits, labels)
            del base_logits
            outputs = model(**inputs, use_cache=False)
            student_hidden = capture.take()
            student_logp = sequence_log_probability(outputs.logits, labels)
            base_margin = float((base_logp[0] - base_logp[1]).cpu())
            student_margin = float((student_logp[0] - student_logp[1]).cpu())
            movement = None
            if row["objective"] == DIRECTIONAL_OBJECTIVE:
                movement = float(
                    (
                        assistant_probe_score(
                            student_hidden[:1],
                            labels[:1],
                            direction,
                            intercept,
                        )
                        - assistant_probe_score(
                            base_hidden[:1],
                            labels[:1],
                            direction,
                            intercept,
                        )
                    )
                    .cpu()
                    .item()
                )
            behavior_results.append(
                {
                    "record_id": row["record_id"],
                    "scenario_id": row["scenario_id"],
                    "family": row["family"],
                    "objective": row["objective"],
                    "base_margin": base_margin,
                    "student_margin": student_margin,
                    "improvement": student_margin - base_margin,
                    "direction_movement": movement,
                }
            )
        for row in preservation_rows:
            inputs, labels, _weights = build_training_batch(
                processor=processor,
                records=[training_record(row)],
                max_length=max_length,
                device=device,
            )
            base_logits, _base_hidden = base_forward(
                model=model,
                capture=capture,
                inputs=inputs,
                parameters=parameters,
            )
            targets = causal_preservation_targets(
                base_logits,
                labels,
            )
            del base_logits
            outputs = model(**inputs, use_cache=False)
            capture.take()
            loss = topk_preservation_kl_loss(
                outputs.logits[:, :-1, :],
                *targets,
            )
            preservation_results.append(
                {
                    "record_id": row["record_id"],
                    "category": row["preservation_category"],
                    "kl": float(loss.cpu()),
                }
            )
    summaries: dict[str, Any] = {}
    for objective in sorted({row["objective"] for row in behavior_results}):
        selected = [row for row in behavior_results if row["objective"] == objective]
        values = [row["improvement"] for row in selected]
        summaries[objective] = {
            "by_scenario": clustered_bootstrap_mean(
                values,
                [row["scenario_id"] for row in selected],
                seed=seed,
            ),
            "by_family": clustered_bootstrap_mean(
                values,
                [row["family"] for row in selected],
                seed=seed + 1,
            ),
        }
    return {
        "behavior": behavior_results,
        "behavior_summary": summaries,
        "preservation": preservation_results,
        "preservation_kl_mean": (
            sum(row["kl"] for row in preservation_results) / len(preservation_results)
            if preservation_results
            else None
        ),
    }


def main() -> int:
    args = parse_args()
    validate_numeric_args(args)
    validate_durability_args(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each Step 5 arm requires exactly one visible CUDA GPU")
    if args.mode == "smoke" and args.max_steps > 2:
        raise ValueError("Smoke mode is capped at two optimizer steps")
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("format") != "tinylora_step5_plan_v1"
        or plan.get("large_run_enabled")
        or plan.get("paid_execution_enabled")
    ):
        raise ValueError("Refusing an unsupported or execution-enabled Step 5 plan")
    arms = {arm["name"]: arm for arm in plan["arms"]}
    if args.arm not in arms:
        raise ValueError(f"Unknown Step 5 arm: {args.arm}")
    arm = arms[args.arm]
    behavior_train = _verified_rows(plan_path, plan, "train_behavior")
    preservation_train = _verified_rows(plan_path, plan, "preservation_train")
    development_behavior = [
        *_verified_rows(plan_path, plan, "development_iid"),
        *_verified_rows(plan_path, plan, "development_heldout_family"),
    ]
    preservation_development = [
        *_verified_rows(plan_path, plan, "preservation_development_text"),
        *_verified_rows(plan_path, plan, "preservation_development_vision"),
    ]
    if args.development_per_objective:
        selected: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in development_behavior:
            grouped[str(row["objective"])].append(row)
        for rows in grouped.values():
            selected.extend(
                sorted(rows, key=lambda row: str(row["record_id"]))[
                    : args.development_per_objective
                ]
            )
        development_behavior = selected
    schedule = preservation_interleaved_schedule(
        behavior_train,
        preservation_train,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_specification = plan["model"]
    if not model_specification.get("revision"):
        raise ValueError("Step 5 requires an exact model revision")
    seed_all(args.seed)
    bundle = load_model_and_processor(
        ModelLoadConfig(
            model_name=model_specification["model_id"],
            revision=model_specification["revision"],
            cache_dir=str(args.cache_dir),
            attention_implementation="flash_attention_2",
            device_map={"": 0},
        )
    )
    model = bundle.model
    if model is None:
        raise RuntimeError("Model weights were not loaded")
    seed_all(args.projection_seed)
    parameters, basis_path = _install_adapter(
        model_bundle=bundle,
        arm=arm,
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
        seed=args.seed,
        projection_seed=args.projection_seed,
    )
    seed_all(args.seed)
    if vars(model.config).get("use_cache", False):
        vars(model.config)["use_cache"] = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    layer_index, direction, intercept, probe_metadata = load_probe(args.probe)
    if layer_index not in arm["train_layers"]:
        raise ValueError("Probe layer differs from the Step 5 arm")
    capture = LayerCapture(model.model.language_model.layers[layer_index])
    direction = direction.to(next(model.parameters()).device)
    started = time.time()
    code_digest = hashlib.sha256()
    for code_path in (
        Path(__file__).resolve(),
        Path(__file__).parents[1] / "src/intelligent_liars/durable_checkpoints.py",
        Path(__file__).parents[1] / "src/intelligent_liars/tinylora_pilot.py",
        Path(__file__).parents[1] / "src/intelligent_liars/tinylora_step5.py",
    ):
        code_digest.update(file_sha256(code_path).encode())
    identity = build_checkpoint_identity(
        plan_sha256=file_sha256(plan_path),
        probe_sha256=file_sha256(args.probe),
        code_sha256=code_digest.hexdigest(),
        basis_sha256=(file_sha256(basis_path) if basis_path else None),
        arm=arm,
        model=model_specification,
        mode=args.mode,
        max_steps=args.max_steps,
        seed=args.seed,
        projection_seed=args.projection_seed,
        max_length=args.max_length,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        runtime_image_digest=(
            args.runtime_image_digest or "local-smoke-image-unpinned"
        ),
        schedule_sha256=schedule_sha256(schedule),
    )
    if args.mode == "reachability":
        result = run_reachability(
            model=model,
            processor=bundle.processor,
            capture=capture,
            parameters=parameters,
            behavior_rows=behavior_train,
            preservation_rows=preservation_train,
            max_length=args.max_length,
            seed=args.seed,
        )
    else:
        target_rows = [
            row for row in behavior_train if row["objective"] == DIRECTIONAL_OBJECTIVE
        ][:16]
        scores: list[float] = []
        for row in target_rows:
            inputs, labels, _weights = build_training_batch(
                processor=bundle.processor,
                records=[training_record(row)],
                max_length=args.max_length,
                device=next(model.parameters()).device,
            )
            _logits, hidden = base_forward(
                model=model,
                capture=capture,
                inputs=inputs,
                parameters=parameters,
            )
            del _logits
            scores.append(
                float(assistant_probe_score(hidden, labels, direction, intercept).cpu())
            )
        values = torch.tensor(scores)
        q1, q3 = torch.quantile(values, torch.tensor([0.25, 0.75]))
        desired_delta = 0.5 * max(float((q3 - q1) / 1.349), 0.1)
        durability_verifier = (
            command_durability_verifier(args.durability_verifier_command)
            if args.durability_verifier_command
            else local_smoke_durability_verifier
        )
        training = train_arm(
            model=model,
            processor=bundle.processor,
            capture=capture,
            parameters=parameters,
            rows=schedule,
            direction=direction,
            intercept=intercept,
            desired_delta=desired_delta,
            max_steps=args.max_steps,
            max_length=args.max_length,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            checkpoint_every=args.checkpoint_every,
            checkpoint_minutes=args.checkpoint_minutes,
            checkpoint_root=args.output_dir / "checkpoint_store",
            identity=identity,
            durability_verifier=durability_verifier,
        )
        evaluation = evaluate_arm(
            model=model,
            processor=bundle.processor,
            capture=capture,
            parameters=parameters,
            behavior_rows=development_behavior,
            preservation_rows=preservation_development,
            direction=direction,
            intercept=intercept,
            max_length=args.max_length,
            seed=args.seed,
        )
        result = {
            "format": "tinylora_step5_arm_result_v1",
            "mode": args.mode,
            "training": training,
            "evaluation": evaluation,
            "desired_direction_delta": desired_delta,
        }
    capture.close()
    result.update(
        {
            "arm": arm,
            "identity": identity,
            "trainable_scalars": sum(parameter.numel() for parameter in parameters),
            "basis_sha256": file_sha256(basis_path) if basis_path else None,
            "attention_configured": getattr(model.config, "_attn_implementation", None),
            "probe_sign": probe_metadata["direction_sign_convention"],
            "gpu": torch.cuda.get_device_name(0),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": time.time() - started,
            "large_run": False,
        }
    )
    atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
