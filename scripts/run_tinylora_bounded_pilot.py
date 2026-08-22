#!/usr/bin/env python3
"""Run one bounded rank candidate with behavior, direction, and preservation losses."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

from intelligent_liars.models import ModelLoadConfig, load_model_and_processor
from intelligent_liars.standalone_models import (
    TinyLoRATrainingConfig,
    build_training_batch,
    install_tinylora_with_cache,
    weighted_causal_lm_loss,
)
from intelligent_liars.tinylora_pilot import (
    DIRECTIONAL_OBJECTIVE,
    ObjectiveWeights,
    assistant_probe_score,
    directional_margin_loss,
    file_sha256,
    select_stratified_rows,
    stable_score,
    topk_preservation_kl_loss,
    topk_preservation_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/cache/huggingface"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--revision")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--calibration-examples", type=int, default=16)
    parser.add_argument("--development-examples", type=int, default=48)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_record(row: dict[str, Any]) -> dict[str, Any]:
    if row["kind"] == "behavior":
        return {
            "id": row["record_id"],
            "messages": [{"role": "user", "content": row["prompt"]}],
            "assistant_content": row["target"],
        }
    messages = list(row["messages"])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"Preservation row lacks a final assistant turn: {row['record_id']}")
    return {
        "id": row["record_id"],
        "messages": messages[:-1],
        "assistant_content": messages[-1]["content"],
    }


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
    if vector.numel() != int(direction["feature_count"]):
        raise ValueError("Probe vector length does not match feature_count")
    return int(direction["layer"]), vector, float(direction["intercept"]), direction


@contextlib.contextmanager
def zero_tinylora(trainable_vector: torch.nn.Parameter):
    saved = trainable_vector.detach().clone()
    with torch.no_grad():
        trainable_vector.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            trainable_vector.copy_(saved)


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


def _base_forward(
    model: Any,
    capture: LayerCapture,
    inputs: dict[str, Any],
    trainable_vector: torch.nn.Parameter,
    *,
    need_hidden: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    with zero_tinylora(trainable_vector), torch.no_grad():
        outputs = model(**inputs, use_cache=False)
        hidden = capture.take().detach() if need_hidden else None
        if not need_hidden:
            capture.take()
    return outputs.logits.detach(), hidden


def calibrate_delta(
    *,
    model: Any,
    processor: Any,
    capture: LayerCapture,
    trainable_vector: torch.nn.Parameter,
    rows: list[dict[str, Any]],
    direction: torch.Tensor,
    intercept: float,
    max_length: int,
    count: int,
) -> tuple[float, list[float]]:
    scores: list[float] = []
    for row in rows[:count]:
        inputs, labels, _ = build_training_batch(
            processor=processor,
            records=[training_record(row)],
            max_length=max_length,
            device=next(model.parameters()).device,
        )
        _, hidden = _base_forward(
            model,
            capture,
            inputs,
            trainable_vector,
            need_hidden=True,
        )
        assert hidden is not None
        scores.append(float(assistant_probe_score(hidden, labels, direction, intercept).cpu()))
    values = torch.tensor(scores, dtype=torch.float32)
    q1, q3 = torch.quantile(values, torch.tensor([0.25, 0.75]))
    robust_sigma = float((q3 - q1) / 1.349)
    return 0.5 * max(robust_sigma, 0.1), scores


def evaluate_losses(
    *,
    model: Any,
    processor: Any,
    capture: LayerCapture,
    trainable_vector: torch.nn.Parameter,
    rows: list[dict[str, Any]],
    direction: torch.Tensor,
    intercept: float,
    max_length: int,
) -> dict[str, Any]:
    result = {
        "records": 0,
        "behavior_ce": [],
        "base_behavior_ce": [],
        "direction_movement": [],
        "preservation_kl": [],
        "by_objective": {},
    }
    model.eval()
    for row in rows:
        inputs, labels, weights = build_training_batch(
            processor=processor,
            records=[training_record(row)],
            max_length=max_length,
            device=next(model.parameters()).device,
        )
        need_hidden = row.get("objective") == DIRECTIONAL_OBJECTIVE
        base_logits, base_hidden = _base_forward(
            model, capture, inputs, trainable_vector, need_hidden=need_hidden
        )
        base_behavior: torch.Tensor | None = None
        preservation_targets: tuple[torch.Tensor, torch.Tensor] | None = None
        if row["kind"] == "preservation":
            preservation_targets = topk_preservation_targets(base_logits)
        else:
            base_behavior = weighted_causal_lm_loss(base_logits, labels, weights)
        del base_logits
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
            student_hidden = capture.take()
            if row["kind"] == "preservation":
                assert preservation_targets is not None
                loss = topk_preservation_kl_loss(
                    outputs.logits,
                    *preservation_targets,
                    inputs["attention_mask"],
                )
                result["preservation_kl"].append(float(loss.cpu()))
            else:
                loss = weighted_causal_lm_loss(outputs.logits, labels, weights)
                assert base_behavior is not None
                result["behavior_ce"].append(float(loss.cpu()))
                result["base_behavior_ce"].append(float(base_behavior.cpu()))
                objective = str(row["objective"])
                objective_values = result["by_objective"].setdefault(
                    objective,
                    {"base_behavior_ce": [], "student_behavior_ce": []},
                )
                objective_values["base_behavior_ce"].append(float(base_behavior.cpu()))
                objective_values["student_behavior_ce"].append(float(loss.cpu()))
                if need_hidden:
                    assert base_hidden is not None
                    movement = assistant_probe_score(
                        student_hidden, labels, direction, intercept
                    ) - assistant_probe_score(base_hidden, labels, direction, intercept)
                    result["direction_movement"].append(float(movement.cpu()))
        result["records"] += 1
    model.train()
    by_objective = {}
    for objective, values in result["by_objective"].items():
        base_mean = sum(values["base_behavior_ce"]) / len(values["base_behavior_ce"])
        student_mean = sum(values["student_behavior_ce"]) / len(
            values["student_behavior_ce"]
        )
        by_objective[objective] = {
            "count": len(values["student_behavior_ce"]),
            "base_behavior_ce_mean": base_mean,
            "student_behavior_ce_mean": student_mean,
            "student_minus_base_ce": student_mean - base_mean,
        }
    return {
        "records": result["records"],
        "by_objective": by_objective,
        **{
            key: {"count": len(values), "mean": sum(values) / len(values) if values else None}
            for key, values in result.items()
            if isinstance(values, list)
        },
    }


def _normalized_answer(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold()))


def _generate_one(model: Any, processor: Any, prompt: str) -> str:
    tokenizer = processor.tokenizer
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
        )
    generated = output[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def evaluate_generation(
    *,
    model: Any,
    processor: Any,
    capture: LayerCapture,
    trainable_vector: torch.nn.Parameter,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_objective: dict[str, dict[str, int]] = {}
    examples: list[dict[str, Any]] = []
    config_values = vars(model.config)
    had_use_cache = "use_cache" in config_values
    previous_use_cache = config_values.get("use_cache")
    config_values["use_cache"] = True
    model.eval()
    try:
        for row in rows:
            target = _normalized_answer(str(row["target"]))
            with zero_tinylora(trainable_vector):
                base_text = _generate_one(model, processor, str(row["prompt"]))
            capture.value = None
            student_text = _generate_one(model, processor, str(row["prompt"]))
            capture.value = None
            base_match = target in _normalized_answer(base_text)
            student_match = target in _normalized_answer(student_text)
            counts = by_objective.setdefault(
                str(row["objective"]),
                {"records": 0, "base_target_matches": 0, "student_target_matches": 0},
            )
            counts["records"] += 1
            counts["base_target_matches"] += int(base_match)
            counts["student_target_matches"] += int(student_match)
            examples.append(
                {
                    "record_id": row["record_id"],
                    "objective": row["objective"],
                    "target": row["target"],
                    "base": base_text,
                    "student": student_text,
                    "base_target_match": base_match,
                    "student_target_match": student_match,
                }
            )
    finally:
        if had_use_cache:
            config_values["use_cache"] = previous_use_cache
        else:
            config_values.pop("use_cache", None)
        model.train()
    return {"by_objective": by_objective, "examples": examples}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The bounded pilot requires exactly one visible CUDA GPU")
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("format") != "tinylora_bounded_pilot_plan_v1" or plan.get("large_run_enabled"):
        raise ValueError("Refusing a missing, unsupported, or large-run-enabled plan")
    if args.max_steps > int(plan["pilot"]["max_optimizer_steps"]):
        raise ValueError("Requested optimizer steps exceed the bounded pilot plan")
    weights = ObjectiveWeights(**plan["pilot"]["objective_weights"])
    weights.validate()
    layer_index, direction, intercept, probe_metadata = load_probe(args.probe)
    if layer_index not in plan["pilot"]["train_layers"]:
        raise ValueError("Probe layer and plan train layer differ")
    train_path = plan_path.parent / plan["outputs"]["train"]["path"]
    development_path = plan_path.parent / plan["outputs"]["development"]["path"]
    if file_sha256(train_path) != plan["outputs"]["train"]["sha256"]:
        raise ValueError("Training data hash does not match plan")
    if file_sha256(development_path) != plan["outputs"]["development"]["sha256"]:
        raise ValueError("Development data hash does not match plan")
    train_rows = read_jsonl(train_path)
    development_rows = read_jsonl(development_path)
    target_rows = sorted(
        [row for row in train_rows if row.get("objective") == DIRECTIONAL_OBJECTIVE],
        key=lambda row: stable_score(args.seed, str(row["record_id"])),
    )
    ordered_train = sorted(
        train_rows,
        key=lambda row: stable_score(args.seed + args.rank, str(row["record_id"])),
    )
    objective_count = len({str(row["objective"]) for row in development_rows})
    if args.development_examples % objective_count:
        raise ValueError("development_examples must divide evenly across objectives")
    ordered_development = select_stratified_rows(
        development_rows,
        per_objective=args.development_examples // objective_count,
        seed=args.seed,
    )
    generation_development = select_stratified_rows(
        development_rows,
        per_objective=4,
        seed=args.seed + 1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_model_and_processor(
        ModelLoadConfig(
            model_name=args.model,
            cache_dir=str(args.cache_dir),
            attention_implementation="flash_attention_2",
            device_map={"": 0},
            revision=args.revision,
        )
    )
    model = bundle.model
    if model is None:
        raise RuntimeError("Model weights were not loaded")
    config = TinyLoRATrainingConfig(
        svd_rank=args.rank,
        projection_dim=int(plan["pilot"]["projection_dim"]),
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation,
        train_layers=(layer_index,),
        gradient_checkpointing=True,
        seed=args.seed,
    )
    basis_path = args.output_dir / f"tinylora_rank{args.rank}_basis.pt"
    installed = install_tinylora_with_cache(
        model_bundle=bundle,
        config=config,
        cache_path=basis_path,
    )
    vectors = {id(item.module.tinylora_v): item.module.tinylora_v for item in installed}
    if len(vectors) != 1:
        raise RuntimeError("Pilot requires one fully tied TinyLoRA vector")
    trainable_vector = next(iter(vectors.values()))
    if vars(model.config).get("use_cache", False):
        vars(model.config)["use_cache"] = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    capture = LayerCapture(model.model.language_model.layers[layer_index])
    direction = direction.to(next(model.parameters()).device)
    desired_delta, calibration_scores = calibrate_delta(
        model=model,
        processor=bundle.processor,
        capture=capture,
        trainable_vector=trainable_vector,
        rows=target_rows,
        direction=direction,
        intercept=intercept,
        max_length=args.max_length,
        count=args.calibration_examples,
    )
    optimizer = torch.optim.AdamW([trainable_vector], lr=args.learning_rate, foreach=False)
    checkpoint_path = args.output_dir / "pilot_state.pt"
    optimizer_steps = 0
    next_example = 0
    skipped_examples: list[dict[str, str]] = []
    history: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("plan_sha256") != file_sha256(plan_path) or state.get("rank") != args.rank:
            raise ValueError("Checkpoint identity differs from this pilot")
        with torch.no_grad():
            trainable_vector.copy_(state["tinylora_vector"].to(trainable_vector.device))
        optimizer.load_state_dict(state["optimizer"])
        optimizer_steps = int(state["optimizer_steps"])
        next_example = int(state["next_example"])
        history = list(state.get("history", []))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    started = time.time()
    while optimizer_steps < args.max_steps:
        row = ordered_train[next_example % len(ordered_train)]
        next_example += 1
        try:
            inputs, labels, sample_weights = build_training_batch(
                processor=bundle.processor,
                records=[training_record(row)],
                max_length=args.max_length,
                device=next(model.parameters()).device,
            )
        except ValueError as error:
            if "exceed max_length" not in str(error):
                raise
            skipped_examples.append(
                {"record_id": str(row["record_id"]), "reason": str(error)}
            )
            continue
        need_hidden = row.get("objective") == DIRECTIONAL_OBJECTIVE
        need_base = need_hidden or row["kind"] == "preservation"
        base_logits: torch.Tensor | None = None
        base_hidden: torch.Tensor | None = None
        preservation_targets: tuple[torch.Tensor, torch.Tensor] | None = None
        if need_base:
            base_logits, base_hidden = _base_forward(
                model, capture, inputs, trainable_vector, need_hidden=need_hidden
            )
            if row["kind"] == "preservation":
                preservation_targets = topk_preservation_targets(base_logits)
            del base_logits
            base_logits = None
        outputs = model(**inputs, use_cache=False)
        student_hidden = capture.take()
        behavior = torch.zeros((), device=student_hidden.device)
        directional = torch.zeros_like(behavior)
        preservation = torch.zeros_like(behavior)
        if row["kind"] == "behavior":
            behavior = weighted_causal_lm_loss(outputs.logits, labels, sample_weights)
            if need_hidden:
                assert base_hidden is not None
                directional = directional_margin_loss(
                    assistant_probe_score(student_hidden, labels, direction, intercept),
                    assistant_probe_score(base_hidden, labels, direction, intercept),
                    desired_delta=desired_delta,
                )
        else:
            assert preservation_targets is not None
            preservation = topk_preservation_kl_loss(
                outputs.logits,
                *preservation_targets,
                inputs["attention_mask"],
            )
        total = (
            weights.behavior_ce * behavior
            + weights.directional_margin * directional
            + weights.preservation_kl * preservation
        )
        (total / args.gradient_accumulation).backward()
        accumulation += 1
        history.append(
            {
                "record_id": row["record_id"],
                "objective": row["objective"],
                "behavior_ce": float(behavior.detach().cpu()),
                "directional_margin": float(directional.detach().cpu()),
                "preservation_kl": float(preservation.detach().cpu()),
                "total": float(total.detach().cpu()),
            }
        )
        if accumulation < args.gradient_accumulation:
            continue
        torch.nn.utils.clip_grad_norm_([trainable_vector], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        optimizer_steps += 1
        if optimizer_steps % args.checkpoint_every == 0 or optimizer_steps == args.max_steps:
            atomic_torch_save(
                checkpoint_path,
                {
                    "format": "tinylora_bounded_pilot_state_v1",
                    "plan_sha256": file_sha256(plan_path),
                    "probe_sha256": file_sha256(args.probe),
                    "rank": args.rank,
                    "optimizer_steps": optimizer_steps,
                    "next_example": next_example,
                    "tinylora_vector": trainable_vector.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
            )
    development = evaluate_losses(
        model=model,
        processor=bundle.processor,
        capture=capture,
        trainable_vector=trainable_vector,
        rows=ordered_development,
        direction=direction,
        intercept=intercept,
        max_length=args.max_length,
    )
    generation = evaluate_generation(
        model=model,
        processor=bundle.processor,
        capture=capture,
        trainable_vector=trainable_vector,
        rows=generation_development,
    )
    capture.close()
    result = {
        "format": "tinylora_bounded_pilot_result_v1",
        "large_run": False,
        "rank": args.rank,
        "trainable_scalars": trainable_vector.numel(),
        "optimizer_steps": optimizer_steps,
        "training_examples_consumed": next_example,
        "skipped_examples": skipped_examples,
        "elapsed_seconds": time.time() - started,
        "plan_sha256": file_sha256(plan_path),
        "probe_sha256": file_sha256(args.probe),
        "probe": {
            "layer": layer_index,
            "sign": probe_metadata["direction_sign_convention"],
            "desired_delta": desired_delta,
            "calibration_scores": calibration_scores,
        },
        "objective_weights": weights.__dict__,
        "development": development,
        "preservation_evaluation_scope": (
            "training-objective-only; the immutable pilot development split contains "
            "no held-out preservation rows"
        ),
        "development_generation": generation,
        "attention_configured": getattr(model.config, "_attn_implementation", None),
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "basis_sha256": file_sha256(basis_path),
        "state_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
