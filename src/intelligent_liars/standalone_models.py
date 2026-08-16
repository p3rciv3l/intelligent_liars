from __future__ import annotations

import hashlib
import json
import math
import os
import random
import gc
import re
import socket
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from intelligent_liars.interventions import (
    DirectionMode,
    InterventionMethod,
    RuntimeIntervention,
    load_intervention_bundle,
)
from intelligent_liars.models import DEFAULT_MODEL_ID, ModelBundle, ModelLoadConfig
from intelligent_liars.rollouts import (
    GeneratedCompletion,
    GenerationSettings,
    Message,
    _render_generation_conversation,
    generate_completions,
    seed_everything,
    split_qwen_thinking,
    write_json_atomic,
)


TEACHER_DATASET_FORMAT = "qwen_intervention_teacher_v1"
STANDALONE_MODEL_FORMAT = "qwen_intervention_student_v1"
FLEET_PLAN_FORMAT = "qwen_intervention_fleet_plan_v1"
DEFAULT_LORA_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class TeacherDatasetSummary:
    output_path: Path
    records: int
    intervention: str | None


@dataclass(frozen=True)
class LoRATrainingConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 4096
    max_grad_norm: float = 1.0
    seed: int = 0
    checkpoint_every_steps: int = 50
    preservation_weight: float = 1.0
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    train_layers: tuple[int, ...] | None = None
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class StandaloneModelSummary:
    output_dir: Path
    optimizer_steps: int
    examples: int
    merged_modules: tuple[str, ...]


@dataclass(frozen=True)
class FleetVariant:
    name: str
    intervention_path: str
    intervention_sha256: str
    teacher_output: str
    model_output: str


@dataclass(frozen=True)
class FleetPlan:
    plan_id: str
    prompt_path: str
    prompt_sha256: str
    preservation_teacher_path: str
    preservation_teacher_sha256: str
    output_root: str
    base_control_output: str
    variants: tuple[FleetVariant, ...]
    generation: GenerationSettings
    training: LoRATrainingConfig
    base_model: str = DEFAULT_MODEL_ID
    base_revision: str = ""


@dataclass(frozen=True)
class FleetClaim:
    variant: FleetVariant
    marker_path: Path
    worker_id: str
    claim_token: str
    attempt: int


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not hasattr(base, "weight"):
            raise TypeError("LoRA target must expose a weight tensor")
        weight = base.weight
        if weight.ndim != 2:
            raise TypeError("LoRA target weight must be a matrix")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(
            torch.empty(rank, weight.shape[1], dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(weight.shape[0], rank, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        update = F.linear(
            F.linear(self.dropout(inputs).float(), self.lora_a), self.lora_b
        )
        return base_output + (self.scaling * update).to(dtype=base_output.dtype)

    def merged_weight(self) -> torch.Tensor:
        update = self.lora_b @ self.lora_a
        return self.base.weight + (self.scaling * update).to(
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )


@dataclass(frozen=True)
class InstalledLoRA:
    name: str
    parent: nn.Module
    attribute: str
    module: LoRALinear


def load_prompt_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = (
        payload.get("records", payload.get("rollouts"))
        if isinstance(payload, Mapping)
        else payload
    )
    if not isinstance(rows, list):
        raise ValueError(
            "Prompt data must be a JSON list or an object with a records list"
        )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Prompt row {index} must be an object")
        messages = row.get("messages", row.get("input_messages"))
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Prompt row {index} must contain non-empty messages")
        metadata = dict(row.get("metadata", {}))
        record_id = str(
            row.get("id", row.get("source_index", metadata.get("source_index", index)))
        )
        if record_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {record_id}")
        seen_ids.add(record_id)
        records.append(
            {
                "id": record_id,
                "messages": [_normalise_message(message) for message in messages],
                "metadata": metadata,
            }
        )
    return records


def validate_lora_training_config(config: LoRATrainingConfig) -> None:
    positive_ints = {
        "rank": config.rank,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length": config.max_length,
    }
    invalid_ints = [name for name, value in positive_ints.items() if value < 1]
    if invalid_ints:
        raise ValueError(f"LoRA training settings must be positive: {invalid_ints}")
    if config.checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be non-negative")
    numeric = {
        "alpha": config.alpha,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "max_grad_norm": config.max_grad_norm,
        "preservation_weight": config.preservation_weight,
    }
    non_finite = [name for name, value in numeric.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"LoRA training settings must be finite: {non_finite}")
    if config.alpha <= 0 or config.learning_rate <= 0 or config.max_grad_norm <= 0:
        raise ValueError(
            "LoRA alpha, learning_rate, and max_grad_norm must be positive"
        )
    if not 0 <= config.dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if config.preservation_weight <= 0:
        raise ValueError("preservation_weight must be positive")
    if not config.target_modules:
        raise ValueError("At least one LoRA target module is required")
    if config.train_layers is not None and len(set(config.train_layers)) != len(
        config.train_layers
    ):
        raise ValueError("LoRA train layers must be unique")


def generate_teacher_dataset(
    *,
    model_bundle: ModelBundle,
    prompt_path: Path,
    output_path: Path,
    generation: GenerationSettings,
    intervention_path: Path | None = None,
    resume: bool = True,
) -> TeacherDatasetSummary:
    prompts = load_prompt_records(prompt_path)
    intervention = (
        load_intervention_bundle(intervention_path)
        if intervention_path is not None
        else None
    )
    metadata = {
        "base_model": model_bundle.model_id,
        "base_revision": _bundle_revision(model_bundle),
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": _sha256_file(prompt_path),
        "intervention_path": (
            str(intervention_path.resolve()) if intervention_path is not None else None
        ),
        "intervention_sha256": (
            _sha256_file(intervention_path) if intervention_path is not None else None
        ),
        "generation": asdict(generation),
    }
    output = _load_or_initialize_teacher_output(
        output_path=output_path,
        metadata=metadata,
        resume=resume,
    )
    completed = {str(row["id"]) for row in output["records"]}
    pending = [row for row in prompts if row["id"] not in completed]
    for batch in _chunks(pending, generation.batch_size):
        seed_everything(
            _stable_seed(generation.seed, *(str(row["id"]) for row in batch))
        )
        context = (
            RuntimeIntervention(model_bundle.model, intervention)
            if intervention is not None
            else nullcontext()
        )
        with context:
            completions = generate_teacher_completions(
                bundle=model_bundle,
                conversations=[row["messages"] for row in batch],
                settings=generation,
            )
        for row, completion in zip(batch, completions, strict=True):
            output["records"].append(
                {
                    "id": row["id"],
                    "messages": row["messages"],
                    "assistant_content": _completion_content(
                        text=completion.text,
                        thinking=completion.thinking,
                        raw_text=completion.raw_text,
                    ),
                    "final_answer": completion.text,
                    "thinking": completion.thinking,
                    "metadata": row["metadata"],
                }
            )
        output["records"].sort(key=lambda item: _natural_id_key(str(item["id"])))
        write_json_atomic(output_path, output)
    return TeacherDatasetSummary(
        output_path=output_path,
        records=len(output["records"]),
        intervention=(
            str(intervention_path) if intervention_path is not None else None
        ),
    )


def generate_teacher_completions(
    *,
    bundle: ModelBundle,
    conversations: Sequence[Sequence[Message]],
    settings: GenerationSettings,
) -> list[GeneratedCompletion]:
    if not any(
        _conversation_has_vision(conversation) for conversation in conversations
    ):
        return generate_completions(
            bundle=bundle,
            conversations=conversations,
            settings=settings,
        )
    if bundle.model is None:
        raise ValueError("Teacher generation requires model weights")
    generation_conversations = [
        _generation_messages(conversation) for conversation in conversations
    ]
    qwen_conversations = [
        [_qwen_message(message) for message in conversation]
        for conversation in generation_conversations
    ]
    texts = [
        _render_generation_conversation(bundle.processor, conversation)
        for conversation in generation_conversations
    ]
    inputs = _processor_inputs(
        processor=bundle.processor,
        conversations=qwen_conversations,
        texts=texts,
        padding=True,
    )
    device = _model_input_device(bundle.model)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(inputs).items()
    }
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
    }
    if settings.do_sample:
        generation_kwargs.update(
            {"temperature": settings.temperature, "top_p": settings.top_p}
        )
        if settings.top_k is not None:
            generation_kwargs["top_k"] = settings.top_k
    outputs = bundle.model.generate(**inputs, **generation_kwargs)
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    decoded = bundle.processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [split_qwen_thinking(text) for text in decoded]


def install_lora(
    model: Any,
    config: LoRATrainingConfig,
) -> list[InstalledLoRA]:
    validate_lora_training_config(config)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.model.language_model.layers
    selected_layers = (
        tuple(range(len(layers)))
        if config.train_layers is None
        else config.train_layers
    )
    invalid = [layer for layer in selected_layers if layer < 0 or layer >= len(layers)]
    if invalid:
        raise ValueError(f"LoRA train layers are outside the model: {invalid}")
    installed: list[InstalledLoRA] = []
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        for target in config.target_modules:
            parent, attribute = _resolve_relative_module(layer, target)
            base = getattr(parent, attribute)
            wrapped = LoRALinear(
                base,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ).to(device=base.weight.device)
            setattr(parent, attribute, wrapped)
            installed.append(
                InstalledLoRA(
                    name=f"model.language_model.layers.{layer_idx}.{target}",
                    parent=parent,
                    attribute=attribute,
                    module=wrapped,
                )
            )
    if not installed:
        raise ValueError("No LoRA modules were installed")
    return installed


def merge_lora(installed: Sequence[InstalledLoRA]) -> tuple[str, ...]:
    merged: list[str] = []
    for item in installed:
        merged_weight = item.module.merged_weight()
        if not torch.isfinite(merged_weight).all():
            raise ValueError(f"LoRA merge produced non-finite weights for {item.name}")
        with torch.no_grad():
            item.module.base.weight.copy_(merged_weight)
        setattr(item.parent, item.attribute, item.module.base)
        merged.append(item.name)
    return tuple(merged)


def train_standalone_model(
    *,
    model_bundle: ModelBundle,
    teacher_path: Path,
    output_dir: Path,
    config: LoRATrainingConfig,
    preservation_teacher_path: Path | None = None,
    resume: bool = True,
    fleet_plan_id: str | None = None,
) -> StandaloneModelSummary:
    validate_lora_training_config(config)
    if model_bundle.model is None:
        raise ValueError("Standalone model training requires model weights")
    teacher_payload = _load_teacher_payload(teacher_path)
    _validate_teacher_compatibility(
        teacher_payload,
        model_bundle=model_bundle,
        require_intervention=True,
    )
    examples = [dict(row, loss_weight=1.0) for row in teacher_payload["records"]]
    preservation_payload = None
    if preservation_teacher_path is not None:
        preservation_payload = _load_teacher_payload(preservation_teacher_path)
        _validate_teacher_compatibility(
            preservation_payload,
            model_bundle=model_bundle,
            require_intervention=False,
        )
        examples.extend(
            dict(row, loss_weight=config.preservation_weight)
            for row in preservation_payload["records"]
        )
    if not examples:
        raise ValueError("Teacher datasets contain no examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_manifest = output_dir / "standalone_model.json"
    if completed_manifest.exists():
        if not resume:
            raise FileExistsError(f"Standalone model is already complete: {output_dir}")
        completed = json.loads(completed_manifest.read_text())
        if not _standalone_output_files_present(output_dir, require_verification=False):
            raise ValueError(
                "Completed standalone model is missing checkpoint artifacts"
            )
        if (
            completed.get("teacher_sha256") != _sha256_file(teacher_path)
            or completed.get("preservation_teacher_sha256")
            != (
                _sha256_file(preservation_teacher_path)
                if preservation_teacher_path is not None
                else None
            )
            or completed.get("training") != _jsonable(asdict(config))
            or completed.get("fleet_plan_id") != fleet_plan_id
        ):
            raise ValueError(
                "Completed standalone model does not match the requested run"
            )
        return StandaloneModelSummary(
            output_dir=output_dir,
            optimizer_steps=int(completed["optimizer_steps"]),
            examples=int(completed["examples"]),
            merged_modules=tuple(completed["merged_modules"]),
        )
    state_path = output_dir / "distillation_state.pt"
    identity_path = output_dir / "distillation_run.json"
    if not resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Standalone output directory is not empty and resume is disabled: {output_dir}"
        )
    identity = {
        "format": "qwen_intervention_distillation_run_v1",
        "base_model": model_bundle.model_id,
        "base_revision": _bundle_revision(model_bundle),
        "teacher_sha256": _sha256_file(teacher_path),
        "preservation_teacher_sha256": (
            _sha256_file(preservation_teacher_path)
            if preservation_teacher_path is not None
            else None
        ),
        "training": _jsonable(asdict(config)),
        "fleet_plan_id": fleet_plan_id,
    }
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "Distillation resume identity does not match the requested run"
            )
    else:
        unknown_files = [path.name for path in output_dir.iterdir()]
        if unknown_files:
            raise ValueError(
                f"Standalone output contains files without a matching run identity: {unknown_files}"
            )
        write_json_atomic(identity_path, identity)

    seed_everything(config.seed)
    model = model_bundle.model
    installed = install_lora(model, config)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, foreach=False)
    start_epoch = 0
    start_batch = 0
    optimizer_steps = 0
    loss_history: list[float] = []
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        _load_lora_state(installed, state["lora"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        start_batch = int(state.get("next_batch", 0))
        optimizer_steps = int(state["optimizer_steps"])
        loss_history = [float(value) for value in state.get("loss_history", [])]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation_step = 0
    for epoch in range(start_epoch, config.epochs):
        order = list(range(len(examples)))
        random.Random(config.seed + epoch).shuffle(order)
        batches = list(_chunks(order, config.batch_size))
        first_batch = start_batch if epoch == start_epoch else 0
        for batch_idx, indices in enumerate(batches[first_batch:], start=first_batch):
            seed_everything(_stable_seed(config.seed, str(epoch), str(batch_idx)))
            batch_rows = [examples[index] for index in indices]
            inputs, labels, weights = build_training_batch(
                processor=model_bundle.processor,
                records=batch_rows,
                max_length=config.max_length,
                device=_model_input_device(model),
            )
            outputs = model(**inputs, use_cache=False)
            loss = weighted_causal_lm_loss(outputs.logits, labels, weights)
            (loss / config.gradient_accumulation_steps).backward()
            loss_history.append(float(loss.detach().cpu().item()))
            accumulation_step += 1
            if accumulation_step % config.gradient_accumulation_steps != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if (
                config.checkpoint_every_steps > 0
                and optimizer_steps % config.checkpoint_every_steps == 0
            ):
                _save_training_state(
                    state_path=state_path,
                    installed=installed,
                    optimizer=optimizer,
                    epoch=epoch,
                    next_batch=batch_idx + 1,
                    optimizer_steps=optimizer_steps,
                    loss_history=loss_history,
                )
        if accumulation_step % config.gradient_accumulation_steps != 0:
            remainder = accumulation_step % config.gradient_accumulation_steps
            correction = config.gradient_accumulation_steps / remainder
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        accumulation_step = 0
        start_batch = 0
        _save_training_state(
            state_path=state_path,
            installed=installed,
            optimizer=optimizer,
            epoch=epoch + 1,
            next_batch=0,
            optimizer_steps=optimizer_steps,
            loss_history=loss_history,
        )

    adapter_path = output_dir / "lora_adapter.pt"
    torch.save(
        {
            "format": "qwen_intervention_lora_v1",
            "training": _jsonable(asdict(config)),
            "modules": _lora_state(installed),
        },
        adapter_path,
    )
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    merged_modules = merge_lora(installed)
    if config.gradient_checkpointing and hasattr(
        model, "gradient_checkpointing_disable"
    ):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    remaining_lora = [
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    lora_state_keys = [key for key in model.state_dict() if "lora_" in key]
    if remaining_lora or lora_state_keys:
        raise RuntimeError(
            "LoRA merge left adapter structure in the standalone model: "
            f"modules={remaining_lora}, keys={lora_state_keys}"
        )
    model.save_pretrained(output_dir, safe_serialization=True)
    model_bundle.processor.save_pretrained(output_dir)
    manifest = {
        "format": STANDALONE_MODEL_FORMAT,
        "base_model": model_bundle.model_id,
        "base_revision": _bundle_revision(model_bundle),
        "teacher_path": str(teacher_path.resolve()),
        "teacher_sha256": _sha256_file(teacher_path),
        "preservation_teacher_path": (
            str(preservation_teacher_path.resolve())
            if preservation_teacher_path is not None
            else None
        ),
        "preservation_teacher_sha256": (
            _sha256_file(preservation_teacher_path)
            if preservation_teacher_path is not None
            else None
        ),
        "intervention": teacher_payload["metadata"].get("intervention_path"),
        "intervention_sha256": teacher_payload["metadata"].get("intervention_sha256"),
        "training": _jsonable(asdict(config)),
        "optimizer_steps": optimizer_steps,
        "examples": len(examples),
        "merged_modules": list(merged_modules),
        "adapter_path": adapter_path.name,
        "mean_training_loss": sum(loss_history) / len(loss_history),
        "standalone": True,
        "runtime_intervention_required": False,
        "fleet_plan_id": fleet_plan_id,
        "structural_validation": {
            "remaining_lora_modules": 0,
            "remaining_lora_state_keys": 0,
        },
    }
    write_json_atomic(completed_manifest, manifest)
    state_path.unlink(missing_ok=True)
    return StandaloneModelSummary(
        output_dir=output_dir,
        optimizer_steps=optimizer_steps,
        examples=len(examples),
        merged_modules=merged_modules,
    )


def verify_standalone_model(
    *,
    model_dir: Path,
    prompt: str = "Answer with exactly one word: ready",
) -> dict[str, Any]:
    manifest_path = model_dir / "standalone_model.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Standalone manifest is missing: {manifest_path}")
    if (model_dir / "adapter_config.json").exists():
        raise ValueError("Standalone checkpoint still contains a PEFT adapter config")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    bundle = ModelBundle(
        model=model,
        processor=processor,
        tokenizer=processor.tokenizer,
        model_id=str(model_dir.resolve()),
        config=ModelLoadConfig(model_name=str(model_dir.resolve())),
    )
    completion = generate_completions(
        bundle=bundle,
        conversations=[[{"role": "user", "content": prompt, "detect": False}]],
        settings=GenerationSettings(max_new_tokens=16, do_sample=False),
    )[0]
    if not completion.text.strip():
        raise ValueError("Standalone checkpoint smoke generation was empty")
    result = {
        "format": "qwen_intervention_standalone_verification_v1",
        "model_dir": str(model_dir.resolve()),
        "prompt": prompt,
        "completion": completion.text,
        "stock_qwen_reload": True,
        "adapter_config_absent": True,
        "artifact_sha256": _standalone_artifact_inventory(model_dir),
    }
    write_json_atomic(model_dir / "standalone_verification.json", result)
    return result


def build_training_batch(
    *,
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    max_length: int,
    device: Any,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    tokenizer = processor.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        prompt_texts: list[str] = []
        full_texts: list[str] = []
        for row in records:
            messages = _generation_messages(row["messages"])
            prompt_text = _render_generation_conversation(
                processor,
                messages,
            )
            completion = str(row["assistant_content"])
            eos = tokenizer.eos_token or ""
            prompt_texts.append(prompt_text)
            full_texts.append(prompt_text + completion + eos)
        qwen_conversations = [
            [
                _qwen_message(message)
                for message in _generation_messages(row["messages"])
            ]
            for row in records
        ]
        if any(_conversation_has_vision(row["messages"]) for row in records):
            prompt_tokens = _processor_inputs(
                processor=processor,
                conversations=qwen_conversations,
                texts=prompt_texts,
                padding=True,
            )
            full_tokens = _processor_inputs(
                processor=processor,
                conversations=qwen_conversations,
                texts=full_texts,
                padding=True,
            )
        else:
            prompt_tokens = tokenizer(
                prompt_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            full_tokens = tokenizer(
                full_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
        if full_tokens["input_ids"].shape[1] > max_length:
            too_long = [
                str(records[index].get("id", index))
                for index, length in enumerate(
                    full_tokens["attention_mask"].sum(dim=1).tolist()
                )
                if int(length) > max_length
            ]
            raise ValueError(
                f"Training examples exceed max_length={max_length}; completion-preserving "
                f"truncation is not implicit. Example ids: {too_long}"
            )
        inputs = full_tokens
    finally:
        tokenizer.padding_side = original_padding_side
    labels = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"].to(dtype=torch.bool)
    prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1)
    for row_idx, prompt_length in enumerate(prompt_lengths.tolist()):
        usable_prompt = prompt_tokens["input_ids"][row_idx][
            prompt_tokens["attention_mask"][row_idx].to(dtype=torch.bool)
        ]
        usable_full = inputs["input_ids"][row_idx][attention_mask[row_idx]]
        if not torch.equal(usable_full[: int(prompt_length)], usable_prompt):
            raise ValueError(
                f"Prompt token ids are not a prefix for training row {row_idx}"
            )
        labels[row_idx, : int(prompt_length)] = -100
    labels[~attention_mask] = -100
    if torch.any((labels != -100).sum(dim=1) == 0):
        raise ValueError("A training example has no assistant tokens after truncation")
    tensor_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(inputs).items()
    }
    weights = torch.tensor(
        [float(row.get("loss_weight", 1.0)) for row in records],
        dtype=torch.float32,
        device=device,
    )
    return tensor_inputs, labels.to(device), weights


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
        ignore_index=-100,
    )
    mask = shifted_labels != -100
    per_example = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return (per_example * weights).mean()


def create_fleet_plan(
    *,
    intervention_paths: Sequence[Path],
    prompt_path: Path,
    output_root: Path,
    generation: GenerationSettings,
    training: LoRATrainingConfig,
    preservation_teacher_path: Path,
    base_revision: str,
    require_complete_suite: bool = True,
) -> FleetPlan:
    validate_lora_training_config(training)
    if not intervention_paths:
        raise ValueError("At least one intervention bundle is required")
    variants: list[FleetVariant] = []
    probe_methods: set[InterventionMethod] = set()
    random_control_methods: set[InterventionMethod] = set()
    names: set[str] = set()
    for path in intervention_paths:
        bundle = load_intervention_bundle(path)
        if bundle.spec.direction_mode == DirectionMode.PROBE:
            probe_methods.add(bundle.spec.method)
        else:
            random_control_methods.add(bundle.spec.method)
        name = path.stem
        if name in names:
            raise ValueError(f"Duplicate intervention variant name: {name}")
        names.add(name)
        variants.append(
            FleetVariant(
                name=name,
                intervention_path=str(path.resolve()),
                intervention_sha256=_sha256_file(path),
                teacher_output=str(
                    (output_root / "teachers" / f"{name}.json").resolve()
                ),
                model_output=str((output_root / "models" / name).resolve()),
            )
        )
    if require_complete_suite:
        missing = set(InterventionMethod) - probe_methods
        if missing or InterventionMethod.FULL_REFLECTION not in random_control_methods:
            raise ValueError(
                "Fleet plan requires all intervention methods and a matched-random full "
                "reflection control; "
                f"missing={sorted(method.value for method in missing)}, "
                f"random_control_methods={sorted(method.value for method in random_control_methods)}"
            )
    if re.fullmatch(r"[0-9a-f]{40}", base_revision) is None:
        raise ValueError("Fleet plan base revision must be a 40-character commit SHA")
    resolved_output_root = output_root.resolve()
    base_control_output = resolved_output_root / "controls" / "base_control.json"
    plan_payload = {
        "base_model": DEFAULT_MODEL_ID,
        "base_revision": base_revision,
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": _sha256_file(prompt_path),
        "preservation_teacher_path": str(preservation_teacher_path.resolve()),
        "preservation_teacher_sha256": _sha256_file(preservation_teacher_path),
        "output_root": str(resolved_output_root),
        "base_control_output": str(base_control_output),
        "variants": [asdict(variant) for variant in variants],
        "generation": _jsonable(asdict(generation)),
        "training": _jsonable(asdict(training)),
    }
    return FleetPlan(
        plan_id=hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        prompt_path=str(prompt_path.resolve()),
        prompt_sha256=_sha256_file(prompt_path),
        preservation_teacher_path=str(preservation_teacher_path.resolve()),
        preservation_teacher_sha256=_sha256_file(preservation_teacher_path),
        output_root=str(resolved_output_root),
        base_control_output=str(base_control_output),
        variants=tuple(variants),
        generation=generation,
        training=training,
        base_revision=base_revision,
    )


def save_fleet_plan(plan: FleetPlan, path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Fleet plan already exists: {path}")
    payload = {
        "format": FLEET_PLAN_FORMAT,
        **asdict(plan),
        "variants": [asdict(variant) for variant in plan.variants],
    }
    write_json_atomic(path, payload)


def load_fleet_plan(path: Path) -> FleetPlan:
    payload = json.loads(path.read_text())
    if payload.get("format") != FLEET_PLAN_FORMAT:
        raise ValueError(f"Unsupported fleet plan format: {payload.get('format')!r}")
    if payload.get("base_model") != DEFAULT_MODEL_ID:
        raise ValueError(f"Fleet plan base model must be {DEFAULT_MODEL_ID}")
    generation = GenerationSettings(**payload["generation"])
    training_raw = dict(payload["training"])
    training_raw["target_modules"] = tuple(training_raw["target_modules"])
    if training_raw.get("train_layers") is not None:
        training_raw["train_layers"] = tuple(training_raw["train_layers"])
    training = LoRATrainingConfig(**training_raw)
    validate_lora_training_config(training)
    plan = FleetPlan(
        plan_id=str(payload["plan_id"]),
        prompt_path=str(payload["prompt_path"]),
        prompt_sha256=str(payload["prompt_sha256"]),
        preservation_teacher_path=str(payload["preservation_teacher_path"]),
        preservation_teacher_sha256=str(payload["preservation_teacher_sha256"]),
        output_root=str(payload["output_root"]),
        base_control_output=str(payload["base_control_output"]),
        variants=tuple(FleetVariant(**row) for row in payload["variants"]),
        generation=generation,
        training=training,
        base_model=str(payload["base_model"]),
        base_revision=str(payload["base_revision"]),
    )
    if plan.plan_id != _fleet_plan_id(plan):
        raise ValueError("Fleet plan id does not match its immutable contents")
    validate_fleet_plan_files(plan)
    return plan


def validate_fleet_plan_files(plan: FleetPlan) -> None:
    expected = {
        Path(plan.prompt_path): plan.prompt_sha256,
        Path(plan.preservation_teacher_path): plan.preservation_teacher_sha256,
        **{
            Path(variant.intervention_path): variant.intervention_sha256
            for variant in plan.variants
        },
    }
    changed = [
        str(path) for path, digest in expected.items() if _sha256_file(path) != digest
    ]
    if changed:
        raise ValueError(f"Fleet plan input files changed after planning: {changed}")


def _fleet_plan_id(plan: FleetPlan) -> str:
    payload = {
        "base_model": plan.base_model,
        "base_revision": plan.base_revision,
        "prompt_path": plan.prompt_path,
        "prompt_sha256": plan.prompt_sha256,
        "preservation_teacher_path": plan.preservation_teacher_path,
        "preservation_teacher_sha256": plan.preservation_teacher_sha256,
        "output_root": plan.output_root,
        "base_control_output": plan.base_control_output,
        "variants": [asdict(variant) for variant in plan.variants],
        "generation": _jsonable(asdict(plan.generation)),
        "training": _jsonable(asdict(plan.training)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fleet_variant(plan: FleetPlan, name: str) -> FleetVariant:
    matches = [variant for variant in plan.variants if variant.name == name]
    if len(matches) != 1:
        raise ValueError(f"Fleet plan has no unique variant named {name!r}")
    return matches[0]


def record_base_control(plan: FleetPlan, model_bundle: ModelBundle) -> None:
    if model_bundle.model_id != plan.base_model:
        raise ValueError("Fleet base control model id does not match the plan")
    if _bundle_revision(model_bundle) != plan.base_revision:
        raise ValueError("Fleet base control revision does not match the plan")
    payload = {
        "format": "qwen_intervention_base_control_v1",
        "plan_id": plan.plan_id,
        "base_model": plan.base_model,
        "base_revision": plan.base_revision,
        "preservation_teacher_path": plan.preservation_teacher_path,
        "preservation_teacher_sha256": plan.preservation_teacher_sha256,
    }
    path = Path(plan.base_control_output)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Existing base control record does not match the fleet plan")
    write_json_atomic(path, payload)


def claim_fleet_variant(
    plan: FleetPlan,
    *,
    name: str | None = None,
    retry_failed: bool = False,
    worker_id: str | None = None,
    max_attempts: int = 2,
) -> FleetClaim | None:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    state_root = Path(plan.output_root) / "fleet_state" / plan.plan_id
    for state in ("locks", "running", "done", "failed", "artifacts"):
        (state_root / state).mkdir(parents=True, exist_ok=True)
    candidates = (
        [fleet_variant(plan, name)] if name is not None else list(plan.variants)
    )
    worker = worker_id or f"{socket.gethostname()}:{os.getpid()}"
    for variant in candidates:
        done = state_root / "done" / f"{variant.name}.json"
        failed = state_root / "failed" / f"{variant.name}.json"
        lock = state_root / "locks" / f"{variant.name}.lock"
        if done.exists():
            _validate_done_marker(plan, variant, done)
            continue
        if lock.exists() or any(
            (state_root / "running").glob(f"{variant.name}.*.json")
        ):
            continue
        previous_attempt = 0
        if failed.exists():
            if not retry_failed:
                continue
            previous_attempt = int(json.loads(failed.read_text()).get("attempt", 1))
            if previous_attempt >= max_attempts:
                continue
            failed.unlink(missing_ok=True)
        claim_token = uuid.uuid4().hex
        attempt = previous_attempt + 1
        running = state_root / "running" / f"{variant.name}.{claim_token}.json"
        artifact_manifest = (
            state_root / "artifacts" / f"{variant.name}.{claim_token}.json"
        )
        payload = {
            "plan_id": plan.plan_id,
            "variant": variant.name,
            "worker_id": worker,
            "claim_token": claim_token,
            "attempt": attempt,
            "status": "running",
            "artifact_manifest": str(artifact_manifest),
        }
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            write_json_atomic(running, payload)
            if (
                not lock.exists()
                or json.loads(lock.read_text()).get("claim_token") != claim_token
            ):
                running.unlink(missing_ok=True)
                continue
            return FleetClaim(
                variant=variant,
                marker_path=running,
                worker_id=worker,
                claim_token=claim_token,
                attempt=attempt,
            )
        except BaseException:
            lock.unlink(missing_ok=True)
            raise
    return None


def finish_fleet_claim(
    plan: FleetPlan,
    claim: FleetClaim,
    *,
    error: BaseException | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> None:
    state_root = Path(plan.output_root) / "fleet_state" / plan.plan_id
    status = "failed" if error is not None else "done"
    assert_fleet_claim_owned(plan, claim)
    if error is None and artifacts is None:
        raise ValueError(
            "Successful fleet claims must publish claim-specific artifacts"
        )
    marker = json.loads(claim.marker_path.read_text())
    write_json_atomic(
        Path(marker["artifact_manifest"]),
        {
            "plan_id": plan.plan_id,
            "variant": claim.variant.name,
            "worker_id": claim.worker_id,
            "claim_token": claim.claim_token,
            "attempt": claim.attempt,
            "status": status,
            "error": repr(error) if error is not None else None,
            "artifacts": dict(artifacts or {}),
        },
    )
    destination = state_root / status / f"{claim.variant.name}.json"
    try:
        os.replace(claim.marker_path, destination)
    except FileNotFoundError as caught:
        raise RuntimeError(
            "Fleet claim ownership was lost during publication"
        ) from caught
    lock = state_root / "locks" / f"{claim.variant.name}.lock"
    if lock.exists():
        lock_payload = json.loads(lock.read_text())
        if lock_payload.get("claim_token") == claim.claim_token:
            lock.unlink()


def _validate_done_marker(
    plan: FleetPlan,
    variant: FleetVariant,
    path: Path,
) -> None:
    marker = json.loads(path.read_text())
    if marker.get("plan_id") != plan.plan_id or marker.get("variant") != variant.name:
        raise ValueError(f"Fleet done marker has the wrong identity: {path}")
    artifact_manifest = Path(str(marker.get("artifact_manifest", "")))
    if not artifact_manifest.exists():
        raise ValueError(f"Fleet done marker is missing its artifact manifest: {path}")
    publication = json.loads(artifact_manifest.read_text())
    if publication.get("claim_token") != marker.get("claim_token"):
        raise ValueError(f"Fleet done publication has the wrong claim token: {path}")
    artifacts = publication.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Fleet done marker is missing artifacts: {path}")
    model_output = Path(str(artifacts.get("model_output", "")))
    teacher_output = Path(str(artifacts.get("teacher_output", "")))
    if not teacher_output.exists() or not _standalone_output_files_present(
        model_output, require_verification=True
    ):
        raise ValueError(f"Fleet done artifacts are missing or corrupt: {path}")


def assert_fleet_claim_owned(plan: FleetPlan, claim: FleetClaim) -> None:
    if not claim.marker_path.exists():
        raise RuntimeError("Fleet claim ownership was lost before completion")
    marker = json.loads(claim.marker_path.read_text())
    lock = (
        Path(plan.output_root)
        / "fleet_state"
        / plan.plan_id
        / "locks"
        / f"{claim.variant.name}.lock"
    )
    if (
        marker.get("plan_id") != plan.plan_id
        or marker.get("worker_id") != claim.worker_id
        or marker.get("claim_token") != claim.claim_token
        or not lock.exists()
        or json.loads(lock.read_text()).get("claim_token") != claim.claim_token
    ):
        raise RuntimeError("Fleet claim marker is owned by a different worker")


def fleet_status(plan: FleetPlan) -> dict[str, Any]:
    state_root = Path(plan.output_root) / "fleet_state" / plan.plan_id
    states: dict[str, list[str]] = {}
    for state in ("running", "done", "failed"):
        names: list[str] = []
        for path in (state_root / state).glob("*.json"):
            try:
                names.append(str(json.loads(path.read_text())["variant"]))
            except (KeyError, json.JSONDecodeError):
                names.append(path.stem)
        states[state] = sorted(set(names))
    corrupt: list[str] = []
    valid_done: list[str] = []
    done_artifacts: dict[str, dict[str, str]] = {}
    for name in states["done"]:
        try:
            _validate_done_marker(
                plan,
                fleet_variant(plan, name),
                state_root / "done" / f"{name}.json",
            )
            valid_done.append(name)
            done_marker = json.loads((state_root / "done" / f"{name}.json").read_text())
            publication = json.loads(Path(done_marker["artifact_manifest"]).read_text())
            done_artifacts[name] = dict(publication["artifacts"])
        except (ValueError, FileNotFoundError, json.JSONDecodeError):
            corrupt.append(name)
    states["done"] = valid_done
    states["corrupt"] = corrupt
    accounted = set(
        states["running"] + states["done"] + states["failed"] + states["corrupt"]
    )
    states["pending"] = sorted(
        variant.name for variant in plan.variants if variant.name not in accounted
    )
    return {
        "plan_id": plan.plan_id,
        "base_control": (
            "recorded" if Path(plan.base_control_output).exists() else "pending"
        ),
        "done_artifacts": done_artifacts,
        **states,
    }


def recover_running_fleet_variants(plan: FleetPlan) -> list[str]:
    state_root = Path(plan.output_root) / "fleet_state" / plan.plan_id
    recovered: list[str] = []
    for running in sorted((state_root / "running").glob("*.json")):
        payload = json.loads(running.read_text())
        variant = str(payload["variant"])
        os.replace(running, state_root / "failed" / f"{variant}.json")
        lock = state_root / "locks" / f"{variant}.lock"
        if lock.exists():
            lock_payload = json.loads(lock.read_text())
            if lock_payload.get("claim_token") == payload.get("claim_token"):
                lock.unlink()
        recovered.append(variant)
    running_variants = set(recovered)
    for lock in sorted((state_root / "locks").glob("*.lock")):
        payload = json.loads(lock.read_text())
        variant = str(payload["variant"])
        if variant in running_variants:
            continue
        os.replace(lock, state_root / "failed" / f"{variant}.json")
        recovered.append(variant)
    return recovered


def _standalone_output_files_present(
    output_dir: Path,
    *,
    require_verification: bool,
) -> bool:
    weights = list(output_dir.glob("*.safetensors")) + list(
        output_dir.glob("model*.pt")
    )
    required = [
        output_dir / "standalone_model.json",
        output_dir / "config.json",
        output_dir / "tokenizer_config.json",
    ]
    processor_metadata = any(
        (output_dir / name).exists()
        for name in (
            "preprocessor_config.json",
            "processor_config.json",
            "processor.json",
        )
    )
    if require_verification:
        verification_path = output_dir / "standalone_verification.json"
        required.append(verification_path)
        if all(path.exists() for path in required):
            verification = json.loads(verification_path.read_text())
            if verification.get("artifact_sha256") != _standalone_artifact_inventory(
                output_dir
            ):
                return False
    return (
        bool(weights) and processor_metadata and all(path.exists() for path in required)
    )


def _standalone_artifact_inventory(output_dir: Path) -> dict[str, str]:
    excluded = {
        "distillation_state.pt",
        "distillation_run.json",
        "lora_adapter.pt",
        "standalone_verification.json",
    }
    return {
        str(path.relative_to(output_dir)): _sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in excluded and ".tmp-" not in path.name
    }


def _load_or_initialize_teacher_output(
    *,
    output_path: Path,
    metadata: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if not output_path.exists():
        return {
            "format": TEACHER_DATASET_FORMAT,
            "metadata": dict(metadata),
            "records": [],
        }
    if not resume:
        raise FileExistsError(f"Teacher dataset already exists: {output_path}")
    payload = json.loads(output_path.read_text())
    if payload.get("format") != TEACHER_DATASET_FORMAT:
        raise ValueError(
            f"Unsupported teacher dataset format: {payload.get('format')!r}"
        )
    if payload.get("metadata") != dict(metadata):
        raise ValueError(
            "Teacher dataset resume metadata does not match the requested run"
        )
    return payload


def _load_teacher_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("format") != TEACHER_DATASET_FORMAT:
        raise ValueError(
            f"Unsupported teacher dataset format: {payload.get('format')!r}"
        )
    if not isinstance(payload.get("records"), list):
        raise ValueError("Teacher dataset is missing records")
    return payload


def _validate_teacher_compatibility(
    payload: Mapping[str, Any],
    *,
    model_bundle: ModelBundle,
    require_intervention: bool,
) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Teacher dataset is missing metadata")
    current_revision = _bundle_revision(model_bundle)
    if metadata.get("base_model") != model_bundle.model_id:
        raise ValueError("Teacher dataset base model does not match the loaded model")
    if metadata.get("base_revision") != current_revision:
        raise ValueError(
            "Teacher dataset base revision does not match the loaded model"
        )
    has_intervention = metadata.get("intervention_path") is not None
    if require_intervention and not has_intervention:
        raise ValueError("Target teacher dataset must identify an intervention")
    if not require_intervention and has_intervention:
        raise ValueError(
            "Preservation teacher dataset must come from the unmodified base model"
        )


def _save_training_state(
    *,
    state_path: Path,
    installed: Sequence[InstalledLoRA],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch: int,
    optimizer_steps: int,
    loss_history: Sequence[float],
) -> None:
    payload = {
        "format": "qwen_intervention_distillation_state_v1",
        "epoch": epoch,
        "next_batch": next_batch,
        "optimizer_steps": optimizer_steps,
        "loss_history": list(loss_history),
        "lora": _lora_state(installed),
        "optimizer": optimizer.state_dict(),
    }
    temporary = state_path.with_name(f".{state_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, state_path)
    finally:
        temporary.unlink(missing_ok=True)


def _lora_state(
    installed: Sequence[InstalledLoRA],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        item.name: {
            "lora_a": item.module.lora_a.detach().cpu(),
            "lora_b": item.module.lora_b.detach().cpu(),
        }
        for item in installed
    }


def _load_lora_state(
    installed: Sequence[InstalledLoRA],
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    expected = {item.name for item in installed}
    if set(state) != expected:
        raise ValueError("Saved LoRA module inventory does not match the current model")
    for item in installed:
        item.module.lora_a.data.copy_(state[item.name]["lora_a"])
        item.module.lora_b.data.copy_(state[item.name]["lora_b"])


def _resolve_relative_module(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _normalise_message(message: Mapping[str, Any]) -> Message:
    content = message.get("content", "")
    return {
        "role": str(message["role"]),
        "content": content,
        "detect": bool(message.get("detect", False)),
    }


def _generation_messages(messages: Sequence[Mapping[str, Any]]) -> list[Message]:
    normalized = [_normalise_message(message) for message in messages]
    if (
        len(normalized) >= 2
        and normalized[-1]["role"] == "assistant"
        and not str(normalized[-1].get("content", "")).strip()
        and bool(normalized[-1].get("detect", False))
        and normalized[-2]["role"] == "assistant"
    ):
        return normalized[:-1]
    return normalized


def _qwen_message(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    qwen_content = (
        content
        if isinstance(content, list)
        else [{"type": "text", "text": str(content)}]
    )
    return {
        "role": str(message["role"]),
        "content": qwen_content,
    }


def _conversation_has_vision(conversation: Sequence[Mapping[str, Any]]) -> bool:
    for message in conversation:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        if any(
            item.get("type") in {"image", "image_url", "video"}
            for item in content
            if isinstance(item, Mapping)
        ):
            return True
    return False


def _processor_inputs(
    *,
    processor: Any,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    texts: Sequence[str],
    padding: bool,
) -> Mapping[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Multimodal distillation requires qwen-vl-utils and a torchvision build "
            "compatible with the installed Torch/CUDA runtime"
        ) from error
    patch_size = int(
        getattr(getattr(processor, "image_processor", None), "patch_size", 14)
    )
    images, videos, video_kwargs = process_vision_info(
        list(conversations),
        return_video_kwargs=True,
        image_patch_size=patch_size,
    )
    return processor(
        text=list(texts),
        images=images,
        videos=videos,
        padding=padding,
        return_tensors="pt",
        **video_kwargs,
    )


def _completion_content(
    *, text: str, thinking: str | None, raw_text: str | None
) -> str:
    if raw_text is not None:
        return raw_text
    if thinking is None:
        return text
    return f"{thinking}\n</think>\n\n{text}"


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\x1f".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _natural_id_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def _model_input_device(model: Any) -> Any:
    embedding = model.get_input_embeddings()
    return embedding.weight.device


def _model_revision(model: Any) -> str | None:
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def _bundle_revision(bundle: ModelBundle) -> str | None:
    return _model_revision(bundle.model) or bundle.config.revision


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))
