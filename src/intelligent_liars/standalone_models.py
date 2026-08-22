from __future__ import annotations

import hashlib
import json
import math
import os
import random
import gc
import fcntl
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
    DECEPTION_DIRECTED_SUITE_METHODS,
    SEEDED_ORTHOGONAL_CONTROL_VARIANT,
    InterventionBundle,
    RuntimeIntervention,
    canonical_intervention_suite_specs,
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
STANDALONE_MODEL_FORMAT = "qwen_intervention_student_v2"
FLEET_PLAN_FORMAT = "qwen_intervention_fleet_plan_v2"
TINYLORA_IMPLEMENTATION = "intelligent_liars_tinylora_v2"
TINYLORA_SVD_OVERSAMPLING = 4
TINYLORA_SVD_POWER_ITERATIONS = 2
TINYLORA_BASIS_SEED_OFFSET = 1_000_000
DEFAULT_TINYLORA_TARGETS = (
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
class TinyLoRATrainingConfig:
    svd_rank: int = 2
    projection_dim: int = 13
    projection_seed: int = 42
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
    target_modules: tuple[str, ...] = DEFAULT_TINYLORA_TARGETS
    train_layers: tuple[int, ...] | None = None
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class StandaloneModelSummary:
    output_dir: Path
    optimizer_steps: int
    examples: int
    merged_modules: tuple[str, ...]
    trainable_scalars: int


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
    training: TinyLoRATrainingConfig
    base_model: str = DEFAULT_MODEL_ID
    base_revision: str = ""


@dataclass(frozen=True)
class FleetClaim:
    variant: FleetVariant
    marker_path: Path
    worker_id: str
    claim_token: str
    attempt: int


class TinyLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        trainable_vector: nn.Parameter,
        group_index: int,
        module_index: int,
        svd_rank: int,
        projection_seed: int,
        dropout: float,
        basis: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if svd_rank < 1:
            raise ValueError("TinyLoRA SVD rank must be positive")
        if not hasattr(base, "weight"):
            raise TypeError("TinyLoRA target must expose a weight tensor")
        weight = base.weight
        if weight.ndim != 2:
            raise TypeError("TinyLoRA target weight must be a matrix")
        self.base = base
        self.group_index = group_index
        self.module_index = module_index
        self.svd_rank = min(svd_rank, min(weight.shape))
        self.dropout = nn.Dropout(dropout)
        self.tinylora_v = trainable_vector
        if basis is None:
            weight_fp32 = weight.detach().float()
            approximation_rank = min(
                self.svd_rank + TINYLORA_SVD_OVERSAMPLING,
                min(weight.shape),
            )
            devices = (
                [
                    weight.device.index
                    if weight.device.index is not None
                    else torch.cuda.current_device()
                ]
                if weight.device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(
                    projection_seed + module_index + TINYLORA_BASIS_SEED_OFFSET
                )
                left, singular_values, right = torch.svd_lowrank(
                    weight_fp32,
                    q=approximation_rank,
                    niter=TINYLORA_SVD_POWER_ITERATIONS,
                )
            right = right.transpose(0, 1)
            tinylora_a = (
                torch.sqrt(singular_values[: self.svd_rank])[:, None]
                * right[: self.svd_rank]
            )
            tinylora_b = (
                left[:, : self.svd_rank]
                * torch.sqrt(singular_values[: self.svd_rank])[None, :]
            )
            generator = torch.Generator(device="cpu").manual_seed(
                projection_seed + module_index
            )
            tinylora_projection = torch.normal(
                mean=0.0,
                std=1.0 / math.sqrt(self.svd_rank),
                size=(trainable_vector.numel(), self.svd_rank, self.svd_rank),
                generator=generator,
            )
        else:
            tinylora_a = basis["tinylora_a"]
            tinylora_b = basis["tinylora_b"]
            tinylora_projection = basis["tinylora_projection"]
            expected_shapes = (
                (self.svd_rank, weight.shape[1]),
                (weight.shape[0], self.svd_rank),
                (trainable_vector.numel(), self.svd_rank, self.svd_rank),
            )
            actual_shapes = (
                tuple(tinylora_a.shape),
                tuple(tinylora_b.shape),
                tuple(tinylora_projection.shape),
            )
            if actual_shapes != expected_shapes:
                raise ValueError(
                    "Cached TinyLoRA basis shapes do not match the target module"
                )
        self.register_buffer(
            "tinylora_a",
            tinylora_a.to(device=weight.device, dtype=weight.dtype).contiguous(),
        )
        self.register_buffer(
            "tinylora_b",
            tinylora_b.to(device=weight.device, dtype=weight.dtype).contiguous(),
        )
        self.register_buffer(
            "tinylora_projection",
            tinylora_projection.to(
                device=weight.device,
                dtype=weight.dtype,
            ).contiguous(),
        )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def _mixing_matrix(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.tinylora_projection.dtype
        return torch.einsum(
            "i,ijk->jk",
            self.tinylora_v.to(dtype=dtype),
            self.tinylora_projection.to(dtype=dtype),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        projected = F.linear(self.dropout(inputs), self.tinylora_a)
        mixed = F.linear(projected, self._mixing_matrix())
        update = F.linear(mixed, self.tinylora_b)
        return base_output + update.to(dtype=base_output.dtype)

    def delta_weight(self) -> torch.Tensor:
        return (
            self.tinylora_b.float()
            @ self._mixing_matrix(dtype=torch.float32)
            @ self.tinylora_a.float()
        )

    def merged_weight(self) -> torch.Tensor:
        return (self.base.weight.float() + self.delta_weight()).to(
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )


@dataclass(frozen=True)
class InstalledTinyLoRA:
    name: str
    parent: nn.Module
    attribute: str
    module: TinyLoRALinear


@dataclass(frozen=True)
class TinyLoRAMergeSummary:
    modules: tuple[str, ...]
    trainable_scalars: int
    parameter_groups: int
    changed_entries: int
    target_entries: int
    delta_l2_norm: float
    target_weight_l2_norm: float
    relative_delta_l2_norm: float
    max_abs_delta: float
    trainable_l2_norm: float


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


def validate_tinylora_training_config(config: TinyLoRATrainingConfig) -> None:
    positive_ints = {
        "svd_rank": config.svd_rank,
        "projection_dim": config.projection_dim,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length": config.max_length,
    }
    invalid_ints = [name for name, value in positive_ints.items() if value < 1]
    if invalid_ints:
        raise ValueError(f"TinyLoRA training settings must be positive: {invalid_ints}")
    if config.checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be non-negative")
    numeric = {
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "max_grad_norm": config.max_grad_norm,
        "preservation_weight": config.preservation_weight,
    }
    non_finite = [name for name, value in numeric.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"TinyLoRA training settings must be finite: {non_finite}")
    if config.learning_rate <= 0 or config.max_grad_norm <= 0:
        raise ValueError("TinyLoRA learning_rate and max_grad_norm must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("TinyLoRA dropout must be in [0, 1)")
    if config.preservation_weight <= 0:
        raise ValueError("preservation_weight must be positive")
    if not config.target_modules:
        raise ValueError("At least one TinyLoRA target module is required")
    if len(set(config.target_modules)) != len(config.target_modules):
        raise ValueError("TinyLoRA target modules must be unique")
    if config.train_layers is not None and len(set(config.train_layers)) != len(
        config.train_layers
    ):
        raise ValueError("TinyLoRA train layers must be unique")


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


def install_tinylora(
    model: Any,
    config: TinyLoRATrainingConfig,
    *,
    basis: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> list[InstalledTinyLoRA]:
    validate_tinylora_training_config(config)
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
        raise ValueError(f"TinyLoRA train layers are outside the model: {invalid}")
    targets: list[tuple[int, str, nn.Module, str, nn.Module]] = []
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        for target in config.target_modules:
            parent, attribute = _resolve_relative_module(layer, target)
            base = getattr(parent, attribute)
            targets.append((layer_idx, target, parent, attribute, base))
    if not targets:
        raise ValueError("No TinyLoRA modules were selected")
    target_names = {
        f"model.language_model.layers.{layer_idx}.{target}"
        for layer_idx, target, _parent, _attribute, _base in targets
    }
    if basis is not None and set(basis) != target_names:
        raise ValueError("Cached TinyLoRA basis module inventory does not match")
    trainable_vector = nn.Parameter(
        torch.zeros(
            config.projection_dim,
            dtype=torch.float32,
            device=targets[0][4].weight.device,
        )
    )
    installed: list[InstalledTinyLoRA] = []
    for module_index, (layer_idx, target, parent, attribute, base) in enumerate(
        targets
    ):
        name = f"model.language_model.layers.{layer_idx}.{target}"
        if trainable_vector.device != base.weight.device:
            raise ValueError(
                "Fully tied TinyLoRA cannot span devices; use one model per GPU"
            )
        wrapped = TinyLoRALinear(
            base,
            trainable_vector=trainable_vector,
            group_index=0,
            module_index=module_index,
            svd_rank=config.svd_rank,
            projection_seed=config.projection_seed,
            dropout=config.dropout,
            basis=(basis[name] if basis is not None else None),
        )
        setattr(parent, attribute, wrapped)
        installed.append(
            InstalledTinyLoRA(
                name=name,
                parent=parent,
                attribute=attribute,
                module=wrapped,
            )
        )
    return installed


def install_tinylora_with_cache(
    *,
    model_bundle: ModelBundle,
    config: TinyLoRATrainingConfig,
    cache_path: Path | None,
) -> list[InstalledTinyLoRA]:
    if cache_path is None:
        return install_tinylora(model_bundle.model, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "format": "qwen_tinylora_basis_v1",
        "implementation": TINYLORA_IMPLEMENTATION,
        "base_model": model_bundle.model_id,
        "base_revision": _bundle_revision(model_bundle),
        "base_dtype": str(next(model_bundle.model.parameters()).dtype),
        "svd_rank": config.svd_rank,
        "svd_method": "seeded_randomized_lowrank",
        "svd_oversampling": TINYLORA_SVD_OVERSAMPLING,
        "svd_power_iterations": TINYLORA_SVD_POWER_ITERATIONS,
        "basis_seed_offset": TINYLORA_BASIS_SEED_OFFSET,
        "projection_dim": config.projection_dim,
        "projection_seed": config.projection_seed,
        "target_modules": list(config.target_modules),
        "train_layers": list(config.train_layers) if config.train_layers else None,
    }
    lock_path = cache_path.with_name(f".{cache_path.name}.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            if cache_path.exists():
                payload = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if payload.get("identity") != identity:
                    raise ValueError(
                        "Cached TinyLoRA basis does not match this model and config"
                    )
                basis = payload.get("modules")
                if not isinstance(basis, Mapping):
                    raise ValueError("Cached TinyLoRA basis is missing module tensors")
                return install_tinylora(
                    model_bundle.model,
                    config,
                    basis=basis,
                )
            installed = install_tinylora(model_bundle.model, config)
            temporary = cache_path.with_name(
                f".{cache_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            )
            try:
                with temporary.open("wb") as output:
                    torch.save(
                        {
                            "identity": identity,
                            "modules": _tinylora_basis_state(installed),
                        },
                        output,
                    )
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
            return installed
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def merge_tinylora(
    installed: Sequence[InstalledTinyLoRA],
) -> TinyLoRAMergeSummary:
    merged: list[str] = []
    vectors = {id(item.module.tinylora_v): item.module.tinylora_v for item in installed}
    changed_entries = 0
    target_entries = 0
    delta_squared = 0.0
    target_squared = 0.0
    max_abs_delta = 0.0
    for item in installed:
        delta = item.module.delta_weight().detach()
        merged_weight = item.module.merged_weight()
        if not torch.isfinite(merged_weight).all():
            raise ValueError(
                f"TinyLoRA merge produced non-finite weights for {item.name}"
            )
        changed_entries += int(
            torch.count_nonzero(merged_weight != item.module.base.weight).item()
        )
        target_entries += item.module.base.weight.numel()
        delta_squared += float(torch.sum(delta.double().square()).cpu().item())
        target_squared += float(
            torch.sum(item.module.base.weight.detach().double().square()).cpu().item()
        )
        max_abs_delta = max(
            max_abs_delta,
            float(delta.abs().max().cpu().item()),
        )
        with torch.no_grad():
            item.module.base.weight.copy_(merged_weight)
        setattr(item.parent, item.attribute, item.module.base)
        merged.append(item.name)
    delta_l2_norm = math.sqrt(delta_squared)
    target_weight_l2_norm = math.sqrt(target_squared)
    return TinyLoRAMergeSummary(
        modules=tuple(merged),
        trainable_scalars=sum(vector.numel() for vector in vectors.values()),
        parameter_groups=len(vectors),
        changed_entries=changed_entries,
        target_entries=target_entries,
        delta_l2_norm=delta_l2_norm,
        target_weight_l2_norm=target_weight_l2_norm,
        relative_delta_l2_norm=(
            delta_l2_norm / target_weight_l2_norm if target_weight_l2_norm else 0.0
        ),
        max_abs_delta=max_abs_delta,
        trainable_l2_norm=math.sqrt(
            sum(
                float(torch.sum(vector.detach().double().square()).cpu().item())
                for vector in vectors.values()
            )
        ),
    )


def train_standalone_model(
    *,
    model_bundle: ModelBundle,
    teacher_path: Path,
    output_dir: Path,
    config: TinyLoRATrainingConfig,
    preservation_teacher_path: Path | None = None,
    resume: bool = True,
    fleet_plan_id: str | None = None,
    tinylora_basis_path: Path | None = None,
) -> StandaloneModelSummary:
    validate_tinylora_training_config(config)
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
            or completed.get("tinylora_basis_path")
            != (
                str(tinylora_basis_path.resolve())
                if tinylora_basis_path is not None
                else None
            )
        ):
            raise ValueError(
                "Completed standalone model does not match the requested run"
            )
        return StandaloneModelSummary(
            output_dir=output_dir,
            optimizer_steps=int(completed["optimizer_steps"]),
            examples=int(completed["examples"]),
            merged_modules=tuple(completed["merged_modules"]),
            trainable_scalars=int(completed["adapter_accounting"]["trainable_scalars"]),
        )
    state_path = output_dir / "distillation_state.pt"
    identity_path = output_dir / "distillation_run.json"
    if not resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Standalone output directory is not empty and resume is disabled: {output_dir}"
        )
    identity = {
        "format": "qwen_intervention_distillation_run_v2",
        "adapter_implementation": TINYLORA_IMPLEMENTATION,
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
        "tinylora_basis_path": (
            str(tinylora_basis_path.resolve())
            if tinylora_basis_path is not None
            else None
        ),
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
    installed = install_tinylora_with_cache(
        model_bundle=model_bundle,
        config=config,
        cache_path=tinylora_basis_path,
    )
    tinylora_basis_sha256 = (
        _sha256_file(tinylora_basis_path) if tinylora_basis_path is not None else None
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    trainable = list(
        {
            id(parameter): parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        }.values()
    )
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, foreach=False)
    start_epoch = 0
    start_batch = 0
    optimizer_steps = 0
    loss_history: list[float] = []
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if (
            state.get("format") != "qwen_intervention_distillation_state_v2"
            or state.get("adapter_implementation") != TINYLORA_IMPLEMENTATION
            or state.get("tinylora_basis_sha256") != tinylora_basis_sha256
        ):
            raise ValueError("Unsupported TinyLoRA distillation checkpoint")
        _load_tinylora_state(installed, state["tinylora"])
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
                    tinylora_basis_sha256=tinylora_basis_sha256,
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
            tinylora_basis_sha256=tinylora_basis_sha256,
        )

    adapter_path = output_dir / "tinylora_adapter.pt"
    torch.save(
        {
            "format": "qwen_intervention_tinylora_v1",
            "implementation": TINYLORA_IMPLEMENTATION,
            "training": _jsonable(asdict(config)),
            "state": _tinylora_state(installed),
        },
        adapter_path,
    )
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    merge_summary = merge_tinylora(installed)
    merged_modules = merge_summary.modules
    if merge_summary.trainable_l2_norm > 0 and merge_summary.changed_entries == 0:
        raise RuntimeError(
            "TinyLoRA training produced a nonzero adapter that vanished during merge"
        )
    if config.gradient_checkpointing and hasattr(
        model, "gradient_checkpointing_disable"
    ):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    remaining_tinylora = [
        name
        for name, module in model.named_modules()
        if isinstance(module, TinyLoRALinear)
    ]
    tinylora_state_keys = [key for key in model.state_dict() if "tinylora_" in key]
    if remaining_tinylora or tinylora_state_keys:
        raise RuntimeError(
            "TinyLoRA merge left adapter structure in the standalone model: "
            f"modules={remaining_tinylora}, keys={tinylora_state_keys}"
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
        "adapter_type": "tinylora",
        "tinylora_basis_path": (
            str(tinylora_basis_path.resolve())
            if tinylora_basis_path is not None
            else None
        ),
        "tinylora_basis_sha256": tinylora_basis_sha256,
        "adapter_accounting": {
            "implementation": TINYLORA_IMPLEMENTATION,
            "trainable_scalars": merge_summary.trainable_scalars,
            "trainable_dtype": "torch.float32",
            "trainable_l2_norm": merge_summary.trainable_l2_norm,
            "parameter_groups": merge_summary.parameter_groups,
            "svd_rank": config.svd_rank,
            "svd_method": "seeded_randomized_lowrank",
            "svd_oversampling": TINYLORA_SVD_OVERSAMPLING,
            "svd_power_iterations": TINYLORA_SVD_POWER_ITERATIONS,
            "basis_seed_offset": TINYLORA_BASIS_SEED_OFFSET,
            "projection_dim_per_group": config.projection_dim,
            "weight_tying": 1.0,
            "projection_seed": config.projection_seed,
            "target_modules": len(merge_summary.modules),
            "effective_update_rank_upper_bound": config.svd_rank,
            "changed_entries_after_merge_cast": merge_summary.changed_entries,
            "target_entries": merge_summary.target_entries,
            "delta_l2_norm": merge_summary.delta_l2_norm,
            "target_weight_l2_norm": merge_summary.target_weight_l2_norm,
            "relative_delta_l2_norm": merge_summary.relative_delta_l2_norm,
            "max_abs_delta": merge_summary.max_abs_delta,
            "module_groups": [
                {
                    "module": item.name,
                    "group_index": item.module.group_index,
                    "module_index": item.module.module_index,
                    "projection_seed": config.projection_seed
                    + item.module.module_index,
                }
                for item in installed
            ],
        },
        "mean_training_loss": sum(loss_history) / len(loss_history),
        "standalone": True,
        "runtime_intervention_required": False,
        "fleet_plan_id": fleet_plan_id,
        "structural_validation": {
            "remaining_tinylora_modules": 0,
            "remaining_tinylora_state_keys": 0,
        },
    }
    write_json_atomic(completed_manifest, manifest)
    state_path.unlink(missing_ok=True)
    return StandaloneModelSummary(
        output_dir=output_dir,
        optimizer_steps=optimizer_steps,
        examples=len(examples),
        merged_modules=merged_modules,
        trainable_scalars=merge_summary.trainable_scalars,
    )


def verify_standalone_model(
    *,
    model_dir: Path,
    prompt: str = "Answer with exactly one word: ready",
) -> dict[str, Any]:
    manifest_path = model_dir / "standalone_model.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Standalone manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    accounting = manifest.get("adapter_accounting")
    if (
        manifest.get("format") != STANDALONE_MODEL_FORMAT
        or manifest.get("adapter_type") != "tinylora"
        or not isinstance(accounting, Mapping)
        or accounting.get("implementation") != TINYLORA_IMPLEMENTATION
        or int(accounting.get("trainable_scalars", 0)) < 1
    ):
        raise ValueError("Standalone manifest has invalid TinyLoRA provenance")
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
    residual_keys = [key for key in model.state_dict() if "tinylora_" in key]
    if residual_keys:
        raise ValueError(
            f"Standalone checkpoint still contains TinyLoRA state: {residual_keys}"
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
        "tinylora_state_absent": True,
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
    training: TinyLoRATrainingConfig,
    preservation_teacher_path: Path,
    base_revision: str,
    require_complete_suite: bool = True,
) -> FleetPlan:
    validate_tinylora_training_config(training)
    if not intervention_paths:
        raise ValueError("At least one intervention bundle is required")
    variants: list[FleetVariant] = []
    names: set[str] = set()
    bundles: dict[str, InterventionBundle] = {}
    for path in intervention_paths:
        bundle = load_intervention_bundle(path)
        name = path.stem
        if name in names:
            raise ValueError(f"Duplicate intervention variant name: {name}")
        names.add(name)
        bundles[name] = bundle
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
        expected_names = {
            *DECEPTION_DIRECTED_SUITE_METHODS,
            SEEDED_ORTHOGONAL_CONTROL_VARIANT,
        }
        if names != expected_names:
            raise ValueError(
                "Fleet plan requires exactly the canonical eight variants; "
                f"missing={sorted(expected_names - names)}, "
                f"extra={sorted(names - expected_names)}"
            )
        control = bundles[SEEDED_ORTHOGONAL_CONTROL_VARIANT]
        semantic_directions = {
            (
                bundle.direction.vector,
                bundle.direction.intercept,
                bundle.direction.layer,
                bundle.direction.task,
                bundle.direction.sign_convention,
            )
            for bundle in bundles.values()
        }
        shared_layers = {bundle.spec.layers for bundle in bundles.values()}
        shared_token_scopes = {
            bundle.spec.token_scope for bundle in bundles.values()
        }
        scalar = bundles["directed_scalar_add_deceptive"].spec
        affine = bundles["directed_affine_project_deceptive"].spec
        if (
            len(semantic_directions) != 1
            or len(shared_layers) != 1
            or len(shared_token_scopes) != 1
            or control.spec.control_seed is None
        ):
            raise ValueError(
                "Fleet plan variants do not share one semantic direction, layer set, "
                "token scope, and seeded control"
            )
        expected_specs = canonical_intervention_suite_specs(
            layers=next(iter(shared_layers)),
            control_seed=control.spec.control_seed,
            deceptive_margin=affine.projection_target,
            score_movement_budget=scalar.score_delta,
        )
        invalid_specs = [
            name for name in expected_names if bundles[name].spec != expected_specs[name]
        ]
        if invalid_specs:
            raise ValueError(
                "Fleet plan variants do not match the canonical directional semantics; "
                f"invalid_variants={sorted(invalid_specs)}"
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
    if payload.get("format") == "qwen_intervention_fleet_plan_v1":
        raise ValueError(
            "Ordinary-LoRA v1 fleet plans cannot be resumed as TinyLoRA; "
            "regenerate the immutable plan"
        )
    if payload.get("format") != FLEET_PLAN_FORMAT:
        raise ValueError(f"Unsupported fleet plan format: {payload.get('format')!r}")
    if payload.get("base_model") != DEFAULT_MODEL_ID:
        raise ValueError(f"Fleet plan base model must be {DEFAULT_MODEL_ID}")
    generation = GenerationSettings(**payload["generation"])
    training_raw = dict(payload["training"])
    training_raw["target_modules"] = tuple(training_raw["target_modules"])
    if training_raw.get("train_layers") is not None:
        training_raw["train_layers"] = tuple(training_raw["train_layers"])
    training = TinyLoRATrainingConfig(**training_raw)
    validate_tinylora_training_config(training)
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
        "tinylora_adapter.pt",
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
    installed: Sequence[InstalledTinyLoRA],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch: int,
    optimizer_steps: int,
    loss_history: Sequence[float],
    tinylora_basis_sha256: str | None,
) -> None:
    payload = {
        "format": "qwen_intervention_distillation_state_v2",
        "adapter_implementation": TINYLORA_IMPLEMENTATION,
        "epoch": epoch,
        "next_batch": next_batch,
        "optimizer_steps": optimizer_steps,
        "loss_history": list(loss_history),
        "tinylora_basis_sha256": tinylora_basis_sha256,
        "tinylora": _tinylora_state(installed),
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


def _tinylora_state(
    installed: Sequence[InstalledTinyLoRA],
) -> dict[str, Any]:
    groups: dict[str, torch.Tensor] = {}
    modules: dict[str, dict[str, Any]] = {}
    for item in installed:
        group_key = str(item.module.group_index)
        groups.setdefault(group_key, item.module.tinylora_v.detach().cpu())
        modules[item.name] = {
            "group_index": item.module.group_index,
            "module_index": item.module.module_index,
            "tinylora_a": item.module.tinylora_a.detach().cpu(),
            "tinylora_b": item.module.tinylora_b.detach().cpu(),
            "tinylora_projection": item.module.tinylora_projection.detach().cpu(),
        }
    return {"groups": groups, "modules": modules}


def _tinylora_basis_state(
    installed: Sequence[InstalledTinyLoRA],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        item.name: {
            "tinylora_a": item.module.tinylora_a.detach().cpu(),
            "tinylora_b": item.module.tinylora_b.detach().cpu(),
            "tinylora_projection": item.module.tinylora_projection.detach().cpu(),
        }
        for item in installed
    }


def _load_tinylora_state(
    installed: Sequence[InstalledTinyLoRA],
    state: Mapping[str, Any],
) -> None:
    expected = {item.name for item in installed}
    modules = state.get("modules")
    groups = state.get("groups")
    if not isinstance(modules, Mapping) or set(modules) != expected:
        raise ValueError(
            "Saved TinyLoRA module inventory does not match the current model"
        )
    if not isinstance(groups, Mapping):
        raise ValueError("Saved TinyLoRA state is missing parameter groups")
    loaded_groups: set[int] = set()
    for item in installed:
        module_state = modules[item.name]
        if (
            int(module_state["group_index"]) != item.module.group_index
            or int(module_state["module_index"]) != item.module.module_index
        ):
            raise ValueError("Saved TinyLoRA grouping does not match the current model")
        item.module.tinylora_a.copy_(module_state["tinylora_a"])
        item.module.tinylora_b.copy_(module_state["tinylora_b"])
        item.module.tinylora_projection.copy_(module_state["tinylora_projection"])
        if item.module.group_index not in loaded_groups:
            item.module.tinylora_v.data.copy_(groups[str(item.module.group_index)])
            loaded_groups.add(item.module.group_index)


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
