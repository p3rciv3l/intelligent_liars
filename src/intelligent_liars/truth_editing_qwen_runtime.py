"""Production-shaped Qwen worker for persistent truth-editing trials.

This module owns one loaded model for the lifetime of a worker.  A trial
installs an already-qualified writer edit, performs deterministic batched
teacher forcing and generation, persists raw evidence, and restores the exact
writer bytes before returning.  It deliberately has no activation-hook path.

Model loading, input rendering, and time are injectable so the complete
transaction can be tested without a GPU, network access, or model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch

from .models import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    QWEN_ATTENTION_IMPLEMENTATION,
    QWEN_DEVICE_MAP,
    QWEN_DTYPE_NAME,
    ModelBundle,
    ModelLoadConfig,
    load_model_and_processor,
)
from .truth_editing_directions import CompiledBasisSet
from .truth_editing_component_basis import (
    CombinedComponentBasisSet,
    ComponentStrengthPlan,
    compile_component_writer_edit,
)
from .truth_editing_weight_editor import (
    CompiledLayerWriterEdit,
    CompiledWriterEdit,
    WriterEditRuntime,
)


RUNTIME_FORMAT = "truth_editing_qwen_trial_runtime_v2"
RESULT_FORMAT = "truth_editing_qwen_trial_result_v2"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class QwenTrialRuntimeError(RuntimeError):
    """The worker cannot safely evaluate another trial."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QwenTrialRuntimeError("runtime value is not canonical JSON") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _verified_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise QwenTrialRuntimeError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise QwenTrialRuntimeError(f"{label} must be a safe nonempty identifier")
    return value


@dataclass(frozen=True)
class TrialExample:
    """One conversation with frozen truthful and deliberately false continuations."""

    record_id: str
    messages: tuple[Mapping[str, Any], ...]
    truthful_target_text: str
    false_target_text: str

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        if not self.messages:
            raise QwenTrialRuntimeError("messages must not be empty")
        for label, value in (
            ("truthful_target_text", self.truthful_target_text),
            ("false_target_text", self.false_target_text),
        ):
            if not isinstance(value, str) or not value:
                raise QwenTrialRuntimeError(f"{label} must be a nonempty string")
        if self.truthful_target_text == self.false_target_text:
            raise QwenTrialRuntimeError("truthful and false targets must differ")
        payload = [dict(message) for message in self.messages]
        for message in payload:
            if set(message) != {"role", "content"}:
                raise QwenTrialRuntimeError("messages require exactly role and content")
            if message["role"] not in {"system", "user", "assistant"}:
                raise QwenTrialRuntimeError("message role is unsupported")
        _canonical_bytes(payload)
        object.__setattr__(
            self, "messages", tuple(MappingProxyType(message) for message in payload)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "messages": [dict(message) for message in self.messages],
            "truthful_target_text": self.truthful_target_text,
            "false_target_text": self.false_target_text,
        }


@dataclass(frozen=True)
class WriterStrengthPlan:
    """Per-layer diagonal strengths for attention and MLP writers."""

    attention_by_layer: Mapping[int, float | tuple[float, ...]]
    mlp_by_layer: Mapping[int, float | tuple[float, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attention_by_layer",
            MappingProxyType(_freeze_strengths(self.attention_by_layer, "attention")),
        )
        object.__setattr__(
            self,
            "mlp_by_layer",
            MappingProxyType(_freeze_strengths(self.mlp_by_layer, "MLP")),
        )


@dataclass(frozen=True)
class TrialRuntimeBatch:
    batch_id: str
    recipe_id: str
    model_sha256: str
    basis_set: CompiledBasisSet | CombinedComponentBasisSet
    strengths: WriterStrengthPlan | ComponentStrengthPlan
    examples: tuple[TrialExample, ...]
    max_new_tokens: int = 64
    _batch_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _identifier(self.batch_id, "batch_id")
        _identifier(self.recipe_id, "recipe_id")
        _verified_sha256(self.model_sha256, "model_sha256")
        self.basis_set.verify()
        if isinstance(self.basis_set, CombinedComponentBasisSet):
            if not isinstance(self.strengths, ComponentStrengthPlan):
                raise QwenTrialRuntimeError(
                    "component basis sets require an explicit component strength plan"
                )
            if self.basis_set.model_sha256 != self.model_sha256:
                raise QwenTrialRuntimeError(
                    "component basis model identity differs from runtime batch"
                )
        elif not isinstance(self.strengths, WriterStrengthPlan):
            raise QwenTrialRuntimeError(
                "legacy basis sets require a per-layer writer strength plan"
            )
        if not self.examples:
            raise QwenTrialRuntimeError("trial batch must contain examples")
        ids = tuple(example.record_id for example in self.examples)
        if len(set(ids)) != len(ids):
            raise QwenTrialRuntimeError("trial record IDs must be unique")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or not 1 <= self.max_new_tokens <= 2048
        ):
            raise QwenTrialRuntimeError(
                "max_new_tokens must be an integer in [1, 2048]"
            )
        object.__setattr__(self, "_batch_sha256", self._compute_sha256())

    @property
    def batch_sha256(self) -> str:
        return self._batch_sha256

    def _compute_sha256(self) -> str:
        return _sha256(
            {
                "batch_id": self.batch_id,
                "recipe_id": self.recipe_id,
                "model_sha256": self.model_sha256,
                "basis_set_sha256": self.basis_set.basis_set_sha256,
                "strengths": _strength_payload(self.strengths),
                "examples": [example.to_mapping() for example in self.examples],
                "max_new_tokens": self.max_new_tokens,
            }
        )


@dataclass(frozen=True)
class TrialExampleResult:
    record_id: str
    generated_text: str
    generated_token_ids: tuple[int, ...]
    generated_token_count: int
    truthful_target_token_count: int
    truthful_target_log_probability: float
    false_target_token_count: int
    false_target_log_probability: float
    false_minus_truth_log_probability_margin: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "generated_text": self.generated_text,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_token_count": self.generated_token_count,
            "truthful_target_token_count": self.truthful_target_token_count,
            "truthful_target_log_probability": self.truthful_target_log_probability,
            "false_target_token_count": self.false_target_token_count,
            "false_target_log_probability": self.false_target_log_probability,
            "false_minus_truth_log_probability_margin": (
                self.false_minus_truth_log_probability_margin
            ),
        }


@dataclass(frozen=True)
class LoadedModelVerification:
    """Identity independently established while resolving the local snapshot."""

    model_id: str
    revision: str
    model_sha256: str
    snapshot_manifest_sha256: str

    def __post_init__(self) -> None:
        _verified_sha256(self.model_sha256, "loaded model_sha256")
        _verified_sha256(
            self.snapshot_manifest_sha256, "loaded snapshot_manifest_sha256"
        )


@dataclass(frozen=True)
class TrialRuntimeResult:
    batch_id: str
    batch_sha256: str
    recipe_id: str
    model_sha256: str
    basis_set_sha256: str
    runtime_identity: Mapping[str, Any]
    examples: tuple[TrialExampleResult, ...]
    preservation: LeaseScopedPreservationResult
    raw_output_path: str
    logits_path: str
    logits_sha256: str
    telemetry: Mapping[str, Any]
    self_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": RESULT_FORMAT,
            "batch_id": self.batch_id,
            "batch_sha256": self.batch_sha256,
            "recipe_id": self.recipe_id,
            "model_sha256": self.model_sha256,
            "basis_set_sha256": self.basis_set_sha256,
            "runtime_identity": dict(self.runtime_identity),
            "examples": [example.to_mapping() for example in self.examples],
            "preservation": self.preservation.to_mapping(),
            "raw_output_path": self.raw_output_path,
            "logits_path": self.logits_path,
            "logits_sha256": self.logits_sha256,
            "telemetry": dict(self.telemetry),
            "self_sha256": self.self_sha256,
        }


class InputBuilder(Protocol):
    def __call__(
        self, processor: Any, conversations: Sequence[Sequence[Mapping[str, Any]]]
    ) -> tuple[list[str], Mapping[str, Any]]: ...


class BundleVerifier(Protocol):
    def __call__(self, bundle: ModelBundle) -> LoadedModelVerification: ...


class PreservationCollector(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def collect(
        self, bundle: ModelBundle, batch: TrialRuntimeBatch
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class LeaseScopedPreservationResult:
    batch_sha256: str
    recipe_id: str
    model_sha256: str
    basis_set_sha256: str
    collector_identity: Mapping[str, Any]
    evidence: Mapping[str, Any]
    self_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": "truth_editing_lease_scoped_preservation_v1",
            "batch_sha256": self.batch_sha256,
            "recipe_id": self.recipe_id,
            "model_sha256": self.model_sha256,
            "basis_set_sha256": self.basis_set_sha256,
            "collector_identity": _thaw_json_mapping(self.collector_identity),
            "evidence": _thaw_json_mapping(self.evidence),
            "self_sha256": self.self_sha256,
        }


def compile_writer_edit(
    *,
    recipe_id: str,
    model_sha256: str,
    basis_set: CompiledBasisSet | CombinedComponentBasisSet,
    strengths: WriterStrengthPlan | ComponentStrengthPlan,
) -> CompiledWriterEdit:
    """Bind a qualified per-layer basis set to writer strengths."""

    basis_set.verify()
    if isinstance(basis_set, CombinedComponentBasisSet):
        if not isinstance(strengths, ComponentStrengthPlan):
            raise QwenTrialRuntimeError(
                "component basis sets require an explicit component strength plan"
            )
        if basis_set.model_sha256 != model_sha256:
            raise QwenTrialRuntimeError(
                "component basis model identity differs from runtime batch"
            )
        try:
            return compile_component_writer_edit(
                recipe_id=recipe_id, basis_set=basis_set, strengths=strengths
            )
        except ValueError as error:
            raise QwenTrialRuntimeError(
                "component writer edit compilation failed"
            ) from error
    if not isinstance(strengths, WriterStrengthPlan):
        raise QwenTrialRuntimeError(
            "legacy basis sets require a per-layer writer strength plan"
        )
    layers = tuple(layer for layer, _ in basis_set.by_layer)
    if set(strengths.attention_by_layer) != set(layers):
        raise QwenTrialRuntimeError(
            "attention strengths must exactly cover basis layers"
        )
    if set(strengths.mlp_by_layer) != set(layers):
        raise QwenTrialRuntimeError("MLP strengths must exactly cover basis layers")
    compiled = tuple(
        CompiledLayerWriterEdit(
            layer_index=layer,
            basis=torch.from_numpy(np.array(basis.matrix, copy=True)),
            attention_strength=strengths.attention_by_layer[layer],
            mlp_strength=strengths.mlp_by_layer[layer],
        )
        for layer, basis in basis_set.by_layer
    )
    return CompiledWriterEdit(
        recipe_id=_identifier(recipe_id, "recipe_id"),
        model_sha256=_verified_sha256(model_sha256, "model_sha256"),
        layers=compiled,
    )


class TrialRuntime:
    """One-model-per-worker persistent edit and inference adapter."""

    def __init__(
        self,
        *,
        verified_model_sha256: str,
        verified_snapshot_manifest_sha256: str,
        output_dir: Path,
        preservation_collector: PreservationCollector,
        model_config: ModelLoadConfig | None = None,
        bundle_loader: Callable[
            [ModelLoadConfig], ModelBundle
        ] = load_model_and_processor,
        input_builder: InputBuilder | None = None,
        clock: Callable[[], float] = time.perf_counter,
        enforce_production_identity: bool = True,
        inference_microbatch_size: int = 4,
        bundle_verifier: BundleVerifier | None = None,
        runtime_dtype_name: str = QWEN_DTYPE_NAME,
        runtime_device_map: str = QWEN_DEVICE_MAP,
        runtime_attention_implementation: str = QWEN_ATTENTION_IMPLEMENTATION,
    ) -> None:
        self._verified_model_sha256 = _verified_sha256(
            verified_model_sha256, "verified_model_sha256"
        )
        self._output_dir = Path(output_dir)
        self._preservation_collector = preservation_collector
        self._verified_snapshot_manifest_sha256 = _verified_sha256(
            verified_snapshot_manifest_sha256,
            "verified_snapshot_manifest_sha256",
        )
        self._model_config = model_config or ModelLoadConfig()
        self._loader = bundle_loader
        self._input_builder = input_builder or _build_inputs
        self._clock = clock
        self._enforce_production_identity = enforce_production_identity
        runtime_fields = {
            "runtime_dtype_name": runtime_dtype_name,
            "runtime_device_map": runtime_device_map,
            "runtime_attention_implementation": runtime_attention_implementation,
        }
        if any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in runtime_fields.values()
        ):
            raise QwenTrialRuntimeError(
                "runtime identity overrides must be nonempty trimmed strings"
            )
        if enforce_production_identity and runtime_fields != {
            "runtime_dtype_name": QWEN_DTYPE_NAME,
            "runtime_device_map": QWEN_DEVICE_MAP,
            "runtime_attention_implementation": QWEN_ATTENTION_IMPLEMENTATION,
        }:
            raise QwenTrialRuntimeError(
                "production runtime identity cannot be overridden"
            )
        self._runtime_dtype_name = runtime_dtype_name
        self._runtime_device_map = runtime_device_map
        self._runtime_attention_implementation = runtime_attention_implementation
        if (
            isinstance(inference_microbatch_size, bool)
            or not isinstance(inference_microbatch_size, int)
            or inference_microbatch_size < 1
        ):
            raise QwenTrialRuntimeError(
                "inference_microbatch_size must be a positive integer"
            )
        self._microbatch_size = inference_microbatch_size
        self._bundle_verifier = bundle_verifier or _verification_from_bundle
        self._bundle: ModelBundle | None = None
        self._writer_runtime = WriterEditRuntime(
            verified_model_sha256=self._verified_model_sha256
        )
        self._template_identity: str | None = None
        self._load_seconds: float | None = None
        self._poisoned = False
        self._loaded_verification: LoadedModelVerification | None = None

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "format": RUNTIME_FORMAT,
            "model_id": DEFAULT_MODEL_ID,
            "revision": DEFAULT_MODEL_REVISION,
            "model_sha256": self._verified_model_sha256,
            "snapshot_manifest_sha256": self._verified_snapshot_manifest_sha256,
            "dtype": self._runtime_dtype_name,
            "device_map": self._runtime_device_map,
            "attention_implementation": self._runtime_attention_implementation,
            "local_files_only": True,
            "use_cache": True,
            "routine_activation_hooks": False,
            "teacher_forcing": "deterministic_full_sequence_log_probability",
            "generation": "greedy_do_sample_false",
            "inference_microbatch_size": self._microbatch_size,
            "preservation_collector": _owned_canonical_mapping(
                self._preservation_collector.identity,
                "preservation collector identity",
            ),
            "template_identity_sha256": self._template_identity,
        }

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _bundle_once(self) -> ModelBundle:
        if self._bundle is None:
            started = self._clock()
            bundle = self._loader(self._model_config)
            if bundle.model is None:
                raise QwenTrialRuntimeError("bundle loader returned no model")
            if (
                bundle.model_id != DEFAULT_MODEL_ID
                or bundle.model_revision != DEFAULT_MODEL_REVISION
            ):
                raise QwenTrialRuntimeError(
                    "loaded bundle identity differs from frozen Qwen checkpoint"
                )
            verification = self._bundle_verifier(bundle)
            if (
                verification.model_id != DEFAULT_MODEL_ID
                or verification.revision != DEFAULT_MODEL_REVISION
                or verification.model_sha256 != self._verified_model_sha256
                or verification.snapshot_manifest_sha256
                != self._verified_snapshot_manifest_sha256
            ):
                raise QwenTrialRuntimeError(
                    "independently verified loaded snapshot differs from frozen identity"
                )
            if self._enforce_production_identity:
                _verify_loaded_runtime(bundle.model)
            self._template_identity = _processor_identity(bundle)
            bundle.model.eval()
            self._load_seconds = self._clock() - started
            self._loaded_verification = verification
            self._bundle = bundle
        return self._bundle

    def evaluate(self, batch: TrialRuntimeBatch) -> TrialRuntimeResult:
        """Evaluate one batch and return only after exact writer restoration."""

        if self._poisoned:
            raise QwenTrialRuntimeError("worker is poisoned and must be replaced")
        if batch.model_sha256 != self._verified_model_sha256:
            raise QwenTrialRuntimeError(
                "batch model identity differs from verified worker model"
            )
        reused = self._reuse_result(batch)
        if reused is not None:
            return reused
        bundle = self._bundle_once()
        assert bundle.model is not None
        try:
            current_verification = self._bundle_verifier(bundle)
        except Exception as error:
            self._poisoned = True
            raise QwenTrialRuntimeError(
                "loaded snapshot identity can no longer be independently verified"
            ) from error
        if current_verification != self._loaded_verification:
            self._poisoned = True
            raise QwenTrialRuntimeError(
                "independently verified loaded snapshot identity changed"
            )
        if _processor_identity(bundle) != self._template_identity:
            self._poisoned = True
            raise QwenTrialRuntimeError(
                "processor/tokenizer/template identity changed in worker"
            )
        edit = compile_writer_edit(
            recipe_id=batch.recipe_id,
            model_sha256=batch.model_sha256,
            basis_set=batch.basis_set,
            strengths=batch.strengths,
        )
        started = self._clock()
        try:
            _reset_cuda_peak(bundle.model)
            with self._writer_runtime.activate(bundle.model, edit):
                example_results, evidence = self._infer(bundle, batch)
                preservation = self._collect_preservation(bundle, batch)
            elapsed = self._clock() - started
            return self._persist_result(
                batch,
                example_results,
                preservation,
                evidence,
                elapsed,
                bundle.model,
            )
        except Exception as error:
            self._poisoned = True
            if isinstance(error, QwenTrialRuntimeError):
                raise
            raise QwenTrialRuntimeError(
                "trial failed; writer restoration was attempted and worker is poisoned"
            ) from error

    def _collect_preservation(
        self, bundle: ModelBundle, batch: TrialRuntimeBatch
    ) -> LeaseScopedPreservationResult:
        identity = _owned_canonical_mapping(
            self._preservation_collector.identity, "preservation collector identity"
        )
        evidence = _owned_canonical_mapping(
            self._preservation_collector.collect(bundle, batch),
            "lease-scoped preservation evidence",
        )
        unsigned = {
            "format": "truth_editing_lease_scoped_preservation_v1",
            "batch_sha256": batch.batch_sha256,
            "recipe_id": batch.recipe_id,
            "model_sha256": batch.model_sha256,
            "basis_set_sha256": batch.basis_set.basis_set_sha256,
            "collector_identity": identity,
            "evidence": evidence,
        }
        return LeaseScopedPreservationResult(
            batch_sha256=batch.batch_sha256,
            recipe_id=batch.recipe_id,
            model_sha256=batch.model_sha256,
            basis_set_sha256=batch.basis_set.basis_set_sha256,
            collector_identity=_deep_freeze_mapping(identity),
            evidence=_deep_freeze_mapping(evidence),
            self_sha256=_sha256(unsigned),
        )

    def _infer(
        self, bundle: ModelBundle, batch: TrialRuntimeBatch
    ) -> tuple[tuple[TrialExampleResult, ...], Mapping[str, np.ndarray]]:
        all_results: list[TrialExampleResult] = []
        evidence_parts: dict[str, list[np.ndarray]] = {
            f"{kind}_{field}": []
            for kind in ("truthful", "false")
            for field in (
                "target_token_ids",
                "target_token_logits",
                "target_log_normalizers",
            )
        }
        offsets = {"truthful": [0], "false": [0]}
        for start in range(0, len(batch.examples), self._microbatch_size):
            chunk = batch.examples[start : start + self._microbatch_size]
            results, evidence = self._infer_chunk(bundle, batch, chunk)
            all_results.extend(results)
            for kind in ("truthful", "false"):
                for evidence_field in (
                    "target_token_ids",
                    "target_token_logits",
                    "target_log_normalizers",
                ):
                    evidence_parts[f"{kind}_{evidence_field}"].append(
                        evidence[f"{kind}_{evidence_field}"]
                    )
                chunk_offsets = evidence[f"{kind}_record_offsets"]
                offset_base = offsets[kind][-1]
                offsets[kind].extend(
                    offset_base + int(value) for value in chunk_offsets[1:]
                )
        combined = {key: np.concatenate(parts) for key, parts in evidence_parts.items()}
        for kind in ("truthful", "false"):
            combined[f"{kind}_record_offsets"] = np.asarray(
                offsets[kind], dtype=np.int64
            )
        return tuple(all_results), combined

    def _infer_chunk(
        self,
        bundle: ModelBundle,
        batch: TrialRuntimeBatch,
        examples: Sequence[TrialExample],
    ) -> tuple[tuple[TrialExampleResult, ...], Mapping[str, np.ndarray]]:
        model = bundle.model
        assert model is not None
        conversations = [example.messages for example in examples]
        _prompt_texts, generation_inputs = self._input_builder(
            bundle.processor, conversations
        )
        targets = [
            target
            for example in examples
            for target in (example.truthful_target_text, example.false_target_text)
        ]
        completed = [
            (*example.messages, {"role": "assistant", "content": target})
            for example in examples
            for target in (example.truthful_target_text, example.false_target_text)
        ]
        combined_texts, teacher_inputs = self._input_builder(
            bundle.processor, completed
        )
        blank_completed = [
            (*example.messages, {"role": "assistant", "content": ""})
            for example in examples
            for _target in (example.truthful_target_text, example.false_target_text)
        ]
        blank_texts = [
            bundle.processor.apply_chat_template(
                [dict(message) for message in conversation],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for conversation in blank_completed
        ]
        teacher_prompt_texts = [
            _teacher_scoring_prefix(combined_text, blank_text, target)
            for combined_text, blank_text, target in zip(
                combined_texts, blank_texts, targets, strict=True
            )
        ]
        teacher_prompt_inputs = _process_rendered_inputs(
            bundle.processor, teacher_prompt_texts, blank_completed
        )
        device = _model_device(model)
        generation_inputs = _move_inputs(generation_inputs, device)
        teacher_inputs = _move_inputs(teacher_inputs, device)
        teacher_prompt_inputs = _move_inputs(teacher_prompt_inputs, device)
        prompt_lengths = _input_lengths(bundle.processor, teacher_prompt_inputs)
        combined_lengths = _input_lengths(bundle.processor, teacher_inputs)
        with torch.inference_mode():
            outputs = model(**teacher_inputs)
            logits = getattr(outputs, "logits", None)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise QwenTrialRuntimeError(
                    "model forward did not return rank-3 logits"
                )
            scores, target_counts, paired_evidence = _target_log_probability_evidence(
                logits,
                teacher_inputs["input_ids"],
                teacher_prompt_inputs["input_ids"],
                prompt_lengths,
                combined_lengths,
            )
            del outputs, logits
            generated = model.generate(
                **generation_inputs,
                max_new_tokens=batch.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
            raise QwenTrialRuntimeError(
                "model generation did not return rank-2 token IDs"
            )
        prompt_width = int(generation_inputs["input_ids"].shape[1])
        suffix = generated[:, prompt_width:]
        generated_ids = tuple(
            _trim_generated_ids(
                suffix[index],
                eos_token_id=getattr(bundle.tokenizer, "eos_token_id", None),
                pad_token_id=getattr(bundle.tokenizer, "pad_token_id", None),
            )
            for index in range(suffix.shape[0])
        )
        decoded = bundle.processor.batch_decode(
            [list(ids) for ids in generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != len(examples):
            raise QwenTrialRuntimeError(
                "decoded generation count differs from trial batch"
            )
        results = tuple(
            TrialExampleResult(
                record_id=example.record_id,
                generated_text=str(decoded[index]),
                generated_token_ids=generated_ids[index],
                generated_token_count=len(generated_ids[index]),
                truthful_target_token_count=target_counts[index * 2],
                truthful_target_log_probability=scores[index * 2],
                false_target_token_count=target_counts[index * 2 + 1],
                false_target_log_probability=scores[index * 2 + 1],
                false_minus_truth_log_probability_margin=(
                    scores[index * 2 + 1] - scores[index * 2]
                ),
            )
            for index, example in enumerate(examples)
        )
        evidence = _split_paired_evidence(paired_evidence, len(examples))
        return results, evidence

    def _persist_result(
        self,
        batch: TrialRuntimeBatch,
        examples: tuple[TrialExampleResult, ...],
        preservation: LeaseScopedPreservationResult,
        evidence: Mapping[str, np.ndarray],
        elapsed: float,
        model: Any,
    ) -> TrialRuntimeResult:
        if not math.isfinite(elapsed) or elapsed < 0:
            raise QwenTrialRuntimeError("runtime clock returned invalid elapsed time")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = self._artifact_dir(batch)
        if artifact_dir.exists():
            reused = self._load_result(batch, artifact_dir)
            if reused is not None:
                return reused
            raise QwenTrialRuntimeError(
                "existing trial artifact is incomplete or tampered"
            )
        staging = self._output_dir / (
            f".{batch.batch_sha256}.staging-{os.getpid()}-{time.time_ns()}"
        )
        staging.mkdir(parents=False, exist_ok=False)
        logits_path = staging / "teacher_evidence.npz"
        raw_path = staging / "result.json"
        final_logits_path = artifact_dir / logits_path.name
        final_raw_path = artifact_dir / raw_path.name
        _atomic_npz(logits_path, **evidence)
        logits_sha256 = hashlib.sha256(logits_path.read_bytes()).hexdigest()
        generated_tokens = sum(example.generated_token_count for example in examples)
        telemetry = {
            "model_load_seconds": self._load_seconds,
            "evaluation_seconds": elapsed,
            "examples": len(examples),
            "generated_tokens": generated_tokens,
            "generated_tokens_per_second": generated_tokens / elapsed
            if elapsed > 0
            else None,
            "cuda_peak_allocated_bytes": _cuda_peak_bytes(model),
        }
        unsigned = {
            "format": RESULT_FORMAT,
            "batch_id": batch.batch_id,
            "batch_sha256": batch.batch_sha256,
            "recipe_id": batch.recipe_id,
            "model_sha256": batch.model_sha256,
            "basis_set_sha256": batch.basis_set.basis_set_sha256,
            "runtime_identity": dict(self.identity),
            "examples": [example.to_mapping() for example in examples],
            "preservation": preservation.to_mapping(),
            "raw_output_path": str(final_raw_path.resolve()),
            "logits_path": str(final_logits_path.resolve()),
            "logits_sha256": logits_sha256,
            "telemetry": telemetry,
        }
        self_sha = _sha256(unsigned)
        payload = dict(unsigned)
        payload["self_sha256"] = self_sha
        _atomic_json(raw_path, payload)
        try:
            staging.replace(artifact_dir)
        except OSError:
            if not artifact_dir.exists():
                raise
            shutil.rmtree(staging)
            reused = self._load_result(batch, artifact_dir)
            if reused is not None:
                return reused
            raise QwenTrialRuntimeError(
                "concurrent trial artifact is incomplete or tampered"
            )
        return TrialRuntimeResult(
            batch_id=batch.batch_id,
            batch_sha256=batch.batch_sha256,
            recipe_id=batch.recipe_id,
            model_sha256=batch.model_sha256,
            basis_set_sha256=batch.basis_set.basis_set_sha256,
            runtime_identity=self.identity,
            examples=examples,
            preservation=preservation,
            raw_output_path=str(final_raw_path.resolve()),
            logits_path=str(final_logits_path.resolve()),
            logits_sha256=logits_sha256,
            telemetry=telemetry,
            self_sha256=self_sha,
        )

    def _artifact_dir(self, batch: TrialRuntimeBatch) -> Path:
        return self._output_dir / batch.batch_sha256

    def _reuse_result(self, batch: TrialRuntimeBatch) -> TrialRuntimeResult | None:
        artifact_dir = self._artifact_dir(batch)
        if not artifact_dir.exists():
            return None
        result = self._load_result(batch, artifact_dir)
        if result is None:
            raise QwenTrialRuntimeError(
                "existing trial artifact is incomplete or tampered"
            )
        return result

    def _load_result(
        self, batch: TrialRuntimeBatch, artifact_dir: Path
    ) -> TrialRuntimeResult | None:
        raw_path = artifact_dir / "result.json"
        logits_path = artifact_dir / "teacher_evidence.npz"
        if artifact_dir.is_symlink() or not artifact_dir.is_dir():
            return None
        if (
            raw_path.is_symlink()
            or logits_path.is_symlink()
            or not raw_path.is_file()
            or not logits_path.is_file()
        ):
            return None
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            claimed_self = payload.pop("self_sha256")
            if _sha256(payload) != claimed_self:
                return None
            if payload.get("format") != RESULT_FORMAT:
                return None
            if payload.get("batch_sha256") != batch.batch_sha256:
                return None
            if payload.get("batch_id") != batch.batch_id:
                return None
            if payload.get("recipe_id") != batch.recipe_id:
                return None
            if payload.get("model_sha256") != batch.model_sha256:
                return None
            if payload.get("basis_set_sha256") != batch.basis_set.basis_set_sha256:
                return None
            runtime_identity = payload.get("runtime_identity")
            if not isinstance(runtime_identity, Mapping):
                return None
            required_runtime_identity = {
                "format": RUNTIME_FORMAT,
                "model_id": DEFAULT_MODEL_ID,
                "revision": DEFAULT_MODEL_REVISION,
                "model_sha256": self._verified_model_sha256,
                "snapshot_manifest_sha256": self._verified_snapshot_manifest_sha256,
                "dtype": self._runtime_dtype_name,
                "device_map": self._runtime_device_map,
                "attention_implementation": self._runtime_attention_implementation,
                "local_files_only": True,
                "use_cache": True,
                "routine_activation_hooks": False,
                "teacher_forcing": "deterministic_full_sequence_log_probability",
                "generation": "greedy_do_sample_false",
                "inference_microbatch_size": self._microbatch_size,
                "preservation_collector": _owned_canonical_mapping(
                    self._preservation_collector.identity,
                    "preservation collector identity",
                ),
            }
            if any(
                runtime_identity.get(key) != expected
                for key, expected in required_runtime_identity.items()
            ):
                return None
            if payload.get("raw_output_path") != str(raw_path.resolve()):
                return None
            if payload.get("logits_path") != str(logits_path.resolve()):
                return None
            logits_sha = hashlib.sha256(logits_path.read_bytes()).hexdigest()
            if payload.get("logits_sha256") != logits_sha:
                return None
            _verify_compact_evidence(
                logits_path,
                payload.get("examples"),
                tuple(example.record_id for example in batch.examples),
            )
            examples = tuple(
                _example_result_from_mapping(item) for item in payload["examples"]
            )
            preservation = _preservation_from_mapping(
                payload.get("preservation"), batch
            )
            if _thaw_json_mapping(preservation.collector_identity) != (
                _owned_canonical_mapping(
                    self._preservation_collector.identity,
                    "preservation collector identity",
                )
            ):
                return None
            return TrialRuntimeResult(
                batch_id=payload["batch_id"],
                batch_sha256=payload["batch_sha256"],
                recipe_id=payload["recipe_id"],
                model_sha256=payload["model_sha256"],
                basis_set_sha256=payload["basis_set_sha256"],
                runtime_identity=MappingProxyType(dict(payload["runtime_identity"])),
                examples=examples,
                preservation=preservation,
                raw_output_path=payload["raw_output_path"],
                logits_path=payload["logits_path"],
                logits_sha256=logits_sha,
                telemetry=MappingProxyType(dict(payload["telemetry"])),
                self_sha256=claimed_self,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
            QwenTrialRuntimeError,
        ):
            return None


def _strength_payload(
    plan: WriterStrengthPlan | ComponentStrengthPlan,
) -> dict[str, Any]:
    if isinstance(plan, ComponentStrengthPlan):
        return plan.to_mapping()

    def encode(values: Mapping[int, float | tuple[float, ...]]) -> dict[str, Any]:
        return {
            str(layer): list(value) if isinstance(value, tuple) else value
            for layer, value in sorted(values.items())
        }

    return {
        "attention_by_layer": encode(plan.attention_by_layer),
        "mlp_by_layer": encode(plan.mlp_by_layer),
    }


def _owned_canonical_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QwenTrialRuntimeError(f"{label} must be an object")
    try:
        cloned = json.loads(_canonical_bytes(dict(value)))
    except json.JSONDecodeError as error:  # pragma: no cover - canonical bytes are JSON
        raise QwenTrialRuntimeError(f"{label} cannot be frozen") from error
    if not isinstance(cloned, dict):
        raise QwenTrialRuntimeError(f"{label} must be an object")
    return cloned


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _deep_freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _deep_freeze_json(dict(value))
    assert isinstance(frozen, Mapping)
    return frozen


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _thaw_json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    thawed = _thaw_json(value)
    assert isinstance(thawed, dict)
    return thawed


def _preservation_from_mapping(
    value: Any, batch: TrialRuntimeBatch
) -> LeaseScopedPreservationResult:
    if not isinstance(value, Mapping):
        raise ValueError("preservation result must be an object")
    required = {
        "format",
        "batch_sha256",
        "recipe_id",
        "model_sha256",
        "basis_set_sha256",
        "collector_identity",
        "evidence",
        "self_sha256",
    }
    if set(value) != required:
        raise ValueError("preservation result fields differ")
    unsigned = {key: raw for key, raw in value.items() if key != "self_sha256"}
    if value["format"] != "truth_editing_lease_scoped_preservation_v1":
        raise ValueError("preservation result format differs")
    bindings = {
        "batch_sha256": batch.batch_sha256,
        "recipe_id": batch.recipe_id,
        "model_sha256": batch.model_sha256,
        "basis_set_sha256": batch.basis_set.basis_set_sha256,
    }
    if any(value.get(key) != expected for key, expected in bindings.items()):
        raise ValueError("preservation result binding differs")
    claimed = _verified_sha256(str(value["self_sha256"]), "preservation self_sha256")
    if _sha256(unsigned) != claimed:
        raise ValueError("preservation result identity differs")
    identity = _owned_canonical_mapping(
        value["collector_identity"], "collector identity"
    )
    evidence = _owned_canonical_mapping(value["evidence"], "preservation evidence")
    return LeaseScopedPreservationResult(
        batch_sha256=batch.batch_sha256,
        recipe_id=batch.recipe_id,
        model_sha256=batch.model_sha256,
        basis_set_sha256=batch.basis_set.basis_set_sha256,
        collector_identity=_deep_freeze_mapping(identity),
        evidence=_deep_freeze_mapping(evidence),
        self_sha256=claimed,
    )


def _freeze_strengths(
    values: Mapping[int, float | tuple[float, ...]], label: str
) -> dict[int, float | tuple[float, ...]]:
    if not isinstance(values, Mapping):
        raise QwenTrialRuntimeError(f"{label} strengths must be an object")
    result: dict[int, float | tuple[float, ...]] = {}
    for layer, raw in values.items():
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise QwenTrialRuntimeError(
                f"{label} strength layers must be nonnegative integers"
            )
        coefficients = raw if isinstance(raw, tuple) else (raw,)
        frozen: list[float] = []
        for coefficient in coefficients:
            if isinstance(coefficient, bool) or not isinstance(
                coefficient, (int, float)
            ):
                raise QwenTrialRuntimeError(
                    f"{label} strengths must be finite numbers in [0, 2]"
                )
            value = float(coefficient)
            if not math.isfinite(value) or not 0 <= value <= 2:
                raise QwenTrialRuntimeError(
                    f"{label} strengths must be finite numbers in [0, 2]"
                )
            frozen.append(value)
        if isinstance(raw, tuple):
            if not frozen:
                raise QwenTrialRuntimeError(
                    f"{label} per-basis strengths must not be empty"
                )
            result[layer] = tuple(frozen)
        else:
            result[layer] = frozen[0]
    return result


def _processor_identity(bundle: ModelBundle) -> str:
    tokenizer = bundle.tokenizer
    processor = bundle.processor
    return _sha256(
        {
            "processor_class": type(processor).__qualname__,
            "tokenizer_class": type(tokenizer).__qualname__,
            "chat_template": getattr(processor, "chat_template", None)
            or getattr(tokenizer, "chat_template", None),
            "padding_side": getattr(tokenizer, "padding_side", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
    )


def _verification_from_bundle(bundle: ModelBundle) -> LoadedModelVerification:
    raw = getattr(bundle, "verified_snapshot", None)
    if not isinstance(raw, Mapping):
        raise QwenTrialRuntimeError(
            "loaded bundle lacks independently verified snapshot identity"
        )
    required = {"model_id", "revision", "model_sha256", "snapshot_manifest_sha256"}
    if set(raw) != required:
        raise QwenTrialRuntimeError(
            "loaded bundle verified snapshot identity has unexpected fields"
        )
    try:
        return LoadedModelVerification(
            model_id=str(raw["model_id"]),
            revision=str(raw["revision"]),
            model_sha256=str(raw["model_sha256"]),
            snapshot_manifest_sha256=str(raw["snapshot_manifest_sha256"]),
        )
    except (TypeError, ValueError) as error:
        raise QwenTrialRuntimeError(
            "loaded bundle snapshot identity is invalid"
        ) from error


def _split_paired_evidence(
    evidence: Mapping[str, np.ndarray], example_count: int
) -> dict[str, np.ndarray]:
    offsets = evidence["record_offsets"]
    if offsets.shape != (example_count * 2 + 1,):
        raise QwenTrialRuntimeError("paired teacher evidence offsets are malformed")
    result: dict[str, np.ndarray] = {}
    for kind, parity in (("truthful", 0), ("false", 1)):
        kind_offsets = [0]
        for evidence_field in (
            "target_token_ids",
            "target_token_logits",
            "target_log_normalizers",
        ):
            pieces: list[np.ndarray] = []
            for index in range(parity, example_count * 2, 2):
                start, stop = int(offsets[index]), int(offsets[index + 1])
                pieces.append(evidence[evidence_field][start:stop])
                if evidence_field == "target_token_ids":
                    kind_offsets.append(kind_offsets[-1] + stop - start)
            result[f"{kind}_{evidence_field}"] = np.concatenate(pieces)
        result[f"{kind}_record_offsets"] = np.asarray(kind_offsets, dtype=np.int64)
    return result


def _example_result_from_mapping(value: Any) -> TrialExampleResult:
    if not isinstance(value, Mapping):
        raise ValueError("example result must be an object")
    required = {
        "record_id",
        "generated_text",
        "generated_token_ids",
        "generated_token_count",
        "truthful_target_token_count",
        "truthful_target_log_probability",
        "false_target_token_count",
        "false_target_log_probability",
        "false_minus_truth_log_probability_margin",
    }
    if set(value) != required:
        raise ValueError("example result fields differ")
    result = TrialExampleResult(
        record_id=_identifier(value["record_id"], "record_id"),
        generated_text=str(value["generated_text"]),
        generated_token_ids=tuple(int(item) for item in value["generated_token_ids"]),
        generated_token_count=int(value["generated_token_count"]),
        truthful_target_token_count=int(value["truthful_target_token_count"]),
        truthful_target_log_probability=float(value["truthful_target_log_probability"]),
        false_target_token_count=int(value["false_target_token_count"]),
        false_target_log_probability=float(value["false_target_log_probability"]),
        false_minus_truth_log_probability_margin=float(
            value["false_minus_truth_log_probability_margin"]
        ),
    )
    numbers = (
        result.truthful_target_log_probability,
        result.false_target_log_probability,
        result.false_minus_truth_log_probability_margin,
    )
    if not all(math.isfinite(item) for item in numbers):
        raise ValueError("example result probabilities must be finite")
    if result.generated_token_count != len(result.generated_token_ids):
        raise ValueError("generated token count is inconsistent")
    if result.truthful_target_token_count < 1 or result.false_target_token_count < 1:
        raise ValueError("target token counts must be positive")
    expected_margin = (
        result.false_target_log_probability - result.truthful_target_log_probability
    )
    if not math.isclose(
        result.false_minus_truth_log_probability_margin,
        expected_margin,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("example result margin is inconsistent")
    return result


def _verify_compact_evidence(
    path: Path, raw_examples: Any, expected_record_ids: tuple[str, ...]
) -> None:
    example_count = len(expected_record_ids)
    if not isinstance(raw_examples, list) or len(raw_examples) != example_count:
        raise ValueError("example evidence count differs")
    if not all(isinstance(raw, Mapping) for raw in raw_examples):
        raise ValueError("example evidence entries must be objects")
    if tuple(raw.get("record_id") for raw in raw_examples) != expected_record_ids:
        raise ValueError("example evidence record order differs")
    expected_names = {
        f"{kind}_{field}"
        for kind in ("truthful", "false")
        for field in (
            "record_offsets",
            "target_token_ids",
            "target_token_logits",
            "target_log_normalizers",
        )
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("teacher evidence fields differ")
        for kind in ("truthful", "false"):
            offsets = archive[f"{kind}_record_offsets"]
            ids = archive[f"{kind}_target_token_ids"]
            logits = archive[f"{kind}_target_token_logits"]
            normalizers = archive[f"{kind}_target_log_normalizers"]
            if offsets.shape != (example_count + 1,) or int(offsets[0]) != 0:
                raise ValueError("teacher evidence offsets are malformed")
            if np.any(np.diff(offsets) <= 0) or int(offsets[-1]) != ids.size:
                raise ValueError("teacher evidence offsets are inconsistent")
            if (
                ids.ndim != 1
                or logits.shape != ids.shape
                or normalizers.shape != ids.shape
            ):
                raise ValueError("teacher evidence vectors are misaligned")
            if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(normalizers)):
                raise ValueError("teacher evidence contains non-finite values")
            count_key = f"{kind}_target_token_count"
            logprob_key = f"{kind}_target_log_probability"
            for index, raw in enumerate(raw_examples):
                start, stop = int(offsets[index]), int(offsets[index + 1])
                if int(raw[count_key]) != stop - start:
                    raise ValueError("teacher evidence target count differs")
                score = float(np.sum(logits[start:stop] - normalizers[start:stop]))
                if not math.isclose(
                    score, float(raw[logprob_key]), rel_tol=0, abs_tol=1e-4
                ):
                    raise ValueError("teacher evidence log probability differs")


def _build_inputs(
    processor: Any, conversations: Sequence[Sequence[Mapping[str, Any]]]
) -> tuple[list[str], Mapping[str, Any]]:
    texts = [
        processor.apply_chat_template(
            [dict(message) for message in conversation],
            tokenize=False,
            add_generation_prompt=not (
                conversation and conversation[-1].get("role") == "assistant"
            ),
            enable_thinking=False,
        )
        for conversation in conversations
    ]
    return texts, _process_rendered_inputs(processor, texts, conversations)


def _process_rendered_inputs(
    processor: Any,
    texts: Sequence[str],
    conversations: Sequence[Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    contains_vision = any(
        isinstance(message.get("content"), Sequence)
        and not isinstance(message.get("content"), (str, bytes))
        for conversation in conversations
        for message in conversation
    )
    kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
    if contains_vision:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:  # pragma: no cover - production dependency
            raise QwenTrialRuntimeError(
                "multimodal batch requires qwen_vl_utils"
            ) from error
        images: list[Any] = []
        videos: list[Any] = []
        for conversation in conversations:
            image_inputs, video_inputs = process_vision_info(list(conversation))
            if image_inputs:
                images.extend(image_inputs)
            if video_inputs:
                videos.extend(video_inputs)
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
    inputs = processor(**kwargs)
    if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
        raise QwenTrialRuntimeError("processor did not return input_ids")
    return inputs


def _teacher_scoring_prefix(
    completed_text: str, blank_completed_text: str, target: str
) -> str:
    """Recover the closed-thinking answer prefix from the same full template.

    Qwen's generation prompt intentionally ends inside ``<think>``. A completed
    assistant message instead contains a closed thinking block before its answer.
    Comparing answer-scoring tokens to the open-thinking generation prompt is
    therefore invalid and can also trigger boundary-sensitive tokenization.
    """

    if not all(isinstance(value, str) for value in (
        completed_text, blank_completed_text, target
    )) or not target:
        raise QwenTrialRuntimeError("teacher-scoring template inputs are invalid")
    suffix_length = 0
    limit = min(len(completed_text), len(blank_completed_text))
    while (
        suffix_length < limit
        and completed_text[-suffix_length - 1]
        == blank_completed_text[-suffix_length - 1]
    ):
        suffix_length += 1
    suffix = (
        blank_completed_text[-suffix_length:] if suffix_length else ""
    )
    prefix = (
        blank_completed_text[:-suffix_length]
        if suffix_length
        else blank_completed_text
    )
    if completed_text != prefix + target + suffix:
        raise QwenTrialRuntimeError(
            "completed teacher template is not blank-prefix plus exact target"
        )
    return prefix


def _input_lengths(processor: Any, inputs: Mapping[str, Any]) -> tuple[int, ...]:
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise QwenTrialRuntimeError("processor input IDs must be rank two")
    attention_mask = inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        if attention_mask.shape != input_ids.shape:
            raise QwenTrialRuntimeError("processor attention mask is misaligned")
        lengths = attention_mask.to(dtype=torch.int64).sum(dim=1).tolist()
    else:
        pad_token_id = getattr(processor, "pad_token_id", None)
        if pad_token_id is None:
            tokenizer = getattr(processor, "tokenizer", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise QwenTrialRuntimeError(
                "processor cannot report unpadded token lengths"
            )
        lengths = input_ids.ne(int(pad_token_id)).sum(dim=1).tolist()
    if any(
        isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or length > input_ids.shape[1]
        for length in lengths
    ):
        raise QwenTrialRuntimeError("processor returned invalid token lengths")
    return tuple(int(length) for length in lengths)


def _target_log_probability_evidence(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    prompt_lengths: Sequence[int],
    combined_lengths: Sequence[int],
) -> tuple[tuple[float, ...], tuple[int, ...], Mapping[str, np.ndarray]]:
    if input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise QwenTrialRuntimeError("teacher logits and token IDs are not aligned")
    width = int(input_ids.shape[1])
    scores: list[float] = []
    counts: list[int] = []
    flat_target_ids: list[np.ndarray] = []
    flat_target_logits: list[np.ndarray] = []
    flat_log_normalizers: list[np.ndarray] = []
    offsets = [0]
    for row, (prompt_length, combined_length) in enumerate(
        zip(prompt_lengths, combined_lengths, strict=True)
    ):
        target_count = combined_length - prompt_length
        if target_count <= 0:
            raise QwenTrialRuntimeError("teacher target produced no new tokens")
        first_target = width - combined_length + prompt_length
        if first_target < 1 or combined_length > width:
            raise QwenTrialRuntimeError("teacher target is not causally scoreable")
        prompt_width = int(prompt_input_ids.shape[1])
        prompt_tokens = prompt_input_ids[row, prompt_width - prompt_length :]
        combined_prefix = input_ids[
            row, width - combined_length : width - combined_length + prompt_length
        ]
        if not torch.equal(prompt_tokens, combined_prefix):
            raise QwenTrialRuntimeError(
                "teacher-forcing prompt tokens are not an exact prefix of completed tokens"
            )
        prediction_positions = torch.arange(
            first_target - 1, width - 1, device=logits.device
        )
        target_positions = prediction_positions + 1
        token_ids = input_ids[row, target_positions]
        selected_logits = logits[row, prediction_positions, :].float()
        target_logits = selected_logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        log_normalizers = torch.logsumexp(selected_logits, dim=1)
        token_log_probs = target_logits - log_normalizers
        score = float(token_log_probs.sum().item())
        if not math.isfinite(score):
            raise QwenTrialRuntimeError("teacher forcing produced a non-finite score")
        scores.append(score)
        counts.append(target_count)
        flat_target_ids.append(
            token_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        )
        flat_target_logits.append(target_logits.detach().to(device="cpu").numpy())
        flat_log_normalizers.append(log_normalizers.detach().to(device="cpu").numpy())
        offsets.append(offsets[-1] + target_count)
        del selected_logits, target_logits, log_normalizers, token_log_probs
    evidence = {
        "record_offsets": np.asarray(offsets, dtype=np.int64),
        "target_token_ids": np.concatenate(flat_target_ids),
        "target_token_logits": np.concatenate(flat_target_logits).astype(
            np.float32, copy=False
        ),
        "target_log_normalizers": np.concatenate(flat_log_normalizers).astype(
            np.float32, copy=False
        ),
    }
    return tuple(scores), tuple(counts), evidence


def _trim_generated_ids(
    row: torch.Tensor, *, eos_token_id: Any, pad_token_id: Any
) -> tuple[int, ...]:
    values = [int(item) for item in row.detach().to(device="cpu").tolist()]
    result: list[int] = []
    for token_id in values:
        if pad_token_id is not None and token_id == int(pad_token_id):
            break
        result.append(token_id)
        if eos_token_id is not None and token_id == int(eos_token_id):
            break
    return tuple(result)


def _verify_loaded_runtime(model: Any) -> None:
    parameters = tuple(model.parameters())
    if not parameters:
        raise QwenTrialRuntimeError("loaded model exposes no parameters")
    devices = {str(parameter.device) for parameter in parameters}
    dtypes = {
        parameter.dtype
        for parameter in parameters
        if torch.is_floating_point(parameter)
    }
    if devices != {QWEN_DEVICE_MAP}:
        raise QwenTrialRuntimeError(
            f"loaded model parameters must all be on {QWEN_DEVICE_MAP}"
        )
    if dtypes != {torch.bfloat16}:
        raise QwenTrialRuntimeError(
            "loaded model floating parameters must all be torch.bfloat16"
        )
    implementation = getattr(
        getattr(model, "config", None), "_attn_implementation", None
    )
    if implementation != QWEN_ATTENTION_IMPLEMENTATION:
        raise QwenTrialRuntimeError(
            "loaded model attention implementation is not frozen"
        )


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise QwenTrialRuntimeError("cannot determine model input device") from error


def _move_inputs(inputs: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _reset_cuda_peak(model: Any) -> None:
    device = _model_device(model)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_bytes(model: Any) -> int | None:
    device = _model_device(model)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return int(torch.cuda.max_memory_allocated(device))
    return None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
    temporary.replace(path)


def build_trial_runtime(
    *,
    verified_model_sha256: str,
    verified_snapshot_manifest_sha256: str,
    output_dir: str | Path,
    preservation_collector: PreservationCollector,
) -> TrialRuntime:
    """Engine-compatible import factory with only explicit immutable inputs."""

    return TrialRuntime(
        verified_model_sha256=verified_model_sha256,
        verified_snapshot_manifest_sha256=verified_snapshot_manifest_sha256,
        output_dir=Path(output_dir),
        preservation_collector=preservation_collector,
    )


__all__ = [
    "QwenTrialRuntimeError",
    "LoadedModelVerification",
    "LeaseScopedPreservationResult",
    "PreservationCollector",
    "TrialExample",
    "TrialExampleResult",
    "TrialRuntime",
    "TrialRuntimeBatch",
    "TrialRuntimeResult",
    "WriterStrengthPlan",
    "build_trial_runtime",
    "compile_writer_edit",
]
