"""Runtime hooks for teacher-forced scoring and cached generation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from intelligent_liars.interventions import (
    InterventionBundle,
    qwen_language_layers,
    transform_activations,
)
from intelligent_liars.step5_intervention_experiments.contracts import (
    InterventionExperimentError,
    TEACHER_FORCED_MASK_SEMANTICS,
)

__all__ = [
    "TEACHER_FORCED_MASK_SEMANTICS",
    "CachedGenerationLastTokenIntervention",
    "MaskedTeacherForcedIntervention",
    "RuntimeExecutionEvidence",
]


@dataclass(frozen=True)
class RuntimeExecutionEvidence:
    invocation_count: int
    token_count: int
    layer_invocations: Mapping[int, int]
    hooks_registered: int
    hooks_active: int


class _InterventionContext(AbstractContextManager["_InterventionContext"]):
    def __init__(self, model: Any, bundle: InterventionBundle | None) -> None:
        self.model = model
        self.bundle = bundle
        self.handles: list[Any] = []
        self.invocation_count = 0
        self.token_count = 0
        self._registered_count = 0
        self._layer_invocations: dict[int, int] = {}
        self._entered = False

    def _layers_and_spec(self) -> tuple[Any, Any, Any] | None:
        if self._entered:
            raise InterventionExperimentError("intervention context cannot be reused")
        self._entered = True
        if self.bundle is None:
            return None
        layers = qwen_language_layers(self.model)
        invalid = [index for index in self.bundle.spec.layers if index >= len(layers)]
        if invalid:
            raise InterventionExperimentError(
                f"intervention layers outside model: {invalid}"
            )
        self._layer_invocations = {index: 0 for index in self.bundle.spec.layers}
        return layers, self.bundle.effective_direction(), self.bundle.spec

    def _record(self, layer_index: int, token_count: int) -> None:
        self.invocation_count += 1
        self.token_count += token_count
        self._layer_invocations[layer_index] += 1

    @property
    def execution_evidence(self) -> RuntimeExecutionEvidence:
        return RuntimeExecutionEvidence(
            invocation_count=self.invocation_count,
            token_count=self.token_count,
            layer_invocations=dict(self._layer_invocations),
            hooks_registered=self._registered_count,
            hooks_active=len(self.handles),
        )

    def assert_executed(self) -> RuntimeExecutionEvidence:
        evidence = self.execution_evidence
        if self.bundle is None:
            if evidence.invocation_count or evidence.token_count:
                raise InterventionExperimentError("no-hook bypass recorded execution")
            return evidence
        missing = [
            index
            for index, invocation_count in evidence.layer_invocations.items()
            if invocation_count == 0
        ]
        if evidence.invocation_count == 0 or evidence.token_count == 0 or missing:
            raise InterventionExperimentError(
                f"intervention hook did not execute on layers: {missing}"
            )
        return evidence

    def __exit__(self, *exc_info: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _hidden_output(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise InterventionExperimentError(
            "Qwen hidden output must be [batch, sequence, hidden]"
        )
    return hidden


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


class MaskedTeacherForcedIntervention(_InterventionContext):
    """Intervene at positions predicting labels selected by ``answer_token_mask``."""

    mask_semantics = TEACHER_FORCED_MASK_SEMANTICS

    def __init__(
        self,
        model: Any,
        bundle: InterventionBundle | None,
        answer_token_mask: torch.Tensor,
    ) -> None:
        if answer_token_mask.ndim != 2 or answer_token_mask.dtype != torch.bool:
            raise InterventionExperimentError(
                "answer token mask must be bool [batch, sequence]"
            )
        if not bool(answer_token_mask.any()):
            raise InterventionExperimentError("answer token mask cannot be all false")
        super().__init__(model, bundle)
        label_mask = answer_token_mask.detach().clone()
        self.answer_token_mask = label_mask
        self.predictor_token_mask = torch.zeros_like(label_mask)
        self.predictor_token_mask[:, :-1] = label_mask[:, 1:]
        if not bool(self.predictor_token_mask.any()):
            raise InterventionExperimentError(
                "answer labels do not have any causal predictor positions"
            )

    def __enter__(self) -> MaskedTeacherForcedIntervention:
        runtime = self._layers_and_spec()
        if runtime is None:
            return self
        layers, direction, spec = runtime
        selected_tokens = int(self.predictor_token_mask.sum().item())
        for layer_index in spec.layers:

            def hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                _layer_index: int = layer_index,
            ) -> Any:
                hidden = _hidden_output(output)
                if tuple(hidden.shape[:2]) != tuple(self.predictor_token_mask.shape):
                    raise InterventionExperimentError(
                        "answer token mask does not match hidden output"
                    )
                mask = self.predictor_token_mask.to(hidden.device).unsqueeze(-1)
                transformed = transform_activations(hidden, direction, spec)
                if not torch.isfinite(transformed).all():
                    raise FloatingPointError(
                        "intervention produced non-finite activations"
                    )
                self._record(_layer_index, selected_tokens)
                return _replace_hidden(output, torch.where(mask, transformed, hidden))

            self.handles.append(layers[layer_index].register_forward_hook(hook))
        self._registered_count = len(self.handles)
        return self


class CachedGenerationLastTokenIntervention(_InterventionContext):
    """Intervene on the current last token during cached prefill and decode."""

    def __enter__(self) -> CachedGenerationLastTokenIntervention:
        runtime = self._layers_and_spec()
        if runtime is None:
            return self
        layers, direction, spec = runtime
        for layer_index in spec.layers:

            def hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                _layer_index: int = layer_index,
            ) -> Any:
                hidden = _hidden_output(output)
                if hidden.shape[1] < 1:
                    raise InterventionExperimentError(
                        "cached generation hidden sequence cannot be empty"
                    )
                transformed = transform_activations(hidden[:, -1:, :], direction, spec)
                if not torch.isfinite(transformed).all():
                    raise FloatingPointError(
                        "intervention produced non-finite activations"
                    )
                replaced = hidden.clone()
                replaced[:, -1:, :] = transformed
                self._record(_layer_index, int(hidden.shape[0]))
                return _replace_hidden(output, replaced)

            self.handles.append(layers[layer_index].register_forward_hook(hook))
        self._registered_count = len(self.handles)
        return self
