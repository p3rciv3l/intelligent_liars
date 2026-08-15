from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from intelligent_liars.models import ModelBundle


QWEN3_VL_DECODER_LAYER_PATH = "model.model.language_model.layers"
DECODER_ANSWER_TOKEN_SURFACE = "decoder_layer_{layer_idx}_answer_tokens"
NON_MODEL_PROCESSOR_KEYS = frozenset(
    {"offset_mapping", "overflow_to_sample_mapping", "detection_mask", "logit_mask"}
)
TEXT_MODEL_INPUT_KEYS = frozenset({"input_ids", "attention_mask"})


@dataclass(frozen=True)
class ActivationSurface:
    """Named model surface for extraction or intervention."""

    name: str
    path: str
    layer_idx: int | None = None
    token_selector: str = "answer_tokens"


@dataclass(frozen=True)
class ActivationIntervention:
    """A request to edit a traced activation surface."""

    layer_idx: int
    edit: Callable[[Any], Any]
    token_positions: Sequence[int] | None = None
    detection_mask: Any | None = None
    surface: str = "decoder"
    enabled: bool = True


class GenerationSite(str, Enum):
    """Offline text-generation sites supported by activation interventions."""

    NONE = "none"
    GENERATED_TEXT = "generated_text"
    POST_REASONING_TEXT = "post_reasoning_text"


@dataclass(frozen=True)
class GenerationPolicy:
    """Conservative token/site gate for offline autoregressive interventions.

    The default is a true no-op. Opted-in generation edits are restricted to
    the last causal token, which is the state used to predict the next token.
    Tool and action generation are intentionally outside this interface.
    """

    site: GenerationSite = GenerationSite.NONE
    reasoning_end_text: str = "</think>"

    def __post_init__(self) -> None:
        if not isinstance(self.site, GenerationSite):
            raise TypeError("site must be a GenerationSite value.")
        if (
            self.site is GenerationSite.POST_REASONING_TEXT
            and not self.reasoning_end_text
        ):
            raise ValueError(
                "POST_REASONING_TEXT requires a non-empty reasoning end marker."
            )


@dataclass(frozen=True)
class ActivationTraceResult:
    activations_by_layer: Mapping[int, Any]
    logits: Any | None = None
    surfaces: Mapping[int, ActivationSurface] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationTraceResult:
    """Generated token IDs and audit metadata from an offline text-only run."""

    sequences: Any
    site: GenerationSite
    installed_intervention_count: int


@runtime_checkable
class ActivationBackend(Protocol):
    """Backend boundary between masking/datasets and model tracing."""

    name: str
    model_id: str
    processor: Any
    tokenizer: Any

    def decoder_layer_count(self) -> int:
        """Return the number of text decoder layers exposed by this backend."""

    def surface_for_decoder_layer(self, layer_idx: int) -> ActivationSurface:
        """Return the canonical extraction/intervention surface for a layer."""

    def capture(
        self,
        *,
        inputs: Mapping[str, Any],
        detection_mask: Any,
        layers: Sequence[int],
        capture_logits: bool = False,
        logit_mask: Any | None = None,
    ) -> ActivationTraceResult:
        """Run the full model input and select hidden states after the forward pass."""

    def run_with_interventions(
        self,
        *,
        inputs: Mapping[str, Any],
        interventions: Sequence[ActivationIntervention],
        return_logits: bool = True,
        generation_kwargs: Mapping[str, Any] | None = None,
        generation_policy: GenerationPolicy | None = None,
    ) -> Any:
        """Run a traced forward or generation pass with activation writes applied."""


class TransformersHookBackend:
    """Activation backend using ordinary PyTorch forward hooks."""

    name = "transformers-hooks"

    def __init__(self, bundle: ModelBundle):
        if bundle.model is None:
            raise ValueError("TransformersHookBackend requires a loaded model.")
        self.model = bundle.model
        self.processor = bundle.processor
        self.tokenizer = bundle.tokenizer
        self.model_id = bundle.model_id
        self.config = bundle.config

    def decoder_layer_count(self) -> int:
        return len(qwen_decoder_layers(self.model))

    def surface_for_decoder_layer(self, layer_idx: int) -> ActivationSurface:
        return qwen3_vl_decoder_surface(layer_idx)

    def capture(
        self,
        *,
        inputs: Mapping[str, Any],
        detection_mask: Any,
        layers: Sequence[int],
        capture_logits: bool = False,
        logit_mask: Any | None = None,
    ) -> ActivationTraceResult:
        import torch

        decoder_layers = qwen_decoder_layers(self.model)
        captured: dict[int, Any] = {}
        handles = []

        def save_layer(layer_idx: int):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                hidden_states = _first_tensor(output)
                selector = detection_mask.to(hidden_states.device)
                captured[layer_idx] = hidden_states[selector].detach().cpu()

            return hook

        for layer_idx in layers:
            handles.append(
                decoder_layers[layer_idx].register_forward_hook(save_layer(layer_idx))
            )

        try:
            model_inputs = prepare_model_inputs(inputs, self.model)
            with torch.inference_mode():
                outputs = self.model(**model_inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        missing = [layer for layer in layers if layer not in captured]
        if missing:
            raise RuntimeError(f"Did not capture activations for layer(s): {missing}")

        logits = None
        if capture_logits:
            logits_tensor = _extract_logits(outputs)
            if logits_tensor is None:
                raise RuntimeError(
                    "capture_logits=True, but model output did not expose logits."
                )
            selector = detection_mask if logit_mask is None else logit_mask
            logits = logits_tensor[selector.to(logits_tensor.device)].detach().cpu()

        return ActivationTraceResult(
            activations_by_layer={layer: captured[layer] for layer in layers},
            logits=logits,
            surfaces={layer: self.surface_for_decoder_layer(layer) for layer in layers},
        )

    def run_with_interventions(
        self,
        *,
        inputs: Mapping[str, Any],
        interventions: Sequence[ActivationIntervention],
        return_logits: bool = True,
        generation_kwargs: Mapping[str, Any] | None = None,
        generation_policy: GenerationPolicy | None = None,
    ) -> Any:
        del inputs, interventions, return_logits, generation_kwargs, generation_policy
        raise NotImplementedError(
            "TransformersHookBackend is extraction-only. Use NnsightActivationBackend for activation writes."
        )


def qwen3_vl_decoder_surface(layer_idx: int) -> ActivationSurface:
    return ActivationSurface(
        name=DECODER_ANSWER_TOKEN_SURFACE.format(layer_idx=layer_idx),
        path=f"{QWEN3_VL_DECODER_LAYER_PATH}[{layer_idx}]",
        layer_idx=layer_idx,
    )


def qwen_decoder_layers(model: Any) -> Sequence[Any]:
    try:
        return model.model.language_model.layers
    except AttributeError as exc:
        raise TypeError(
            "Expected Qwen3-VL model path model.model.language_model.layers."
        ) from exc


def prepare_model_inputs(inputs: Mapping[str, Any], model: Any) -> dict[str, Any]:
    """Move text-only processor outputs to the model device."""
    candidate_inputs = {
        key: _move_to_model_device(value, model)
        for key, value in inputs.items()
        if key in TEXT_MODEL_INPUT_KEYS and key not in NON_MODEL_PROCESSOR_KEYS
    }
    accepted_keys = _accepted_forward_keys(model)
    if accepted_keys is None:
        return candidate_inputs
    return {
        key: value for key, value in candidate_inputs.items() if key in accepted_keys
    }


def _accepted_forward_keys(model: Any) -> set[str] | None:
    forward = getattr(model, "forward", None)
    if forward is None:
        return None
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return None
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return None
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def _move_to_model_device(value: Any, model: Any) -> Any:
    if not hasattr(value, "to"):
        return value
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return value
    return value.to(device)


def _first_tensor(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def _extract_logits(outputs: Any) -> Any | None:
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits
    if isinstance(outputs, Mapping):
        return outputs.get("logits")
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    return None
