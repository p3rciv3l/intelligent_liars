from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intelligent_liars.activation_backends import (
    ActivationIntervention,
    ActivationSurface,
    ActivationTraceResult,
    prepare_model_inputs,
    qwen3_vl_decoder_surface,
)
from intelligent_liars.models import (
    ModelLoadConfig,
    load_processor,
    model_config_from_env,
    qwen_model_load_kwargs,
    resolve_model_id,
)


@dataclass
class NnsightBundle:
    model: Any
    processor: Any
    tokenizer: Any
    model_id: str
    config: ModelLoadConfig


@dataclass(frozen=True)
class TraceSmokeResult:
    layer_idx: int
    input_shape: tuple[int, ...]
    layer_output_shape: tuple[int, ...]


class NnsightActivationBackend:
    """NNsight backend for decoder activation reads and future activation writes."""

    name = "nnsight"
    supports_generation_interventions = False

    def __init__(self, bundle: NnsightBundle):
        self.model = bundle.model
        self.processor = bundle.processor
        self.tokenizer = bundle.tokenizer
        self.model_id = bundle.model_id
        self.config = bundle.config

    def decoder_layer_count(self) -> int:
        return len(self.model.model.language_model.layers)

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
        saved_layers: dict[int, Any] = {}
        saved_logits = None

        model_inputs = prepare_model_inputs(inputs, self.model)
        with self.model.trace(model_inputs, scan=False, validate=False):
            for layer_idx in layers:
                saved_layers[layer_idx] = _save_selected_rows(
                    qwen3_vl_text_decoder_layer(self.model, layer_idx).output,
                    detection_mask,
                )
            if capture_logits:
                selector = detection_mask if logit_mask is None else logit_mask
                saved_logits = _save_selected_rows(_nnsight_logits_proxy(self.model), selector)

        activations_by_layer: dict[int, Any] = {}
        for layer_idx in layers:
            activations_by_layer[layer_idx] = _saved_value(saved_layers[layer_idx]).detach().cpu()

        logits = None
        if saved_logits is not None:
            logits = _saved_value(saved_logits).detach().cpu()

        return ActivationTraceResult(
            activations_by_layer=activations_by_layer,
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
    ) -> Any:
        if generation_kwargs is not None:
            raise NotImplementedError(
                "Generation-time NNsight interventions are not wired yet; "
                "this backend currently supports traced forward passes only."
            )

        saved_output = None
        model_inputs = prepare_model_inputs(inputs, self.model)
        with self.model.trace(model_inputs, scan=False, validate=False):
            for intervention in interventions:
                layer = qwen3_vl_text_decoder_layer(self.model, intervention.layer_idx)
                original_output = layer.output
                tensor = original_output[0] if isinstance(original_output, tuple) else original_output
                edited_tensor = _apply_intervention(tensor, intervention)
                layer.output = (
                    (edited_tensor, *original_output[1:])
                    if isinstance(original_output, tuple)
                    else edited_tensor
                )
            saved_output = _nnsight_logits_proxy(self.model).save() if return_logits else self.model.output.save()

        return _saved_value(saved_output)


def load_nnsight_bundle(config: ModelLoadConfig | None = None) -> NnsightBundle:
    """Load Qwen3-VL through NNsight and pair it with the official processor."""
    config = config or model_config_from_env()
    model_id = resolve_model_id(config.model_name)

    if config.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    from nnsight import VisionLanguageModel

    processor_bundle = load_processor(config)
    model = VisionLanguageModel(
        model_id,
        processor=processor_bundle.processor,
        **qwen_model_load_kwargs(
            cache_dir=config.cache_dir,
            revision=config.revision,
            attn_implementation=config.attn_implementation,
            device_map=config.device_map,
            local_files_only=config.local_files_only,
            use_cache=config.use_cache,
        ),
    )
    return NnsightBundle(
        model=model,
        processor=processor_bundle.processor,
        tokenizer=processor_bundle.tokenizer,
        model_id=model_id,
        config=config,
    )


def make_text_inputs(processor: Any, prompt: str) -> Mapping[str, Any]:
    """Build official Qwen3-VL processor inputs for a text-only prompt."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(text=[text], padding=True, return_tensors="pt")


def trace_text_decoder_layer_once(
    bundle: NnsightBundle,
    *,
    prompt: str = "Please answer with the single word: ready.",
    layer_idx: int = 0,
) -> TraceSmokeResult:
    """Trace one Qwen3-VL text decoder layer and return the saved activation shape."""
    inputs = make_text_inputs(bundle.processor, prompt)
    input_shape = tuple(inputs["input_ids"].shape)

    with bundle.model.trace(dict(inputs), scan=False, validate=False):
        layer_output = qwen3_vl_text_decoder_layer(bundle.model, layer_idx).output.save()

    layer_tensor = _saved_value(layer_output)
    return TraceSmokeResult(
        layer_idx=layer_idx,
        input_shape=input_shape,
        layer_output_shape=tuple(layer_tensor.shape),
    )


def qwen3_vl_text_decoder_layer(nnsight_model: Any, layer_idx: int) -> Any:
    """Return the NNsight proxy for a Qwen3-VL text decoder layer."""
    return nnsight_model.model.language_model.layers[layer_idx]


def _nnsight_logits_proxy(nnsight_model: Any) -> Any:
    try:
        return nnsight_model.lm_head.output
    except AttributeError:
        pass
    try:
        return nnsight_model.output.logits
    except AttributeError as exc:
        raise RuntimeError("Could not locate an NNsight logits proxy for Qwen3-VL.") from exc


def _nnsight_first_tensor(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def _save_selected_rows(output: Any, selector: Any) -> Any:
    tensor = _nnsight_first_tensor(output)
    return tensor[selector.to(tensor.device)].save()


def _apply_intervention(tensor: Any, intervention: ActivationIntervention) -> Any:
    if intervention.detection_mask is not None:
        selector = intervention.detection_mask.to(tensor.device)
        edited = tensor.clone()
        edited[selector] = intervention.edit(tensor[selector])
        return edited
    if intervention.token_positions is not None:
        edited = tensor.clone()
        positions = list(intervention.token_positions)
        edited[:, positions, :] = intervention.edit(tensor[:, positions, :])
        return edited
    return intervention.edit(tensor)


def _saved_value(saved: Any) -> Any:
    value = getattr(saved, "value", saved)
    if isinstance(value, tuple):
        return value[0]
    return value
