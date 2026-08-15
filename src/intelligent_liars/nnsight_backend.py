from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intelligent_liars.activation_backends import (
    ActivationIntervention,
    ActivationSurface,
    ActivationTraceResult,
    GenerationPolicy,
    GenerationSite,
    GenerationTraceResult,
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
    """NNsight backend for offline decoder activation reads and writes."""

    name = "nnsight"

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
                saved_logits = _save_selected_rows(
                    _nnsight_logits_proxy(self.model), selector
                )

        activations_by_layer: dict[int, Any] = {}
        for layer_idx in layers:
            activations_by_layer[layer_idx] = (
                _saved_value(saved_layers[layer_idx]).detach().cpu()
            )

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
        generation_policy: GenerationPolicy | None = None,
    ) -> Any:
        if generation_kwargs is not None:
            return self._generate_with_interventions(
                inputs=inputs,
                interventions=interventions,
                return_logits=return_logits,
                generation_kwargs=generation_kwargs,
                generation_policy=generation_policy,
            )

        if generation_policy is not None:
            raise ValueError("generation_policy requires generation_kwargs.")

        saved_output = None
        model_inputs = prepare_model_inputs(inputs, self.model)
        with self.model.trace(model_inputs, scan=False, validate=False):
            for intervention in interventions:
                if not intervention.enabled:
                    continue
                layer = qwen3_vl_text_decoder_layer(self.model, intervention.layer_idx)
                original_output = layer.output
                tensor = (
                    original_output[0]
                    if isinstance(original_output, tuple)
                    else original_output
                )
                edited_tensor = _apply_intervention(tensor, intervention)
                layer.output = (
                    (edited_tensor, *original_output[1:])
                    if isinstance(original_output, tuple)
                    else edited_tensor
                )
            saved_output = (
                _nnsight_logits_proxy(self.model).save()
                if return_logits
                else self.model.output.save()
            )

        return _saved_value(saved_output)

    def _generate_with_interventions(
        self,
        *,
        inputs: Mapping[str, Any],
        interventions: Sequence[ActivationIntervention],
        return_logits: bool,
        generation_kwargs: Mapping[str, Any],
        generation_policy: GenerationPolicy | None,
    ) -> GenerationTraceResult:
        """Generate text with explicitly gated, non-persistent activation writes."""
        _refuse_non_text_generation(inputs, generation_kwargs)
        if return_logits:
            raise ValueError(
                "Generation returns token IDs, not a stable logits trace; pass return_logits=False."
            )

        policy = generation_policy or GenerationPolicy()
        active_interventions = [
            intervention
            for intervention in interventions
            if intervention.enabled and policy.site is not GenerationSite.NONE
        ]
        _validate_generation_interventions(active_interventions)
        if active_interventions:
            _positive_max_new_tokens(generation_kwargs)
        model_inputs = prepare_model_inputs(inputs, self.model)
        prompt_width = _prompt_width(model_inputs)
        reasoning_end_token_ids = _reasoning_end_token_ids(self.tokenizer, policy)

        if active_interventions and policy.site is GenerationSite.POST_REASONING_TEXT:
            if generation_kwargs.get("do_sample", False):
                raise ValueError(
                    "POST_REASONING_TEXT uses deterministic boundary discovery and requires "
                    "do_sample=False."
                )
            baseline_sequences = self._run_generation(
                model_inputs=model_inputs,
                generation_kwargs=generation_kwargs,
                interventions=(),
            )
            boundary_token_count = _post_reasoning_boundary_token_count(
                baseline_sequences,
                prompt_width=prompt_width,
                reasoning_end_token_ids=reasoning_end_token_ids,
            )
            generated_count = int(baseline_sequences.shape[-1]) - prompt_width
            if boundary_token_count is None or boundary_token_count >= generated_count:
                return GenerationTraceResult(
                    sequences=baseline_sequences,
                    site=policy.site,
                    installed_intervention_count=0,
                )
            model_inputs = _post_reasoning_continuation_inputs(
                model_inputs,
                baseline_sequences=baseline_sequences,
                prompt_width=prompt_width,
                boundary_token_count=boundary_token_count,
            )
            generation_kwargs = {
                **generation_kwargs,
                "max_new_tokens": _positive_max_new_tokens(generation_kwargs)
                - boundary_token_count,
            }

        sequences = self._run_generation(
            model_inputs=model_inputs,
            generation_kwargs=generation_kwargs,
            interventions=active_interventions,
        )

        return GenerationTraceResult(
            sequences=sequences,
            site=policy.site,
            installed_intervention_count=len(active_interventions),
        )

    def _run_generation(
        self,
        *,
        model_inputs: Mapping[str, Any],
        generation_kwargs: Mapping[str, Any],
        interventions: Sequence[ActivationIntervention],
    ) -> Any:
        with self.model.generate(model_inputs, **dict(generation_kwargs)) as tracer:
            if interventions:
                with _all_generation_steps(tracer, self.model):
                    for intervention in interventions:
                        layer = qwen3_vl_text_decoder_layer(
                            self.model, intervention.layer_idx
                        )
                        original_output = layer.output
                        tensor = _nnsight_first_tensor(original_output)
                        edited_tensor = _apply_generation_intervention(
                            tensor, intervention
                        )
                        layer.output = (
                            (edited_tensor, *original_output[1:])
                            if isinstance(original_output, tuple)
                            else edited_tensor
                        )
            saved_sequences = self.model.generator.output.save()
        return _saved_value(saved_sequences)


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
        **qwen_model_load_kwargs(cache_dir=config.cache_dir),
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
        layer_output = qwen3_vl_text_decoder_layer(
            bundle.model, layer_idx
        ).output.save()

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
        raise RuntimeError(
            "Could not locate an NNsight logits proxy for Qwen3-VL."
        ) from exc


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


_ALLOWED_OFFLINE_GENERATION_KWARGS = frozenset(
    {
        "do_sample",
        "eos_token_id",
        "max_new_tokens",
        "pad_token_id",
        "temperature",
        "top_k",
        "top_p",
        "use_cache",
    }
)


def _refuse_non_text_generation(
    inputs: Mapping[str, Any], generation_kwargs: Mapping[str, Any]
) -> None:
    unsupported_inputs = sorted(set(inputs) - {"attention_mask", "input_ids"})
    unsupported_generation_kwargs = sorted(
        set(generation_kwargs) - _ALLOWED_OFFLINE_GENERATION_KWARGS
    )
    if unsupported_inputs or unsupported_generation_kwargs:
        raise ValueError(
            "This backend is restricted to offline text-only generation and refuses "
            "unsupported or tool/action/OSWorld pathways: "
            f"inputs={unsupported_inputs}, generation_kwargs={unsupported_generation_kwargs}."
        )


def _validate_generation_interventions(
    interventions: Sequence[ActivationIntervention],
) -> None:
    for intervention in interventions:
        if intervention.surface != "decoder":
            raise ValueError(
                "Generation interventions support only the text decoder surface."
            )
        if (
            intervention.detection_mask is not None
            or intervention.token_positions is not None
        ):
            raise ValueError(
                "Generation token selection is owned by GenerationPolicy; per-intervention "
                "masks and token positions are not accepted."
            )


def _all_generation_steps(tracer: Any, model: Any) -> Any:
    """Use the current NNsight API with compatibility for the pinned minimum."""
    tracer_all = getattr(tracer, "all", None)
    return tracer_all() if tracer_all is not None else model.all()


def _prompt_width(model_inputs: Mapping[str, Any]) -> int:
    input_ids = model_inputs.get("input_ids")
    if (
        input_ids is None
        or not hasattr(input_ids, "shape")
        or len(input_ids.shape) != 2
    ):
        raise ValueError("Generation requires rank-2 input_ids.")
    if int(input_ids.shape[0]) != 1:
        raise ValueError(
            "Offline activation-intervention generation currently requires batch size 1."
        )
    return int(input_ids.shape[-1])


def _reasoning_end_token_ids(
    tokenizer: Any, policy: GenerationPolicy
) -> tuple[int, ...]:
    if policy.site is not GenerationSite.POST_REASONING_TEXT:
        return ()
    encoded = tokenizer.encode(policy.reasoning_end_text, add_special_tokens=False)
    token_ids = tuple(int(token_id) for token_id in encoded)
    if not token_ids:
        raise ValueError("The tokenizer produced no IDs for the reasoning end marker.")
    return token_ids


def _post_reasoning_boundary_token_count(
    sequences: Any,
    *,
    prompt_width: int,
    reasoning_end_token_ids: tuple[int, ...],
) -> int | None:
    generated = sequences[0, prompt_width:].detach().cpu().tolist()
    marker_size = len(reasoning_end_token_ids)
    for start in range(len(generated) - marker_size + 1):
        if tuple(generated[start : start + marker_size]) == reasoning_end_token_ids:
            return start + marker_size
    return None


def _post_reasoning_continuation_inputs(
    model_inputs: Mapping[str, Any],
    *,
    baseline_sequences: Any,
    prompt_width: int,
    boundary_token_count: int,
) -> dict[str, Any]:
    """Continue from the observed marker prefix so reasoning is never replay-edited."""

    import torch

    input_ids = model_inputs["input_ids"]
    prefix_width = prompt_width + boundary_token_count
    continuation = dict(model_inputs)
    continuation["input_ids"] = baseline_sequences[:, :prefix_width].to(
        input_ids.device
    )
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        generated_mask = attention_mask.new_ones(
            (int(attention_mask.shape[0]), boundary_token_count)
        )
        continuation["attention_mask"] = torch.cat(
            (attention_mask, generated_mask), dim=-1
        )
    return continuation


def _positive_max_new_tokens(generation_kwargs: Mapping[str, Any]) -> int:
    max_new_tokens = generation_kwargs.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError(
            "Intervened generation requires a positive integer max_new_tokens."
        )
    return max_new_tokens


def _apply_generation_intervention(
    tensor: Any,
    intervention: ActivationIntervention,
) -> Any:
    selected = tensor[:, -1:, :]
    transformed = intervention.edit(selected)
    edited = tensor.clone()
    edited[:, -1:, :] = transformed
    return edited


def _saved_value(saved: Any) -> Any:
    value = getattr(saved, "value", saved)
    if isinstance(value, tuple):
        return value[0]
    return value
