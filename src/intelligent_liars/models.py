from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_MODEL_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
QWEN_ATTENTION_IMPLEMENTATION = "flash_attention_2"
QWEN_DTYPE_NAME = "torch.bfloat16"


@dataclass(frozen=True)
class ModelLoadConfig:
    model_name: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_MODEL_REVISION
    cache_dir: str | None = None
    gpu_ids: tuple[int, ...] | None = None
    cuda_visible_devices: str | None = None


@dataclass
class ModelBundle:
    model: Any | None
    processor: Any
    tokenizer: Any
    model_id: str
    config: ModelLoadConfig


def resolve_model_id(model_name: str | None = None) -> str:
    """Return the single supported Hugging Face model id."""
    model_id = model_name or DEFAULT_MODEL_ID
    if model_id != DEFAULT_MODEL_ID:
        raise ValueError(f"Only {DEFAULT_MODEL_ID} is supported for this project, got {model_id}.")
    return model_id


def model_config_from_env(
    *,
    model_name: str | None = None,
    cache_dir: str | None = None,
    gpu_ids: Sequence[int] | str | None = None,
) -> ModelLoadConfig:
    parsed_gpu_ids = _parse_gpu_ids(gpu_ids) if gpu_ids is not None else []
    cuda_visible_devices = ",".join(str(gpu_id) for gpu_id in parsed_gpu_ids) if parsed_gpu_ids else os.getenv("CUDA_VISIBLE_DEVICES")
    return ModelLoadConfig(
        model_name=resolve_model_id(model_name),
        revision=DEFAULT_MODEL_REVISION,
        cache_dir=cache_dir or os.getenv("HF_HOME") or None,
        gpu_ids=tuple(parsed_gpu_ids) if parsed_gpu_ids else None,
        cuda_visible_devices=cuda_visible_devices or None,
    )


def load_processor(config: ModelLoadConfig | None = None) -> ModelBundle:
    """Load only the processor/tokenizer; no model weights."""
    config = config or model_config_from_env()
    model_id = resolve_model_id(config.model_name)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=config.cache_dir,
        revision=config.revision,
    )
    tokenizer = _get_tokenizer(processor)

    _configure_tokenizer(tokenizer)
    return ModelBundle(
        model=None,
        processor=processor,
        tokenizer=tokenizer,
        model_id=model_id,
        config=config,
    )


def load_model_and_processor(config: ModelLoadConfig | None = None) -> ModelBundle:
    """Load Qwen3-VL through Transformers with its matching processor."""
    config = config or model_config_from_env()
    model_id = resolve_model_id(config.model_name)

    if config.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    bundle = load_processor(config)
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        **qwen_model_load_kwargs(
            cache_dir=config.cache_dir,
            revision=config.revision,
        ),
    )
    model.eval()
    bundle.model = model
    return bundle


def qwen_model_load_kwargs(
    *,
    cache_dir: str | None = None,
    revision: str = DEFAULT_MODEL_REVISION,
) -> dict[str, Any]:
    """Return the shared Qwen3-VL load kwargs for Transformers and NNsight."""
    import torch

    return {
        "cache_dir": cache_dir,
        "revision": revision,
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "attn_implementation": QWEN_ATTENTION_IMPLEMENTATION,
    }


def qwen_model_load_description() -> str:
    return (
        f'dtype="{QWEN_DTYPE_NAME}", '
        'device_map="auto", '
        f'attn_implementation="{QWEN_ATTENTION_IMPLEMENTATION}", '
        f'revision="{DEFAULT_MODEL_REVISION}"'
    )


def get_model_and_tokenizer(
    model_name: str | None = None,
    models_directory: str | None = None,
    omit_model: bool = False,
    gpu_ids: Sequence[int] | str | None = None,
) -> tuple[Any | None, Any]:
    """Compatibility wrapper matching Truth Spec's loader return shape."""
    config = model_config_from_env(
        model_name=model_name,
        cache_dir=models_directory,
        gpu_ids=gpu_ids,
    )
    bundle = load_processor(config) if omit_model else load_model_and_processor(config)
    return bundle.model, bundle.tokenizer


def _parse_gpu_ids(gpu_ids: Sequence[int] | str | None) -> list[int]:
    if gpu_ids is None:
        return []
    if isinstance(gpu_ids, str):
        if not gpu_ids.strip():
            return []
        return [int(part.strip()) for part in gpu_ids.split(",") if part.strip()]
    return [int(gpu_id) for gpu_id in gpu_ids]


def _get_tokenizer(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("Loaded processor does not expose a tokenizer.")
    return tokenizer


def _configure_tokenizer(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
