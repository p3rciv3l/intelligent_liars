from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_MODEL_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
DEFAULT_MODEL_CONTENT_SHA256 = (
    "bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8"
)
DEFAULT_SNAPSHOT_MANIFEST_SHA256 = (
    "3ba7152c45c7bda7526e1ccb64c949b2358aba430acf40b7390288d0df6ba5c5"
)
QWEN_ATTENTION_IMPLEMENTATION = "flash_attention_2"
QWEN_DTYPE_NAME = "torch.bfloat16"
QWEN_DEVICE_MAP = "cuda:0"


@dataclass(frozen=True)
class ModelLoadConfig:
    model_name: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_MODEL_REVISION
    attn_implementation: str = QWEN_ATTENTION_IMPLEMENTATION
    device_map: str = QWEN_DEVICE_MAP
    local_files_only: bool = True
    use_cache: bool = True
    cache_dir: str | None = None
    snapshot_manifest_path: str | None = None
    expected_model_sha256: str = DEFAULT_MODEL_CONTENT_SHA256
    expected_snapshot_manifest_sha256: str = DEFAULT_SNAPSHOT_MANIFEST_SHA256
    gpu_ids: tuple[int, ...] | None = None
    cuda_visible_devices: str | None = None

    def __post_init__(self) -> None:
        resolve_model_revision(self.revision)
        if self.attn_implementation != QWEN_ATTENTION_IMPLEMENTATION:
            raise ValueError(
                "Only attention implementation "
                f"{QWEN_ATTENTION_IMPLEMENTATION!r} is supported, "
                f"got {self.attn_implementation!r}."
            )
        if self.device_map != QWEN_DEVICE_MAP:
            raise ValueError(
                f"Only the explicit single-GPU device map {QWEN_DEVICE_MAP!r} "
                f"is supported, got {self.device_map!r}."
            )
        if self.local_files_only is not True:
            raise ValueError(
                "local_files_only must be true for pinned snapshot loading."
            )
        if self.use_cache is not True:
            raise ValueError(
                "use_cache must be true for the production inference runtime."
            )
        for label, value in (
            ("expected_model_sha256", self.expected_model_sha256),
            (
                "expected_snapshot_manifest_sha256",
                self.expected_snapshot_manifest_sha256,
            ),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256.")
        if self.snapshot_manifest_path is not None and self.cache_dir is None:
            raise ValueError("cache_dir is required with snapshot_manifest_path.")


@dataclass
class ModelBundle:
    model: Any | None
    processor: Any
    tokenizer: Any
    model_id: str
    config: ModelLoadConfig
    verified_snapshot: dict[str, str] | None = None

    @property
    def model_revision(self) -> str:
        return self.config.revision

    @property
    def model_identity(self) -> dict[str, str]:
        return self.verified_snapshot or {
            "model_id": self.model_id,
            "revision": self.model_revision,
        }


def resolve_model_id(model_name: str | None = None) -> str:
    """Return the single supported Hugging Face model id."""
    model_id = model_name or DEFAULT_MODEL_ID
    if model_id != DEFAULT_MODEL_ID:
        raise ValueError(
            f"Only {DEFAULT_MODEL_ID} is supported for this project, got {model_id}."
        )
    return model_id


def resolve_model_revision(revision: str | None = None) -> str:
    """Return the one immutable checkpoint revision supported by the runtime."""
    if revision != DEFAULT_MODEL_REVISION:
        raise ValueError(
            "Only target checkpoint revision "
            f"{DEFAULT_MODEL_REVISION} is supported for this project, got {revision!r}."
        )
    return revision


def model_config_from_env(
    *,
    model_name: str | None = None,
    cache_dir: str | None = None,
    gpu_ids: Sequence[int] | str | None = None,
) -> ModelLoadConfig:
    parsed_gpu_ids = _parse_gpu_ids(gpu_ids) if gpu_ids is not None else []
    cuda_visible_devices = (
        ",".join(str(gpu_id) for gpu_id in parsed_gpu_ids)
        if parsed_gpu_ids
        else os.getenv("CUDA_VISIBLE_DEVICES")
    )
    return ModelLoadConfig(
        model_name=resolve_model_id(model_name),
        cache_dir=cache_dir or os.getenv("HF_HOME") or None,
        snapshot_manifest_path=os.getenv("TRUTH_EDITING_MODEL_CACHE_MANIFEST") or None,
        gpu_ids=tuple(parsed_gpu_ids) if parsed_gpu_ids else None,
        cuda_visible_devices=cuda_visible_devices or None,
    )


def load_processor(config: ModelLoadConfig | None = None) -> ModelBundle:
    """Load only the processor/tokenizer; no model weights."""
    config = config or model_config_from_env()
    model_id = resolve_model_id(config.model_name)
    verified_snapshot = _verify_configured_snapshot(config)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=config.cache_dir,
        revision=resolve_model_revision(config.revision),
        local_files_only=config.local_files_only,
    )
    tokenizer = _get_tokenizer(processor)

    _configure_tokenizer(tokenizer)
    if _verify_configured_snapshot(config) != verified_snapshot:
        raise RuntimeError("model cache identity changed while loading the processor")
    return ModelBundle(
        model=None,
        processor=processor,
        tokenizer=tokenizer,
        model_id=model_id,
        config=config,
        verified_snapshot=verified_snapshot,
    )


def load_model_and_processor(config: ModelLoadConfig | None = None) -> ModelBundle:
    """Load Qwen3-VL through Transformers with its matching processor."""
    config = config or model_config_from_env()
    model_id = resolve_model_id(config.model_name)

    if config.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    bundle = load_processor(config)
    from transformers import Qwen3VLForConditionalGeneration

    model_kwargs = qwen_model_load_kwargs(
        cache_dir=config.cache_dir,
        revision=config.revision,
        attn_implementation=config.attn_implementation,
        device_map=config.device_map,
        local_files_only=config.local_files_only,
        use_cache=config.use_cache,
    )
    # Transformers 4.57 forwards unknown direct-class kwargs to Qwen3-VL's
    # constructor, whose signature does not accept ``use_cache``.  Preserve the
    # frozen setting on the loaded model config instead.
    model_kwargs.pop("use_cache")
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = config.use_cache
    if model.config.use_cache is not True:
        raise RuntimeError("Qwen model config did not retain use_cache=true")
    model.eval()
    if _verify_configured_snapshot(config) != bundle.verified_snapshot:
        raise RuntimeError("model cache identity changed while loading model weights")
    bundle.model = model
    return bundle


def qwen_model_load_kwargs(
    *,
    cache_dir: str | None = None,
    revision: str = DEFAULT_MODEL_REVISION,
    attn_implementation: str = QWEN_ATTENTION_IMPLEMENTATION,
    device_map: str = QWEN_DEVICE_MAP,
    local_files_only: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Return the shared Qwen3-VL load kwargs for Transformers and NNsight."""
    import torch

    config = ModelLoadConfig(
        revision=revision,
        attn_implementation=attn_implementation,
        device_map=device_map,
        local_files_only=local_files_only,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    return {
        "cache_dir": cache_dir,
        "revision": config.revision,
        "local_files_only": config.local_files_only,
        "device_map": config.device_map,
        "dtype": torch.bfloat16,
        "attn_implementation": config.attn_implementation,
        "use_cache": config.use_cache,
    }


def qwen_model_load_description() -> str:
    return (
        f'revision="{DEFAULT_MODEL_REVISION}", '
        f'dtype="{QWEN_DTYPE_NAME}", '
        f'device_map="{QWEN_DEVICE_MAP}", '
        f'attn_implementation="{QWEN_ATTENTION_IMPLEMENTATION}", '
        "local_files_only=true, use_cache=true"
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


def _verify_configured_snapshot(config: ModelLoadConfig) -> dict[str, str] | None:
    """Return the independently verified local snapshot identity when configured."""

    if config.snapshot_manifest_path is None:
        return None
    assert config.cache_dir is not None
    from .model_cache import verify_huggingface_cache_for_loading

    return verify_huggingface_cache_for_loading(
        cache_dir=Path(config.cache_dir),
        manifest_path=Path(config.snapshot_manifest_path),
        expected_model_sha256=config.expected_model_sha256,
        expected_manifest_sha256=config.expected_snapshot_manifest_sha256,
    ).to_mapping()
