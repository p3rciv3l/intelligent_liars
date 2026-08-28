import sys
import types

import pytest

from intelligent_liars.models import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ModelBundle,
    ModelLoadConfig,
    load_model_and_processor,
    model_config_from_env,
    qwen_model_load_description,
    resolve_model_id,
)


TARGET_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"


def test_model_name_env_does_not_override_hardcoded_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_NAME", "some/other-model")

    assert resolve_model_id() == DEFAULT_MODEL_ID
    assert model_config_from_env().model_name == DEFAULT_MODEL_ID


def test_model_load_config_pins_exact_target_revision() -> None:
    config = ModelLoadConfig()

    assert DEFAULT_MODEL_REVISION == TARGET_REVISION
    assert config.revision == TARGET_REVISION
    assert config.attn_implementation == "flash_attention_2"
    assert config.device_map == "cuda:0"
    assert config.local_files_only is True
    assert config.use_cache is True


@pytest.mark.parametrize("revision", ["main", "latest", "0" * 40, None])
def test_model_load_config_rejects_floating_or_wrong_revision(revision: object) -> None:
    with pytest.raises(ValueError, match="Only target checkpoint revision"):
        ModelLoadConfig(revision=revision)  # type: ignore[arg-type]


def test_model_load_config_rejects_runtime_identity_drift() -> None:
    with pytest.raises(ValueError, match="attention implementation"):
        ModelLoadConfig(attn_implementation="eager")
    with pytest.raises(ValueError, match="device map"):
        ModelLoadConfig(device_map="auto")
    with pytest.raises(ValueError, match="local_files_only"):
        ModelLoadConfig(local_files_only=False)
    with pytest.raises(ValueError, match="use_cache"):
        ModelLoadConfig(use_cache=False)


def test_cuda_visible_devices_env_is_preserved_without_integer_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")

    config = model_config_from_env()

    assert config.gpu_ids is None
    assert config.cuda_visible_devices == "GPU-abc123"


def test_explicit_gpu_ids_are_serialized_to_cuda_visible_devices() -> None:
    config = model_config_from_env(gpu_ids=(0, 2))

    assert config.gpu_ids == (0, 2)
    assert config.cuda_visible_devices == "0,2"


def test_explicit_unsupported_model_name_fails() -> None:
    with pytest.raises(ValueError, match="Only Qwen/Qwen3-VL-8B-Thinking is supported"):
        resolve_model_id("some/other-model")


def test_qwen_model_load_description_mentions_flash_attention() -> None:
    description = qwen_model_load_description()

    assert "torch.bfloat16" in description
    assert 'device_map="cuda:0"' in description
    assert 'attn_implementation="flash_attention_2"' in description
    assert "local_files_only=true" in description
    assert "use_cache=true" in description
    assert f'revision="{TARGET_REVISION}"' in description


def test_processor_load_uses_exact_target_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    tokenizer = types.SimpleNamespace(
        pad_token_id=0,
        pad_token=None,
        eos_token="<eos>",
        padding_side="right",
    )
    processor = types.SimpleNamespace(tokenizer=tokenizer)

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return processor

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoProcessor=FakeAutoProcessor),
    )

    from intelligent_liars.models import load_processor

    bundle = load_processor(ModelLoadConfig(cache_dir="/tmp/hf-cache"))

    assert captured == {
        "model_id": DEFAULT_MODEL_ID,
        "kwargs": {
            "cache_dir": "/tmp/hf-cache",
            "revision": TARGET_REVISION,
            "local_files_only": True,
        },
    }
    assert bundle.model_revision == TARGET_REVISION
    assert bundle.model_identity == {
        "model_id": DEFAULT_MODEL_ID,
        "revision": TARGET_REVISION,
    }


def test_transformers_model_load_uses_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_bfloat16 = object()

    class FakeModel:
        eval_called = False

        def __init__(self) -> None:
            self.config = types.SimpleNamespace(use_cache=False)

        def eval(self) -> None:
            self.eval_called = True

    class FakeQwenModelClass:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return FakeModel()

    fake_transformers = types.SimpleNamespace(
        Qwen3VLForConditionalGeneration=FakeQwenModelClass,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(bfloat16=fake_bfloat16)
    )
    monkeypatch.setattr(
        "intelligent_liars.models.load_processor",
        lambda config: ModelBundle(
            model=None,
            processor=object(),
            tokenizer=object(),
            model_id=config.model_name,
            config=config,
        ),
    )

    bundle = load_model_and_processor(ModelLoadConfig(cache_dir="/tmp/hf-cache"))

    assert captured["model_id"] == DEFAULT_MODEL_ID
    assert captured["kwargs"] == {
        "cache_dir": "/tmp/hf-cache",
        "revision": TARGET_REVISION,
        "local_files_only": True,
        "device_map": "cuda:0",
        "dtype": fake_bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    assert "quantization_config" not in captured["kwargs"]
    assert bundle.model is not None
    assert bundle.model.eval_called is True
    assert bundle.model.config.use_cache is True
