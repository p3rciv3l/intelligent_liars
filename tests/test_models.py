import sys
import types

import pytest

from intelligent_liars.models import (
    DEFAULT_MODEL_ID,
    ModelBundle,
    ModelLoadConfig,
    load_model_and_processor,
    model_config_from_env,
    qwen_model_load_description,
    qwen_model_load_kwargs,
    resolve_model_id,
)


def test_model_name_env_does_not_override_hardcoded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "some/other-model")

    assert resolve_model_id() == DEFAULT_MODEL_ID
    assert model_config_from_env().model_name == DEFAULT_MODEL_ID


def test_cuda_visible_devices_env_is_preserved_without_integer_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert "device_map=\"auto\"" in description
    assert "attn_implementation=\"flash_attention_2\"" in description


def test_qwen_model_load_kwargs_support_training_attention_and_placement() -> None:
    kwargs = qwen_model_load_kwargs(
        attention_implementation="sdpa",
        device_map=None,
        revision="abc123",
    )

    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["device_map"] is None
    assert kwargs["revision"] == "abc123"


def test_transformers_model_load_uses_flash_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fake_bfloat16 = object()

    class FakeModel:
        eval_called = False

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
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16=fake_bfloat16))
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
        "device_map": "auto",
        "dtype": fake_bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    assert bundle.model is not None
    assert bundle.model.eval_called is True
