import sys
import types

from intelligent_liars.models import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ModelBundle,
    ModelLoadConfig,
)
from intelligent_liars.nnsight_backend import load_nnsight_bundle


def test_nnsight_model_load_uses_flash_attention(monkeypatch):
    captured: dict[str, object] = {}
    fake_bfloat16 = object()

    class FakeLanguageModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "nnsight", types.SimpleNamespace(LanguageModel=FakeLanguageModel))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16=fake_bfloat16))
    monkeypatch.setattr(
        "intelligent_liars.nnsight_backend.load_processor",
        lambda config: ModelBundle(
            model=None,
            processor=object(),
            tokenizer=object(),
            model_id=config.model_name,
            config=config,
        ),
    )

    bundle = load_nnsight_bundle(ModelLoadConfig(cache_dir="/tmp/hf-cache"))

    assert captured["model_id"] == DEFAULT_MODEL_ID
    assert captured["kwargs"] == {
        "cache_dir": "/tmp/hf-cache",
        "revision": DEFAULT_MODEL_REVISION,
        "device_map": "auto",
        "dtype": fake_bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    assert isinstance(bundle.model, FakeLanguageModel)
