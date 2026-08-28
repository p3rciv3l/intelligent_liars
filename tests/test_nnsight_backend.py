from types import SimpleNamespace

import pytest

from intelligent_liars.models import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ModelBundle,
    ModelLoadConfig,
)
from intelligent_liars.nnsight_backend import NnsightActivationBackend, load_nnsight_bundle


def test_nnsight_model_load_uses_installed_vision_language_model(monkeypatch):
    nnsight = pytest.importorskip("nnsight")
    captured: dict[str, object] = {}
    fake_bfloat16 = object()
    processor = object()
    tokenizer = object()

    def capture_init(self, model_id: str, **kwargs: object) -> None:
        captured["class"] = type(self)
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs

    monkeypatch.setattr(nnsight.VisionLanguageModel, "__init__", capture_init)
    monkeypatch.setattr("torch.bfloat16", fake_bfloat16)
    monkeypatch.setattr(
        "intelligent_liars.nnsight_backend.load_processor",
        lambda config: ModelBundle(
            model=None,
            processor=processor,
            tokenizer=tokenizer,
            model_id=config.model_name,
            config=config,
        ),
    )

    bundle = load_nnsight_bundle(ModelLoadConfig(cache_dir="/tmp/hf-cache"))

    assert captured["class"] is nnsight.VisionLanguageModel
    assert captured["model_id"] == DEFAULT_MODEL_ID
    assert captured["kwargs"] == {
        "processor": processor,
        "cache_dir": "/tmp/hf-cache",
        "revision": DEFAULT_MODEL_REVISION,
        "local_files_only": True,
        "device_map": "cuda:0",
        "dtype": fake_bfloat16,
        "attn_implementation": "flash_attention_2",
        "use_cache": True,
    }
    assert isinstance(bundle.model, nnsight.VisionLanguageModel)
    assert bundle.processor is processor
    assert bundle.tokenizer is tokenizer


def test_nnsight_backend_explicitly_rejects_even_empty_generation_request():
    backend = NnsightActivationBackend(
        SimpleNamespace(
            model=object(),
            processor=object(),
            tokenizer=object(),
            model_id=DEFAULT_MODEL_ID,
            config=ModelLoadConfig(),
        )
    )

    assert backend.supports_generation_interventions is False
    with pytest.raises(NotImplementedError, match="Generation-time NNsight interventions"):
        backend.run_with_interventions(
            inputs={},
            interventions=(),
            generation_kwargs={},
        )
