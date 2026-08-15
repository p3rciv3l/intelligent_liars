import sys
import types

import pytest
import torch

from intelligent_liars.activation_backends import (
    ActivationIntervention,
    GenerationPolicy,
    GenerationSite,
)
from intelligent_liars.models import DEFAULT_MODEL_ID, ModelBundle, ModelLoadConfig
from intelligent_liars.nnsight_backend import NnsightActivationBackend, NnsightBundle, load_nnsight_bundle


def test_nnsight_model_load_uses_vlm_wrapper_and_flash_attention(monkeypatch):
    captured: dict[str, object] = {}
    fake_bfloat16 = object()
    fake_processor = object()

    class FakeVisionLanguageModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "nnsight",
        types.SimpleNamespace(VisionLanguageModel=FakeVisionLanguageModel),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16=fake_bfloat16))
    monkeypatch.setattr(
        "intelligent_liars.nnsight_backend.load_processor",
        lambda config: ModelBundle(
            model=None,
            processor=fake_processor,
            tokenizer=object(),
            model_id=config.model_name,
            config=config,
        ),
    )

    bundle = load_nnsight_bundle(ModelLoadConfig(cache_dir="/tmp/hf-cache"))

    assert captured["model_id"] == DEFAULT_MODEL_ID
    assert captured["kwargs"] == {
        "processor": fake_processor,
        "cache_dir": "/tmp/hf-cache",
        "device_map": "auto",
        "dtype": fake_bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    assert isinstance(bundle.model, FakeVisionLanguageModel)


class _Saved:
    def __init__(self, value):
        self.value = value


class _Proxy:
    def __init__(self, value):
        self.value = value

    def save(self):
        return _Saved(self.value)

    def __getitem__(self, key):
        return self.value[key]


class _Context:
    def __init__(self, model, generated_ids):
        self.model = model
        self.generated_ids = generated_ids

    def __enter__(self):
        self.model.generator = types.SimpleNamespace(output=_Proxy(self.generated_ids))
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    @property
    def iter(self):
        return self

    def __getitem__(self, key):
        self.model.iteration_slice = key
        return self


class _FakeNnsightModel:
    device = "cpu"

    def __init__(self, generated_ids):
        self.generated_ids = generated_ids
        self.generate_calls = []
        self.iteration_slice = None
        self.layer = types.SimpleNamespace(output=torch.tensor([[[1.0, 2.0]]]))
        self.model = types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=[self.layer])
        )
        self.lm_head = types.SimpleNamespace(output=_Proxy(torch.tensor([[[0.1, 0.9]]])))
        self.output = _Proxy(torch.tensor([[[0.1, 0.9]]]))

    def forward(self, input_ids, attention_mask=None):
        del input_ids, attention_mask

    def generate(self, inputs, **kwargs):
        self.generate_calls.append((inputs, kwargs))
        return _Context(self, self.generated_ids)

    def all(self):
        return _Context(self, self.generated_ids)


class _FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        assert text == "</think>"
        return [7, 8]


def _backend(generated_ids):
    model = _FakeNnsightModel(generated_ids)
    config = ModelLoadConfig()
    return NnsightActivationBackend(
        NnsightBundle(
            model=model,
            processor=object(),
            tokenizer=_FakeTokenizer(),
            model_id="fake-qwen",
            config=config,
        )
    )


def _inputs():
    return {
        "input_ids": torch.tensor([[101]]),
        "attention_mask": torch.tensor([[1]]),
    }


def test_generation_defaults_to_a_true_no_op_even_when_an_intervention_is_supplied():
    backend = _backend(torch.tensor([[101, 42]]))
    edit_calls = []

    result = backend.run_with_interventions(
        inputs=_inputs(),
        interventions=[
            ActivationIntervention(
                layer_idx=0,
                edit=lambda value: edit_calls.append(value) or value + 1000,
            )
        ],
        return_logits=False,
        generation_kwargs={"max_new_tokens": 1, "do_sample": False},
    )

    assert torch.equal(result.sequences, torch.tensor([[101, 42]]))
    assert torch.equal(backend.model.layer.output, torch.tensor([[[1.0, 2.0]]]))
    assert edit_calls == []
    assert result.installed_intervention_count == 0
    assert backend.model.generate_calls[0][1] == {
        "max_new_tokens": 1,
        "do_sample": False,
    }


def test_generated_text_site_edits_only_the_last_causal_token():
    backend = _backend(torch.tensor([[101, 42]]))

    backend.run_with_interventions(
        inputs=_inputs(),
        interventions=[ActivationIntervention(layer_idx=0, edit=lambda value: value + 3)],
        return_logits=False,
        generation_kwargs={"max_new_tokens": 1},
        generation_policy=GenerationPolicy(site=GenerationSite.GENERATED_TEXT),
    )

    assert torch.equal(backend.model.layer.output, torch.tensor([[[4.0, 5.0]]]))


def test_explicit_identity_edit_preserves_generation_and_activation_exactly():
    expected_sequences = torch.tensor([[101, 42]])
    backend = _backend(expected_sequences)

    result = backend.run_with_interventions(
        inputs=_inputs(),
        interventions=[ActivationIntervention(layer_idx=0, edit=lambda value: value)],
        return_logits=False,
        generation_kwargs={"max_new_tokens": 1, "do_sample": False},
        generation_policy=GenerationPolicy(site=GenerationSite.GENERATED_TEXT),
    )

    assert torch.equal(result.sequences, expected_sequences)
    assert torch.equal(backend.model.layer.output, torch.tensor([[[1.0, 2.0]]]))
    assert result.installed_intervention_count == 1


@pytest.mark.parametrize(
    ("generated_ids", "expected"),
    [
        (torch.tensor([[101, 7, 3]]), torch.tensor([[[1.0, 2.0]]])),
        (torch.tensor([[101, 7, 8, 3]]), torch.tensor([[[4.0, 5.0]]])),
    ],
)
def test_post_reasoning_site_requires_the_complete_end_think_marker(generated_ids, expected):
    backend = _backend(generated_ids)

    backend.run_with_interventions(
        inputs=_inputs(),
        interventions=[ActivationIntervention(layer_idx=0, edit=lambda value: value + 3)],
        return_logits=False,
        generation_kwargs={"max_new_tokens": 3},
        generation_policy=GenerationPolicy(site=GenerationSite.POST_REASONING_TEXT),
    )

    assert torch.equal(backend.model.layer.output, expected)
    if generated_ids.shape[-1] == 4:
        assert backend.model.iteration_slice == slice(2, 3)


@pytest.mark.parametrize(
    "unsafe_inputs, unsafe_kwargs",
    [
        ({"input_ids": torch.tensor([[101]]), "tools": [{"name": "click"}]}, {}),
        ({"input_ids": torch.tensor([[101]])}, {"tool_choice": "auto"}),
        ({"input_ids": torch.tensor([[101]])}, {"osworld": True}),
    ],
)
def test_generation_refuses_tool_action_and_osworld_pathways(unsafe_inputs, unsafe_kwargs):
    backend = _backend(torch.tensor([[101, 42]]))

    with pytest.raises(ValueError, match="offline text-only"):
        backend.run_with_interventions(
            inputs=unsafe_inputs,
            interventions=[],
            return_logits=False,
            generation_kwargs=unsafe_kwargs,
        )


def test_generation_policy_rejects_unsafe_token_positions():
    with pytest.raises(ValueError, match="last causal token"):
        GenerationPolicy(
            site=GenerationSite.GENERATED_TEXT,
            token_positions=(0,),
        )


def test_generation_policy_requires_a_named_safe_site():
    with pytest.raises(TypeError, match="GenerationSite"):
        GenerationPolicy(site="generated_text")  # type: ignore[arg-type]


def test_generation_refuses_ambiguous_logits_return_mode():
    backend = _backend(torch.tensor([[101, 42]]))

    with pytest.raises(ValueError, match="return_logits=False"):
        backend.run_with_interventions(
            inputs=_inputs(),
            interventions=[],
            return_logits=True,
            generation_kwargs={"max_new_tokens": 1},
        )


def test_post_reasoning_generation_refuses_sampling_before_running_model():
    backend = _backend(torch.tensor([[101, 7, 8, 42]]))

    with pytest.raises(ValueError, match="do_sample=False"):
        backend.run_with_interventions(
            inputs=_inputs(),
            interventions=[ActivationIntervention(layer_idx=0, edit=lambda value: value)],
            return_logits=False,
            generation_kwargs={"max_new_tokens": 3, "do_sample": True},
            generation_policy=GenerationPolicy(site=GenerationSite.POST_REASONING_TEXT),
        )

    assert backend.model.generate_calls == []
