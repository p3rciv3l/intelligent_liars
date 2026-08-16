from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from intelligent_liars.models import ModelBundle, ModelLoadConfig
from intelligent_liars.rollouts import GeneratedCompletion, GenerationSettings
from intelligent_liars.standalone_models import (
    FLEET_PLAN_FORMAT,
    TEACHER_DATASET_FORMAT,
    LoRALinear,
    LoRATrainingConfig,
    build_training_batch,
    claim_fleet_variant,
    create_fleet_plan,
    finish_fleet_claim,
    fleet_status,
    generate_teacher_dataset,
    load_fleet_plan,
    load_prompt_records,
    merge_lora,
    recover_running_fleet_variants,
    save_fleet_plan,
    train_standalone_model,
    weighted_causal_lm_loss,
)


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.padding_side = "left"

    def __call__(
        self, texts, *, padding, truncation, return_tensors, add_special_tokens
    ):
        del truncation, return_tensors, add_special_tokens
        rows = [[ord(char) % 251 + 1 for char in text] for text in texts]
        width = max(len(row) for row in rows) if padding else None
        padded = []
        masks = []
        for row in rows:
            amount = int(width or len(row)) - len(row)
            if self.padding_side == "right":
                padded.append([*row, *([0] * amount)])
                masks.append([*([1] * len(row)), *([0] * amount)])
            else:
                padded.append([*([0] * amount), *row])
                masks.append([*([0] * amount), *([1] * len(row))])
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(patch_size=16)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        continue_final_message=False,
    ):
        assert tokenize is False
        text = "".join(
            f"<{message['role']}>"
            + "".join(item.get("text", "<image>") for item in message["content"])
            for message in messages
        )
        if continue_final_message:
            return text
        return text + ("<assistant><think>\n" if add_generation_prompt else "")

    def __call__(
        self,
        *,
        text,
        images=None,
        videos=None,
        padding,
        return_tensors,
        **kwargs,
    ):
        del images, videos, kwargs
        return self.tokenizer(
            text,
            padding=padding,
            truncation=False,
            return_tensors=return_tensors,
            add_special_tokens=False,
        )

    def save_pretrained(self, path: Path) -> None:
        (path / "processor.json").write_text("processor")
        (path / "tokenizer_config.json").write_text("{}")


class TinyAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.o_proj(
            torch.tanh(self.q_proj(inputs) + self.k_proj(inputs) + self.v_proj(inputs))
        )


class TinyMlp(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.sigmoid(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


class TinyLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(hidden)
        self.mlp = TinyMlp(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.self_attn(inputs)
        return hidden + self.mlp(hidden)


class TinyQwen(nn.Module):
    def __init__(self, hidden: int = 8, vocab: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=nn.ModuleList([TinyLayer(hidden)]))
        )
        self.layers = self.model.language_model.layers
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(use_cache=True)

    def parameters(self, recurse: bool = True):
        yield from self.embed.parameters(recurse=recurse)
        yield from self.layers.parameters(recurse=recurse)
        yield from self.head.parameters(recurse=recurse)

    def get_input_embeddings(self):
        return self.embed

    def gradient_checkpointing_enable(self):
        return None

    def enable_input_require_grads(self):
        return None

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.head(hidden))

    def save_pretrained(self, path: Path, *, safe_serialization: bool):
        assert safe_serialization is True
        torch.save(self.state_dict(), path / "model.pt")
        (path / "config.json").write_text("{}")


def write_teacher(
    path: Path, *, content: str = "answer", intervention: str | None = "reflection.json"
) -> None:
    path.write_text(
        json.dumps(
            {
                "format": TEACHER_DATASET_FORMAT,
                "metadata": {
                    "base_model": "Qwen/Qwen3-VL-8B-Thinking",
                    "base_revision": None,
                    "intervention_path": intervention,
                    "intervention_sha256": "abc" if intervention is not None else None,
                },
                "records": [
                    {
                        "id": "0",
                        "messages": [{"role": "user", "content": "question"}],
                        "assistant_content": content,
                    }
                ],
            }
        )
    )


def test_load_prompt_records_accepts_rollout_input_messages(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            [
                {
                    "source_index": 4,
                    "input_messages": [{"role": "user", "content": "hello"}],
                }
            ]
        )
    )

    records = load_prompt_records(path)

    assert records == [
        {
            "id": "4",
            "messages": [{"role": "user", "content": "hello", "detect": False}],
            "metadata": {},
        }
    ]


def test_generate_teacher_dataset_is_resume_safe(monkeypatch, tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps(
            [
                {"id": "a", "messages": [{"role": "user", "content": "one"}]},
                {"id": "b", "messages": [{"role": "user", "content": "two"}]},
            ]
        )
    )
    output = tmp_path / "teacher.json"
    calls = []

    def fake_generate_completions(*, bundle, conversations, settings):
        del bundle, settings
        calls.extend(conversations)
        return [
            GeneratedCompletion(
                text="final", thinking="reason", raw_text="reason</think>final"
            )
            for _ in conversations
        ]

    monkeypatch.setattr(
        "intelligent_liars.standalone_models.generate_completions",
        fake_generate_completions,
    )
    bundle = ModelBundle(
        model=object(),
        processor=FakeProcessor(),
        tokenizer=FakeTokenizer(),
        model_id="Qwen/Qwen3-VL-8B-Thinking",
        config=ModelLoadConfig(),
    )
    settings = GenerationSettings(batch_size=1, max_new_tokens=4, do_sample=False)

    first = generate_teacher_dataset(
        model_bundle=bundle,
        prompt_path=prompts,
        output_path=output,
        generation=settings,
    )
    second = generate_teacher_dataset(
        model_bundle=bundle,
        prompt_path=prompts,
        output_path=output,
        generation=settings,
    )

    assert first.records == second.records == 2
    assert len(calls) == 2
    payload = json.loads(output.read_text())
    assert payload["records"][0]["assistant_content"] == "reason</think>final"


def test_lora_merge_preserves_wrapped_forward_result() -> None:
    base = nn.Linear(3, 2, bias=False)
    wrapped = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    wrapped.lora_b.data.fill_(0.25)
    inputs = torch.randn(4, 3)
    expected = wrapped(inputs)
    parent = nn.Module()
    parent.proj = wrapped
    installed = [
        SimpleNamespace(name="proj", parent=parent, attribute="proj", module=wrapped)
    ]

    merge_lora(installed)

    assert isinstance(parent.proj, nn.Linear)
    assert torch.allclose(parent.proj(inputs), expected, atol=1e-6)


def test_build_training_batch_masks_prompt_and_rejects_truncation() -> None:
    processor = FakeProcessor()
    records = [
        {
            "id": "x",
            "messages": [{"role": "user", "content": "question"}],
            "assistant_content": "answer",
            "loss_weight": 2.0,
        }
    ]

    inputs, labels, weights = build_training_batch(
        processor=processor,
        records=records,
        max_length=100,
        device="cpu",
    )

    supervised = labels[0] != -100
    assert supervised.sum().item() == len("answer<eos>")
    assert torch.equal(inputs["input_ids"][0][supervised], labels[0][supervised])
    assert weights.tolist() == [2.0]
    assert processor.tokenizer.padding_side == "left"
    with pytest.raises(ValueError, match="exceed max_length"):
        build_training_batch(
            processor=processor,
            records=records,
            max_length=4,
            device="cpu",
        )


def test_build_training_batch_supports_qwen_visual_message_blocks(monkeypatch) -> None:
    processor = FakeProcessor()
    captured = {}

    def process_vision_info(conversations, **kwargs):
        captured["conversations"] = conversations
        captured["kwargs"] = kwargs
        return [object()], None, {}

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=process_vision_info),
    )
    records = [
        {
            "id": "visual",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screen.png"},
                        {"type": "text", "text": "What action next?"},
                    ],
                }
            ],
            "assistant_content": "click(10, 20)",
        }
    ]

    _inputs, labels, _weights = build_training_batch(
        processor=processor,
        records=records,
        max_length=200,
        device="cpu",
    )

    assert (labels != -100).any()
    assert captured["kwargs"]["image_patch_size"] == 16
    assert captured["conversations"][0][0]["content"][0]["image"] == "screen.png"


def test_training_continues_assistant_prefix_without_starting_new_turn() -> None:
    processor = FakeProcessor()
    records = [
        {
            "id": "prefix",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer: ", "detect": False},
                {"role": "assistant", "content": "", "detect": True},
            ],
            "assistant_content": "B",
        }
    ]

    _inputs, labels, _weights = build_training_batch(
        processor=processor,
        records=records,
        max_length=200,
        device="cpu",
    )

    assert (labels != -100).sum().item() == len("B<eos>")


def test_weighted_causal_loss_scales_single_example_gradient_weight() -> None:
    logits = torch.tensor([[[5.0, -5.0], [0.0, 0.0], [0.0, 0.0]]])
    labels = torch.tensor([[-100, 0, -100]])
    base = weighted_causal_lm_loss(logits, labels, torch.tensor([1.0]))
    doubled = weighted_causal_lm_loss(logits, labels, torch.tensor([2.0]))

    assert doubled.item() == pytest.approx(2 * base.item())


def test_tiny_model_distillation_merges_lora_and_saves_stock_weights(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher.json"
    write_teacher(teacher)
    model = TinyQwen()
    bundle = ModelBundle(
        model=model,
        processor=FakeProcessor(),
        tokenizer=model,
        model_id="Qwen/Qwen3-VL-8B-Thinking",
        config=ModelLoadConfig(),
    )
    output = tmp_path / "student"

    summary = train_standalone_model(
        model_bundle=bundle,
        teacher_path=teacher,
        output_dir=output,
        config=LoRATrainingConfig(
            rank=2,
            alpha=2.0,
            epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_length=100,
            target_modules=("self_attn.q_proj", "mlp.down_proj"),
            gradient_checkpointing=False,
        ),
    )

    assert summary.optimizer_steps == 1
    assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(model.layers[0].mlp.down_proj, nn.Linear)
    manifest = json.loads((output / "standalone_model.json").read_text())
    assert manifest["standalone"] is True
    assert manifest["runtime_intervention_required"] is False
    assert (output / "model.pt").exists()
    assert not (output / "distillation_state.pt").exists()

    resumed = train_standalone_model(
        model_bundle=bundle,
        teacher_path=teacher,
        output_dir=output,
        config=LoRATrainingConfig(
            rank=2,
            alpha=2.0,
            epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_length=100,
            target_modules=("self_attn.q_proj", "mlp.down_proj"),
            gradient_checkpointing=False,
        ),
    )
    assert resumed == summary


def test_fleet_plan_round_trip(tmp_path: Path) -> None:
    intervention = tmp_path / "reflection.json"
    intervention.write_text(
        json.dumps(
            {
                "format": "qwen_truth_intervention_v1",
                "direction": {
                    "vector": [1.0, 0.0],
                    "intercept": 0.0,
                    "layer": 1,
                    "task": "general_domain",
                    "sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
                    "source_path": None,
                    "original_intercept": None,
                },
                "spec": {
                    "method": "full_reflection",
                    "layers": [1],
                    "token_scope": "last_token",
                    "direction_mode": "probe",
                    "random_seed": None,
                    "score_delta": 0.0,
                    "projection_target": 0.0,
                    "reflection_strength": 1.0,
                    "selected_side": "honest",
                    "remap_input_min": -1.0,
                    "remap_input_max": 1.0,
                    "remap_output_min": 1.0,
                    "remap_output_max": -1.0,
                    "margin": 1.0,
                    "max_score_delta": None,
                },
            }
        )
    )
    prompts = tmp_path / "prompts.json"
    prompts.write_text("[]")
    preservation = tmp_path / "preservation.json"
    write_teacher(preservation, intervention=None)
    path = tmp_path / "plan.json"
    plan = create_fleet_plan(
        intervention_paths=[intervention],
        prompt_path=prompts,
        preservation_teacher_path=preservation,
        output_root=tmp_path / "fleet",
        generation=GenerationSettings(max_new_tokens=2, do_sample=False),
        training=LoRATrainingConfig(epochs=1),
        base_revision="d" * 40,
        require_complete_suite=False,
    )

    save_fleet_plan(plan, path)
    loaded = load_fleet_plan(path)

    assert loaded == plan
    assert json.loads(path.read_text())["format"] == FLEET_PLAN_FORMAT

    tampered_path = tmp_path / "tampered-plan.json"
    tampered = json.loads(path.read_text())
    tampered["training"]["seed"] = 999
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="plan id"):
        load_fleet_plan(tampered_path)

    claim = claim_fleet_variant(loaded, worker_id="worker-1")
    assert claim is not None
    assert claim.variant.name == "reflection"
    assert claim_fleet_variant(loaded, worker_id="worker-2") is None
    finish_fleet_claim(loaded, claim, error=RuntimeError("boom"))
    assert fleet_status(loaded)["failed"] == ["reflection"]

    retry = claim_fleet_variant(loaded, retry_failed=True, worker_id="worker-2")
    assert retry is not None
    finish_fleet_claim(
        loaded,
        retry,
        artifacts={"teacher_output": "teacher.json", "model_output": "model"},
    )
    status = fleet_status(loaded)
    assert status["done"] == []
    assert status["corrupt"] == ["reflection"]
    assert status["failed"] == []

    second_plan_root = tmp_path / "fleet-2"
    second_plan = create_fleet_plan(
        intervention_paths=[intervention],
        prompt_path=prompts,
        preservation_teacher_path=preservation,
        output_root=second_plan_root,
        generation=GenerationSettings(max_new_tokens=2, do_sample=False),
        training=LoRATrainingConfig(epochs=1),
        base_revision="d" * 40,
        require_complete_suite=False,
    )
    stale = claim_fleet_variant(second_plan, worker_id="stale")
    assert stale is not None
    assert recover_running_fleet_variants(second_plan) == ["reflection"]
    replacement = claim_fleet_variant(
        second_plan,
        retry_failed=True,
        worker_id="replacement",
    )
    assert replacement is not None
    with pytest.raises(RuntimeError, match="ownership"):
        finish_fleet_claim(second_plan, stale)
