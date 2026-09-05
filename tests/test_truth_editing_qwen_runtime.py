from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from intelligent_liars.models import ModelBundle, ModelLoadConfig
from intelligent_liars.truth_editing_directions import (
    CompiledBasis,
    CompiledBasisSet,
    _basis_set_hash,
    vector_sha256,
)
from intelligent_liars.truth_editing_qwen_runtime import (
    QwenTrialRuntimeError,
    TrialExample,
    TrialRuntime,
    TrialRuntimeBatch,
    WriterStrengthPlan,
    compile_writer_edit,
)


MODEL_SHA = "a" * 64
MANIFEST_SHA = "b" * 64


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(3, 3, bias=False)


class FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(4, 3, bias=False)


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = FakeAttention()
        self.mlp = FakeMlp()


class FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([FakeLayer()])
        self.fail_generation = False
        self.forward_writer_snapshots: list[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        writer = self.model.language_model.layers[0].self_attn.o_proj.weight
        self.forward_writer_snapshots.append(writer.detach().clone())
        vocab = 32
        logits = torch.zeros((*input_ids.shape, vocab), device=input_ids.device)
        logits.scatter_(2, (input_ids % vocab).unsqueeze(-1), 2.0)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        if self.fail_generation:
            raise RuntimeError("synthetic generation failure")
        suffix = torch.tensor([[7, 8]] * input_ids.shape[0], device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


class FakeProcessor:
    chat_template = "frozen-template-v1"

    def __init__(self) -> None:
        self.tokenizer = self
        self.padding_side = "left"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert enable_thinking is False
        body = "|".join(f"{item['role']}:{item['content']}" for item in messages)
        return body + ("|assistant:<think>\n" if add_generation_prompt else "|end")

    def __call__(
        self,
        *,
        text: list[str],
        padding: bool,
        return_tensors: str | None,
        **_: object,
    ) -> dict[str, object]:
        rows = [[2 + (ord(character) % 20) for character in item] for item in text]
        if not padding:
            return {"input_ids": rows}
        width = max(len(row) for row in rows)
        padded = [[0] * (width - len(row)) + row for row in rows]
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}

    def batch_decode(self, rows: torch.Tensor, **_: object) -> list[str]:
        return [" ".join(str(int(token)) for token in row) for row in rows]


class FakePreservationCollector:
    identity = {"collector": "fake-preservation-v1"}

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.writer_snapshots: list[torch.Tensor] = []

    def collect(
        self, bundle: ModelBundle, batch: TrialRuntimeBatch
    ) -> dict[str, object]:
        del batch
        writer = bundle.model.model.language_model.layers[0].self_attn.o_proj.weight
        self.writer_snapshots.append(writer.detach().clone())
        if self.fail:
            raise RuntimeError("synthetic preservation failure")
        return {"strata": {"text": {"kl": 0.01}}, "edited_logits_collected": True}


def _basis_set() -> CompiledBasisSet:
    matrix = np.array([[1.0], [0.0], [0.0]], dtype="<f8")
    basis = CompiledBasis(
        direction_ids=("general-layer-0",),
        method="qr",
        requested_rank=1,
        matrix=matrix,
        basis_sha256=vector_sha256(matrix[:, 0]),
    )
    by_layer = ((0, basis),)
    return CompiledBasisSet(
        manifest_sha256=MANIFEST_SHA,
        method="qr",
        requested_rank=1,
        by_layer=by_layer,
        basis_set_sha256=_basis_set_hash(MANIFEST_SHA, "qr", 1, by_layer),
    )


def _batch(batch_id: str = "batch-1") -> TrialRuntimeBatch:
    return TrialRuntimeBatch(
        batch_id=batch_id,
        recipe_id="recipe-1",
        model_sha256=MODEL_SHA,
        basis_set=_basis_set(),
        strengths=WriterStrengthPlan(
            attention_by_layer={0: 1.0},
            mlp_by_layer={0: 0.5},
        ),
        examples=(
            TrialExample(
                record_id="record-1",
                messages=({"role": "user", "content": "What is 2+2?"},),
                truthful_target_text="4",
                false_target_text="5",
            ),
            TrialExample(
                record_id="record-2",
                messages=({"role": "user", "content": "Sky color?"},),
                truthful_target_text="blue",
                false_target_text="green",
            ),
        ),
        max_new_tokens=8,
    )


def _bundle(model: FakeQwen, processor: FakeProcessor) -> ModelBundle:
    bundle = ModelBundle(
        model=model,
        processor=processor,
        tokenizer=processor,
        model_id="Qwen/Qwen3-VL-8B-Thinking",
        config=ModelLoadConfig(),
    )
    bundle.verified_snapshot = {
        "model_id": "Qwen/Qwen3-VL-8B-Thinking",
        "revision": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
        "model_sha256": MODEL_SHA,
        "snapshot_manifest_sha256": MANIFEST_SHA,
    }
    return bundle


def test_compiles_basis_set_to_exact_layer_writer_edit() -> None:
    result = compile_writer_edit(
        recipe_id="recipe-1",
        model_sha256=MODEL_SHA,
        basis_set=_basis_set(),
        strengths=WriterStrengthPlan({0: (0.2,)}, {0: 1.7}),
    )
    assert result.recipe_id == "recipe-1"
    assert result.model_sha256 == MODEL_SHA
    assert len(result.layers) == 1
    assert result.layers[0].attention_strength == (0.2,)
    assert result.layers[0].mlp_strength == 1.7
    assert torch.equal(result.layers[0].basis, torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64))


def test_compiler_fails_closed_when_strength_layers_are_incomplete() -> None:
    with pytest.raises(QwenTrialRuntimeError, match="exactly cover"):
        compile_writer_edit(
            recipe_id="recipe-1",
            model_sha256=MODEL_SHA,
            basis_set=_basis_set(),
            strengths=WriterStrengthPlan({}, {0: 1.0}),
        )


def test_worker_loads_once_batches_inference_restores_and_persists(tmp_path: Path) -> None:
    model = FakeQwen()
    processor = FakeProcessor()
    original = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }
    loads = 0

    def loader(_config: ModelLoadConfig) -> ModelBundle:
        nonlocal loads
        loads += 1
        return _bundle(model, processor)

    preservation_collector = FakePreservationCollector()
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=preservation_collector,
        bundle_loader=loader,
        enforce_production_identity=False,
    )
    first = runtime.evaluate(_batch())
    second = runtime.evaluate(_batch("batch-2"))

    assert loads == 1
    assert not runtime.poisoned
    assert len(first.examples) == 2
    assert first.examples[0].generated_text == "7 8"
    assert first.examples[0].truthful_target_token_count > 0
    assert first.examples[0].false_target_token_count > 0
    assert first.examples[0].false_minus_truth_log_probability_margin == pytest.approx(
        first.examples[0].false_target_log_probability
        - first.examples[0].truthful_target_log_probability
    )
    assert first.telemetry["generated_tokens"] == 4
    assert first.telemetry["projection_restoration_verified"] == 1.0
    assert first.telemetry["projection_total_weight_delta_norm"] > 0.0
    assert first.projection_evidence
    assert all(
        item["exact_restoration_verified"] is True
        for item in first.projection_evidence
    )
    assert second.runtime_identity["routine_activation_hooks"] is False
    assert second.runtime_identity["revision"] == "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
    assert first.preservation.batch_sha256 == _batch().batch_sha256
    assert first.preservation.recipe_id == "recipe-1"
    assert first.preservation.model_sha256 == MODEL_SHA
    assert first.preservation.basis_set_sha256 == _basis_set().basis_set_sha256
    with pytest.raises(TypeError):
        first.preservation.evidence["strata"]["text"] = {"kl": 9.0}
    for name, value in model.named_parameters():
        assert torch.equal(value, original[name])
    assert not torch.equal(
        model.forward_writer_snapshots[0],
        original["model.language_model.layers.0.self_attn.o_proj.weight"],
    )
    assert not torch.equal(
        preservation_collector.writer_snapshots[0],
        original["model.language_model.layers.0.self_attn.o_proj.weight"],
    )
    assert torch.equal(
        preservation_collector.writer_snapshots[0],
        model.forward_writer_snapshots[0],
    )

    raw = json.loads(Path(first.raw_output_path).read_text(encoding="utf-8"))
    assert raw["self_sha256"] == first.self_sha256
    assert raw["batch_sha256"] == _batch().batch_sha256
    assert raw["projection_evidence"] == [dict(item) for item in first.projection_evidence]
    assert raw["logits_sha256"] == hashlib.sha256(
        Path(first.logits_path).read_bytes()
    ).hexdigest()
    with np.load(first.logits_path) as archive:
        assert archive["truthful_record_offsets"].shape == (3,)
        assert archive["false_record_offsets"].shape == (3,)
        assert archive["truthful_target_token_ids"].ndim == 1
        assert archive["truthful_target_token_logits"].shape == archive[
            "truthful_target_token_ids"
        ].shape
        assert not any("vocab" in name for name in archive.files)


def test_nonproduction_runtime_records_explicit_backend_identity(tmp_path: Path) -> None:
    model = FakeQwen()
    processor = FakeProcessor()
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(model, processor),
        enforce_production_identity=False,
        runtime_dtype_name="torch.bfloat16",
        runtime_device_map="mps:0",
        runtime_attention_implementation="eager",
    )
    result = runtime.evaluate(_batch())
    assert result.runtime_identity["device_map"] == "mps:0"
    assert result.runtime_identity["attention_implementation"] == "eager"


def test_production_runtime_rejects_backend_identity_override(tmp_path: Path) -> None:
    with pytest.raises(
        QwenTrialRuntimeError, match="production runtime identity cannot be overridden"
    ):
        TrialRuntime(
            verified_model_sha256=MODEL_SHA,
            verified_snapshot_manifest_sha256=MANIFEST_SHA,
            output_dir=tmp_path,
            preservation_collector=FakePreservationCollector(),
            runtime_device_map="mps:0",
        )


def test_inference_failure_restores_weights_and_poisons_worker(tmp_path: Path) -> None:
    model = FakeQwen()
    model.fail_generation = True
    processor = FakeProcessor()
    original = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(model, processor),
        enforce_production_identity=False,
    )

    with pytest.raises(QwenTrialRuntimeError, match="worker is poisoned"):
        runtime.evaluate(_batch())
    assert runtime.poisoned
    for name, value in model.named_parameters():
        assert torch.equal(value, original[name])
    with pytest.raises(QwenTrialRuntimeError, match="must be replaced"):
        runtime.evaluate(_batch("later"))


def test_preservation_failure_restores_weights_and_poisons_worker(tmp_path: Path) -> None:
    model = FakeQwen()
    original = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }
    collector = FakePreservationCollector(fail=True)
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=collector,
        bundle_loader=lambda _config: _bundle(model, FakeProcessor()),
        enforce_production_identity=False,
    )
    with pytest.raises(QwenTrialRuntimeError, match="worker is poisoned"):
        runtime.evaluate(_batch())
    assert collector.writer_snapshots
    assert runtime.poisoned
    for name, value in model.named_parameters():
        assert torch.equal(value, original[name])


def test_template_mutation_is_detected_before_another_edit(tmp_path: Path) -> None:
    model = FakeQwen()
    processor = FakeProcessor()
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(model, processor),
        enforce_production_identity=False,
    )
    runtime.evaluate(_batch())
    processor.chat_template = "changed-template"

    with pytest.raises(QwenTrialRuntimeError, match="identity changed"):
        runtime.evaluate(_batch("batch-2"))
    assert runtime.poisoned


def test_batch_rejects_wrong_verified_model_before_loading(tmp_path: Path) -> None:
    loads = 0

    def loader(_config: ModelLoadConfig) -> ModelBundle:
        nonlocal loads
        loads += 1
        return _bundle(FakeQwen(), FakeProcessor())

    runtime = TrialRuntime(
        verified_model_sha256="c" * 64,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=loader,
        enforce_production_identity=False,
    )
    with pytest.raises(QwenTrialRuntimeError, match="differs"):
        runtime.evaluate(_batch())
    assert loads == 0


def test_teacher_forcing_rejects_non_prefix_template_tokenization(tmp_path: Path) -> None:
    class NonPrefixProcessor(FakeProcessor):
        def __call__(self, **kwargs: object) -> dict[str, object]:
            result = super().__call__(**kwargs)
            texts = kwargs["text"]
            assert isinstance(texts, list)
            if any("|end" in item for item in texts):
                ids = result["input_ids"]
                if isinstance(ids, torch.Tensor):
                    for row in ids:
                        first = int(torch.nonzero(row, as_tuple=False)[0].item())
                        row[first] = 31
                else:
                    assert isinstance(ids, list)
                    for row in ids:
                        row[0] = 31
            return result

    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(FakeQwen(), NonPrefixProcessor()),
        enforce_production_identity=False,
    )
    with pytest.raises(QwenTrialRuntimeError, match="exact prefix"):
        runtime.evaluate(_batch())
    assert runtime.poisoned


def test_teacher_forcing_uses_closed_thinking_prefix_for_answer_scoring(
    tmp_path: Path,
) -> None:
    class ThinkingTemplateProcessor(FakeProcessor):
        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            assert tokenize is False
            assert enable_thinking is False
            if add_generation_prompt:
                body = "|".join(
                    f"{item['role']}:{item['content']}" for item in messages
                )
                return body + "|assistant:<think>\n"
            final = messages[-1]
            prompt = "|".join(
                f"{item['role']}:{item['content']}" for item in messages[:-1]
            )
            return (
                prompt
                + "|assistant:<think>\n\n</think>\n\n"
                + str(final["content"])
                + "|end"
            )

        def __call__(self, **kwargs: object) -> dict[str, object]:
            result = super().__call__(**kwargs)
            texts = kwargs["text"]
            assert isinstance(texts, list)
            ids = result["input_ids"]
            for index, text in enumerate(texts):
                if not text.endswith("<think>\n"):
                    continue
                if isinstance(ids, torch.Tensor):
                    ids[index, -1] = 31
                else:
                    assert isinstance(ids, list)
                    ids[index][-1] = 31
            return result

    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(
            FakeQwen(), ThinkingTemplateProcessor()
        ),
        enforce_production_identity=False,
    )

    result = runtime.evaluate(_batch())

    assert len(result.examples) == 2
    assert all(example.truthful_target_token_count > 0 for example in result.examples)
    assert all(example.false_target_token_count > 0 for example in result.examples)


def test_false_target_must_share_the_exact_frozen_truth_prompt_prefix(
    tmp_path: Path,
) -> None:
    class FalseOnlyPrefixMutationProcessor(FakeProcessor):
        def __call__(self, **kwargs: object) -> dict[str, object]:
            result = super().__call__(**kwargs)
            texts = kwargs["text"]
            assert isinstance(texts, list)
            ids = result["input_ids"]
            for index, text in enumerate(texts):
                if "assistant:5|end" not in text:
                    continue
                if isinstance(ids, torch.Tensor):
                    first = int(torch.nonzero(ids[index], as_tuple=False)[0].item())
                    ids[index, first] = 31
                else:
                    assert isinstance(ids, list)
                    ids[index][0] = 31
            return result

    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(
            FakeQwen(), FalseOnlyPrefixMutationProcessor()
        ),
        enforce_production_identity=False,
    )
    with pytest.raises(QwenTrialRuntimeError, match="exact prefix"):
        runtime.evaluate(_batch())


def test_dtos_freeze_caller_owned_messages_and_strengths() -> None:
    message = {"role": "user", "content": "original"}
    attention = {0: 1.0}
    example = TrialExample("record-1", (message,), "answer", "wrong")
    strengths = WriterStrengthPlan(attention, {0: 0.0})
    batch = TrialRuntimeBatch(
        "frozen-batch",
        "recipe-1",
        MODEL_SHA,
        _basis_set(),
        strengths,
        (example,),
    )
    identity = batch.batch_sha256

    message["content"] = "mutated"
    attention[0] = 2.0
    assert example.messages[0]["content"] == "original"
    assert strengths.attention_by_layer[0] == 1.0
    assert batch.batch_sha256 == identity


def test_repeated_content_address_reuses_verified_exact_pair(tmp_path: Path) -> None:
    model = FakeQwen()
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(model, FakeProcessor()),
        enforce_production_identity=False,
    )
    first = runtime.evaluate(_batch())
    original_raw = Path(first.raw_output_path).read_bytes()

    forwards = len(model.forward_writer_snapshots)
    second = runtime.evaluate(_batch())
    assert second.self_sha256 == first.self_sha256
    assert len(model.forward_writer_snapshots) == forwards
    assert Path(first.raw_output_path).read_bytes() == original_raw
    assert not runtime.poisoned


def test_incomplete_content_address_fails_before_model_load(tmp_path: Path) -> None:
    batch = _batch()
    artifact_dir = tmp_path / batch.batch_sha256
    artifact_dir.mkdir()
    (artifact_dir / "result.json").write_text("{}\n", encoding="utf-8")
    loads = 0

    def loader(_config: ModelLoadConfig) -> ModelBundle:
        nonlocal loads
        loads += 1
        return _bundle(FakeQwen(), FakeProcessor())

    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=loader,
        enforce_production_identity=False,
    )
    with pytest.raises(QwenTrialRuntimeError, match="incomplete or tampered"):
        runtime.evaluate(batch)
    assert loads == 0


def test_tampered_compact_evidence_fails_closed_on_restart(tmp_path: Path) -> None:
    first_runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(FakeQwen(), FakeProcessor()),
        enforce_production_identity=False,
    )
    result = first_runtime.evaluate(_batch())
    Path(result.logits_path).write_bytes(b"tampered")
    restarted = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(FakeQwen(), FakeProcessor()),
        enforce_production_identity=False,
    )
    with pytest.raises(QwenTrialRuntimeError, match="incomplete or tampered"):
        restarted.evaluate(_batch())


def test_loaded_snapshot_verification_is_not_replaced_by_caller_assertion(
    tmp_path: Path,
) -> None:
    bundle = _bundle(FakeQwen(), FakeProcessor())
    bundle.verified_snapshot = {
        **bundle.verified_snapshot,
        "snapshot_manifest_sha256": "c" * 64,
    }
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: bundle,
        enforce_production_identity=False,
    )
    with pytest.raises(QwenTrialRuntimeError, match="independently verified"):
        runtime.evaluate(_batch())
    assert not runtime.poisoned


def test_loaded_bundle_without_independent_snapshot_receipt_fails_closed(
    tmp_path: Path,
) -> None:
    bundle = _bundle(FakeQwen(), FakeProcessor())
    del bundle.verified_snapshot
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: bundle,
        enforce_production_identity=False,
    )
    with pytest.raises(QwenTrialRuntimeError, match="lacks independently verified"):
        runtime.evaluate(_batch())


def test_default_runtime_identity_rejects_non_bf16_non_cuda_fake(tmp_path: Path) -> None:
    runtime = TrialRuntime(
        verified_model_sha256=MODEL_SHA,
        verified_snapshot_manifest_sha256=MANIFEST_SHA,
        output_dir=tmp_path,
        preservation_collector=FakePreservationCollector(),
        bundle_loader=lambda _config: _bundle(FakeQwen(), FakeProcessor()),
    )
    with pytest.raises(QwenTrialRuntimeError, match="cuda:0"):
        runtime.evaluate(_batch())
