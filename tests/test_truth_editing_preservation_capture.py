from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from intelligent_liars.models import ModelBundle, ModelLoadConfig
from intelligent_liars.truth_editing_preservation_capture import (
    BaselineCaptureOutput,
    PreservationBaselineCaptureError,
    VerifiedQwenBaselineCaptureBackend,
    capture_preservation_baselines,
    open_preservation_baseline_capture,
)
from intelligent_liars.tinylora_pilot import topk_preservation_kl_loss


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/capture_truth_editing_preservation_baselines.py"
    )
    spec = importlib.util.spec_from_file_location("preservation_capture_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_plan(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    image = source / "pixel.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(image)
    inputs = [
        {
            "messages": [
                {"role": "user", "content": "Add one and one."},
                {"role": "assistant", "content": "Two."},
            ],
            "media": [],
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "media_id": "pixel"},
                        {"type": "text", "text": "Name the color."},
                    ],
                },
                {"role": "assistant", "content": "Dark blue."},
            ],
            "media": [
                {
                    "media_id": "pixel",
                    "media_type": "image",
                    "path": "pixel.png",
                    "sha256": _sha_bytes(image.read_bytes()),
                }
            ],
        },
    ]
    records = []
    for index, payload in enumerate(inputs):
        path = source / f"input-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        records.append(
            {
                "record_id": f"preserve-{index}",
                "input_path": path.name,
                "input_sha256": _sha_bytes(path.read_bytes()),
                "required_action_token_id": None,
            }
        )
    plan = {
        "format": "truth_editing_preservation_baseline_capture_plan_v2",
        "base_model_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "processor_sha256": "3" * 64,
        "chat_template_sha256": "4" * 64,
        "inference_runtime_sha256": "5" * 64,
        "batch_size": 2,
        "top_k": 64,
        "temperature": 1.0,
        "records": records,
    }
    path = source / "capture-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


class FakeBackend:
    identity = {
        "base_model_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "processor_sha256": "3" * 64,
        "chat_template_sha256": "4" * 64,
        "inference_runtime_sha256": "5" * 64,
    }

    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    def capture_batch(self, records):
        self.batches.append(tuple(record.record_id for record in records))
        outputs = []
        for index, record in enumerate(records):
            indices = torch.arange(64, dtype=torch.int64).reshape(1, 1, 64)
            probabilities = torch.full((1, 1, 65), 1.0 / 65, dtype=torch.float32)
            positions = torch.tensor([[2]], dtype=torch.int64)
            outputs.append(
                BaselineCaptureOutput(
                    record.record_id,
                    indices + index,
                    probabilities,
                    positions,
                    4,
                )
            )
        return tuple(outputs)


def test_capture_publishes_materializer_compatible_artifacts_atomically(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    output = tmp_path / "captured"
    backend = FakeBackend()

    receipt = capture_preservation_baselines(plan, output, backend=backend)

    assert backend.batches == [("preserve-0", "preserve-1")]
    assert open_preservation_baseline_capture(output) == receipt
    assert receipt["record_count"] == 2
    tensors = load_file(output / "base-logits/0000.safetensors")
    assert set(tensors) == {
        "assistant_positions",
        "base_indices",
        "base_probabilities",
        "sequence_length",
    }
    assert tensors["assistant_positions"].tolist() == [[2]]
    assert tensors["base_indices"].shape == (1, 1, 64)
    assert tensors["base_probabilities"].shape == (1, 1, 65)
    capture = json.loads((output / "capture-receipts/0000.json").read_text())
    assert capture["format"] == "truth_editing_preservation_base_logits_capture_receipt_v2"
    assert (
        capture["representation"]
        == "assistant_top64_plus_other_token_id_tiebreak_v2"
    )
    assert capture["assistant_position_count"] == 1
    assert capture["sequence_length"] == 4
    assert capture["record_id"] == "preserve-0"
    assert capture["base_logits_sha256"] == _sha_bytes(
        (output / "base-logits/0000.safetensors").read_bytes()
    )
    unsigned = {key: value for key, value in capture.items() if key != "self_sha256"}
    assert capture["self_sha256"] == _canonical_sha(unsigned)

    with pytest.raises(PreservationBaselineCaptureError, match="already exists"):
        capture_preservation_baselines(plan, output, backend=backend)


def test_capture_rejects_identity_or_output_substitution_without_partial_output(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    backend = FakeBackend()
    backend.identity = {**backend.identity, "processor_sha256": "9" * 64}
    output = tmp_path / "captured"
    with pytest.raises(PreservationBaselineCaptureError, match="backend identity"):
        capture_preservation_baselines(plan, output, backend=backend)
    assert not output.exists()

    backend = FakeBackend()
    backend.capture_batch = lambda records: (
        BaselineCaptureOutput(
            "substituted",
            torch.arange(64).reshape(1, 1, 64),
            torch.full((1, 1, 65), 1.0 / 65),
            torch.tensor([[0]]),
            2,
        ),
    )
    plan_raw = json.loads(plan.read_text())
    plan_raw["batch_size"] = 1
    plan.write_text(json.dumps(plan_raw))
    with pytest.raises(PreservationBaselineCaptureError, match="output record order"):
        capture_preservation_baselines(plan, output, backend=backend)
    assert not output.exists()


def test_capture_opener_rejects_self_consistent_unknown_compact_representation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "captured"
    capture_preservation_baselines(_write_plan(tmp_path), output, backend=FakeBackend())
    receipt_path = output / "capture-receipts/0000.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["representation"] = "full_vocab_logits_disguised_as_compact"
    receipt_unsigned = {key: value for key, value in receipt.items() if key != "self_sha256"}
    receipt["self_sha256"] = _canonical_sha(receipt_unsigned)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    run_path = output / "capture-run-receipt.json"
    run = json.loads(run_path.read_text())
    receipt_sha = _sha_bytes(receipt_path.read_bytes())
    run["records"][0]["base_logits_capture_receipt_sha256"] = receipt_sha
    run["artifact_sha256"]["capture-receipts/0000.json"] = receipt_sha
    run_unsigned = {key: value for key, value in run.items() if key != "self_sha256"}
    run["self_sha256"] = _canonical_sha(run_unsigned)
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")

    with pytest.raises(PreservationBaselineCaptureError, match="representation"):
        open_preservation_baseline_capture(output)

class FakeProcessor:
    chat_template = "frozen-template"

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(
            chat_template=self.chat_template,
            padding_side="left",
            pad_token_id=0,
            eos_token_id=2,
        )
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        text = "<image>" if any(isinstance(m["content"], list) for m in messages) else ""
        text += "<|im_start|>user\nquestion<|im_end|>\n"
        if add_generation_prompt:
            return text + "<|im_start|>assistant\n"
        return text + "<|im_start|>assistant\nanswer<|im_end|>\n"

    def __call__(self, *, text, padding, return_tensors, **kwargs):
        assert padding is True and return_tensors == "pt"
        self.calls.append({"text": list(text), **kwargs})
        rows = []
        for value in text:
            prefix = value.endswith("assistant\n")
            ids = ([21, 22] if value.startswith("<image>") else [11]) + [31, 32]
            if not prefix:
                ids += [41, 42]
            rows.append(ids)
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(masks),
            **({"pixel_values": torch.ones((1, 3, 2, 2))} if "images" in kwargs else {}),
        }


class FakeQwenModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2", use_cache=True)
        self.calls: list[dict[str, object]] = []

    def eval(self):
        return self

    def parameters(self):
        yield torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        logits = torch.arange(ids.shape[0] * ids.shape[1] * 80, dtype=torch.float32)
        return SimpleNamespace(logits=logits.reshape(ids.shape[0], ids.shape[1], 80))


def test_verified_qwen_backend_batches_multimodal_teacher_forcing_and_masks_prefix(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    model, processor = FakeQwenModel(), FakeProcessor()
    config = ModelLoadConfig(
        cache_dir=str(tmp_path / "cache"),
        snapshot_manifest_path=str(tmp_path / "manifest.json"),
        expected_model_sha256="1" * 64,
        expected_snapshot_manifest_sha256="a" * 64,
    )
    (tmp_path / "manifest.json").write_text("{}")

    def loader(requested: ModelLoadConfig) -> ModelBundle:
        return ModelBundle(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            model_id=requested.model_name,
            config=requested,
            verified_snapshot={
                "model_id": requested.model_name,
                "revision": requested.revision,
                "model_sha256": requested.expected_model_sha256,
                "snapshot_manifest_sha256": requested.expected_snapshot_manifest_sha256,
            },
        )

    backend = VerifiedQwenBaselineCaptureBackend(
        model_config=config,
        bundle_loader=loader,
        expected_tokenizer_sha256="2" * 64,
        expected_processor_sha256="3" * 64,
        expected_chat_template_sha256=_sha_bytes(b"frozen-template"),
        inference_runtime_sha256="5" * 64,
        enforce_production_runtime=False,
        vision_info_loader=lambda conversations: (["decoded-image"], []),
        processor_identity_resolver=lambda bundle: "3" * 64,
    )
    raw = json.loads(plan.read_text())
    raw["chat_template_sha256"] = _sha_bytes(b"frozen-template")
    plan.write_text(json.dumps(raw))

    receipt = capture_preservation_baselines(plan, tmp_path / "captured", backend=backend)

    assert receipt["record_count"] == 2
    assert len(model.calls) == 1
    assert "pixel_values" in model.calls[0]
    first = load_file(tmp_path / "captured/base-logits/0000.safetensors")
    second = load_file(tmp_path / "captured/base-logits/0001.safetensors")
    assert first["assistant_positions"].tolist() == [[2, 3]]
    assert second["assistant_positions"].tolist() == [[3, 4]]
    assert first["base_indices"].shape == (1, 2, 64)
    assert second["base_indices"].shape == (1, 2, 64)
    assert first["base_probabilities"].shape == (1, 2, 65)
    assert second["base_probabilities"].shape == (1, 2, 65)
    assert first["sequence_length"].item() == 5
    assert second["sequence_length"].item() == 6
    # Full-vocabulary logits are never persisted.
    assert "base_logits" not in first and "labels" not in first
    assert first["base_indices"][0, 0].tolist() == list(range(79, 15, -1))
    expected = torch.softmax(torch.arange(240, 320, dtype=torch.float32), dim=-1)
    assert torch.allclose(
        first["base_probabilities"][0, 0, :64],
        expected[torch.arange(79, 15, -1)],
        atol=2e-6,
        rtol=2e-6,
    )
    assert first["base_probabilities"][0, 0, 64].item() > 0
    assert first["base_probabilities"][0, 0].sum().item() == pytest.approx(1.0)


def test_verified_qwen_backend_breaks_top_k_logit_ties_by_lowest_token_id(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    processor = FakeProcessor()
    config = ModelLoadConfig(
        cache_dir=str(tmp_path / "cache"),
        snapshot_manifest_path=str(tmp_path / "manifest.json"),
        expected_model_sha256="1" * 64,
        expected_snapshot_manifest_sha256="a" * 64,
    )
    (tmp_path / "manifest.json").write_text("{}")

    class TiedQwenModel(FakeQwenModel):
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            ids = kwargs["input_ids"]
            return SimpleNamespace(
                logits=torch.zeros((*ids.shape, 80), dtype=torch.float32)
            )

    model = TiedQwenModel()

    def loader(requested: ModelLoadConfig) -> ModelBundle:
        return ModelBundle(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            model_id=requested.model_name,
            config=requested,
            verified_snapshot={
                "model_id": requested.model_name,
                "revision": requested.revision,
                "model_sha256": requested.expected_model_sha256,
                "snapshot_manifest_sha256": requested.expected_snapshot_manifest_sha256,
            },
        )

    backend = VerifiedQwenBaselineCaptureBackend(
        model_config=config,
        bundle_loader=loader,
        expected_tokenizer_sha256="2" * 64,
        expected_processor_sha256="3" * 64,
        expected_chat_template_sha256=_sha_bytes(b"frozen-template"),
        inference_runtime_sha256="5" * 64,
        enforce_production_runtime=False,
        vision_info_loader=lambda conversations: (["decoded-image"], []),
        processor_identity_resolver=lambda bundle: "3" * 64,
    )
    raw = json.loads(plan.read_text())
    raw["chat_template_sha256"] = _sha_bytes(b"frozen-template")
    raw["records"][1]["required_action_token_id"] = 79
    plan.write_text(json.dumps(raw))

    capture_preservation_baselines(plan, tmp_path / "captured", backend=backend)

    first = load_file(tmp_path / "captured/base-logits/0000.safetensors")
    second = load_file(tmp_path / "captured/base-logits/0001.safetensors")
    expected_indices = list(range(64))
    assert first["base_indices"][0, 0].tolist() == expected_indices
    assert first["base_indices"][0, 1].tolist() == expected_indices
    assert second["base_indices"][0, 0].tolist() == [*range(63), 79]
    assert first["base_probabilities"][0, 0, :64].tolist() == pytest.approx(
        [1.0 / 80.0] * 64
    )
    assert first["base_probabilities"][0, 0, 64].item() == pytest.approx(16.0 / 80.0)
    assert first["base_probabilities"][0, 0].sum().item() == pytest.approx(1.0)

    mask = torch.ones((1, 2), dtype=torch.bool)
    selected_token_boosted = torch.zeros((1, 2, 80), dtype=torch.float32)
    selected_token_boosted[..., 63] = 1.0
    selected_token_boosted[..., 64] = -1.0
    omitted_token_boosted = -selected_token_boosted
    selected_loss = topk_preservation_kl_loss(
        selected_token_boosted,
        first["base_indices"],
        first["base_probabilities"],
        mask,
    )
    omitted_loss = topk_preservation_kl_loss(
        omitted_token_boosted,
        first["base_indices"],
        first["base_probabilities"],
        mask,
    )
    assert selected_loss.item() == pytest.approx(0.0090473, abs=1e-6)
    assert omitted_loss.item() == pytest.approx(0.0055839, abs=1e-6)
    assert selected_loss > omitted_loss


def test_capture_cli_wires_verified_cache_configuration(tmp_path: Path, capsys) -> None:
    cli = _load_cli()
    plan = _write_plan(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text("{}")
    observed = {}

    def fake_capture(plan_path, output_dir, *, model_config):
        observed.update(
            plan=plan_path,
            output=output_dir,
            cache=model_config.cache_dir,
            manifest=model_config.snapshot_manifest_path,
        )
        return {"format": "truth_editing_preservation_baseline_capture_run_v1"}

    cli.capture_preservation_baselines = fake_capture
    output = tmp_path / "captured"
    assert (
        cli.main(
            [
                "--plan",
                str(plan),
                "--output-dir",
                str(output),
                "--model-cache",
                str(cache),
                "--model-cache-manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert observed == {
        "plan": plan,
        "output": output,
        "cache": str(cache),
        "manifest": str(manifest),
    }
    assert json.loads(capsys.readouterr().out)["format"].endswith("capture_run_v1")
