from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from intelligent_liars.models import (
    DEFAULT_MODEL_CONTENT_SHA256,
    ModelBundle,
    ModelLoadConfig,
)
from intelligent_liars.truth_editing_base_known import (
    BaseKnownError,
    FrozenBaseIdentity,
    QualificationBatchStore,
    QualificationRequest,
    validate_backend_batch_execution,
)
from intelligent_liars.truth_editing_qwen_qualification import (
    FrozenQwenQualificationBackend,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _payload_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


IDENTITY = FrozenBaseIdentity.from_mapping(
    {
        "repository": "Qwen/Qwen3-VL-8B-Thinking",
        "revision": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
        "model_sha256": "b" * 64,
        "tokenizer_sha256": _sha("tokenizer"),
        "chat_template_sha256": _sha("template"),
        "inference_backend": "transformers",
        "dtype": "torch.bfloat16",
        "attention_implementation": "flash_attention_2",
        "device_map": "cuda:0",
        "local_files_only": True,
        "use_cache": True,
    }
)


def _request(number: int, max_new_tokens: int = 8) -> QualificationRequest:
    return QualificationRequest(
        request_id=f"bk_{number:064x}",
        record_id=f"record-{number}",
        rotation_index=number % 2,
        repeat_index=0,
        labels=("A", "B"),
        ordered_choices=("truth", "false"),
        correct_answer="truth",
        prompt=f"question {number}",
        seed=number,
        max_new_tokens=max_new_tokens,
    )


class FakeProcessor:
    chat_template = "template"

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=9,
            padding_side="left",
            chat_template="template",
            encode=lambda text, add_special_tokens: {
                "A": [3],
                "B": [4],
            }[text],
        )
        self.rendered: list[list[dict[str, str]]] = []
        self.encoded_texts: list[str] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        self.rendered.append(messages)
        return messages[0]["content"][0]["text"] + "<|im_start|>assistant\n<think>\n"

    def __call__(self, *, text, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        self.encoded_texts.extend(text)
        return {
            "input_ids": torch.tensor([[0, 1, 2] for _ in text]),
            "attention_mask": torch.tensor([[0, 1, 1] for _ in text]),
        }

    def batch_decode(self, rows, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return ["A" if row[0] == 3 else "B" for row in rows]


class FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        self.calls: list[dict[str, object]] = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([[3, 9] for _ in range(input_ids.shape[0])])
        return torch.cat((input_ids, suffix), dim=1)


def _manifest(path: Path, *, tokenizer_sha: str | None = None) -> None:
    files = [
        {"path": "chat_template.json", "bytes": 1, "sha256": IDENTITY.chat_template_sha256},
        {"path": "tokenizer.json", "bytes": 1, "sha256": tokenizer_sha or IDENTITY.tokenizer_sha256},
    ]
    path.write_text(json.dumps({"format": "fixture", "files": files}))


def _backend(tmp_path: Path, loader, clock=None) -> FrozenQwenQualificationBackend:
    manifest = tmp_path / "model-manifest.json"
    _manifest(manifest)
    config = ModelLoadConfig(
        cache_dir=str(tmp_path / "cache"),
        snapshot_manifest_path=str(manifest),
        expected_model_sha256=IDENTITY.model_sha256,
        expected_snapshot_manifest_sha256="c" * 64,
    )
    return FrozenQwenQualificationBackend(
        model_config=config,
        bundle_loader=loader,
        enforce_production_runtime=False,
        clock=clock or iter((1.0, 2.0)).__next__,
        execution_receipt_path=tmp_path / "qwen-execution-receipt.json",
    )


def test_backend_batches_greedy_decoding_and_binds_execution_telemetry(tmp_path: Path) -> None:
    model, processor = FakeModel(), FakeProcessor()
    loads = 0

    def loader(config: ModelLoadConfig) -> ModelBundle:
        nonlocal loads
        loads += 1
        return ModelBundle(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            model_id=config.model_name,
            config=config,
            verified_snapshot={
                "model_id": config.model_name,
                "revision": config.revision,
                "model_sha256": config.expected_model_sha256,
                "snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256,
            },
        )

    backend = _backend(tmp_path, loader)
    requests = (_request(1), _request(2))
    responses = backend.generate_batch(requests, IDENTITY)
    execution = backend.batch_execution_receipt(requests, responses, IDENTITY)

    assert backend.evidence_receipt(IDENTITY) is None
    assert [response.text for response in responses] == ["A", "A"]
    assert [response.token_ids for response in responses] == [(3, 9), (3, 9)]
    assert loads == 1
    assert processor.rendered == [
        [{"role": "user", "content": [{"type": "text", "text": "question 1"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "question 2"}]}],
    ]
    assert processor.encoded_texts == [
        "question 1<|im_start|>assistant\n<think>\n</think>\n\n",
        "question 2<|im_start|>assistant\n<think>\n</think>\n\n",
    ]
    call = model.calls[0]
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert call["temperature"] == 0.0
    assert call["use_cache"] is True
    assert call["max_new_tokens"] == 8
    allowed = call["prefix_allowed_tokens_fn"]
    assert allowed(0, torch.tensor([0, 1, 2])) == [3, 4]
    assert allowed(0, torch.tensor([0, 1, 2, 3])) == [9]
    assert execution is None


def test_backend_fails_closed_when_an_option_label_is_not_one_token(tmp_path: Path) -> None:
    model, processor = FakeModel(), FakeProcessor()
    processor.tokenizer.encode = lambda text, add_special_tokens: [3, 4]

    def loader(config: ModelLoadConfig) -> ModelBundle:
        return ModelBundle(
            model, processor, processor.tokenizer, config.model_name, config,
            {"model_id": config.model_name, "revision": config.revision,
             "model_sha256": config.expected_model_sha256,
             "snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256},
        )

    with pytest.raises(BaseKnownError, match="single tokenizer token"):
        _backend(tmp_path, loader).generate_batch((_request(1),), IDENTITY)


def test_backend_rejects_an_unexpected_thinking_template_suffix(tmp_path: Path) -> None:
    model, processor = FakeModel(), FakeProcessor()
    processor.apply_chat_template = lambda *args, **kwargs: "unexpected assistant prompt"

    def loader(config: ModelLoadConfig) -> ModelBundle:
        return ModelBundle(
            model, processor, processor.tokenizer, config.model_name, config,
            {"model_id": config.model_name, "revision": config.revision,
             "model_sha256": config.expected_model_sha256,
             "snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256},
        )

    with pytest.raises(BaseKnownError, match="thinking prompt suffix"):
        _backend(tmp_path, loader).generate_batch((_request(1),), IDENTITY)


def test_batch_store_does_not_publish_production_receipt_for_mock_qwen(tmp_path: Path) -> None:
    model, processor = FakeModel(), FakeProcessor()

    def loader(config: ModelLoadConfig) -> ModelBundle:
        return ModelBundle(
            model, processor, processor.tokenizer, config.model_name, config,
            {"model_id": config.model_name, "revision": config.revision,
             "model_sha256": config.expected_model_sha256,
             "snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256},
        )

    backend = _backend(tmp_path, loader)
    requests = (_request(1), _request(2))
    store = QualificationBatchStore(
        tmp_path / "out",
        IDENTITY,
        backend,
        {"mode": "synthetic_mock_only", "receipt_sha256": "0" * 64},
    )
    first = store.run(0, requests, "a" * 64)
    receipt = json.loads((tmp_path / "out/batches/batch_000000/receipt.json").read_text())
    assert "backend_execution" not in receipt
    assert store.run(0, requests, "a" * 64) == first
    assert len(model.calls) == 1


def test_one_backend_reuses_one_model_across_qualification_lanes(tmp_path: Path) -> None:
    model, processor, loads = FakeModel(), FakeProcessor(), []

    def loader(config: ModelLoadConfig) -> ModelBundle:
        loads.append(config)
        return ModelBundle(
            model, processor, processor.tokenizer, config.model_name, config,
            {"model_id": config.model_name, "revision": config.revision,
             "model_sha256": config.expected_model_sha256,
             "snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256},
        )

    backend = _backend(tmp_path, loader, iter((0.0, 1.0, 2.0, 3.0)).__next__)
    backend.generate_batch((_request(1),), IDENTITY)
    backend.generate_batch((_request(2),), IDENTITY)
    assert len(loads) == 1
    assert len(model.calls) == 2


def test_backend_fails_closed_on_manifest_or_identity_drift(tmp_path: Path) -> None:
    model, processor = FakeModel(), FakeProcessor()

    def loader(config: ModelLoadConfig) -> ModelBundle:
        return ModelBundle(
            model, processor, processor.tokenizer, config.model_name, config,
            {"model_id": config.model_name, "revision": config.revision,
             "model_sha256": "d" * 64,
             "snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256},
        )

    backend = _backend(tmp_path, loader)
    with pytest.raises(BaseKnownError, match="loaded snapshot"):
        backend.generate_batch((_request(1),), IDENTITY)

    manifest = tmp_path / "other-manifest.json"
    _manifest(manifest, tokenizer_sha="e" * 64)
    raw_manifest = json.loads(manifest.read_text())
    raw_manifest["files"] = [
        item for item in raw_manifest["files"] if item["path"] != "tokenizer.json"
    ]
    manifest.write_text(json.dumps(raw_manifest))
    config = ModelLoadConfig(
        cache_dir=str(tmp_path / "cache"), snapshot_manifest_path=str(manifest),
        expected_model_sha256=IDENTITY.model_sha256,
        expected_snapshot_manifest_sha256="c" * 64,
    )
    other = FrozenQwenQualificationBackend(
        model_config=config, bundle_loader=loader, enforce_production_runtime=False,
        execution_receipt_path=tmp_path / "other-execution-receipt.json",
    )
    with pytest.raises(BaseKnownError, match="tokenizer"):
        other.generate_batch((_request(1),), IDENTITY)


def test_backend_rejects_mixed_decoding_contracts(tmp_path: Path) -> None:
    model, processor = FakeModel(), FakeProcessor()

    def loader(config: ModelLoadConfig) -> ModelBundle:
        return ModelBundle(
            model, processor, processor.tokenizer, config.model_name, config,
            {"model_id": config.model_name, "revision": config.revision,
             "model_sha256": config.expected_model_sha256,
             "snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256},
        )

    backend = _backend(tmp_path, loader)
    with pytest.raises(BaseKnownError, match="max_new_tokens"):
        backend.generate_batch((_request(1, 8), _request(2, 9)), IDENTITY)


def test_injected_loader_cannot_mint_production_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "model-manifest.json"
    _manifest(manifest)
    config = ModelLoadConfig(
        cache_dir=str(tmp_path / "cache"), snapshot_manifest_path=str(manifest),
    )
    with pytest.raises(BaseKnownError, match="verified model loader"):
        FrozenQwenQualificationBackend(
            model_config=config,
            bundle_loader=lambda _: None,  # type: ignore[arg-type,return-value]
            execution_receipt_path=tmp_path / "receipt.json",
        )


def test_production_backend_rejects_nonproject_processor_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "model-manifest.json"
    _manifest(manifest)
    backend = FrozenQwenQualificationBackend(
        model_config=ModelLoadConfig(
            cache_dir=str(tmp_path / "cache"), snapshot_manifest_path=str(manifest)
        ),
        execution_receipt_path=tmp_path / "receipt.json",
    )
    identity_payload = IDENTITY.to_payload()
    identity_payload["model_sha256"] = DEFAULT_MODEL_CONTENT_SHA256
    with pytest.raises(BaseKnownError, match="project pin"):
        backend.evidence_receipt(FrozenBaseIdentity.from_mapping(identity_payload))


def test_cli_reports_malformed_config_without_traceback(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_truth_editing_qualifications.py",
            "--config",
            str(config),
            "--base-output-dir",
            str(tmp_path / "output"),
            "--responses",
            str(tmp_path / "responses.json"),
        ],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "truth-editing qualification failed" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generated_token_count", 3, "generated-token"),
        ("generated_tokens_per_second", 9.0, "throughput"),
    ],
)
def test_batch_execution_telemetry_is_semantically_bound(
    field: str, value: object, message: str
) -> None:
    requests = [_request(1).to_payload()]
    responses = [{"request_id": _request(1).request_id, "text": "A", "token_ids": [3, 9]}]
    evidence = {"mode": "verified_frozen_qwen", "receipt_sha256": "f" * 64}
    unsigned = {
        "format": "truth_editing_frozen_qwen_batch_execution_v1",
        "backend_receipt_sha256": "f" * 64,
        "request_set_sha256": _payload_sha(requests),
        "response_set_sha256": _payload_sha(responses),
        "request_count": 1,
        "prompt_token_count": 2,
        "generated_token_count": 2,
        "elapsed_seconds": 1.0,
        "generated_tokens_per_second": 2.0,
        "cuda_peak_allocated_bytes": 1,
    }
    unsigned[field] = value
    execution = {**unsigned, "self_sha256": _payload_sha(unsigned)}
    with pytest.raises(BaseKnownError, match=message):
        validate_backend_batch_execution(execution, evidence, requests, responses)
