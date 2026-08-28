from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_base_known import (
    BaseKnownError,
    BaseKnownQualification,
    BaseKnownRunner,
    QualificationConfig,
    QualificationResponse,
)


MODEL = {
    "repository": "Qwen/Qwen3-VL-8B-Thinking",
    "revision": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
    "model_sha256": "b" * 64,
    "tokenizer_sha256": "8" * 64,
    "chat_template_sha256": "f" * 64,
    "inference_backend": "transformers",
    "dtype": "torch.bfloat16",
    "attention_implementation": "flash_attention_2",
    "device_map": "cuda:0",
    "local_files_only": True,
    "use_cache": True,
}


def _dataset(root: Path, *, split: str = "validation") -> Path:
    root.mkdir()
    row = {
        "format": "truth_editing_canonical_qa_record_v2",
        "record_id": "qa_one",
        "canonical_key": "one",
        "collision_cluster_id": "cluster_one",
        "question": "Capital of France?",
        "choices": ["Paris", "Rome", "Lima"],
        "correct_answer": "Paris",
        "wrong_answers": ["Rome", "Lima"],
        "family": "geography",
        "truth_authority": "fixture",
        "split": split,
    }
    raw = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    (root / f"{split}.jsonl").write_text(raw)
    import hashlib

    sha = hashlib.sha256(raw.encode()).hexdigest()
    manifest = {
        "format": "truth_editing_canonical_qa_manifest_v2",
        "dataset_id": "fixture-v2",
        "file_sha256": {f"{split}.jsonl": sha},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class CorrectBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate_batch(self, requests, identity):
        self.calls += 1
        assert identity.to_payload() == MODEL
        answers = []
        for request in requests:
            correct_label = request.labels[request.ordered_choices.index("Paris")]
            answers.append(QualificationResponse(request.request_id, correct_label, (1,)))
        return tuple(answers)


def _config(**changes) -> QualificationConfig:
    values = dict(
        split="validation", rotations="all", repeats=2, batch_size=2,
        max_new_tokens=8, seed=20260827, strict_all_correct=True,
    )
    values.update(changes)
    return QualificationConfig(**values)


def test_all_rotations_and_repeats_qualify_and_resume_without_backend(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "data")
    backend = CorrectBackend()
    runner = BaseKnownRunner(dataset, tmp_path / "out", MODEL, _config(), backend)
    result = runner.run()
    assert result.qualified_record_ids == ("qa_one",)
    assert result.records[0].request_count == 6
    assert result.records[0].correct_count == 6
    assert result.records[0].stable is True
    with pytest.raises(BaseKnownError, match="not production"):
        BaseKnownQualification.open(tmp_path / "out")
    opened = BaseKnownQualification.open(tmp_path / "out", allow_nonproduction=True)
    assert opened.qualified_record_ids == ("qa_one",)
    assert opened.dataset_manifest_sha256
    calls = backend.calls
    resumed = runner.run()
    assert resumed.manifest_sha256 == result.manifest_sha256
    assert backend.calls == calls


def test_one_wrong_rotation_fails_strict_qualification(tmp_path: Path) -> None:
    class OneWrong(CorrectBackend):
        def generate_batch(self, requests, identity):
            responses = list(super().generate_batch(requests, identity))
            if requests[0].rotation_index == 1:
                responses[0] = QualificationResponse(requests[0].request_id, "Z", ())
            return tuple(responses)

    result = BaseKnownRunner(
        _dataset(tmp_path / "data"), tmp_path / "out", MODEL,
        _config(batch_size=1), OneWrong(),
    ).run()
    assert result.qualified_record_ids == ()
    assert result.records[0].base_known is False
    assert result.records[0].parse_failure_count == 2


def test_test_split_is_sealed_even_when_config_constructed(tmp_path: Path) -> None:
    with pytest.raises(BaseKnownError, match="sealed"):
        BaseKnownRunner(
            _dataset(tmp_path / "data", split="test"), tmp_path / "out", MODEL,
            _config(split="test"), CorrectBackend(),
        ).run()


def test_identity_or_dataset_tamper_fails_closed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "data")
    runner = BaseKnownRunner(dataset, tmp_path / "out", MODEL, _config(), CorrectBackend())
    runner.run()
    changed = dict(MODEL, tokenizer_sha256="9" * 64)
    with pytest.raises(BaseKnownError, match="run identity"):
        BaseKnownRunner(dataset, tmp_path / "out", changed, _config(), CorrectBackend()).run()
    with (dataset / "validation.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(BaseKnownError, match="dataset split hash"):
        BaseKnownRunner(dataset, tmp_path / "other", MODEL, _config(), CorrectBackend()).run()


def test_missing_duplicate_or_unknown_backend_responses_fail_closed(tmp_path: Path) -> None:
    class Broken:
        def generate_batch(self, requests, identity):
            return (QualificationResponse("unknown", "A", ()),) * len(requests)

    with pytest.raises(BaseKnownError, match="response"):
        BaseKnownRunner(
            _dataset(tmp_path / "data"), tmp_path / "out", MODEL,
            _config(), Broken(),
        ).run()


def test_existing_raw_batch_tamper_is_detected(tmp_path: Path) -> None:
    runner = BaseKnownRunner(
        _dataset(tmp_path / "data"), tmp_path / "out", MODEL,
        _config(), CorrectBackend(),
    )
    runner.run()
    response = next((tmp_path / "out" / "batches").glob("*/responses.json"))
    response.write_text("[]")
    with pytest.raises(BaseKnownError, match="immutable batch evidence"):
        runner.run()


def test_strict_loader_rejects_aggregate_tamper(tmp_path: Path) -> None:
    runner = BaseKnownRunner(
        _dataset(tmp_path / "data"), tmp_path / "out", MODEL,
        _config(), CorrectBackend(),
    )
    runner.run()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    manifest["qualified_count"] = 0
    (tmp_path / "out" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BaseKnownError, match="self identity"):
        BaseKnownQualification.open(tmp_path / "out", allow_nonproduction=True)


def test_batch_store_never_deletes_another_writers_unique_staging(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "data")
    foreign = tmp_path / "out" / "batches" / ".batch_000000.foreign"
    foreign.mkdir(parents=True)
    sentinel = foreign / "sentinel"
    sentinel.write_text("other writer")
    BaseKnownRunner(dataset, tmp_path / "out", MODEL, _config(), CorrectBackend()).run()
    assert sentinel.read_text() == "other writer"
