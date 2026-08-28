from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from intelligent_liars.models import (
    DEFAULT_MODEL_CONTENT_SHA256,
    DEFAULT_MODEL_REVISION,
)

from intelligent_liars.truth_editing_refusal_directions import (
    RefusalDirectionConfig,
    RefusalPromptManifest,
    build_refusal_extraction_plan,
    canonical_sha256,
    parse_refusal_direction_bank,
    parse_refusal_direction_config,
    parse_refusal_prompt_manifest,
)
from intelligent_liars.truth_editing_refusal_extraction import (
    BackendBatchResult,
    RefusalExtractionError,
    RefusalExtractionRunner,
    RuntimeIdentity,
    StoredResidualBackend,
    TransformersQwenResidualBackend,
    parse_run_receipt,
)


def _signed(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["self_sha256"] = canonical_sha256(result)
    return result


def _inputs() -> tuple[RefusalDirectionConfig, RefusalPromptManifest]:
    config_raw = _signed(
        {
            "format": "truth_editing_refusal_direction_config_v1",
            "config_id": "runner-test",
            "model": {
                "repository": "Qwen/Qwen3-VL-8B-Thinking",
                "revision": "9" * 40,
                "model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "chat_template_sha256": "c" * 64,
                "decoder_layer_count": 2,
                "hidden_width": 2,
            },
            "extraction": {
                "transformers_version": "4.57.1",
                "system_prompt": "You are a helpful assistant.",
                "message_layout": "system_then_user_text_v1",
                "add_generation_prompt": True,
                "tokenize_chat_template": False,
                "response_prefix": "",
                "max_new_tokens": 1,
                "do_sample": False,
                "use_cache": False,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
                "residual_location": "decoder_layer_output_first_generated_token_v1",
                "direction_formula": "unit_l2(mean_harmful_minus_mean_harmless)",
                "dtype": "float64",
                "layers": [0, 1],
            },
            "sources": [
                {
                    "role": "harmless",
                    "repository": "source/harmless",
                    "revision": "1" * 40,
                    "split": "train",
                    "text_field": "text",
                    "construction_range": {"start": 0, "stop": 2},
                    "evaluation_range": {"start": 2, "stop": 3},
                },
                {
                    "role": "harmful",
                    "repository": "source/harmful",
                    "revision": "2" * 40,
                    "split": "train",
                    "text_field": "text",
                    "construction_range": {"start": 0, "stop": 2},
                    "evaluation_range": {"start": 2, "stop": 3},
                },
            ],
            "output_root": "unused",
        }
    )
    config = parse_refusal_direction_config(config_raw)
    rows = []
    for role, revision in (("harmless", "1" * 40), ("harmful", "2" * 40)):
        for index, partition in (
            (0, "construction"),
            (1, "construction"),
            (2, "evaluation"),
        ):
            text = f"{role}-{index}"
            rows.append(
                {
                    "prompt_id": text,
                    "role": role,
                    "partition": partition,
                    "source_repository": f"source/{role}",
                    "source_revision": revision,
                    "source_split": "train",
                    "source_index": index,
                    "prompt_text": text,
                    "formatted_prompt_sha256": canonical_sha256(
                        {
                            "chat_template_sha256": "c" * 64,
                            "transformers_version": "4.57.1",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant.",
                                },
                                {"role": "user", "content": text},
                            ],
                            "add_generation_prompt": True,
                            "tokenize": False,
                            "response_prefix": "",
                        }
                    ),
                }
            )
    prompts = parse_refusal_prompt_manifest(
        _signed(
            {
                "format": "truth_editing_refusal_prompt_manifest_v1",
                "config_sha256": config.self_sha256,
                "rows": rows,
            }
        ),
        config,
    )
    return config, prompts


def _identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        backend="stored_mock_v1",
        repository="Qwen/Qwen3-VL-8B-Thinking",
        revision="9" * 40,
        model_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        chat_template_sha256="c" * 64,
        transformers_version="4.57.1",
        torch_version="2.5.1+cu124",
        dtype="torch.bfloat16",
        attention_implementation="flash_attention_2",
        device="cuda:0",
        decoder_layer_count=2,
        hidden_width=2,
    )


class _Backend:
    identity = _identity()

    def __init__(
        self, values: dict[str, tuple[float, float]], *, fail_after: int | None = None
    ):
        self.values = values
        self.fail_after = fail_after
        self.calls: list[tuple[str, ...]] = []

    def extract(self, rows):
        ids = tuple(row.prompt_id for row in rows)
        self.calls.append(ids)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("interrupted")
        base = np.asarray([self.values[item] for item in ids], dtype=np.float64)
        return BackendBatchResult(
            residuals_by_layer={0: base, 1: base * 2.0},
            input_token_count=len(ids) * 3,
            elapsed_seconds=0.5,
        )


def _values() -> dict[str, tuple[float, float]]:
    return {
        "harmless-0": (1.0, 0.0),
        "harmless-1": (1.0, 0.0),
        "harmful-0": (2.0, 2.0),
        "harmful-1": (2.0, 2.0),
    }


def _stored_payload(config, prompts) -> dict:
    plan = build_refusal_extraction_plan(config, prompts)
    return _signed(
        {
            "format": "truth_editing_stored_refusal_residuals_v1",
            "config_sha256": config.self_sha256,
            "prompt_manifest_sha256": prompts.self_sha256,
            "plan_sha256": plan.self_sha256,
            "runtime_identity": _identity().to_dict(),
            "records": [
                {
                    "prompt_id": prompt_id,
                    "formatted_prompt_sha256": next(
                        row.formatted_prompt_sha256
                        for row in prompts.rows
                        if row.prompt_id == prompt_id
                    ),
                    "input_token_count": 3,
                    "residuals_by_layer": {
                        "0": list(value),
                        "1": list(np.asarray(value) * 2),
                    },
                }
                for prompt_id, value in _values().items()
            ],
        }
    )


def _rebind_runtime_hashes(config, prompts, *, tokenizer_sha: str, template_sha: str):
    raw = config.to_dict()
    raw["model"]["revision"] = DEFAULT_MODEL_REVISION
    raw["model"]["model_sha256"] = DEFAULT_MODEL_CONTENT_SHA256
    raw["model"]["tokenizer_sha256"] = tokenizer_sha
    raw["model"]["chat_template_sha256"] = template_sha
    raw["sources"] = [
        {
            "role": source.role,
            "repository": source.repository,
            "revision": source.revision,
            "split": source.split,
            "text_field": source.text_field,
            "construction_range": {
                "start": source.construction_indices[0],
                "stop": source.construction_indices[-1] + 1,
            },
            "evaluation_range": {
                "start": source.evaluation_indices[0],
                "stop": source.evaluation_indices[-1] + 1,
            },
        }
        for source in config.sources
    ]
    raw.pop("self_sha256")
    rebound_config = parse_refusal_direction_config(_signed(raw))
    rows = []
    for row in prompts.rows:
        item = row.__dict__.copy()
        item["formatted_prompt_sha256"] = canonical_sha256(
            {
                "chat_template_sha256": template_sha,
                "transformers_version": "4.57.1",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": row.prompt_text},
                ],
                "add_generation_prompt": True,
                "tokenize": False,
                "response_prefix": "",
            }
        )
        rows.append(item)
    rebound_prompts = parse_refusal_prompt_manifest(
        _signed(
            {
                "format": "truth_editing_refusal_prompt_manifest_v1",
                "config_sha256": rebound_config.self_sha256,
                "rows": rows,
            }
        ),
        rebound_config,
    )
    return rebound_config, rebound_prompts


def test_runner_builds_parseable_unit_direction_bank_without_raw_residuals(
    tmp_path: Path,
) -> None:
    config, prompts = _inputs()
    result = RefusalExtractionRunner(
        config, prompts, _Backend(_values()), tmp_path, batch_size=1
    ).run()

    bank = parse_refusal_direction_bank(result.bank.to_dict(), config, prompts)
    assert len(bank.per_layer_receipts) == 2
    np.testing.assert_allclose(
        np.load(tmp_path / "vectors/layer-00.npy"), [1 / np.sqrt(5), 2 / np.sqrt(5)]
    )
    assert not list(tmp_path.rglob("*residual*"))
    receipt = parse_run_receipt(json.loads((tmp_path / "run_receipt.json").read_text()))
    assert receipt.completed_prompt_count == 4
    assert receipt.input_token_count == 12
    assert receipt.prompt_throughput == 2.0
    assert receipt.runtime_identity_sha256 == _identity().self_sha256


def test_runner_resumes_only_after_verified_atomic_batch_receipts(
    tmp_path: Path,
) -> None:
    config, prompts = _inputs()
    first = _Backend(_values(), fail_after=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        RefusalExtractionRunner(config, prompts, first, tmp_path, batch_size=1).run()
    assert first.calls == [("harmless-0",), ("harmless-1",)]

    resumed = _Backend(_values())
    resumed_result = RefusalExtractionRunner(
        config, prompts, resumed, tmp_path, batch_size=1
    ).run()
    assert resumed.calls == [("harmless-1",), ("harmful-0",), ("harmful-1",)]
    assert resumed_result.resumed_batch_count == 1

    contribution = tmp_path / "batches/0000/contribution.npz"
    contribution.write_bytes(contribution.read_bytes() + b"tamper")
    with pytest.raises(RefusalExtractionError, match="contribution file identity"):
        RefusalExtractionRunner(
            config, prompts, _Backend(_values()), tmp_path, batch_size=1
        ).run()


def test_runner_fails_closed_on_runtime_identity_or_backend_shape(
    tmp_path: Path,
) -> None:
    config, prompts = _inputs()
    backend = _Backend(_values())
    backend.identity = (
        copy.replace(_identity(), device="mps")
        if hasattr(copy, "replace")
        else RuntimeIdentity(**{**_identity().to_dict(), "device": "mps"})
    )
    with pytest.raises(RefusalExtractionError, match="runtime identity"):
        RefusalExtractionRunner(config, prompts, backend, tmp_path).run()
    assert backend.calls == []

    class MissingLayer(_Backend):
        def extract(self, rows):
            result = super().extract(rows)
            return BackendBatchResult({0: result.residuals_by_layer[0]}, 1, 1.0)

    with pytest.raises(RefusalExtractionError, match="every configured layer"):
        RefusalExtractionRunner(
            config, prompts, MissingLayer(_values()), tmp_path / "bad"
        ).run()


def test_stored_backend_round_trip_and_production_environment_rejection(
    tmp_path: Path,
) -> None:
    config, prompts = _inputs()
    stored_path = tmp_path / "stored.json"
    stored_path.write_text(json.dumps(_stored_payload(config, prompts)))
    result = RefusalExtractionRunner(
        config,
        prompts,
        StoredResidualBackend.from_path(stored_path),
        tmp_path / "out",
        batch_size=2,
    ).run()
    assert result.bank.model_sha256 == "a" * 64

    with pytest.raises(RefusalExtractionError, match="CUDA"):
        TransformersQwenResidualBackend.validate_environment(
            cuda_available=False,
            transformers_version="4.57.1",
            torch_version="2.5.1+cu124",
        )


def test_mocked_production_backend_verifies_identity_and_extracts_prefill_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, prompts = _inputs()
    template = "frozen-template"
    template_sha = hashlib.sha256(template.encode()).hexdigest()
    tokenizer_sha = "d" * 64
    config, prompts = _rebind_runtime_hashes(
        config, prompts, tokenizer_sha=tokenizer_sha, template_sha=template_sha
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "tokenizer.json", "sha256": tokenizer_sha}]})
    )

    class Processor:
        chat_template = template

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            return "rendered"

        def __call__(self, **kwargs):
            return {
                "input_ids": FakeInput(2),
                "attention_mask": FakeInput(2),
            }

    class FakeInput:
        def __init__(self, value):
            self.value = value

        def to(self, device):
            return self

        def sum(self):
            return self

        def item(self):
            return self.value

        def numel(self):
            return self.value

    class Parameter:
        device = SimpleNamespace(type="cuda", index=0)
        dtype = torch.bfloat16

    class Model:
        config = SimpleNamespace(_attn_implementation="flash_attention_2")

        def parameters(self):
            return iter((Parameter(), Parameter()))

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 1
            embedding = torch.zeros((1, 2, 2))
            layer0 = torch.tensor([[[0.0, 0.0], [2.0, 3.0]]])
            layer1 = layer0 * 2
            return SimpleNamespace(hidden_states=((embedding, layer0, layer1),))

    bundle = SimpleNamespace(
        model=Model(),
        processor=Processor(),
        tokenizer=SimpleNamespace(chat_template=template),
        verified_snapshot={
            "model_id": config.model.repository,
            "revision": config.model.revision,
            "model_sha256": config.model.model_sha256,
            "snapshot_manifest_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu124")
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.57.1")
    backend = TransformersQwenResidualBackend(
        config,
        cache_dir=tmp_path,
        snapshot_manifest_path=manifest,
        bundle=bundle,
    )
    backend.identity.verify_for(config)
    batch = backend.extract((prompts.rows[0],))
    np.testing.assert_array_equal(batch.residuals_by_layer[0], [[2.0, 3.0]])
    assert batch.input_token_count == 2


def test_cli_runs_stored_residuals_and_reports_receipt(tmp_path: Path) -> None:
    config, prompts = _inputs()
    config_path = tmp_path / "config.json"
    prompts_path = tmp_path / "prompts.json"
    stored_path = tmp_path / "stored.json"
    config_disk = config.to_dict()
    config_disk["sources"] = [
        {
            "role": source.role,
            "repository": source.repository,
            "revision": source.revision,
            "split": source.split,
            "text_field": source.text_field,
            "construction_range": {
                "start": source.construction_indices[0],
                "stop": source.construction_indices[-1] + 1,
            },
            "evaluation_range": {
                "start": source.evaluation_indices[0],
                "stop": source.evaluation_indices[-1] + 1,
            },
        }
        for source in config.sources
    ]
    config_disk.pop("self_sha256")
    config_path.write_text(json.dumps(_signed(config_disk)))
    prompts_path.write_text(json.dumps(prompts.to_dict()))
    stored_path.write_text(json.dumps(_stored_payload(config, prompts)))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/extract_truth_editing_refusal_directions.py",
            "--config",
            str(config_path),
            "--prompts",
            str(prompts_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--stored-residuals",
            str(stored_path),
            "--batch-size",
            "3",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["completed_prompt_count"] == 4
    assert summary["direction_count"] == 2
    assert summary["resumed_batch_count"] == 0
