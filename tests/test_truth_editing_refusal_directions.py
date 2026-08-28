from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_refusal_directions import (
    RefusalDirectionError,
    build_refusal_extraction_plan,
    canonical_sha256,
    parse_raw_joint_basis_manifest,
    parse_refusal_direction_bank,
    parse_refusal_direction_config,
    parse_refusal_prompt_manifest,
    parse_truth_orthogonalized_manifest,
)


def _signed(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["self_sha256"] = canonical_sha256(result)
    return result


def _config() -> dict:
    return _signed(
        {
            "format": "truth_editing_refusal_direction_config_v1",
            "config_id": "test",
            "model": {
                "repository": "Qwen/Qwen3-VL-8B-Thinking",
                "revision": "9" * 40,
                "model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "chat_template_sha256": "c" * 64,
                "decoder_layer_count": 2,
                "hidden_width": 4,
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
                    "repository": "mlabonne/harmless_alpaca",
                    "revision": "1" * 40,
                    "split": "train",
                    "text_field": "text",
                    "construction_range": {"start": 0, "stop": 2},
                    "evaluation_range": {"start": 2, "stop": 3},
                },
                {
                    "role": "harmful",
                    "repository": "mlabonne/harmful_behaviors",
                    "revision": "2" * 40,
                    "split": "train",
                    "text_field": "text",
                    "construction_range": {"start": 0, "stop": 2},
                    "evaluation_range": {"start": 2, "stop": 3},
                },
            ],
            "output_root": "artifacts/directions/refusal-v1",
        }
    )


def _prompts(config_sha: str) -> dict:
    rows = []
    for role, revision in (("harmless", "1" * 40), ("harmful", "2" * 40)):
        for index, partition in ((0, "construction"), (1, "construction"), (2, "evaluation")):
            rows.append(
                {
                    "prompt_id": f"{role}-{index}",
                    "role": role,
                    "partition": partition,
                    "source_repository": f"mlabonne/{role}_alpaca" if role == "harmless" else "mlabonne/harmful_behaviors",
                    "source_revision": revision,
                    "source_split": "train",
                    "source_index": index,
                    "prompt_text": f"{role} prompt {index}",
                    "formatted_prompt_sha256": canonical_sha256(
                        {
                            "chat_template_sha256": "c" * 64,
                            "transformers_version": "4.57.1",
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": f"{role} prompt {index}"},
                            ],
                            "add_generation_prompt": True,
                            "tokenize": False,
                            "response_prefix": "",
                        }
                    ),
                }
            )
    return _signed(
        {
            "format": "truth_editing_refusal_prompt_manifest_v1",
            "config_sha256": config_sha,
            "rows": rows,
        }
    )


def test_prompt_manifest_round_trip_and_source_disjoint_plan() -> None:
    config = parse_refusal_direction_config(_config())
    prompts = parse_refusal_prompt_manifest(_prompts(config.self_sha256), config)
    plan = build_refusal_extraction_plan(config, prompts)

    assert plan.ready
    assert plan.requires_model_inference
    assert plan.prompt_count == 6
    assert plan.construction_prompt_count == 4
    assert plan.evaluation_prompt_count == 2
    assert plan.model_forward_prompt_count == 4
    assert plan.residual_vector_count == 8
    assert plan.estimated_residual_bytes == 4 * 2 * 4 * 8
    assert parse_refusal_prompt_manifest(prompts.to_dict(), config) == prompts


def test_missing_prompt_snapshot_fails_closed_but_reports_exact_workload() -> None:
    config = parse_refusal_direction_config(_config())
    plan = build_refusal_extraction_plan(config, None)
    assert not plan.ready
    assert plan.blockers == ("materialized_prompt_manifest_missing",)
    assert plan.prompt_count == 6
    assert plan.residual_vector_count == 8


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda x: x["rows"][0].update(partition="evaluation"), "selection"),
        (lambda x: x["rows"][2].update(prompt_text=x["rows"][0]["prompt_text"]), "prompt text overlap"),
        (lambda x: x["rows"][0].update(formatted_prompt_sha256="f" * 64), "formatted prompt identity"),
    ],
)
def test_prompt_manifest_rejects_leakage_and_identity_drift(mutation, message) -> None:
    config = parse_refusal_direction_config(_config())
    raw = _prompts(config.self_sha256)
    mutation(raw)
    raw.pop("self_sha256")
    with pytest.raises(RefusalDirectionError, match=message):
        parse_refusal_prompt_manifest(_signed(raw), config)


def _bank(config_sha: str, prompt_sha: str) -> dict:
    receipts = []
    for layer in (0, 1):
        receipts.append(
            _signed(
                {
                    "format": "truth_editing_refusal_direction_layer_receipt_v1",
                    "receipt_id": f"layer-{layer}",
                    "source_layer": layer,
                    "width": 4,
                    "construction_harmless_count": 2,
                    "construction_harmful_count": 2,
                    "harmless_mean_sha256": "3" * 64,
                    "harmful_mean_sha256": "4" * 64,
                    "vector_path": f"layer-{layer}.npy",
                    "vector_file_sha256": str(layer + 5) * 64,
                    "vector_sha256": str(layer + 7) * 64,
                    "finite": True,
                    "unit_norm": True,
                }
            )
        )
    return _signed(
        {
            "format": "truth_editing_refusal_direction_bank_v1",
            "bank_id": "refusal-test",
            "config_sha256": config_sha,
            "prompt_manifest_sha256": prompt_sha,
            "model_sha256": "a" * 64,
            "chat_template_sha256": "c" * 64,
            "per_layer_receipts": receipts,
            "global_source_receipt_ids": ["layer-0", "layer-1"],
        }
    )


def test_bank_binds_every_layer_and_global_choices() -> None:
    config = parse_refusal_direction_config(_config())
    prompts = parse_refusal_prompt_manifest(_prompts(config.self_sha256), config)
    bank = parse_refusal_direction_bank(_bank(config.self_sha256, prompts.self_sha256), config, prompts)
    assert len(bank.per_layer_receipts) == 2
    assert bank.global_source_receipt_ids == ("layer-0", "layer-1")
    assert parse_refusal_direction_bank(bank.to_dict(), config, prompts) == bank


def test_bank_rejects_checkpoint_or_layer_substitution() -> None:
    config = parse_refusal_direction_config(_config())
    prompts = parse_refusal_prompt_manifest(_prompts(config.self_sha256), config)
    raw = _bank(config.self_sha256, prompts.self_sha256)
    raw["model_sha256"] = "e" * 64
    raw.pop("self_sha256")
    with pytest.raises(RefusalDirectionError, match="model identity"):
        parse_refusal_direction_bank(_signed(raw), config, prompts)

    raw = _bank(config.self_sha256, prompts.self_sha256)
    raw["per_layer_receipts"].pop()
    raw.pop("self_sha256")
    with pytest.raises(RefusalDirectionError, match="every configured layer"):
        parse_refusal_direction_bank(_signed(raw), config, prompts)


def test_joint_basis_manifests_keep_raw_and_orthogonalized_semantics_distinct() -> None:
    orth = _signed(
        {
            "format": "truth_editing_truth_orthogonalized_refusal_basis_v1",
            "truth_basis_set_sha256": "1" * 64,
            "raw_refusal_bank_sha256": "2" * 64,
            "layers": [
                {
                    "source_layer": 0,
                    "truth_basis_sha256": "3" * 64,
                    "raw_refusal_vector_sha256": "4" * 64,
                    "orthogonal_vector_path": "orth-0.npy",
                    "orthogonal_vector_sha256": "5" * 64,
                    "projection_norm": 0.2,
                    "residual_norm": 0.8,
                    "principal_angle_degrees": 78.0,
                    "qualified": True,
                }
            ],
        }
    )
    raw = _signed(
        {
            "format": "truth_editing_raw_joint_basis_manifest_v1",
            "diagnostic_only": True,
            "truth_basis_set_sha256": "1" * 64,
            "raw_refusal_bank_sha256": "2" * 64,
            "layers": [
                {
                    "source_layer": 0,
                    "truth_basis_sha256": "3" * 64,
                    "raw_refusal_vector_sha256": "4" * 64,
                    "joint_basis_path": "raw-joint-0.npy",
                    "joint_basis_sha256": "6" * 64,
                    "rank": 2,
                }
            ],
        }
    )
    assert parse_truth_orthogonalized_manifest(orth).layers[0].qualified
    assert parse_raw_joint_basis_manifest(raw).diagnostic_only

    mislabeled = copy.deepcopy(raw)
    mislabeled["diagnostic_only"] = False
    mislabeled.pop("self_sha256")
    with pytest.raises(RefusalDirectionError, match="diagnostic_only"):
        parse_raw_joint_basis_manifest(_signed(mislabeled))


def test_config_file_is_strict_and_current(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()))
    parsed = parse_refusal_direction_config(json.loads(path.read_text()))
    assert parsed.model.chat_template_sha256 == "c" * 64

    raw = _config()
    raw["unknown"] = True
    raw.pop("self_sha256")
    with pytest.raises(RefusalDirectionError, match="fields differ"):
        parse_refusal_direction_config(_signed(raw))


def test_cli_returns_nonzero_for_unmaterialized_prompt_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/plan_truth_editing_refusal_directions.py",
            "--config",
            str(config_path),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["blockers"] == [
        "materialized_prompt_manifest_missing"
    ]
