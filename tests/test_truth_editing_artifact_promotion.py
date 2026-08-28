from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from intelligent_liars.truth_editing_artifact_promotion import (
    ArtifactPromotionError,
    promote_recovered_refusal_bank,
)
from intelligent_liars.truth_editing_refusal_directions import canonical_sha256


def _signed(value: dict[str, object]) -> dict[str, object]:
    return {**value, "self_sha256": canonical_sha256(value)}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Any]:
    config = _signed(
        {
            "format": "truth_editing_refusal_direction_config_v1",
            "config_id": "test-refusal",
            "model": {
                "repository": "Qwen/Qwen3-VL-8B-Thinking",
                "revision": "9" * 40,
                "model_sha256": "1" * 64,
                "tokenizer_sha256": "2" * 64,
                "chat_template_sha256": "3" * 64,
                "decoder_layer_count": 1,
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
                "layers": [0],
            },
            "sources": [
                {
                    "role": "harmless",
                    "repository": "fixture/harmless",
                    "revision": "a" * 40,
                    "split": "train",
                    "text_field": "text",
                    "construction_range": {"start": 0, "stop": 1},
                    "evaluation_range": {"start": 1, "stop": 2},
                },
                {
                    "role": "harmful",
                    "repository": "fixture/harmful",
                    "revision": "b" * 40,
                    "split": "train",
                    "text_field": "text",
                    "construction_range": {"start": 0, "stop": 1},
                    "evaluation_range": {"start": 1, "stop": 2},
                },
            ],
            "output_root": "artifacts/truth-editing/refusal-directions",
        }
    )
    rows = []
    for role, repository, revision in (
        ("harmless", "fixture/harmless", "a" * 40),
        ("harmful", "fixture/harmful", "b" * 40),
    ):
        for partition, source_index in (("construction", 0), ("evaluation", 1)):
            rows.append(
                {
                    "prompt_id": hashlib.sha256(
                        f"{role}:{partition}".encode()
                    ).hexdigest(),
                    "role": role,
                    "partition": partition,
                    "source_repository": repository,
                    "source_revision": revision,
                    "source_split": "train",
                    "source_index": source_index,
                    "prompt_text": f"{role} {partition}",
                    "formatted_prompt_sha256": canonical_sha256(
                        {
                            "chat_template_sha256": "3" * 64,
                            "transformers_version": "4.57.1",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant.",
                                },
                                {
                                    "role": "user",
                                    "content": f"{role} {partition}",
                                },
                            ],
                            "add_generation_prompt": True,
                            "tokenize": False,
                            "response_prefix": "",
                        }
                    ),
                }
            )
    prompts = _signed(
        {
            "format": "truth_editing_refusal_prompt_manifest_v1",
            "config_sha256": config["self_sha256"],
            "rows": rows,
        }
    )
    outputs = tmp_path / "outputs"
    refusal = outputs / "refusal"
    vector = refusal / "vectors/layer-00.npy"
    vector.parent.mkdir(parents=True)
    np.save(vector, np.array([0.6, 0.8], dtype=np.float64), allow_pickle=False)
    vector_file_sha = hashlib.sha256(vector.read_bytes()).hexdigest()
    vector_value_sha = hashlib.sha256(
        np.array([0.6, 0.8], dtype=np.float64).tobytes()
    ).hexdigest()
    layer = _signed(
        {
            "format": "truth_editing_refusal_direction_layer_receipt_v1",
            "receipt_id": "raw-refusal-layer-00",
            "source_layer": 0,
            "width": 2,
            "construction_harmless_count": 1,
            "construction_harmful_count": 1,
            "harmless_mean_sha256": "4" * 64,
            "harmful_mean_sha256": "5" * 64,
            "vector_path": "vectors/layer-00.npy",
            "vector_file_sha256": vector_file_sha,
            "vector_sha256": vector_value_sha,
            "finite": True,
            "unit_norm": True,
        }
    )
    bank = _signed(
        {
            "format": "truth_editing_refusal_direction_bank_v1",
            "bank_id": "fixture-refusal-bank",
            "config_sha256": config["self_sha256"],
            "prompt_manifest_sha256": prompts["self_sha256"],
            "model_sha256": "1" * 64,
            "chat_template_sha256": "3" * 64,
            "per_layer_receipts": [layer],
            "global_source_receipt_ids": ["raw-refusal-layer-00"],
        }
    )
    runtime = {
        "backend": "transformers_qwen_residual_v1",
        "repository": "Qwen/Qwen3-VL-8B-Thinking",
        "revision": "9" * 40,
        "model_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "chat_template_sha256": "3" * 64,
        "transformers_version": "4.57.1",
        "torch_version": "2.5.1+cu124",
        "dtype": "torch.bfloat16",
        "attention_implementation": "flash_attention_2",
        "device": "cuda:0",
        "decoder_layer_count": 1,
        "hidden_width": 2,
    }
    runtime_sha = canonical_sha256(runtime)
    run_receipt = _signed(
        {
            "format": "truth_editing_refusal_extraction_run_receipt_v1",
            "config_sha256": config["self_sha256"],
            "prompt_manifest_sha256": prompts["self_sha256"],
            "plan_sha256": "6" * 64,
            "runtime_identity": runtime,
            "runtime_identity_sha256": runtime_sha,
            "batch_size": 1,
            "batch_count": 2,
            "batch_receipt_sha256s": ["7" * 64, "8" * 64],
            "completed_prompt_count": 2,
            "input_token_count": 10,
            "elapsed_seconds": 1.0,
            "prompt_throughput": 2.0,
            "token_throughput": 10.0,
            "direction_bank_sha256": bank["self_sha256"],
        }
    )
    _write_json(refusal / "direction_bank.json", bank)
    _write_json(refusal / "receipts/layer-00.json", layer)
    _write_json(refusal / "run_receipt.json", run_receipt)
    (outputs / "other.txt").write_text("bound whole-tree input\n", encoding="utf-8")

    archive = tmp_path / "outputs.tar.gz"
    with archive.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(outputs.rglob("*")):
                    tar.add(path, arcname=path.relative_to(outputs), recursive=False)

    lifecycle = _signed(
        {
            "format": "truth_editing_vast_prerequisite_lifecycle_v1",
            "offer": {"id": 123},
            "image": "pinned-image@sha256:" + "a" * 64,
            "label": "fixture-lifecycle",
            "instance_id": "456",
            "events": [
                {"event": "created", "instance_id": "456"},
                {"event": "workload_finished", "exit_code": 0},
                {"event": "destroyed"},
            ],
            "elapsed_seconds": 2.0,
            "estimated_cost_usd": 0.01,
            "maximum_network_cost_usd": 0.01,
            "projected_all_in_max_cost_usd": 0.02,
            "exit_code": 0,
            "artifact_archive": {
                "format": "truth_editing_vast_output_archive_v1",
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "size_bytes": archive.stat().st_size,
                "expected_outputs": ["refusal/direction_bank.json"],
                "published_directory": "recovered/outputs",
            },
            "error": "KeyboardInterrupt: ",
            "destroyed": True,
            "destroy_error": None,
        }
    )
    paths = {
        "config": tmp_path / "config.json",
        "prompts": tmp_path / "prompts.json",
        "outputs": outputs,
        "archive": archive,
        "lifecycle": tmp_path / "lifecycle.json",
        "destination": tmp_path / "canonical/refusal-directions",
    }
    paths["expected_archive_sha"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    _write_json(paths["config"], config)
    _write_json(paths["prompts"], prompts)
    _write_json(paths["lifecycle"], lifecycle)
    return paths


def _promote(paths: dict[str, Any]) -> dict[str, object]:
    return promote_recovered_refusal_bank(
        lifecycle_receipt_path=paths["lifecycle"],
        output_archive_path=paths["archive"],
        expected_output_archive_sha256=paths["expected_archive_sha"],
        extracted_outputs_dir=paths["outputs"],
        refusal_config_path=paths["config"],
        refusal_prompt_manifest_path=paths["prompts"],
        destination=paths["destination"],
    )


def test_verified_recovery_is_atomically_promoted_with_bound_provenance(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    receipt = _promote(paths)

    destination = paths["destination"]
    assert receipt["format"] == "truth_editing_recovered_artifact_promotion_v1"
    assert receipt["artifact_kind"] == "refusal_direction_bank"
    assert receipt["lifecycle_receipt_file_sha256"] == hashlib.sha256(
        paths["lifecycle"].read_bytes()
    ).hexdigest()
    assert receipt["lifecycle_receipt_sha256"] == json.loads(
        paths["lifecycle"].read_text()
    )["self_sha256"]
    assert receipt["output_archive_sha256"] == hashlib.sha256(
        paths["archive"].read_bytes()
    ).hexdigest()
    assert receipt["refusal_bank_sha256"] == json.loads(
        (destination / "direction_bank.json").read_text()
    )["self_sha256"]
    assert json.loads((destination / "promotion_receipt.json").read_text()) == receipt
    assert np.load(destination / "vectors/layer-00.npy", allow_pickle=False).tolist() == [
        0.6,
        0.8,
    ]


@pytest.mark.parametrize("tamper", ["archive", "tree", "lifecycle", "vector"])
def test_tampered_recovery_fails_closed_without_destination(
    tmp_path: Path, tamper: str
) -> None:
    paths = _fixture(tmp_path)
    if tamper == "archive":
        paths["archive"].write_bytes(paths["archive"].read_bytes() + b"tamper")
    elif tamper == "tree":
        (paths["outputs"] / "other.txt").write_text("changed", encoding="utf-8")
    elif tamper == "lifecycle":
        raw = json.loads(paths["lifecycle"].read_text())
        raw["destroyed"] = False
        _write_json(paths["lifecycle"], raw)
    else:
        np.save(
            paths["outputs"] / "refusal/vectors/layer-00.npy",
            np.array([1.0, 0.0]),
            allow_pickle=False,
        )

    with pytest.raises(ArtifactPromotionError):
        _promote(paths)

    assert not paths["destination"].exists()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["destination"].mkdir(parents=True)
    sentinel = paths["destination"] / "user-work.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ArtifactPromotionError, match="destination already exists"):
        _promote(paths)

    assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_lifecycle_archive_identity_must_match_recovered_archive(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    lifecycle = json.loads(paths["lifecycle"].read_text())
    lifecycle["artifact_archive"] = {
        "format": "truth_editing_vast_output_archive_v1",
        "archive_sha256": "0" * 64,
        "size_bytes": paths["archive"].stat().st_size,
        "expected_outputs": ["refusal/direction_bank.json"],
        "published_directory": "recovered",
    }
    unsigned = dict(lifecycle)
    unsigned.pop("self_sha256")
    lifecycle["self_sha256"] = canonical_sha256(unsigned)
    _write_json(paths["lifecycle"], lifecycle)

    with pytest.raises(ArtifactPromotionError, match="archive identity mismatch"):
        _promote(paths)

    assert not paths["destination"].exists()


def test_source_change_during_staging_leaves_no_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    import intelligent_liars.truth_editing_artifact_promotion as promotion

    original = promotion._tree_inventory

    def inventory_after_change(root: Path) -> tuple[dict[str, Any], ...]:
        if root.name == "artifact":
            (root / "direction_bank.json").write_text("{}", encoding="utf-8")
        return original(root)

    monkeypatch.setattr(promotion, "_tree_inventory", inventory_after_change)

    with pytest.raises(ArtifactPromotionError, match="changed during staging"):
        _promote(paths)

    assert not paths["destination"].exists()
    assert not list(paths["destination"].parent.glob("*.promotion-claim"))


def test_cli_uses_the_same_fail_closed_promotion_seam(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/promote_truth_editing_recovered_artifact.py"),
            "--lifecycle-receipt",
            str(paths["lifecycle"]),
            "--output-archive",
            str(paths["archive"]),
            "--expected-output-archive-sha256",
            paths["expected_archive_sha"],
            "--extracted-outputs-dir",
            str(paths["outputs"]),
            "--refusal-config",
            str(paths["config"]),
            "--refusal-prompt-manifest",
            str(paths["prompts"]),
            "--destination",
            str(paths["destination"]),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["artifact_kind"] == "refusal_direction_bank"
    assert (paths["destination"] / "promotion_receipt.json").is_file()


def test_archive_with_one_enclosing_outputs_directory_is_compatible(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    archive = paths["archive"]
    with archive.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(paths["outputs"].rglob("*")):
                    relative = path.relative_to(paths["outputs"])
                    tar.add(path, arcname=Path("outputs") / relative, recursive=False)
    paths["expected_archive_sha"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    lifecycle = json.loads(paths["lifecycle"].read_text())
    lifecycle["artifact_archive"]["archive_sha256"] = paths["expected_archive_sha"]
    lifecycle["artifact_archive"]["size_bytes"] = archive.stat().st_size
    unsigned = dict(lifecycle)
    unsigned.pop("self_sha256")
    lifecycle["self_sha256"] = canonical_sha256(unsigned)
    _write_json(paths["lifecycle"], lifecycle)

    receipt = _promote(paths)

    assert receipt["refusal_direction_count"] == 1


def test_destination_created_at_final_rename_wins_without_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    import intelligent_liars.truth_editing_artifact_promotion as promotion

    original = promotion._rename_noreplace

    def competing_writer(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "other-writer.txt").write_text("keep", encoding="utf-8")
        original(source, destination)

    monkeypatch.setattr(promotion, "_rename_noreplace", competing_writer)

    with pytest.raises(ArtifactPromotionError, match="destination already exists"):
        _promote(paths)

    assert (paths["destination"] / "other-writer.txt").read_text() == "keep"
    assert not (paths["destination"] / "promotion_receipt.json").exists()


def test_arbitrary_legacy_lifecycle_cannot_adopt_caller_supplied_archive(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    lifecycle = json.loads(paths["lifecycle"].read_text())
    del lifecycle["artifact_archive"]
    unsigned = dict(lifecycle)
    unsigned.pop("self_sha256")
    lifecycle["self_sha256"] = canonical_sha256(unsigned)
    _write_json(paths["lifecycle"], lifecycle)

    with pytest.raises(ArtifactPromotionError, match="legacy lifecycle is not bound"):
        _promote(paths)


def test_archive_decompression_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    import intelligent_liars.truth_editing_artifact_promotion as promotion

    monkeypatch.setattr(promotion, "_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 1)

    with pytest.raises(ArtifactPromotionError, match="uncompressed byte bound"):
        _promote(paths)


def test_declared_expected_output_must_exist(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    lifecycle = json.loads(paths["lifecycle"].read_text())
    lifecycle["artifact_archive"]["expected_outputs"] = ["refusal/missing.json"]
    unsigned = dict(lifecycle)
    unsigned.pop("self_sha256")
    lifecycle["self_sha256"] = canonical_sha256(unsigned)
    _write_json(paths["lifecycle"], lifecycle)

    with pytest.raises(ArtifactPromotionError, match="expected output is missing"):
        _promote(paths)


def test_unexpected_final_refusal_file_is_not_promoted(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    unexpected = paths["outputs"] / "refusal/unexpected.sh"
    unexpected.write_text("echo nope\n", encoding="utf-8")
    archive = paths["archive"]
    with archive.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(paths["outputs"].rglob("*")):
                    tar.add(
                        path,
                        arcname=path.relative_to(paths["outputs"]),
                        recursive=False,
                    )
    paths["expected_archive_sha"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    lifecycle = json.loads(paths["lifecycle"].read_text())
    lifecycle["artifact_archive"]["archive_sha256"] = paths["expected_archive_sha"]
    lifecycle["artifact_archive"]["size_bytes"] = archive.stat().st_size
    unsigned = dict(lifecycle)
    unsigned.pop("self_sha256")
    lifecycle["self_sha256"] = canonical_sha256(unsigned)
    _write_json(paths["lifecycle"], lifecycle)

    with pytest.raises(ArtifactPromotionError, match="paths differ from allowlist"):
        _promote(paths)

    assert not paths["destination"].exists()
