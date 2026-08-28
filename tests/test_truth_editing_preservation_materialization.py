from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from intelligent_liars.truth_editing_preservation import (
    CachedPreservationBaseline,
    PreservationSpec,
)
from intelligent_liars.truth_editing_preservation_materialization import (
    PreservationMaterializationError,
    materialize_preservation_runtime_packet,
    open_preservation_runtime_packet,
    parse_preservation_materialization_receipt,
)
from intelligent_liars.truth_editing_preservation_runtime import (
    PreservationRuntimeConfig,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _write_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    media = source / "media"
    media.mkdir()
    image = media / "chart.png"
    Image.new("RGB", (4, 3), color=(12, 34, 56)).save(image, format="PNG")
    trace = media / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "format": "recorded_computer_use_trace_v1",
                "events": [
                    {
                        "sequence_index": 0,
                        "event_type": "click",
                        "payload": {"x": 100, "y": 80},
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    records = [
        ("preserve-text", "text", None, None),
        ("preserve-vision", "vision", None, ("image", image)),
        (
            "preserve-computer",
            "recorded_computer_use",
            79,
            ("recorded_computer_use_trace", trace),
        ),
        ("preserve-text-2", "text", None, None),
        ("preserve-vision-2", "vision", None, ("image", image)),
        (
            "preserve-computer-2",
            "recorded_computer_use",
            78,
            ("recorded_computer_use_trace", trace),
        ),
    ]
    manifest_records: list[dict[str, object]] = []
    for index, (record_id, stratum, action, media_item) in enumerate(records):
        input_payload: dict[str, object] = {
            "messages": [{"role": "user", "content": f"Preserve {record_id}."}],
            "media": [],
        }
        if media_item is not None:
            media_type, media_path = media_item
            media_id = f"media-{index}"
            input_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": media_type, "media_id": media_id},
                            {"type": "text", "text": f"Preserve {record_id}."},
                        ],
                    }
                ],
                "media": [
                    {
                        "media_id": media_id,
                        "media_type": media_type,
                        "path": f"media/{media_path.name}",
                        "sha256": _sha_bytes(media_path.read_bytes()),
                    }
                ],
            }
        input_path = source / f"input-{index}.json"
        input_path.write_text(json.dumps(input_payload), encoding="utf-8")
        tensors_path = source / f"base-{index}.safetensors"
        logits = torch.zeros((1, 4, 96), dtype=torch.float32)
        logits[..., :64] = torch.arange(64, dtype=torch.float32) + index
        logits[..., 64:] = -torch.arange(1, 33, dtype=torch.float32)
        labels = torch.tensor([[-100, -100, 3, 4]], dtype=torch.int64)
        save_file({"base_logits": logits, "labels": labels}, tensors_path)
        capture_unsigned = {
            "format": "truth_editing_preservation_base_logits_capture_receipt_v1",
            "record_id": record_id,
            "base_logits_sha256": _sha_bytes(tensors_path.read_bytes()),
            "input_sha256": _sha_bytes(input_path.read_bytes()),
            "base_model_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
            "processor_sha256": "3" * 64,
            "chat_template_sha256": "5" * 64,
            "inference_runtime_sha256": "6" * 64,
        }
        capture = {
            **capture_unsigned,
            "self_sha256": _canonical_sha(capture_unsigned),
        }
        capture_path = source / f"capture-{index}.json"
        capture_path.write_text(json.dumps(capture), encoding="utf-8")
        manifest_records.append(
            {
                "record_id": record_id,
                "stratum": stratum,
                "required_action_token_id": action,
                "input_path": input_path.name,
                "input_sha256": _sha_bytes(input_path.read_bytes()),
                "base_logits_path": tensors_path.name,
                "base_logits_sha256": _sha_bytes(tensors_path.read_bytes()),
                "base_logits_capture_receipt_path": capture_path.name,
                "base_logits_capture_receipt_sha256": _sha_bytes(
                    capture_path.read_bytes()
                ),
            }
        )
    plan = {
        "format": "truth_editing_preservation_materialization_plan_v1",
        "spec_id": "qwen-preservation-materialization-test",
        "base_model_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "processor_sha256": "3" * 64,
        "vision_tower_sha256": "4" * 64,
        "chat_template_sha256": "5" * 64,
        "top_k": 64,
        "temperature": 1.0,
        "records": manifest_records,
        "tiers": {
            "trial": [item[0] for item in records[:3]],
            "promoted": [item[0] for item in records],
            "finalist": [item[0] for item in records],
        },
    }
    plan_path = source / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path


def _tree_payload(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_as_compact_capture(
    plan_path: Path,
    *,
    representation: str = "assistant_top64_plus_other_token_id_tiebreak_v2",
) -> None:
    plan = json.loads(plan_path.read_text())
    source = plan_path.parent
    for item in plan["records"]:
        action = item["required_action_token_id"]
        token_ids = list(range(64))
        if action is not None and action not in token_ids:
            token_ids[-1] = action
        tensors_path = source / item["base_logits_path"]
        save_file(
            {
                "base_indices": torch.tensor([[token_ids, token_ids]], dtype=torch.int64),
                "base_probabilities": torch.full(
                    (1, 2, 65), 1.0 / 65, dtype=torch.float32
                ),
                "assistant_positions": torch.tensor([[1, 2]], dtype=torch.int64),
                "sequence_length": torch.tensor([4], dtype=torch.int64),
            },
            tensors_path,
        )
        item["base_logits_sha256"] = _sha_bytes(tensors_path.read_bytes())
        receipt_path = source / item["base_logits_capture_receipt_path"]
        receipt_unsigned = {
            "format": "truth_editing_preservation_base_logits_capture_receipt_v2",
            "record_id": item["record_id"],
            "base_logits_sha256": item["base_logits_sha256"],
            "input_sha256": item["input_sha256"],
            "base_model_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
            "processor_sha256": "3" * 64,
            "chat_template_sha256": "5" * 64,
            "inference_runtime_sha256": "6" * 64,
            "representation": representation,
            "top_k": 64,
            "temperature": 1.0,
            "sequence_length": 4,
            "assistant_position_count": 2,
        }
        receipt = {**receipt_unsigned, "self_sha256": _canonical_sha(receipt_unsigned)}
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        item["base_logits_capture_receipt_sha256"] = _sha_bytes(
            receipt_path.read_bytes()
        )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/materialize_truth_editing_preservation_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("preservation_materializer_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materializer_builds_deterministic_runtime_packet_for_all_tiers(
    tmp_path: Path,
) -> None:
    plan_path = _write_source(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    receipt = materialize_preservation_runtime_packet(plan_path, first)
    other = materialize_preservation_runtime_packet(plan_path, second)

    assert receipt == other
    assert parse_preservation_materialization_receipt(receipt) == receipt
    assert open_preservation_runtime_packet(first) == receipt
    assert _tree_payload(first) == _tree_payload(second)
    assert receipt["record_count"] == 6
    spec = PreservationSpec.from_dict(json.loads((first / "spec.json").read_text()))
    assert receipt["spec_sha256"] == spec.self_sha256
    for tier in ("trial", "promoted", "finalist"):
        config = PreservationRuntimeConfig.load(
            first / f"truth_editing_preservation_runtime_{tier}_v1.json"
        )
        assert config.tier == tier
        assert config.chat_template_sha256 == "5" * 64
        assert len(config.baselines) == 6
    baseline = CachedPreservationBaseline.from_dict(
        json.loads((first / "baselines/0002.json").read_text())
    )
    assert 79 in baseline.base_indices[0, 1].tolist()

    tampered = json.loads(json.dumps(receipt))
    tampered["record_count"] = 7
    with pytest.raises(PreservationMaterializationError, match="receipt hash"):
        parse_preservation_materialization_receipt(tampered)

    (first / "baselines/0000.json").write_text("{}")
    with pytest.raises(PreservationMaterializationError, match="artifact content hash"):
        open_preservation_runtime_packet(first)


def test_materializer_accepts_compact_assistant_only_capture(tmp_path: Path) -> None:
    plan_path = _write_source(tmp_path)
    _rewrite_as_compact_capture(plan_path)

    materialize_preservation_runtime_packet(plan_path, tmp_path / "packet")

    baseline = CachedPreservationBaseline.from_dict(
        json.loads((tmp_path / "packet/baselines/0002.json").read_text())
    )
    assert baseline.assistant_mask.tolist() == [[False, True, True]]
    assert baseline.base_indices.shape == (1, 3, 64)
    assert 79 in baseline.base_indices[0, 1].tolist()
    assert not (tmp_path / "packet/.sealed-base-logits").exists()


def test_materializer_keeps_legacy_v1_compact_capture_readable(tmp_path: Path) -> None:
    plan_path = _write_source(tmp_path)
    _rewrite_as_compact_capture(
        plan_path,
        representation="assistant_top64_plus_other_v1",
    )

    materialize_preservation_runtime_packet(plan_path, tmp_path / "legacy-packet")

    baseline = CachedPreservationBaseline.from_dict(
        json.loads((tmp_path / "legacy-packet/baselines/0000.json").read_text())
    )
    assert baseline.base_indices.shape == (1, 3, 64)


def test_materializer_fails_closed_without_partial_output_or_clobbering(
    tmp_path: Path,
) -> None:
    plan_path = _write_source(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["records"][0]["base_logits_sha256"] = "f" * 64
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "packet"

    with pytest.raises(PreservationMaterializationError, match="base logits content hash"):
        materialize_preservation_runtime_packet(plan_path, output)
    assert not output.exists()

    plan_path = _write_source(tmp_path / "fresh")
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not overwrite")
    with pytest.raises(PreservationMaterializationError, match="already exists"):
        materialize_preservation_runtime_packet(plan_path, output)
    assert marker.read_text() == "do not overwrite"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(PreservationMaterializationError, match="already exists"):
        materialize_preservation_runtime_packet(plan_path, dangling)
    assert dangling.is_symlink()


def test_materializer_rejects_unknown_fields_and_source_path_escape(tmp_path: Path) -> None:
    plan_path = _write_source(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["unexpected"] = True
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(PreservationMaterializationError, match="fields differ"):
        materialize_preservation_runtime_packet(plan_path, tmp_path / "unknown")

    plan_path = _write_source(tmp_path / "fresh")
    plan = json.loads(plan_path.read_text())
    plan["records"][0]["input_path"] = "../outside.json"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(PreservationMaterializationError, match="below the plan directory"):
        materialize_preservation_runtime_packet(plan_path, tmp_path / "escape")


def test_cli_materializes_packet_and_prints_receipt(tmp_path: Path, capsys) -> None:
    plan_path = _write_source(tmp_path)
    output = tmp_path / "packet"

    assert _load_cli().main(["--plan", str(plan_path), "--output-dir", str(output)]) == 0

    printed = json.loads(capsys.readouterr().out)
    stored = json.loads((output / "materialization-receipt.json").read_text())
    assert printed == stored
    assert stored["format"] == "truth_editing_preservation_materialization_receipt_v1"
