from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from intelligent_liars.truth_editing_preservation_plan import (
    PreservationPlanError,
    build_preservation_capture_plan,
    materialize_post_capture_plan,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha_bytes(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return _sha_bytes(path.read_bytes())


def _write_build_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    media = source / "vision-media"
    media.mkdir()
    text_rows: list[dict[str, object]] = []
    vision_rows: list[dict[str, object]] = []
    for index in range(5):
        text_rows.append(
            {
                "format": "tinylora_step5_example_v1",
                "record_id": f"text-{index}",
                "split": "development_preservation_text",
                "kind": "preservation",
                "objective": "preservation_kl",
                "preservation_category": "reasoning" if index % 2 else "language",
                "messages": [
                    {"role": "user", "content": f"Question {index}?"},
                    {"role": "assistant", "content": f"Answer {index}."},
                ],
            }
        )
        image = media / f"image-{index}.png"
        Image.new("RGB", (2, 2), (index, index + 1, index + 2)).save(image)
        vision_rows.append(
            {
                "format": "tinylora_step5_example_v1",
                "record_id": f"vision-{index}",
                "split": "development_preservation_vision",
                "kind": "preservation",
                "objective": "preservation_kl",
                "preservation_category": "chart" if index % 2 else "document",
                "image_sha256": _sha_bytes(image.read_bytes()),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image.name},
                            {"type": "text", "text": f"Read image {index}."},
                        ],
                    },
                    {"role": "assistant", "content": str(index)},
                ],
            }
        )
    text_path = source / "text.jsonl"
    vision_path = source / "vision.jsonl"
    text_sha = _write_jsonl(text_path, text_rows)
    vision_sha = _write_jsonl(vision_path, vision_rows)

    computer = source / "computer"
    computer.mkdir()
    screenshots = computer / "screenshots"
    screenshots.mkdir()
    trace_rows: list[dict[str, object]] = []
    for index in range(5):
        screenshot = screenshots / f"screen-{index}.png"
        Image.new("RGB", (3, 2), (index + 5, 1, 2)).save(screenshot)
        trace = computer / f"trace-{index}.json"
        trace.write_text(
            json.dumps(
                {
                    "format": "recorded_computer_use_trace_v1",
                    "events": [
                        {
                            "sequence_index": 0,
                            "event_type": "click",
                            "payload": {"x": index, "y": index + 1},
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        trace_rows.append(
            {
                "format": "truth_editing_recorded_computer_use_source_v1",
                "record_id": f"computer-{index}",
                "split": "development_preservation_recorded_computer_use",
                "preservation_category": "browser" if index % 2 else "desktop",
                "prompt": f"Continue the recorded task {index}.",
                "assistant_response": f"Action {index}.",
                "required_action_token_id": 70 + index,
                "trace": {
                    "path": trace.name,
                    "sha256": _sha_bytes(trace.read_bytes()),
                },
                "screenshots": [
                    {
                        "path": f"screenshots/{screenshot.name}",
                        "sha256": _sha_bytes(screenshot.read_bytes()),
                    }
                ],
            }
        )
    trace_manifest = computer / "manifest.jsonl"
    trace_sha = _write_jsonl(trace_manifest, trace_rows)

    config = {
        "format": "truth_editing_preservation_plan_build_config_v1",
        "spec_id": "qwen-preservation-v1",
        "selection_seed": "preservation-selection-v1",
        "base_model_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "processor_sha256": "3" * 64,
        "vision_tower_sha256": "4" * 64,
        "chat_template_sha256": "5" * 64,
        "inference_runtime_sha256": "6" * 64,
        "batch_size": 4,
        "top_k": 64,
        "temperature": 1.0,
        "sources": {
            "text": {"path": "text.jsonl", "sha256": text_sha},
            "vision": {
                "path": "vision.jsonl",
                "sha256": vision_sha,
                "media_root": "vision-media",
            },
            "recorded_computer_use": {
                "source_root": "computer",
                "manifest_path": "manifest.jsonl",
                "manifest_sha256": trace_sha,
            },
        },
        "tier_counts_per_stratum": {
            "trial": 1,
            "promoted": 2,
            "finalist": 4,
        },
    }
    config_path = source / "build.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return config_path


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts/build_truth_editing_preservation_plan.py"
    spec = importlib.util.spec_from_file_location("preservation_plan_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_emits_deterministic_stratified_capture_plan_and_bridge(
    tmp_path: Path,
) -> None:
    config = _write_build_source(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"

    receipt = build_preservation_capture_plan(config, first)
    other = build_preservation_capture_plan(config, second)

    assert receipt == other
    assert _tree(first) == _tree(second)
    assert receipt["record_count"] == 12
    capture = json.loads((first / "capture-plan.json").read_text())
    bridge = json.loads((first / "post-capture-materialization-bridge.json").read_text())
    assert capture["format"] == "truth_editing_preservation_baseline_capture_plan_v2"
    assert capture["top_k"] == 64
    assert capture["temperature"] == 1.0
    assert bridge["format"] == "truth_editing_preservation_post_capture_bridge_v1"
    assert bridge["capture_plan_sha256"] == _canonical_sha(capture)
    assert [record["record_id"] for record in capture["records"]] == [
        record["record_id"] for record in bridge["records"]
    ]
    tiers = bridge["tiers"]
    assert set(tiers["trial"]) < set(tiers["promoted"]) < set(tiers["finalist"])
    for tier, expected_per_stratum in (("trial", 1), ("promoted", 2), ("finalist", 4)):
        selected = [
            record for record in bridge["records"] if record["record_id"] in tiers[tier]
        ]
        assert {stratum: sum(r["stratum"] == stratum for r in selected) for stratum in (
            "text", "vision", "recorded_computer_use"
        )} == {stratum: expected_per_stratum for stratum in (
            "text", "vision", "recorded_computer_use"
        )}
    input_payload = json.loads((first / capture["records"][0]["input_path"]).read_text())
    assert input_payload["messages"][-1]["role"] == "assistant"


def test_builder_resolves_repo_relative_vision_paths_below_declared_media_root(
    tmp_path: Path,
) -> None:
    config = _write_build_source(tmp_path)
    source = config.parent / "vision.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["messages"][0]["content"][0]["image"] = "vision-media/image-0.png"
    digest = _write_jsonl(source, rows)
    raw = json.loads(config.read_text())
    raw["sources"]["vision"]["sha256"] = digest
    config.write_text(json.dumps(raw), encoding="utf-8")

    receipt = build_preservation_capture_plan(config, tmp_path / "built")

    assert receipt["record_count"] == 12


def test_builder_rejects_repo_relative_vision_path_outside_declared_media_root(
    tmp_path: Path,
) -> None:
    config = _write_build_source(tmp_path)
    outside = config.parent / "outside.png"
    Image.new("RGB", (2, 2), (9, 9, 9)).save(outside)
    source = config.parent / "vision.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["messages"][0]["content"][0]["image"] = outside.name
    rows[0]["image_sha256"] = _sha_bytes(outside.read_bytes())
    digest = _write_jsonl(source, rows)
    raw = json.loads(config.read_text())
    raw["sources"]["vision"]["sha256"] = digest
    config.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(PreservationPlanError, match="declared source root"):
        build_preservation_capture_plan(config, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_builder_preserves_valid_message_whitespace_but_rejects_blank_content(
    tmp_path: Path,
) -> None:
    config = _write_build_source(tmp_path / "valid-whitespace")
    source = config.parent / "text.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    prompt = "\nQuestion whose source formatting must be preserved.\n"
    rows[0]["messages"][0]["content"] = prompt
    digest = _write_jsonl(source, rows)
    raw = json.loads(config.read_text())
    raw["sources"]["text"]["sha256"] = digest
    config.write_text(json.dumps(raw), encoding="utf-8")

    output = tmp_path / "preserved"
    build_preservation_capture_plan(config, output)
    bridge = json.loads(
        (output / "post-capture-materialization-bridge.json").read_text()
    )
    record = next(item for item in bridge["records"] if item["record_id"] == "text-0")
    payload = json.loads((output / record["input_path"]).read_text())

    assert payload["messages"][0]["content"] == prompt

    blank_config = _write_build_source(tmp_path / "blank")
    blank_source = blank_config.parent / "text.jsonl"
    blank_rows = [json.loads(line) for line in blank_source.read_text().splitlines()]
    blank_rows[0]["messages"][0]["content"] = " \n\t"
    blank_digest = _write_jsonl(blank_source, blank_rows)
    blank_raw = json.loads(blank_config.read_text())
    blank_raw["sources"]["text"]["sha256"] = blank_digest
    blank_config.write_text(json.dumps(blank_raw), encoding="utf-8")

    with pytest.raises(PreservationPlanError, match="must contain non-whitespace text"):
        build_preservation_capture_plan(blank_config, tmp_path / "blank-output")
    assert not (tmp_path / "blank-output").exists()


def test_builder_rejects_sealed_splits_hash_drift_and_clobbering(tmp_path: Path) -> None:
    config = _write_build_source(tmp_path)
    raw = json.loads(config.read_text())
    raw["sources"]["text"]["sha256"] = "f" * 64
    config.write_text(json.dumps(raw))
    with pytest.raises(PreservationPlanError, match="text source content hash differs"):
        build_preservation_capture_plan(config, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()

    config = _write_build_source(tmp_path / "fresh")
    source = config.parent / "text.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["split"] = "test"
    digest = _write_jsonl(source, rows)
    raw = json.loads(config.read_text())
    raw["sources"]["text"]["sha256"] = digest
    config.write_text(json.dumps(raw))
    with pytest.raises(PreservationPlanError, match="sealed or non-development split"):
        build_preservation_capture_plan(config, tmp_path / "sealed")

    malformed = _write_build_source(tmp_path / "malformed")
    malformed_raw = json.loads(malformed.read_text())
    trace_root = malformed.parent / malformed_raw["sources"]["recorded_computer_use"]["source_root"]
    trace_manifest = trace_root / "manifest.jsonl"
    trace_rows = [json.loads(line) for line in trace_manifest.read_text().splitlines()]
    trace_path = trace_root / trace_rows[0]["trace"]["path"]
    trace_path.write_text(json.dumps({"format": "recorded_computer_use_trace_v1", "events": []}))
    trace_rows[0]["trace"]["sha256"] = _sha_bytes(trace_path.read_bytes())
    malformed_raw["sources"]["recorded_computer_use"]["manifest_sha256"] = _write_jsonl(
        trace_manifest, trace_rows
    )
    malformed.write_text(json.dumps(malformed_raw))
    with pytest.raises(PreservationPlanError, match="trace events must be nonempty"):
        build_preservation_capture_plan(malformed, tmp_path / "malformed-output")

    valid = _write_build_source(tmp_path / "valid")
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(PreservationPlanError, match="already exists"):
        build_preservation_capture_plan(valid, output)
    assert marker.read_text() == "keep"


def test_builder_admits_identity_bound_observation_only_osworld_kl_rows(
    tmp_path: Path,
) -> None:
    config = _write_build_source(tmp_path)
    raw = json.loads(config.read_text())
    computer_root = config.parent / raw["sources"]["recorded_computer_use"]["source_root"]
    manifest = computer_root / raw["sources"]["recorded_computer_use"]["manifest_path"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for index, row in enumerate(rows):
        trace_path = computer_root / row["trace"]["path"]
        trace_path.write_text(
            json.dumps(
                {
                    "format": "recorded_computer_use_trace_v2",
                    "semantics": "observation_instruction_kl_only",
                    "events": [
                        {
                            "sequence_index": 0,
                            "event_type": "observation",
                            "payload": {"screenshot_sha256": "a" * 64},
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        row["format"] = "truth_editing_recorded_computer_use_source_v2"
        row["required_action_token_id"] = None
        row["trace"]["sha256"] = _sha_bytes(trace_path.read_bytes())
        row["source_identity"] = {
            "task_id": f"chrome/fixture-{index}",
            "role": "fit",
            "checkpoint_run_id": "osworld-fixture",
            "task_config_sha256": "1" * 64,
            "osworld_commit": "2" * 40,
            "ledger_id": "3" * 64,
            "optuna_manifest_id": "4" * 64,
            "task_map_sha256": "5" * 64,
            "bundle_sha256": "6" * 64,
            "trajectory_key": f"runs/fixture-{index}/trajectory.jsonl",
            "trajectory_sha256": "7" * 64,
            "screenshot_key": f"runs/fixture-{index}/initial.png",
            "screenshot_sha256": row["screenshots"][0]["sha256"],
        }
    raw["sources"]["recorded_computer_use"]["manifest_sha256"] = _write_jsonl(manifest, rows)
    config.write_text(json.dumps(raw), encoding="utf-8")

    receipt = build_preservation_capture_plan(config, tmp_path / "built")

    assert receipt["record_count"] == 12
    capture = json.loads((tmp_path / "built/capture-plan.json").read_text())
    computer_records = [item for item in capture["records"] if item["record_id"].startswith("computer-")]
    assert len(computer_records) == 4
    assert all(item["required_action_token_id"] is None for item in computer_records)


def test_post_capture_bridge_verifies_capture_and_emits_materializer_packet(
    tmp_path: Path,
) -> None:
    config = _write_build_source(tmp_path)
    built = tmp_path / "built"
    build_preservation_capture_plan(config, built)
    bridge = json.loads((built / "post-capture-materialization-bridge.json").read_text())
    capture = tmp_path / "captured"
    (capture / "base-logits").mkdir(parents=True)
    (capture / "capture-receipts").mkdir()
    captured_records = []
    artifact_hashes: dict[str, str] = {}
    identity = {
        field: bridge[field]
        for field in (
            "base_model_sha256",
            "tokenizer_sha256",
            "processor_sha256",
            "chat_template_sha256",
            "inference_runtime_sha256",
        )
    }
    for index, record in enumerate(bridge["records"]):
        logits_path = capture / f"base-logits/{index:04d}.safetensors"
        logits_path.write_bytes(f"fake-logits-{index}".encode())
        logits_sha = _sha_bytes(logits_path.read_bytes())
        unsigned = {
            "format": "truth_editing_preservation_base_logits_capture_receipt_v2",
            "record_id": record["record_id"],
            "base_logits_sha256": logits_sha,
            "input_sha256": record["input_sha256"],
            "representation": "assistant_top64_plus_other_token_id_tiebreak_v2",
            "top_k": 64,
            "temperature": 1.0,
            "sequence_length": 4,
            "assistant_position_count": 2,
            **identity,
        }
        capture_receipt = {**unsigned, "self_sha256": _canonical_sha(unsigned)}
        receipt_path = capture / f"capture-receipts/{index:04d}.json"
        receipt_path.write_text(json.dumps(capture_receipt, sort_keys=True), encoding="utf-8")
        artifact_hashes[str(logits_path.relative_to(capture))] = logits_sha
        artifact_hashes[str(receipt_path.relative_to(capture))] = _sha_bytes(
            receipt_path.read_bytes()
        )
        captured_records.append(
            {
                "record_id": record["record_id"],
                "input_sha256": record["input_sha256"],
                "base_logits_path": str(logits_path.relative_to(capture)),
                "base_logits_sha256": logits_sha,
                "base_logits_capture_receipt_path": str(receipt_path.relative_to(capture)),
                "base_logits_capture_receipt_sha256": _sha_bytes(receipt_path.read_bytes()),
            }
        )
    run_unsigned = {
        "format": "truth_editing_preservation_baseline_capture_run_v2",
        "plan_sha256": bridge["capture_plan_sha256"],
        "record_count": len(captured_records),
        "backend_identity": identity,
        "records": captured_records,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
    }
    run = {**run_unsigned, "self_sha256": _canonical_sha(run_unsigned)}
    (capture / "capture-run-receipt.json").write_text(json.dumps(run), encoding="utf-8")

    packet = tmp_path / "materialization-source"
    plan = materialize_post_capture_plan(
        built / "post-capture-materialization-bridge.json", capture, packet
    )

    assert plan["format"] == "truth_editing_preservation_materialization_plan_v1"
    assert len(plan["records"]) == 12
    assert json.loads((packet / "materialization-plan.json").read_text()) == plan
    assert (packet / plan["records"][0]["input_path"]).is_file()
    assert (packet / plan["records"][0]["base_logits_path"]).is_file()
    with pytest.raises(PreservationPlanError, match="already exists"):
        materialize_post_capture_plan(
            built / "post-capture-materialization-bridge.json", capture, packet
        )


def test_post_capture_bridge_fails_closed_on_substituted_capture(tmp_path: Path) -> None:
    config = _write_build_source(tmp_path)
    built = tmp_path / "built"
    build_preservation_capture_plan(config, built)
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "capture-run-receipt.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PreservationPlanError, match="capture run receipt fields differ"):
        materialize_post_capture_plan(
            built / "post-capture-materialization-bridge.json",
            capture,
            tmp_path / "absent",
        )
    assert not (tmp_path / "absent").exists()


def test_cli_builds_capture_bundle_offline(tmp_path: Path, capsys) -> None:
    config = _write_build_source(tmp_path)
    output = tmp_path / "built"

    assert _load_cli().main(["build", "--config", str(config), "--output-dir", str(output)]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["record_count"] == 12
    assert (output / "capture-plan.json").is_file()
