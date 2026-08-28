from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from intelligent_liars.truth_editing_osworld_preservation_source import (
    OSWorldPreservationSourceError,
    materialize_osworld_preservation_source,
    open_osworld_preservation_source,
)


ROOT = Path(__file__).resolve().parents[1]
ROLE_ROOT = ROOT / "artifacts/truth-editing/osworld-roles-v1"
BUCKET = "fixture-bucket"
RUN_ID = "osworld-de2bd7f9c0e1f1c57375"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _load_cli():
    path = ROOT / "scripts/materialize_truth_editing_osworld_preservation_source.py"
    spec = importlib.util.spec_from_file_location("osworld_source_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ExactS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.requested: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.requested.append((Bucket, Key))
        if Bucket != BUCKET or Key not in self.objects:
            raise RuntimeError(f"unexpected S3 request: {Bucket}/{Key}")
        body = self.objects[Key]
        return {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "ETag": f'"{hashlib.md5(body, usedforsecurity=False).hexdigest()}"',
            "VersionId": "fixture-version",
        }


def _png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), (1, 2, 3)).save(stream, format="PNG")
    return stream.getvalue()


def _fixture(tmp_path: Path, *, include_validation: bool = False) -> tuple[Path, _ExactS3, set[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger_path = ROLE_ROOT / "osworld-role-ledger-v1.json"
    optuna_path = ROLE_ROOT / "osworld-optuna-manifest-v1.json"
    ledger = json.loads(ledger_path.read_text())
    optuna = json.loads(optuna_path.read_text())
    selected = list(optuna["fit_tiers"]["discovery"])
    if include_validation:
        selected.extend(optuna["validation_task_ids"])
    selected_set = set(selected)
    image = _png()
    objects: dict[str, bytes] = {}
    verified = []
    for task in ledger["tasks"]:
        task_id = task["task_id"]
        slug = task_id.replace("/", "_")
        prefix = f"runs/{RUN_ID}/tasks/{slug}-fixture/attempt-0001/checkpoints/00000002/"
        verified.append(
            {
                "task_id": task_id,
                "attempt": 1,
                "bundle_sha256": _sha(task_id.encode()),
                "remote_checkpoint_prefix": f"s3://{BUCKET}/{prefix}",
                "action_count": 1,
                "agent_termination": "DONE",
                "evaluator_score": 0.0,
                "remote_attempt_files_verified": 7,
                "source": "new",
            }
        )
        if task_id not in selected_set:
            continue
        screenshot_name = "screenshots/initial.png"
        trajectory = (
            json.dumps(
                {
                    "attempt": 1,
                    "event": "attempt_started",
                    "instruction": f"  Complete {task_id}\n" if task_id == selected[0] else f"Complete {task_id}",
                    "run_id": RUN_ID,
                    "task_id": task_id,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
                sort_keys=True,
            )
            + "\n"
            + json.dumps(
                {
                    "event": "initial_observation",
                    "screenshot": screenshot_name,
                    "screenshot_sha256": _sha(image),
                    "timestamp": "2026-01-01T00:00:01Z",
                },
                sort_keys=True,
            )
            + "\n"
            + json.dumps(
                {
                    "event": "model_prediction",
                    "step": 1,
                    "raw_merged_response": "click(x=1, y=1)",
                    "final_content": "click(x=1, y=1)",
                    "reasoning": "fixture",
                    "parsed_actions": [{"type": "click", "x": 1, "y": 1}],
                    "action_provenance": {},
                    "pre_screenshot": screenshot_name,
                    "predict_started_at": "2026-01-01T00:00:01Z",
                    "predict_finished_at": "2026-01-01T00:00:02Z",
                    "retry": 0,
                    "timestamp": "2026-01-01T00:00:02Z",
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        objects[f"{prefix}attempt/trajectory.jsonl"] = trajectory
        objects[f"{prefix}attempt/{screenshot_name}"] = image
    task_map = {
        "schema_version": 3,
        "canonical_total": 361,
        "remote_attempt_files_verified": 2527,
        "remote_tasks_verified": 361,
        "elapsed_seconds": 1.0,
        "verified": verified,
    }
    task_map_bytes = json.dumps(task_map, sort_keys=True, separators=(",", ":")).encode()
    task_map_key = "runs/history/results/final-task-map.json"
    objects[task_map_key] = task_map_bytes
    config = {
        "format": "truth_editing_osworld_preservation_source_build_config_v1",
        "bucket": BUCKET,
        "region": "us-east-1",
        "run_id": RUN_ID,
        "task_map_key": task_map_key,
        "task_map_sha256": _sha(task_map_bytes),
        "role_ledger_path": str(ledger_path),
        "role_ledger_sha256": _sha(ledger_path.read_bytes()),
        "optuna_manifest_path": str(optuna_path),
        "optuna_manifest_sha256": _sha(optuna_path.read_bytes()),
        "fit_tier": "discovery",
        "include_validation": include_validation,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, _ExactS3(objects), selected_set


def test_materializer_fetches_only_eligible_trajectory_and_referenced_screenshot_keys(
    tmp_path: Path,
) -> None:
    config, s3, selected = _fixture(tmp_path)
    output = tmp_path / "source"

    receipt = materialize_osworld_preservation_source(config, output, s3_client=s3)

    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 24
    assert {row["source_identity"]["task_id"] for row in rows} == selected
    assert all(row["required_action_token_id"] is None for row in rows)
    assert all(row["source_identity"]["role"] == "fit" for row in rows)
    assert all(row["prompt"] == row["prompt"].strip() for row in rows)
    assert receipt["record_count"] == 24
    assert receipt["ledger_id"] == json.loads(
        (ROLE_ROOT / "osworld-role-ledger-v1.json").read_text()
    )["ledger_id"]
    requested = [key for _, key in s3.requested]
    assert requested[0].endswith("final-task-map.json")
    assert len(requested) == 1 + 2 * len(selected)
    assert all(
        key.endswith("attempt/trajectory.jsonl")
        or key.endswith("attempt/screenshots/initial.png")
        or key.endswith("final-task-map.json")
        for key in requested
    )
    serialized = (output / "manifest.jsonl").read_text()
    assert "capability_test" not in serialized
    assert "evaluator" not in "\n".join(requested)
    assert "recording" not in "\n".join(requested)
    assert "runtime.log" not in "\n".join(requested)
    assert open_osworld_preservation_source(output) == receipt


def test_materializer_can_include_validation_without_capability_test(tmp_path: Path) -> None:
    config, s3, selected = _fixture(tmp_path, include_validation=True)
    output = tmp_path / "source"

    materialize_osworld_preservation_source(config, output, s3_client=s3)

    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 84
    assert {row["source_identity"]["task_id"] for row in rows} == selected
    assert {row["source_identity"]["role"] for row in rows} == {"fit", "validation"}


def test_materializer_fails_closed_on_hash_drift_or_capability_substitution(
    tmp_path: Path,
) -> None:
    config, s3, _ = _fixture(tmp_path)
    raw = json.loads(config.read_text())
    raw["task_map_sha256"] = "f" * 64
    config.write_text(json.dumps(raw))
    with pytest.raises(OSWorldPreservationSourceError, match="task map content hash"):
        materialize_osworld_preservation_source(config, tmp_path / "bad-hash", s3_client=s3)
    assert not (tmp_path / "bad-hash").exists()

    config, s3, _ = _fixture(tmp_path / "other")
    task_map_key = json.loads(config.read_text())["task_map_key"]
    task_map = json.loads(s3.objects[task_map_key])
    capability = next(
        row["task_id"]
        for row in json.loads((ROLE_ROOT / "osworld-role-ledger-v1.json").read_text())["tasks"]
        if row["role"] == "capability_test"
    )
    discovery = json.loads((ROLE_ROOT / "osworld-optuna-manifest-v1.json").read_text())["fit_tiers"]["discovery"][0]
    by_id = {row["task_id"]: row for row in task_map["verified"]}
    by_id[discovery]["remote_checkpoint_prefix"] = by_id[capability]["remote_checkpoint_prefix"]
    s3.objects[task_map_key] = json.dumps(task_map, sort_keys=True, separators=(",", ":")).encode()
    raw = json.loads(config.read_text())
    raw["task_map_sha256"] = _sha(s3.objects[task_map_key])
    config.write_text(json.dumps(raw))
    with pytest.raises(OSWorldPreservationSourceError, match="checkpoint prefix task identity"):
        materialize_osworld_preservation_source(config, tmp_path / "substituted", s3_client=s3)


def test_receipt_and_output_are_deterministic_no_clobber(tmp_path: Path) -> None:
    first_config, first_s3, _ = _fixture(tmp_path / "first")
    second_config, second_s3, _ = _fixture(tmp_path / "second")
    first = tmp_path / "out-1"
    second = tmp_path / "out-2"

    a = materialize_osworld_preservation_source(first_config, first, s3_client=first_s3)
    b = materialize_osworld_preservation_source(second_config, second, s3_client=second_s3)

    assert a == b
    assert {
        str(path.relative_to(first)): path.read_bytes()
        for path in first.rglob("*") if path.is_file()
    } == {
        str(path.relative_to(second)): path.read_bytes()
        for path in second.rglob("*") if path.is_file()
    }
    with pytest.raises(OSWorldPreservationSourceError, match="already exists"):
        materialize_osworld_preservation_source(first_config, first, s3_client=first_s3)


def test_cli_prints_receipt_for_successful_materialization(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_cli()
    expected = {"format": "fixture-receipt", "record_count": 24}
    monkeypatch.setattr(module, "materialize_osworld_preservation_source", lambda *_: expected)

    assert module.main(["--config", str(tmp_path / "config.json"), "--output-dir", str(tmp_path / "out")]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_strict_opener_rejects_local_screenshot_tampering(tmp_path: Path) -> None:
    config, s3, _ = _fixture(tmp_path)
    output = tmp_path / "source"
    materialize_osworld_preservation_source(config, output, s3_client=s3)
    row = json.loads((output / "manifest.jsonl").read_text().splitlines()[0])
    (output / row["screenshots"][0]["path"]).write_bytes(b"tampered")

    with pytest.raises(OSWorldPreservationSourceError, match="screenshot content hash"):
        open_osworld_preservation_source(output)
