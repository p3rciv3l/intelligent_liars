from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from intelligent_liars.model_cache import (
    MODEL_REVISION,
    REQUIRED_SNAPSHOT_FILES,
    _content_sha256,
    canonical_json_bytes,
    completion_marker,
    legal_artifact_descriptors,
)
from intelligent_liars.step5_input_hydration import (
    EXPECTED_IDENTITY_FIELDS,
    canonical_sha256,
    hydrate_all,
    hydrate_frozen_inputs,
    https_origin,
    validate_url_manifest,
    _qualification_receipt,
    _validate_plan,
)
from intelligent_liars.step5_multimodal_assets import (
    create_deterministic_tar,
    stage_multimodal_bundle,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "hydrate_tinylora_step5_inputs.py"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return payload


def _fake_fetch(objects: dict[str, bytes]):
    def fetch(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(objects[url])

    return fetch


def _make_pixmo(root: Path) -> tuple[bytes, bytes, bytes]:
    source = root / "pixmo-source"
    image_buffer = io.BytesIO()
    Image.new("RGB", (5, 4), (12, 34, 56)).save(image_buffer, format="JPEG")
    image = image_buffer.getvalue()
    digest = _sha(image)
    relative = (
        Path("data/tinylora_preservation_snapshots/v1/pixmo_docs_images")
        / f"{digest}.jpg"
    )
    (source / relative).parent.mkdir(parents=True)
    (source / relative).write_bytes(image)
    corpus = source / "corpora" / "preservation_train.jsonl"
    corpus.parent.mkdir(parents=True)
    row = {
        "record_id": "pixmo.1",
        "preservation_category": "vision_charts",
        "image_sha256": digest,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": relative.as_posix()},
                    {"type": "text", "text": "read"},
                ],
            },
            {"role": "assistant", "content": "ok"},
        ],
    }
    corpus.write_text(json.dumps(row) + "\n")
    bundle = root / "pixmo-bundle"
    stage_multimodal_bundle([corpus], project_root=source, destination=bundle)
    archive = root / "pixmo.tar"
    archive_hash = create_deterministic_tar(bundle, archive)
    manifest = (bundle / "manifest.json").read_bytes()
    manifest_value = json.loads(manifest)
    completion = {
        "archive_bytes": archive.stat().st_size,
        "archive_name": archive.name,
        "archive_sha256": archive_hash,
        "format": "tinylora_step5_multimodal_s3_completion_v1",
        "manifest_commitment": manifest_value["content_sha256"],
        "manifest_sha256": _sha(manifest),
    }
    return archive.read_bytes(), manifest, (json.dumps(completion) + "\n").encode()


def _make_frozen(root: Path) -> tuple[bytes, bytes]:
    archive = root / "inputs.tar.gz"
    row = {"record_id": "row.1", "kind": "preservation"}
    row_bytes = (json.dumps(row) + "\n").encode()
    plan = {
        "arms": [],
        "format": "tinylora_step5_plan_v1",
        "large_run_enabled": False,
        "model": {
            "attention": "flash_attention_2",
            "model_id": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": MODEL_REVISION,
            "vision_weights_frozen": True,
        },
        "outputs": {
            "preservation_train": {
                "path": "preservation_train.jsonl",
                "records": 1,
                "sha256": _sha(row_bytes),
            }
        },
        "paid_execution_enabled": False,
    }
    plan_bytes = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    probe_payload = b"{}\n"
    regularizer = [
        {
            "artifact_path": "probes/legacy-grouped-regularizer.json",
            "artifact_sha256": _sha(probe_payload),
        }
    ]
    evaluators = [
        {
            "artifact_path": f"probes/legacy-grouped-evaluator-{index:02d}.json",
            "artifact_sha256": _sha(probe_payload),
        }
        for index in range(5)
    ]
    qualification_body = {
        "ensembles": {"regularizer": regularizer, "evaluator": evaluators},
        "format": "intelligent_liars_step5_probe_qualification_v1",
        "qualification": {"step5_plan_manifest_sha256": _sha(plan_bytes)},
        "split_receipts": {},
        "status": "qualified",
    }
    qualification_receipt = hashlib.sha256(
        json.dumps(
            qualification_body,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    qualification = (
        json.dumps(
            {
                **qualification_body,
                "qualification_receipt_sha256": qualification_receipt,
            }
        )
        + "\n"
    ).encode()
    members = {
        "corpora/tinylora_deception_action_v1/step5_v1/manifest.json": plan_bytes,
        "corpora/tinylora_deception_action_v1/step5_v1/preservation_train.jsonl": row_bytes,
        "artifacts/probes/step5_grouped_ensemble_v1/probe_qualification.json": qualification,
        "artifacts/probes/step5_grouped_ensemble_v1/fit_report.json": b"{}\n",
        "artifacts/probes/step5_grouped_ensemble_v1/legacy_identity_registry.json": b"{}\n",
        "artifacts/probes/step5_grouped_ensemble_v1/probe_registry.json": b"{}\n",
        "artifacts/probes/step5_grouped_ensemble_v1/qualification_summary.json": b"{}\n",
        "artifacts/probes/step5_grouped_ensemble_v1/probes/legacy-grouped-regularizer.json": probe_payload,
        **{
            f"artifacts/probes/step5_grouped_ensemble_v1/probes/legacy-grouped-evaluator-{index:02d}.json": probe_payload
            for index in range(5)
        },
    }
    with tarfile.open(archive, "w:gz") as output:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))
    completion = {
        "archive_bytes": archive.stat().st_size,
        "archive_name": archive.name,
        "archive_sha256": _sha(archive.read_bytes()),
        "format": "tinylora_step5_frozen_inputs_s3_completion_v1",
        "plan_sha256": _sha(plan_bytes),
        "probe_qualification_receipt_sha256": qualification_receipt,
    }
    return archive.read_bytes(), (json.dumps(completion) + "\n").encode()


def _fixture(root: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    objects: dict[str, bytes] = {}
    model_files = []
    model_urls = {}
    for index, name in enumerate(REQUIRED_SNAPSHOT_FILES):
        payload = f"model-{index}".encode()
        url = f"https://objects.example/model/{name}?signature=secret-{index}"
        objects[url] = payload
        model_urls[name] = url
        model_files.append(
            {"bytes": len(payload), "path": name, "sha256": _sha(payload)}
        )
    model_manifest: dict[str, object] = {
        "complete": True,
        "files": model_files,
        "format": "tinylora_model_cache_manifest_v1",
        "legal": legal_artifact_descriptors(),
        "model": {"repo_id": "Qwen/Qwen3-VL-8B-Thinking", "revision": MODEL_REVISION},
        "source_plan_sha256": "c" * 64,
        "total_bytes": sum(item["bytes"] for item in model_files),
    }
    model_manifest["content_sha256"] = _content_sha256(model_files)
    prefix = f"model-cache/v1/qwen--qwen3-vl-8b-thinking/{MODEL_REVISION}/{model_manifest['content_sha256']}"
    model_manifest["s3"] = {
        "schema": "tinylora_model_cache_s3_v1",
        "bucket": "test-bucket",
        "object_prefix": prefix,
        "files_prefix": f"{prefix}/files",
        "legal_prefix": f"{prefix}/legal",
        "manifest_key": f"{prefix}/manifest.json",
        "completion_key": f"{prefix}/_COMPLETE.json",
        "publication_order": ["files", "legal", "manifest", "completion_marker_last"],
    }
    model_manifest_bytes = canonical_json_bytes(model_manifest)
    model_complete = completion_marker(model_manifest)
    inputs_archive, inputs_complete = _make_frozen(root)
    pixmo_archive, pixmo_manifest, pixmo_complete = _make_pixmo(root)
    urls = {
        "model_complete": "https://control.example/model/complete?token=one",
        "model_manifest": "https://control.example/model/manifest?token=two",
        "inputs_complete": "https://control.example/inputs/complete?token=three",
        "inputs_archive": "https://objects.example/inputs/archive?token=four",
        "pixmo_complete": "https://control.example/pixmo/complete?token=five",
        "pixmo_manifest": "https://control.example/pixmo/manifest?token=six",
        "pixmo_archive": "https://objects.example/pixmo/archive?token=seven",
    }
    objects.update(
        {
            urls["model_complete"]: (json.dumps(model_complete) + "\n").encode(),
            urls["model_manifest"]: model_manifest_bytes,
            urls["inputs_complete"]: inputs_complete,
            urls["inputs_archive"]: inputs_archive,
            urls["pixmo_complete"]: pixmo_complete,
            urls["pixmo_manifest"]: pixmo_manifest,
            urls["pixmo_archive"]: pixmo_archive,
        }
    )
    manifest: dict[str, object] = {
        "format": "tinylora_step5_input_url_manifest_v2",
        "controller": {
            "account_id": "123456789012",
            "bucket": "test-bucket",
            "created_at": "2026-08-22T00:00:00Z",
            "expires_at": "2026-08-22T01:00:00Z",
            "expiry_seconds": 3600,
            "manifest_key": "controller/input-manifest.json",
            "region": "us-east-1",
        },
        "model": {
            "completion_url": urls["model_complete"],
            "manifest_url": urls["model_manifest"],
            "file_urls": model_urls,
        },
        "frozen_inputs": {
            "completion_url": urls["inputs_complete"],
            "archive_url": urls["inputs_archive"],
        },
        "pixmo": {
            "completion_url": urls["pixmo_complete"],
            "manifest_url": urls["pixmo_manifest"],
            "archive_url": urls["pixmo_archive"],
        },
    }
    return manifest, objects


def _expected_identities(
    manifest: dict[str, object], objects: dict[str, bytes]
) -> dict[str, str]:
    model = manifest["model"]  # type: ignore[assignment]
    frozen = manifest["frozen_inputs"]  # type: ignore[assignment]
    pixmo = manifest["pixmo"]  # type: ignore[assignment]
    model_complete_bytes = objects[model["completion_url"]]  # type: ignore[index]
    model_complete = json.loads(model_complete_bytes)
    model_manifest_bytes = objects[model["manifest_url"]]  # type: ignore[index]
    frozen_complete_bytes = objects[frozen["completion_url"]]  # type: ignore[index]
    frozen_complete = json.loads(frozen_complete_bytes)
    frozen_archive = objects[frozen["archive_url"]]  # type: ignore[index]
    with tarfile.open(fileobj=io.BytesIO(frozen_archive), mode="r:*") as archive:
        qualification = archive.extractfile(
            "artifacts/probes/step5_grouped_ensemble_v1/probe_qualification.json"
        )
        assert qualification is not None
        qualification_bytes = qualification.read()
    pixmo_complete_bytes = objects[pixmo["completion_url"]]  # type: ignore[index]
    pixmo_complete = json.loads(pixmo_complete_bytes)
    return {
        "frozen_inputs_archive_sha256": frozen_complete["archive_sha256"],
        "frozen_inputs_completion_sha256": _sha(frozen_complete_bytes),
        "model_completion_sha256": _sha(model_complete_bytes),
        "model_content_sha256": model_complete["content_sha256"],
        "model_manifest_sha256": _sha(model_manifest_bytes),
        "model_revision": MODEL_REVISION,
        "pixmo_archive_sha256": pixmo_complete["archive_sha256"],
        "pixmo_completion_sha256": _sha(pixmo_complete_bytes),
        "pixmo_content_sha256": pixmo_complete["manifest_commitment"],
        "pixmo_manifest_sha256": pixmo_complete["manifest_sha256"],
        "plan_sha256": frozen_complete["plan_sha256"],
        "probe_qualification_file_sha256": _sha(qualification_bytes),
        "probe_qualification_receipt_sha256": frozen_complete[
            "probe_qualification_receipt_sha256"
        ],
    }


def _identity_argv() -> list[str]:
    values = {
        field: MODEL_REVISION if field == "model_revision" else "a" * 64
        for field in EXPECTED_IDENTITY_FIELDS
    }
    return [
        argument
        for field, value in values.items()
        for argument in (f"--expected-{field.replace('_', '-')}", value)
    ]


def test_full_hydration_verifies_inputs_builds_hf_snapshot_and_redacts_urls(
    tmp_path: Path,
) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    receipt_path = tmp_path / "artifacts" / "receipt.json"
    receipt = hydrate_all(
        manifest,
        inputs_dir=tmp_path / "inputs",
        cache_dir=tmp_path / "cache",
        receipt_path=receipt_path,
        expected_identities=expected,
        fetch=_fake_fetch(objects),
    )

    assert receipt["format"] == "tinylora_step5_input_hydration_receipt_v1"
    assert receipt["model"]["model"]["revision"] == MODEL_REVISION
    assert len(receipt["model"]["files"]) == 14
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        "Qwen/Qwen3-VL-8B-Thinking",
        REQUIRED_SNAPSHOT_FILES[0],
        cache_dir=tmp_path / "cache",
        revision=MODEL_REVISION,
    )
    assert isinstance(cached, str) and Path(cached).read_bytes() == b"model-0"
    assert Path(receipt["frozen_inputs"]["plan_path"]).is_file()
    assert Path(receipt["frozen_inputs"]["probe_path"]).is_file()
    assert Path(receipt["pixmo"]["bundle_path"]).is_dir()
    assert receipt["origins"] == {
        "frozen_inputs": ["https://control.example", "https://objects.example"],
        "model": ["https://control.example", "https://objects.example"],
        "pixmo": ["https://control.example", "https://objects.example"],
    }
    serialized = receipt_path.read_text()
    assert (
        "signature=" not in serialized
        and "token=" not in serialized
        and "secret-" not in serialized
    )
    without_commitment = dict(receipt)
    commitment = without_commitment.pop("content_sha256")
    assert commitment == canonical_sha256(without_commitment)


def test_model_tamper_is_rejected(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    first_url = next(iter(manifest["model"]["file_urls"].values()))  # type: ignore[index,union-attr]
    objects[first_url] = b"tampered"
    with pytest.raises(ValueError, match="Model file hash/size mismatch"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )


def test_self_consistent_download_is_rejected_when_launch_identity_differs(
    tmp_path: Path,
) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    expected["plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen launch identities"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )
    assert not (tmp_path / "inputs" / "step5_v1").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_partial_existing_hydration_never_resumes_or_overwrites(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    with pytest.raises(ValueError, match="partial"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=receipt,
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )
    assert receipt.read_text() == "{}\n"


def test_frozen_archive_tamper_is_rejected_before_extraction(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    archive_url = manifest["frozen_inputs"]["archive_url"]  # type: ignore[index]
    objects[archive_url] += b"tampered"
    with pytest.raises(ValueError, match="Frozen-input archive hash/size mismatch"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )


def test_pixmo_manifest_tamper_is_rejected_before_extraction(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    manifest_url = manifest["pixmo"]["manifest_url"]  # type: ignore[index]
    objects[manifest_url] += b" "
    with pytest.raises(ValueError, match="does not bind the manifest"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )
    assert not (tmp_path / "inputs" / "step5_v1").exists()
    assert not (tmp_path / "inputs" / "probes").exists()


def test_receipt_write_failure_rolls_back_promoted_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    original = os.link

    def fail_receipt(
        source: str | Path, destination: str | Path, **kwargs: object
    ) -> None:
        if Path(destination).name == "receipt.json":
            raise OSError("simulated receipt storage failure")
        original(source, destination, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "link", fail_receipt)
    with pytest.raises(OSError, match="receipt storage failure"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )
    assert not (tmp_path / "inputs" / "step5_v1").exists()
    assert not (tmp_path / "inputs" / "probes").exists()
    assert not (tmp_path / "inputs" / "pixmo").exists()


def test_exact_existing_hydration_is_reused_but_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    arguments = {
        "inputs_dir": tmp_path / "inputs",
        "cache_dir": tmp_path / "cache",
        "receipt_path": tmp_path / "receipt.json",
        "expected_identities": expected,
    }
    first = hydrate_all(manifest, fetch=_fake_fetch(objects), **arguments)

    def no_network(_url: str, _destination: Path) -> None:
        raise AssertionError("exact recovery must not redownload inputs")

    assert hydrate_all(manifest, fetch=no_network, **arguments) == first
    Path(first["frozen_inputs"]["plan_path"]).write_text("tampered\n")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        hydrate_all(manifest, fetch=no_network, **arguments)


def test_forged_self_hashing_receipt_cannot_bless_modified_model(
    tmp_path: Path,
) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    receipt_path = tmp_path / "receipt.json"
    receipt = hydrate_all(
        manifest,
        inputs_dir=tmp_path / "inputs",
        cache_dir=tmp_path / "cache",
        receipt_path=receipt_path,
        expected_identities=expected,
        fetch=_fake_fetch(objects),
    )
    target = Path(receipt["model"]["files"][0]["path"])
    target.write_bytes(b"evil")
    forged = json.loads(receipt_path.read_text())
    forged["model"]["files"][0].update({"bytes": 4, "sha256": _sha(b"evil")})
    unsigned = dict(forged)
    unsigned.pop("content_sha256")
    forged["content_sha256"] = canonical_sha256(unsigned)
    receipt_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="Authenticated model file"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=receipt_path,
            expected_identities=expected,
            fetch=lambda *_args: (_ for _ in ()).throw(AssertionError("network")),
        )


def test_wrong_model_identity_and_partial_cache_fail_before_snapshot_writes(
    tmp_path: Path,
) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    wrong = dict(expected)
    wrong["model_content_sha256"] = "0" * 64
    cache = tmp_path / "wrong-cache"
    with pytest.raises(ValueError, match="model does not match frozen"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "wrong-inputs",
            cache_dir=cache,
            receipt_path=tmp_path / "wrong-receipt.json",
            expected_identities=wrong,
            fetch=_fake_fetch(objects),
        )
    snapshot = (
        cache / "models--Qwen--Qwen3-VL-8B-Thinking" / "snapshots" / MODEL_REVISION
    )
    assert not snapshot.exists()

    partial_cache = tmp_path / "partial-cache"
    partial_snapshot = (
        partial_cache
        / "models--Qwen--Qwen3-VL-8B-Thinking"
        / "snapshots"
        / MODEL_REVISION
    )
    partial_snapshot.mkdir(parents=True)
    first_name = REQUIRED_SNAPSHOT_FILES[0]
    first_url = manifest["model"]["file_urls"][first_name]  # type: ignore[index]
    (partial_snapshot / first_name).write_bytes(objects[first_url])
    with pytest.raises(ValueError, match="partial"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "partial-inputs",
            cache_dir=partial_cache,
            receipt_path=tmp_path / "partial-receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )
    assert {path.name for path in partial_snapshot.iterdir()} == {first_name}


def test_empty_snapshot_and_unexpected_snapshot_directory_fail_closed(
    tmp_path: Path,
) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)

    def snapshot(cache: Path) -> Path:
        return (
            cache / "models--Qwen--Qwen3-VL-8B-Thinking" / "snapshots" / MODEL_REVISION
        )

    empty_cache = tmp_path / "empty-cache"
    snapshot(empty_cache).mkdir(parents=True)
    with pytest.raises(ValueError, match="partial"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "empty-inputs",
            cache_dir=empty_cache,
            receipt_path=tmp_path / "empty-receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )

    extra_cache = tmp_path / "extra-cache"
    extra_snapshot = snapshot(extra_cache)
    extra_snapshot.mkdir(parents=True)
    for name, url in manifest["model"]["file_urls"].items():  # type: ignore[index,union-attr]
        (extra_snapshot / name).write_bytes(objects[url])
    (extra_snapshot / "unexpected").mkdir()
    with pytest.raises(ValueError, match="unexpected"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "extra-inputs",
            cache_dir=extra_cache,
            receipt_path=tmp_path / "extra-receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )


def test_model_cache_rejects_symlinked_root(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    real_cache = tmp_path / "real-cache"
    real_cache.mkdir()
    linked_cache = tmp_path / "linked-cache"
    linked_cache.symlink_to(real_cache, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs-a",
            cache_dir=linked_cache,
            receipt_path=tmp_path / "receipt-a.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )


def test_output_ancestors_and_model_snapshot_leaf_reject_symlinks(
    tmp_path: Path,
) -> None:
    manifest, objects = _fixture(tmp_path)
    expected = _expected_identities(manifest, objects)
    real_inputs = tmp_path / "real-inputs"
    real_inputs.mkdir()
    linked_inputs = tmp_path / "linked-inputs"
    linked_inputs.symlink_to(real_inputs, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        hydrate_all(
            manifest,
            inputs_dir=linked_inputs,
            cache_dir=tmp_path / "cache-a",
            receipt_path=tmp_path / "receipt-a.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )
    assert not any(real_inputs.iterdir())

    real_receipts = tmp_path / "real-receipts"
    real_receipts.mkdir()
    linked_receipts = tmp_path / "linked-receipts"
    linked_receipts.symlink_to(real_receipts, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs-b",
            cache_dir=tmp_path / "cache-b",
            receipt_path=linked_receipts / "receipt.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )
    assert not any(real_receipts.iterdir())

    cache = tmp_path / "cache"
    snapshot = (
        cache / "models--Qwen--Qwen3-VL-8B-Thinking" / "snapshots" / MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (snapshot / REQUIRED_SNAPSHOT_FILES[0]).symlink_to(outside)
    with pytest.raises(ValueError, match="partial|regular file"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs-b",
            cache_dir=cache,
            receipt_path=tmp_path / "receipt-b.json",
            expected_identities=expected,
            fetch=_fake_fetch(objects),
        )


def test_plan_and_probe_cross_identity_mismatches_are_rejected(tmp_path: Path) -> None:
    plan_path = tmp_path / "manifest.json"
    plan_path.write_text(
        json.dumps(
            {
                "format": "tinylora_step5_plan_v1",
                "large_run_enabled": False,
                "model": {
                    "attention": "flash_attention_2",
                    "model_id": "Qwen/Qwen3-VL-8B-Thinking",
                    "revision": "0" * 40,
                    "vision_weights_frozen": True,
                },
                "outputs": {},
                "paid_execution_enabled": False,
            }
        )
    )
    with pytest.raises(ValueError, match="approved model"):
        _validate_plan(plan_path)

    qualification_body = {
        "format": "intelligent_liars_step5_probe_qualification_v1",
        "qualification": {"step5_plan_manifest_sha256": "1" * 64},
    }
    receipt = hashlib.sha256(
        json.dumps(qualification_body, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text(
        json.dumps({**qualification_body, "qualification_receipt_sha256": receipt})
    )
    with pytest.raises(ValueError, match="different Step 5 plan"):
        _qualification_receipt(qualification_path, expected_plan_sha256="2" * 64)


@pytest.mark.parametrize(
    "url",
    [
        "http://objects.example/file",
        "file:///tmp/object",
        "https://user:password@example.com/file",
    ],
)
def test_url_contract_rejects_non_https_and_authority_credentials(url: str) -> None:
    with pytest.raises(ValueError, match="credentialless HTTPS"):
        https_origin(url)


def test_url_manifest_rejects_extra_fields_and_missing_file_map() -> None:
    payload = {
        "format": "tinylora_step5_input_url_manifest_v2",
        "controller": {
            "account_id": "123456789012",
            "bucket": "test-bucket",
            "created_at": "2026-08-22T00:00:00Z",
            "expires_at": "2026-08-22T01:00:00Z",
            "expiry_seconds": 3600,
            "manifest_key": "controller/input-manifest.json",
            "region": "us-east-1",
        },
        "model": {
            "completion_url": "https://x/c",
            "manifest_url": "https://x/m",
            "file_urls": {},
            "aws_access_key": "bad",
        },
        "frozen_inputs": {
            "completion_url": "https://x/c",
            "archive_url": "https://x/a",
        },
        "pixmo": {
            "completion_url": "https://x/c",
            "manifest_url": "https://x/m",
            "archive_url": "https://x/a",
        },
    }
    with pytest.raises(ValueError, match="invalid model fields"):
        validate_url_manifest(payload)


def test_hydrator_cli_takes_url_from_environment_not_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = importlib.util.spec_from_file_location("step5_hydrator_cli", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(
        "sys.argv",
        [str(SCRIPT), "--receipt", "/tmp/receipt.json", *_identity_argv()],
    )

    args = module.parse_args()

    assert args.url_manifest_url_env == "CANARY_INPUT_URL_MANIFEST_URL"
    assert args.url_manifest_url_file is None
    assert not hasattr(args, "url_manifest_url")


def test_hydrator_reads_signed_url_from_private_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    specification = importlib.util.spec_from_file_location(
        "step5_hydrator_file_cli", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    secret = tmp_path / "url"
    secret.write_text("https://example.test/private-manifest\n")
    secret.chmod(0o600)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--url-manifest-url-file",
            str(secret),
            "--receipt",
            str(tmp_path / "receipt.json"),
            *_identity_argv(),
        ],
    )
    monkeypatch.setattr(module, "fetch_https", lambda _url, path: path.write_text("{}"))
    monkeypatch.setattr(module, "hydrate_all", lambda *_args, **_kwargs: {"ok": True})
    assert module.main() == 0
    assert "private-manifest" not in capsys.readouterr().out


def test_hydrator_rejects_world_readable_url_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification = importlib.util.spec_from_file_location(
        "step5_hydrator_bad_mode", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    secret = tmp_path / "url"
    secret.write_text("https://example.test/private\n")
    secret.chmod(0o644)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--url-manifest-url-file",
            str(secret),
            "--receipt",
            str(tmp_path / "receipt.json"),
            *_identity_argv(),
        ],
    )
    with pytest.raises(ValueError, match="0600"):
        module.main()


def test_hydrator_rejects_symlinked_file_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification = importlib.util.spec_from_file_location(
        "step5_hydrator_symlink", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    real = tmp_path / "real"
    real.mkdir()
    secret = real / "url"
    secret.write_text("https://example.test/private\n")
    secret.chmod(0o600)
    direct = tmp_path / "direct"
    direct.symlink_to(secret)
    with pytest.raises(ValueError, match="symlinks"):
        module.read_private_url(direct)
    parent = tmp_path / "parent"
    parent.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        module.read_private_url(parent / "url")


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.CHRTYPE])
def test_frozen_archive_rejects_links_devices_and_traversal(
    tmp_path: Path, member_type: bytes
) -> None:
    archive = tmp_path / "malicious.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo(
            "../escape"
            if member_type == tarfile.SYMTYPE
            else "corpora/tinylora_deception_action_v1/step5_v1/device"
        )
        member.type = member_type
        if member_type == tarfile.SYMTYPE:
            member.linkname = "/etc/passwd"
        output.addfile(member)
    completion = {
        "archive_bytes": archive.stat().st_size,
        "archive_name": archive.name,
        "archive_sha256": _sha(archive.read_bytes()),
        "format": "tinylora_step5_frozen_inputs_s3_completion_v1",
        "plan_sha256": "0" * 64,
        "probe_qualification_receipt_sha256": "0" * 64,
    }
    objects = {
        "https://x/complete": (json.dumps(completion) + "\n").encode(),
        "https://x/archive": archive.read_bytes(),
    }
    with pytest.raises(
        ValueError, match="Unsafe archive member|link or special device"
    ):
        hydrate_frozen_inputs(
            {
                "completion_url": "https://x/complete",
                "archive_url": "https://x/archive",
            },
            inputs_dir=tmp_path / "inputs",
            temporary_dir=tmp_path / "temporary",
            fetch=_fake_fetch(objects),
        )
