from __future__ import annotations

import hashlib
import importlib.util
import io
import json
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
    relative = Path("data/tinylora_preservation_snapshots/v1/pixmo_docs_images") / f"{digest}.jpg"
    (source / relative).parent.mkdir(parents=True)
    (source / relative).write_bytes(image)
    corpus = source / "corpora" / "preservation_train.jsonl"
    corpus.parent.mkdir(parents=True)
    row = {
        "record_id": "pixmo.1",
        "preservation_category": "vision_charts",
        "image_sha256": digest,
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": relative.as_posix()}, {"type": "text", "text": "read"}]},
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
        "outputs": {"preservation_train": {"path": "preservation_train.jsonl", "records": 1, "sha256": _sha(row_bytes)}},
        "paid_execution_enabled": False,
    }
    plan_bytes = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    qualification_body = {
        "ensembles": {},
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
            {**qualification_body, "qualification_receipt_sha256": qualification_receipt}
        )
        + "\n"
    ).encode()
    members = {
        "corpora/tinylora_deception_action_v1/step5_v1/manifest.json": plan_bytes,
        "corpora/tinylora_deception_action_v1/step5_v1/preservation_train.jsonl": row_bytes,
        "artifacts/probes/step5_grouped_ensemble_v1/probe_qualification.json": qualification,
        "artifacts/probes/step5_grouped_ensemble_v1/probes/legacy-grouped-regularizer.json": b"{}\n",
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
        model_files.append({"bytes": len(payload), "path": name, "sha256": _sha(payload)})
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
        "format": "tinylora_step5_input_url_manifest_v1",
        "model": {"completion_url": urls["model_complete"], "manifest_url": urls["model_manifest"], "file_urls": model_urls},
        "frozen_inputs": {"completion_url": urls["inputs_complete"], "archive_url": urls["inputs_archive"]},
        "pixmo": {"completion_url": urls["pixmo_complete"], "manifest_url": urls["pixmo_manifest"], "archive_url": urls["pixmo_archive"]},
    }
    return manifest, objects


def test_full_hydration_verifies_inputs_builds_hf_snapshot_and_redacts_urls(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    receipt_path = tmp_path / "artifacts" / "receipt.json"
    receipt = hydrate_all(
        manifest,
        inputs_dir=tmp_path / "inputs",
        cache_dir=tmp_path / "cache",
        receipt_path=receipt_path,
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
    assert "signature=" not in serialized and "token=" not in serialized and "secret-" not in serialized
    without_commitment = dict(receipt)
    commitment = without_commitment.pop("content_sha256")
    assert commitment == canonical_sha256(without_commitment)


def test_model_tamper_is_rejected(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    first_url = next(iter(manifest["model"]["file_urls"].values()))  # type: ignore[index,union-attr]
    objects[first_url] = b"tampered"
    with pytest.raises(ValueError, match="Model file hash/size mismatch"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            fetch=_fake_fetch(objects),
        )


def test_frozen_archive_tamper_is_rejected_before_extraction(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    archive_url = manifest["frozen_inputs"]["archive_url"]  # type: ignore[index]
    objects[archive_url] += b"tampered"
    with pytest.raises(ValueError, match="Frozen-input archive hash/size mismatch"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            fetch=_fake_fetch(objects),
        )


def test_pixmo_manifest_tamper_is_rejected_before_extraction(tmp_path: Path) -> None:
    manifest, objects = _fixture(tmp_path)
    manifest_url = manifest["pixmo"]["manifest_url"]  # type: ignore[index]
    objects[manifest_url] += b" "
    with pytest.raises(ValueError, match="does not bind the manifest"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            fetch=_fake_fetch(objects),
        )
    assert not (tmp_path / "inputs" / "step5_v1").exists()
    assert not (tmp_path / "inputs" / "probes").exists()


def test_receipt_write_failure_rolls_back_promoted_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, objects = _fixture(tmp_path)
    original = Path.write_text

    def fail_receipt(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == "receipt.json.tmp":
            raise OSError("simulated receipt storage failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_receipt)
    with pytest.raises(OSError, match="receipt storage failure"):
        hydrate_all(
            manifest,
            inputs_dir=tmp_path / "inputs",
            cache_dir=tmp_path / "cache",
            receipt_path=tmp_path / "receipt.json",
            fetch=_fake_fetch(objects),
        )
    assert not (tmp_path / "inputs" / "step5_v1").exists()
    assert not (tmp_path / "inputs" / "probes").exists()
    assert not (tmp_path / "inputs" / "pixmo").exists()


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
    ["http://objects.example/file", "file:///tmp/object", "https://user:password@example.com/file"],
)
def test_url_contract_rejects_non_https_and_authority_credentials(url: str) -> None:
    with pytest.raises(ValueError, match="credentialless HTTPS"):
        https_origin(url)


def test_url_manifest_rejects_extra_fields_and_missing_file_map() -> None:
    payload = {
        "format": "tinylora_step5_input_url_manifest_v1",
        "model": {"completion_url": "https://x/c", "manifest_url": "https://x/m", "file_urls": {}, "aws_access_key": "bad"},
        "frozen_inputs": {"completion_url": "https://x/c", "archive_url": "https://x/a"},
        "pixmo": {"completion_url": "https://x/c", "manifest_url": "https://x/m", "archive_url": "https://x/a"},
    }
    with pytest.raises(ValueError, match="invalid model fields"):
        validate_url_manifest(payload)


def test_hydrator_cli_takes_url_from_environment_not_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    specification = importlib.util.spec_from_file_location("step5_hydrator_cli", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--receipt", "/tmp/receipt.json"])

    args = module.parse_args()

    assert args.url_manifest_url_env == "CANARY_INPUT_URL_MANIFEST_URL"
    assert not hasattr(args, "url_manifest_url")


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.CHRTYPE])
def test_frozen_archive_rejects_links_devices_and_traversal(tmp_path: Path, member_type: bytes) -> None:
    archive = tmp_path / "malicious.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("../escape" if member_type == tarfile.SYMTYPE else "corpora/tinylora_deception_action_v1/step5_v1/device")
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
    with pytest.raises(ValueError, match="Unsafe archive member|link or special device"):
        hydrate_frozen_inputs(
            {"completion_url": "https://x/complete", "archive_url": "https://x/archive"},
            inputs_dir=tmp_path / "inputs",
            temporary_dir=tmp_path / "temporary",
            fetch=_fake_fetch(objects),
        )
