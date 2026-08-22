from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from intelligent_liars.step5_multimodal_assets import (
    build_asset_manifest,
    create_deterministic_tar,
    rebase_image_references,
    stage_multimodal_bundle,
    validate_staged_bundle,
)

PIXMO_RELATIVE = Path("data/tinylora_preservation_snapshots/v1/pixmo_docs_images")


def _image_bytes(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (7, 5), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_fixture(root: Path) -> tuple[list[Path], dict[str, bytes]]:
    image_dir = root / "data" / "tinylora_preservation_snapshots" / "v1" / "pixmo_docs_images"
    image_dir.mkdir(parents=True)
    images: dict[str, bytes] = {}
    rows_by_file: list[list[dict[str, object]]] = [[], []]
    for index, (category, colour) in enumerate(
        (("vision_charts", (255, 0, 0)), ("vision_tables", (0, 255, 0)))
    ):
        payload = _image_bytes(colour)
        digest = hashlib.sha256(payload).hexdigest()
        relative = f"data/tinylora_preservation_snapshots/v1/pixmo_docs_images/{digest}.jpg"
        (root / relative).write_bytes(payload)
        images[digest] = payload
        rows_by_file[index].append(
            {
                "record_id": f"pixmo.{index}",
                "preservation_category": category,
                "image_sha256": digest,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": relative},
                            {"type": "text", "text": "question"},
                        ],
                    },
                    {"role": "assistant", "content": "answer"},
                ],
            }
        )
    corpus_dir = root / "corpora"
    corpus_dir.mkdir()
    paths = []
    for name, rows in zip(("preservation_train.jsonl", "preservation_development_vision.jsonl"), rows_by_file, strict=True):
        path = corpus_dir / name
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        paths.append(path)
    return paths, images


def test_manifest_is_content_addressed_and_has_category_smoke_inventory(tmp_path: Path) -> None:
    corpora, images = _write_fixture(tmp_path)

    manifest = build_asset_manifest(corpora, project_root=tmp_path)

    assert manifest["format"] == "tinylora_step5_multimodal_assets_v1"
    assert len(manifest["content_sha256"]) == 64
    assert manifest["totals"] == {"bytes": sum(map(len, images.values())), "image_references": 2, "unique_images": 2}
    assert manifest["categories"]["vision_charts"]["records"] == 1
    assert manifest["categories"]["vision_tables"]["unique_images"] == 1
    assert sorted(manifest["smoke_selection"]) == ["vision_charts", "vision_tables"]
    assert [item["sha256"] for item in manifest["assets"]] == sorted(images)


def test_build_rejects_traversal_symlink_hash_mismatch_and_invalid_decode(tmp_path: Path) -> None:
    corpora, _images = _write_fixture(tmp_path)
    row = json.loads(corpora[0].read_text())

    row["messages"][0]["content"][0]["image"] = "../outside.jpg"
    corpora[0].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="relative path|traversal"):
        build_asset_manifest(corpora, project_root=tmp_path)

    corpora, _images = _write_fixture(tmp_path / "hash")
    row = json.loads(corpora[0].read_text())
    row["image_sha256"] = "0" * 64
    corpora[0].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash"):
        build_asset_manifest(corpora, project_root=tmp_path / "hash")

    corpora, _images = _write_fixture(tmp_path / "decode")
    row = json.loads(corpora[0].read_text())
    image = tmp_path / "decode" / row["messages"][0]["content"][0]["image"]
    bad = b"not an image"
    image.write_bytes(bad)
    row["image_sha256"] = hashlib.sha256(bad).hexdigest()
    row["messages"][0]["content"][0]["image"] = str(image.relative_to(tmp_path / "decode")).replace(image.stem, row["image_sha256"])
    renamed = image.with_name(f"{row['image_sha256']}.jpg")
    image.rename(renamed)
    corpora[0].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="decode"):
        build_asset_manifest(corpora, project_root=tmp_path / "decode")

    corpora, _images = _write_fixture(tmp_path / "link")
    image_dir = tmp_path / "link" / PIXMO_RELATIVE
    real_dir = image_dir.with_name("real_pixmo_docs_images")
    image_dir.rename(real_dir)
    image_dir.symlink_to(real_dir.name)
    with pytest.raises(ValueError, match="symlink"):
        build_asset_manifest(corpora, project_root=tmp_path / "link")


def test_staging_rebase_and_archive_are_deterministic(tmp_path: Path) -> None:
    corpora, _images = _write_fixture(tmp_path / "source")
    first = tmp_path / "bundle-a"
    second = tmp_path / "bundle-b"
    stage_multimodal_bundle(corpora, project_root=tmp_path / "source", destination=first)
    stage_multimodal_bundle(corpora, project_root=tmp_path / "source", destination=second)

    first_manifest = validate_staged_bundle(first)
    second_manifest = validate_staged_bundle(second)
    assert first_manifest["content_sha256"] == second_manifest["content_sha256"]

    source_rows = [json.loads(corpora[0].read_text())]
    rebased = rebase_image_references(source_rows, bundle_root=first, manifest=first_manifest)
    image_ref = rebased[0]["messages"][0]["content"][0]["image"]
    assert Path(image_ref).is_absolute()
    assert Path(image_ref).is_file()

    archive_a = tmp_path / "a.tar"
    archive_b = tmp_path / "b.tar"
    create_deterministic_tar(first, archive_a)
    create_deterministic_tar(second, archive_b)
    assert hashlib.sha256(archive_a.read_bytes()).digest() == hashlib.sha256(archive_b.read_bytes()).digest()
    with tarfile.open(archive_a) as archive:
        assert archive.getnames()[0] == "manifest.json"
        assert all(member.uid == member.gid == 0 and member.mtime == 0 for member in archive.getmembers())


def test_stage_refuses_to_overwrite_an_existing_destination(tmp_path: Path) -> None:
    corpora, _images = _write_fixture(tmp_path / "source")
    destination = tmp_path / "bundle"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        stage_multimodal_bundle(corpora, project_root=tmp_path / "source", destination=destination)
