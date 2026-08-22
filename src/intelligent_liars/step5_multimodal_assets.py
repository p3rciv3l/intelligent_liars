"""Portable, content-addressed assets for the Step 5 PixMo evaluation rows."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError


FORMAT = "tinylora_step5_multimodal_assets_v1"
PIXMO_IMAGE_PREFIX = PurePosixPath(
    "data/tinylora_preservation_snapshots/v1/pixmo_docs_images"
)
_FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _checked_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a traversal-free relative path: {value!r}")
    return path


def _checked_local_path(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    root = root.resolve(strict=True)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink: {relative}")
    try:
        current.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{label} is missing or escapes its root: {relative}") from error
    if not current.is_file():
        raise ValueError(f"{label} is not a regular file: {relative}")
    return current


def _relative_to_root(path: Path, root: Path, *, label: str) -> PurePosixPath:
    root = root.resolve(strict=True)
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {path}")
    try:
        relative = path.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside project root: {path}") from error
    return _checked_relative_path(relative.as_posix(), label=label)


def _decode_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image_format = image.format
            frame_count = getattr(image, "n_frames", 1)
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"Image decode failed for {path}") from error
    if image_format not in _FORMAT_EXTENSIONS or width < 1 or height < 1 or frame_count != 1:
        raise ValueError(f"Unsupported or invalid image decode for {path}")
    return {"format": image_format, "width": width, "height": height}


def _image_contents(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for message in row.get("messages", []):
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, list):
            images.extend(
                item
                for item in content
                if isinstance(item, dict) and item.get("type") == "image"
            )
    return images


def _rewrite_row(row: Mapping[str, Any], asset_paths: Mapping[str, str]) -> dict[str, Any]:
    rewritten: dict[str, Any] = copy.deepcopy(dict(row))
    images = _image_contents(rewritten)
    if not images:
        return rewritten
    expected = str(rewritten.get("image_sha256", ""))
    if expected not in asset_paths:
        raise ValueError(f"No bundled asset for record {rewritten.get('record_id')}")
    for content in images:
        content["image"] = asset_paths[expected]
    return rewritten


def build_asset_manifest(
    corpus_paths: Sequence[Path],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Validate every referenced PixMo image and return a canonical inventory."""

    if not corpus_paths:
        raise ValueError("At least one corpus is required")
    root = project_root.resolve(strict=True)
    if project_root.is_symlink():
        raise ValueError("Project root may not be a symlink")
    corpus_names: set[str] = set()
    asset_records: dict[str, dict[str, Any]] = {}
    category_records: dict[str, list[tuple[str, str]]] = defaultdict(list)
    corpus_rows: list[tuple[PurePosixPath, list[dict[str, Any]], str]] = []
    seen_records: set[str] = set()
    image_references = 0

    for corpus_path in sorted(corpus_paths, key=lambda path: path.as_posix()):
        relative_corpus = _relative_to_root(corpus_path, root, label="Corpus")
        checked_corpus = _checked_local_path(root, relative_corpus, label="Corpus")
        if checked_corpus.name in corpus_names:
            raise ValueError(f"Corpus basenames must be unique: {checked_corpus.name}")
        corpus_names.add(checked_corpus.name)
        rows = [json.loads(line) for line in checked_corpus.read_text().splitlines() if line.strip()]
        corpus_rows.append((relative_corpus, rows, _sha256_file(checked_corpus)))
        for row in rows:
            images = _image_contents(row)
            if not images:
                continue
            if len(images) != 1:
                raise ValueError(f"PixMo row must have exactly one image: {row.get('record_id')}")
            record_id = str(row.get("record_id", ""))
            if not record_id or record_id in seen_records:
                raise ValueError(f"Missing or duplicate image record id: {record_id!r}")
            seen_records.add(record_id)
            expected = str(row.get("image_sha256", ""))
            if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
                raise ValueError(f"Invalid expected image hash for {record_id}")
            source_reference = _checked_relative_path(str(images[0].get("image", "")), label="Image")
            if source_reference.parent != PIXMO_IMAGE_PREFIX:
                raise ValueError(f"Image is not a canonical PixMo snapshot reference: {source_reference}")
            if Path(source_reference.name).stem != expected:
                raise ValueError(f"Image filename/hash mismatch for {record_id}")
            source_path = _checked_local_path(root, source_reference, label="Image")
            actual = _sha256_file(source_path)
            if actual != expected:
                raise ValueError(f"Image content hash mismatch for {record_id}")
            decoded = _decode_image(source_path)
            extension = _FORMAT_EXTENSIONS[decoded["format"]]
            bundle_path = f"images/{expected[:2]}/{expected}{extension}"
            category = str(row.get("preservation_category", ""))
            if not category.startswith("vision_"):
                raise ValueError(f"Image row lacks a vision preservation category: {record_id}")
            image_references += 1
            category_records[category].append((record_id, expected))
            asset = asset_records.setdefault(
                expected,
                {
                    "bundle_path": bundle_path,
                    "bytes": source_path.stat().st_size,
                    **decoded,
                    "record_ids": [],
                    "categories": [],
                    "sha256": expected,
                    "source_references": [],
                },
            )
            if any(asset[key] != decoded[key] for key in ("format", "width", "height")):
                raise ValueError(f"Conflicting decode metadata for image {expected}")
            asset["record_ids"].append(record_id)
            asset["categories"].append(category)
            asset["source_references"].append(source_reference.as_posix())

    if not asset_records:
        raise ValueError("No PixMo image references found")
    for asset in asset_records.values():
        for key in ("record_ids", "categories", "source_references"):
            asset[key] = sorted(set(asset[key]))
    asset_paths = {digest: item["bundle_path"] for digest, item in asset_records.items()}
    source_corpora: list[dict[str, Any]] = []
    bundled_corpora: list[dict[str, Any]] = []
    rewritten_by_record: dict[str, dict[str, Any]] = {}
    for relative, rows, source_hash in corpus_rows:
        rewritten = [_rewrite_row(row, asset_paths) for row in rows]
        for row in rewritten:
            if _image_contents(row):
                rewritten_by_record[str(row["record_id"])] = row
        source_corpora.append(
            {"path": relative.as_posix(), "records": len(rows), "sha256": source_hash}
        )
        output_path = f"corpora/{relative.name}"
        bundled_corpora.append(
            {
                "path": output_path,
                "records": len(rewritten),
                "sha256": _sha256_bytes(_jsonl_bytes(rewritten)),
            }
        )
    categories: dict[str, dict[str, int]] = {}
    smoke_selection: dict[str, dict[str, str]] = {}
    smoke_rows: list[dict[str, Any]] = []
    for category, records in sorted(category_records.items()):
        unique_hashes = {digest for _record_id, digest in records}
        categories[category] = {
            "bytes": sum(asset_records[digest]["bytes"] for digest in unique_hashes),
            "records": len(records),
            "unique_images": len(unique_hashes),
        }
        record_id, digest = min(records)
        smoke_selection[category] = {
            "bundle_path": asset_paths[digest],
            "image_sha256": digest,
            "record_id": record_id,
        }
        smoke_rows.append(rewritten_by_record[record_id])
    smoke_bytes = _jsonl_bytes(smoke_rows)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "assets": sorted(asset_records.values(), key=lambda item: item["sha256"]),
        "bundled_corpora": bundled_corpora,
        "categories": categories,
        "smoke_corpus": {
            "path": "smoke/one_per_category.jsonl",
            "records": len(smoke_rows),
            "sha256": _sha256_bytes(smoke_bytes),
        },
        "smoke_selection": smoke_selection,
        "source_corpora": source_corpora,
        "totals": {
            "bytes": sum(item["bytes"] for item in asset_records.values()),
            "image_references": image_references,
            "unique_images": len(asset_records),
        },
    }
    manifest["content_sha256"] = _sha256_bytes(_json_bytes(manifest))
    return manifest


def _manifest_asset_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported multimodal asset manifest")
    return {str(asset["sha256"]): str(asset["bundle_path"]) for asset in manifest["assets"]}


def rebase_image_references(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return copies whose image references are verified absolute bundle paths."""

    root = bundle_root.resolve(strict=True)
    asset_paths = _manifest_asset_map(manifest)
    absolute_paths: dict[str, str] = {}
    for digest, relative_value in asset_paths.items():
        relative = _checked_relative_path(relative_value, label="Bundled image")
        path = _checked_local_path(root, relative, label="Bundled image")
        if _sha256_file(path) != digest:
            raise ValueError(f"Bundled image hash mismatch: {relative}")
        absolute_paths[digest] = str(path)
    return [_rewrite_row(row, absolute_paths) for row in rows]


def stage_multimodal_bundle(
    corpus_paths: Sequence[Path],
    *,
    project_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Atomically stage validated images, portable corpora, and a manifest."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_asset_manifest(corpus_paths, project_root=project_root)
    asset_paths = _manifest_asset_map(manifest)
    root = project_root.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for asset in manifest["assets"]:
            source = _checked_local_path(
                root,
                _checked_relative_path(asset["source_references"][0], label="Image"),
                label="Image",
            )
            target = temporary / asset["bundle_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        rewritten_by_record: dict[str, dict[str, Any]] = {}
        for corpus_path, specification in zip(
            sorted(corpus_paths, key=lambda path: path.as_posix()),
            manifest["bundled_corpora"],
            strict=True,
        ):
            rows = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
            rewritten = [_rewrite_row(row, asset_paths) for row in rows]
            for row in rewritten:
                if _image_contents(row):
                    rewritten_by_record[str(row["record_id"])] = row
            target = temporary / specification["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_jsonl_bytes(rewritten))
        smoke_rows = [
            rewritten_by_record[item["record_id"]]
            for _category, item in sorted(manifest["smoke_selection"].items())
        ]
        smoke_path = temporary / manifest["smoke_corpus"]["path"]
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_bytes(_jsonl_bytes(smoke_rows))
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        validate_staged_bundle(temporary)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def validate_staged_bundle(bundle_root: Path) -> dict[str, Any]:
    """Fail closed on unexpected, missing, linked, corrupt, or undecodable members."""

    root = bundle_root.resolve(strict=True)
    manifest_path = _checked_local_path(root, PurePosixPath("manifest.json"), label="Manifest")
    manifest = json.loads(manifest_path.read_text())
    claimed = str(manifest.pop("content_sha256", ""))
    actual = _sha256_bytes(_json_bytes(manifest))
    manifest["content_sha256"] = claimed
    if claimed != actual:
        raise ValueError("Multimodal manifest content hash mismatch")
    expected_files = {"manifest.json"}
    for asset in manifest["assets"]:
        relative = _checked_relative_path(str(asset["bundle_path"]), label="Bundled image")
        path = _checked_local_path(root, relative, label="Bundled image")
        expected_files.add(relative.as_posix())
        if _sha256_file(path) != asset["sha256"] or path.stat().st_size != asset["bytes"]:
            raise ValueError(f"Bundled image hash/size mismatch: {relative}")
        decoded = _decode_image(path)
        if any(decoded[key] != asset[key] for key in ("format", "width", "height")):
            raise ValueError(f"Bundled image decode metadata mismatch: {relative}")
    for specification in [*manifest["bundled_corpora"], manifest["smoke_corpus"]]:
        relative = _checked_relative_path(str(specification["path"]), label="Bundled corpus")
        path = _checked_local_path(root, relative, label="Bundled corpus")
        expected_files.add(relative.as_posix())
        if _sha256_file(path) != specification["sha256"]:
            raise ValueError(f"Bundled corpus hash mismatch: {relative}")
        rows = [line for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != specification["records"]:
            raise ValueError(f"Bundled corpus record count mismatch: {relative}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError(
            f"Bundle member inventory mismatch; missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    return manifest


def create_deterministic_tar(bundle_root: Path, archive_path: Path) -> str:
    """Write a byte-reproducible uncompressed archive and return its SHA-256."""

    if archive_path.exists() or archive_path.is_symlink():
        raise FileExistsError(archive_path)
    validate_staged_bundle(bundle_root)
    root = bundle_root.resolve(strict=True)
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: (path.name != "manifest.json", path.relative_to(root).as_posix()),
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                info = tarfile.TarInfo(relative)
                info.size = path.stat().st_size
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        os.rename(temporary, archive_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_file(archive_path)
