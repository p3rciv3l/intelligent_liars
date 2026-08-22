"""Credentialless, fail-closed hydration of immutable Step 5 inputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from intelligent_liars.model_cache import (
    MODEL_REVISION,
    REQUIRED_SNAPSHOT_FILES,
    completion_marker,
)
from intelligent_liars.step5_multimodal_assets import validate_staged_bundle


FORMAT = "tinylora_step5_input_url_manifest_v1"
RECEIPT_FORMAT = "tinylora_step5_input_hydration_receipt_v1"
MODEL_FILES = 14
MODEL_REPO = "Qwen/Qwen3-VL-8B-Thinking"
Fetch = Callable[[str, Path], None]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def https_origin(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Every hydration URL must be credentialless HTTPS")
    return urllib.parse.urlunsplit(("https", parsed.netloc, "", "", ""))


def fetch_https(url: str, destination: Path) -> None:
    """Stream one public or presigned HTTPS GET without ambient credentials."""

    https_origin(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "step5-hydrator/1"})
    opener = urllib.request.build_opener(_RejectRedirects)
    try:
        with opener.open(request, timeout=120) as response, temporary.open("xb") as output:
            if response.status != 200:
                raise ValueError(f"Hydration GET returned HTTP {response.status}")
            shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_download(url: str, directory: Path, name: str, fetch: Fetch) -> tuple[dict[str, Any], str]:
    path = directory / name
    fetch(url, path)
    try:
        value = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Downloaded {name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"Downloaded {name} must be a JSON object")
    return value, file_sha256(path)


def _relative_file(value: str, *, label: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError(f"{label} must be a POSIX relative file path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return path


def _regular_path(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError(f"{label} root may not be a symlink")
    resolved_root = root.resolve(strict=True)
    current = resolved_root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink")
        current.mkdir(exist_ok=True)
    target = current / relative.name
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError(f"{label} must be a regular file")
    return target


def _verify_file(path: Path, specification: Mapping[str, Any], *, label: str) -> None:
    expected_hash = str(specification.get("sha256", ""))
    expected_bytes = specification.get("bytes")
    if len(expected_hash) != 64 or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ValueError(f"Invalid {label} hash/size specification")
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_hash:
        raise ValueError(f"{label} hash/size mismatch: {path.name}")


def hydrate_model(
    specification: Mapping[str, Any],
    *,
    cache_dir: Path,
    temporary_dir: Path,
    fetch: Fetch,
) -> dict[str, Any]:
    complete, complete_hash = _json_download(str(specification["completion_url"]), temporary_dir, "model.complete.json", fetch)
    manifest, manifest_hash = _json_download(str(specification["manifest_url"]), temporary_dir, "model.manifest.json", fetch)
    expected_complete = completion_marker(manifest)
    if complete != expected_complete:
        raise ValueError("Model completion marker or manifest contract is invalid")
    if manifest.get("format") != "tinylora_model_cache_manifest_v1" or manifest.get("complete") is not True:
        raise ValueError("Unsupported or incomplete model manifest")
    if expected_complete["manifest_sha256"] != manifest_hash:
        raise ValueError("Model completion marker does not bind the downloaded manifest")
    if complete.get("content_sha256") != manifest.get("content_sha256"):
        raise ValueError("Model content identity mismatch")
    if complete.get("model") != manifest.get("model") or manifest.get("model") != {"repo_id": MODEL_REPO, "revision": MODEL_REVISION}:
        raise ValueError("Model identity mismatch")
    files = manifest.get("files")
    urls = specification.get("file_urls")
    if not isinstance(files, list) or len(files) != MODEL_FILES or not isinstance(urls, Mapping):
        raise ValueError("Model manifest must contain exactly 14 files and matching URLs")
    paths = [str(item.get("path", "")) for item in files if isinstance(item, Mapping)]
    if paths != list(REQUIRED_SNAPSHOT_FILES) or set(urls) != set(paths):
        raise ValueError("Model file URL inventory does not exactly match the manifest")
    total_bytes = sum(item.get("bytes", -1) for item in files)
    if total_bytes != complete.get("total_bytes") or total_bytes != manifest.get("total_bytes"):
        raise ValueError("Model total byte count mismatch")
    revision = str(manifest["model"]["revision"])
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Model revision must be an exact commit hash")
    snapshot = cache_dir / "models--Qwen--Qwen3-VL-8B-Thinking" / "snapshots" / revision
    inventory: list[dict[str, Any]] = []
    for item in files:
        relative = _relative_file(str(item["path"]), label="model file")
        if len(relative.parts) != 1:
            raise ValueError("Model files must live at the snapshot root")
        target = _regular_path(snapshot, relative, label="model cache")
        reused = False
        if target.exists():
            try:
                _verify_file(target, item, label="Model file")
                reused = True
            except ValueError:
                target.unlink()
        if not reused:
            fetch(str(urls[relative.as_posix()]), target)
            _verify_file(target, item, label="Model file")
        inventory.append({"bytes": item["bytes"], "path": str(target.resolve()), "sha256": item["sha256"], "reused": reused})
    refs = snapshot.parents[1] / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)
    return {
        "content_sha256": manifest["content_sha256"],
        "completion_sha256": complete_hash,
        "files": inventory,
        "manifest_sha256": manifest_hash,
        "model": manifest["model"],
        "snapshot_path": str(snapshot.resolve()),
        "total_bytes": total_bytes,
    }


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    seen: set[str] = set()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        relative = _relative_file(member.name, label="archive member")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"Duplicate archive member: {normalized}")
        seen.add(normalized)
        if not (member.isdir() or member.isreg()):
            raise ValueError(f"Archive contains a link or special device: {normalized}")
        members.append(member)
    return members


def _extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Cannot read archive member: {member.name}")
    with stream, target.open("xb") as output:
        shutil.copyfileobj(stream, output, length=1024 * 1024)


def _download_verified_archive(
    specification: Mapping[str, Any],
    complete: Mapping[str, Any],
    *,
    temporary_dir: Path,
    name: str,
    label: str,
    fetch: Fetch,
) -> Path:
    archive = temporary_dir / name
    fetch(str(specification["archive_url"]), archive)
    if archive.stat().st_size != complete.get("archive_bytes") or file_sha256(archive) != complete.get("archive_sha256"):
        raise ValueError(f"{label} archive hash/size mismatch")
    return archive


def _qualification_receipt(path: Path, *, expected_plan_sha256: str) -> str:
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Probe qualification must be a JSON object")
    if (
        manifest.get("qualification", {}).get("step5_plan_manifest_sha256")
        != expected_plan_sha256
    ):
        raise ValueError("Probe qualification is bound to a different Step 5 plan")
    claimed = manifest.pop("qualification_receipt_sha256", None)
    actual = hashlib.sha256(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if claimed != actual:
        raise ValueError("Probe qualification canonical receipt mismatch")
    return actual


def _validate_plan(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = json.loads(plan_path.read_text())
    if plan.get("format") != "tinylora_step5_plan_v1" or plan.get("large_run_enabled") or plan.get("paid_execution_enabled"):
        raise ValueError("Frozen plan is unsupported or execution-enabled")
    if plan.get("model") != {
        "attention": "flash_attention_2",
        "model_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "vision_weights_frozen": True,
    }:
        raise ValueError("Frozen plan does not bind the approved model contract")
    inventory = [{"path": str(plan_path.resolve()), "sha256": file_sha256(plan_path), "bytes": plan_path.stat().st_size}]
    for name, item in sorted(plan.get("outputs", {}).items()):
        relative = _relative_file(str(item.get("path", "")), label=f"plan output {name}")
        if len(relative.parts) != 1:
            raise ValueError("Plan output paths must be colocated with the manifest")
        path = plan_path.parent / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != item.get("sha256"):
            raise ValueError(f"Frozen plan output hash mismatch: {name}")
        rows = sum(1 for line in path.read_text().splitlines() if line.strip())
        if rows != item.get("records"):
            raise ValueError(f"Frozen plan output count mismatch: {name}")
        inventory.append({"path": str(path.resolve()), "sha256": item["sha256"], "bytes": path.stat().st_size})
    return plan, inventory


def hydrate_frozen_inputs(
    specification: Mapping[str, Any],
    *,
    inputs_dir: Path,
    temporary_dir: Path,
    fetch: Fetch,
) -> dict[str, Any]:
    complete, complete_hash = _json_download(str(specification["completion_url"]), temporary_dir, "inputs.complete.json", fetch)
    if complete.get("format") != "tinylora_step5_frozen_inputs_s3_completion_v1":
        raise ValueError("Unsupported frozen-input completion marker")
    archive = _download_verified_archive(
        specification,
        complete,
        temporary_dir=temporary_dir,
        name="frozen-inputs.tar",
        label="Frozen-input",
        fetch=fetch,
    )
    stage = Path(tempfile.mkdtemp(prefix="step5-inputs-", dir=temporary_dir))
    plan_root = stage / "step5_v1"
    probe_root = stage / "probes" / "step5_grouped_ensemble_v1"
    try:
        with tarfile.open(archive, mode="r:*") as source:
            for member in _safe_members(source):
                name = PurePosixPath(member.name)
                plan_prefix = PurePosixPath("corpora/tinylora_deception_action_v1/step5_v1")
                probe_prefix = PurePosixPath("artifacts/probes/step5_grouped_ensemble_v1")
                if name.is_relative_to(plan_prefix):
                    relative = name.relative_to(plan_prefix)
                    target_root = plan_root
                elif name.is_relative_to(probe_prefix):
                    relative = name.relative_to(probe_prefix)
                    target_root = probe_root
                else:
                    raise ValueError(f"Unexpected frozen-input archive member: {name}")
                if member.isdir():
                    continue
                if name.name.startswith("._"):
                    raise ValueError(f"Unexpected AppleDouble frozen-input member: {name}")
                target = _regular_path(target_root, relative, label="frozen inputs")
                _extract_member(source, member, target)
        plan_path = plan_root / "manifest.json"
        plan, inventory = _validate_plan(plan_path)
        if file_sha256(plan_path) != complete.get("plan_sha256"):
            raise ValueError("Frozen-input completion marker does not bind the plan")
        qualification = probe_root / "probe_qualification.json"
        if not qualification.is_file() or _qualification_receipt(
            qualification,
            expected_plan_sha256=complete["plan_sha256"],
        ) != complete.get("probe_qualification_receipt_sha256"):
            raise ValueError("Probe qualification receipt hash mismatch")
        for path in sorted(probe_root.rglob("*")):
            if path.is_file():
                inventory.append({"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size})
        final_plan = inputs_dir / "step5_v1"
        final_probe = inputs_dir / "probes" / "step5_grouped_ensemble_v1"
        if final_plan.exists() or final_probe.exists():
            raise FileExistsError("Frozen input destination already exists")
        final_plan.parent.mkdir(parents=True, exist_ok=True)
        final_probe.parent.mkdir(parents=True, exist_ok=True)
        os.rename(plan_root, final_plan)
        os.rename(probe_root, final_probe)
        for item in inventory:
            path = Path(item["path"])
            if path.is_relative_to(plan_root):
                item["path"] = str(final_plan / path.relative_to(plan_root))
            elif path.is_relative_to(probe_root):
                item["path"] = str(final_probe / path.relative_to(probe_root))
        return {
            "archive_sha256": complete["archive_sha256"],
            "completion_sha256": complete_hash,
            "files": inventory,
            "plan_path": str((final_plan / "manifest.json").resolve()),
            "plan_sha256": complete["plan_sha256"],
            "probe_path": str((final_probe / "probes" / "legacy-grouped-regularizer.json").resolve()),
            "probe_qualification_receipt_sha256": complete["probe_qualification_receipt_sha256"],
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def hydrate_pixmo(
    specification: Mapping[str, Any],
    *,
    inputs_dir: Path,
    temporary_dir: Path,
    fetch: Fetch,
) -> dict[str, Any]:
    complete, complete_hash = _json_download(str(specification["completion_url"]), temporary_dir, "pixmo.complete.json", fetch)
    manifest, manifest_hash = _json_download(str(specification["manifest_url"]), temporary_dir, "pixmo.manifest.json", fetch)
    if complete.get("format") != "tinylora_step5_multimodal_s3_completion_v1":
        raise ValueError("Unsupported PixMo completion marker")
    if manifest_hash != complete.get("manifest_sha256") or manifest.get("content_sha256") != complete.get("manifest_commitment"):
        raise ValueError("PixMo completion marker does not bind the manifest")
    archive = _download_verified_archive(
        specification,
        complete,
        temporary_dir=temporary_dir,
        name="pixmo.tar",
        label="PixMo",
        fetch=fetch,
    )
    stage = Path(tempfile.mkdtemp(prefix="pixmo-", dir=temporary_dir))
    final = inputs_dir / "pixmo"
    try:
        with tarfile.open(archive, mode="r:*") as source:
            for member in _safe_members(source):
                if member.isdir():
                    continue
                target = _regular_path(stage, _relative_file(member.name, label="PixMo member"), label="PixMo bundle")
                _extract_member(source, member, target)
        if file_sha256(stage / "manifest.json") != manifest_hash:
            raise ValueError("PixMo archive manifest differs from published manifest")
        validated = validate_staged_bundle(stage)
        if validated.get("content_sha256") != complete["manifest_commitment"]:
            raise ValueError("PixMo bundle commitment mismatch")
        if final.exists():
            raise FileExistsError("PixMo destination already exists")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, final)
        files = [
            {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(final.rglob("*"))
            if path.is_file()
        ]
        return {
            "archive_sha256": complete["archive_sha256"],
            "bundle_path": str(final.resolve()),
            "completion_sha256": complete_hash,
            "files": files,
            "manifest_commitment": complete["manifest_commitment"],
            "manifest_sha256": manifest_hash,
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def validate_url_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("format") != FORMAT or set(payload) != {"format", "model", "frozen_inputs", "pixmo"}:
        raise ValueError("Unsupported hydration URL manifest")
    expected = {
        "model": {"completion_url", "manifest_url", "file_urls"},
        "frozen_inputs": {"completion_url", "archive_url"},
        "pixmo": {"completion_url", "manifest_url", "archive_url"},
    }
    result = dict(payload)
    for group, keys in expected.items():
        item = payload.get(group)
        if not isinstance(item, Mapping) or set(item) != keys:
            raise ValueError(f"Hydration URL manifest has invalid {group} fields")
        for key, value in item.items():
            values = value.values() if key == "file_urls" and isinstance(value, Mapping) else [value]
            for url in values:
                https_origin(str(url))
    return result


def hydrate_all(
    url_manifest: Mapping[str, Any],
    *,
    inputs_dir: Path,
    cache_dir: Path,
    receipt_path: Path,
    fetch: Fetch = fetch_https,
) -> dict[str, Any]:
    """Hydrate all immutable worker inputs and emit a secret-free receipt."""

    manifest = validate_url_manifest(url_manifest)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(receipt_path)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_input_paths = (
        inputs_dir / "step5_v1",
        inputs_dir / "probes" / "step5_grouped_ensemble_v1",
        inputs_dir / "pixmo",
    )
    if any(path.exists() or path.is_symlink() for path in final_input_paths):
        raise FileExistsError("Step 5 hydration destination already exists")
    temporary = Path(tempfile.mkdtemp(prefix="step5-hydration-", dir=inputs_dir.parent))
    try:
        model = hydrate_model(manifest["model"], cache_dir=cache_dir, temporary_dir=temporary, fetch=fetch)
        frozen = hydrate_frozen_inputs(manifest["frozen_inputs"], inputs_dir=inputs_dir, temporary_dir=temporary, fetch=fetch)
        pixmo = hydrate_pixmo(manifest["pixmo"], inputs_dir=inputs_dir, temporary_dir=temporary, fetch=fetch)
        origins = {
            group: sorted(
                {
                    https_origin(str(url))
                    for key, value in manifest[group].items()
                    for url in (value.values() if key == "file_urls" else [value])
                }
            )
            for group in ("model", "frozen_inputs", "pixmo")
        }
        receipt: dict[str, Any] = {
            "format": RECEIPT_FORMAT,
            "frozen_inputs": frozen,
            "model": model,
            "origins": origins,
            "pixmo": pixmo,
            "origin_contract_sha256": canonical_sha256(origins),
        }
        receipt["content_sha256"] = canonical_sha256(receipt)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_receipt, receipt_path)
        return receipt
    except BaseException:
        if not receipt_path.exists():
            for path in final_input_paths:
                shutil.rmtree(path, ignore_errors=True)
            try:
                (inputs_dir / "probes").rmdir()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
