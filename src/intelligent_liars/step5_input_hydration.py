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
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from intelligent_liars.model_cache import (
    MODEL_REVISION,
    REQUIRED_SNAPSHOT_FILES,
    canonical_json_bytes,
    completion_marker,
)
from intelligent_liars.step5_multimodal_assets import validate_staged_bundle


FORMAT = "tinylora_step5_input_url_manifest_v2"
RECEIPT_FORMAT = "tinylora_step5_input_hydration_receipt_v1"
MODEL_FILES = 14
MODEL_REPO = "Qwen/Qwen3-VL-8B-Thinking"
Fetch = Callable[[str, Path], None]
EXPECTED_IDENTITY_FIELDS = (
    "frozen_inputs_archive_sha256",
    "frozen_inputs_completion_sha256",
    "model_completion_sha256",
    "model_content_sha256",
    "model_manifest_sha256",
    "model_revision",
    "pixmo_archive_sha256",
    "pixmo_completion_sha256",
    "pixmo_content_sha256",
    "pixmo_manifest_sha256",
    "plan_sha256",
    "probe_qualification_file_sha256",
    "probe_qualification_receipt_sha256",
)
EXPECTED_PROBE_FILES = {
    "fit_report.json",
    "legacy_identity_registry.json",
    "probe_qualification.json",
    "probe_registry.json",
    "qualification_summary.json",
    "probes/legacy-grouped-evaluator-00.json",
    "probes/legacy-grouped-evaluator-01.json",
    "probes/legacy-grouped-evaluator-02.json",
    "probes/legacy-grouped-evaluator-03.json",
    "probes/legacy-grouped-evaluator-04.json",
    "probes/legacy-grouped-regularizer.json",
}


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
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


def validate_expected_identities(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate the immutable launch-packet identities trusted by this worker."""

    if set(value) != set(EXPECTED_IDENTITY_FIELDS):
        raise ValueError("Expected hydration identity fields do not match the contract")
    identities = {field: str(value[field]) for field in EXPECTED_IDENTITY_FIELDS}
    for field, item in identities.items():
        length = 40 if field == "model_revision" else 64
        if len(item) != length or any(
            character not in "0123456789abcdef" for character in item
        ):
            raise ValueError(f"Invalid expected hydration identity: {field}")
    if identities["model_revision"] != MODEL_REVISION:
        raise ValueError("Expected model revision is not the approved revision")
    return identities


def https_origin(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Every hydration URL must be credentialless HTTPS")
    return urllib.parse.urlunsplit(("https", parsed.netloc, "", "", ""))


def fetch_https(url: str, destination: Path) -> None:
    """Stream one public or presigned HTTPS GET without ambient credentials."""

    https_origin(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url, method="GET", headers={"User-Agent": "step5-hydrator/1"}
    )
    opener = urllib.request.build_opener(_RejectRedirects)
    try:
        with (
            opener.open(request, timeout=120) as response,
            temporary.open("xb") as output,
        ):
            if response.status != 200:
                raise ValueError(f"Hydration GET returned HTTP {response.status}")
            shutil.copyfileobj(response, output, length=1024 * 1024)
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_download(
    url: str, directory: Path, name: str, fetch: Fetch
) -> tuple[dict[str, Any], str]:
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
    _reject_symlink_ancestors(root, label=label)
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


def _reject_symlink_ancestors(path: Path, *, label: str) -> None:
    """Reject a path if any already-existing component is a symlink."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink: {current}")


def _verify_file(path: Path, specification: Mapping[str, Any], *, label: str) -> None:
    expected_hash = str(specification.get("sha256", ""))
    expected_bytes = specification.get("bytes")
    if (
        len(expected_hash) != 64
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValueError(f"Invalid {label} hash/size specification")
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_hash:
        raise ValueError(f"{label} hash/size mismatch: {path.name}")


def hydrate_model(
    specification: Mapping[str, Any],
    *,
    cache_dir: Path,
    temporary_dir: Path,
    expected_identities: Mapping[str, Any],
    fetch: Fetch,
) -> dict[str, Any]:
    _reject_symlink_ancestors(cache_dir, label="Model cache")
    expected = validate_expected_identities(expected_identities)
    complete, complete_hash = _json_download(
        str(specification["completion_url"]),
        temporary_dir,
        "model.complete.json",
        fetch,
    )
    manifest, manifest_hash = _json_download(
        str(specification["manifest_url"]), temporary_dir, "model.manifest.json", fetch
    )
    expected_complete = completion_marker(manifest)
    if complete != expected_complete:
        raise ValueError("Model completion marker or manifest contract is invalid")
    if (
        manifest.get("format") != "tinylora_model_cache_manifest_v1"
        or manifest.get("complete") is not True
    ):
        raise ValueError("Unsupported or incomplete model manifest")
    if expected_complete["manifest_sha256"] != manifest_hash:
        raise ValueError(
            "Model completion marker does not bind the downloaded manifest"
        )
    if complete.get("content_sha256") != manifest.get("content_sha256"):
        raise ValueError("Model content identity mismatch")
    if complete.get("model") != manifest.get("model") or manifest.get("model") != {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
    }:
        raise ValueError("Model identity mismatch")
    if (
        complete_hash != expected["model_completion_sha256"]
        or manifest_hash != expected["model_manifest_sha256"]
        or manifest.get("content_sha256") != expected["model_content_sha256"]
        or manifest.get("model", {}).get("revision") != expected["model_revision"]
    ):
        raise ValueError("Downloaded model does not match frozen launch identities")
    files = manifest.get("files")
    urls = specification.get("file_urls")
    if (
        not isinstance(files, list)
        or len(files) != MODEL_FILES
        or not isinstance(urls, Mapping)
    ):
        raise ValueError(
            "Model manifest must contain exactly 14 files and matching URLs"
        )
    paths = [str(item.get("path", "")) for item in files if isinstance(item, Mapping)]
    if paths != list(REQUIRED_SNAPSHOT_FILES) or set(urls) != set(paths):
        raise ValueError("Model file URL inventory does not exactly match the manifest")
    total_bytes = sum(item.get("bytes", -1) for item in files)
    if total_bytes != complete.get("total_bytes") or total_bytes != manifest.get(
        "total_bytes"
    ):
        raise ValueError("Model total byte count mismatch")
    revision = str(manifest["model"]["revision"])
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("Model revision must be an exact commit hash")
    snapshot = cache_dir / "models--Qwen--Qwen3-VL-8B-Thinking" / "snapshots" / revision
    if snapshot.exists() or snapshot.is_symlink():
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ValueError("Existing model snapshot root must be a regular directory")
        snapshot_children = list(snapshot.iterdir())
        if {path.name for path in snapshot_children} != set(
            REQUIRED_SNAPSHOT_FILES
        ) or any(path.is_symlink() or not path.is_file() for path in snapshot_children):
            raise ValueError(
                "Existing model snapshot is partial or has unexpected files"
            )
    inventory: list[dict[str, Any]] = []
    for item in files:
        relative = _relative_file(str(item["path"]), label="model file")
        if len(relative.parts) != 1:
            raise ValueError("Model files must live at the snapshot root")
        target = _regular_path(snapshot, relative, label="model cache")
        reused = False
        if target.exists():
            _verify_file(target, item, label="Model file")
            reused = True
        if not reused:
            fetch(str(urls[relative.as_posix()]), target)
            _verify_file(target, item, label="Model file")
        inventory.append(
            {
                "bytes": item["bytes"],
                "path": str(target.resolve()),
                "sha256": item["sha256"],
                "reused": reused,
            }
        )
    return {
        "content_sha256": manifest["content_sha256"],
        "completion_sha256": complete_hash,
        "files": inventory,
        "manifest_sha256": manifest_hash,
        "manifest": manifest,
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


def _extract_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, target: Path
) -> None:
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
    if archive.stat().st_size != complete.get("archive_bytes") or file_sha256(
        archive
    ) != complete.get("archive_sha256"):
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
    if (
        plan.get("format") != "tinylora_step5_plan_v1"
        or plan.get("large_run_enabled")
        or plan.get("paid_execution_enabled")
    ):
        raise ValueError("Frozen plan is unsupported or execution-enabled")
    if plan.get("model") != {
        "attention": "flash_attention_2",
        "model_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "vision_weights_frozen": True,
    }:
        raise ValueError("Frozen plan does not bind the approved model contract")
    inventory = [
        {
            "path": str(plan_path.resolve()),
            "sha256": file_sha256(plan_path),
            "bytes": plan_path.stat().st_size,
        }
    ]
    for name, item in sorted(plan.get("outputs", {}).items()):
        relative = _relative_file(
            str(item.get("path", "")), label=f"plan output {name}"
        )
        if len(relative.parts) != 1:
            raise ValueError("Plan output paths must be colocated with the manifest")
        path = plan_path.parent / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"Frozen plan output hash mismatch: {name}")
        rows = sum(1 for line in path.read_text().splitlines() if line.strip())
        if rows != item.get("records"):
            raise ValueError(f"Frozen plan output count mismatch: {name}")
        inventory.append(
            {
                "path": str(path.resolve()),
                "sha256": item["sha256"],
                "bytes": path.stat().st_size,
            }
        )
    return plan, inventory


def hydrate_frozen_inputs(
    specification: Mapping[str, Any],
    *,
    inputs_dir: Path,
    temporary_dir: Path,
    fetch: Fetch,
) -> dict[str, Any]:
    complete, complete_hash = _json_download(
        str(specification["completion_url"]),
        temporary_dir,
        "inputs.complete.json",
        fetch,
    )
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
                plan_prefix = PurePosixPath(
                    "corpora/tinylora_deception_action_v1/step5_v1"
                )
                probe_prefix = PurePosixPath(
                    "artifacts/probes/step5_grouped_ensemble_v1"
                )
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
                    raise ValueError(
                        f"Unexpected AppleDouble frozen-input member: {name}"
                    )
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
        qualification_file_sha256 = file_sha256(qualification)
        for path in sorted(probe_root.rglob("*")):
            if path.is_file():
                inventory.append(
                    {
                        "path": str(path.resolve()),
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
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
            "probe_path": str(
                (final_probe / "probes" / "legacy-grouped-regularizer.json").resolve()
            ),
            "probe_qualification_file_sha256": qualification_file_sha256,
            "probe_qualification_receipt_sha256": complete[
                "probe_qualification_receipt_sha256"
            ],
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
    complete, complete_hash = _json_download(
        str(specification["completion_url"]),
        temporary_dir,
        "pixmo.complete.json",
        fetch,
    )
    manifest, manifest_hash = _json_download(
        str(specification["manifest_url"]), temporary_dir, "pixmo.manifest.json", fetch
    )
    if complete.get("format") != "tinylora_step5_multimodal_s3_completion_v1":
        raise ValueError("Unsupported PixMo completion marker")
    if manifest_hash != complete.get("manifest_sha256") or manifest.get(
        "content_sha256"
    ) != complete.get("manifest_commitment"):
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
                target = _regular_path(
                    stage,
                    _relative_file(member.name, label="PixMo member"),
                    label="PixMo bundle",
                )
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
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
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
    if payload.get("format") != FORMAT or set(payload) != {
        "controller",
        "format",
        "model",
        "frozen_inputs",
        "pixmo",
    }:
        raise ValueError("Unsupported hydration URL manifest")
    controller = payload.get("controller")
    if not isinstance(controller, Mapping) or set(controller) != {
        "account_id",
        "bucket",
        "created_at",
        "expires_at",
        "expiry_seconds",
        "manifest_key",
        "region",
    }:
        raise ValueError("Hydration URL manifest has invalid controller binding")
    if (
        not str(controller["account_id"]).isdigit()
        or len(str(controller["account_id"])) != 12
        or not isinstance(controller["expiry_seconds"], int)
        or not 60 <= controller["expiry_seconds"] <= 604800
        or any(
            not str(controller[field])
            for field in (
                "bucket",
                "created_at",
                "expires_at",
                "manifest_key",
                "region",
            )
        )
    ):
        raise ValueError("Hydration URL manifest controller binding is malformed")
    try:
        created = datetime.fromisoformat(
            str(controller["created_at"]).replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(
            str(controller["expires_at"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("Hydration URL manifest timestamps are invalid") from error
    if (
        created.tzinfo is None
        or expires.tzinfo is None
        or created.astimezone(timezone.utc)
        + timedelta(seconds=controller["expiry_seconds"])
        != expires.astimezone(timezone.utc)
    ):
        raise ValueError("Hydration URL manifest expiry binding is inconsistent")
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
            values = (
                value.values()
                if key == "file_urls" and isinstance(value, Mapping)
                else [value]
            )
            for url in values:
                https_origin(str(url))
    return result


def _origins(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        group: sorted(
            {
                https_origin(str(url))
                for key, value in manifest[group].items()
                for url in (value.values() if key == "file_urls" else [value])
            }
        )
        for group in ("model", "frozen_inputs", "pixmo")
    }


def _receipt_identities(receipt: Mapping[str, Any]) -> dict[str, str]:
    return {
        "frozen_inputs_archive_sha256": str(receipt["frozen_inputs"]["archive_sha256"]),
        "frozen_inputs_completion_sha256": str(
            receipt["frozen_inputs"]["completion_sha256"]
        ),
        "model_completion_sha256": str(receipt["model"]["completion_sha256"]),
        "model_content_sha256": str(receipt["model"]["content_sha256"]),
        "model_manifest_sha256": str(receipt["model"]["manifest_sha256"]),
        "model_revision": str(receipt["model"]["model"]["revision"]),
        "pixmo_archive_sha256": str(receipt["pixmo"]["archive_sha256"]),
        "pixmo_completion_sha256": str(receipt["pixmo"]["completion_sha256"]),
        "pixmo_content_sha256": str(receipt["pixmo"]["manifest_commitment"]),
        "pixmo_manifest_sha256": str(receipt["pixmo"]["manifest_sha256"]),
        "plan_sha256": str(receipt["frozen_inputs"]["plan_sha256"]),
        "probe_qualification_file_sha256": str(
            receipt["frozen_inputs"]["probe_qualification_file_sha256"]
        ),
        "probe_qualification_receipt_sha256": str(
            receipt["frozen_inputs"]["probe_qualification_receipt_sha256"]
        ),
    }


def _verify_inventory(items: Any) -> set[Path]:
    if not isinstance(items, list) or not items:
        raise ValueError("Hydration receipt has an invalid file inventory")
    paths: set[Path] = set()
    for item in items:
        if not isinstance(item, Mapping) or set(item) not in (
            {"bytes", "path", "sha256"},
            {"bytes", "path", "reused", "sha256"},
        ):
            raise ValueError("Hydration receipt has an invalid file entry")
        path = Path(str(item["path"]))
        if (
            not path.is_absolute()
            or path in paths
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(
                "Hydration receipt file path is missing, linked, or duplicated"
            )
        _verify_file(path, item, label="Hydrated receipt file")
        paths.add(path)
    return paths


def validate_existing_hydration(
    receipt_path: Path,
    *,
    inputs_dir: Path,
    cache_dir: Path,
    expected_identities: Mapping[str, Any],
    origins: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate and reuse one exact completed hydration without network downloads."""

    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Existing hydration receipt must be a regular file")
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or receipt.get("format") != RECEIPT_FORMAT:
        raise ValueError("Existing hydration receipt has an unsupported format")
    claimed = receipt.get("content_sha256")
    unsigned = dict(receipt)
    unsigned.pop("content_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("Existing hydration receipt content hash mismatch")
    expected = validate_expected_identities(expected_identities)
    if _receipt_identities(receipt) != expected:
        raise ValueError("Existing hydration receipt does not match frozen identities")
    if receipt.get("origins") != origins or receipt.get(
        "origin_contract_sha256"
    ) != canonical_sha256(origins):
        raise ValueError("Existing hydration receipt does not match URL origins")

    plan_root = (inputs_dir / "step5_v1").resolve()
    probe_root = (inputs_dir / "probes" / "step5_grouped_ensemble_v1").resolve()
    pixmo_root = (inputs_dir / "pixmo").resolve()
    model_root = (
        cache_dir / "models--Qwen--Qwen3-VL-8B-Thinking" / "snapshots" / MODEL_REVISION
    ).resolve()
    if receipt["frozen_inputs"].get("plan_path") != str(plan_root / "manifest.json"):
        raise ValueError("Existing hydration receipt has the wrong plan path")
    if receipt["frozen_inputs"].get("probe_path") != str(
        probe_root / "probes" / "legacy-grouped-regularizer.json"
    ):
        raise ValueError("Existing hydration receipt has the wrong probe path")
    if receipt["pixmo"].get("bundle_path") != str(pixmo_root):
        raise ValueError("Existing hydration receipt has the wrong PixMo path")
    if receipt["model"].get("snapshot_path") != str(model_root):
        raise ValueError("Existing hydration receipt has the wrong model path")

    frozen_paths = _verify_inventory(receipt["frozen_inputs"].get("files"))
    actual_frozen = {
        path.resolve()
        for root in (plan_root, probe_root)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if frozen_paths != actual_frozen:
        raise ValueError("Existing frozen input inventory has changed")
    plan, _ = _validate_plan(plan_root / "manifest.json")
    expected_plan_paths = {
        plan_root / "manifest.json",
        *(plan_root / str(item["path"]) for item in plan["outputs"].values()),
    }
    if {
        path.resolve()
        for path in plan_root.iterdir()
        if path.is_file() or path.is_symlink()
    } != {path.resolve() for path in expected_plan_paths}:
        raise ValueError("Existing frozen plan file inventory has changed")
    qualification = probe_root / "probe_qualification.json"
    if file_sha256(plan_root / "manifest.json") != expected["plan_sha256"]:
        raise ValueError("Existing frozen plan hash mismatch")
    if file_sha256(qualification) != expected["probe_qualification_file_sha256"]:
        raise ValueError("Existing probe qualification file hash mismatch")
    if (
        _qualification_receipt(
            qualification, expected_plan_sha256=expected["plan_sha256"]
        )
        != expected["probe_qualification_receipt_sha256"]
    ):
        raise ValueError("Existing probe qualification receipt mismatch")
    actual_probe_paths = {
        path.relative_to(probe_root).as_posix()
        for path in probe_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_probe_paths != EXPECTED_PROBE_FILES:
        raise ValueError("Existing probe bundle inventory has changed")
    qualification_manifest = json.loads(qualification.read_text())
    ensembles = qualification_manifest.get("ensembles", {})
    if (
        set(ensembles) != {"regularizer", "evaluator"}
        or len(ensembles["regularizer"]) != 1
        or len(ensembles["evaluator"]) != 5
    ):
        raise ValueError(
            "Existing probe qualification has the wrong ensemble inventory"
        )
    committed_probe_paths: set[str] = set()
    for probe in [*ensembles["regularizer"], *ensembles["evaluator"]]:
        relative = _relative_file(
            str(probe.get("artifact_path", "")), label="probe artifact"
        )
        path = probe_root / relative
        _verify_file(
            path,
            {"bytes": path.stat().st_size, "sha256": probe.get("artifact_sha256")},
            label="Qualified probe artifact",
        )
        committed_probe_paths.add(relative.as_posix())
    if committed_probe_paths != {
        path for path in EXPECTED_PROBE_FILES if path.startswith("probes/")
    }:
        raise ValueError("Existing qualification commits the wrong probe artifacts")

    model_manifest = receipt["model"].get("manifest")
    if (
        not isinstance(model_manifest, Mapping)
        or hashlib.sha256(canonical_json_bytes(model_manifest)).hexdigest()
        != expected["model_manifest_sha256"]
    ):
        raise ValueError("Existing model manifest is not authenticated")
    if (
        completion_marker(model_manifest)["content_sha256"]
        != expected["model_content_sha256"]
    ):
        raise ValueError("Existing model manifest content identity mismatch")
    model_specifications = {str(item["path"]): item for item in model_manifest["files"]}
    model_paths = _verify_inventory(receipt["model"].get("files"))
    model_children = list(model_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in model_children):
        raise ValueError("Existing model snapshot contains a linked or non-file child")
    actual_model = {path.resolve() for path in model_children}
    if model_paths != actual_model or {path.name for path in model_paths} != set(
        REQUIRED_SNAPSHOT_FILES
    ):
        raise ValueError("Existing model snapshot inventory has changed")
    for path in model_paths:
        _verify_file(
            path,
            model_specifications[path.name],
            label="Authenticated model file",
        )
    pixmo_manifest = validate_staged_bundle(pixmo_root)
    if pixmo_manifest.get("content_sha256") != expected["pixmo_content_sha256"]:
        raise ValueError("Existing PixMo content identity mismatch")
    if file_sha256(pixmo_root / "manifest.json") != expected["pixmo_manifest_sha256"]:
        raise ValueError("Existing PixMo manifest file hash mismatch")
    if _verify_inventory(receipt["pixmo"].get("files")) != {
        path.resolve() for path in pixmo_root.rglob("*") if path.is_file()
    }:
        raise ValueError("Existing PixMo receipt inventory has changed")
    return receipt


def hydrate_all(
    url_manifest: Mapping[str, Any],
    *,
    inputs_dir: Path,
    cache_dir: Path,
    receipt_path: Path,
    expected_identities: Mapping[str, Any],
    fetch: Fetch = fetch_https,
) -> dict[str, Any]:
    """Hydrate all immutable worker inputs and emit a secret-free receipt."""

    manifest = validate_url_manifest(url_manifest)
    expected = validate_expected_identities(expected_identities)
    _reject_symlink_ancestors(inputs_dir, label="Input destination")
    _reject_symlink_ancestors(receipt_path, label="Hydration receipt")
    _reject_symlink_ancestors(cache_dir, label="Model cache")
    inputs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_input_paths = (
        inputs_dir / "step5_v1",
        inputs_dir / "probes" / "step5_grouped_ensemble_v1",
        inputs_dir / "pixmo",
    )
    origins = _origins(manifest)
    if receipt_path.exists() or receipt_path.is_symlink():
        if not all(
            path.exists() and not path.is_symlink() for path in final_input_paths
        ):
            raise ValueError("Existing hydration is partial")
        return validate_existing_hydration(
            receipt_path,
            inputs_dir=inputs_dir,
            cache_dir=cache_dir,
            expected_identities=expected,
            origins=origins,
        )
    if any(path.exists() or path.is_symlink() for path in final_input_paths):
        raise FileExistsError("Step 5 hydration destination already exists")
    temporary = Path(tempfile.mkdtemp(prefix="step5-hydration-", dir=inputs_dir.parent))
    try:
        model = hydrate_model(
            manifest["model"],
            cache_dir=cache_dir,
            temporary_dir=temporary,
            expected_identities=expected,
            fetch=fetch,
        )
        frozen = hydrate_frozen_inputs(
            manifest["frozen_inputs"],
            inputs_dir=inputs_dir,
            temporary_dir=temporary,
            fetch=fetch,
        )
        pixmo = hydrate_pixmo(
            manifest["pixmo"],
            inputs_dir=inputs_dir,
            temporary_dir=temporary,
            fetch=fetch,
        )
        receipt: dict[str, Any] = {
            "format": RECEIPT_FORMAT,
            "frozen_inputs": frozen,
            "model": model,
            "origins": origins,
            "pixmo": pixmo,
            "origin_contract_sha256": canonical_sha256(origins),
        }
        if _receipt_identities(receipt) != expected:
            raise ValueError("Hydrated inputs do not match frozen launch identities")
        receipt["content_sha256"] = canonical_sha256(receipt)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_receipt = receipt_path.with_name(
            f".{receipt_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary_receipt.open("x") as output:
                output.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            os.link(temporary_receipt, receipt_path, follow_symlinks=False)
        finally:
            temporary_receipt.unlink(missing_ok=True)
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
