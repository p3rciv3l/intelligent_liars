"""Trusted-controller preparation of credentialless Step 5 input URLs."""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlsplit

from intelligent_liars.model_cache import REQUIRED_SNAPSHOT_FILES, completion_marker
from intelligent_liars.step5_input_hydration import FORMAT


RECEIPT_FORMAT = "tinylora_step5_input_url_controller_receipt_v1"
STREAM_VERIFY_LIMIT = 64 * 1024 * 1024


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _s3_uri(value: str) -> tuple[str, str]:
    parsed = urlsplit(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid frozen S3 URI: {value!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _safe_key(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe S3 key: {value!r}")
    return path.as_posix()


def _read_body(response: Mapping[str, Any]) -> bytes:
    body = response.get("Body")
    if body is None:
        raise ValueError("S3 response lacks an object body")
    return body.read()


def _checksum_sha256(head: Mapping[str, Any]) -> tuple[str, str] | None:
    encoded = head.get("ChecksumSHA256")
    if isinstance(encoded, str) and head.get("ChecksumType") in {None, "FULL_OBJECT"}:
        return base64.b64decode(encoded, validate=True).hex(), "head_full_object_sha256"
    metadata = head.get("Metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("sha256")
        if isinstance(value, str):
            return value.lower(), "head_sha256_metadata"
    return None


def _verify_object(
    s3: Any,
    *,
    bucket: str,
    key: str,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    if head.get("ContentLength") != expected_bytes:
        raise ValueError(f"S3 object size mismatch: {key}")
    checksum = _checksum_sha256(head)
    claimed, verification = checksum or (None, "")
    if checksum is None and expected_bytes <= STREAM_VERIFY_LIMIT:
        payload = _read_body(s3.get_object(Bucket=bucket, Key=key))
        claimed = _sha256_bytes(payload)
        verification = "stream_sha256"
    if claimed != expected_sha256:
        raise ValueError(f"S3 object lacks the exact SHA-256 commitment: {key}")
    return {
        "bytes": expected_bytes,
        "key": key,
        "sha256": expected_sha256,
        "verification": verification,
    }


def _get_exact_json(
    s3: Any,
    *,
    bucket: str,
    key: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    response = s3.get_object(Bucket=bucket, Key=key)
    payload = _read_body(response)
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"Frozen JSON object hash mismatch: {key}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Frozen JSON object must contain an object: {key}")
    verified = _verify_object(
        s3,
        bucket=bucket,
        key=key,
        expected_bytes=len(payload),
        expected_sha256=expected_sha256,
    )
    return value, payload, verified


def _presign(s3: Any, *, bucket: str, key: str, expiry_seconds: int) -> str:
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry_seconds,
        HttpMethod="GET",
    )
    if urlsplit(url).scheme != "https":
        raise ValueError("S3 presigner did not return HTTPS")
    return url


def _open_private_parent(path: Path) -> tuple[int, str]:
    """Open/create a lexical parent using no-follow directory descriptors."""

    absolute = path.absolute()
    if absolute.name in {"", ".", ".."}:
        raise ValueError("Protected output must name a file")
    flags = os.O_RDONLY | os.O_DIRECTORY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parent.parts[1:]:
            try:
                child = os.open(part, flags | nofollow, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                child = os.open(part, flags | nofollow, dir_fd=descriptor)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        "Protected output may not traverse a symlink or non-directory"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute.name
    except BaseException:
        os.close(descriptor)
        raise


def _preflight_private_outputs(paths: tuple[Path, ...]) -> None:
    for path in paths:
        descriptor, name = _open_private_parent(path)
        try:
            try:
                os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise FileExistsError(f"Protected output already exists: {path}")
        finally:
            os.close(descriptor)


def _stage_private_outputs(
    payloads: tuple[tuple[Path, bytes], ...],
) -> list[tuple[int, str, str]]:
    staged: list[tuple[int, str, str]] = []
    try:
        for path, payload in payloads:
            parent, name = _open_private_parent(path)
            temporary = f".{name}.{secrets.token_hex(12)}.part"
            temporary_created = False
            try:
                try:
                    os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise FileExistsError(f"Protected output already exists: {path}")
                file_descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent,
                )
                temporary_created = True
                with os.fdopen(file_descriptor, "wb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                staged.append((parent, temporary, name))
            except BaseException:
                if temporary_created:
                    try:
                        os.unlink(temporary, dir_fd=parent)
                    except FileNotFoundError:
                        pass
                os.close(parent)
                raise
        return staged
    except BaseException:
        _cleanup_private_stages(staged)
        raise


def _cleanup_private_stages(staged: list[tuple[int, str, str]]) -> None:
    for parent, temporary, _name in staged:
        try:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
        finally:
            os.close(parent)


def _promote_private_outputs(staged: list[tuple[int, str, str]]) -> None:
    promoted: list[tuple[int, str, str, int, int]] = []
    try:
        for parent, temporary, name in staged:
            os.link(
                temporary,
                name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
            identity = os.stat(temporary, dir_fd=parent, follow_symlinks=False)
            promoted.append((parent, temporary, name, identity.st_dev, identity.st_ino))
            os.fsync(parent)
    except BaseException:
        for parent, _temporary, name, expected_device, expected_inode in reversed(
            promoted
        ):
            try:
                actual = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if (actual.st_dev, actual.st_ino) == (expected_device, expected_inode):
                    os.unlink(name, dir_fd=parent)
            except FileNotFoundError:
                pass
        _cleanup_private_stages(staged)
        raise
    _cleanup_private_stages(staged)


def _bucket_region(s3: Any, bucket: str) -> str:
    location = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint")
    if location is None:
        return "us-east-1"
    if location == "EU":
        return "eu-west-1"
    if not isinstance(location, str) or not location:
        raise ValueError("S3 returned an invalid bucket region")
    return location


def _publish_or_adopt_manifest(
    s3: Any, *, bucket: str, key: str, payload: bytes, sha256: str
) -> None:
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            Metadata={"sha256": sha256},
            ServerSideEncryption="AES256",
            IfNoneMatch="*",
        )
        return
    except BaseException as publish_error:
        try:
            existing = _read_body(s3.get_object(Bucket=bucket, Key=key))
            if existing != payload:
                raise ValueError("Remote controller manifest collision")
            _verify_object(
                s3,
                bucket=bucket,
                key=key,
                expected_bytes=len(payload),
                expected_sha256=sha256,
            )
        except BaseException:
            raise publish_error


def prepare_input_urls(
    packet: Mapping[str, Any],
    *,
    s3: Any,
    sts: Any,
    account_id: str,
    region: str,
    manifest_bucket: str,
    manifest_key: str,
    manifest_output: Path,
    url_file: Path,
    host_gate_url_file: Path,
    receipt_path: Path,
    expiry_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify frozen S3 objects, publish the URL manifest, and emit no secrets."""

    if packet.get("format") != "tinylora_step5_canary_launch_packet_v1":
        raise ValueError("Unsupported launch packet")
    if packet.get("execution", {}).get("enabled") is not False:
        raise ValueError("URL preparation requires an inert launch packet")
    if not account_id.isdigit() or len(account_id) != 12:
        raise ValueError("AWS account id must be 12 digits")
    if sts.get_caller_identity().get("Account") != account_id:
        raise ValueError("Active AWS identity does not match the approved account")
    if not 60 <= expiry_seconds <= 604800:
        raise ValueError("URL expiry must be between 60 and 604800 seconds")
    manifest_key = _safe_key(manifest_key)
    outputs = (manifest_output, url_file, host_gate_url_file, receipt_path)
    if len({path.absolute() for path in outputs}) != len(outputs):
        raise ValueError("Protected controller outputs must use distinct paths")
    _preflight_private_outputs(outputs)
    identity = packet["identity"]
    remote = packet["remote_inputs"]
    objects: list[dict[str, Any]] = []

    model_bucket, model_prefix = _s3_uri(remote["model_s3_prefix"])
    model_prefix = model_prefix.rstrip("/")
    model_manifest_key = f"{model_prefix}/manifest.json"
    model_complete_key = f"{model_prefix}/_COMPLETE.json"
    model_manifest, _manifest_bytes, verified = _get_exact_json(
        s3,
        bucket=model_bucket,
        key=model_manifest_key,
        expected_sha256=remote["model_manifest_sha256"],
    )
    objects.append(verified)
    model_complete, _complete_bytes, verified = _get_exact_json(
        s3,
        bucket=model_bucket,
        key=model_complete_key,
        expected_sha256=remote["model_completion_sha256"],
    )
    objects.append(verified)
    if model_complete != completion_marker(model_manifest):
        raise ValueError("Model completion marker does not match its manifest")
    if (
        model_manifest["content_sha256"] != identity["model_content_sha256"]
        or model_manifest["model"]["revision"] != identity["model_revision"]
    ):
        raise ValueError("Model objects do not match launch identity")
    model_urls: dict[str, str] = {}
    largest_model: tuple[int, str] | None = None
    if [item["path"] for item in model_manifest["files"]] != list(
        REQUIRED_SNAPSHOT_FILES
    ):
        raise ValueError("Model manifest has the wrong file inventory")
    for item in model_manifest["files"]:
        key = f"{model_prefix}/files/{item['path']}"
        objects.append(
            _verify_object(
                s3,
                bucket=model_bucket,
                key=key,
                expected_bytes=item["bytes"],
                expected_sha256=item["sha256"],
            )
        )
        model_urls[item["path"]] = _presign(
            s3, bucket=model_bucket, key=key, expiry_seconds=expiry_seconds
        )
        if largest_model is None or item["bytes"] > largest_model[0]:
            largest_model = (item["bytes"], key)
    assert largest_model is not None

    frozen_bucket, frozen_archive_key = _s3_uri(remote["plan_s3_uri"])
    frozen_prefix = str(PurePosixPath(frozen_archive_key).parent)
    frozen_complete_key = f"{frozen_prefix}/_COMPLETE.json"
    frozen_complete, _payload, verified = _get_exact_json(
        s3,
        bucket=frozen_bucket,
        key=frozen_complete_key,
        expected_sha256=remote["frozen_inputs_completion_sha256"],
    )
    objects.append(verified)
    if (
        frozen_complete["archive_sha256"] != remote["frozen_inputs_tar_sha256"]
        or frozen_complete["plan_sha256"] != identity["plan_sha256"]
        or frozen_complete["probe_qualification_receipt_sha256"]
        != identity["probe_qualification_receipt_sha256"]
    ):
        raise ValueError("Frozen input completion does not match launch identity")
    objects.append(
        _verify_object(
            s3,
            bucket=frozen_bucket,
            key=frozen_archive_key,
            expected_bytes=frozen_complete["archive_bytes"],
            expected_sha256=frozen_complete["archive_sha256"],
        )
    )

    pixmo_bucket, pixmo_prefix = _s3_uri(remote["pixmo_s3_prefix"])
    pixmo_prefix = pixmo_prefix.rstrip("/")
    pixmo_complete_key = f"{pixmo_prefix}/_COMPLETE.json"
    pixmo_manifest_key = f"{pixmo_prefix}/manifest.json"
    pixmo_complete, _payload, verified = _get_exact_json(
        s3,
        bucket=pixmo_bucket,
        key=pixmo_complete_key,
        expected_sha256=remote["pixmo_completion_sha256"],
    )
    objects.append(verified)
    pixmo_manifest, _payload, verified = _get_exact_json(
        s3,
        bucket=pixmo_bucket,
        key=pixmo_manifest_key,
        expected_sha256=remote["pixmo_manifest_sha256"],
    )
    objects.append(verified)
    if (
        pixmo_complete["archive_sha256"] != remote["pixmo_tar_sha256"]
        or pixmo_complete["manifest_commitment"] != identity["pixmo_content_sha256"]
        or pixmo_manifest["content_sha256"] != identity["pixmo_content_sha256"]
    ):
        raise ValueError("PixMo objects do not match launch identity")
    pixmo_archive_key = f"{pixmo_prefix}/{pixmo_complete['archive_name']}"
    objects.append(
        _verify_object(
            s3,
            bucket=pixmo_bucket,
            key=pixmo_archive_key,
            expected_bytes=pixmo_complete["archive_bytes"],
            expected_sha256=pixmo_complete["archive_sha256"],
        )
    )

    buckets = {model_bucket, frozen_bucket, pixmo_bucket, manifest_bucket}
    if len(buckets) != 1:
        raise ValueError(
            "All frozen inputs and controller manifest must use one bucket"
        )
    if _bucket_region(s3, manifest_bucket) != region:
        raise ValueError("Frozen input bucket is not in the approved AWS region")
    if largest_model[0] < STREAM_VERIFY_LIMIT:
        raise ValueError("Model cache has no large object suitable for the host gate")
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = created + timedelta(seconds=expiry_seconds)
    controller = {
        "account_id": account_id,
        "bucket": manifest_bucket,
        "created_at": created.isoformat().replace("+00:00", "Z"),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
        "expiry_seconds": expiry_seconds,
        "manifest_key": manifest_key,
        "region": region,
    }
    url_manifest = {
        "controller": controller,
        "format": FORMAT,
        "frozen_inputs": {
            "archive_url": _presign(
                s3,
                bucket=frozen_bucket,
                key=frozen_archive_key,
                expiry_seconds=expiry_seconds,
            ),
            "completion_url": _presign(
                s3,
                bucket=frozen_bucket,
                key=frozen_complete_key,
                expiry_seconds=expiry_seconds,
            ),
        },
        "model": {
            "completion_url": _presign(
                s3,
                bucket=model_bucket,
                key=model_complete_key,
                expiry_seconds=expiry_seconds,
            ),
            "file_urls": model_urls,
            "manifest_url": _presign(
                s3,
                bucket=model_bucket,
                key=model_manifest_key,
                expiry_seconds=expiry_seconds,
            ),
        },
        "pixmo": {
            "archive_url": _presign(
                s3,
                bucket=pixmo_bucket,
                key=pixmo_archive_key,
                expiry_seconds=expiry_seconds,
            ),
            "completion_url": _presign(
                s3,
                bucket=pixmo_bucket,
                key=pixmo_complete_key,
                expiry_seconds=expiry_seconds,
            ),
            "manifest_url": _presign(
                s3,
                bucket=pixmo_bucket,
                key=pixmo_manifest_key,
                expiry_seconds=expiry_seconds,
            ),
        },
    }
    manifest_bytes = _canonical_bytes(url_manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    manifest_url = _presign(
        s3, bucket=manifest_bucket, key=manifest_key, expiry_seconds=expiry_seconds
    )
    host_gate_url = _presign(
        s3,
        bucket=model_bucket,
        key=largest_model[1],
        expiry_seconds=expiry_seconds,
    )
    receipt = {
        "account_id": account_id,
        "created_at": controller["created_at"],
        "expires_at": controller["expires_at"],
        "expiry_seconds": expiry_seconds,
        "format": RECEIPT_FORMAT,
        "host_gate": {
            "bytes": largest_model[0],
            "key": largest_model[1],
            "sha256": next(
                item["sha256"]
                for item in model_manifest["files"]
                if f"{model_prefix}/files/{item['path']}" == largest_model[1]
            ),
        },
        "manifest": {
            "bucket": manifest_bucket,
            "key": manifest_key,
            "sha256": manifest_sha256,
        },
        "objects": sorted(objects, key=lambda item: item["key"]),
        "region": region,
    }
    receipt["content_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    staged = _stage_private_outputs(
        (
            (manifest_output, manifest_bytes),
            (url_file, (manifest_url + "\n").encode()),
            (host_gate_url_file, (host_gate_url + "\n").encode()),
            (
                receipt_path,
                json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n",
            ),
        )
    )
    try:
        _publish_or_adopt_manifest(
            s3,
            bucket=manifest_bucket,
            key=manifest_key,
            payload=manifest_bytes,
            sha256=manifest_sha256,
        )
    except BaseException:
        _cleanup_private_stages(staged)
        raise
    _promote_private_outputs(staged)
    return receipt
