"""Trusted receipt contract for the Step 5 artifact presigned PUT."""

from __future__ import annotations

import hashlib
import json
import errno
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import parse_qs, quote, unquote, urlsplit


RECEIPT_FORMAT = "tinylora_step5_artifact_presigner_receipt_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_QUERY_KEYS = {
    "X-Amz-Algorithm",
    "X-Amz-Credential",
    "X-Amz-Date",
    "X-Amz-Expires",
    "X-Amz-SignedHeaders",
    "X-Amz-Signature",
}


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_utc(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def effective_expiry_seconds(
    requested: int,
    *,
    current: datetime,
    credential_expiry: datetime | None,
    credential_is_temporary: bool = False,
) -> int:
    if credential_expiry is None:
        if credential_is_temporary:
            raise ValueError("temporary AWS credential expiry is unavailable")
        return requested
    if credential_expiry.tzinfo is None:
        raise ValueError("AWS credential expiry must include a timezone")
    remaining = int(
        (credential_expiry.astimezone(timezone.utc) - current).total_seconds()
    )
    effective = min(requested, remaining - 300)
    if effective < 60:
        raise ValueError("AWS credentials expire too soon to mint artifact PUT URL")
    return effective


def _exact_query(url: str) -> tuple[Any, dict[str, str]]:
    parsed = urlsplit(url)
    values = parse_qs(parsed.query, keep_blank_values=True)
    allowed = _QUERY_KEYS | {"X-Amz-Security-Token"}
    if not _QUERY_KEYS.issubset(values) or not set(values).issubset(allowed):
        raise ValueError("presigned PUT query fields differ from SigV4 contract")
    if any(len(items) != 1 for items in values.values()):
        raise ValueError("presigned PUT query fields must be singular")
    return parsed, {key: items[0] for key, items in values.items()}


def build_receipt(
    *,
    url: str,
    bucket: str,
    key: str,
    region: str,
    account_id: str,
    approved_at: str,
    generated_at: datetime,
    expiry_seconds: int,
) -> dict[str, Any]:
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None:
        raise ValueError("artifact bucket is invalid")
    pure_key = PurePosixPath(key)
    if (
        _SAFE_KEY.fullmatch(key) is None
        or pure_key.is_absolute()
        or ".." in pure_key.parts
        or pure_key.as_posix() != key
    ):
        raise ValueError("artifact key must be a safe relative S3 key")
    if re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", region) is None:
        raise ValueError("artifact region is invalid")
    if re.fullmatch(r"\d{12}", account_id) is None:
        raise ValueError("artifact account ID must contain 12 digits")
    parsed, query = _exact_query(url)
    generated_at = generated_at.astimezone(timezone.utc)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("presigned PUT URL must be credentialless HTTPS")
    expected_host = f"{bucket}.s3.{region}.amazonaws.com"
    if parsed.hostname != expected_host or parsed.port is not None:
        raise ValueError("presigned PUT host differs from exact regional S3 endpoint")
    if unquote(parsed.path.lstrip("/")) != key or parsed.path != "/" + quote(key, safe="/~"):
        raise ValueError("presigned PUT path differs from exact immutable key")
    if query["X-Amz-Algorithm"] != "AWS4-HMAC-SHA256":
        raise ValueError("presigned PUT algorithm must be AWS4-HMAC-SHA256")
    try:
        expires = int(query["X-Amz-Expires"])
    except ValueError as error:
        raise ValueError("presigned PUT expiry is invalid") from error
    if expires != expiry_seconds or not 60 <= expires <= 604800:
        raise ValueError("presigned PUT expiry differs from controller request")
    try:
        signed_at = datetime.strptime(query["X-Amz-Date"], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError("presigned PUT signing time is invalid") from error
    if abs((signed_at - generated_at).total_seconds()) > 300:
        raise ValueError("presigned PUT signing time differs from controller time")
    credential = query["X-Amz-Credential"].split("/")
    if len(credential) != 5 or credential[1:] != [
        signed_at.strftime("%Y%m%d"),
        region,
        "s3",
        "aws4_request",
    ]:
        raise ValueError("presigned PUT credential scope differs")
    if query["X-Amz-SignedHeaders"] != "host;if-none-match":
        raise ValueError("presigned PUT must sign host and if-none-match")
    signature = query["X-Amz-Signature"]
    if _SHA256.fullmatch(signature) is None:
        raise ValueError("presigned PUT signature is invalid")
    approval = parse_utc(approved_at, label="approved_at")
    if approval > generated_at + timedelta(minutes=5):
        raise ValueError("approval timestamp is later than URL generation")
    core = {
        "format": RECEIPT_FORMAT,
        "account_id": account_id,
        "approved_at": approval.isoformat().replace("+00:00", "Z"),
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "expires_at": (signed_at + timedelta(seconds=expires)).isoformat().replace(
            "+00:00", "Z"
        ),
        "expiry_seconds": expires,
        "method": "PUT",
        "durable_uri": f"s3://{bucket}/{key}",
        "endpoint": {
            "bucket": bucket,
            "key": key,
            "region": region,
            "host": expected_host,
            "path": parsed.path,
        },
        "required_headers": {"if-none-match": "*"},
        "sigv4": {
            "algorithm": query["X-Amz-Algorithm"],
            "credential_access_key_sha256": sha256_bytes(credential[0].encode()),
            "credential_date": credential[1],
            "credential_region": credential[2],
            "credential_service": credential[3],
            "credential_terminal": credential[4],
            "signed_headers": query["X-Amz-SignedHeaders"].split(";"),
            "signature_sha256": sha256_bytes(signature.encode()),
            "session_token_present": "X-Amz-Security-Token" in query,
        },
        "url_sha256": sha256_bytes((url.strip() + "\n").encode()),
    }
    return {**core, "receipt_id": sha256_bytes(canonical_bytes(core))}


def validate_receipt(
    value: Mapping[str, Any],
    *,
    url_bytes: bytes,
    expected_receipt_sha256: str,
    expected_durable_uri: str,
    expected_approved_at: str,
    now: datetime,
    max_approval_age_seconds: int | None,
) -> dict[str, Any]:
    if sha256_bytes(canonical_bytes(value)) != expected_receipt_sha256:
        raise ValueError("artifact presigner receipt differs from frozen SHA-256")
    if value.get("format") != RECEIPT_FORMAT or value.get("method") != "PUT":
        raise ValueError("artifact presigner receipt format or method differs")
    if value.get("durable_uri") != expected_durable_uri:
        raise ValueError("artifact presigner durable URI differs")
    if value.get("approved_at") != expected_approved_at:
        raise ValueError("artifact presigner approval binding differs")
    if value.get("url_sha256") != sha256_bytes(url_bytes):
        raise ValueError("artifact PUT URL differs from presigner receipt")
    endpoint = value.get("endpoint")
    sigv4 = value.get("sigv4")
    if not isinstance(endpoint, Mapping) or not isinstance(sigv4, Mapping):
        raise ValueError("artifact presigner endpoint or SigV4 receipt is missing")
    bucket = endpoint.get("bucket")
    key = endpoint.get("key")
    region = endpoint.get("region")
    if value.get("durable_uri") != f"s3://{bucket}/{key}":
        raise ValueError("artifact presigner endpoint differs from durable URI")
    url = url_bytes.decode("utf-8").removesuffix("\n")
    rebuilt = build_receipt(
        url=url,
        bucket=str(bucket),
        key=str(key),
        region=str(region),
        account_id=str(value.get("account_id")),
        approved_at=str(value.get("approved_at")),
        generated_at=parse_utc(str(value.get("generated_at")), label="generated_at"),
        expiry_seconds=int(value.get("expiry_seconds", -1)),
    )
    if rebuilt != dict(value):
        raise ValueError("artifact presigner receipt fields differ from signed URL")
    current = now.astimezone(timezone.utc)
    approval = parse_utc(str(value["approved_at"]), label="approved_at")
    expiry = parse_utc(str(value["expires_at"]), label="expires_at")
    generated = parse_utc(str(value["generated_at"]), label="generated_at")
    if current < approval - timedelta(minutes=5):
        raise ValueError("artifact approval is in the future")
    if generated > current + timedelta(minutes=5):
        raise ValueError("artifact signing time is in the future")
    if (
        max_approval_age_seconds is not None
        and (current - approval).total_seconds() > max_approval_age_seconds
    ):
        raise ValueError("artifact approval is stale")
    if expiry <= current:
        raise ValueError("artifact PUT URL is expired")
    return rebuilt


def create_presigned_put_authorization(
    *,
    bucket: str,
    key: str,
    region: str,
    account_id: str,
    approved_at: str,
    expiry_seconds: int,
    generated_at: datetime | None = None,
    network_timeout_seconds: float | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Authenticate AWS identity/bucket and mint a receipt-bound PUT capability."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:
        raise RuntimeError("boto3 and botocore are required on the controller") from error
    session = boto3.session.Session(region_name=region)
    credentials = session.get_credentials()
    if credentials is None:
        raise ValueError("controller has no AWS credentials")
    frozen_credentials = credentials.get_frozen_credentials()
    current = generated_at or datetime.now(timezone.utc)
    expiry_seconds = effective_expiry_seconds(
        expiry_seconds,
        current=current,
        credential_expiry=getattr(credentials, "_expiry_time", None),
        credential_is_temporary=bool(frozen_credentials.token),
    )
    if network_timeout_seconds is not None and network_timeout_seconds <= 0:
        raise ValueError("artifact presigner network timeout must be positive")
    per_request_timeout = (
        None
        if network_timeout_seconds is None
        else max(0.25, network_timeout_seconds / 4)
    )
    client_config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual"},
        connect_timeout=10 if per_request_timeout is None else min(10, per_request_timeout),
        read_timeout=30 if per_request_timeout is None else min(30, per_request_timeout),
        retries=(
            {"max_attempts": 3, "mode": "standard"}
            if per_request_timeout is None
            else {"total_max_attempts": 1, "mode": "standard"}
        ),
    )
    identity = session.client(
        "sts", region_name=region, config=client_config
    ).get_caller_identity()
    if str(identity.get("Account")) != account_id:
        raise ValueError("active AWS account differs from frozen account ID")
    s3 = session.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=client_config,
    )
    location = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint")
    actual_region = location or "us-east-1"
    if actual_region == "EU":
        actual_region = "eu-west-1"
    if actual_region != region:
        raise ValueError("S3 bucket region differs from frozen region")
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "IfNoneMatch": "*"},
        ExpiresIn=expiry_seconds,
        HttpMethod="PUT",
    )
    receipt = build_receipt(
        url=url,
        bucket=bucket,
        key=key,
        region=region,
        account_id=account_id,
        approved_at=approved_at,
        generated_at=current,
        expiry_seconds=expiry_seconds,
    )
    return (url.strip() + "\n").encode(), receipt


def write_private_outputs(payloads: tuple[tuple[Path, bytes], ...]) -> None:
    """Publish mode-0600 outputs without symlink traversal or overwrite."""
    def open_parent(path: Path) -> tuple[int, str]:
        absolute = path.absolute()
        if absolute.name in {"", ".", ".."}:
            raise ValueError("protected output must name a file")
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
                            "protected output may not traverse a symlink or non-directory"
                        ) from error
                    raise
                os.close(descriptor)
                descriptor = child
            return descriptor, absolute.name
        except BaseException:
            os.close(descriptor)
            raise

    if len({str(path.absolute()) for path, _payload in payloads}) != len(payloads):
        raise ValueError("protected outputs must be distinct")
    staged: list[dict[str, Any]] = []
    try:
        for path, _payload in payloads:
            parent, name = open_parent(path)
            try:
                try:
                    os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise FileExistsError(f"protected output already exists: {path}")
            finally:
                os.close(parent)
        for path, payload in payloads:
            parent, name = open_parent(path)
            temporary = f".{name}.{secrets.token_hex(16)}.part"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append(
                {
                    "parent": parent,
                    "temporary": temporary,
                    "name": name,
                    "installed": False,
                }
            )
        # Publish the non-capability receipt first and the URL capability last.
        for item in reversed(staged):
            parent = item["parent"]
            temporary = item["temporary"]
            name = item["name"]
            os.link(
                temporary,
                name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
            item["installed"] = True
    except BaseException:
        for item in staged:
            if not item["installed"]:
                continue
            parent = item["parent"]
            try:
                temporary_stat = os.stat(
                    item["temporary"], dir_fd=parent, follow_symlinks=False
                )
                installed_stat = os.stat(
                    item["name"], dir_fd=parent, follow_symlinks=False
                )
                if (temporary_stat.st_dev, temporary_stat.st_ino) == (
                    installed_stat.st_dev,
                    installed_stat.st_ino,
                ):
                    os.unlink(item["name"], dir_fd=parent)
            except FileNotFoundError:
                pass
        raise
    finally:
        for item in staged:
            parent = item["parent"]
            try:
                os.unlink(item["temporary"], dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)
