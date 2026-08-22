from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pytest
from botocore.config import Config

from intelligent_liars.step5_artifact_presigner import (
    build_receipt,
    canonical_bytes,
    effective_expiry_seconds,
    sha256_bytes,
    validate_receipt,
    write_private_outputs,
)


BUCKET = "step5-artifacts-example"
KEY = "immutable/run-7/artifacts.tar"
REGION = "us-west-2"
ACCOUNT = "123456789012"


def _signed_url() -> str:
    session = boto3.session.Session(
        aws_access_key_id="AKIA" + "A" * 16,
        aws_secret_access_key="b" * 40,
        aws_session_token="test-session-token",
        region_name=REGION,
    )
    client = session.client(
        "s3",
        endpoint_url=f"https://s3.{REGION}.amazonaws.com",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
        ),
    )
    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": KEY, "IfNoneMatch": "*"},
        ExpiresIn=3600,
        HttpMethod="PUT",
    )


def _fixture() -> tuple[str, dict, bytes, str]:
    url = _signed_url()
    query_date = url.split("X-Amz-Date=", 1)[1].split("&", 1)[0]
    generated = datetime.strptime(query_date, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    approved = (generated - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    receipt = build_receipt(
        url=url,
        bucket=BUCKET,
        key=KEY,
        region=REGION,
        account_id=ACCOUNT,
        approved_at=approved,
        generated_at=generated,
        expiry_seconds=3600,
    )
    receipt_bytes = canonical_bytes(receipt)
    return url, receipt, receipt_bytes, approved


def test_real_botocore_put_shape_is_fully_bound_and_validates():
    url, receipt, receipt_bytes, approved = _fixture()
    verified = validate_receipt(
        receipt,
        url_bytes=(url + "\n").encode(),
        expected_receipt_sha256=sha256_bytes(receipt_bytes),
        expected_durable_uri=f"s3://{BUCKET}/{KEY}",
        expected_approved_at=approved,
        now=datetime.fromisoformat(receipt["generated_at"].replace("Z", "+00:00")),
        max_approval_age_seconds=3600,
    )
    assert verified["method"] == "PUT"
    assert verified["endpoint"]["host"] == f"{BUCKET}.s3.{REGION}.amazonaws.com"
    assert verified["sigv4"]["signed_headers"] == ["host", "if-none-match"]
    assert "test-session-token" not in json.dumps(receipt)
    assert "AKIA" not in json.dumps(receipt)


def test_temporary_credential_expiry_shortens_or_rejects_url_lifetime():
    current = datetime.now(timezone.utc)
    assert effective_expiry_seconds(
        21600,
        current=current,
        credential_expiry=current + timedelta(hours=2),
    ) == 6900
    assert effective_expiry_seconds(
        21600,
        current=current,
        credential_expiry=current + timedelta(minutes=15),
        credential_is_temporary=True,
    ) == 600
    with pytest.raises(ValueError, match="expire too soon"):
        effective_expiry_seconds(
            21600,
            current=current,
            credential_expiry=current + timedelta(minutes=5),
        )
    with pytest.raises(ValueError, match="expiry is unavailable"):
        effective_expiry_seconds(
            21600,
            current=current,
            credential_expiry=None,
            credential_is_temporary=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("durable_uri", "s3://other/key", "durable URI"),
        ("method", "GET", "format or method"),
        ("approved_at", "2026-01-01T00:00:00Z", "approval binding"),
    ],
)
def test_receipt_drift_fails_closed(field: str, value: str, message: str):
    url, receipt, _receipt_bytes, approved = _fixture()
    receipt[field] = value
    drifted_bytes = canonical_bytes(receipt)
    with pytest.raises(ValueError, match=message):
        validate_receipt(
            receipt,
            url_bytes=(url + "\n").encode(),
            expected_receipt_sha256=sha256_bytes(drifted_bytes),
            expected_durable_uri=f"s3://{BUCKET}/{KEY}",
            expected_approved_at=approved,
            now=datetime.now(timezone.utc),
            max_approval_age_seconds=3600,
        )


def test_url_drift_and_stale_approval_fail_closed():
    url, receipt, receipt_bytes, approved = _fixture()
    with pytest.raises(ValueError, match="URL differs"):
        validate_receipt(
            receipt,
            url_bytes=(url.replace(BUCKET, "other-bucket", 1) + "\n").encode(),
            expected_receipt_sha256=sha256_bytes(receipt_bytes),
            expected_durable_uri=f"s3://{BUCKET}/{KEY}",
            expected_approved_at=approved,
            now=datetime.now(timezone.utc),
            max_approval_age_seconds=3600,
        )
    with pytest.raises(ValueError, match="stale"):
        validate_receipt(
            receipt,
            url_bytes=(url + "\n").encode(),
            expected_receipt_sha256=sha256_bytes(receipt_bytes),
            expected_durable_uri=f"s3://{BUCKET}/{KEY}",
            expected_approved_at=approved,
            now=datetime.fromisoformat(approved.replace("Z", "+00:00"))
            + timedelta(hours=2),
            max_approval_age_seconds=3600,
        )


def test_future_signing_time_and_ambiguous_keys_fail_closed():
    url, _receipt, _receipt_bytes, _approved = _fixture()
    original_date = url.split("X-Amz-Date=", 1)[1].split("&", 1)[0]
    original_time = datetime.strptime(original_date, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    future = original_time + timedelta(hours=2)
    future_url = url.replace(original_date, future.strftime("%Y%m%dT%H%M%SZ"))
    future_url = future_url.replace(
        f"%2F{original_time:%Y%m%d}%2F",
        f"%2F{future:%Y%m%d}%2F",
    )
    approved = original_time.isoformat().replace("+00:00", "Z")
    receipt = build_receipt(
        url=future_url,
        bucket=BUCKET,
        key=KEY,
        region=REGION,
        account_id=ACCOUNT,
        approved_at=approved,
        generated_at=future,
        expiry_seconds=3600,
    )
    receipt_bytes = canonical_bytes(receipt)
    with pytest.raises(ValueError, match="signing time is in the future"):
        validate_receipt(
            receipt,
            url_bytes=(future_url + "\n").encode(),
            expected_receipt_sha256=sha256_bytes(receipt_bytes),
            expected_durable_uri=f"s3://{BUCKET}/{KEY}",
            expected_approved_at=approved,
            now=original_time,
            max_approval_age_seconds=3600,
        )

    with pytest.raises(ValueError, match="artifact key"):
        build_receipt(
            url=url,
            bucket=BUCKET,
            key="ambiguous/object?version=other",
            region=REGION,
            account_id=ACCOUNT,
            approved_at=approved,
            generated_at=original_time,
            expiry_seconds=3600,
        )


def test_private_outputs_are_0600_no_clobber_and_reject_symlink_parent(tmp_path: Path):
    url_path = tmp_path / "protected" / "url"
    receipt_path = tmp_path / "protected" / "receipt.json"
    write_private_outputs(((url_path, b"url\n"), (receipt_path, b"{}\n")))
    assert url_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="already exists"):
        write_private_outputs(((url_path, b"other\n"),))

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    os.symlink(real, linked)
    with pytest.raises(ValueError, match="symlink"):
        write_private_outputs(((linked / "secret", b"secret"),))


def test_private_output_second_publish_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    url_path = tmp_path / "url"
    receipt_path = tmp_path / "receipt"
    real_link = os.link
    calls = 0

    def failing_second_link(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-link failure")
        return real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", failing_second_link)
    with pytest.raises(OSError, match="injected"):
        write_private_outputs(((url_path, b"url\n"), (receipt_path, b"{}\n")))
    assert not url_path.exists()
    assert not receipt_path.exists()
