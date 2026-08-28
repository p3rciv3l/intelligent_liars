from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_osworld_roles import (
    OSWorldRoleLedgerError,
    build_osworld_role_ledger,
    open_osworld_build_receipt,
    open_osworld_optuna_manifest,
    open_osworld_role_ledger,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha(json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode())


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo(root: Path) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")


def _sources(tmp_path: Path) -> dict[str, object]:
    osworld = tmp_path / "osworld"
    _repo(osworld)
    examples = osworld / "evaluation_examples/examples"
    groups = {"chrome": 46, "gimp": 26, "libreoffice_calc": 47, "libreoffice_impress": 47, "libreoffice_writer": 23, "multi_apps": 93, "os": 24, "thunderbird": 15, "vlc": 17, "vs_code": 23}
    index: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    task_ids: list[str] = []
    for group, count in groups.items():
        index[group] = []
        for number in range(count):
            suffix = f"{group}-{number:03d}"
            task_id = f"{group}/{suffix}"
            index[group].append(suffix)
            task_ids.append(task_id)
            url = "https://assets/shared-family" if task_id in {"chrome/chrome-000", "chrome/chrome-001"} else f"https://assets/{task_id}"
            payload = {"id": suffix, "instruction": f"Perform {group} token{number} alpha{number} beta{number} gamma{number}", "source": f"unit-source-{task_id}", "related_apps": [group], "config": [{"type": "download", "parameters": {"url": url}}], "evaluator": {"func": "check", "result": {"type": "task_result"}}}
            path = examples / f"{task_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            hashes[task_id] = _sha(path.read_bytes())
    index_path = osworld / "evaluation_examples/test_nogdrive.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    _git(osworld, "add", ".")
    _git(osworld, "commit", "-qm", "fixture")
    osworld_commit = _git(osworld, "rev-parse", "HEAD")
    template_repo = tmp_path / "template-repo"
    _repo(template_repo)
    template = template_repo / "docs/evaluation/osworld_templates/test_nogdrive_361_50.template.json"
    template.parent.mkdir(parents=True)
    grid = {"name": "test_nogdrive_361_50", "sha256": _sha(b"grid"), "source": "evaluation_examples/test_nogdrive.json", "source_sha256": _sha(index_path.read_bytes()), "task_config_sha256": hashes, "task_ids": task_ids}
    template.write_text(json.dumps({"template_kind": "immutable-osworld-run-template", "template_schema_version": 1, "osworld": {"repository": "https://github.com/xlang-ai/OSWorld.git", "commit": osworld_commit}, "evaluation": {"task_config_sha256": hashes}, "task_grid": grid}, sort_keys=True), encoding="utf-8")
    _git(template_repo, "add", ".")
    _git(template_repo, "commit", "-qm", "template")
    commit = _git(template_repo, "rev-parse", "HEAD")
    relative = template.relative_to(template_repo).as_posix()
    return {"osworld": osworld, "template_repo": template_repo, "template": template, "commit": commit, "blob": _git(template_repo, "rev-parse", f"{commit}:{relative}"), "template_sha": _sha(template.read_bytes())}


def _build(tmp_path: Path) -> tuple[dict[str, object], Path, Path, dict[str, object]]:
    source = _sources(tmp_path)
    private, optimizer = tmp_path / "private-v2", tmp_path / "optimizer-v2"
    build_osworld_role_ledger(template_path=source["template"], ledger_output_dir=private, optimizer_output_dir=optimizer, template_git_root=source["template_repo"], osworld_checkout=source["osworld"], source_git_ref="HEAD", source_git_commit=source["commit"], source_git_blob=source["blob"], expected_template_sha256=source["template_sha"])
    receipt = open_osworld_build_receipt(private / "build-receipt-v2.json", expected_source_git_commit=source["commit"], expected_source_git_blob=source["blob"], expected_template_sha256=source["template_sha"])
    ledger = open_osworld_role_ledger(private / "osworld-role-ledger-v2.json", verified_receipt=receipt, template_path=source["template"], osworld_checkout=source["osworld"])
    return source, private, optimizer, ledger


def test_v2_round_trip_exact_roles_family_isolation_and_optimizer_membership(tmp_path: Path) -> None:
    source, private, optimizer, ledger = _build(tmp_path)
    receipt = open_osworld_build_receipt(private / "build-receipt-v2.json", expected_source_git_commit=source["commit"], expected_source_git_blob=source["blob"], expected_template_sha256=source["template_sha"])
    manifest = open_osworld_optuna_manifest(optimizer / "osworld-optuna-manifest-v2.json", verified_ledger=ledger, verified_receipt=receipt)
    assert {role: sum(row["role"] == role for row in ledger["tasks"]) for role in ("fit", "validation", "capability_test")} == {"fit": 265, "validation": 60, "capability_test": 36}
    families: dict[str, set[str]] = {}
    for row in ledger["tasks"]:
        families.setdefault(row["family_id"], set()).add(row["role"])
    assert all(len(roles) == 1 for roles in families.values())
    paired = [row for row in ledger["tasks"] if row["task_id"] in {"chrome/chrome-000", "chrome/chrome-001"}]
    assert paired[0]["family_id"] == paired[1]["family_id"] and paired[0]["role"] == paired[1]["role"]
    assert {tier: len(ids) for tier, ids in manifest["fit_tiers"].items()} == {"discovery": 24, "promoted": 96, "finalist": 265}
    hidden = {row["task_id"] for row in ledger["tasks"] if row["role"] == "capability_test"}
    assert all(task_id not in json.dumps(manifest) for task_id in hidden)


def test_build_fails_without_authoritative_semantic_material(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    next((source["osworld"] / "evaluation_examples/examples").rglob("*.json")).unlink()
    with pytest.raises(OSWorldRoleLedgerError, match="missing"):
        build_osworld_role_ledger(template_path=source["template"], ledger_output_dir=tmp_path / "p", optimizer_output_dir=tmp_path / "o", template_git_root=source["template_repo"], osworld_checkout=source["osworld"], source_git_ref="HEAD", source_git_commit=source["commit"], source_git_blob=source["blob"], expected_template_sha256=source["template_sha"])


@pytest.mark.parametrize("field", ["commit", "blob", "template_sha"])
def test_build_fails_on_wrong_explicit_trust_root(tmp_path: Path, field: str) -> None:
    source = _sources(tmp_path)
    source[field] = "a" * (64 if field == "template_sha" else 40)
    with pytest.raises(OSWorldRoleLedgerError):
        build_osworld_role_ledger(template_path=source["template"], ledger_output_dir=tmp_path / "p", optimizer_output_dir=tmp_path / "o", template_git_root=source["template_repo"], osworld_checkout=source["osworld"], source_git_ref="HEAD", source_git_commit=source["commit"], source_git_blob=source["blob"], expected_template_sha256=source["template_sha"])


def test_coordinated_ledger_receipt_rewrite_is_rejected(tmp_path: Path) -> None:
    source, private, _, ledger = _build(tmp_path)
    receipt_path = private / "build-receipt-v2.json"
    receipt = json.loads(receipt_path.read_text())
    first = next(row for row in ledger["tasks"] if row["role"] == "fit")
    second = next(row for row in ledger["tasks"] if row["role"] == "validation")
    first["role"], second["role"] = second["role"], first["role"]
    ledger["ledger_id"] = _canonical_sha({k: v for k, v in ledger.items() if k != "ledger_id"})
    ledger_path = private / "osworld-role-ledger-v2.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    receipt["ledger_id"], receipt["ledger_sha256"] = ledger["ledger_id"], _canonical_sha(ledger)
    receipt["receipt_id"] = _canonical_sha({k: v for k, v in receipt.items() if k != "receipt_id"})
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    verified = open_osworld_build_receipt(receipt_path, expected_source_git_commit=source["commit"], expected_source_git_blob=source["blob"], expected_template_sha256=source["template_sha"])
    with pytest.raises(OSWorldRoleLedgerError):
        open_osworld_role_ledger(ledger_path, verified_receipt=verified, template_path=source["template"], osworld_checkout=source["osworld"])


def test_optimizer_coordinated_rewrite_cannot_choose_arbitrary_task(tmp_path: Path) -> None:
    source, private, optimizer, ledger = _build(tmp_path)
    receipt_path = private / "build-receipt-v2.json"
    receipt = json.loads(receipt_path.read_text())
    path = optimizer / "osworld-optuna-manifest-v2.json"
    manifest = json.loads(path.read_text())
    manifest["fit_tiers"]["discovery"][0] = manifest["validation_task_ids"][0]
    manifest["manifest_id"] = _canonical_sha({k: v for k, v in manifest.items() if k != "manifest_id"})
    receipt["optuna_manifest_id"], receipt["optimizer_manifest_sha256"] = manifest["manifest_id"], _canonical_sha(manifest)
    receipt["receipt_id"] = _canonical_sha({k: v for k, v in receipt.items() if k != "receipt_id"})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    verified = open_osworld_build_receipt(receipt_path, expected_source_git_commit=source["commit"], expected_source_git_blob=source["blob"], expected_template_sha256=source["template_sha"])
    with pytest.raises(OSWorldRoleLedgerError):
        open_osworld_optuna_manifest(path, verified_ledger=ledger, verified_receipt=verified)


def test_coordinated_family_collision_rewrite_is_rejected(tmp_path: Path) -> None:
    source, private, _, ledger = _build(tmp_path)
    receipt_path = private / "build-receipt-v2.json"
    receipt = json.loads(receipt_path.read_text())
    fit_rows = [row for row in ledger["tasks"] if row["role"] == "fit"]
    first = fit_rows[0]
    second = next(row for row in fit_rows[1:] if row["family_id"] != first["family_id"])
    second["family_id"] = first["family_id"]
    ledger["ledger_id"] = _canonical_sha({k: v for k, v in ledger.items() if k != "ledger_id"})
    ledger_path = private / "osworld-role-ledger-v2.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    receipt["ledger_id"], receipt["ledger_sha256"] = ledger["ledger_id"], _canonical_sha(ledger)
    receipt["receipt_id"] = _canonical_sha({k: v for k, v in receipt.items() if k != "receipt_id"})
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    verified = open_osworld_build_receipt(receipt_path, expected_source_git_commit=source["commit"], expected_source_git_blob=source["blob"], expected_template_sha256=source["template_sha"])
    with pytest.raises(OSWorldRoleLedgerError, match="deterministic authoritative"):
        open_osworld_role_ledger(ledger_path, verified_receipt=verified, template_path=source["template"], osworld_checkout=source["osworld"])
