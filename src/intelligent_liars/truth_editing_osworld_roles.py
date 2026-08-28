"""Strict, immutable OSWorld role ledgers for truth-editing preservation.

The full ledger is the controlled artifact that records every task role.  The
separate Optuna manifest deliberately omits capability-test identifiers, so an
optimizer consumer cannot accidentally evaluate or select against them.  This
is an execution-boundary control, not secrecy: OSWorld's public task catalog
still contains every benchmark identifier.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROLE_LEDGER_FORMAT = "truth_editing_osworld_role_ledger_v2"
OPTUNA_MANIFEST_FORMAT = "truth_editing_osworld_optuna_manifest_v2"
BUILD_RECEIPT_FORMAT = "truth_editing_osworld_role_ledger_build_receipt_v2"
EXPECTED_TASK_COUNT = 361
ROLE_COUNTS = {"fit": 265, "validation": 60, "capability_test": 36}
FIT_TIER_COUNTS = {"discovery": 24, "promoted": 96, "finalist": 265}
DEFAULT_SELECTION_SEED = "truth-editing-osworld-partition-v1"
_HEX = frozenset("0123456789abcdef")
_STOPWORDS = frozenset(
    "a an and are as at be for from i in is it me my of on please that the this to want "
    "with would you your".split()
)


class OSWorldRoleLedgerError(RuntimeError):
    """The OSWorld source catalog or a derived role artifact is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OSWorldRoleLedgerError("value is not canonical JSON") from error


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise OSWorldRoleLedgerError(f"file is unreadable: {path}") from error
    return digest.hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OSWorldRoleLedgerError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OSWorldRoleLedgerError(f"{name} must be an array")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise OSWorldRoleLedgerError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise OSWorldRoleLedgerError(f"{name} must be nonempty trimmed text")
    return value


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise OSWorldRoleLedgerError(f"{name} must be a lowercase SHA-256")
    return text


def _git_oid(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) != 40 or any(character not in _HEX for character in text):
        raise OSWorldRoleLedgerError(f"{name} must be a lowercase SHA-1 git object ID")
    return text


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OSWorldRoleLedgerError(f"{name} must be a regular non-symlink file")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OSWorldRoleLedgerError(f"{name} is not strict JSON") from error


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _identity(payload: Mapping[str, Any], field: str) -> str:
    return _hash_json({key: value for key, value in payload.items() if key != field})


def _rank(seed: str, purpose: str, task_id: str, task_hash: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{purpose}\0{task_id}\0{task_hash}".encode("utf-8")
    ).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OSWorldRoleLedgerError("could not verify the declared git object relationship") from error
    return completed.stdout.strip()


def _tokens(value: Any, name: str) -> frozenset[str]:
    if not isinstance(value, str) or not value.strip():
        raise OSWorldRoleLedgerError(f"{name} must be nonempty text")
    text = value.strip().lower()
    return frozenset(
        token for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 2 and token not in _STOPWORDS
    )


def _typed_signature(value: Any) -> tuple[str, ...]:
    result: set[str] = set()
    def visit(current: Any) -> None:
        if isinstance(current, Mapping):
            if isinstance(current.get("type"), str):
                result.add(current["type"])
            for child in current.values():
                visit(child)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            for child in current:
                visit(child)
    visit(value)
    return tuple(sorted(result))


def _urls(value: Any) -> frozenset[str]:
    result: set[str] = set()
    def visit(current: Any) -> None:
        if isinstance(current, Mapping):
            for key, child in current.items():
                if key == "url" and isinstance(child, str) and child.startswith(("http://", "https://")):
                    result.add(child)
                visit(child)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            for child in current:
                visit(child)
    visit(value)
    return frozenset(result)


def _derive_families(
    rows: list[dict[str, str]], task_payloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    evidence: list[dict[str, Any]] = []
    source_frequency: dict[str, int] = {}
    for row in rows:
        payload = task_payloads[row["task_id"]]
        source = str(payload.get("source", "")).strip().lower()
        source_frequency[source] = source_frequency.get(source, 0) + 1
        evaluator = _object(payload.get("evaluator"), "authoritative task evaluator")
        evidence.append({
            "tokens": _tokens(payload.get("instruction"), "authoritative instruction"),
            "source": source,
            "urls": _urls(payload.get("config")),
            "apps": tuple(sorted(str(value).strip().lower().replace(" ", "_") for value in _array(payload.get("related_apps"), "related_apps"))),
            "shape": (
                str(evaluator.get("func")),
                _typed_signature(evaluator),
                _typed_signature(payload.get("config")),
            ),
        })
    parent = list(range(len(rows)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            a, b = evidence[left], evidence[right]
            union_tokens = a["tokens"] | b["tokens"]
            similarity = len(a["tokens"] & b["tokens"]) / max(1, len(union_tokens))
            shared_asset = bool(a["urls"] & b["urls"])
            bounded_source = (
                bool(a["source"])
                and a["source"] == b["source"]
                and source_frequency[a["source"]] <= 10
                and similarity >= 0.30
            )
            instruction_family = (
                bool(set(a["apps"]) & set(b["apps"]))
                and a["shape"] == b["shape"]
                and similarity >= 0.68
            )
            if shared_asset or bounded_source or instruction_family:
                union(left, right)
    components: dict[int, list[str]] = {}
    for index, row in enumerate(rows):
        components.setdefault(find(index), []).append(row["task_id"])
    family_by_task: dict[str, str] = {}
    for members in components.values():
        family_id = f"authoritative-semantic-v1:{_hash_json(sorted(members))}"
        for task_id in members:
            family_by_task[task_id] = family_id
    return family_by_task


def _select_family_safe(
    rows: Sequence[dict[str, str]], target: int, *, seed: str, purpose: str
) -> set[str]:
    families: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        families.setdefault(row["family_id"], []).append(row)
    total_by_group: dict[str, int] = {}
    for row in rows:
        total_by_group[row["group"]] = total_by_group.get(row["group"], 0) + 1
    soft_capacity = {
        group: math.ceil(target * count / len(rows)) + 1
        for group, count in total_by_group.items()
    }
    buckets: dict[str, list[tuple[str, list[dict[str, str]]]]] = {}
    for item in families.items():
        member_counts: dict[str, int] = {}
        for row in item[1]:
            member_counts[row["group"]] = member_counts.get(row["group"], 0) + 1
        if any(count > soft_capacity[group] for group, count in member_counts.items()):
            continue
        group_key = "+".join(sorted({row["group"] for row in item[1]}))
        buckets.setdefault(group_key, []).append(item)
    for group_key, items in buckets.items():
        items.sort(key=lambda item: (
            len(item[1]),
            _rank(seed, purpose, item[0], _hash_json(sorted(row["task_id"] for row in item[1]))),
            item[0],
        ))
    # Round-robin application buckets before subset-sum selection. Families are
    # indivisible, while early reachable solutions remain application-balanced.
    ordered: list[tuple[str, list[dict[str, str]]]] = []
    offset = 0
    while any(offset < len(items) for items in buckets.values()):
        for group_key in sorted(buckets):
            if offset < len(buckets[group_key]):
                ordered.append(buckets[group_key][offset])
        offset += 1
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for family_id, members in ordered:
        size = len(members)
        for count, chosen in sorted(tuple(reachable.items()), reverse=True):
            new_count = count + size
            if new_count <= target and new_count not in reachable:
                reachable[new_count] = chosen + (family_id,)
    if target not in reachable:
        raise OSWorldRoleLedgerError(
            f"authoritative families cannot produce exact {purpose} target {target}"
        )
    selected_families = set(reachable[target])
    selected = {
        row["task_id"] for family_id in selected_families for row in families[family_id]
    }
    if len(selected) != target:
        raise OSWorldRoleLedgerError("family-safe selection did not reach its exact target")
    return selected


def _catalog(
    template_path: Path, osworld_checkout: Path
) -> tuple[Mapping[str, Any], list[dict[str, str]]]:
    template = _load_json(template_path, "OSWorld template")
    if template.get("template_kind") != "immutable-osworld-run-template" or template.get(
        "template_schema_version"
    ) != 1:
        raise OSWorldRoleLedgerError("OSWorld template identity is unsupported")
    osworld = _object(template.get("osworld"), "template.osworld")
    grid = _object(template.get("task_grid"), "template.task_grid")
    evaluation = _object(template.get("evaluation"), "template.evaluation")
    task_ids_raw = _array(grid.get("task_ids"), "template.task_grid.task_ids")
    if len(task_ids_raw) != EXPECTED_TASK_COUNT:
        raise OSWorldRoleLedgerError(
            f"expected exactly {EXPECTED_TASK_COUNT} no-GDrive tasks; found {len(task_ids_raw)}"
        )
    task_ids = [_text(value, "task ID") for value in task_ids_raw]
    if len(set(task_ids)) != EXPECTED_TASK_COUNT:
        raise OSWorldRoleLedgerError("OSWorld task IDs must be unique")
    hashes = _object(grid.get("task_config_sha256"), "template.task_grid.task_config_sha256")
    evaluator_hashes = _object(
        evaluation.get("task_config_sha256"), "template.evaluation.task_config_sha256"
    )
    if set(hashes) != set(task_ids) or dict(hashes) != dict(evaluator_hashes):
        raise OSWorldRoleLedgerError("task IDs and evaluator task hashes differ")

    rows: list[dict[str, str]] = []
    task_payloads: dict[str, Mapping[str, Any]] = {}
    seen_hashes: dict[str, str] = {}
    for task_id in task_ids:
        if task_id.count("/") != 1:
            raise OSWorldRoleLedgerError(f"task ID has no single application group: {task_id}")
        group, suffix = task_id.split("/", 1)
        _text(group, "task group")
        _text(suffix, "task suffix")
        task_hash = _sha256(hashes[task_id], f"task hash for {task_id}")
        previous = seen_hashes.setdefault(task_hash, task_id)
        if previous != task_id:
            raise OSWorldRoleLedgerError(
                f"task-config family collision between {previous} and {task_id}"
            )
        task_path = osworld_checkout / "evaluation_examples" / "examples" / f"{task_id}.json"
        try:
            resolved_task = task_path.resolve(strict=True)
            resolved_root = osworld_checkout.resolve(strict=True)
        except OSError as error:
            raise OSWorldRoleLedgerError(f"authoritative task JSON is missing: {task_id}") from error
        if task_path.is_symlink() or not resolved_task.is_file() or not resolved_task.is_relative_to(resolved_root):
            raise OSWorldRoleLedgerError(f"authoritative task JSON is unsafe: {task_id}")
        if _hash_file(resolved_task) != task_hash:
            raise OSWorldRoleLedgerError(f"authoritative task JSON hash differs: {task_id}")
        payload = _load_json(resolved_task, f"authoritative task {task_id}")
        if payload.get("id") != suffix:
            raise OSWorldRoleLedgerError(f"authoritative task ID differs: {task_id}")
        task_payloads[task_id] = payload
        rows.append(
            {
                "task_id": task_id,
                "group": group,
                "task_config_sha256": task_hash,
            }
        )
    index_path = osworld_checkout / _text(grid.get("source"), "task index path")
    if index_path.is_symlink() or not index_path.is_file() or _hash_file(index_path) != grid.get("source_sha256"):
        raise OSWorldRoleLedgerError("authoritative task index hash differs")
    families = _derive_families(rows, task_payloads)
    for row in rows:
        row["family_id"] = families[row["task_id"]]
    source = {
        "repository": _text(osworld.get("repository"), "OSWorld repository"),
        "osworld_commit": _git_oid(osworld.get("commit"), "OSWorld commit"),
        "task_index_path": _text(grid.get("source"), "task index path"),
        "task_index_sha256": _sha256(grid.get("source_sha256"), "task index SHA-256"),
        "task_grid_name": _text(grid.get("name"), "task grid name"),
        "task_grid_sha256": _sha256(grid.get("sha256"), "task grid SHA-256"),
        "task_hashes_sha256": _hash_json(dict(sorted((str(k), v) for k, v in hashes.items()))),
        "family_derivation": "authoritative-assets-source-evaluator-instruction-v1",
        "authoritative_task_tree_sha256": _hash_json(
            {task_id: _hash_json(task_payloads[task_id]) for task_id in sorted(task_payloads)}
        ),
    }
    return source, rows


def _remove(rows: Sequence[dict[str, str]], selected: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row["task_id"] not in selected]


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    system = platform.system()
    if system == "Darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(
            os.fsencode(source), os.fsencode(destination), ctypes.c_uint(0x00000004)
        )
    elif system == "Linux" and hasattr(libc, "renameat2"):
        result = libc.renameat2(
            ctypes.c_int(-100),
            os.fsencode(source),
            ctypes.c_int(-100),
            os.fsencode(destination),
            ctypes.c_uint(1),
        )
    else:  # pragma: no cover
        raise OSWorldRoleLedgerError("atomic no-replace publication is unsupported")
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise OSWorldRoleLedgerError(f"output already exists: {destination}")
    raise OSError(number, os.strerror(number), str(destination))


def build_osworld_role_ledger(
    *,
    template_path: Path | str,
    ledger_output_dir: Path | str,
    optimizer_output_dir: Path | str,
    template_git_root: Path | str,
    osworld_checkout: Path | str,
    source_git_ref: str,
    source_git_commit: str,
    source_git_blob: str,
    expected_template_sha256: str,
    selection_seed: str = DEFAULT_SELECTION_SEED,
) -> dict[str, Any]:
    """Build the controlled full ledger and redacted Optuna manifest."""

    template = Path(template_path).expanduser().resolve(strict=True)
    ledger_output = Path(ledger_output_dir).expanduser().absolute()
    optimizer_output = Path(optimizer_output_dir).expanduser().absolute()
    for output in (ledger_output, optimizer_output):
        if os.path.lexists(output):
            raise OSWorldRoleLedgerError(f"output already exists: {output}")
    if ledger_output == optimizer_output:
        raise OSWorldRoleLedgerError("private ledger and optimizer outputs must be separate")
    seed = _text(selection_seed, "selection seed")
    git_ref = _text(source_git_ref, "source git ref")
    git_commit = _git_oid(source_git_commit, "source git commit")
    git_blob = _git_oid(source_git_blob, "source git blob")
    template_sha = _sha256(expected_template_sha256, "expected template SHA-256")
    git_root = Path(template_git_root).expanduser().resolve(strict=True)
    checkout = Path(osworld_checkout).expanduser().resolve(strict=True)
    if _git(git_root, "rev-parse", f"{git_ref}^{{commit}}") != git_commit:
        raise OSWorldRoleLedgerError("source git ref does not resolve to the expected commit")
    relative_template = template.relative_to(git_root).as_posix()
    if _git(git_root, "rev-parse", f"{git_commit}:{relative_template}") != git_blob:
        raise OSWorldRoleLedgerError("source git commit does not contain the expected template blob")
    if _hash_file(template) != template_sha:
        raise OSWorldRoleLedgerError("template hash differs from the explicit trust root")
    source, rows = _catalog(template, checkout)
    if _git(checkout, "rev-parse", "HEAD^{commit}") != source["osworld_commit"]:
        raise OSWorldRoleLedgerError("OSWorld checkout is not at the catalog's pinned commit")
    source = dict(source)
    source.update(
        {
            "template_git_ref": git_ref,
            "template_git_commit": git_commit,
            "template_git_blob": git_blob,
            "template_sha256": template_sha,
        }
    )

    capability = _select_family_safe(
        rows, ROLE_COUNTS["capability_test"], seed=seed, purpose="capability-test"
    )
    remainder = _remove(rows, capability)
    validation = _select_family_safe(
        remainder, ROLE_COUNTS["validation"], seed=seed, purpose="validation"
    )
    fit_rows = _remove(remainder, validation)
    fit = {row["task_id"] for row in fit_rows}
    role_by_id = {
        **{task_id: "fit" for task_id in fit},
        **{task_id: "validation" for task_id in validation},
        **{task_id: "capability_test" for task_id in capability},
    }
    tasks = [dict(row, role=role_by_id[row["task_id"]]) for row in sorted(rows, key=lambda row: row["task_id"])]
    ledger: dict[str, Any] = {
        "format": ROLE_LEDGER_FORMAT,
        "partition_policy": "authoritative-semantic-family-v1",
        "selection_seed": seed,
        "source": source,
        "role_counts": dict(ROLE_COUNTS),
        "tasks": tasks,
    }
    ledger["ledger_id"] = _identity(ledger, "ledger_id")

    discovery = _select_family_safe(
        fit_rows, FIT_TIER_COUNTS["discovery"], seed=seed, purpose="fit-discovery"
    )
    promoted_extra = _select_family_safe(
        _remove(fit_rows, discovery),
        FIT_TIER_COUNTS["promoted"] - FIT_TIER_COUNTS["discovery"],
        seed=seed,
        purpose="fit-promoted-extra",
    )
    promoted = discovery | promoted_extra
    optuna: dict[str, Any] = {
        "format": OPTUNA_MANIFEST_FORMAT,
        "ledger_id": ledger["ledger_id"],
        "source_identity": {
            "osworld_commit": source["osworld_commit"],
            "task_index_sha256": source["task_index_sha256"],
            "task_hashes_sha256": source["task_hashes_sha256"],
        },
        "fit_tiers": {
            "discovery": sorted(discovery),
            "promoted": sorted(promoted),
            "finalist": sorted(fit),
        },
        "validation_task_ids": sorted(validation),
        "optimizer_visibility": "fit-and-promoted-validation-only",
    }
    optuna["manifest_id"] = _identity(optuna, "manifest_id")
    receipt: dict[str, Any] = {
        "format": BUILD_RECEIPT_FORMAT,
        "ledger_id": ledger["ledger_id"],
        "optuna_manifest_id": optuna["manifest_id"],
        "role_counts": dict(ROLE_COUNTS),
        "fit_tier_counts": dict(FIT_TIER_COUNTS),
        "source_trust": {
            "template_git_commit": git_commit,
            "template_git_blob": git_blob,
            "template_sha256": template_sha,
        },
    }
    receipt["ledger_sha256"] = _hash_json(ledger)
    receipt["optimizer_manifest_sha256"] = _hash_json(optuna)
    receipt["receipt_id"] = _identity(receipt, "receipt_id")

    ledger_output.parent.mkdir(parents=True, exist_ok=True)
    optimizer_output.parent.mkdir(parents=True, exist_ok=True)
    ledger_staging = Path(tempfile.mkdtemp(prefix=f".{ledger_output.name}.staging-", dir=ledger_output.parent))
    optimizer_staging = Path(tempfile.mkdtemp(prefix=f".{optimizer_output.name}.staging-", dir=optimizer_output.parent))
    try:
        _write_json(ledger_staging / "osworld-role-ledger-v2.json", ledger)
        _write_json(ledger_staging / "build-receipt-v2.json", receipt)
        _write_json(optimizer_staging / "osworld-optuna-manifest-v2.json", optuna)
        _rename_no_replace(ledger_staging, ledger_output)
        try:
            _rename_no_replace(optimizer_staging, optimizer_output)
        except Exception:
            shutil.rmtree(ledger_output, ignore_errors=True)
            raise
    except Exception:
        shutil.rmtree(ledger_staging, ignore_errors=True)
        shutil.rmtree(optimizer_staging, ignore_errors=True)
        raise
    return receipt


def open_osworld_build_receipt(
    path: Path | str,
    *,
    expected_source_git_commit: str,
    expected_source_git_blob: str,
    expected_template_sha256: str,
) -> dict[str, Any]:
    raw = _load_json(Path(path), "OSWorld role build receipt")
    _exact(
        raw,
        {"format", "ledger_id", "optuna_manifest_id", "role_counts", "fit_tier_counts", "source_trust", "ledger_sha256", "optimizer_manifest_sha256", "receipt_id"},
        "OSWorld role build receipt",
    )
    if raw["format"] != BUILD_RECEIPT_FORMAT or dict(_object(raw["role_counts"], "receipt role counts")) != ROLE_COUNTS or dict(_object(raw["fit_tier_counts"], "receipt tier counts")) != FIT_TIER_COUNTS:
        raise OSWorldRoleLedgerError("build receipt format or frozen counts differ")
    trust = _object(raw["source_trust"], "receipt source trust")
    _exact(trust, {"template_git_commit", "template_git_blob", "template_sha256"}, "receipt source trust")
    expected = {
        "template_git_commit": _git_oid(expected_source_git_commit, "expected source git commit"),
        "template_git_blob": _git_oid(expected_source_git_blob, "expected source git blob"),
        "template_sha256": _sha256(expected_template_sha256, "expected template SHA-256"),
    }
    if dict(trust) != expected:
        raise OSWorldRoleLedgerError("build receipt does not match the explicit source trust root")
    for field in ("ledger_id", "optuna_manifest_id", "ledger_sha256", "optimizer_manifest_sha256"):
        _sha256(raw[field], f"receipt.{field}")
    if raw["receipt_id"] != _identity(raw, "receipt_id"):
        raise OSWorldRoleLedgerError("build receipt identity does not match its contents")
    return dict(raw)


def open_osworld_role_ledger(
    path: Path | str,
    *,
    verified_receipt: Mapping[str, Any],
    template_path: Path | str,
    osworld_checkout: Path | str,
) -> dict[str, Any]:
    if verified_receipt.get("format") != BUILD_RECEIPT_FORMAT or verified_receipt.get("receipt_id") != _identity(verified_receipt, "receipt_id"):
        raise OSWorldRoleLedgerError("verified receipt is not a valid opened v2 receipt")
    raw = _load_json(Path(path), "OSWorld role ledger")
    _exact(
        raw,
        {"format", "partition_policy", "selection_seed", "source", "role_counts", "tasks", "ledger_id"},
        "OSWorld role ledger",
    )
    if raw["format"] != ROLE_LEDGER_FORMAT or raw["partition_policy"] != "authoritative-semantic-family-v1":
        raise OSWorldRoleLedgerError("OSWorld role ledger format or policy is unsupported")
    _text(raw["selection_seed"], "selection seed")
    source = _object(raw["source"], "ledger.source")
    _exact(
        source,
        {
            "repository", "osworld_commit", "task_index_path", "task_index_sha256",
            "task_grid_name", "task_grid_sha256", "task_hashes_sha256", "template_git_ref",
            "template_git_commit", "template_git_blob", "template_sha256", "family_derivation",
            "authoritative_task_tree_sha256",
        },
        "ledger.source",
    )
    _text(source["repository"], "source repository")
    _git_oid(source["osworld_commit"], "OSWorld commit")
    _text(source["task_index_path"], "task index path")
    _text(source["task_grid_name"], "task grid name")
    _text(source["template_git_ref"], "template git ref")
    _git_oid(source["template_git_commit"], "template git commit")
    _git_oid(source["template_git_blob"], "template git blob")
    if source["family_derivation"] != "authoritative-assets-source-evaluator-instruction-v1":
        raise OSWorldRoleLedgerError("family derivation policy is unsupported")
    for field in ("task_index_sha256", "task_grid_sha256", "task_hashes_sha256", "template_sha256", "authoritative_task_tree_sha256"):
        _sha256(source[field], f"source.{field}")
    counts = _object(raw["role_counts"], "ledger.role_counts")
    if dict(counts) != ROLE_COUNTS:
        raise OSWorldRoleLedgerError("role counts differ from the frozen 265/60/36 partition")
    tasks_raw = _array(raw["tasks"], "ledger.tasks")
    if len(tasks_raw) != EXPECTED_TASK_COUNT:
        raise OSWorldRoleLedgerError("ledger must contain exactly 361 tasks")
    ids: set[str] = set()
    family_roles: dict[str, set[str]] = {}
    observed = {role: 0 for role in ROLE_COUNTS}
    previous = ""
    tasks: list[dict[str, str]] = []
    for index, value in enumerate(tasks_raw):
        row = _object(value, f"ledger.tasks[{index}]")
        _exact(row, {"task_id", "group", "task_config_sha256", "family_id", "role"}, f"ledger.tasks[{index}]")
        task_id = _text(row["task_id"], "task ID")
        group = _text(row["group"], "task group")
        _sha256(row["task_config_sha256"], "task config SHA-256")
        family = _text(row["family_id"], "family ID")
        role = _text(row["role"], "task role")
        if task_id <= previous or not task_id.startswith(f"{group}/") or not family.startswith("authoritative-semantic-v1:"):
            raise OSWorldRoleLedgerError("ledger task ordering, group, or family identity is invalid")
        previous = task_id
        if task_id in ids or role not in ROLE_COUNTS:
            raise OSWorldRoleLedgerError("ledger contains a duplicate task or unknown role")
        ids.add(task_id)
        observed[role] += 1
        family_roles.setdefault(family, set()).add(role)
        tasks.append(dict(row))
    if observed != ROLE_COUNTS or any(len(roles) != 1 for roles in family_roles.values()):
        raise OSWorldRoleLedgerError("ledger role counts or family isolation are invalid")
    observed_hash_identity = _hash_json(
        {row["task_id"]: row["task_config_sha256"] for row in tasks}
    )
    if observed_hash_identity != source["task_hashes_sha256"]:
        raise OSWorldRoleLedgerError("ledger task hashes differ from the bound source identity")
    if raw["ledger_id"] != _identity(raw, "ledger_id"):
        raise OSWorldRoleLedgerError("ledger identity does not match its contents")
    if raw["ledger_id"] != verified_receipt.get("ledger_id") or _hash_json(raw) != verified_receipt.get("ledger_sha256"):
        raise OSWorldRoleLedgerError("ledger does not match the verified build receipt")
    resolved_template = Path(template_path).expanduser().resolve(strict=True)
    resolved_checkout = Path(osworld_checkout).expanduser().resolve(strict=True)
    trust = _object(verified_receipt.get("source_trust"), "verified receipt source trust")
    if _hash_file(resolved_template) != trust.get("template_sha256"):
        raise OSWorldRoleLedgerError("verification template differs from receipt trust root")
    authoritative_source, authoritative_rows = _catalog(resolved_template, resolved_checkout)
    if _git(resolved_checkout, "rev-parse", "HEAD^{commit}") != authoritative_source["osworld_commit"]:
        raise OSWorldRoleLedgerError("verification OSWorld checkout is not at the pinned commit")
    for field in ("osworld_commit", "task_index_sha256", "task_grid_sha256", "task_hashes_sha256", "family_derivation", "authoritative_task_tree_sha256"):
        if source[field] != authoritative_source[field]:
            raise OSWorldRoleLedgerError("ledger differs from authoritative OSWorld task material")
    capability = _select_family_safe(authoritative_rows, 36, seed=raw["selection_seed"], purpose="capability-test")
    remainder = _remove(authoritative_rows, capability)
    validation = _select_family_safe(remainder, 60, seed=raw["selection_seed"], purpose="validation")
    expected_rows = []
    for row in authoritative_rows:
        role = "capability_test" if row["task_id"] in capability else "validation" if row["task_id"] in validation else "fit"
        expected_rows.append(dict(row, role=role))
    if tasks != sorted(expected_rows, key=lambda row: row["task_id"]):
        raise OSWorldRoleLedgerError("ledger roles or families differ from deterministic authoritative derivation")
    return dict(raw)


def open_osworld_optuna_manifest(
    path: Path | str, *, verified_ledger: Mapping[str, Any], verified_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    if verified_receipt.get("format") != BUILD_RECEIPT_FORMAT or verified_receipt.get("receipt_id") != _identity(verified_receipt, "receipt_id"):
        raise OSWorldRoleLedgerError("verified receipt is not a valid opened v2 receipt")
    if verified_ledger.get("ledger_id") != _identity(verified_ledger, "ledger_id") or _hash_json(verified_ledger) != verified_receipt.get("ledger_sha256"):
        raise OSWorldRoleLedgerError("verified ledger is not bound by the verified receipt")
    raw = _load_json(Path(path), "OSWorld Optuna manifest")
    _exact(
        raw,
        {"format", "ledger_id", "source_identity", "fit_tiers", "validation_task_ids", "optimizer_visibility", "manifest_id"},
        "OSWorld Optuna manifest",
    )
    if raw["format"] != OPTUNA_MANIFEST_FORMAT or raw["optimizer_visibility"] != "fit-and-promoted-validation-only":
        raise OSWorldRoleLedgerError("OSWorld Optuna manifest format or visibility policy is unsupported")
    ledger_id = _sha256(raw["ledger_id"], "ledger ID")
    if ledger_id != verified_ledger.get("ledger_id") or ledger_id != verified_receipt.get("ledger_id"):
        raise OSWorldRoleLedgerError("OSWorld Optuna manifest binds a different role ledger")
    source = _object(raw["source_identity"], "Optuna source identity")
    _exact(source, {"osworld_commit", "task_index_sha256", "task_hashes_sha256"}, "Optuna source identity")
    _git_oid(source["osworld_commit"], "OSWorld commit")
    _sha256(source["task_index_sha256"], "task index SHA-256")
    _sha256(source["task_hashes_sha256"], "task hashes SHA-256")
    tiers = _object(raw["fit_tiers"], "Optuna fit tiers")
    _exact(tiers, set(FIT_TIER_COUNTS), "Optuna fit tiers")
    normalized: dict[str, list[str]] = {}
    for tier, count in FIT_TIER_COUNTS.items():
        values = [_text(value, f"{tier} task ID") for value in _array(tiers[tier], f"Optuna {tier} tier")]
        if values != sorted(values) or len(values) != count or len(set(values)) != count:
            raise OSWorldRoleLedgerError(f"Optuna {tier} tier is not a sorted exact unique set")
        normalized[tier] = values
    if not set(normalized["discovery"]) < set(normalized["promoted"]) < set(normalized["finalist"]):
        raise OSWorldRoleLedgerError("Optuna fit tiers are not strictly nested")
    validation = [_text(value, "validation task ID") for value in _array(raw["validation_task_ids"], "validation task IDs")]
    if validation != sorted(validation) or len(validation) != ROLE_COUNTS["validation"] or len(set(validation)) != len(validation):
        raise OSWorldRoleLedgerError("Optuna validation tasks are not a sorted exact unique set")
    if set(validation) & set(normalized["finalist"]):
        raise OSWorldRoleLedgerError("Optuna fit and validation tasks overlap")
    if raw["manifest_id"] != _identity(raw, "manifest_id"):
        raise OSWorldRoleLedgerError("Optuna manifest identity does not match its contents")
    if raw["manifest_id"] != verified_receipt.get("optuna_manifest_id") or _hash_json(raw) != verified_receipt.get("optimizer_manifest_sha256"):
        raise OSWorldRoleLedgerError("Optuna manifest does not match the verified build receipt")
    ledger_tasks = [dict(_object(row, "verified ledger task")) for row in _array(verified_ledger.get("tasks"), "verified ledger tasks")]
    expected_fit = {row["task_id"] for row in ledger_tasks if row["role"] == "fit"}
    expected_validation = {row["task_id"] for row in ledger_tasks if row["role"] == "validation"}
    if set(normalized["finalist"]) != expected_fit or set(validation) != expected_validation:
        raise OSWorldRoleLedgerError("Optuna fit or validation membership differs from verified ledger")
    fit_rows = [row for row in ledger_tasks if row["role"] == "fit"]
    seed = _text(verified_ledger.get("selection_seed"), "verified ledger selection seed")
    expected_discovery = _select_family_safe(fit_rows, 24, seed=seed, purpose="fit-discovery")
    expected_promoted = expected_discovery | _select_family_safe(
        _remove(fit_rows, expected_discovery), 72, seed=seed, purpose="fit-promoted-extra"
    )
    if set(normalized["discovery"]) != expected_discovery or set(normalized["promoted"]) != expected_promoted:
        raise OSWorldRoleLedgerError("Optuna tier membership differs from deterministic verified ledger")
    return dict(raw)
