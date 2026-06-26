from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import socket
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, Mapping, Sequence
from uuid import uuid4


class LockHeldError(RuntimeError):
    """Raised when a live or unsafe lock prevents a run-control operation."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    return str(uuid4())


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_line(argv: Sequence[str] | None = None) -> str:
    return " ".join(shlex.quote(part) for part in (argv or sys.argv))


def current_git_commit(project_root: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def current_host() -> str:
    return socket.gethostname()


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class FileLock:
    path: Path
    payload: Mapping[str, Any]
    acquired: bool = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = json.loads(self.path.read_text())
        except Exception:
            self.acquired = False
            return
        if (
            existing.get("owner_pid") == self.payload.get("owner_pid")
            and existing.get("host") == self.payload.get("host")
            and existing.get("started_at") == self.payload.get("started_at")
            and existing.get("run_id") == self.payload.get("run_id")
        ):
            self.path.unlink(missing_ok=True)
        self.acquired = False


@dataclass
class SignalCleanup:
    previous_handlers: Mapping[int, Any]

    def restore(self) -> None:
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)


def lock_payload(
    *,
    run_id: str,
    queue_plan_id: str | None,
    command: str | None = None,
    kind: str = "run",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "owner_pid": os.getpid(),
        "host": current_host(),
        "started_at": now_iso(),
        "command": command or command_line(),
        "run_id": run_id,
    }
    if queue_plan_id is not None:
        payload["queue_plan_id"] = queue_plan_id
    if extra:
        payload.update(dict(extra))
    return payload


def acquire_lock(
    path: Path,
    payload: Mapping[str, Any],
    *,
    force_stale_lock: bool = False,
) -> FileLock:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _create_lock(path, payload)
    except FileExistsError:
        if not force_stale_lock:
            raise LockHeldError(_lock_error_message(path)) from None
        _break_same_host_stale_lock(path)
        return _create_lock(path, payload)


def _create_lock(path: Path, payload: Mapping[str, Any]) -> FileLock:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    return FileLock(path=path, payload=dict(payload))


def _lock_error_message(path: Path) -> str:
    try:
        existing = json.loads(path.read_text())
    except Exception:
        return f"Lock exists and is not valid JSON: {path}"
    return (
        f"Lock exists: {path} "
        f"host={existing.get('host')} pid={existing.get('owner_pid')} "
        f"run_id={existing.get('run_id')} started_at={existing.get('started_at')}"
    )


def _break_same_host_stale_lock(path: Path) -> None:
    try:
        existing = json.loads(path.read_text())
    except Exception as exc:
        raise LockHeldError(f"Cannot prove malformed lock is stale: {path}") from exc

    lock_host = existing.get("host")
    owner_pid = existing.get("owner_pid")
    if lock_host != current_host():
        raise LockHeldError(
            f"Refusing to break cross-host lock {path}: host={lock_host!r} current_host={current_host()!r}"
        )
    if not isinstance(owner_pid, int):
        raise LockHeldError(f"Cannot prove lock owner pid is stale: {path}")
    if pid_is_alive(owner_pid):
        raise LockHeldError(f"Refusing to break live lock {path}: pid={owner_pid}")
    path.unlink()


def install_signal_cleanup(lock: FileLock) -> SignalCleanup:
    previous_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }

    def _handler(signum: int, _frame: FrameType | None) -> None:
        lock.release()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    return SignalCleanup(previous_handlers=previous_handlers)
