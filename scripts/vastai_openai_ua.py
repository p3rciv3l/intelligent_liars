#!/usr/bin/env python3
"""Run the installed Vast CLI with the repository-required HTTP user agent."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


_BOOTSTRAPPED = "INTELLIGENT_LIARS_VAST_WRAPPER_BOOTSTRAPPED"


def _vast_launchers() -> tuple[Path, ...]:
    launchers: list[Path] = []
    seen: set[Path] = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory or ".") / "vastai"
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            launchers.append(candidate)
    return tuple(launchers)


def _launcher_interpreter(vastai: Path) -> str:
    try:
        first_line = vastai.read_text(errors="replace").splitlines()[0]
    except (OSError, IndexError) as error:
        raise ValueError(f"cannot read launcher: {error}") from error
    if not first_line.startswith("#!"):
        raise ValueError("launcher has no Python shebang")
    shebang = shlex.split(first_line[2:].strip())
    if not shebang:
        raise ValueError("launcher has an empty shebang")
    if Path(shebang[0]).name == "env":
        commands = [part for part in shebang[1:] if not part.startswith("-")]
        if not commands:
            raise ValueError("launcher has no shebang command")
        interpreter = shutil.which(commands[0])
        if interpreter is None:
            raise ValueError("launcher Python interpreter is not on PATH")
        return interpreter
    return shebang[0]


def _vast_interpreter() -> str:
    launchers = _vast_launchers()
    if not launchers:
        raise SystemExit(
            "vastai is not installed or is not on PATH; install the Vast CLI first"
        )
    diagnostics: list[str] = []
    for launcher in launchers:
        try:
            interpreter = _launcher_interpreter(launcher)
        except ValueError as error:
            diagnostics.append(f"{launcher}: {error}")
            continue
        probe = subprocess.run(
            [interpreter, "-c", "import vast, requests"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if probe.returncode == 0:
            return interpreter
        diagnostics.append(f"{launcher}: interpreter cannot import vast and requests")
    raise SystemExit("no working Vast CLI launcher found; " + "; ".join(diagnostics))


if os.environ.get(_BOOTSTRAPPED) != "1":
    environment = dict(os.environ)
    environment[_BOOTSTRAPPED] = "1"
    interpreter = _vast_interpreter()
    os.execve(
        interpreter,
        [interpreter, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )

import requests  # noqa: E402


REQUIRED_USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"
_original_request = requests.sessions.Session.request


def _request_with_required_user_agent(
    self: requests.Session,
    method: str,
    url: str,
    **kwargs: object,
) -> object:
    headers = dict(kwargs.pop("headers", None) or {})  # type: ignore[arg-type]
    headers["User-Agent"] = REQUIRED_USER_AGENT
    return _original_request(self, method, url, headers=headers, **kwargs)


requests.sessions.Session.request = _request_with_required_user_agent  # type: ignore[method-assign]

from vast import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
