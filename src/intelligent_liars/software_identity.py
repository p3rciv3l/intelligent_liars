"""Stable content identities for explicitly enumerated software dependencies."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path


def software_dependency_sha256(
    *, project_root: Path, dependency_paths: Sequence[Path]
) -> str:
    """Hash ordered project-relative names and their exact file contents."""

    root = Path(project_root).resolve(strict=True)
    if not dependency_paths:
        raise ValueError("software dependency inventory must not be empty")
    digest = hashlib.sha256()
    for candidate in dependency_paths:
        path = Path(candidate)
        if not path.is_absolute():
            path = root / path
        if path.is_symlink():
            raise ValueError(f"software dependency must not be a symlink: {path}")
        path = path.resolve(strict=True)
        relative = path.relative_to(root)
        if not path.is_file():
            raise ValueError(f"software dependency is not a regular file: {relative}")
        file_digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()
