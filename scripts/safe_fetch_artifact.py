#!/usr/bin/env python3
"""Safely fetch a large artifact over SSH and print DVC promotion steps."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BUFFER_SIZE = 1024 * 1024
SHA256_HEX_LENGTH = 64


class FetchError(RuntimeError):
    """Raised when a safe fetch precondition or verification step fails."""


@dataclass(frozen=True)
class SshSpec:
    target: str
    port: int | None = None

    def command_prefix(self) -> list[str]:
        command = ["ssh"]
        if self.port is not None:
            command.extend(["-p", str(self.port)])
        command.append(self.target)
        return command


@dataclass(frozen=True)
class RemoteMetadata:
    byte_count: int
    sha256: str


def temp_path_for(destination: Path) -> Path:
    """Return the sibling temp path used for a destination."""
    return destination.with_name(f"{destination.name}.tmp")


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError as error:
        raise FetchError(f"could not stat {path}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(BUFFER_SIZE), b""):
                digest.update(chunk)
    except OSError as error:
        raise FetchError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def remove_temp_file(temp_path: Path) -> None:
    try:
        temp_path.unlink()
    except FileNotFoundError:
        return


def atomic_promote(
    temp_path: Path,
    destination: Path,
    *,
    overwrite: bool = False,
) -> None:
    if not temp_path.exists():
        raise FetchError(f"temp file does not exist: {temp_path}")
    if destination.exists() and not overwrite:
        raise FetchError(
            f"destination already exists: {destination}; pass --overwrite to replace"
        )
    try:
        os.replace(temp_path, destination)
    except OSError as error:
        raise FetchError(
            f"could not promote {temp_path} to {destination}: {error}"
        ) from error


def _remote_path_arg(remote_path: str) -> str:
    return shlex.quote(remote_path)


def remote_size_command(remote_path: str) -> str:
    return f"LC_ALL=C stat -c %s -- {_remote_path_arg(remote_path)}"


def remote_sha256_command(remote_path: str) -> str:
    return f"LC_ALL=C sha256sum -- {_remote_path_arg(remote_path)}"


def remote_cat_command(remote_path: str) -> str:
    return f"cat -- {_remote_path_arg(remote_path)}"


def run_ssh_text(ssh_spec: SshSpec, remote_command: str) -> str:
    try:
        completed = subprocess.run(
            [*ssh_spec.command_prefix(), remote_command],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise FetchError(f"could not run ssh: {error}") from error
    if completed.returncode != 0:
        raise FetchError(
            "remote command failed "
            f"({completed.returncode}): {remote_command}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def parse_remote_size(output: str) -> int:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].isdigit():
        raise FetchError(f"could not parse remote byte size from: {output!r}")
    return int(lines[0])


def parse_sha256sum_output(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise FetchError("remote sha256sum returned no output")
    digest = lines[0].split(maxsplit=1)[0].lower()
    valid_hex = all(character in string.hexdigits for character in digest)
    if len(digest) != SHA256_HEX_LENGTH or not valid_hex:
        raise FetchError(f"could not parse sha256 from: {output!r}")
    return digest


def query_remote_metadata(ssh_spec: SshSpec, remote_path: str) -> RemoteMetadata:
    size_output = run_ssh_text(ssh_spec, remote_size_command(remote_path))
    sha_output = run_ssh_text(ssh_spec, remote_sha256_command(remote_path))
    return RemoteMetadata(
        byte_count=parse_remote_size(size_output),
        sha256=parse_sha256sum_output(sha_output),
    )


def prepare_temp_path(temp_path: Path, *, replace_temp: bool) -> None:
    if not temp_path.exists():
        return
    if not replace_temp:
        raise FetchError(
            f"temp file already exists: {temp_path}; "
            "remove it or pass --replace-temp"
        )
    try:
        temp_path.unlink()
    except OSError as error:
        raise FetchError(f"could not remove temp file {temp_path}: {error}") from error


def copy_remote_to_temp(
    ssh_spec: SshSpec,
    remote_path: str,
    temp_path: Path,
) -> None:
    try:
        temp_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FetchError(
            f"could not create destination directory {temp_path.parent}: {error}"
        ) from error
    command = [*ssh_spec.command_prefix(), remote_cat_command(remote_path)]
    try:
        with temp_path.open("xb") as output_file:
            completed = subprocess.run(
                command,
                check=False,
                stdout=output_file,
                stderr=subprocess.PIPE,
            )
    except FileExistsError as error:
        raise FetchError(
            f"temp file already exists: {temp_path}; "
            "remove it or pass --replace-temp"
        ) from error
    except OSError as error:
        raise FetchError(f"could not copy to {temp_path}: {error}") from error
    if completed.returncode != 0:
        raise FetchError(
            "remote copy failed "
            f"({completed.returncode}) while writing {temp_path}\n"
            f"{completed.stderr.strip()}"
        )


def verify_temp_file(temp_path: Path, metadata: RemoteMetadata) -> None:
    actual_size = file_size(temp_path)
    if actual_size != metadata.byte_count:
        raise FetchError(
            "byte-size mismatch: "
            f"remote={metadata.byte_count} local={actual_size}"
        )

    actual_sha256 = sha256_file(temp_path)
    if actual_sha256 != metadata.sha256:
        raise FetchError(
            "sha256 mismatch: "
            f"remote={metadata.sha256} local={actual_sha256}"
        )


def build_ssh_spec(
    *,
    ssh_target: str | None,
    host: str | None,
    user: str | None,
    port: int | None,
) -> SshSpec:
    if ssh_target is not None:
        if user is not None:
            raise FetchError("--user can only be used with --host")
        target = ssh_target
    elif host is not None:
        target = f"{user}@{host}" if user else host
    else:
        raise FetchError("provide either --ssh-target or --host")
    return SshSpec(target=target, port=port)


def cleanup_after_failure(temp_path: Path, *, keep_temp: bool) -> str:
    if keep_temp:
        return f"left temp file: {temp_path}"
    remove_temp_file(temp_path)
    return f"removed temp file: {temp_path}"


def print_fetch_preamble(
    *,
    remote_path: str,
    destination: Path,
    temp_path: Path,
) -> None:
    print(f"remote path: {remote_path}")
    print(f"local path: {destination}")
    print(f"temp path: {temp_path}")


def fetch_artifact(args: argparse.Namespace) -> None:
    destination = args.destination
    temp_path = temp_path_for(destination)
    ssh_spec = build_ssh_spec(
        ssh_target=args.ssh_target,
        host=args.host,
        user=args.user,
        port=args.port,
    )

    print_fetch_preamble(
        remote_path=args.remote_path,
        destination=destination,
        temp_path=temp_path,
    )

    if args.dry_run:
        print("bytes: dry-run (not queried)")
        print("sha256: dry-run (not queried)")
        print("copy mode: ssh cat to a fresh sibling temp file; no resume")
        print("final status: dry-run")
        return

    metadata = query_remote_metadata(ssh_spec, args.remote_path)
    print(f"bytes: {metadata.byte_count}")
    print(f"sha256: {metadata.sha256}")

    try:
        prepare_temp_path(temp_path, replace_temp=args.replace_temp)
        copy_remote_to_temp(ssh_spec, args.remote_path, temp_path)
        verify_temp_file(temp_path, metadata)
        atomic_promote(temp_path, destination, overwrite=args.overwrite)
    except FetchError:
        cleanup_status = cleanup_after_failure(
            temp_path,
            keep_temp=args.keep_temp_on_failure,
        )
        print(cleanup_status, file=sys.stderr)
        raise

    print("final status: ok")
    print()
    print("DVC promotion command, after validating the HDF5:")
    quoted_destination = shlex.quote(str(destination))
    print(f"  {Path(__file__).as_posix()} dvc-flow {quoted_destination}")


def print_dvc_flow(artifact_path: str) -> None:
    quoted_path = shlex.quote(artifact_path)
    quoted_pointer = shlex.quote(f"{artifact_path}.dvc")
    print("DVC promotion flow:")
    print(f"# 1. Validate activation HDF5 first: {quoted_path}")
    print("#    Use the repo's activation/HDF5 validation command before DVC add.")
    print(f'uvx --from "dvc[gdrive]" dvc add {quoted_path}')
    print('uvx --from "dvc[gdrive]" dvc push')
    print(f"git add {quoted_pointer} .gitignore")
    print("# Only .dvc pointers are git-tracked; large HDF5 files remain ignored.")


def positive_port(value: str) -> int:
    port = int(value)
    if port <= 0 or port > 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely fetch a large remote artifact over SSH. The fetch path "
            "always writes a fresh sibling temp file, verifies byte size and "
            "sha256, then atomically promotes the temp file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  scripts/safe_fetch_artifact.py fetch "
            "--ssh-target user@example-host --port 22 "
            "/remote/path/activations.h5 data/activations.h5\n"
            "  scripts/safe_fetch_artifact.py fetch "
            "--host example-host --user user "
            "/remote/path/activations.h5 data/activations.h5\n"
            "  scripts/safe_fetch_artifact.py dvc-flow data/activations.h5\n\n"
            "DVC promotion after validation:\n"
            '  uvx --from "dvc[gdrive]" dvc add <artifact path or dir>\n'
            '  uvx --from "dvc[gdrive]" dvc push\n'
            "  git tracks .dvc pointer files; large HDF5 files stay ignored.\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch and verify a remote artifact without append-resume.",
    )
    target_group = fetch_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--ssh-target",
        "--target",
        dest="ssh_target",
        help="Full target accepted by ssh, such as user@example-host.",
    )
    target_group.add_argument("--host", help="SSH host.")
    fetch_parser.add_argument("--user", help="SSH user, only with --host.")
    fetch_parser.add_argument("--port", type=positive_port, help="SSH port.")
    fetch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned paths and safety mode without contacting SSH.",
    )
    fetch_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow verified temp file to replace an existing destination.",
    )
    fetch_parser.add_argument(
        "--replace-temp",
        action="store_true",
        help="Remove an existing sibling temp file before copying.",
    )
    fetch_parser.add_argument(
        "--keep-temp-on-failure",
        action="store_true",
        help="Leave the temp file in place after copy/verification failure.",
    )
    fetch_parser.add_argument("remote_path", help="Remote artifact path.")
    fetch_parser.add_argument(
        "destination",
        type=Path,
        help="Local final destination path.",
    )
    fetch_parser.set_defaults(func=fetch_artifact)

    dvc_parser = subparsers.add_parser(
        "dvc-flow",
        help="Print the separate DVC promotion flow for a validated artifact.",
    )
    dvc_parser.add_argument("artifact_path", help="Artifact path or directory.")
    dvc_parser.set_defaults(
        func=lambda args: print_dvc_flow(args.artifact_path),
    )
    return parser


def normalize_argv(argv: Sequence[str]) -> list[str]:
    if not argv:
        return []
    known_commands = {"fetch", "dvc-flow", "-h", "--help"}
    if argv[0] in known_commands:
        return list(argv)
    return ["fetch", *argv]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed_argv = normalize_argv(sys.argv[1:] if argv is None else argv)
    try:
        args = parser.parse_args(parsed_argv)
        args.func(args)
    except FetchError as error:
        print(f"final status: failed ({error})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
