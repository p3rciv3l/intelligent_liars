from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_LOCAL = Path(
    "artifacts/activations/activation_all_text_20260624/"
    "extracted_feats_all_text_qwen3-vl-8b-thinking.h5"
)
DEFAULT_REMOTE = "ubuntu@45.250.254.57"
DEFAULT_REMOTE_ROOT = Path("/home/ubuntu/intelligent_liars")
DEFAULT_EXPECTED_SHA256 = "c6e687f69256544121d0718cf8bba142ed5837221976121e1899552dc76f1a5a"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the canonical HDF5 to Latitude as parallel remote chunks."
    )
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size-mib", type=int, default=512)
    parser.add_argument("--expected-sha256", default=DEFAULT_EXPECTED_SHA256)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Upload at most this many missing chunks, then stop before assembly.",
    )
    parser.add_argument("--skip-assemble", action="store_true")
    parser.add_argument("--cleanup-chunks", action="store_true")
    args = parser.parse_args()

    local_path = args.local
    if not local_path.exists():
        raise SystemExit(f"Missing local file: {local_path}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.chunk_size_mib < 1:
        raise SystemExit("--chunk-size-mib must be at least 1")

    remote_final = args.remote_root / local_path
    remote_chunks = remote_final.parent / f".chunks_{remote_final.name}"
    size = local_path.stat().st_size
    chunk_size = args.chunk_size_mib * 1024 * 1024
    chunk_count = math.ceil(size / chunk_size)
    print(
        f"uploading {size} bytes as {chunk_count} chunks of {chunk_size} bytes "
        f"to {args.remote}:{remote_chunks}",
        flush=True,
    )

    _ssh(args.remote, f"mkdir -p {shlex.quote(str(remote_chunks))}")
    remote_sizes = _remote_chunk_sizes(args.remote, remote_chunks)

    missing_jobs = [
        (index, index * chunk_size, min(chunk_size, size - index * chunk_size))
        for index in range(chunk_count)
        if remote_sizes.get(_chunk_name(index), -1) != min(chunk_size, size - index * chunk_size)
    ]
    jobs = missing_jobs
    if args.max_chunks is not None:
        if args.max_chunks < 1:
            raise SystemExit("--max-chunks must be at least 1 when provided")
        jobs = missing_jobs[: args.max_chunks]
        args.skip_assemble = True
    skipped = chunk_count - len(missing_jobs)
    print(f"chunks already complete: {skipped}; chunks to upload: {len(jobs)}", flush=True)

    start = time.monotonic()
    completed_bytes = 0
    for index in range(chunk_count):
        name = _chunk_name(index)
        expected = min(chunk_size, size - index * chunk_size)
        if remote_sizes.get(name) == expected:
            completed_bytes += expected

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(_upload_chunk, args.remote, local_path, remote_chunks, index, offset, length): (
                index,
                length,
            )
            for index, offset, length in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            index, length = future_to_job[future]
            future.result()
            completed_bytes += length
            elapsed = max(time.monotonic() - start, 1e-6)
            rate = completed_bytes / elapsed
            remaining = max(size - completed_bytes, 0)
            print(
                f"chunk {index:06d} complete; "
                f"{completed_bytes / size * 100:.2f}% uploaded; "
                f"rate={rate / 1024**2:.2f} MiB/s; "
                f"eta={remaining / rate / 3600:.2f} h",
                flush=True,
            )

    _verify_remote_chunks(args.remote, remote_chunks, chunk_count, chunk_size, size)
    if args.skip_assemble:
        print("skip assemble requested; chunks are uploaded", flush=True)
        return

    _assemble_remote_file(
        remote=args.remote,
        remote_chunks=remote_chunks,
        remote_final=remote_final,
        chunk_count=chunk_count,
        expected_size=size,
        expected_sha256=args.expected_sha256,
        cleanup_chunks=args.cleanup_chunks,
    )


def _chunk_name(index: int) -> str:
    return f"chunk_{index:06d}"


def _ssh(remote: str, command: str, *, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", remote, command],
        input=stdin,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _remote_chunk_sizes(remote: str, remote_chunks: Path) -> dict[str, int]:
    command = (
        f"if [ -d {shlex.quote(str(remote_chunks))} ]; then "
        f"find {shlex.quote(str(remote_chunks))} -maxdepth 1 -type f -name 'chunk_*' "
        "-printf '%f %s\\n'; fi"
    )
    result = _ssh(remote, command)
    sizes: dict[str, int] = {}
    for line in result.stdout.decode().splitlines():
        name, raw_size = line.rsplit(" ", 1)
        sizes[name] = int(raw_size)
    return sizes


def _upload_chunk(
    remote: str,
    local_path: Path,
    remote_chunks: Path,
    index: int,
    offset: int,
    length: int,
) -> None:
    remote_name = _chunk_name(index)
    remote_path = remote_chunks / remote_name
    remote_tmp = remote_chunks / f".{remote_name}.{os.getpid()}.tmp"
    command = (
        f"cat > {shlex.quote(str(remote_tmp))} && "
        f"test $(stat -c %s {shlex.quote(str(remote_tmp))}) -eq {length} && "
        f"mv -f {shlex.quote(str(remote_tmp))} {shlex.quote(str(remote_path))}"
    )
    with local_path.open("rb") as handle:
        handle.seek(offset)
        process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", remote, command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        remaining = length
        digest = hashlib.sha256()
        while remaining:
            block = handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                process.kill()
                raise RuntimeError(f"Unexpected EOF while reading chunk {index}")
            digest.update(block)
            process.stdin.write(block)
            remaining -= len(block)
        process.stdin.close()
        stdout = process.stdout.read() if process.stdout is not None else b""
        stderr = process.stderr.read() if process.stderr is not None else b""
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"chunk {index} upload failed with code {returncode}: "
            f"stdout={stdout.decode(errors='replace')} stderr={stderr.decode(errors='replace')}"
        )


def _verify_remote_chunks(
    remote: str,
    remote_chunks: Path,
    chunk_count: int,
    chunk_size: int,
    total_size: int,
) -> None:
    sizes = _remote_chunk_sizes(remote, remote_chunks)
    missing: list[str] = []
    wrong: list[str] = []
    for index in range(chunk_count):
        name = _chunk_name(index)
        expected = min(chunk_size, total_size - index * chunk_size)
        actual = sizes.get(name)
        if actual is None:
            missing.append(name)
        elif actual != expected:
            wrong.append(f"{name}: expected {expected}, got {actual}")
    if missing or wrong:
        raise SystemExit(f"remote chunks incomplete; missing={missing[:10]} wrong={wrong[:10]}")
    print("remote chunk sizes verified", flush=True)


def _assemble_remote_file(
    *,
    remote: str,
    remote_chunks: Path,
    remote_final: Path,
    chunk_count: int,
    expected_size: int,
    expected_sha256: str,
    cleanup_chunks: bool,
) -> None:
    final_tmp = remote_final.with_name(f".{remote_final.name}.parallel_tmp")
    cleanup = f"rm -rf {shlex.quote(str(remote_chunks))}" if cleanup_chunks else "true"
    command = f"""
set -euo pipefail
tmp={shlex.quote(str(final_tmp))}
final={shlex.quote(str(remote_final))}
chunks={shlex.quote(str(remote_chunks))}
rm -f "$tmp"
: > "$tmp"
for i in $(seq -f "%06g" 0 {chunk_count - 1}); do
  cat "$chunks/chunk_$i" >> "$tmp"
done
actual_size=$(stat -c %s "$tmp")
test "$actual_size" -eq {expected_size}
actual_sha=$(sha256sum "$tmp" | awk '{{print $1}}')
test "$actual_sha" = {shlex.quote(expected_sha256)}
mv -f "$tmp" "$final"
{cleanup}
printf 'assembled %s bytes sha256=%s\\n' "$actual_size" "$actual_sha"
"""
    print("assembling and verifying remote file", flush=True)
    result = _ssh(remote, command)
    sys.stdout.write(result.stdout.decode())
    if result.stderr:
        sys.stderr.write(result.stderr.decode())


if __name__ == "__main__":
    main()
