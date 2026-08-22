# TinyLoRA Step 5 RTX runtime

This directory defines a reusable, immutable CUDA 12.1 runtime for the bounded
Step 5 candidate screen. It reproduces the software stack qualified on an RTX
3090 while keeping `large_run_enabled` false. It does not embed model weights,
credentials, experiment data, or checkpoints.

## Why this avoids slow and fragile startup

- The PyTorch/CUDA base is selected by digest, not by a mutable tag.
- Python dependencies are transitively locked with hashes.
- The 256 MB upstream FlashAttention wheel is downloaded and verified while the
  image is built. A rented machine never compiles FlashAttention.
- Hugging Face and Torch caches live below `/workspace/cache`. A launcher should
  attach or restore a durable cache there, key it by exact model revision, and
  reject the host before model download if the cache has less than 80 GiB free.
- The full validator exercises BF16 and an actual FlashAttention CUDA kernel.
  An image build only runs metadata validation because it has no GPU.
- The image carries FlashAttention's BSD-3-Clause notice plus a deterministic
  SPDX 2.3 inventory and license inventory for every installed Python package.

The model cache is intentionally separate from the runtime image. This keeps the
image reusable across model revisions and allows a cached model snapshot to be
checksummed, restored, and updated independently. A run must pin the Hugging Face
model revision; `main` is not an acceptable revision.

## Build, publish, and qualify

No build or push is performed merely by adding these files. The manual
`Publish TinyLoRA Step 5 runtime` GitHub Actions workflow is the supported
publication path. It uses only the workflow's automatic `GITHUB_TOKEN`, with
read-only source access and package-write access. The workflow publishes:

```text
ghcr.io/p3rciv3l/intelligent_liars/tinylora-step5:sha-<full-source-commit>
```

It also records the registry digest, attaches BuildKit SBOM and provenance
attestations, and uploads a checksummed evidence bundle to the workflow run.
Select the intended source commit in GitHub, dispatch the workflow, and use the
resulting digest-qualified reference from `image-reference.txt`. Do not treat a
successful CPU-only image build as GPU qualification, and do not use the tag by
itself for a rented host.

For a local build that is not pushed:

```bash
docker build \
  --build-arg IMAGE_REVISION="$(git rev-parse HEAD)" \
  --tag tinylora-step5:cu121-torch251-fa283post1-r2 \
  docker/tinylora-step5
```

Push it to the chosen registry, resolve the resulting image digest, and give Vast
the published `repository@sha256:...` reference. Do not launch from the local tag.
On the first candidate host, before downloading a model or starting training:

```bash
python /opt/tinylora/validate_runtime.py
```

The host is eligible only if this exits zero. The run record should capture the
published image digest, runtime-manifest checksum, actual NVIDIA driver, GPU,
cache path/free space, and validator JSON.

## Updating the lock

Edit `requirements.in`, then regenerate the lock on macOS or Linux with:

```bash
./compile_lock.sh
```

Update both build-input checksums in `runtime-manifest.json`. Package changes,
base-digest changes, and FlashAttention-wheel changes each require a new runtime
ID and a fresh single-host GPU qualification before any parallel screen.

The Docker context is a positive allowlist in `.dockerignore`: only the
Dockerfile, dependency input/lock, runtime/SBOM tools, manifest, and explicit
third-party notice can enter a build.
This fails closed if a developer later places an `.env`, token, repository
config, dataset, checkpoint, or cache inside this directory.
