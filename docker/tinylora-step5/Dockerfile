# syntax=docker/dockerfile:1.7

# This exact base was used by the successful RTX 3090 qualification. The tag
# is kept for readability; the digest is the identity Docker actually pulls.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel@sha256:e8e63dd7baca894ba11fe1ba48a52a550793c8974f89b533d697784dd20a4dc0
FROM ${BASE_IMAGE}

ARG IMAGE_REVISION=uncommitted

LABEL org.opencontainers.image.title="tinylora-step5" \
      org.opencontainers.image.description="Pinned RTX runtime for bounded TinyLoRA Step 5 screening" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      ai.intelligent-liars.runtime-manifest="/opt/tinylora/runtime-manifest.json"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/cache/huggingface \
    HF_HUB_CACHE=/workspace/cache/huggingface/hub \
    HF_ASSETS_CACHE=/workspace/cache/huggingface/assets \
    TORCH_HOME=/workspace/cache/torch \
    XDG_CACHE_HOME=/workspace/cache/xdg \
    TINYLORA_RUNTIME_MANIFEST=/opt/tinylora/runtime-manifest.json

WORKDIR /opt/tinylora

COPY requirements.in requirements.lock /opt/tinylora/

# Every registry artifact is hash-pinned. Torch and CUDA are supplied by the
# digest-pinned base; the lock deliberately excludes torch so pip cannot replace
# the CUDA build with a PyPI CPU wheel.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --require-hashes -r /opt/tinylora/requirements.lock

# FlashAttention is installed from the upstream prebuilt cp311/cu12/torch2.5
# wheel. This avoids a per-host source build and rejects an altered asset before
# installation. The cxx11abiFALSE wheel matches the qualified PyTorch base.
ADD --checksum=sha256:1c455db7a1e5d58e9d43370ecfc75a3a2eb1e37ec12353db66d12c23dbbc3ac6 \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    /opt/tinylora/wheels/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

RUN python -m pip install --no-deps \
      /opt/tinylora/wheels/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && python -m pip check \
    && mkdir -p /workspace/cache/huggingface/hub /workspace/cache/torch /workspace/cache/xdg

COPY Dockerfile runtime-manifest.json validate_runtime.py write_sbom.py /opt/tinylora/
COPY third_party/ /opt/tinylora/third_party/

RUN python /opt/tinylora/write_sbom.py \
      --runtime-id tinylora-step5-cu121-torch251-fa283post1-r2 \
      --output-dir /opt/tinylora/sbom

# Image construction can validate metadata and packages, but a real RTX host is
# still required for the CUDA/FlashAttention kernel check.
RUN python /opt/tinylora/validate_runtime.py --metadata-only

WORKDIR /workspace/intelligent_liars
