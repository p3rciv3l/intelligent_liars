"""Frozen Qwen backend shared by base-known qualification lanes.

The backend is deliberately narrow: text-only prompts, greedy decoding, one
verified model load per process, and immutable execution identities.  Stored
and synthetic qualification backends remain in ``truth_editing_base_known``;
this is the only backend permitted to claim ``verified_frozen_qwen`` evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .models import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_CONTENT_SHA256,
    DEFAULT_MODEL_REVISION,
    DEFAULT_SNAPSHOT_MANIFEST_SHA256,
    QWEN_ATTENTION_IMPLEMENTATION,
    QWEN_DEVICE_MAP,
    ModelBundle,
    ModelLoadConfig,
    load_model_and_processor,
)
from .truth_editing_base_known import (
    BaseKnownError,
    FrozenBaseIdentity,
    QualificationRequest,
    QualificationResponse,
    VerifiedFrozenQwenEvidence,
)

BACKEND_FORMAT = "truth_editing_frozen_qwen_backend_v2"
BATCH_EXECUTION_FORMAT = "truth_editing_frozen_qwen_batch_execution_v2"
EXECUTION_RECEIPT_FORMAT = "frozen_qwen_execution_receipt_v2"
_THINKING_PROMPT_SUFFIX = "<|im_start|>assistant\n<think>\n"
_EMPTY_THINK_PREFILL = "</think>\n\n"
FROZEN_TOKENIZER_SHA256 = "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
FROZEN_CHAT_TEMPLATE_SHA256 = "36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956"
FROZEN_CHAT_TEMPLATE_FILE_SHA256 = (
    "7dc0b863a3cf9320e063574e0adb1382a22a089f1eeb8bfbe33cc69c5c2cc1e5"
)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise BaseKnownError("Qwen qualification evidence is not canonical JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _single_token_option_ids(tokenizer: Any, labels: Sequence[str]) -> tuple[int, ...]:
    """Resolve the request's option labels without relaxing response parsing."""

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise BaseKnownError("Qwen tokenizer cannot encode option labels")
    token_ids: list[int] = []
    for label in labels:
        encoded = encode(label, add_special_tokens=False)
        if (
            not isinstance(encoded, Sequence)
            or isinstance(encoded, (str, bytes))
            or len(encoded) != 1
            or not isinstance(encoded[0], int)
            or encoded[0] < 0
        ):
            raise BaseKnownError("each option label must map to a single tokenizer token")
        token_ids.append(encoded[0])
    if len(set(token_ids)) != len(token_ids):
        raise BaseKnownError("option labels must map to distinct tokenizer tokens")
    return tuple(token_ids)


class FrozenQwenQualificationBackend:
    """Batched Transformers inference over one exactly verified Qwen bundle."""

    def __init__(
        self,
        *,
        model_config: ModelLoadConfig | None = None,
        bundle_loader: Callable[[ModelLoadConfig], ModelBundle] = load_model_and_processor,
        enforce_production_runtime: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        execution_receipt_path: Path | None = None,
    ) -> None:
        self._config = model_config or ModelLoadConfig()
        if enforce_production_runtime and (
            self._config.expected_model_sha256 != DEFAULT_MODEL_CONTENT_SHA256
            or self._config.expected_snapshot_manifest_sha256
            != DEFAULT_SNAPSHOT_MANIFEST_SHA256
        ):
            raise BaseKnownError(
                "production Qwen evidence requires the exact pinned snapshot identities"
            )
        if enforce_production_runtime and bundle_loader is not load_model_and_processor:
            raise BaseKnownError(
                "production Qwen evidence requires the verified model loader"
            )
        self._loader = bundle_loader
        self._enforce_production_runtime = enforce_production_runtime
        self._clock = clock
        self._execution_receipt_path = (
            Path(execution_receipt_path) if execution_receipt_path is not None else None
        )
        self._bundle: ModelBundle | None = None
        self._bound_identity: FrozenBaseIdentity | None = None
        self._execution_by_request_set: dict[str, dict[str, Any]] = {}

    def evidence_receipt(
        self, identity: FrozenBaseIdentity
    ) -> VerifiedFrozenQwenEvidence | None:
        """Verify the loaded model before returning production evidence identity."""

        if not self._enforce_production_runtime:
            return None
        self._bind_identity(identity)
        self._bundle_once(identity)
        runtime_identity = {
            "format": BACKEND_FORMAT,
            "generation": {
                "batch_scheduled": True,
                "do_sample": False,
                "num_beams": 1,
                "temperature": 0.0,
                "use_cache": True,
                "prompt": "chat_template_user_message_add_generation_prompt",
                "enable_thinking": False,
                "assistant_prefill": "empty_think",
                "output": "single_option_label_then_eos",
                "option_constraint": "prefix_allowed_tokens_fn",
            },
        }
        path = self._execution_receipt_path
        if path is None:
            raise BaseKnownError("production Qwen backend requires an execution receipt path")
        unsigned = {
            "format": EXECUTION_RECEIPT_FORMAT,
            "model": identity.to_payload(),
            "snapshot_manifest_sha256": self._config.expected_snapshot_manifest_sha256,
            "runtime_identity_sha256": _hash(runtime_identity),
            "software_sha256": _software_identity_sha256(),
        }
        payload = dict(unsigned)
        payload["self_sha256"] = _hash(unsigned)
        encoded = _canonical(payload) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_first_write(path, encoded)
        return VerifiedFrozenQwenEvidence.open(path, identity)

    def generate_batch(
        self,
        requests: Sequence[QualificationRequest],
        identity: FrozenBaseIdentity,
    ) -> Sequence[QualificationResponse]:
        self._bind_identity(identity)
        if not requests:
            return ()
        max_tokens = {request.max_new_tokens for request in requests}
        if len(max_tokens) != 1:
            raise BaseKnownError("Qwen batch requests must share max_new_tokens")
        request_ids = tuple(request.request_id for request in requests)
        if len(set(request_ids)) != len(request_ids):
            raise BaseKnownError("Qwen batch request IDs must be unique")

        bundle = self._bundle_once(identity)
        model = bundle.model
        assert model is not None
        texts = []
        for request in requests:
            rendered = bundle.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": request.prompt}],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not isinstance(rendered, str) or not rendered.endswith(
                _THINKING_PROMPT_SUFFIX
            ):
                raise BaseKnownError(
                    "Qwen chat template has an unexpected thinking prompt suffix"
                )
            texts.append(rendered + _EMPTY_THINK_PREFILL)
        inputs = bundle.processor(text=texts, padding=True, return_tensors="pt")
        if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
            raise BaseKnownError("Qwen processor did not return batched input_ids")
        device = _model_device(model)
        moved = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_ids = moved["input_ids"]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise BaseKnownError("Qwen processor input_ids must be a rank-2 tensor")
        tokenizer = bundle.tokenizer
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise BaseKnownError("Qwen tokenizer requires one non-negative EOS token")
        allowed_by_request: tuple[tuple[int, ...], ...] = tuple(
            _single_token_option_ids(tokenizer, request.labels) for request in requests
        )
        prompt_width = int(input_ids.shape[1])

        def allowed_tokens(batch_id: int, sequence: torch.Tensor) -> list[int]:
            if batch_id < 0 or batch_id >= len(allowed_by_request):
                raise BaseKnownError("Qwen constrained decoder batch index is invalid")
            if int(sequence.shape[-1]) == prompt_width:
                return list(allowed_by_request[batch_id])
            return [eos_token_id]

        _reset_cuda_peak(device)
        started = self._clock()
        with torch.inference_mode():
            generated = model.generate(
                **moved,
                max_new_tokens=next(iter(max_tokens)),
                do_sample=False,
                num_beams=1,
                temperature=0.0,
                use_cache=True,
                prefix_allowed_tokens_fn=allowed_tokens,
            )
        elapsed = self._clock() - started
        if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
            raise BaseKnownError("Qwen generation did not return rank-2 token IDs")
        if generated.shape[0] != len(requests) or generated.shape[1] < input_ids.shape[1]:
            raise BaseKnownError("Qwen generation shape differs from the request batch")
        suffix = generated[:, input_ids.shape[1] :]
        token_ids = tuple(
            _trim_tokens(
                suffix[index],
                eos_token_id=getattr(bundle.tokenizer, "eos_token_id", None),
                pad_token_id=getattr(bundle.tokenizer, "pad_token_id", None),
            )
            for index in range(suffix.shape[0])
        )
        decoded = bundle.processor.batch_decode(
            [list(values) for values in token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)) or len(decoded) != len(requests):
            raise BaseKnownError("Qwen decoded response count differs from requests")
        responses = tuple(
            QualificationResponse(request.request_id, str(decoded[index]), token_ids[index])
            for index, request in enumerate(requests)
        )
        request_set_sha = _hash([request.to_payload() for request in requests])
        generated_count = sum(len(values) for values in token_ids)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise BaseKnownError("Qwen runtime clock returned an invalid duration")
        evidence = self.evidence_receipt(identity) if self._enforce_production_runtime else None
        if self._enforce_production_runtime and evidence is None:
            raise BaseKnownError("production Qwen execution lacks verified evidence")
        unsigned_execution = {
            "format": BATCH_EXECUTION_FORMAT,
            "backend_receipt_sha256": (
                evidence.receipt_sha256
                if evidence is not None
                else "0" * 64
            ),
            "request_set_sha256": request_set_sha,
            "response_set_sha256": _hash(
                [response.to_payload() for response in responses]
            ),
            "request_count": len(requests),
            "prompt_token_count": _prompt_token_count(moved),
            "generated_token_count": generated_count,
            "elapsed_seconds": elapsed,
            "generated_tokens_per_second": generated_count / elapsed if elapsed > 0 else None,
            "cuda_peak_allocated_bytes": _cuda_peak_bytes(device),
        }
        execution = dict(unsigned_execution)
        execution["self_sha256"] = _hash(unsigned_execution)
        self._execution_by_request_set[request_set_sha] = execution
        return responses

    def batch_execution_receipt(
        self,
        requests: Sequence[QualificationRequest],
        responses: Sequence[QualificationResponse],
        identity: FrozenBaseIdentity,
    ) -> Mapping[str, Any] | None:
        """Return telemetry bound to the exact generated requests and responses."""

        self._bind_identity(identity)
        if not self._enforce_production_runtime:
            return None
        if tuple(response.request_id for response in responses) != tuple(
            request.request_id for request in requests
        ):
            raise BaseKnownError("Qwen execution receipt response order differs")
        request_set_sha = _hash([request.to_payload() for request in requests])
        try:
            receipt = dict(self._execution_by_request_set[request_set_sha])
        except KeyError as error:
            raise BaseKnownError("Qwen execution receipt is unavailable for this batch") from error
        if receipt["response_set_sha256"] != _hash(
            [response.to_payload() for response in responses]
        ):
            raise BaseKnownError("Qwen execution receipt response payload differs")
        return receipt

    def _bind_identity(self, identity: FrozenBaseIdentity) -> None:
        if self._bound_identity is not None and identity != self._bound_identity:
            raise BaseKnownError("Qwen backend cannot change frozen model identity")
        if identity.model_sha256 != self._config.expected_model_sha256:
            raise BaseKnownError("Qwen configured model SHA differs from qualification identity")
        if self._enforce_production_runtime and (
            identity.tokenizer_sha256 != FROZEN_TOKENIZER_SHA256
            or identity.chat_template_sha256 != FROZEN_CHAT_TEMPLATE_SHA256
        ):
            raise BaseKnownError(
                "Qwen tokenizer or chat-template identity differs from the project pin"
            )
        manifest_path = self._config.snapshot_manifest_path
        if manifest_path is None or self._config.cache_dir is None:
            raise BaseKnownError("Qwen production qualification requires a verified cache manifest")
        try:
            manifest = json.loads(Path(manifest_path).read_text())
            files = manifest["files"]
            by_path = {item["path"]: item["sha256"] for item in files}
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise BaseKnownError("Qwen cache manifest file inventory is unreadable") from error
        if self._enforce_production_runtime:
            if by_path.get("tokenizer.json") != identity.tokenizer_sha256:
                raise BaseKnownError("Qwen cache tokenizer file identity differs")
            if by_path.get("chat_template.json") != FROZEN_CHAT_TEMPLATE_FILE_SHA256:
                raise BaseKnownError("Qwen cache chat-template file identity differs")
        elif "tokenizer.json" not in by_path or "chat_template.json" not in by_path:
            raise BaseKnownError("Qwen cache lacks tokenizer or chat-template files")
        self._bound_identity = identity

    def _bundle_once(self, identity: FrozenBaseIdentity) -> ModelBundle:
        if self._bundle is None:
            bundle = self._loader(self._config)
            if bundle.model is None:
                raise BaseKnownError("Qwen loader returned no model")
            expected_snapshot = {
                "model_id": DEFAULT_MODEL_ID,
                "revision": DEFAULT_MODEL_REVISION,
                "model_sha256": identity.model_sha256,
                "snapshot_manifest_sha256": self._config.expected_snapshot_manifest_sha256,
            }
            if bundle.model_id != DEFAULT_MODEL_ID or bundle.model_revision != DEFAULT_MODEL_REVISION:
                raise BaseKnownError("loaded Qwen checkpoint identity differs")
            if bundle.verified_snapshot != expected_snapshot:
                raise BaseKnownError("loaded snapshot identity differs from frozen Qwen")
            template = getattr(bundle.processor, "chat_template", None) or getattr(
                bundle.tokenizer, "chat_template", None
            )
            if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != identity.chat_template_sha256:
                raise BaseKnownError("loaded Qwen chat template identity differs")
            if self._enforce_production_runtime:
                _verify_production_runtime(bundle.model)
            bundle.model.eval()
            self._bundle = bundle
        return self._bundle


def _verify_production_runtime(model: Any) -> None:
    try:
        parameters = tuple(model.parameters())
    except (AttributeError, TypeError) as error:
        raise BaseKnownError("loaded Qwen model exposes no parameters") from error
    if not parameters:
        raise BaseKnownError("loaded Qwen model exposes no parameters")
    devices = {str(parameter.device) for parameter in parameters}
    floating_dtypes = {
        parameter.dtype for parameter in parameters if torch.is_floating_point(parameter)
    }
    implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if devices != {QWEN_DEVICE_MAP}:
        raise BaseKnownError(f"production Qwen parameters must all be on {QWEN_DEVICE_MAP}")
    if floating_dtypes != {torch.bfloat16}:
        raise BaseKnownError("production Qwen floating parameters must be BF16")
    if implementation != QWEN_ATTENTION_IMPLEMENTATION:
        raise BaseKnownError("production Qwen must use FlashAttention 2")


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise BaseKnownError("cannot determine Qwen model device") from error


def _trim_tokens(
    values: torch.Tensor, *, eos_token_id: int | None, pad_token_id: int | None
) -> tuple[int, ...]:
    output: list[int] = []
    for raw in values.detach().cpu().tolist():
        token_id = int(raw)
        if pad_token_id is not None and token_id == int(pad_token_id):
            break
        output.append(token_id)
        if eos_token_id is not None and token_id == int(eos_token_id):
            break
    return tuple(output)


def _prompt_token_count(inputs: Mapping[str, Any]) -> int:
    attention = inputs.get("attention_mask")
    if isinstance(attention, torch.Tensor):
        return int(attention.detach().sum().item())
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise BaseKnownError("Qwen inputs lack token-count evidence")
    return int(input_ids.numel())


def _reset_cuda_peak(device: Any) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_bytes(device: Any) -> int | None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return int(torch.cuda.max_memory_allocated(device))
    return None


def _atomic_first_write(path: Path, content: bytes) -> None:
    """Publish immutable bytes without allowing concurrent replacement."""

    if path.exists():
        if path.read_bytes() != content:
            raise BaseKnownError("existing frozen Qwen execution receipt differs")
        return
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{time.time_ns()}")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except FileExistsError:
            if path.read_bytes() != content:
                raise BaseKnownError("concurrent frozen Qwen execution receipt differs")
    finally:
        temporary.unlink(missing_ok=True)


def _software_identity_sha256() -> str:
    """Bind the receipt to the local verification chain and ML runtimes."""

    source_root = Path(__file__).parent
    sources = {}
    for name in (
        "model_cache.py",
        "models.py",
        "truth_editing_base_known.py",
        "truth_editing_qwen_qualification.py",
    ):
        sources[name] = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
    return _hash(
        {
            "sources": sources,
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
        }
    )
