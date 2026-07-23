from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import io
import ipaddress
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

from PIL import Image, UnidentifiedImageError

from intelligent_liars.models import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ModelBundle,
    load_model_and_processor,
)

MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGES = 4
MAX_MESSAGES = 128
MAX_TEXT_CHARS = 200_000
MAX_OUTPUT_TOKENS = 32768
FROZEN_TOP_P = 0.9
CLIENT_HISTORY_N = 4
CLIENT_IMAGE_MAX = 4
SERIALIZATION = "single-inference-lock"
MODEL_WORKERS_PER_GPU = 1
MAX_ENCODED_IMAGE_BYTES = ((MAX_IMAGE_BYTES + 2) // 3) * 4
MAX_JSON_ESCAPED_TEXT_BYTES = MAX_TEXT_CHARS * (len(json.dumps("\U0010ffff")) - 2)
MAX_REQUEST_STRUCTURE_BYTES = len(
    json.dumps(
        {
            "model": DEFAULT_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,"},
                            }
                            for _ in range(MAX_IMAGES)
                        ],
                        {"type": "text", "text": ""},
                    ],
                },
                *[
                    {"role": "assistant", "content": ""}
                    for _ in range(MAX_MESSAGES - 1)
                ],
            ],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "top_p": FROZEN_TOP_P,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()
)
MAX_REQUEST_BYTES = (
    MAX_IMAGES * MAX_ENCODED_IMAGE_BYTES
    + MAX_JSON_ESCAPED_TEXT_BYTES
    + MAX_REQUEST_STRUCTURE_BYTES
)
FINGERPRINT_SOURCE = {
    "model": DEFAULT_MODEL_ID,
    "revision": DEFAULT_MODEL_REVISION,
    "processor_revision": DEFAULT_MODEL_REVISION,
    "dtype": "bfloat16",
    "attention": "flash_attention_2",
    "quantization": None,
    "do_sample": False,
    "temperature": 0,
    "top_p": FROZEN_TOP_P,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "client_history_n": CLIENT_HISTORY_N,
    "client_image_max": CLIENT_IMAGE_MAX,
    "serialization": SERIALIZATION,
    "model_workers_per_gpu": MODEL_WORKERS_PER_GPU,
}
MODEL_FINGERPRINT = "sha256:" + hashlib.sha256(
    json.dumps(FINGERPRINT_SOURCE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class RequestError(ValueError):
    def __init__(self, status: HTTPStatus, message: str, *, code: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class EndpointResponse:
    status: int
    body: dict[str, Any]


class QwenEndpoint:
    def __init__(
        self,
        api_key: str,
        *,
        loader: Callable[[], ModelBundle] = load_model_and_processor,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not api_key:
            raise ValueError("QWEN_ENDPOINT_API_KEY must be set")
        self._api_key = api_key
        self._loader = loader
        self._clock = clock
        self._bundle: ModelBundle | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def handle(self, method: str, path: str, headers: Mapping[str, str], body: bytes = b"") -> EndpointResponse:
        try:
            self._authenticate(headers)
            _validate_request_body_size(len(body))
            if method == "GET" and path == "/health":
                return EndpointResponse(HTTPStatus.OK, self._health())
            if method == "GET" and path == "/v1/models":
                return EndpointResponse(HTTPStatus.OK, self._models())
            if method == "POST" and path == "/v1/chat/completions":
                payload = self._parse_json(body)
                return EndpointResponse(HTTPStatus.OK, self._complete(payload))
            raise RequestError(HTTPStatus.NOT_FOUND, "not found", code="not_found")
        except RequestError as exc:
            return EndpointResponse(
                exc.status,
                {"error": {"message": str(exc), "type": exc.code, "code": exc.code}},
            )
        except Exception:
            return EndpointResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": {
                        "message": "internal server error",
                        "type": "server_error",
                        "code": "internal_error",
                    }
                },
            )

    def _authenticate(self, headers: Mapping[str, str]) -> None:
        authorization = next(
            (value for key, value in headers.items() if key.lower() == "authorization"),
            "",
        )
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not token
            or not hmac.compare_digest(token, self._api_key)
        ):
            raise RequestError(
                HTTPStatus.UNAUTHORIZED,
                "authentication required",
                code="authentication_error",
            )

    @staticmethod
    def _parse_json(body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
        return payload

    def _health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": DEFAULT_MODEL_ID,
            "revision": DEFAULT_MODEL_REVISION,
            "fingerprint": MODEL_FINGERPRINT,
            "loaded": self.loaded,
        }

    @staticmethod
    def _models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": DEFAULT_MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "Qwen",
                    "revision": DEFAULT_MODEL_REVISION,
                    "fingerprint": MODEL_FINGERPRINT,
                }
            ],
        }

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages, images = _validate_and_convert_payload(payload)
        max_tokens = payload.get("max_tokens", MAX_OUTPUT_TOKENS)
        bundle = self._get_bundle()
        rendered = bundle.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [rendered],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        inputs = bundle.processor(**processor_kwargs)
        model = bundle.model
        if model is None:
            raise RuntimeError("model loader returned no model")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
        }
        with self._inference_lock:
            generated = model.generate(**inputs, **generation_kwargs)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated_tokens = [output[prompt_tokens:] for output in generated]
        decoded = bundle.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        completion_tokens = len(generated_tokens[0])
        created = int(self._clock())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": DEFAULT_MODEL_ID,
            "system_fingerprint": MODEL_FINGERPRINT,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": decoded},
                    "finish_reason": "length" if completion_tokens >= max_tokens else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _get_bundle(self) -> ModelBundle:
        if self._bundle is None:
            with self._load_lock:
                if self._bundle is None:
                    self._bundle = self._loader()
        return self._bundle


def _validate_and_convert_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "stream"}
    unsupported = sorted(set(payload) - allowed)
    if unsupported:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"unsupported fields: {', '.join(unsupported)}")
    if payload.get("model") != DEFAULT_MODEL_ID:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"model must be {DEFAULT_MODEL_ID}")
    if payload.get("stream", False) is not False:
        raise RequestError(HTTPStatus.BAD_REQUEST, "streaming is not supported")
    temperature = payload.get("temperature", 0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature != 0:
        raise RequestError(HTTPStatus.BAD_REQUEST, "temperature must be 0")
    top_p = payload.get("top_p", FROZEN_TOP_P)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or top_p != FROZEN_TOP_P:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"top_p must be {FROZEN_TOP_P}")
    max_tokens = payload.get("max_tokens", MAX_OUTPUT_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            f"max_tokens must be an integer from 1 to {MAX_OUTPUT_TOKENS}",
        )
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not 1 <= len(raw_messages) <= MAX_MESSAGES:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"messages must contain 1 to {MAX_MESSAGES} items")

    messages: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    text_chars = 0
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict) or set(raw_message) != {"role", "content"}:
            raise RequestError(HTTPStatus.BAD_REQUEST, "each message must contain only role and content")
        role = raw_message["role"]
        if role not in {"system", "user", "assistant"}:
            raise RequestError(HTTPStatus.BAD_REQUEST, "message role is invalid")
        content = raw_message["content"]
        if isinstance(content, str):
            text_chars += len(content)
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list) or not content:
            raise RequestError(HTTPStatus.BAD_REQUEST, "message content must be text or a non-empty parts list")
        converted_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise RequestError(HTTPStatus.BAD_REQUEST, "message parts must be objects")
            if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
                text_chars += len(part["text"])
                converted_parts.append(part)
            elif part.get("type") == "image_url" and set(part) == {"type", "image_url"}:
                image = _decode_image_url(part["image_url"])
                images.append(image)
                converted_parts.append({"type": "image", "image": image})
            else:
                raise RequestError(HTTPStatus.BAD_REQUEST, "only text and base64 image_url parts are supported")
        messages.append({"role": role, "content": converted_parts})
    if text_chars > MAX_TEXT_CHARS:
        raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "message text is too large")
    if len(images) > MAX_IMAGES:
        raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"at most {MAX_IMAGES} images are supported")
    return messages, images


def _decode_image_url(value: Any) -> Image.Image:
    if isinstance(value, dict):
        value = value.get("url") if set(value) == {"url"} else None
    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise RequestError(HTTPStatus.BAD_REQUEST, "image_url must be a base64 image data URL")
    metadata, separator, encoded = value.partition(",")
    if not separator or not metadata.endswith(";base64"):
        raise RequestError(HTTPStatus.BAD_REQUEST, "image_url must use base64 encoding")
    if len(encoded) > MAX_ENCODED_IMAGE_BYTES:
        raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "image is too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, "image_url contains invalid base64") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "image is too large")
    try:
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "image dimensions are too large")
        image.load()
    except RequestError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, "image_url does not contain a valid image") from exc
    return image.convert("RGB")


def _validate_request_body_size(length: int) -> None:
    if length < 0 or length > MAX_REQUEST_BYTES:
        raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")


def make_handler(endpoint: QwenEndpoint) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._dispatch()

        def do_POST(self) -> None:
            self._dispatch()

        def _dispatch(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = MAX_REQUEST_BYTES + 1
            try:
                _validate_request_body_size(length)
            except RequestError:
                response = EndpointResponse(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": {"message": "request body is too large", "type": "invalid_request_error"}},
                )
            else:
                response = endpoint.handle(
                    self.command,
                    self.path.split("?", 1)[0],
                    dict(self.headers.items()),
                    self.rfile.read(length),
                )
            encoded = json.dumps(response.body, separators=(",", ":")).encode()
            self.send_response(response.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return Handler


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the pinned Qwen3-VL OpenAI-compatible endpoint.")
    parser.add_argument("--host", default=os.getenv("QWEN_ENDPOINT_HOST") or "127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("QWEN_ENDPOINT_PORT") or "8000"))
    parser.add_argument(
        "--unsafe-allow-public-bind",
        action="store_true",
        help="Explicitly allow a non-loopback bind; use only with a secured transport.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not _is_loopback_host(args.host) and not args.unsafe_allow_public_bind:
        parser.error("non-loopback bind requires --unsafe-allow-public-bind and a secured transport")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "host": args.host,
                    "port": args.port,
                    "public_bind": not _is_loopback_host(args.host),
                    "environment_names": ["QWEN_ENDPOINT_API_KEY", "HF_HOME"],
                }
            )
        )
        return
    api_key = os.getenv("QWEN_ENDPOINT_API_KEY", "")
    endpoint = QwenEndpoint(api_key)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(endpoint))
    server.serve_forever()


if __name__ == "__main__":
    main()
