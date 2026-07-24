from __future__ import annotations

import base64
import http.client
import io
import json
import threading
import tomllib
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from intelligent_liars.models import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ModelBundle,
    ModelLoadConfig,
)
from intelligent_liars.qwen_endpoint import (
    CLIENT_HISTORY_N,
    CLIENT_IMAGE_MAX,
    FINGERPRINT_SOURCE,
    FROZEN_TOP_P,
    MAX_ENCODED_IMAGE_BYTES,
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    MAX_MESSAGES,
    MAX_OUTPUT_TOKENS,
    MAX_REQUEST_BYTES,
    MAX_TEXT_CHARS,
    MODEL_FINGERPRINT,
    QwenEndpoint,
    RequestError,
    _validate_request_body_size,
    main,
    make_handler,
)

API_KEY = "unit-test-key-that-must-stay-private"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


class FakeInputs(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(input_ids=np.array([[10, 11, 12]]))
        self.to_device: object | None = None

    def to(self, device: object) -> FakeInputs:
        self.to_device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.template_messages: list[dict[str, Any]] | None = None
        self.template_kwargs: dict[str, Any] | None = None
        self.processor_kwargs: dict[str, Any] | None = None
        self.decoded_ids: list[Any] | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.template_messages = messages
        self.template_kwargs = kwargs
        return "rendered prompt"

    def __call__(self, **kwargs: Any) -> FakeInputs:
        self.processor_kwargs = kwargs
        return FakeInputs()

    def batch_decode(self, ids: list[Any], **kwargs: Any) -> list[str]:
        self.decoded_ids = ids
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return ["<think>inspect carefully</think>\n<final>click(1, 2)</final>"]


class FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.generate_kwargs = kwargs
        return np.array([[10, 11, 12, 20, 21, 22, 23]])


def make_endpoint() -> tuple[QwenEndpoint, FakeProcessor, FakeModel, list[int]]:
    processor = FakeProcessor()
    model = FakeModel()
    loads: list[int] = []

    def loader() -> ModelBundle:
        loads.append(1)
        return ModelBundle(
            model=model,
            processor=processor,
            tokenizer=object(),
            model_id=DEFAULT_MODEL_ID,
            config=ModelLoadConfig(),
        )

    return QwenEndpoint(API_KEY, loader=loader, clock=lambda: 1234), processor, model, loads


def request(endpoint: QwenEndpoint, payload: dict[str, Any], headers: dict[str, str] | None = None):
    return endpoint.handle(
        "POST",
        "/v1/chat/completions",
        AUTH if headers is None else headers,
        json.dumps(payload).encode(),
    )


def basic_payload(**updates: Any) -> dict[str, Any]:
    payload = {
        "model": DEFAULT_MODEL_ID,
        "messages": [{"role": "user", "content": "inspect the desktop"}],
        "temperature": 0,
        "top_p": FROZEN_TOP_P,
        "max_tokens": 12,
    }
    payload.update(updates)
    return payload


def png_data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (2, 3), (1, 2, 3)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def padded_png_data_url(size: int) -> str:
    output = io.BytesIO()
    Image.new("RGB", (2, 3), (1, 2, 3)).save(output, format="PNG")
    raw = output.getvalue()
    assert len(raw) <= size
    return "data:image/png;base64," + base64.b64encode(raw.ljust(size, b"\0")).decode()


def maximum_openai_request_body() -> bytes:
    return json.dumps(
        {
            "model": DEFAULT_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": padded_png_data_url(MAX_IMAGE_BYTES)},
                            }
                            for _ in range(MAX_IMAGES)
                        ],
                        {"type": "text", "text": "\U0010ffff" * MAX_TEXT_CHARS},
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


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": API_KEY},
        {"Authorization": "Basic abc"},
    ],
)
def test_auth_is_required_for_every_route_and_never_loads_model(headers):
    endpoint, _, _, loads = make_endpoint()

    for method, path in [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
    ]:
        response = endpoint.handle(method, path, headers, b"{}")
        assert response.status == 401
        assert response.body["error"]["code"] == "authentication_error"
    assert loads == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"model": DEFAULT_MODEL_ID}, "messages"),
        (basic_payload(model="other/model"), "model must be"),
        (basic_payload(temperature=0.1), "temperature must be 0"),
        (basic_payload(top_p=0.8), "top_p must be 0.9"),
        (basic_payload(top_p=True), "top_p must be 0.9"),
        (basic_payload(max_tokens=0), "max_tokens"),
        (basic_payload(max_tokens=32769), "max_tokens"),
        (basic_payload(stream=True), "streaming"),
        (basic_payload(messages=[{"role": "tool", "content": "x"}]), "role"),
        (basic_payload(extra=True), "unsupported fields"),
        (
            basic_payload(messages=[{"role": "user", "content": [{"type": "audio", "audio": "x"}]}]),
            "only text and base64",
        ),
    ],
)
def test_malformed_payloads_are_rejected_without_loading(payload, message):
    endpoint, _, _, loads = make_endpoint()

    response = request(endpoint, payload)

    assert response.status == 400
    assert message in response.body["error"]["message"]
    assert loads == []


def test_invalid_json_is_rejected():
    endpoint, _, _, loads = make_endpoint()

    response = endpoint.handle("POST", "/v1/chat/completions", AUTH, b"{")

    assert response.status == 400
    assert loads == []


def test_maximum_five_image_openai_request_shape_fits_derived_body_limit():
    body = maximum_openai_request_body()

    assert len(body) == MAX_REQUEST_BYTES
    assert MAX_ENCODED_IMAGE_BYTES == 8 * 1024 * 1024
    _validate_request_body_size(len(body))
    with pytest.raises(RequestError) as exc_info:
        _validate_request_body_size(len(body) + 1)
    assert exc_info.value.status == 413


def test_real_http_boundary_accepts_runtime_sized_multimodal_body_and_rejects_over_cap():
    endpoint, _, _, _ = make_endpoint()
    image_url = padded_png_data_url(1_671_573)
    payload = basic_payload(
        messages=[
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image_url", "image_url": {"url": image_url}}
                        for _ in range(MAX_IMAGES)
                    ],
                    {"type": "text", "text": "Continue from the frozen history."},
                ],
            }
        ]
    )
    body = json.dumps(payload, separators=(",", ":")).encode()
    assert 8 * 1024 * 1024 < len(body) <= MAX_REQUEST_BYTES
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(endpoint))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={**AUTH, "Content-Type": "application/json"},
        )
        assert connection.getresponse().status == 200
        connection.close()

        connection = http.client.HTTPConnection(*server.server_address)
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Authorization", f"Bearer {API_KEY}")
        connection.putheader("Content-Length", str(MAX_REQUEST_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 413
        assert json.loads(response.read())["error"]["message"] == "request body is too large"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_base64_image_is_decoded_and_passed_to_template_and_processor():
    endpoint, processor, _, _ = make_endpoint()
    payload = basic_payload(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is visible?"},
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                ],
            }
        ]
    )

    response = request(endpoint, payload)

    assert response.status == 200
    assert processor.template_kwargs == {"tokenize": False, "add_generation_prompt": True}
    image = processor.template_messages[0]["content"][1]["image"]
    assert image.mode == "RGB"
    assert image.size == (2, 3)
    assert processor.processor_kwargs == {
        "text": ["rendered prompt"],
        "images": [image],
        "padding": True,
        "return_tensors": "pt",
    }


def test_exact_official_qwen_agent_payload_fields_are_accepted():
    endpoint, processor, model, _ = make_endpoint()
    official_payload = {
        "model": DEFAULT_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "# Tools\nUse computer_use."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                    {"type": "text", "text": "Please generate the next move."},
                ],
            },
        ],
        "max_tokens": 32768,
        "top_p": 0.9,
        "temperature": 0.0,
    }

    response = request(endpoint, official_payload)

    assert response.status == 200
    assert [message["role"] for message in processor.template_messages] == ["system", "user"]
    assert MAX_OUTPUT_TOKENS == 32768
    assert model.generate_kwargs["max_new_tokens"] == 32768
    assert model.generate_kwargs["do_sample"] is False
    assert "top_p" not in model.generate_kwargs
    assert "temperature" not in model.generate_kwargs


@pytest.mark.parametrize(
    "image_url",
    [
        "https://example.invalid/image.png",
        "data:image/png,not-base64",
        "data:image/png;base64,%%%%",
        "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
    ],
)
def test_non_base64_or_invalid_images_are_rejected(image_url):
    endpoint, _, _, loads = make_endpoint()
    payload = basic_payload(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}}],
            }
        ]
    )

    response = request(endpoint, payload)

    assert response.status == 400
    assert loads == []


def test_generation_is_deterministic_and_full_decoded_text_is_preserved():
    endpoint, processor, model, loads = make_endpoint()

    first = request(endpoint, basic_payload(max_tokens=99))
    second = request(endpoint, basic_payload(max_tokens=99))

    assert first.status == 200
    assert len(loads) == 1
    assert model.generate_kwargs["max_new_tokens"] == 99
    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs
    assert np.array_equal(model.generate_kwargs["input_ids"], [[10, 11, 12]])
    assert np.array_equal(processor.decoded_ids[0], [20, 21, 22, 23])
    assert first.body["object"] == "chat.completion"
    assert first.body["created"] == 1234
    assert first.body["model"] == DEFAULT_MODEL_ID
    assert first.body["system_fingerprint"] == MODEL_FINGERPRINT
    assert first.body["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>inspect carefully</think>\n<final>click(1, 2)</final>",
            },
            "finish_reason": "stop",
        }
    ]
    assert first.body["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
    assert second.status == 200


def test_health_and_models_have_pinned_fingerprint_without_loading():
    endpoint, _, _, loads = make_endpoint()

    health = endpoint.handle("GET", "/health", AUTH)
    models = endpoint.handle("GET", "/v1/models", AUTH)

    assert health.status == 200
    assert health.body == {
        "status": "ok",
        "model": DEFAULT_MODEL_ID,
        "revision": DEFAULT_MODEL_REVISION,
        "fingerprint": MODEL_FINGERPRINT,
        "loaded": False,
    }
    assert models.body["data"][0]["revision"] == DEFAULT_MODEL_REVISION
    assert models.body["data"][0]["fingerprint"] == MODEL_FINGERPRINT
    assert FINGERPRINT_SOURCE["dtype"] == "bfloat16"
    assert FINGERPRINT_SOURCE["attention"] == "flash_attention_2"
    assert FINGERPRINT_SOURCE["quantization"] is None
    assert FINGERPRINT_SOURCE["top_p"] == FROZEN_TOP_P
    assert FINGERPRINT_SOURCE["max_output_tokens"] == 32768
    assert FINGERPRINT_SOURCE["client_history_n"] == CLIENT_HISTORY_N == 4
    assert FINGERPRINT_SOURCE["client_image_max"] == CLIENT_IMAGE_MAX == 5
    assert FINGERPRINT_SOURCE["serialization"] == "single-inference-lock"
    assert FINGERPRINT_SOURCE["model_workers_per_gpu"] == 1
    assert loads == []


def test_secret_is_redacted_from_all_errors_and_responses():
    endpoint, _, _, _ = make_endpoint()
    responses = [
        endpoint.handle("GET", "/health", {"Authorization": f"Bearer {API_KEY}-wrong"}),
        request(endpoint, basic_payload(temperature=1)),
        endpoint.handle("GET", "/missing", AUTH),
    ]

    assert all(API_KEY not in json.dumps(response.body) for response in responses)


@pytest.mark.parametrize("failure_stage", ["loader", "processor", "model"])
def test_unexpected_inference_errors_return_generic_redacted_openai_500(failure_stage):
    internal_text = f"CUDA OOM contained {API_KEY}"
    endpoint, processor, model, _ = make_endpoint()
    if failure_stage == "loader":
        endpoint = QwenEndpoint(API_KEY, loader=lambda: (_ for _ in ()).throw(RuntimeError(internal_text)))
    elif failure_stage == "processor":
        processor.apply_chat_template = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(internal_text))
    else:
        model.generate = lambda **kwargs: (_ for _ in ()).throw(RuntimeError(internal_text))

    response = request(endpoint, basic_payload())
    serialized = json.dumps(response.body)

    assert response.status == 500
    assert response.body == {
        "error": {
            "message": "internal server error",
            "type": "server_error",
            "code": "internal_error",
        }
    }
    assert API_KEY not in serialized
    assert internal_text not in serialized


def test_default_bind_is_loopback_and_public_bind_requires_explicit_unsafe_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.delenv("QWEN_ENDPOINT_HOST", raising=False)
    monkeypatch.delenv("QWEN_ENDPOINT_PORT", raising=False)

    main(["--dry-run"])
    default_dry_run = json.loads(capsys.readouterr().out)
    assert default_dry_run["host"] == "127.0.0.1"
    assert default_dry_run["public_bind"] is False

    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0", "--dry-run"])
    assert "--unsafe-allow-public-bind" in capsys.readouterr().err

    main(["--host", "0.0.0.0", "--unsafe-allow-public-bind", "--dry-run"])
    explicit_dry_run = json.loads(capsys.readouterr().out)
    assert explicit_dry_run["public_bind"] is True


def test_endpoint_manifest_freezes_revision_runtime_and_scaling():
    manifest_path = Path(__file__).parents[1] / "model_deployments.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    deployment = manifest["models"][0]

    assert deployment["model"] == DEFAULT_MODEL_ID
    assert deployment["revision"] == DEFAULT_MODEL_REVISION
    assert deployment["processor_revision"] == DEFAULT_MODEL_REVISION
    endpoint = deployment["endpoint"]
    assert endpoint["environment_names"] == ["QWEN_ENDPOINT_API_KEY", "HF_HOME"]
    assert endpoint["image"] == (
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:"
        "14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
    )
    assert endpoint["python"] == "3.11"
    assert endpoint["install_command"] == (
        "uv sync --frozen --no-dev --group endpoint --no-editable"
    )
    assert endpoint["preflight_command"] == "qwen-endpoint --dry-run"
    assert endpoint["command"] == "qwen-endpoint --host 127.0.0.1 --port 8000"
    assert "uv run" not in endpoint["preflight_command"]
    assert "uv run" not in endpoint["command"]
    assert endpoint["provisioning_script"] == "scripts/provision_qwen_endpoint.sh"
    assert endpoint["secret_injection"] == {
        "source_environment": "QWEN_ENDPOINT_API_KEY",
        "path": "/run/qwen-endpoint/api-key",
        "mode": "0600",
    }
    assert endpoint["runtime"] == {
        "accelerate": "1.1.1",
        "flash-attn": "2.7.4.post1",
        "ninja": "1.11.1.3",
        "packaging": "24.2",
        "qwen-vl-utils": "0.0.14",
        "setuptools": "75.8.0",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "transformers": "4.57.1",
        "wheel": "0.45.1",
    }
    assert deployment["inference"] == {
        "dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "quantization": None,
        "do_sample": False,
        "temperature": 0,
        "top_p": FROZEN_TOP_P,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "serialization": "single-inference-lock",
        "model_workers_per_gpu": 1,
        "horizontal_scaling": "multiple identical workers behind the AWS controller",
    }
    assert deployment["client_policy"] == {
        "history_n": CLIENT_HISTORY_N,
        "image_max": CLIENT_IMAGE_MAX,
    }
    assert deployment["limits"]["max_request_bytes"] == MAX_REQUEST_BYTES
    assert "QWEN_ENDPOINT_API_KEY=" not in deployment["endpoint"]["command"]


def test_endpoint_dependency_group_and_lock_are_exact_without_cuda13_drift():
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock_text = (root / "uv.lock").read_text(encoding="utf-8")
    lock = tomllib.loads(lock_text)
    expected = {
        "accelerate": "1.1.1",
        "flash-attn": "2.7.4.post1",
        "ninja": "1.11.1.3",
        "packaging": "24.2",
        "qwen-vl-utils": "0.0.14",
        "setuptools": "75.8.0",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "transformers": "4.57.1",
        "wheel": "0.45.1",
    }
    requirements = {
        requirement.rsplit("==", 1)[0]: requirement.rsplit("==", 1)[1]
        for requirement in project["dependency-groups"]["endpoint"]
    }
    packages = {package["name"]: package["version"] for package in lock["package"]}

    assert requirements == expected
    assert {name: packages[name] for name in expected} == expected
    assert packages["nvidia-cuda-runtime-cu12"] == "12.4.127"
    assert packages["nvidia-cuda-nvrtc-cu12"] == "12.4.127"
    assert "torch==2.12" not in lock_text
    assert "cuda13" not in lock_text.lower()
    assert "cu13" not in lock_text.lower()
    assert project["tool"]["uv"]["extra-build-dependencies"]["flash-attn"] == [
        {"requirement": "torch", "match-runtime": True},
        "ninja==1.11.1.3",
        "packaging==24.2",
        "setuptools==75.8.0",
        "wheel==0.45.1",
    ]
    assert project["tool"]["uv"]["dependency-metadata"] == [
        {
            "name": "flash-attn",
            "version": "2.7.4.post1",
            "requires-dist": ["einops", "torch"],
        }
    ]


def test_endpoint_provisioning_uses_only_frozen_uv_sync_and_loopback_start():
    root = Path(__file__).parents[1]
    script_path = root / "scripts/provision_qwen_endpoint.sh"
    script = script_path.read_text(encoding="utf-8")
    proposal = (
        root / "docs/evaluation/qwen3_vl_osworld_evaluation_plan.md"
    ).read_text(encoding="utf-8")

    assert script_path.stat().st_mode & 0o777 == 0o755
    assert "uv sync --frozen --no-dev --group endpoint --no-editable" in script
    runtime_import = (
        "import torch, torchvision; "
        'assert torch.__version__.split("+")[0] == "2.5.1"; '
        'assert torchvision.__version__.split("+")[0] == "0.20.1"'
    )
    assert runtime_import in script
    assert "qwen-endpoint --dry-run" in script
    assert "exec qwen-endpoint --host 127.0.0.1 --port 8000" in script
    assert script.index("uv sync") < script.index("qwen-endpoint --dry-run")
    assert script.index(runtime_import) < script.index("qwen-endpoint --dry-run")
    assert script.index("qwen-endpoint --dry-run") < script.index("exec qwen-endpoint")
    assert "chmod 0600" in script
    assert "pip " not in script
    assert "uv run" not in script
    assert "`uv sync --frozen --no-dev --group endpoint --no-editable`" in proposal
    assert "installed binary directly as `qwen-endpoint --dry-run`" in proposal
    assert "it does not use `uv run`" in proposal
