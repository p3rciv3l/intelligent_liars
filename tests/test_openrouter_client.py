from __future__ import annotations

from pathlib import Path

import pytest

from intelligent_liars.clients.openrouter_client import (
    OpenRouterAPIError,
    OpenRouterClient,
    default_model_deployments_path,
    fetch_openrouter_key_metadata,
    get_model_client,
)


class FakeResponse:
    status_code = 200
    headers = {}

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse()


class FakeRequests:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, *, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self.response


def test_openrouter_client_builds_chat_completion_request():
    session = FakeSession()
    client = OpenRouterClient(
        "test/provider-model",
        api_key="sk-test",
        timeout=7.0,
        temperature=0.2,
        max_tokens=11,
        provider={"sort": "price", "quantizations": []},
    )

    response = client.generate(
        [{"role": "user", "content": "hello"}],
        session=session,
        top_p=0.5,
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    call = session.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["headers"]["X-OpenRouter-Title"] == "intelligent-liars"
    assert call["timeout"] == 7.0
    assert call["json"]["model"] == "test/provider-model"
    assert call["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert call["json"]["temperature"] == 0.2
    assert call["json"]["top_p"] == 0.5
    assert call["json"]["max_tokens"] == 11
    assert call["json"]["provider"]["data_collection"] == "deny"
    assert call["json"]["provider"]["sort"] == "price"
    assert call["json"]["provider"]["quantizations"] == []


def test_openrouter_client_sends_only_explicit_defaults():
    session = FakeSession()
    client = OpenRouterClient("test/provider-model", api_key="sk-test")

    client.generate([{"role": "user", "content": "hello"}], session=session)

    payload = session.calls[0]["json"]
    assert payload == {
        "model": "test/provider-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "provider": {
            "require_parameters": False,
            "allow_fallbacks": True,
            "data_collection": "deny",
        },
    }


def test_fetch_openrouter_key_metadata_validates_configured_key(monkeypatch):
    class KeyResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {"data": {"label": "test-key", "usage": 0, "limit": 1000}}

    fake_requests = FakeRequests(KeyResponse())
    monkeypatch.setattr("intelligent_liars.clients.openrouter_client._get_requests", lambda: fake_requests)

    metadata = fetch_openrouter_key_metadata(api_key="sk-test", timeout=9.0)

    assert metadata["label"] == "test-key"
    assert fake_requests.calls == [
        {
            "url": "https://openrouter.ai/api/v1/key",
                "headers": {
                    "Authorization": "Bearer sk-test",
                    "User-Agent": "OpenAI File Downloader, XaiImageApiFetch/1.0",
                    "X-OpenRouter-Title": "intelligent-liars",
                },
            "timeout": 9.0,
        }
    ]


def test_fetch_openrouter_key_metadata_raises_typed_auth_error(monkeypatch):
    class UnauthorizedResponse:
        status_code = 401
        headers = {}

        def json(self):
            return {"error": {"message": "invalid key"}}

    monkeypatch.setattr(
        "intelligent_liars.clients.openrouter_client._get_requests",
        lambda: FakeRequests(UnauthorizedResponse()),
    )

    with pytest.raises(OpenRouterAPIError) as exc_info:
        fetch_openrouter_key_metadata(api_key="bad-key")

    assert exc_info.value.status_code == 401
    assert "invalid key" in str(exc_info.value)


def test_default_model_deployments_path_walks_up_from_nested_cwd(tmp_path, monkeypatch):
    nested = tmp_path / "src" / "intelligent_liars"
    nested.mkdir(parents=True)
    deployments = tmp_path / "model_deployments.yaml"
    deployments.write_text("models: []\n")
    monkeypatch.chdir(nested)

    assert default_model_deployments_path() == deployments


def test_default_model_deployments_path_falls_back_to_prod_env(tmp_path, monkeypatch):
    nested = tmp_path / "src" / "intelligent_liars"
    nested.mkdir(parents=True)
    deployments = tmp_path / "prod_env" / "model_deployments.yaml"
    deployments.parent.mkdir()
    deployments.write_text("models: []\n")
    monkeypatch.chdir(nested)

    assert default_model_deployments_path() == deployments


def test_get_model_client_loads_yaml_alias(tmp_path):
    yaml_path = tmp_path / "model_deployments.yaml"
    yaml_path.write_text(
        """
models:
  - name: cheap-judge
    model: example/judge-model:nitro
    temperature: 0.1
    max_tokens: 42
    provider:
      sort: price
"""
    )

    session = FakeSession()
    client = get_model_client("cheap-judge", api_key="sk-test", yaml_path=yaml_path, timeout=3.0)

    assert client.model == "example/judge-model:nitro"
    client.generate([{"role": "user", "content": "hi"}], session=session)

    payload = session.calls[0]["json"]
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 42
    assert payload["provider"]["sort"] == "price"
    assert session.calls[0]["timeout"] == 3.0
    assert client.provider_config["sort"] == "price"


def test_project_glm_flash_judge_contract_is_frozen():
    yaml_path = Path(__file__).resolve().parents[1] / "model_deployments.yaml"
    session = FakeSession()

    client = get_model_client(
        "glm-5.3-flash",
        api_key="sk-test",
        yaml_path=yaml_path,
    )
    client.generate([{"role": "user", "content": "grade"}], session=session)

    payload = session.calls[0]["json"]
    assert client.model == "z-ai/glm-5.3-flash"
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 2048
    assert payload["reasoning"] == {"effort": "high", "exclude": True}
    assert payload["provider"] == {
        "order": ["z-ai/fp8"],
        "only": ["z-ai/fp8"],
        "quantizations": ["fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }


def test_project_gemini_flash_readability_contract_omits_sampling():
    yaml_path = Path(__file__).resolve().parents[1] / "model_deployments.yaml"
    session = FakeSession()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "done_step",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    client = get_model_client(
        "gemini-3.8-flash",
        api_key="sk-test",
        yaml_path=yaml_path,
    )
    client.generate(
        [{"role": "user", "content": "hi"}],
        session=session,
        tools=tools,
        tool_choice="auto",
    )

    payload = session.calls[0]["json"]
    assert client.model == "google/gemini-3.8-flash:nitro"
    assert client.timeout == 1200
    assert session.calls[0]["timeout"] == 1200
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload
    assert "max_tokens" not in payload
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"
    assert payload["reasoning"] == {"effort": "high", "exclude": False}
    assert payload["provider"] == {
        "only": [
            "google-ai-studio/priority",
            "google-vertex/global/priority",
            "google-ai-studio",
            "google-vertex",
        ],
        "sort": "throughput",
        "allow_fallbacks": True,
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert "quantizations" not in payload["provider"]


def test_get_model_client_merges_provider_with_call_site_winning(tmp_path):
    yaml_path = tmp_path / "model_deployments.yaml"
    yaml_path.write_text(
        """
models:
  - name: judge
    model: example/judge-model:nitro
    provider:
      quantizations: []
      require_parameters: false
"""
    )

    session = FakeSession()
    client = get_model_client(
        "judge",
        api_key="sk-test",
        yaml_path=yaml_path,
        provider={"require_parameters": True},
    )

    client.generate([{"role": "user", "content": "hi"}], session=session)

    payload = session.calls[0]["json"]
    assert payload["provider"]["quantizations"] == []
    assert payload["provider"]["require_parameters"] is True


def test_get_model_client_rejects_alias_typo_without_raw_model_shape(tmp_path):
    yaml_path = tmp_path / "model_deployments.yaml"
    yaml_path.write_text(
        """
models:
  - name: judge
    model: example/judge-model:nitro
"""
    )

    with pytest.raises(ValueError, match="Model 'judeg' not found"):
        get_model_client("judeg", api_key="sk-test", yaml_path=yaml_path)


def test_get_model_client_allows_raw_openrouter_model_id_when_alias_missing(tmp_path):
    yaml_path = tmp_path / "model_deployments.yaml"
    yaml_path.write_text(
        """
models:
  - name: judge
    model: example/judge-model:nitro
"""
    )

    client = get_model_client("z-ai/glm-5.2:floor", api_key="sk-test", yaml_path=yaml_path)

    assert client.model == "z-ai/glm-5.2:floor"


def test_openrouter_client_raises_typed_retryable_error():
    class RateLimitResponse:
        status_code = 429
        headers = {"Retry-After": "3"}

        def json(self):
            return {
                "error": {
                    "message": "slow down",
                    "metadata": {
                        "error_type": "rate_limit_exceeded",
                        "provider_name": "ExampleProvider",
                    },
                }
            }

    class RateLimitSession(FakeSession):
        def post(self, url, *, headers, json, timeout):
            self.calls.append(
                {
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }
            )
            return RateLimitResponse()

    session = RateLimitSession()
    client = OpenRouterClient("test/provider-model", api_key="sk-test")

    with pytest.raises(OpenRouterAPIError) as exc_info:
        client.generate([{"role": "user", "content": "hello"}], session=session)

    assert exc_info.value.status_code == 429
    assert exc_info.value.error_type == "rate_limit_exceeded"
    assert exc_info.value.provider_name == "ExampleProvider"
    assert exc_info.value.retry_after == 3.0
    assert exc_info.value.retryable is True
