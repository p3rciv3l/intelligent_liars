from __future__ import annotations

import pytest

from intelligent_liars.clients.openrouter_client import OpenRouterAPIError, OpenRouterClient, get_model_client


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
    assert call["timeout"] == 7.0
    assert call["json"]["model"] == "test/provider-model"
    assert call["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert call["json"]["temperature"] == 0.2
    assert call["json"]["top_p"] == 0.5
    assert call["json"]["max_tokens"] == 11
    assert call["json"]["provider"]["data_collection"] == "deny"
    assert call["json"]["provider"]["sort"] == "price"
    assert call["json"]["provider"]["quantizations"] == []


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
