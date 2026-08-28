from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from collections.abc import Sequence
from typing import Any, cast

from dotenv import load_dotenv

from intelligent_liars.clients.openrouter_client import OpenRouterAPIError
from intelligent_liars.judge_config import JudgeConfig, Judgement, Message, PromptGetter


logger = logging.getLogger(__name__)
_thread_local = threading.local()


def _judging() -> Any:
    from intelligent_liars import judging

    return judging


def parse_judgement(response_text: str) -> tuple[Judgement, str]:
    """Parse Apollo-style <judgement>...</judgement> grader output."""
    try:
        parts = response_text.split("<judgement>")
        if len(parts) != 2:
            logger.warning(
                "Missing or multiple <judgement> tags in response: %s",
                _safe_text_context(response_text),
            )
            return "unknown", response_text

        judgement_section = parts[1].split("</judgement>")
        if len(judgement_section) != 2:
            logger.warning(
                "Missing or multiple </judgement> tags in response: %s",
                _safe_text_context(response_text),
            )
            return "unknown", response_text

        judgement: Judgement = judgement_section[0].strip().lower()
        if isinstance(judgement, str) and judgement.isdigit():
            judgement = int(judgement)
        return judgement, response_text
    except Exception as exc:  # pragma: no cover - defensive diagnostic
        logger.warning("Error parsing judgement: %s", exc)
        return "unknown", response_text


def convert_response_to_json(response: str) -> dict[str, str]:
    """Extract the first JSON object from a judge response."""
    response = response.strip()
    start_idx = response.find("{")
    end_idx = response.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError(
            f"Could not find JSON in response: {_safe_text_context(response)}"
        )
    parsed = json.loads(response[start_idx : end_idx + 1])
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Judge JSON must be an object: {_safe_text_context(response)}"
        )
    return {str(key): str(value) for key, value in parsed.items()}


def convert_response_to_object(response: str) -> dict[str, Any]:
    """Extract the first JSON object from a judge response while preserving value types."""
    response = response.strip()
    start_idx = response.find("{")
    end_idx = response.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError(
            f"Could not find JSON in response: {_safe_text_context(response)}"
        )
    parsed = json.loads(response[start_idx : end_idx + 1])
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Judge JSON must be an object: {_safe_text_context(response)}"
        )
    return cast(dict[str, Any], parsed)


def json_judge_response_format(
    expected_judgements: Sequence[str] | None,
) -> dict[str, Any]:
    judgement_schema: dict[str, Any] = {"type": "string"}
    if expected_judgements is not None:
        judgement_schema["enum"] = list(expected_judgements)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "deception_judge_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief evidence-based justification using only the supplied transcript or message.",
                    },
                    "judgement": judgement_schema,
                },
                "required": ["reasoning", "judgement"],
                "additionalProperties": False,
            },
        },
    }


def call_judge(
    *,
    system_prompt: str | None,
    user_prompt: str,
    config: JudgeConfig,
    response_format: dict[str, Any] | None = None,
    provider_routing: dict[str, Any] | None = None,
) -> str:
    """Call the configured judge model with Apollo-style retry behavior."""
    if config.mock_response is not None:
        return config.mock_response

    load_dotenv()
    last_error: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            if config.provider == "openrouter":
                return _judging()._call_openrouter_judge(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    config=config,
                    response_format=response_format,
                    provider_routing=provider_routing,
                )
            raise ValueError(f"Unsupported judge provider: {config.provider}")
        except Exception as exc:  # pragma: no cover - network/API diagnostic
            last_error = exc
            if not _judging()._should_retry_judge_error(exc):
                raise
            if attempt < config.max_retries - 1:
                _judging()._handle_error(attempt, exc, config.base_delay)
    raise RuntimeError(
        f"Max retries exceeded. Last error: {last_error}"
    ) from last_error


def grade_single_dialogue_for_deception(
    grading_prompt_getter: PromptGetter,
    input_messages: list[Message],
    output_message: Message,
    metadata: dict[str, Any] | None,
    config: JudgeConfig,
) -> tuple[Judgement, str]:
    """Grade a dialogue using the generic Apollo <judgement> judge harness."""
    try:
        prompt = grading_prompt_getter(input_messages, output_message, metadata)
    except Exception as exc:
        raise RuntimeError(f"Could not build grading prompt: {exc}") from exc

    return _judging()._run_tagged_judge(
        system_prompt=prompt.get("system"),
        user_prompt=prompt["user"],
        config=config,
    )


def _run_tagged_judge(
    *,
    system_prompt: str | None,
    user_prompt: str,
    config: JudgeConfig,
) -> tuple[Judgement, str]:
    """Call a `<judgement>` grader and retry malformed judge text."""

    last_error: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            response_text = _judging().call_judge(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=config,
            )
        except Exception as exc:
            raise RuntimeError(f"Grader API error: {exc}") from exc
        try:
            return _parse_required_judgement(response_text)
        except ValueError as exc:
            last_error = exc
            if attempt < config.max_retries - 1:
                logger.warning(
                    "Judge parse error on attempt %s: %s",
                    attempt,
                    exc.__class__.__name__,
                )
    raise ValueError(
        f"Invalid tagged judge response after retries: {last_error}"
    ) from last_error


def _parse_required_judgement(response_text: str) -> tuple[Judgement, str]:
    judgement, reasoning = parse_judgement(response_text)
    if judgement == "unknown":
        raise ValueError(
            f"Could not parse required judge score: {_safe_text_context(response_text)}"
        )
    return judgement, reasoning


def run_json_judge(
    prompt: str,
    *,
    expected_judgements: Sequence[str] | None,
    config: JudgeConfig,
) -> dict[str, str]:
    last_error: Exception | None = None
    for _ in range(config.max_retries):
        user_prompt = prompt
        response_format = None
        provider_routing = None
        if config.structured_outputs:
            user_prompt = prompt + _json_judge_structured_instruction(
                expected_judgements
            )
            response_format = json_judge_response_format(expected_judgements)
            provider_routing = _provider_routing_for_structured_outputs(config)
        response = _judging().call_judge(
            system_prompt=None,
            user_prompt=user_prompt,
            config=config,
            response_format=response_format,
            provider_routing=provider_routing,
        )
        try:
            response_json = convert_response_to_json(response)
            if set(response_json) != {"reasoning", "judgement"}:
                raise ValueError(
                    f"Expected reasoning/judgement keys, got {sorted(response_json)}"
                )
            response_json["judgement"] = response_json["judgement"].lower()
            if (
                expected_judgements is not None
                and response_json["judgement"] not in expected_judgements
            ):
                raise ValueError(
                    f"Expected judgement in {expected_judgements}, got {response_json['judgement']}"
                )
            return response_json
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Invalid judge response after retries: {last_error}")


def _json_judge_structured_instruction(
    expected_judgements: Sequence[str] | None,
) -> str:
    choices = (
        ""
        if expected_judgements is None
        else f" Judgement must be one of: {', '.join(expected_judgements)}."
    )
    return (
        "\n\nReturn only a JSON object with exactly two keys: reasoning and judgement."
        + choices
    )


def _provider_routing_for_structured_outputs(config: JudgeConfig) -> dict[str, Any]:
    provider = dict(config.provider_routing or {})
    provider["require_parameters"] = config.require_structured_outputs
    return provider


def _call_openrouter_judge(
    *,
    system_prompt: str | None,
    user_prompt: str,
    config: JudgeConfig,
    response_format: dict[str, Any] | None = None,
    provider_routing: dict[str, Any] | None = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    client_kwargs: dict[str, Any] = {
        "timeout": config.timeout,
        "max_tokens": config.max_tokens,
    }
    if provider_routing is not None:
        client_kwargs["provider"] = provider_routing
    elif config.provider_routing is not None:
        client_kwargs["provider"] = config.provider_routing
    if response_format is not None:
        client_kwargs["response_format"] = response_format
        client_kwargs["plugins"] = (
            [{"id": "response-healing"}] if config.json_response_healing else []
        )

    client = _judging().get_model_client(
        config.resolved_model,
        api_key=config.api_key,
        yaml_path=config.model_config_path,
        **client_kwargs,
    )
    response = client.generate(messages, session=_judging()._get_openrouter_session())
    return _extract_openrouter_text(response)


def _get_openrouter_session() -> Any:
    if not hasattr(_thread_local, "openrouter_session"):
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency diagnostic
            raise RuntimeError(
                "requests is required for OpenRouter judge calls"
            ) from exc

        _thread_local.openrouter_session = requests.Session()
    return _thread_local.openrouter_session


def _extract_openrouter_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"OpenRouter response missing choices; keys={sorted(response)}"
        )
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError(
            f"OpenRouter response missing message; keys={sorted(response)}"
        )

    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    raise RuntimeError(
        f"OpenRouter response did not contain text content; keys={sorted(response)}"
    )


def _handle_error(attempt: int, error: Exception, base_delay: float) -> None:
    retry_after = getattr(error, "retry_after", None)
    if isinstance(retry_after, int | float):
        delay = max(float(retry_after), 0.0)
    else:
        delay = random.uniform(0.0, min(60.0, base_delay * (2**attempt)))
    logger.warning("Judge error on attempt %s: %s", attempt, error.__class__.__name__)
    time.sleep(delay)


def _should_retry_judge_error(error: Exception) -> bool:
    if isinstance(error, OpenRouterAPIError):
        return error.retryable
    if isinstance(error, NotImplementedError | ValueError):
        return False
    return True


def _safe_text_context(text: str, limit: int = 160) -> str:
    normalized = " ".join(text.split())
    snippet = normalized[:limit]
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"sha256={digest} snippet={snippet!r}"
