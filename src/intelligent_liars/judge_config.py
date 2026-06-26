from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias


DEFAULT_OPENROUTER_JUDGE_MODEL = "glm-5.2"
ROLEPLAYING_SYSTEM_POSTFIX = (
    "Answer the question directly, without asides or internal thoughts."
)

Message: TypeAlias = dict[str, Any]
Judgement: TypeAlias = int | str
JudgeProvider = Literal["openrouter"]
PromptGetter: TypeAlias = Callable[
    [list[Message], Message, dict[str, Any] | None], dict[str, str]
]


def _judging() -> Any:
    from intelligent_liars import judging

    return judging


@dataclass(frozen=True)
class JudgeConfig:
    provider: JudgeProvider = "openrouter"
    model: str | None = None
    max_tokens: int = 1000
    max_retries: int = 4
    base_delay: float = 2.0
    timeout: float = 120.0
    provider_routing: dict[str, Any] | None = None
    model_config_path: Path | None = None
    api_key: str | None = None
    structured_outputs: bool = True
    require_structured_outputs: bool = True
    mock_response: str | None = None

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return DEFAULT_OPENROUTER_JUDGE_MODEL


# Shared defaults used by judging call sites and Typer options.
DEFAULT_JUDGE_CONFIG = JudgeConfig()
DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY = 25
DEFAULT_ROLLOUT_GRADING_MAX_WORKERS = 1
DEFAULT_INSIDER_GRADING_FLUSH_EVERY = 10


@dataclass(frozen=True)
class GradingSummary:
    task: str
    input_path: Path
    output_path: Path
    total_items: int
    graded_items: int
    skipped_items: int


@dataclass(frozen=True)
class JudgePreflight:
    provider: JudgeProvider
    alias: str
    resolved_model: str
    structured_outputs: bool
    require_structured_outputs: bool
    checked_api_key: bool
    checked_live_metadata: bool


def preflight_judge_config(config: JudgeConfig) -> JudgePreflight:
    """Validate judge configuration before generation or grading mutates files."""

    if config.mock_response is not None:
        return JudgePreflight(
            provider=config.provider,
            alias=config.resolved_model,
            resolved_model=config.resolved_model,
            structured_outputs=config.structured_outputs,
            require_structured_outputs=config.require_structured_outputs,
            checked_api_key=False,
            checked_live_metadata=False,
        )
    if config.provider != "openrouter":
        raise ValueError(f"Unsupported judge provider: {config.provider}")

    _judging().fetch_openrouter_key_metadata(
        api_key=config.api_key, timeout=config.timeout
    )

    client = _judging().get_model_client(
        config.resolved_model,
        api_key=config.api_key,
        yaml_path=config.model_config_path,
        timeout=config.timeout,
        max_tokens=config.max_tokens,
    )
    checked_live_metadata = False
    if config.structured_outputs and config.require_structured_outputs:
        metadata = _judging().fetch_openrouter_model_endpoint_metadata(
            client.model, timeout=config.timeout
        )
        checked_live_metadata = True
        endpoints = metadata.get("endpoints")
        if not isinstance(endpoints, list):
            raise RuntimeError(
                f"OpenRouter model metadata missing endpoints for {client.model}"
            )
        routeable_endpoints = _filter_endpoints_for_provider_routing(
            endpoints,
            getattr(client, "provider_config", {}),
        )
        if not routeable_endpoints:
            raise ValueError(
                f"OpenRouter model {client.model!r} has no live endpoint matching provider routing; "
                "check model_deployments.yaml provider.order/provider.only."
            )
        capable_endpoints = [
            endpoint
            for endpoint in routeable_endpoints
            if isinstance(endpoint, dict)
            and endpoint.get("status", 0) == 0
            and "response_format" in set(endpoint.get("supported_parameters") or [])
        ]
        if not capable_endpoints:
            raise ValueError(
                f"OpenRouter model {client.model!r} has no routed live endpoint supporting response_format; "
                "choose a structured-output-capable judge or pass --no-structured-outputs."
            )

    return JudgePreflight(
        provider=config.provider,
        alias=config.resolved_model,
        resolved_model=client.model,
        structured_outputs=config.structured_outputs,
        require_structured_outputs=config.require_structured_outputs,
        checked_api_key=True,
        checked_live_metadata=checked_live_metadata,
    )


def _filter_endpoints_for_provider_routing(
    endpoints: Sequence[Any],
    provider_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply local OpenRouter provider filters to endpoint metadata.

    OpenRouter endpoint metadata exposes concrete route tags such as
    `z-ai/fp8` and `wafer/fp4`. Preflight uses those tags to make sure the
    configured restricted provider set still has at least one live route before
    generation starts.
    """

    only = _normalised_provider_values(provider_config.get("only"))
    ignore = _normalised_provider_values(provider_config.get("ignore"))
    quantizations = _normalised_provider_values(provider_config.get("quantizations"))
    filtered: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        endpoint_keys = _endpoint_provider_keys(endpoint)
        if only and endpoint_keys.isdisjoint(only):
            continue
        if ignore and not endpoint_keys.isdisjoint(ignore):
            continue
        if quantizations:
            quantization = str(endpoint.get("quantization") or "").lower()
            if quantization not in quantizations:
                continue
        if endpoint.get("status", 0) != 0:
            continue
        filtered.append(endpoint)
    return filtered


def _normalised_provider_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, Sequence) or isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    return {str(item).strip().lower() for item in values if str(item).strip()}


def _endpoint_provider_keys(endpoint: dict[str, Any]) -> set[str]:
    tag = str(endpoint.get("tag") or "").strip().lower()
    provider_name = str(endpoint.get("provider_name") or "").strip().lower()
    keys = {
        tag,
        tag.split("/", maxsplit=1)[0],
        provider_name,
        provider_name.replace(".", "-").replace(" ", "-"),
    }
    return {key for key in keys if key}
