"""OpenRouter chat-completion client used by the judging harness.

Standalone OpenRouter Client

A simple client for OpenRouter's unified AI API that provides intelligent
provider routing across multiple model providers with built-in defaults and
easy parameter overrides.

References:
    - Parameters: https://openrouter.ai/docs/api/reference/parameters
    - Provider Routing: https://openrouter.ai/docs/guides/routing/provider-selection
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv

# Note: We use requests directly instead of the OpenAI SDK to avoid import hang issues
# The OpenAI SDK can hang on import in some environments due to httpx/HTTP2 issues

# Lazy imports to avoid any module-level execution
_yaml = None
_requests = None

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_TYPES = {
    "rate_limit_exceeded",
    "provider_overloaded",
    "provider_unavailable",
    "timeout",
    "server",
    "unmapped",
}


class OpenRouterAPIError(RuntimeError):
    """Error returned by OpenRouter or an upstream provider."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        retry_after: float | None = None,
        provider_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.retry_after = retry_after
        self.provider_name = provider_name
        self.payload = payload

    @property
    def retryable(self) -> bool:
        if self.status_code is not None and (
            self.status_code in RETRYABLE_STATUS_CODES or self.status_code >= 500
        ):
            return True
        return self.error_type in RETRYABLE_ERROR_TYPES


def _get_yaml():
    """Lazy import of yaml."""
    global _yaml
    if _yaml is None:
        try:
            import yaml
            _yaml = yaml
        except ImportError:
            raise ImportError(
                "pyyaml package is required. Install with: pip install pyyaml"
            )
    return _yaml


def _get_requests():
    """Lazy import of requests."""
    global _requests
    if _requests is None:
        try:
            import requests
            _requests = requests
        except ImportError:
            raise ImportError(
                "requests package is required. Install with: pip install requests"
            )
    return _requests


# =============================================================================
# Type Definitions
# =============================================================================

class ProviderPreferences(TypedDict, total=False):
    """
    OpenRouter provider routing configuration.
    
    Controls how requests are routed across providers for optimal cost,
    performance, and reliability.
    
    Reference: https://openrouter.ai/docs/guides/routing/provider-selection
    """
    order: list[str]
    """Provider slugs to try in order (e.g., ["anthropic", "openai"])."""
    
    allow_fallbacks: bool
    """Whether to allow backup providers when primary is unavailable. Default: True."""
    
    require_parameters: bool
    """Only use providers supporting all request parameters. Default: False."""
    
    data_collection: Literal["allow", "deny"]
    """Control whether to use providers that may store data. Default: "allow"."""
    
    zdr: bool
    """Restrict to Zero Data Retention endpoints only."""
    
    enforce_distillable_text: bool
    """Restrict to models allowing text distillation."""
    
    only: list[str]
    """Provider slugs to exclusively allow."""
    
    ignore: list[str]
    """Provider slugs to skip."""
    
    quantizations: list[str]
    """Quantization levels to filter by (e.g., ["fp8", "fp16", "bf16"])."""
    
    sort: Literal["price", "throughput", "latency"]
    """Sort providers by attribute. Disables load balancing when set."""
    
    max_price: dict[str, float]
    """Maximum pricing (e.g., {"prompt": 1.0, "completion": 2.0} for $/M tokens)."""


# =============================================================================
# OpenRouter Client
# =============================================================================

class OpenRouterClient:
    """
    Standalone client for OpenRouter's unified AI API.
    
    Provides intelligent routing across multiple AI providers (OpenAI, Anthropic,
    Google, etc.) with support for cost optimization, latency preferences, and
    privacy controls.
    
    Example:
        >>> client = OpenRouterClient(
        ...     model="openai/gpt-4",
        ...     api_key="sk-...",
        ...     temperature=0.7,
        ...     provider={"sort": "price"},
        ... )
        >>> response = client.generate([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response.choices[0].message.content)
    """

    
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float | None = 30.0,
        # Sampling parameters
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int | None = 0,
        frequency_penalty: float | None = 0.0,
        presence_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        min_p: float | None = 0.0,
        top_a: float | None = 0.0,
        # Generation control
        seed: int | None = 42069,
        max_tokens: int | None = 5120,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        response_format: dict[str, Any] | None = None,
        verbosity: Literal["low", "medium", "high"] | None = "medium",
        # Tool configuration
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        parallel_tool_calls: bool | None = True,
        plugins: list[dict[str, Any]] | None = None,
        # Provider routing
        provider: ProviderPreferences | dict[str, Any] | None = None,
        # Reasoning
        reasoning: dict[str, Any] | None = None,
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            model: OpenRouter model identifier (e.g., "openai/gpt-4")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            default_headers: HTTP headers for all requests (e.g., {"x-anthropic-beta": "..."})
            timeout: Request timeout in seconds (default: 30.0, None for no timeout)
            
            # Sampling (OpenAI-compatible)
            temperature: Controls randomness (0.0-2.0). Default: 1.0
            top_p: Nucleus sampling threshold (0.0-1.0). Default: 1.0
            top_k: Top-k sampling (0=disabled). Default: 0
            frequency_penalty: Penalize frequent tokens (-2.0-2.0). Default: 0.0
            presence_penalty: Penalize repeated tokens (-2.0-2.0). Default: 0.0
            
            # Sampling (OpenRouter-specific)
            repetition_penalty: Alternative repetition control (0.0-2.0). Default: 1.0
            min_p: Minimum probability threshold (0.0-1.0). Default: 0.0
            top_a: Dynamic top-p based on max probability (0.0-1.0). Default: 0.0
            
            # Generation
            seed: Random seed for deterministic outputs. Default: 42069
            max_tokens: Maximum tokens to generate. Default: 1000
            logit_bias: Token bias mapping
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top logprobs to return (0-20)
            response_format: Output format specification
            verbosity: Response verbosity ("low", "medium", "high"). Default: "medium"
            reasoning: Reasoning configuration (e.g., {"effort": "low/medium/high"}). Default: {"effort": "medium"}           
            # Tools
            tools: Tool definitions for function calling
            tool_choice: Tool selection mode ("none", "auto", "required", or specific)
            parallel_tool_calls: Allow parallel function calls. Default: True
            
            # Provider Routing
            provider: Provider routing preferences (see ProviderPreferences)
            
        """
        load_dotenv()
        # Resolve API key
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        # Store model name and HTTP settings
        self.model = model
        self.timeout = timeout
        
        # Store default parameters (None means use default, value means override)
        self._defaults = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
            "top_a": top_a,
            "seed": seed,
            "max_tokens": max_tokens,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "response_format": response_format,
            "verbosity": verbosity,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "plugins": plugins,
            "reasoning": reasoning if reasoning is not None else {"effort": "medium"},
        }
        
        # Build provider configuration with defaults
        self._default_provider = self._build_provider_config(provider)
    
    def _build_provider_config(
        self, 
        user_config: ProviderPreferences | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Build provider configuration by merging user settings with defaults.
        
        Args:
            user_config: User-provided provider preferences
            
        Returns:
            Complete provider configuration dictionary
        """

        config = {
            "require_parameters": False,
            "allow_fallbacks": True,
            "data_collection": "deny",
        }
        
        # Merge user config (user values override defaults)
        if user_config:
            config.update(user_config)
        
        return config
    
    def generate(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        session: Any | None = None,
        **override_params
    ) -> dict[str, Any]:
        """
        Generate a completion using the OpenRouter API.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            stream: Whether to stream the response
            session: Optional requests.Session for connection reuse (improves
                    performance for high-volume requests by reducing TLS handshakes)
            **override_params: Parameters to override client defaults and YAML config
            
        Returns:
            Response dict from OpenRouter API (OpenAI-compatible format)
            
        Example:
            >>> response = client.generate(
            ...     [{"role": "user", "content": "Hello!"}],
            ...     temperature=0.8,  # Override default
            ... )
            >>> print(response["choices"][0]["message"]["content"])
            
            # With session for connection reuse:
            >>> import requests
            >>> session = requests.Session()
            >>> response = client.generate(messages, session=session)
        """
        if stream:
            raise NotImplementedError("OpenRouter streaming requires a separate SSE client path.")

        # Build request parameters with precedence:
        # override_params > client defaults
        params = self._build_request_params(**override_params)
        
        # Add messages and model
        params["model"] = self.model
        params["messages"] = messages
        params["stream"] = stream
        
        # Use provided session or fall back to lazy-loaded requests module
        requester = session if session is not None else _get_requests()
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title

        response = requester.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=params,
            timeout=self.timeout,
        )

        try:
            payload = response.json()
        except ValueError as exc:
            text = getattr(response, "text", "")
            raise RuntimeError(f"OpenRouter returned non-JSON response: {text[:500]}") from exc

        status_code = getattr(response, "status_code", 200)
        if status_code >= 400:
            raise _build_openrouter_error(payload, status_code=status_code, response=response)
        if isinstance(payload, dict) and payload.get("error"):
            raise _build_openrouter_error(payload, status_code=status_code, response=response)
        if not isinstance(payload, dict):
            raise RuntimeError(f"OpenRouter returned unexpected payload: {payload!r}")
        _raise_for_choice_error(payload, status_code=status_code, response=response)
        return payload
    
    def _build_request_params(self, **override_params) -> dict[str, Any]:
        """
        Build request parameters applying precedence:
        override_params > client defaults
        
        Args:
            **override_params: Parameters from method call
            
        Returns:
            Complete request parameters dictionary
        """
        params = {}
        
        # Apply client defaults (only if not None)
        for key, value in self._defaults.items():
            if value is not None and key not in override_params:
                params[key] = value
        
        # Apply method-level overrides (highest precedence)
        params.update(override_params)
        
        # Always include provider config (merge with any override)
        provider_config = dict(self._default_provider)
        if "provider" in override_params:
            provider_config.update(override_params["provider"])
        params["provider"] = provider_config
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return params


# =============================================================================
# YAML Configuration Loading
# =============================================================================

def default_model_deployments_path() -> Path:
    """Return the first likely model-deployments YAML path for this checkout."""
    env_path = os.getenv("OPENROUTER_MODEL_DEPLOYMENTS")
    if env_path:
        return Path(env_path).expanduser()

    client_path = Path(__file__).resolve()
    repo_root = client_path.parents[3]
    candidates = [
        Path.cwd() / "prod_env" / "model_deployments.yaml",
        repo_root / "prod_env" / "model_deployments.yaml",
        client_path.with_name("model_deployments.yaml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_model_deployments(
    yaml_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load model deployments from YAML file.
    
    Args:
        yaml_path: Path to model_deployments.yaml. If None, looks for
                  prod_env/model_deployments.yaml relative to this file.
    
    Returns:
        Dictionary mapping model names to their configurations
    
    Example:
        >>> configs = load_model_deployments()
        >>> print(configs["gpt-5.1"])
    """
    if yaml_path is None:
        yaml_path = default_model_deployments_path()
    
    yaml = _get_yaml()  # Lazy import
    
    yaml_path = Path(yaml_path).expanduser()
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Model deployments file not found: {yaml_path}"
        )
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        raise ValueError(
            f"Invalid YAML file: {yaml_path} is empty or contains no data"
        )
    
    if "models" not in data:
        raise ValueError(
            f"Invalid YAML structure: expected 'models' key in {yaml_path}"
        )
    
    # Build dictionary mapping name -> config
    configs = {}
    for model_config in data["models"]:
        if "name" not in model_config:
            raise ValueError(
                f"Model config missing 'name' field: {model_config}"
            )
        if "model" not in model_config:
            raise ValueError(
                f"Model config missing 'model' field: {model_config}"
            )
        
        name = model_config["name"]
        configs[name] = model_config
    
    return configs


def load_model_config(
    name: str,
    yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load configuration for a specific model by name.
    
    Args:
        name: Model name from YAML file
        yaml_path: Path to model_deployments.yaml (optional)
    
    Returns:
        Model configuration dictionary
    
    Example:
        >>> config = load_model_config("gpt-5.1")
        >>> print(config["model"])  # "openai/gpt-5.1:floor"
    """
    configs = load_model_deployments(yaml_path)
    
    if name not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(
            f"Model '{name}' not found. Available models: {available}"
        )
    
    return dict(configs[name])


def get_model_client(
    name: str,
    api_key: str | None = None,
    yaml_path: str | Path | None = None,
    **override_params
) -> OpenRouterClient:
    """
    Get a configured OpenRouterClient instance from YAML configuration.
    
    Args:
        name: Model name from YAML file
        api_key: OpenRouter API key (optional, uses env var if not provided)
        yaml_path: Path to model_deployments.yaml (optional)
        **override_params: Additional parameters to override YAML config
    
    Returns:
        Configured OpenRouterClient instance
    
    Example:
        >>> client = get_model_client("gpt-5.1")
        >>> response = client.generate([{"role": "user", "content": "Hello!"}])
    """
    try:
        config = load_model_config(name, yaml_path)
        
        # Extract model name (required)
        model = config.pop("model")
        
        # Extract provider config if present
        provider = config.pop("provider", None)
        
        # Extract name (not needed for client)
        config.pop("name", None)
        
        # Merge YAML config with override params (override_params take precedence)
        client_params = {**config, **override_params}
        
        # Add provider back if it exists
        if provider is not None:
            client_params["provider"] = provider
    except (ValueError, FileNotFoundError):
        # Fallback: If name is not an alias in YAML, treat it as a raw model ID
        model = name
        client_params = override_params

    # Create client
    return OpenRouterClient(
        model=model,
        api_key=api_key,
        **client_params
    )


def _build_openrouter_error(
    payload: dict[str, Any],
    *,
    status_code: int | None,
    response: Any,
) -> OpenRouterAPIError:
    error = payload.get("error") if isinstance(payload, dict) else None
    metadata = error.get("metadata", {}) if isinstance(error, dict) else {}
    message = error.get("message") if isinstance(error, dict) else None
    if not message:
        message = f"OpenRouter HTTP {status_code}"
    return OpenRouterAPIError(
        str(message),
        status_code=status_code,
        error_type=_coerce_str(metadata.get("error_type")),
        retry_after=_parse_retry_after(getattr(response, "headers", {}).get("Retry-After")),
        provider_name=_coerce_str(metadata.get("provider_name")),
        payload=payload,
    )


def _raise_for_choice_error(
    payload: dict[str, Any],
    *,
    status_code: int | None,
    response: Any,
) -> None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    first = choices[0]
    if not isinstance(first, dict) or first.get("finish_reason") != "error":
        return
    error = first.get("error")
    if not isinstance(error, dict):
        return
    metadata = error.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    raise OpenRouterAPIError(
        str(error.get("message") or "OpenRouter generation failed"),
        status_code=_coerce_int(error.get("code")) or status_code,
        error_type=_coerce_str(metadata.get("error_type")),
        retry_after=_parse_retry_after(getattr(response, "headers", {}).get("Retry-After")),
        payload=payload,
    )


def _parse_retry_after(value: Any) -> float | None:
    try:
        if value is None:
            return None
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return max(seconds, 0.0)


def _coerce_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
