"""
OpenRouter client helpers for intelligent-liars judging.

This module provides a simple client for OpenRouter's unified AI API with
built-in defaults and easy parameter overrides.
"""

from intelligent_liars.clients.openrouter_client import (
    OpenRouterAPIError,
    OpenRouterClient,
    ProviderPreferences,
    load_model_deployments,
    load_model_config,
    get_model_client,
)

__all__ = [
    "OpenRouterClient",
    "OpenRouterAPIError",
    "ProviderPreferences",
    "load_model_deployments",
    "load_model_config",
    "get_model_client",
]
