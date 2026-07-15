"""Model Manager — Core Business Layer module.

ModelManager owns the provider registry, the current model selection,
and the factory for creating fresh ``ChatOpenAI`` instances on demand.

It does NOT cache model instances — every call to ``get_current_model()``
returns a new object so that temperature / timeout / max_tokens changes
take effect immediately.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from langchain_chat.core.config_models import LLMConfig
from langchain_chat.core.provider import (
    BaseProvider,
    DeepSeekProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

# ------------------------------------------------------------------
# Built-in provider registry (extensible without touching ModelManager)
# ------------------------------------------------------------------

_BUILTIN_PROVIDERS: list[type[BaseProvider]] = [
    OpenAIProvider,
    DeepSeekProvider,
    OpenRouterProvider,
]

# Fallback model lists used when config.yaml provides none.
_DEFAULT_MODEL_LISTS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4.1", "o4-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "openrouter": ["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-sonnet-4"],
}


class ModelNotFoundError(Exception):
    """Raised when a requested provider or model is unknown."""


class ModelManager:
    """Manages provider registration and model selection.

    Belongs to the Core Business Layer.  It reads model *lists* from
    ``LLMConfig.models`` (falling back to built-in defaults) so that
    adding or removing models requires a config change, not a code change.

    Usage::

        mgr = ModelManager(config.llm)
        model = mgr.get_current_model()
        mgr.switch_model("deepseek", "deepseek-chat")
    """

    def __init__(self, config: LLMConfig) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._model_lists: dict[str, list[str]] = {}
        self._current_provider_name: str
        self._current_model_name: str
        self._config = config

        # Register built-in providers.
        for cls in _BUILTIN_PROVIDERS:
            self.register_provider(cls())

        # Load model lists: config first, fall back to defaults.
        for name in self._providers:
            if name in config.models:
                self._model_lists[name] = list(config.models[name])
            else:
                self._model_lists[name] = list(_DEFAULT_MODEL_LISTS.get(name, []))

        # Initial selection from config.
        provider = config.provider or "openai"
        model = config.model or self._providers[provider].default_model
        self.switch_model(provider, model)

    # ------------------------------------------------------------------
    # Provider registry
    # ------------------------------------------------------------------

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a provider instance.

        If a provider with the same name already exists it is replaced,
        allowing test code to inject custom providers.  Model lists are
        loaded from config (or defaults) at registration time.
        """
        self._providers[provider.name] = provider
        # Load model list for this provider if not already present.
        if provider.name not in self._model_lists:
            if provider.name in self._config.models:
                self._model_lists[provider.name] = list(self._config.models[provider.name])
            else:
                self._model_lists[provider.name] = list(_DEFAULT_MODEL_LISTS.get(provider.name, []))

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def switch_model(self, provider_name: str, model_name: str) -> None:
        """Switch to *model_name* under *provider_name*.

        Raises:
            ModelNotFoundError: If the provider or model is unknown.
        """
        if provider_name not in self._providers:
            raise ModelNotFoundError(
                f"Unknown provider: {provider_name!r}. "
                f"Available: {', '.join(self.list_providers())}"
            )
        available = self._model_lists.get(provider_name, [])
        if model_name not in available:
            raise ModelNotFoundError(
                f"Unknown model: {model_name!r} for provider {provider_name!r}. "
                f"Available: {', '.join(available)}"
            )
        self._current_provider_name = provider_name
        self._current_model_name = model_name

    def switch_provider(self, provider_name: str) -> None:
        """Switch to *provider_name*, auto-selecting its default model."""
        if provider_name not in self._providers:
            raise ModelNotFoundError(f"Unknown provider: {provider_name!r}")
        default = self._providers[provider_name].default_model
        self.switch_model(provider_name, default)

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def get_current_model(self) -> ChatOpenAI:
        """Create and return a fresh ``ChatOpenAI`` instance.

        No caching — every call produces a new object so that parameter
        changes (temperature, timeout, etc.) are always honoured.
        """
        provider = self._providers[self._current_provider_name]
        return provider.create_model(
            model_name=self._current_model_name,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
        )

    def get_model_for(self, provider_name: str, model_name: str) -> ChatOpenAI:
        """Create a model for an arbitrary provider+model combo (for preview)."""
        if provider_name not in self._providers:
            raise ModelNotFoundError(f"Unknown provider: {provider_name!r}")
        provider = self._providers[provider_name]
        return provider.create_model(
            model_name=model_name,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def current_provider(self) -> str:
        """Name of the currently active provider."""
        return self._current_provider_name

    @property
    def current_model(self) -> str:
        """Name of the currently active model."""
        return self._current_model_name

    def list_providers(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers.keys())

    def list_models(self, provider_name: str | None = None) -> list[str]:
        """Return model list for *provider_name* (defaults to current)."""
        name = provider_name or self._current_provider_name
        return list(self._model_lists.get(name, []))

    def get_provider_info(self, provider_name: str) -> dict[str, Any]:
        """Return metadata for *provider_name* (name, base_url, models)."""
        if provider_name not in self._providers:
            raise ModelNotFoundError(f"Unknown provider: {provider_name!r}")
        p = self._providers[provider_name]
        return {
            "name": p.name,
            "base_url": p.base_url,
            "default_model": p.default_model,
            "models": self._model_lists.get(provider_name, []),
        }
