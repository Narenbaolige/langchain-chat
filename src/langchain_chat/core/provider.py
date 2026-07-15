"""Provider abstraction for OpenAI-compatible LLM backends.

Defines a BaseProvider ABC and built-in concrete providers.  New providers
can be added by subclassing BaseProvider — ChatEngine and ModelManager
require no changes (Open-Closed Principle).

All providers use ``ChatOpenAI`` under the hood, varying only ``base_url``
and ``api_key``.  Model *lists* come from configuration, not from the
provider classes — see :class:`ModelManager`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import ChatOpenAI


class BaseProvider(ABC):
    """Abstract OpenAI-compatible provider.

    Each concrete provider specifies its API endpoint URL and the
    environment variable that holds its API key.  Model lists are
    managed externally (in config.yaml) so they can be changed
    without touching code.
    """

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier, e.g. ``"openai"``."""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str | None:
        """API base URL.  Return ``None`` for the OpenAI default."""
        ...

    @property
    @abstractmethod
    def api_key_env(self) -> str:
        """Environment variable that holds the API key."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model name when switching to this provider."""
        ...

    # ------------------------------------------------------------------
    # Shared factory
    # ------------------------------------------------------------------

    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> ChatOpenAI:
        """Build a fresh ``ChatOpenAI`` instance configured for this provider.

        Args:
            model_name: Model identifier recognised by the provider.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on transient failures.

        Returns:
            A configured ``ChatOpenAI`` instance.
        """
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url

        api_key = os.getenv(self.api_key_env)
        if api_key:
            kwargs["api_key"] = api_key

        return ChatOpenAI(**kwargs)


# ------------------------------------------------------------------
# Built-in providers
# ------------------------------------------------------------------


class OpenAIProvider(BaseProvider):
    """OpenAI official API."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def base_url(self) -> str | None:
        return None

    @property
    def api_key_env(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"


class DeepSeekProvider(BaseProvider):
    """DeepSeek API (OpenAI-compatible)."""

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def base_url(self) -> str | None:
        return "https://api.deepseek.com/v1"

    @property
    def api_key_env(self) -> str:
        return "DEEPSEEK_API_KEY"

    @property
    def default_model(self) -> str:
        return "deepseek-chat"


class OpenRouterProvider(BaseProvider):
    """OpenRouter API gateway."""

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def base_url(self) -> str | None:
        return "https://openrouter.ai/api/v1"

    @property
    def api_key_env(self) -> str:
        return "OPENROUTER_API_KEY"

    @property
    def default_model(self) -> str:
        return "openai/gpt-4o-mini"


class ChatAnywhereProvider(BaseProvider):
    """ChatAnywhere free API (OpenAI-compatible proxy).

    Provides free access to GPT and DeepSeek models via a single endpoint.
    Uses the same ``OPENAI_API_KEY`` environment variable — set it to your
    ChatAnywhere key.
    """

    @property
    def name(self) -> str:
        return "chatanywhere"

    @property
    def base_url(self) -> str | None:
        return "https://api.chatanywhere.tech/v1"

    @property
    def api_key_env(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini-ca"
