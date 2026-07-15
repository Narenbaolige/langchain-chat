"""Tests for ModelManager and Provider abstraction."""

from __future__ import annotations

import pytest

from langchain_chat.core.config_models import LLMConfig
from langchain_chat.core.model_manager import ModelManager, ModelNotFoundError
from langchain_chat.core.provider import (
    BaseProvider,
    ChatAnywhereProvider,
    DeepSeekProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

# ------------------------------------------------------------------
# Provider tests
# ------------------------------------------------------------------


class TestProviders:
    """Tests for built-in provider metadata."""

    def test_openai_provider_metadata(self) -> None:
        p = OpenAIProvider()
        assert p.name == "openai"
        assert p.base_url is None
        assert p.api_key_env == "OPENAI_API_KEY"
        assert p.default_model == "gpt-4o-mini"

    def test_deepseek_provider_metadata(self) -> None:
        p = DeepSeekProvider()
        assert p.name == "deepseek"
        assert p.base_url == "https://api.deepseek.com/v1"
        assert p.api_key_env == "DEEPSEEK_API_KEY"
        assert p.default_model == "deepseek-chat"

    def test_openrouter_provider_metadata(self) -> None:
        p = OpenRouterProvider()
        assert p.name == "openrouter"
        assert p.base_url == "https://openrouter.ai/api/v1"
        assert p.api_key_env == "OPENROUTER_API_KEY"
        assert p.default_model == "openai/gpt-4o-mini"

    def test_chatanywhere_provider_metadata(self) -> None:
        p = ChatAnywhereProvider()
        assert p.name == "chatanywhere"
        assert p.base_url == "https://api.chatanywhere.tech/v1"
        assert p.api_key_env == "OPENAI_API_KEY"
        assert p.default_model == "gpt-4o-mini-ca"

    def test_all_providers_extend_base(self) -> None:
        for cls in [
            OpenAIProvider,
            DeepSeekProvider,
            OpenRouterProvider,
            ChatAnywhereProvider,
        ]:
            assert issubclass(cls, BaseProvider)

    def test_create_model_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
        p = OpenAIProvider()
        model = p.create_model("gpt-4o-mini", temperature=0.5, max_tokens=100)
        assert model is not None
        assert model.model_name == "gpt-4o-mini"
        assert model.temperature == 0.5
        assert model.max_tokens == 100


# ------------------------------------------------------------------
# ModelManager tests
# ------------------------------------------------------------------


@pytest.fixture
def mgr() -> ModelManager:
    """Return a ModelManager with built-in providers and default config."""
    return ModelManager(LLMConfig(model="gpt-4o-mini"))


class TestModelManagerInit:
    """Tests for ModelManager initialisation."""

    def test_registers_builtin_providers(self, mgr: ModelManager) -> None:
        providers = mgr.list_providers()
        assert "openai" in providers
        assert "deepseek" in providers
        assert "openrouter" in providers
        assert "chatanywhere" in providers

    def test_default_provider_is_from_config(self, mgr: ModelManager) -> None:
        assert mgr.current_provider == "openai"

    def test_default_model_is_from_config(self, mgr: ModelManager) -> None:
        assert mgr.current_model == "gpt-4o-mini"

    def test_current_model_exists_in_list(self, mgr: ModelManager) -> None:
        models = mgr.list_models()
        assert mgr.current_model in models


class TestModelSwitching:
    """Tests for switch_model and switch_provider."""

    def test_switch_model_success(self, mgr: ModelManager) -> None:
        mgr.switch_model("openai", "gpt-4o")
        assert mgr.current_model == "gpt-4o"
        assert mgr.current_provider == "openai"

    def test_switch_provider_success(self, mgr: ModelManager) -> None:
        mgr.switch_provider("deepseek")
        assert mgr.current_provider == "deepseek"
        assert mgr.current_model == "deepseek-chat"

    def test_switch_to_nonexistent_provider_raises(self, mgr: ModelManager) -> None:
        with pytest.raises(ModelNotFoundError, match="Unknown provider"):
            mgr.switch_model("nonexistent", "x")

    def test_switch_to_nonexistent_model_raises(self, mgr: ModelManager) -> None:
        with pytest.raises(ModelNotFoundError, match="Unknown model"):
            mgr.switch_model("openai", "nonexistent-model-xyz")

    def test_switch_provider_nonexistent_raises(self, mgr: ModelManager) -> None:
        with pytest.raises(ModelNotFoundError, match="Unknown provider"):
            mgr.switch_provider("nonexistent")


class TestModelFactory:
    """Tests for get_current_model."""

    def test_get_current_model_returns_object(
        self, mgr: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
        model = mgr.get_current_model()
        assert model is not None
        assert model.model_name == "gpt-4o-mini"

    def test_get_current_model_no_cache(
        self, mgr: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each call creates a fresh instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
        m1 = mgr.get_current_model()
        m2 = mgr.get_current_model()
        assert m1 is not m2  # different objects — no caching

    def test_get_current_model_reflects_switch(
        self, mgr: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
        mgr.switch_model("openai", "gpt-4-turbo")
        model = mgr.get_current_model()
        assert model.model_name == "gpt-4-turbo"

    def test_get_model_for_arbitrary_combo(
        self, mgr: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy")
        model = mgr.get_model_for("deepseek", "deepseek-reasoner")
        assert model.model_name == "deepseek-reasoner"


class TestQueries:
    """Tests for list_providers, list_models, get_provider_info."""

    def test_list_providers_sorted(self, mgr: ModelManager) -> None:
        providers = mgr.list_providers()
        assert providers == sorted(providers)

    def test_list_models_defaults_to_current(self, mgr: ModelManager) -> None:
        models = mgr.list_models()
        assert "gpt-4o-mini" in models

    def test_list_models_for_specific_provider(self, mgr: ModelManager) -> None:
        models = mgr.list_models("deepseek")
        assert "deepseek-chat" in models

    def test_get_provider_info(self, mgr: ModelManager) -> None:
        info = mgr.get_provider_info("openai")
        assert info["name"] == "openai"
        assert info["base_url"] is None
        assert "models" in info

    def test_get_provider_info_missing_raises(self, mgr: ModelManager) -> None:
        with pytest.raises(ModelNotFoundError):
            mgr.get_provider_info("nonexistent")


class TestConfigModelLists:
    """Tests that config model lists override defaults."""

    def test_config_provides_model_lists(self) -> None:
        """Config model lists are used when available, falling back to defaults."""

        class _TestProvider(BaseProvider):
            @property
            def name(self) -> str:
                return "testprovider"

            @property
            def base_url(self) -> str | None:
                return None

            @property
            def api_key_env(self) -> str:
                return "TEST_KEY"

            @property
            def default_model(self) -> str:
                return "m1"

        # Register provider first, then verify its model list.
        mgr = ModelManager(LLMConfig())
        mgr.register_provider(_TestProvider())
        # Unknown provider → empty fallback list.
        assert mgr.list_models("testprovider") == []

        # Re-register with config that has model lists.
        config = LLMConfig(
            models={"testprovider": ["m1", "m2"]},
        )
        mgr2 = ModelManager(config)
        mgr2.register_provider(_TestProvider())
        mgr2.switch_model("testprovider", "m1")
        assert mgr2.list_models("testprovider") == ["m1", "m2"]


class TestRegisterProvider:
    """Tests for register_provider."""

    def test_register_custom_provider(self, mgr: ModelManager) -> None:
        class _Custom(BaseProvider):
            @property
            def name(self) -> str:
                return "custom"

            @property
            def base_url(self) -> str | None:
                return "http://localhost"

            @property
            def api_key_env(self) -> str:
                return "CUSTOM_KEY"

            @property
            def default_model(self) -> str:
                return "custom-model"

        mgr.register_provider(_Custom())
        assert "custom" in mgr.list_providers()

    def test_register_replaces_existing(self, mgr: ModelManager) -> None:
        class _Replacement(BaseProvider):
            @property
            def name(self) -> str:
                return "openai"

            @property
            def base_url(self) -> str | None:
                return "http://override"

            @property
            def api_key_env(self) -> str:
                return "OVERRIDE_KEY"

            @property
            def default_model(self) -> str:
                return "override-model"

        mgr.register_provider(_Replacement())
        assert mgr.get_provider_info("openai")["base_url"] == "http://override"
