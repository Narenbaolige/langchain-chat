"""Tests for configuration management."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from langchain_chat.core.config_manager import ConfigManager
from langchain_chat.core.config_models import (
    AppConfig,
    LLMConfig,
    LoggingConfig,
    ProjectConfig,
    StorageConfig,
)

# ------------------------------------------------------------------
# Pydantic model tests
# ------------------------------------------------------------------


class TestAppConfig:
    """Tests for AppConfig model."""

    def test_defaults(self) -> None:
        cfg = AppConfig()
        assert cfg.name == "langchain-chat"
        assert cfg.env == "dev"
        assert cfg.debug is False

    def test_custom_values(self) -> None:
        cfg = AppConfig(name="my-app", env="prod", debug=True)
        assert cfg.name == "my-app"
        assert cfg.env == "prod"
        assert cfg.debug is True


class TestStorageConfig:
    """Tests for StorageConfig model."""

    def test_defaults(self) -> None:
        cfg = StorageConfig()
        assert cfg.type == "sqlite"
        assert cfg.database == "data/chat.db"


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.timeout == 60
        assert cfg.max_retries == 3
        assert cfg.models == {}


class TestProjectConfig:
    """Tests for the root ProjectConfig."""

    def test_default_factory(self) -> None:
        cfg = ProjectConfig()
        assert isinstance(cfg.app, AppConfig)
        assert isinstance(cfg.storage, StorageConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.logging, LoggingConfig)


# ------------------------------------------------------------------
# ConfigManager tests
# ------------------------------------------------------------------


class TestConfigManager:
    """Integration tests for ConfigManager with real config files."""

    @pytest.fixture
    def temp_config_dir(self) -> str:
        """Create a temporary config directory with a minimal config.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            yaml_content = """\
app:
  name: test-app
  env: testing
  debug: true

storage:
  type: memory
  database: ":memory:"

llm:
  provider: openai
  model: gpt-4o-mini

logging:
  level: DEBUG
"""
            (config_dir / "config.yaml").write_text(yaml_content, encoding="utf-8")
            yield str(config_dir)

    def test_load_yaml_from_temp_dir(self, temp_config_dir: str) -> None:
        """ConfigManager loads YAML correctly from a custom config dir."""
        cm = ConfigManager(config_dir=temp_config_dir)
        config = cm.load()

        assert config.app.name == "test-app"
        assert config.app.env == "testing"
        assert config.app.debug is True
        assert config.storage.type == "memory"
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o-mini"
        assert config.logging.level == "DEBUG"

    def test_load_uses_default_config_dir(self) -> None:
        """ConfigManager resolves the real project config directory."""
        cm = ConfigManager()
        config = cm.load()

        assert config.app.name == "langchain-chat"
        # env should be 'dev' from the actual config.yaml
        assert config.app.env == "dev"

    def test_config_cache(self, temp_config_dir: str) -> None:
        """Calling .config multiple times returns the same object."""
        cm = ConfigManager(config_dir=temp_config_dir)
        c1 = cm.config
        c2 = cm.config
        assert c1 is c2

    def test_missing_config_file(self) -> None:
        """ConfigManager raises FileNotFoundError for missing config.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty_config"
            empty_dir.mkdir()
            cm = ConfigManager(config_dir=str(empty_dir))
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                cm.load()

    def test_project_root_property(self, temp_config_dir: str) -> None:
        """project_root points to the parent of config_dir."""
        cm = ConfigManager(config_dir=temp_config_dir)
        assert cm.project_root == Path(temp_config_dir).parent


# ------------------------------------------------------------------
# Environment variable tests
# ------------------------------------------------------------------


class TestEnvLoading:
    """Tests for .env and environment variable support."""

    @pytest.fixture
    def temp_config_with_env(self) -> str:
        """Create a temp project with config.yaml and .env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            config_dir.mkdir()

            (config_dir / "config.yaml").write_text(
                "app:\n  name: env-test\n  env: dev\n", encoding="utf-8"
            )
            (root / ".env").write_text("OPENAI_API_KEY=sk-test-key-12345\n", encoding="utf-8")
            yield str(config_dir)

    def test_env_file_loads_into_os_environ(self, temp_config_with_env: str) -> None:
        """Loading config also loads .env into os.environ."""
        cm = ConfigManager(config_dir=temp_config_with_env)
        cm.load()

        assert os.getenv("OPENAI_API_KEY") == "sk-test-key-12345"
