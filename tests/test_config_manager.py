"""Tests for ConfigManager — multi-env loading, deep merge, env overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from langchain_chat.core.config_manager import ConfigManager, _coerce_env_value

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


# ------------------------------------------------------------------
# Deep merge tests
# ------------------------------------------------------------------


class TestDeepMerge:
    """Unit tests for _deep_merge."""

    def test_merge_nested_dicts(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99}}
        result = ConfigManager._deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99}, "b": 3}

    def test_merge_scalar_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"a": 10}
        result = ConfigManager._deep_merge(base, override)
        assert result == {"a": 10, "b": 2}

    def test_merge_list_replaces(self) -> None:
        base = {"a": [1, 2, 3]}
        override = {"a": [4, 5]}
        result = ConfigManager._deep_merge(base, override)
        assert result == {"a": [4, 5]}

    def test_merge_new_key(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        result = ConfigManager._deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_merge_empty_override(self) -> None:
        base = {"a": 1}
        result = ConfigManager._deep_merge(base, {})
        assert result == {"a": 1}

    def test_merge_three_levels(self) -> None:
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = ConfigManager._deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}


# ------------------------------------------------------------------
# Env value coercion tests
# ------------------------------------------------------------------


class TestCoerceEnvValue:
    def test_true_false(self) -> None:
        assert _coerce_env_value("true") is True
        assert _coerce_env_value("True") is True
        assert _coerce_env_value("false") is False

    def test_int(self) -> None:
        assert _coerce_env_value("42") == 42
        assert _coerce_env_value("0") == 0

    def test_float(self) -> None:
        assert _coerce_env_value("3.14") == 3.14

    def test_string(self) -> None:
        assert _coerce_env_value("hello") == "hello"


# ------------------------------------------------------------------
# Env var override tests
# ------------------------------------------------------------------


class TestEnvVarOverrides:
    """Tests for _apply_env_overrides."""

    def test_simple_override(self) -> None:
        raw = {"storage": {"type": "sqlite"}}
        os.environ["LANGCHAIN_STORAGE__TYPE"] = "mysql"
        try:
            ConfigManager._apply_env_overrides(raw)
            assert raw["storage"]["type"] == "mysql"
        finally:
            del os.environ["LANGCHAIN_STORAGE__TYPE"]

    def test_nested_override(self) -> None:
        raw = {"storage": {"mysql": {"host": "localhost"}}}
        os.environ["LANGCHAIN_STORAGE__MYSQL__HOST"] = "db.prod.com"
        try:
            ConfigManager._apply_env_overrides(raw)
            assert raw["storage"]["mysql"]["host"] == "db.prod.com"
        finally:
            del os.environ["LANGCHAIN_STORAGE__MYSQL__HOST"]

    def test_bool_coercion(self) -> None:
        raw = {"app": {"debug": False}}
        os.environ["LANGCHAIN_APP__DEBUG"] = "true"
        try:
            ConfigManager._apply_env_overrides(raw)
            assert raw["app"]["debug"] is True
        finally:
            del os.environ["LANGCHAIN_APP__DEBUG"]

    def test_ignores_non_langchain_vars(self) -> None:
        raw = {"app": {"name": "test"}}
        os.environ["OTHER_VAR"] = "should_be_ignored"
        try:
            ConfigManager._apply_env_overrides(raw)
            assert raw["app"]["name"] == "test"
        finally:
            del os.environ["OTHER_VAR"]

    def test_empty_value_ignored(self) -> None:
        raw = {"app": {"name": "test"}}
        os.environ["LANGCHAIN_APP_NAME"] = ""
        try:
            ConfigManager._apply_env_overrides(raw)
            assert raw["app"]["name"] == "test"
        finally:
            del os.environ["LANGCHAIN_APP_NAME"]


# ------------------------------------------------------------------
# Environment resolution tests
# ------------------------------------------------------------------


class TestResolveEnv:
    def test_default_is_dev(self) -> None:
        old = os.environ.pop("APP_ENV", None)
        try:
            assert ConfigManager._resolve_env() == "dev"
        finally:
            if old is not None:
                os.environ["APP_ENV"] = old

    def test_reads_app_env(self) -> None:
        os.environ["APP_ENV"] = "prod"
        try:
            assert ConfigManager._resolve_env() == "prod"
        finally:
            del os.environ["APP_ENV"]

    def test_case_insensitive(self) -> None:
        os.environ["APP_ENV"] = "PROD"
        try:
            assert ConfigManager._resolve_env() == "prod"
        finally:
            del os.environ["APP_ENV"]


# ------------------------------------------------------------------
# Config loading tests (uses real project config files)
# ------------------------------------------------------------------


class TestConfigLoading:
    """Tests that load the actual project config files."""

    def test_load_dev_config(self) -> None:
        """Default env loads config.yaml + config.dev.yaml."""
        old = os.environ.pop("APP_ENV", None)
        try:
            cm = ConfigManager(str(CONFIG_DIR))
            cfg = cm.load()
            assert cfg.app.env == "dev"
            assert cfg.app.debug is True
            assert cfg.logging.level == "DEBUG"
        finally:
            if old is not None:
                os.environ["APP_ENV"] = old

    def test_load_test_config(self) -> None:
        """APP_ENV=test loads config.yaml + config.test.yaml."""
        os.environ["APP_ENV"] = "test"
        try:
            cm = ConfigManager(str(CONFIG_DIR))
            cfg = cm.load()
            assert cfg.app.env == "test"
            assert cfg.storage.type == "sqlite"
            assert cfg.storage.database == ":memory:"
            assert cfg.logging.level == "WARNING"
        finally:
            del os.environ["APP_ENV"]

    def test_load_prod_config(self) -> None:
        """APP_ENV=prod loads config.yaml + config.prod.yaml."""
        os.environ["APP_ENV"] = "prod"
        try:
            cm = ConfigManager(str(CONFIG_DIR))
            cfg = cm.load()
            assert cfg.app.env == "prod"
            assert cfg.app.debug is False
            assert cfg.storage.type == "mysql"
            assert cfg.logging.level == "INFO"
        finally:
            del os.environ["APP_ENV"]

    def test_invalid_env_graceful(self) -> None:
        """Unknown APP_ENV just skips the env file — loads base config."""
        os.environ["APP_ENV"] = "nonexistent"
        try:
            cm = ConfigManager(str(CONFIG_DIR))
            cfg = cm.load()
            # No config.nonexistent.yaml → base config.yaml used
            # Base config has app.env = "dev"
            assert cfg.storage.type == "sqlite"
        finally:
            del os.environ["APP_ENV"]

    def test_deep_merge_preserves_llm_models(self) -> None:
        """Environment config must not wipe llm.models from base config."""
        os.environ["APP_ENV"] = "test"
        try:
            cm = ConfigManager(str(CONFIG_DIR))
            cfg = cm.load()
            # test config doesn't define llm.models — should inherit from base
            assert "openai" in cfg.llm.models
            assert "gpt-4o-mini" in cfg.llm.models["openai"]
        finally:
            del os.environ["APP_ENV"]
