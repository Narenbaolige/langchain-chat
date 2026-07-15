"""Configuration Manager.

Responsible for loading and providing unified access to all configuration:
- YAML configuration files (base + environment-specific)
- Environment variables (via .env)
- LANGCHAIN_* environment variable overrides
- Pydantic-validated config objects

Loading priority (highest last):
    1. config.yaml           (base defaults)
    2. config.{APP_ENV}.yaml  (environment overrides, deep merged)
    3. LANGCHAIN_* env vars   (individual key overrides)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from langchain_chat.core.config_models import ProjectConfig


class ConfigManager:
    """Loads and manages project configuration across environments.

    Usage::

        cm = ConfigManager()
        config = cm.load()       # auto-detects APP_ENV
        print(config.app.name)
    """

    def __init__(self, config_dir: Optional[str | Path] = None) -> None:
        """Initialize ConfigManager.

        Args:
            config_dir: Path to the config directory.
                       Defaults to ``<project_root>/config/``.
        """
        if config_dir is None:
            config_dir = self._resolve_config_dir()
        self._config_dir: Path = Path(config_dir)
        self._project_root: Path = self._config_dir.parent
        self._config: Optional[ProjectConfig] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> ProjectConfig:
        """Return the loaded (and cached) configuration."""
        if self._config is None:
            self._config = self.load()
        return self._config

    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return self._project_root

    def load(self) -> ProjectConfig:
        """Load all configuration sources and return a validated config object.

        Loading order (each step overrides the previous):
        1. ``.env`` file — populates ``os.environ``
        2. ``config.yaml`` — base defaults
        3. ``config.{APP_ENV}.yaml`` — environment-specific overrides (deep merge)
        4. ``LANGCHAIN_*`` env vars — individual key overrides
        5. Validate with Pydantic models
        """
        self._load_dotenv()

        # 1. Base config (required)
        raw = self._load_yaml("config.yaml", required=True)

        # 2. Environment-specific config (deep merge, optional)
        env = self._resolve_env()
        env_config = self._load_yaml(f"config.{env}.yaml", required=False)
        if env_config:
            raw = self._deep_merge(raw, env_config)

        # 3. LANGCHAIN_* env var overrides
        self._apply_env_overrides(raw)

        self._config = ProjectConfig(**raw)
        return self._config

    # ------------------------------------------------------------------
    # Internal — environment resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_env() -> str:
        """Return the current environment name from APP_ENV (default ``"dev"``)."""
        return os.getenv("APP_ENV", "dev").strip().lower()

    # ------------------------------------------------------------------
    # Internal — loading
    # ------------------------------------------------------------------

    def _load_dotenv(self) -> None:
        """Load ``.env`` file from the project root."""
        env_path = self._project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path))
        # Also load from cwd as fallback
        load_dotenv(override=False)

    def _load_yaml(self, filename: str, *, required: bool = False) -> dict[str, Any]:
        """Load a YAML file from the config directory.

        Args:
            filename: YAML file name (e.g. ``"config.yaml"``).
            required: If ``True``, raise ``FileNotFoundError`` when missing.

        Returns an empty dict if the file does not exist and *required* is False.
        """
        yaml_path = self._config_dir / filename
        if not yaml_path.exists():
            if required:
                raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            return {}
        with open(yaml_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # ------------------------------------------------------------------
    # Internal — deep merge
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge *override* into *base*.

        - Dicts are merged recursively.
        - Lists and scalars from *override* replace *base* values entirely.
        - Keys only in *base* are preserved.
        """
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Internal — env var overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_env_overrides(raw: dict[str, Any]) -> None:
        """Apply ``LANGCHAIN_*`` environment variables on top of *raw* (mutated in place).

        ``__`` (double underscore) separates nesting levels::

            LANGCHAIN_STORAGE__TYPE          → raw["storage"]["type"]
            LANGCHAIN_LLM__MODEL             → raw["llm"]["model"]
            LANGCHAIN_STORAGE__MYSQL__HOST   → raw["storage"]["mysql"]["host"]
            LANGCHAIN_LOG__FILE              → raw["logging"]["file"]
        """
        prefix = "LANGCHAIN_"
        for var_name, var_value in os.environ.items():
            if not var_name.startswith(prefix) or not var_value:
                continue
            path_str = var_name[len(prefix):].lower()
            parts = [p for p in path_str.split("__") if p]
            if not parts:
                continue

            # Navigate to the target dict, creating intermediate dicts as needed.
            target = raw
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    target[part] = _coerce_env_value(var_value)
                else:
                    if part not in target or not isinstance(target[part], dict):
                        target[part] = {}
                    target = target[part]

    # ------------------------------------------------------------------
    # Internal — config directory discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config_dir() -> Path:
        """Resolve the config directory relative to this source file.

        Heuristic: walk up from this file until we find a ``config/``
        directory that contains ``config.yaml``.
        """
        current = Path(__file__).resolve().parent
        for _ in range(10):
            candidate = current / "config"
            if (candidate / "config.yaml").exists():
                return candidate
            current = current.parent
        # Fallback: assume cwd is project root
        return Path.cwd() / "config"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _coerce_env_value(raw: str) -> Any:
    """Convert an env-var string to the most specific Python type."""
    v = raw.strip()
    # bool
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    # int
    try:
        return int(v)
    except ValueError:
        pass
    # float
    try:
        return float(v)
    except ValueError:
        pass
    return v


# ------------------------------------------------------------------
# Module-level convenience
# ------------------------------------------------------------------

_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: Optional[str | Path] = None) -> ConfigManager:
    """Return a module-level ConfigManager instance (lazy-init)."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


def get_config() -> ProjectConfig:
    """Return the current project configuration."""
    return get_config_manager().config
