"""Configuration Manager.

Responsible for loading and providing unified access to all configuration:
- YAML configuration files
- Environment variables (via .env)
- Pydantic-validated config objects
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from langchain_chat.core.config_models import ProjectConfig


class ConfigManager:
    """Loads and manages project configuration.

    Usage::

        cm = ConfigManager()
        config = cm.load()
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

        Loading order:
        1. ``.env`` file (if present) — populates ``os.environ``
        2. ``config.yaml`` — application settings
        3. Validate with Pydantic models
        """
        self._load_dotenv()
        raw = self._load_yaml()
        self._config = ProjectConfig(**raw)
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dotenv(self) -> None:
        """Load ``.env`` file from the project root."""
        env_path = self._project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path))
        # Also load from cwd as fallback
        load_dotenv(override=False)

    def _load_yaml(self) -> dict:
        """Load and parse the main YAML config file."""
        yaml_path = self._config_dir / "config.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_path}"
            )
        with open(yaml_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

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
