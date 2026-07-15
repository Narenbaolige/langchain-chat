"""Configuration models using Pydantic.

Provides typed configuration schemas with built-in validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application-level configuration."""

    name: str = "langchain-chat"
    env: str = "dev"
    debug: bool = False


class MySQLConfig(BaseModel):
    """MySQL connection parameters (used when storage type is ``"mysql"``)."""

    host: str = "localhost"
    port: int = 3306
    database: str = "langchain_chat"
    user: str = "root"
    password: str = ""
    pool_size: int = 5


class StorageConfig(BaseModel):
    """Storage backend configuration.

    When ``type`` is ``"sqlite"`` the ``database`` field is a file path
    (or ``":memory:"``).  When ``type`` is ``"mysql"`` the nested
    ``mysql`` config is used for connection parameters.
    """

    type: str = "sqlite"
    database: str = "data/chat.db"
    mysql: MySQLConfig | None = None


class LLMConfig(BaseModel):
    """LLM provider and model configuration.

    ``models`` maps provider names to their available model lists.
    When empty, built-in defaults are used.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3
    models: dict[str, list[str]] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class ProjectConfig(BaseModel):
    """Root configuration aggregating all sub-configs."""

    app: AppConfig = Field(default_factory=AppConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
