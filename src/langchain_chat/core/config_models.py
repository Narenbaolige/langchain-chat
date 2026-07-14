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


class StorageConfig(BaseModel):
    """Storage backend configuration."""

    type: str = "sqlite"
    database: str = "data/chat.db"


class LLMConfig(BaseModel):
    """LLM provider and model configuration."""

    provider: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3


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
