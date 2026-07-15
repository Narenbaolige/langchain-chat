"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest
from fakes import FakeModel  # noqa: F401 — re-exported for test modules

from langchain_chat.core.config_models import LLMConfig, StorageConfig
from langchain_chat.storage.sqlite_backend import SQLiteBackend

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def fake_model() -> FakeModel:
    """Return a fresh FakeModel instance."""
    return FakeModel()


@pytest.fixture
def llm_config() -> LLMConfig:
    """A minimal LLMConfig for testing."""
    return LLMConfig(
        model="gpt-4o-mini", temperature=0.7, max_tokens=100, timeout=30, max_retries=2
    )


@pytest.fixture
async def storage_backend():
    """Return an initialized :memory: SQLiteBackend."""
    config = StorageConfig(type="sqlite", database=":memory:")
    be = SQLiteBackend(config)
    await be.initialize()
    yield be
    await be.close()
