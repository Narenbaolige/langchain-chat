"""Tests for PromptManager — the prompt / preset management module."""

from __future__ import annotations

import pytest

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.core.prompt_manager import (
    DuplicatePresetError,
    PresetNotFoundError,
    PromptManager,
)
from langchain_chat.models.preset import Preset
from langchain_chat.storage.sqlite_backend import SQLiteBackend

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def manager() -> PromptManager:
    """Return a PromptManager backed by an initialized :memory: SQLite."""
    config = StorageConfig(type="sqlite", database=":memory:")
    be = SQLiteBackend(config)
    await be.initialize()
    yield PromptManager(be)
    await be.close()


# ------------------------------------------------------------------
# Create preset tests
# ------------------------------------------------------------------


class TestCreatePreset:
    """Tests for PromptManager.create_preset()."""

    async def test_create_user_preset(self, manager: PromptManager) -> None:
        preset = await manager.create_preset("greeting", "Hello, {name}!")
        assert isinstance(preset, Preset)
        assert preset.name == "greeting"
        assert preset.content == "Hello, {name}!"
        assert preset.prompt_type == "user"
        assert isinstance(preset.id, int)
        assert preset.id > 0

    async def test_create_system_preset(self, manager: PromptManager) -> None:
        preset = await manager.create_preset(
            "system-prompt", "You are a helpful assistant.", prompt_type="system"
        )
        assert preset.prompt_type == "system"
        assert preset.name == "system-prompt"

    async def test_trims_whitespace(self, manager: PromptManager) -> None:
        preset = await manager.create_preset("  hello  ", "  content  ")
        assert preset.name == "hello"
        assert preset.content == "content"

    async def test_duplicate_name_raises(self, manager: PromptManager) -> None:
        await manager.create_preset("dup", "content")
        with pytest.raises(DuplicatePresetError, match="dup"):
            await manager.create_preset("dup", "other content")

    async def test_empty_name_raises(self, manager: PromptManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            await manager.create_preset("", "content")
        with pytest.raises(ValueError, match="must not be empty"):
            await manager.create_preset("   ", "content")

    async def test_empty_content_raises(self, manager: PromptManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            await manager.create_preset("name", "")
        with pytest.raises(ValueError, match="must not be empty"):
            await manager.create_preset("name", "   ")


# ------------------------------------------------------------------
# Get preset tests
# ------------------------------------------------------------------


class TestGetPreset:
    """Tests for PromptManager.get_preset()."""

    async def test_get_preset_exists(self, manager: PromptManager) -> None:
        created = await manager.create_preset("test", "content")
        fetched = await manager.get_preset(created.id)
        assert fetched.id == created.id
        assert fetched.name == "test"

    async def test_get_preset_missing_raises(self, manager: PromptManager) -> None:
        with pytest.raises(PresetNotFoundError, match="9999"):
            await manager.get_preset(9999)


class TestGetPresetByName:
    """Tests for PromptManager.get_preset_by_name()."""

    async def test_get_by_name_exists(self, manager: PromptManager) -> None:
        created = await manager.create_preset("alpha", "a")
        fetched = await manager.get_preset_by_name("alpha")
        assert fetched.id == created.id

    async def test_get_by_name_missing_raises(self, manager: PromptManager) -> None:
        with pytest.raises(PresetNotFoundError, match="nope"):
            await manager.get_preset_by_name("nope")


# ------------------------------------------------------------------
# List presets tests
# ------------------------------------------------------------------


class TestListPresets:
    """Tests for PromptManager.list_presets()."""

    async def test_list_empty(self, manager: PromptManager) -> None:
        assert await manager.list_presets() == []

    async def test_list_multiple(self, manager: PromptManager) -> None:
        await manager.create_preset("a", "1")
        await manager.create_preset("b", "2", prompt_type="system")
        await manager.create_preset("c", "3")

        presets = await manager.list_presets()
        assert len(presets) == 3
        assert all(isinstance(p, Preset) for p in presets)
        names = {p.name for p in presets}
        assert names == {"a", "b", "c"}


# ------------------------------------------------------------------
# Delete preset tests
# ------------------------------------------------------------------


class TestDeletePreset:
    """Tests for PromptManager.delete_preset()."""

    async def test_delete_exists(self, manager: PromptManager) -> None:
        created = await manager.create_preset("del", "x")
        assert await manager.delete_preset(created.id) is True
        with pytest.raises(PresetNotFoundError):
            await manager.get_preset(created.id)

    async def test_delete_missing_idempotent(self, manager: PromptManager) -> None:
        assert await manager.delete_preset(9999) is False


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


class TestIntegration:
    """End-to-end preset lifecycle scenarios."""

    async def test_full_lifecycle(self, manager: PromptManager) -> None:
        # Create system + user presets
        sys = await manager.create_preset("system", "You are helpful.", prompt_type="system")
        usr = await manager.create_preset("user-greeting", "Hi!")

        # List
        all_presets = await manager.list_presets()
        assert len(all_presets) == 2

        # Get by id
        assert (await manager.get_preset(sys.id)).prompt_type == "system"

        # Get by name
        assert (await manager.get_preset_by_name("user-greeting")).id == usr.id

        # Delete
        assert await manager.delete_preset(sys.id) is True
        assert len(await manager.list_presets()) == 1

        # Verify gone
        with pytest.raises(PresetNotFoundError):
            await manager.get_preset(sys.id)
