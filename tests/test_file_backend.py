"""Tests for FileBackend — structure, factory, and full CRUD integration."""

from __future__ import annotations

import pytest

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.storage.base import StorageBackend
from langchain_chat.storage.factory import StorageFactory
from langchain_chat.storage.file_backend import FileBackend

# ------------------------------------------------------------------
# Fixture
# ------------------------------------------------------------------


@pytest.fixture
async def backend(tmp_path) -> FileBackend:
    """Return an initialised FileBackend using a temp directory."""
    cfg = StorageConfig(type="file", database=str(tmp_path / "file_storage"))
    be = FileBackend(cfg)
    await be.initialize()
    yield be
    await be.close()


# ------------------------------------------------------------------
# Structure tests
# ------------------------------------------------------------------


class TestFileBackendStructure:
    """Verify FileBackend satisfies the StorageBackend contract."""

    def test_extends_storage_backend(self) -> None:
        assert issubclass(FileBackend, StorageBackend)

    def test_all_abstract_methods_implemented(self) -> None:
        abstract = set(StorageBackend.__abstractmethods__)
        concrete = set(dir(FileBackend))
        missing = abstract - concrete
        assert not missing, f"FileBackend missing methods: {missing}"


# ------------------------------------------------------------------
# Factory tests
# ------------------------------------------------------------------


class TestFactoryFile:
    """Verify StorageFactory creates FileBackend correctly."""

    def test_factory_creates_file_backend(self) -> None:
        cfg = StorageConfig(type="file", database="/tmp/test_file")
        backend = StorageFactory.create(cfg)
        assert isinstance(backend, FileBackend)

    def test_factory_file_returns_storage_backend(self) -> None:
        cfg = StorageConfig(type="file", database="/tmp/test_file")
        backend = StorageFactory.create(cfg)
        assert isinstance(backend, StorageBackend)


# ------------------------------------------------------------------
# CRUD integration tests
# ------------------------------------------------------------------


class TestFileUserCRUD:
    """User CRUD via FileBackend."""

    async def test_create_and_get_user(self, backend: FileBackend) -> None:
        user = await backend.create_user("alice")
        assert user["username"] == "alice"
        assert user["id"] > 0

        fetched = await backend.get_user(user["id"])
        assert fetched is not None
        assert fetched["username"] == "alice"

    async def test_get_user_missing(self, backend: FileBackend) -> None:
        assert await backend.get_user(9999) is None

    async def test_get_user_by_name(self, backend: FileBackend) -> None:
        await backend.create_user("bob")
        fetched = await backend.get_user_by_name("bob")
        assert fetched is not None
        assert fetched["username"] == "bob"

    async def test_list_users(self, backend: FileBackend) -> None:
        await backend.create_user("a")
        await backend.create_user("b")
        users = await backend.list_users()
        assert len(users) == 2

    async def test_delete_user(self, backend: FileBackend) -> None:
        user = await backend.create_user("temp")
        assert await backend.delete_user(user["id"]) is True
        assert await backend.get_user(user["id"]) is None

    async def test_delete_user_missing(self, backend: FileBackend) -> None:
        assert await backend.delete_user(9999) is False


class TestFileSessionCRUD:
    """Session CRUD via FileBackend."""

    async def test_create_and_get_session(self, backend: FileBackend) -> None:
        await backend.create_user("u1")
        session = await backend.create_session(user_id=1, title="Test")
        assert session["title"] == "Test"
        assert "created_at" in session

        fetched = await backend.get_session(session["id"])
        assert fetched is not None
        assert fetched["title"] == "Test"

    async def test_list_sessions_filtered(self, backend: FileBackend) -> None:
        await backend.create_user("u1")
        await backend.create_user("u2")
        await backend.create_session(user_id=1, title="S1")
        await backend.create_session(user_id=2, title="S2")

        u1 = await backend.list_sessions(user_id=1)
        assert len(u1) == 1

    async def test_update_and_search(self, backend: FileBackend) -> None:
        await backend.create_user("u1")
        s = await backend.create_session(user_id=1, title="Alpha")
        updated = await backend.update_session(s["id"], "Beta")
        assert updated["title"] == "Beta"

        results = await backend.search_sessions(1, "Beta")
        assert len(results) == 1

    async def test_delete_session_cascades(self, backend: FileBackend) -> None:
        await backend.create_user("u1")
        s = await backend.create_session(user_id=1, title="X")
        await backend.add_message(s["id"], "user", "hello")
        await backend.delete_session(s["id"])
        msgs = await backend.get_messages(s["id"])
        assert msgs == []


class TestFileMessageCRUD:
    """Message CRUD via FileBackend."""

    async def test_add_and_get_messages(self, backend: FileBackend) -> None:
        await backend.create_user("u1")
        s = await backend.create_session(user_id=1)
        m1 = await backend.add_message(s["id"], "user", "Q1")
        m2 = await backend.add_message(s["id"], "assistant", "A1")

        assert m1["role"] == "user"
        assert m2["role"] == "assistant"

        msgs = await backend.get_messages(s["id"])
        assert len(msgs) == 2

    async def test_search_messages(self, backend: FileBackend) -> None:
        await backend.create_user("u1")
        s = await backend.create_session(user_id=1)
        await backend.add_message(s["id"], "user", "Python question")
        await backend.add_message(s["id"], "assistant", "Java answer")

        results = await backend.search_messages(s["id"], "Python")
        assert len(results) == 1
        assert results[0]["content"] == "Python question"


class TestFilePresetCRUD:
    """Preset CRUD via FileBackend."""

    async def test_create_and_get_preset(self, backend: FileBackend) -> None:
        p = await backend.create_preset("greeting", "Hello!", "system")
        assert p["name"] == "greeting"
        assert p["prompt_type"] == "system"

        fetched = await backend.get_preset(p["id"])
        assert fetched is not None

    async def test_get_preset_by_name(self, backend: FileBackend) -> None:
        await backend.create_preset("p1", "content")
        p = await backend.get_preset_by_name("p1")
        assert p is not None

    async def test_list_and_delete(self, backend: FileBackend) -> None:
        p = await backend.create_preset("tmp", "x")
        presets = await backend.get_presets()
        assert len(presets) == 1

        assert await backend.delete_preset(p["id"]) is True
        assert await backend.get_preset(p["id"]) is None


class TestFileConfigOps:
    """Key-value config via FileBackend."""

    async def test_save_and_get(self, backend: FileBackend) -> None:
        await backend.save_config("theme", "dark")
        assert await backend.get_config("theme") == "dark"

    async def test_missing_key(self, backend: FileBackend) -> None:
        assert await backend.get_config("nonexistent") is None

    async def test_overwrite(self, backend: FileBackend) -> None:
        await backend.save_config("k", "v1")
        await backend.save_config("k", "v2")
        assert await backend.get_config("k") == "v2"
