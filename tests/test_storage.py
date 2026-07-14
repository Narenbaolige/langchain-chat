"""Tests for the storage layer.

Covers:
- Database initialization
- SQLiteBackend CRUD operations
- StorageFactory
"""

from __future__ import annotations

import pytest

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.storage.base import StorageBackend
from langchain_chat.storage.database import init_database
from langchain_chat.storage.factory import StorageFactory
from langchain_chat.storage.sqlite_backend import SQLiteBackend

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def backend() -> SQLiteBackend:
    """Return an initialized SQLiteBackend backed by :memory:."""
    config = StorageConfig(type="sqlite", database=":memory:")
    be = SQLiteBackend(config)
    await be.initialize()
    yield be
    await be.close()


# ------------------------------------------------------------------
# Database init tests
# ------------------------------------------------------------------


class TestDatabaseInit:
    """Tests for init_database()."""

    async def test_init_creates_tables_in_memory(self) -> None:
        """Database init runs without error on :memory:."""
        await init_database(":memory:")

    async def test_init_creates_tables_on_disk(self, tmp_path) -> None:
        """Database init creates the db file and parent dirs on disk."""
        db_path = str(tmp_path / "subdir" / "test.db")
        await init_database(db_path)

        import os

        assert os.path.isfile(db_path)


# ------------------------------------------------------------------
# User tests
# ------------------------------------------------------------------


class TestUserOperations:
    """Tests for create_user / get_user / delete_user."""

    async def test_create_user(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("alice")
        assert user["username"] == "alice"
        assert isinstance(user["id"], int)
        assert "created_at" in user

    async def test_get_user_exists(self, backend: SQLiteBackend) -> None:
        created = await backend.create_user("bob")
        fetched = await backend.get_user(created["id"])
        assert fetched is not None
        assert fetched["username"] == "bob"

    async def test_get_user_missing(self, backend: SQLiteBackend) -> None:
        assert await backend.get_user(9999) is None

    async def test_delete_user(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("carol")
        assert await backend.delete_user(user["id"]) is True
        assert await backend.get_user(user["id"]) is None

    async def test_delete_user_missing(self, backend: SQLiteBackend) -> None:
        assert await backend.delete_user(9999) is False


# ------------------------------------------------------------------
# Session tests
# ------------------------------------------------------------------


class TestSessionOperations:
    """Tests for create_session / get_session / list_sessions / delete_session."""

    async def test_create_session(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("dave")
        session = await backend.create_session(user["id"], "My Chat")
        assert session["title"] == "My Chat"
        assert session["user_id"] == user["id"]
        assert isinstance(session["id"], int)

    async def test_create_session_default_title(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("eve")
        session = await backend.create_session(user["id"])
        assert session["title"] == ""

    async def test_get_session_exists(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("frank")
        created = await backend.create_session(user["id"])
        fetched = await backend.get_session(created["id"])
        assert fetched is not None
        assert fetched["title"] == ""

    async def test_get_session_missing(self, backend: SQLiteBackend) -> None:
        assert await backend.get_session(9999) is None

    async def test_list_sessions(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("grace")
        await backend.create_session(user["id"], "Chat A")
        await backend.create_session(user["id"], "Chat B")

        sessions = await backend.list_sessions(user["id"])
        assert len(sessions) == 2

    async def test_list_sessions_filter_by_user(self, backend: SQLiteBackend) -> None:
        u1 = await backend.create_user("henry")
        u2 = await backend.create_user("iris")
        await backend.create_session(u1["id"], "H1")
        await backend.create_session(u1["id"], "H2")
        await backend.create_session(u2["id"], "I1")

        assert len(await backend.list_sessions(u1["id"])) == 2
        assert len(await backend.list_sessions(u2["id"])) == 1

    async def test_list_sessions_all(self, backend: SQLiteBackend) -> None:
        u1 = await backend.create_user("jack")
        u2 = await backend.create_user("kate")
        await backend.create_session(u1["id"])
        await backend.create_session(u2["id"])

        all_sessions = await backend.list_sessions()
        assert len(all_sessions) == 2

    async def test_delete_session(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("leo")
        session = await backend.create_session(user["id"])
        assert await backend.delete_session(session["id"]) is True
        assert await backend.get_session(session["id"]) is None

    async def test_delete_session_missing(self, backend: SQLiteBackend) -> None:
        assert await backend.delete_session(9999) is False


# ------------------------------------------------------------------
# Message tests
# ------------------------------------------------------------------


class TestMessageOperations:
    """Tests for add_message / get_messages."""

    async def test_add_message(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("mia")
        session = await backend.create_session(user["id"])
        msg = await backend.add_message(session["id"], "user", "Hello!")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello!"
        assert msg["session_id"] == session["id"]

    async def test_get_messages_ordered(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("nick")
        session = await backend.create_session(user["id"])
        await backend.add_message(session["id"], "user", "First")
        await backend.add_message(session["id"], "assistant", "Second")
        await backend.add_message(session["id"], "user", "Third")

        messages = await backend.get_messages(session["id"])
        assert len(messages) == 3
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"

    async def test_get_messages_empty_session(self, backend: SQLiteBackend) -> None:
        user = await backend.create_user("olivia")
        session = await backend.create_session(user["id"])
        assert await backend.get_messages(session["id"]) == []


# ------------------------------------------------------------------
# Preset tests
# ------------------------------------------------------------------


class TestPresetOperations:
    """Tests for create_preset / get_presets."""

    async def test_create_preset(self, backend: SQLiteBackend) -> None:
        preset = await backend.create_preset("Greeting", "Hello, {name}!")
        assert preset["name"] == "Greeting"
        assert preset["content"] == "Hello, {name}!"
        assert isinstance(preset["id"], int)

    async def test_get_presets(self, backend: SQLiteBackend) -> None:
        await backend.create_preset("P1", "Content 1")
        await backend.create_preset("P2", "Content 2")

        presets = await backend.get_presets()
        assert len(presets) == 2
        names = {p["name"] for p in presets}
        assert names == {"P1", "P2"}

    async def test_get_presets_empty(self, backend: SQLiteBackend) -> None:
        assert await backend.get_presets() == []


# ------------------------------------------------------------------
# Config tests
# ------------------------------------------------------------------


class TestConfigOperations:
    """Tests for get_config / save_config."""

    async def test_save_and_get_config(self, backend: SQLiteBackend) -> None:
        await backend.save_config("theme", "dark")
        assert await backend.get_config("theme") == "dark"

    async def test_get_config_missing(self, backend: SQLiteBackend) -> None:
        assert await backend.get_config("nonexistent") is None

    async def test_save_config_overwrite(self, backend: SQLiteBackend) -> None:
        await backend.save_config("key1", "val1")
        await backend.save_config("key1", "val2")
        assert await backend.get_config("key1") == "val2"


# ------------------------------------------------------------------
# Factory tests
# ------------------------------------------------------------------


class TestStorageFactory:
    """Tests for StorageFactory.create()."""

    def test_create_sqlite(self) -> None:
        config = StorageConfig(type="sqlite", database=":memory:")
        backend = StorageFactory.create(config)
        assert isinstance(backend, SQLiteBackend)

    def test_create_unknown_type(self) -> None:
        config = StorageConfig(type="postgres", database="")
        with pytest.raises(ValueError, match="Unknown storage type"):
            StorageFactory.create(config)

    def test_factory_returns_storage_backend_interface(self) -> None:
        """Factory must return a StorageBackend, never a concrete type directly."""
        config = StorageConfig(type="sqlite", database=":memory:")
        backend = StorageFactory.create(config)
        assert isinstance(backend, StorageBackend)


# ------------------------------------------------------------------
# Interface contract tests
# ------------------------------------------------------------------


class TestStorageBackendInterface:
    """Verify that SQLiteBackend is a proper StorageBackend."""

    def test_sqlite_backend_is_storage_backend(self) -> None:
        assert issubclass(SQLiteBackend, StorageBackend)

    async def test_all_abstract_methods_implemented(self) -> None:
        """Instantiate + call every method to confirm no NotImplementedError."""
        config = StorageConfig(type="sqlite", database=":memory:")
        be = SQLiteBackend(config)
        await be.initialize()

        try:
            user = await be.create_user("testuser")
            uid = user["id"]

            session = await be.create_session(uid, "test")
            sid = session["id"]

            await be.add_message(sid, "user", "hello")
            await be.get_messages(sid)

            await be.get_user(uid)
            await be.list_sessions(uid)
            await be.list_sessions()

            await be.create_preset("p", "c")
            await be.get_presets()

            await be.save_config("k", "v")
            await be.get_config("k")

            await be.delete_session(sid)
            await be.delete_user(uid)
        finally:
            await be.close()
