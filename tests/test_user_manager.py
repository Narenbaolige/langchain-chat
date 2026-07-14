"""Tests for UserManager — the first Core Business Layer module."""

from __future__ import annotations

import pytest

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.core.user_manager import (
    DuplicateUserError,
    UserManager,
    UserNotFoundError,
)
from langchain_chat.models.user import User
from langchain_chat.storage.sqlite_backend import SQLiteBackend

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def manager() -> UserManager:
    """Return a UserManager backed by an initialized :memory: SQLite."""
    config = StorageConfig(type="sqlite", database=":memory:")
    be = SQLiteBackend(config)
    await be.initialize()
    yield UserManager(be)
    await be.close()


# ------------------------------------------------------------------
# Create user tests
# ------------------------------------------------------------------


class TestCreateUser:
    """Tests for UserManager.create_user()."""

    async def test_create_user_success(self, manager: UserManager) -> None:
        user = await manager.create_user("alice")
        assert isinstance(user, User)
        assert user.username == "alice"
        assert isinstance(user.id, int)
        assert user.id > 0

    async def test_create_user_trims_whitespace(self, manager: UserManager) -> None:
        user = await manager.create_user("  bob  ")
        assert user.username == "bob"

    async def test_create_duplicate_raises(self, manager: UserManager) -> None:
        await manager.create_user("carol")
        with pytest.raises(DuplicateUserError, match="carol"):
            await manager.create_user("carol")

    async def test_create_duplicate_case_sensitive(self, manager: UserManager) -> None:
        await manager.create_user("Dave")
        with pytest.raises(DuplicateUserError, match="Dave"):
            await manager.create_user("Dave")

    async def test_create_empty_username_raises(self, manager: UserManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            await manager.create_user("")
        with pytest.raises(ValueError, match="must not be empty"):
            await manager.create_user("   ")


# ------------------------------------------------------------------
# Get user tests
# ------------------------------------------------------------------


class TestGetUser:
    """Tests for UserManager.get_user()."""

    async def test_get_user_exists(self, manager: UserManager) -> None:
        created = await manager.create_user("eve")
        fetched = await manager.get_user(created.id)
        assert fetched.id == created.id
        assert fetched.username == "eve"

    async def test_get_user_missing_raises(self, manager: UserManager) -> None:
        with pytest.raises(UserNotFoundError, match="9999"):
            await manager.get_user(9999)

    async def test_get_user_returns_user_model(self, manager: UserManager) -> None:
        created = await manager.create_user("frank")
        user = await manager.get_user(created.id)
        assert isinstance(user, User)
        assert user.created_at is not None


class TestGetUserByName:
    """Tests for UserManager.get_user_by_name()."""

    async def test_get_user_by_name_exists(self, manager: UserManager) -> None:
        created = await manager.create_user("grace")
        fetched = await manager.get_user_by_name("grace")
        assert fetched.id == created.id

    async def test_get_user_by_name_missing_raises(self, manager: UserManager) -> None:
        with pytest.raises(UserNotFoundError, match="noone"):
            await manager.get_user_by_name("noone")


# ------------------------------------------------------------------
# List users tests
# ------------------------------------------------------------------


class TestListUsers:
    """Tests for UserManager.list_users()."""

    async def test_list_users_empty(self, manager: UserManager) -> None:
        users = await manager.list_users()
        assert users == []

    async def test_list_users_multiple(self, manager: UserManager) -> None:
        await manager.create_user("a")
        await manager.create_user("b")
        await manager.create_user("c")

        users = await manager.list_users()
        assert len(users) == 3
        assert all(isinstance(u, User) for u in users)
        names = {u.username for u in users}
        assert names == {"a", "b", "c"}


# ------------------------------------------------------------------
# Delete user tests
# ------------------------------------------------------------------


class TestDeleteUser:
    """Tests for UserManager.delete_user()."""

    async def test_delete_user_exists(self, manager: UserManager) -> None:
        created = await manager.create_user("henry")
        assert await manager.delete_user(created.id) is True
        with pytest.raises(UserNotFoundError):
            await manager.get_user(created.id)

    async def test_delete_user_idempotent(self, manager: UserManager) -> None:
        """Deleting a missing user returns False (idempotent)."""
        assert await manager.delete_user(9999) is False


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


class TestIntegration:
    """End-to-end scenarios through UserManager."""

    async def test_full_lifecycle(self, manager: UserManager) -> None:
        # Create
        u1 = await manager.create_user("iris")
        u2 = await manager.create_user("jack")

        # List
        users = await manager.list_users()
        assert len(users) == 2

        # Get by id
        assert (await manager.get_user(u1.id)).username == "iris"

        # Get by name
        assert (await manager.get_user_by_name("jack")).id == u2.id

        # Delete
        assert await manager.delete_user(u1.id) is True
        assert len(await manager.list_users()) == 1

        # Verify gone
        with pytest.raises(UserNotFoundError):
            await manager.get_user(u1.id)
