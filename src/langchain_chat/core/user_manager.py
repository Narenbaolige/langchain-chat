"""User management — the first Core Business Layer module.

UserManager orchestrates user operations through the StorageBackend
interface. It contains business logic (validation, duplicate handling,
etc.) but never touches the database directly.
"""

from __future__ import annotations

from langchain_chat.models.user import User
from langchain_chat.storage.base import StorageBackend


class DuplicateUserError(Exception):
    """Raised when attempting to create a user with an existing username."""


class UserNotFoundError(Exception):
    """Raised when a requested user does not exist."""


class UserManager:
    """Business-layer user management.

    Depends on StorageBackend (not SQLiteBackend!) so it works with
    any backend the factory provides.

    Usage::

        storage = StorageFactory.create(config.storage)
        await storage.initialize()
        manager = UserManager(storage)
        user = await manager.create_user("alice")
    """

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_user(self, username: str) -> User:
        """Create a new user.

        Args:
            username: Desired username (must be unique).

        Returns:
            The newly created User.

        Raises:
            DuplicateUserError: If *username* is already taken.
        """
        if not username or not username.strip():
            raise ValueError("Username must not be empty")

        username = username.strip()

        existing = await self._storage.get_user_by_name(username)
        if existing is not None:
            raise DuplicateUserError(f"User {username!r} already exists")

        data = await self._storage.create_user(username)
        return User.model_validate(data)

    async def get_user(self, user_id: int) -> User:
        """Retrieve a user by ID.

        Args:
            user_id: The user's unique ID.

        Returns:
            The matching User.

        Raises:
            UserNotFoundError: If no user has that ID.
        """
        data = await self._storage.get_user(user_id)
        if data is None:
            raise UserNotFoundError(f"User id={user_id} not found")
        return User.model_validate(data)

    async def get_user_by_name(self, username: str) -> User:
        """Retrieve a user by username.

        Args:
            username: Exact username to look up.

        Returns:
            The matching User.

        Raises:
            UserNotFoundError: If no user has that username.
        """
        data = await self._storage.get_user_by_name(username)
        if data is None:
            raise UserNotFoundError(f"User {username!r} not found")
        return User.model_validate(data)

    async def list_users(self) -> list[User]:
        """List all users.

        Returns:
            List of Users (may be empty).
        """
        rows = await self._storage.list_users()
        return [User.model_validate(r) for r in rows]

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user by ID (idempotent — no error if missing).

        Returns:
            True if a user was deleted, False if already gone.
        """
        return await self._storage.delete_user(user_id)
