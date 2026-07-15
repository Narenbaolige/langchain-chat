"""Abstract storage backend interface.

Defines the contract that ALL storage backends (SQLite, MySQL, File, etc.)
must fulfill. Business-layer managers depend on this interface — never on
concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """Abstract interface for storage backends.

    Every concrete backend (SQLiteBackend, future MySQLBackend, etc.)
    must implement all methods defined here.

    All methods are async to support both sync and async backends uniformly.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (create tables, open connections, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the backend."""
        ...

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_user(self, username: str) -> dict[str, Any]:
        """Create a new user.

        Args:
            username: Unique username.

        Returns:
            dict with keys: id, username, created_at.
        """
        ...

    @abstractmethod
    async def get_user(self, user_id: int) -> dict[str, Any] | None:
        """Retrieve a user by ID.

        Returns:
            User dict or None if not found.
        """
        ...

    @abstractmethod
    async def delete_user(self, user_id: int) -> bool:
        """Delete a user by ID.

        Returns:
            True if deleted, False if user did not exist.
        """
        ...

    @abstractmethod
    async def get_user_by_name(self, username: str) -> dict[str, Any] | None:
        """Retrieve a user by username.

        Returns:
            User dict or None if not found.
        """
        ...

    @abstractmethod
    async def list_users(self) -> list[dict[str, Any]]:
        """List all users.

        Returns:
            List of user dicts.
        """
        ...

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_session(
        self, user_id: int, title: str = "", preset_id: int | None = None
    ) -> dict[str, Any]:
        """Create a new chat session.

        Args:
            user_id: Owner of the session.
            title: Optional human-readable title.
            preset_id: Optional preset bound to this session (reserved).

        Returns:
            dict with keys: id, user_id, preset_id, title, created_at, updated_at.
        """
        ...

    @abstractmethod
    async def get_session(self, session_id: int) -> dict[str, Any] | None:
        """Retrieve a session by ID.

        Returns:
            Session dict or None if not found.
        """
        ...

    @abstractmethod
    async def list_sessions(
        self, user_id: int | None = None, limit: int = 0, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by user, with pagination.

        Args:
            user_id: If provided, only return sessions owned by this user.
            limit: Max records to return (0 = no limit).
            offset: Records to skip before returning.

        Returns:
            List of session dicts, ordered by updated_at DESC.
        """
        ...

    @abstractmethod
    async def delete_session(self, session_id: int) -> bool:
        """Delete a session by ID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def update_session(self, session_id: int, title: str) -> dict[str, Any]:
        """Rename a session (and bump its updated_at).

        Args:
            session_id: Session to rename.
            title: New title.

        Returns:
            Updated session dict.

        Raises:
            The backend should raise if session_id does not exist.
        """
        ...

    @abstractmethod
    async def search_sessions(self, user_id: int, query: str) -> list[dict[str, Any]]:
        """Search sessions by title substring for a given user.

        Args:
            user_id: Owner user ID.
            query: Substring to match against session titles.

        Returns:
            List of matching session dicts, ordered by updated_at DESC.
        """
        ...

    @abstractmethod
    async def search_messages(self, session_id: int, query: str) -> list[dict[str, Any]]:
        """Search messages by content substring within a session.

        Args:
            session_id: Session to search within.
            query: Substring to match against message content.

        Returns:
            List of matching message dicts, ordered by created_at ASC.
        """
        ...

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def add_message(self, session_id: int, role: str, content: str) -> dict[str, Any]:
        """Append a message to a session.

        Args:
            session_id: Target session.
            role: Message role (e.g. "user", "assistant", "system").
            content: Message body.

        Returns:
            dict with keys: id, session_id, role, content, created_at.
        """
        ...

    @abstractmethod
    async def get_messages(self, session_id: int) -> list[dict[str, Any]]:
        """Retrieve all messages for a session, ordered by creation time.

        Args:
            session_id: Session to fetch messages for.

        Returns:
            List of message dicts.
        """
        ...

    # ------------------------------------------------------------------
    # Preset operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_preset(
        self, name: str, content: str, prompt_type: str = "user"
    ) -> dict[str, Any]:
        """Create a new prompt preset.

        Args:
            name: Preset name (must be unique).
            content: Preset content / template text.
            prompt_type: "system" or "user" (default).

        Returns:
            dict with keys: id, name, content, prompt_type, created_at.
        """
        ...

    @abstractmethod
    async def get_preset(self, preset_id: int) -> dict[str, Any] | None:
        """Retrieve a single preset by ID.

        Returns:
            Preset dict or None if not found.
        """
        ...

    @abstractmethod
    async def get_preset_by_name(self, name: str) -> dict[str, Any] | None:
        """Retrieve a preset by name.

        Returns:
            Preset dict or None if not found.
        """
        ...

    @abstractmethod
    async def get_presets(self) -> list[dict[str, Any]]:
        """Retrieve all presets.

        Returns:
            List of preset dicts.
        """
        ...

    @abstractmethod
    async def delete_preset(self, preset_id: int) -> bool:
        """Delete a preset by ID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    # ------------------------------------------------------------------
    # Config operations (key-value store)
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_config(self, key: str) -> str | None:
        """Read a config value by key.

        Returns:
            Value string or None if key not found.
        """
        ...

    @abstractmethod
    async def save_config(self, key: str, value: str) -> None:
        """Save (insert or update) a config key-value pair.

        Args:
            key: Config key.
            value: Config value.
        """
        ...
