"""Session & message management — Core Business Layer module.

SessionManager orchestrates session and message persistence through the
StorageBackend interface.  It contains business logic (validation, error
handling) but never touches the database directly.

Interface naming is kept consistent with StorageBackend (add_message,
get_messages, etc.) to avoid cognitive overhead.
"""

from __future__ import annotations

from langchain_chat.models.message import Message
from langchain_chat.models.session import Session
from langchain_chat.storage.base import StorageBackend


class SessionNotFoundError(Exception):
    """Raised when a requested session does not exist."""


class SessionManager:
    """Business-layer session & message management.

    Depends on StorageBackend (not SQLiteBackend!) so it works with
    any backend the factory provides.

    Usage::

        storage = StorageFactory.create(config.storage)
        await storage.initialize()
        manager = SessionManager(storage)
        session = await manager.create_session(user_id=1, title="My Chat")
        msg = await manager.add_message(session.id, "user", "Hello!")
    """

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    async def create_session(
        self, user_id: int, title: str = "", preset_id: int | None = None
    ) -> Session:
        """Create a new chat session.

        Args:
            user_id: Owner of the session.
            title: Optional human-readable title.
            preset_id: Optional preset to bind (reserved for Step 8).

        Returns:
            The newly created Session.
        """
        if title:
            title = title.strip()
        data = await self._storage.create_session(user_id, title, preset_id)
        return Session.model_validate(data)

    async def get_session(self, session_id: int) -> Session:
        """Retrieve a session by ID.

        Raises:
            SessionNotFoundError: If no session has that ID.
        """
        data = await self._storage.get_session(session_id)
        if data is None:
            raise SessionNotFoundError(f"Session id={session_id} not found")
        return Session.model_validate(data)

    async def list_sessions(self, user_id: int | None = None) -> list[Session]:
        """List sessions, optionally filtered by user.

        Args:
            user_id: If provided, only return sessions owned by this user.

        Returns:
            List of Sessions (may be empty).
        """
        rows = await self._storage.list_sessions(user_id)
        return [Session.model_validate(r) for r in rows]

    async def delete_session(self, session_id: int) -> bool:
        """Delete a session by ID (idempotent — no error if missing).

        Returns:
            True if a session was deleted, False if already gone.
        """
        return await self._storage.delete_session(session_id)

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def add_message(self, session_id: int, role: str, content: str) -> Message:
        """Append a message to a session.

        Args:
            session_id: Target session.
            role: Message role (``"user"``, ``"assistant"``, or ``"system"``).
            content: Message body.

        Returns:
            The newly created Message.
        """
        if not content:
            raise ValueError("Message content must not be empty")
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role {role!r}; expected user, assistant, or system")

        data = await self._storage.add_message(session_id, role, content)
        return Message.model_validate(data)

    async def get_messages(self, session_id: int) -> list[Message]:
        """Retrieve all messages for a session, ordered by creation time.

        Args:
            session_id: Session to fetch messages for.

        Returns:
            List of Messages (may be empty).
        """
        rows = await self._storage.get_messages(session_id)
        return [Message.model_validate(r) for r in rows]
