"""Integration tests — full-chain workflows across multiple modules."""

from __future__ import annotations

import pytest

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.core.session_manager import SessionManager
from langchain_chat.core.user_manager import UserManager
from langchain_chat.storage.sqlite_backend import SQLiteBackend

# ------------------------------------------------------------------
# Fixture
# ------------------------------------------------------------------


@pytest.fixture
async def storage():
    """In-memory SQLite backend for integration tests."""
    cfg = StorageConfig(type="sqlite", database=":memory:")
    be = SQLiteBackend(cfg)
    await be.initialize()
    yield be
    await be.close()


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


class TestFullChatLifecycle:
    """User → Session → Message → Storage — full chain."""

    async def test_create_user_session_and_messages(self, storage: SQLiteBackend) -> None:
        """End-to-end: create user, session, add messages, verify."""
        user_mgr = UserManager(storage)
        session_mgr = SessionManager(storage)

        user = await user_mgr.create_user("alice")
        session = await session_mgr.create_session(user.id, title="My Chat")

        m1 = await session_mgr.add_message(session.id, "user", "What is AI?")
        m2 = await session_mgr.add_message(session.id, "assistant", "AI is artificial intelligence.")

        assert m1.role == "user"
        assert m2.role == "assistant"

        msgs = await session_mgr.get_messages(session.id)
        assert len(msgs) == 2
        assert msgs[0].content == "What is AI?"

    async def test_user_isolation(self, storage: SQLiteBackend) -> None:
        """Two users' sessions are isolated."""
        user_mgr = UserManager(storage)
        session_mgr = SessionManager(storage)

        u1 = await user_mgr.create_user("alice")
        u2 = await user_mgr.create_user("bob")

        await session_mgr.create_session(u1.id, title="Alice Chat")
        await session_mgr.create_session(u2.id, title="Bob Chat")

        alice_sessions = await session_mgr.list_sessions(user_id=u1.id)
        bob_sessions = await session_mgr.list_sessions(user_id=u2.id)
        assert len(alice_sessions) == 1
        assert len(bob_sessions) == 1
        assert alice_sessions[0].title == "Alice Chat"

    async def test_session_reopen_flow(self, storage: SQLiteBackend) -> None:
        """Create session → add messages → reopen → verify messages restored."""
        user_mgr = UserManager(storage)
        session_mgr = SessionManager(storage)

        user = await user_mgr.create_user("alice")
        session = await session_mgr.create_session(user.id, title="Test")
        await session_mgr.add_message(session.id, "user", "Q1")
        await session_mgr.add_message(session.id, "assistant", "A1")

        reopened, msgs = await session_mgr.reopen_session(session.id)
        assert reopened.id == session.id
        assert len(msgs) == 2

    async def test_cascade_delete(self, storage: SQLiteBackend) -> None:
        """Deleting a session cascade-deletes its messages."""
        user_mgr = UserManager(storage)
        session_mgr = SessionManager(storage)

        user = await user_mgr.create_user("alice")
        session = await session_mgr.create_session(user.id)
        await session_mgr.add_message(session.id, "user", "Hello")

        await session_mgr.delete_session(session.id)
        msgs = await session_mgr.get_messages(session.id)
        assert msgs == []

    async def test_pagination(self, storage: SQLiteBackend) -> None:
        """list_sessions pagination with limit/offset."""
        user_mgr = UserManager(storage)
        session_mgr = SessionManager(storage)

        user = await user_mgr.create_user("alice")
        for i in range(5):
            await session_mgr.create_session(user.id, title=f"Session {i}")

        # First page: limit 3
        page1 = await session_mgr.list_sessions(user_id=user.id, limit=3, offset=0)
        assert len(page1) == 3

        # Second page: limit 3, offset 3
        page2 = await session_mgr.list_sessions(user_id=user.id, limit=3, offset=3)
        assert len(page2) == 2  # remaining 2

        # No limit: all
        all_sessions = await session_mgr.list_sessions(user_id=user.id, limit=0, offset=0)
        assert len(all_sessions) == 5
