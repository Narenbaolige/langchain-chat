"""Tests for SessionManager — session & message persistence.

Includes Step 8 enhancements: update, search, reopen.
"""

from __future__ import annotations

import pytest

from langchain_chat.core.session_manager import SessionManager, SessionNotFoundError
from langchain_chat.models.message import Message
from langchain_chat.models.session import Session
from langchain_chat.storage.base import StorageBackend

# ------------------------------------------------------------------
# Fixture
# ------------------------------------------------------------------


@pytest.fixture
async def session_manager(storage_backend: StorageBackend) -> SessionManager:
    """Return a SessionManager backed by an initialized :memory: SQLite.

    Pre-creates a couple of users so FK constraints are satisfied.
    """
    await storage_backend.create_user("user1")
    await storage_backend.create_user("user2")
    return SessionManager(storage_backend)


# ------------------------------------------------------------------
# Session creation tests
# ------------------------------------------------------------------


class TestCreateSession:
    """Tests for SessionManager.create_session()."""

    async def test_create_session_success(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1, title="Test Chat")
        assert isinstance(session, Session)
        assert session.user_id == 1
        assert session.title == "Test Chat"
        assert session.id > 0
        assert session.preset_id is None

    async def test_create_session_with_preset(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=2, preset_id=5)
        assert session.preset_id == 5
        assert session.title == ""

    async def test_create_session_default_title(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        assert session.title == ""

    async def test_create_session_strips_title(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1, title="  Hello  ")
        assert session.title == "Hello"

    async def test_create_session_has_timestamps(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        assert session.created_at is not None
        assert session.updated_at is not None


# ------------------------------------------------------------------
# Session retrieval tests
# ------------------------------------------------------------------


class TestGetSession:
    """Tests for SessionManager.get_session()."""

    async def test_get_session_exists(self, session_manager: SessionManager) -> None:
        created = await session_manager.create_session(user_id=1)
        fetched = await session_manager.get_session(created.id)
        assert fetched.id == created.id
        assert fetched.user_id == 1

    async def test_get_session_missing_raises(self, session_manager: SessionManager) -> None:
        with pytest.raises(SessionNotFoundError, match="9999"):
            await session_manager.get_session(9999)

    async def test_get_session_returns_session_model(self, session_manager: SessionManager) -> None:
        created = await session_manager.create_session(user_id=1)
        fetched = await session_manager.get_session(created.id)
        assert isinstance(fetched, Session)


# ------------------------------------------------------------------
# List sessions tests
# ------------------------------------------------------------------


class TestListSessions:
    """Tests for SessionManager.list_sessions()."""

    async def test_list_sessions_empty(self, session_manager: SessionManager) -> None:
        assert await session_manager.list_sessions() == []

    async def test_list_sessions_all(self, session_manager: SessionManager) -> None:
        await session_manager.create_session(user_id=1, title="A")
        await session_manager.create_session(user_id=2, title="B")
        sessions = await session_manager.list_sessions()
        assert len(sessions) == 2
        assert all(isinstance(s, Session) for s in sessions)

    async def test_list_sessions_filter_by_user(self, session_manager: SessionManager) -> None:
        await session_manager.create_session(user_id=1, title="U1")
        await session_manager.create_session(user_id=2, title="U2")
        await session_manager.create_session(user_id=1, title="U1-again")

        u1_sessions = await session_manager.list_sessions(user_id=1)
        assert len(u1_sessions) == 2
        assert all(s.user_id == 1 for s in u1_sessions)

        u2_sessions = await session_manager.list_sessions(user_id=2)
        assert len(u2_sessions) == 1


# ------------------------------------------------------------------
# Delete session tests
# ------------------------------------------------------------------


class TestDeleteSession:
    """Tests for SessionManager.delete_session()."""

    async def test_delete_session_exists(self, session_manager: SessionManager) -> None:
        created = await session_manager.create_session(user_id=1)
        assert await session_manager.delete_session(created.id) is True
        with pytest.raises(SessionNotFoundError):
            await session_manager.get_session(created.id)

    async def test_delete_session_idempotent(self, session_manager: SessionManager) -> None:
        assert await session_manager.delete_session(9999) is False


# ------------------------------------------------------------------
# Message tests
# ------------------------------------------------------------------


class TestAddMessage:
    """Tests for SessionManager.add_message()."""

    async def test_add_user_message(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        msg = await session_manager.add_message(session.id, "user", "Hello!")
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.session_id == session.id

    async def test_add_assistant_message(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        msg = await session_manager.add_message(session.id, "assistant", "Hi there!")
        assert msg.role == "assistant"

    async def test_add_system_message(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        msg = await session_manager.add_message(session.id, "system", "System prompt")
        assert msg.role == "system"

    async def test_add_message_invalid_role(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        with pytest.raises(ValueError, match="Invalid role"):
            await session_manager.add_message(session.id, "invalid", "content")

    async def test_add_message_empty_content(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        with pytest.raises(ValueError, match="must not be empty"):
            await session_manager.add_message(session.id, "user", "")


class TestGetMessages:
    """Tests for SessionManager.get_messages()."""

    async def test_get_messages_empty(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        assert await session_manager.get_messages(session.id) == []

    async def test_get_messages_ordered(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        await session_manager.add_message(session.id, "user", "Q1")
        await session_manager.add_message(session.id, "assistant", "A1")
        await session_manager.add_message(session.id, "user", "Q2")

        msgs = await session_manager.get_messages(session.id)
        assert len(msgs) == 3
        assert all(isinstance(m, Message) for m in msgs)
        assert [m.content for m in msgs] == ["Q1", "A1", "Q2"]

    async def test_get_messages_returns_message_models(
        self, session_manager: SessionManager
    ) -> None:
        session = await session_manager.create_session(user_id=1)
        await session_manager.add_message(session.id, "user", "Hello")
        msgs = await session_manager.get_messages(session.id)
        assert isinstance(msgs[0], Message)
        assert msgs[0].created_at is not None


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


class TestIntegration:
    """End-to-end session + message lifecycle."""

    async def test_full_conversation_persistence(self, session_manager: SessionManager) -> None:
        # Create session
        session = await session_manager.create_session(user_id=1, title="Full Test")
        assert isinstance(session, Session)

        # Add messages
        m1 = await session_manager.add_message(session.id, "user", "What is AI?")
        m2 = await session_manager.add_message(session.id, "assistant", "AI is...")
        assert m1.role == "user"
        assert m2.role == "assistant"

        # Retrieve
        msgs = await session_manager.get_messages(session.id)
        assert len(msgs) == 2
        assert msgs[0].content == "What is AI?"
        assert msgs[1].content == "AI is..."

        # List sessions
        sessions = await session_manager.list_sessions(user_id=1)
        assert len(sessions) == 1
        assert sessions[0].title == "Full Test"

        # Delete and verify cascade
        assert await session_manager.delete_session(session.id) is True
        # Messages cascade-deleted with session (SQLite FK CASCADE)
        remaining = await session_manager.get_messages(session.id)
        assert remaining == []


# ------------------------------------------------------------------
# Step 8 — Update session tests
# ------------------------------------------------------------------


class TestUpdateSession:
    """Tests for SessionManager.update_session()."""

    async def test_rename_success(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1, title="Old")
        updated = await session_manager.update_session(session.id, "New Title")
        assert updated.title == "New Title"
        assert updated.id == session.id

    async def test_rename_strips_whitespace(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1, title="Old")
        updated = await session_manager.update_session(session.id, "  Trimmed  ")
        assert updated.title == "Trimmed"

    async def test_rename_empty_title_raises(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1, title="X")
        with pytest.raises(ValueError, match="must not be empty"):
            await session_manager.update_session(session.id, "   ")

    async def test_rename_missing_session_raises(self, session_manager: SessionManager) -> None:
        with pytest.raises(SessionNotFoundError, match="9999"):
            await session_manager.update_session(9999, "X")


# ------------------------------------------------------------------
# Step 8 — Search sessions tests
# ------------------------------------------------------------------


class TestSearchSessions:
    """Tests for SessionManager.search_sessions()."""

    async def test_search_by_title(self, session_manager: SessionManager) -> None:
        await session_manager.create_session(user_id=1, title="Project Alpha")
        await session_manager.create_session(user_id=1, title="Project Beta")
        await session_manager.create_session(user_id=1, title="Random")

        results = await session_manager.search_sessions(1, "Project")
        assert len(results) == 2

    async def test_search_empty_query_returns_all(self, session_manager: SessionManager) -> None:
        await session_manager.create_session(user_id=1, title="A")
        await session_manager.create_session(user_id=1, title="B")
        results = await session_manager.search_sessions(1, "")
        assert len(results) == 2

    async def test_search_no_match(self, session_manager: SessionManager) -> None:
        await session_manager.create_session(user_id=1, title="Hello")
        results = await session_manager.search_sessions(1, "zzz_nonexistent")
        assert results == []

    async def test_search_case_sensitive(self, session_manager: SessionManager) -> None:
        await session_manager.create_session(user_id=1, title="UPPERCASE")
        # SQLite LIKE is case-insensitive for ASCII by default
        results = await session_manager.search_sessions(1, "upper")
        assert len(results) == 1


# ------------------------------------------------------------------
# Step 8 — Search messages tests
# ------------------------------------------------------------------


class TestSearchMessages:
    """Tests for SessionManager.search_messages()."""

    async def test_search_message_content(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        await session_manager.add_message(session.id, "user", "What is Python?")
        await session_manager.add_message(session.id, "assistant", "Python is a language.")
        await session_manager.add_message(session.id, "user", "Tell me about Java.")

        results = await session_manager.search_messages(session.id, "Python")
        assert len(results) == 2

    async def test_search_no_match(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        await session_manager.add_message(session.id, "user", "Hello")
        results = await session_manager.search_messages(session.id, "zzz")
        assert results == []

    async def test_search_empty_query_returns_all(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1)
        await session_manager.add_message(session.id, "user", "A")
        await session_manager.add_message(session.id, "assistant", "B")
        results = await session_manager.search_messages(session.id, "")
        assert len(results) == 2


# ------------------------------------------------------------------
# Step 8 — Reopen session tests
# ------------------------------------------------------------------


class TestReopenSession:
    """Tests for SessionManager.reopen_session()."""

    async def test_reopen_returns_session_and_messages(
        self, session_manager: SessionManager
    ) -> None:
        session = await session_manager.create_session(user_id=1, title="Saved Chat")
        await session_manager.add_message(session.id, "user", "Q1")
        await session_manager.add_message(session.id, "assistant", "A1")

        reopened, msgs = await session_manager.reopen_session(session.id)
        assert reopened.id == session.id
        assert reopened.title == "Saved Chat"
        assert len(msgs) == 2
        assert msgs[0].content == "Q1"
        assert msgs[1].content == "A1"

    async def test_reopen_missing_raises(self, session_manager: SessionManager) -> None:
        with pytest.raises(SessionNotFoundError, match="9999"):
            await session_manager.reopen_session(9999)

    async def test_reopen_empty_session(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(user_id=1, title="Empty")
        _, msgs = await session_manager.reopen_session(session.id)
        assert msgs == []


# ------------------------------------------------------------------
# Edge case tests
# ------------------------------------------------------------------


class TestSessionManagerEdgeCases:
    """Boundary and error scenarios for sessions and messages."""

    async def test_search_sessions_special_chars(self, session_manager: SessionManager) -> None:
        await session_manager.create_session(user_id=1, title="test@#$%")
        results = await session_manager.search_sessions(1, "@#$")
        assert len(results) == 1

    async def test_search_messages_special_chars(self, session_manager: SessionManager) -> None:
        s = await session_manager.create_session(user_id=1)
        await session_manager.add_message(s.id, "user", "contains @#$% specials")
        results = await session_manager.search_messages(s.id, "@#$%")
        assert len(results) == 1

    async def test_update_session_same_title(self, session_manager: SessionManager) -> None:
        s = await session_manager.create_session(user_id=1, title="Same")
        updated = await session_manager.update_session(s.id, "Same")
        assert updated.title == "Same"

    async def test_add_message_invalid_role_empty(self, session_manager: SessionManager) -> None:
        s = await session_manager.create_session(user_id=1)
        with pytest.raises(ValueError, match="Invalid role"):
            await session_manager.add_message(s.id, "", "content")

    async def test_add_message_very_long_content(self, session_manager: SessionManager) -> None:
        s = await session_manager.create_session(user_id=1)
        long_text = "x" * 10000
        msg = await session_manager.add_message(s.id, "user", long_text)
        assert len(msg.content) == 10000

    async def test_get_messages_non_existent_session(self, session_manager: SessionManager) -> None:
        msgs = await session_manager.get_messages(9999)
        assert msgs == []
