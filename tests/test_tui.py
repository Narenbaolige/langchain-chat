"""Tests for TUI components — command handling, dispatch, and app logic.

Does NOT test pixel-level Rich rendering; that is ChatView's concern.
"""

from __future__ import annotations

import pytest
from fakes import FakeModel

from langchain_chat.core.chat_engine import ChatEngine
from langchain_chat.core.config_models import LLMConfig, StorageConfig
from langchain_chat.core.model_manager import ModelManager
from langchain_chat.core.prompt_manager import PromptManager
from langchain_chat.core.session_manager import SessionManager
from langchain_chat.core.user_manager import UserManager
from langchain_chat.storage.sqlite_backend import SQLiteBackend
from langchain_chat.ui.app import TuiChatApp
from langchain_chat.ui.chat_view import ChatView
from langchain_chat.ui.commands import EXIT, CommandContext, CommandHandler

# ------------------------------------------------------------------
# CommandHandler unit tests
# ------------------------------------------------------------------


class TestCommandParsing:
    """Unit tests for command detection and parsing."""

    def test_is_command_slash_prefix(self) -> None:
        handler = CommandHandler()
        assert handler.is_command("/help") is True
        assert handler.is_command("/quit") is True

    def test_is_command_not_command(self) -> None:
        handler = CommandHandler()
        assert handler.is_command("hello") is False
        assert handler.is_command("") is False
        assert handler.is_command("  ") is False

    def test_is_command_with_leading_whitespace(self) -> None:
        handler = CommandHandler()
        assert handler.is_command("  /help") is True


class TestCommandExecution:
    """Integration tests for command dispatch."""

    @pytest.fixture
    async def ctx(self) -> CommandContext:
        """Build a CommandContext backed by real :memory: managers."""
        config = StorageConfig(type="sqlite", database=":memory:")
        backend = SQLiteBackend(config)
        await backend.initialize()

        user_manager = UserManager(backend)
        prompt_manager = PromptManager(backend)
        session_manager = SessionManager(backend)
        model_manager = ModelManager(LLMConfig(model="gpt-4o-mini"))

        # Ensure a default user exists
        try:
            user = await user_manager.create_user("testuser")
        except Exception:
            user = await user_manager.get_user_by_name("testuser")

        session = await session_manager.create_session(user.id, title="Test")
        engine = ChatEngine(model=FakeModel(response="OK"))
        view = ChatView()

        return CommandContext(
            user_manager=user_manager,
            prompt_manager=prompt_manager,
            session_manager=session_manager,
            model_manager=model_manager,
            engine=engine,
            view=view,
            current_user_name="testuser",
            session=session,
        )

    async def test_help_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/help", ctx)
        assert result is not EXIT

    async def test_quit_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/quit", ctx)
        assert result is EXIT

    async def test_exit_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/exit", ctx)
        assert result is EXIT

    async def test_clear_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        await ctx.engine.chat("Hello")
        assert ctx.engine.message_count == 2
        result = await handler.handle("/clear", ctx)
        assert result is not EXIT
        assert ctx.engine.message_count == 0

    async def test_stats_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/stats", ctx)
        assert result is not EXIT

    async def test_users_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/users", ctx)
        assert result is not EXIT

    async def test_presets_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/presets", ctx)
        assert result is not EXIT

    async def test_unknown_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/unknown_cmd_xyz", ctx)
        assert result is not EXIT  # Should continue gracefully

    async def test_user_command_missing_name(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/user", ctx)
        assert result is not EXIT

    async def test_preset_command_missing_name(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/preset", ctx)
        assert result is not EXIT

    async def test_system_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()

        system_set = False

        async def _on_preset(info: dict) -> None:
            nonlocal system_set
            system_set = True

        ctx.on_preset_change = _on_preset
        result = await handler.handle("/system You are helpful", ctx)
        assert result is not EXIT
        assert system_set is True

    async def test_system_command_missing_text(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/system", ctx)
        assert result is not EXIT


# ------------------------------------------------------------------
# TuiChatApp construction
# ------------------------------------------------------------------


class TestTuiChatAppConstruction:
    """Smoke tests for TuiChatApp dependency injection."""

    async def test_constructor_accepts_managers(self) -> None:
        """Verify TuiChatApp wires up without errors."""
        config = StorageConfig(type="sqlite", database=":memory:")
        backend = SQLiteBackend(config)
        await backend.initialize()

        try:
            user_manager = UserManager(backend)
            prompt_manager = PromptManager(backend)
            session_manager = SessionManager(backend)
            engine = ChatEngine(model=FakeModel(response="Hi"))

            model_mgr = ModelManager(LLMConfig(model="gpt-4o-mini"))
            app = TuiChatApp(user_manager, prompt_manager, session_manager, model_mgr, engine)
            assert app is not None
        finally:
            await backend.close()


# ------------------------------------------------------------------
# Step 8 — Session management command tests
# ------------------------------------------------------------------


class TestStep8Commands:
    """Tests for the Step 8 session management commands."""

    @pytest.fixture
    async def ctx(self) -> CommandContext:
        """Build a CommandContext with session_manager."""
        config = StorageConfig(type="sqlite", database=":memory:")
        backend = SQLiteBackend(config)
        await backend.initialize()

        user_manager = UserManager(backend)
        prompt_manager = PromptManager(backend)
        session_manager = SessionManager(backend)

        try:
            user = await user_manager.create_user("testuser")
        except Exception:
            user = await user_manager.get_user_by_name("testuser")

        session = await session_manager.create_session(user.id, title="Test")
        model_manager = ModelManager(LLMConfig(model="gpt-4o-mini"))
        engine = ChatEngine(model=FakeModel(response="OK"))
        view = ChatView()

        return CommandContext(
            user_manager=user_manager,
            prompt_manager=prompt_manager,
            session_manager=session_manager,
            model_manager=model_manager,
            engine=engine,
            view=view,
            current_user_name="testuser",
            session=session,
        )

    async def test_sessions_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/sessions", ctx)
        assert result is not EXIT

    async def test_search_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/search Test", ctx)
        assert result is not EXIT

    async def test_search_missing_query(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/search", ctx)
        assert result is not EXIT

    async def test_rename_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/rename New Name", ctx)
        assert result is not EXIT

    async def test_rename_missing_title(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/rename", ctx)
        assert result is not EXIT

    async def test_open_command(self, ctx: CommandContext) -> None:
        opened = False

        async def _on_open(sid: int) -> None:
            nonlocal opened
            opened = True

        ctx.on_session_open = _on_open
        handler = CommandHandler()
        result = await handler.handle(f"/open {ctx.session.id}", ctx)
        assert result is not EXIT
        assert opened is True

    async def test_open_missing_id(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/open", ctx)
        assert result is not EXIT

    async def test_open_invalid_id(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/open abc", ctx)
        assert result is not EXIT

    async def test_delete_session_command(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle(f"/delete-session {ctx.session.id}", ctx)
        assert result is not EXIT

    async def test_delete_session_missing_id(self, ctx: CommandContext) -> None:
        handler = CommandHandler()
        result = await handler.handle("/delete-session", ctx)
        assert result is not EXIT


# ------------------------------------------------------------------
# Step 9 — ChatView polish tests
# ------------------------------------------------------------------


class TestChatView:
    """Tests for the enhanced ChatView display methods."""

    def test_build_prompt_basic(self) -> None:
        view = ChatView()
        prompt = view.build_prompt("alice")
        assert "alice" in prompt

    def test_build_prompt_with_session(self) -> None:
        view = ChatView()
        prompt = view.build_prompt("alice", session_id=5)
        assert "alice" in prompt
        assert "5" in prompt

    def test_build_prompt_with_system_active(self) -> None:
        view = ChatView()
        view.set_system_prompt_active(True)
        prompt = view.build_prompt("alice", session_id=1)
        assert "⚡" in prompt

    def test_system_prompt_active_property(self) -> None:
        view = ChatView()
        assert view.system_prompt_active is False
        view.set_system_prompt_active(True)
        assert view.system_prompt_active is True

    def test_show_warning_output(self) -> None:
        """Smoke test — ensure show_warning doesn't crash."""
        view = ChatView()
        view.show_warning("Test warning")

    def test_show_error_output(self) -> None:
        """Smoke test — ensure show_error panel doesn't crash."""
        view = ChatView()
        view.show_error("Test error")

    def test_show_sessions_no_data(self) -> None:
        """Smoke test — empty list."""
        view = ChatView()
        view.show_sessions([])

    def test_show_sessions_with_data(self) -> None:
        """Smoke test — sessions with message counts."""
        view = ChatView()
        view.show_sessions(
            [{"id": 1, "title": "Test", "updated_at": "2026-01-01", "message_count": "3"}]
        )

    def test_show_help_grouped(self) -> None:
        """Smoke test — grouped help display."""
        view = ChatView()
        view.show_help()

    def test_show_welcome_with_session(self) -> None:
        """Smoke test — welcome with session context."""
        view = ChatView()
        view.show_welcome("testuser", session_id=42, system_prompt=True)


# ------------------------------------------------------------------
# Edge case tests
# ------------------------------------------------------------------


class TestCommandHandlerEdgeCases:
    """Edge cases for command parsing and dispatch."""

    def test_is_command_none(self) -> None:
        handler = CommandHandler()
        assert handler.is_command("") is False

    def test_is_command_multiple_slashes(self) -> None:
        handler = CommandHandler()
        assert handler.is_command("//help") is True

    def test_command_case_insensitive(self) -> None:
        handler = CommandHandler()
        assert handler.is_command("/HELP") is True

    async def test_command_with_extra_whitespace(self) -> None:
        """Commands with extra internal whitespace in args."""
        config = StorageConfig(type="sqlite", database=":memory:")
        backend = SQLiteBackend(config)
        await backend.initialize()
        try:
            um = UserManager(backend)
            pm = PromptManager(backend)
            sm = SessionManager(backend)
            mm = ModelManager(LLMConfig(model="gpt-4o-mini"))
            await um.create_user("testuser")
            session = await sm.create_session(1, title="Test")
            engine = ChatEngine(model=FakeModel(response="OK"))
            view = ChatView()

            ctx = CommandContext(
                user_manager=um,
                prompt_manager=pm,
                session_manager=sm,
                model_manager=mm,
                engine=engine,
                view=view,
                current_user_name="testuser",
                session=session,
            )
            handler = CommandHandler()
            # /system with multiple spaces in arg
            result = await handler.handle("/system   You are helpful   ", ctx)
            assert result is not EXIT
        finally:
            await backend.close()


class TestChatViewEdgeCases:
    """Edge cases for ChatView display methods."""

    def test_show_users_empty_list(self) -> None:
        view = ChatView()
        view.show_users([])

    def test_show_presets_empty_list(self) -> None:
        view = ChatView()
        view.show_presets([])

    def test_build_prompt_empty_model(self) -> None:
        view = ChatView()
        prompt = view.build_prompt("alice", session_id=1, model_name="")
        assert "alice" in prompt
        assert "1" in prompt

    def test_build_prompt_all_fields(self) -> None:
        view = ChatView()
        view.set_system_prompt_active(True)
        prompt = view.build_prompt("alice", session_id=5, model_name="gpt-4o")
        assert "alice" in prompt
        assert "5" in prompt
        assert "gpt-4o" in prompt
        assert "⚡" in prompt

    def test_show_welcome_all_params(self) -> None:
        view = ChatView()
        view.show_welcome(
            "u", session_id=1, system_prompt=True, provider="openai", model="gpt-4o-mini"
        )

    def test_show_stats_active_prompt(self) -> None:
        view = ChatView()
        view.set_system_prompt_active(True)
        view.show_stats(10, 1, "user")
