"""TUI Chat Application — the main UI entry point.

TuiChatApp orchestrates the terminal chat experience: startup, input
loop, command dispatch, streaming responses, and message persistence.
It depends on the Core Business Layer managers — never on storage directly.
"""

from __future__ import annotations

import asyncio
import signal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_chat.core.chat_engine import ChatEngine
from langchain_chat.core.model_manager import ModelManager
from langchain_chat.core.prompt_manager import PromptManager
from langchain_chat.core.session_manager import SessionManager
from langchain_chat.core.user_manager import DuplicateUserError, UserManager
from langchain_chat.models.session import Session
from langchain_chat.ui.chat_view import ChatView
from langchain_chat.ui.commands import EXIT, CommandContext, CommandHandler

# ------------------------------------------------------------------
# Prompt shown to the user
# ------------------------------------------------------------------


class TuiChatApp:
    """Async terminal chat application.

    Wires together managers, the chat engine, and the display layer.
    Call :meth:`run` to start the interactive loop.

    Usage::

        app = TuiChatApp(user_mgr, prompt_mgr, session_mgr, model_mgr, engine)
        await app.run()
    """

    def __init__(
        self,
        user_manager: UserManager,
        prompt_manager: PromptManager,
        session_manager: SessionManager,
        model_manager: ModelManager,
        engine: ChatEngine,
        *,
        view: ChatView | None = None,
        max_input_length: int = 5000,
        context_max_tokens: int = 4000,
    ) -> None:
        self._user_manager = user_manager
        self._prompt_manager = prompt_manager
        self._session_manager = session_manager
        self._model_manager = model_manager
        self._engine = engine
        self._view = view or ChatView()
        self._max_input_length = max_input_length
        self._context_max_tokens = context_max_tokens

        # Mutable state
        self._running: bool = False
        self._session: Session | None = None
        self._user_name: str = ""
        self._system_prompt: str | None = None
        self._command_handler = CommandHandler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the TUI: welcome → startup wizard → input loop."""
        self._running = True
        self._setup_signal_handlers()

        try:
            await self._startup()
            await self._input_loop()
        except asyncio.CancelledError:
            pass
        finally:
            self._view.show_info("Goodbye!")

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def _startup(self) -> None:
        """Walk through user selection, preset loading, and session creation."""
        # -- Pick or create a user --
        users = await self._user_manager.list_users()
        if not users:
            try:
                user = await self._user_manager.create_user("default")
                self._user_name = user.username
                self._view.show_info(f"Created default user: {user.username}")
            except DuplicateUserError:
                user = await self._user_manager.get_user_by_name("default")
                self._user_name = user.username
        elif len(users) == 1:
            self._user_name = users[0].username
        else:
            self._view.show_users(
                [
                    {"id": u.id, "username": u.username, "created_at": str(u.created_at)}
                    for u in users
                ]
            )
            choice = input(f"Select user [1-{len(users)}, default=1]: ").strip()
            try:
                idx = int(choice) - 1 if choice else 0
                if 0 <= idx < len(users):
                    self._user_name = users[idx].username
                else:
                    self._user_name = users[0].username
            except ValueError:
                self._user_name = users[0].username

        # -- Optionally load a preset --
        presets = await self._prompt_manager.list_presets()
        if presets:
            self._view.show_presets(
                [
                    {"name": p.name, "content": p.content, "prompt_type": p.prompt_type}
                    for p in presets
                ]
            )
            choice = input("Load preset? Enter name or press Enter to skip: ").strip()
            if choice:
                try:
                    preset = await self._prompt_manager.get_preset_by_name(choice)
                    self._system_prompt = preset.content
                    self._view.show_success(f"Loaded preset: {preset.name}")
                except Exception as exc:
                    self._view.show_error(str(exc))

        # -- Resolve user ID for session creation --
        user = await self._user_manager.get_user_by_name(self._user_name)
        self._session = await self._session_manager.create_session(
            user.id, title=f"Chat with {self._user_name}"
        )

        # -- Welcome --
        self._view.set_system_prompt_active(self._system_prompt is not None)
        self._view.show_welcome(
            self._user_name,
            session_id=self._session.id if self._session else None,
            system_prompt=self._system_prompt is not None,
            provider=self._model_manager.current_provider,
            model=self._model_manager.current_model,
        )

    # ------------------------------------------------------------------
    # Input loop
    # ------------------------------------------------------------------

    async def _input_loop(self) -> None:
        """Main REPL: read line → dispatch command or chat."""
        while self._running:
            prompt = self._view.build_prompt(
                self._user_name,
                session_id=self._session.id if self._session else None,
                model_name=f"{self._model_manager.current_provider}/{self._model_manager.current_model}",
            )
            try:
                line = await asyncio.to_thread(input, prompt)
            except EOFError:
                # Ctrl+D / Ctrl+Z — exit gracefully
                self._running = False
                break

            # None from input() should not happen, but guard anyway.
            if line is None:
                self._running = False
                break

            line = line.strip()
            if not line:
                continue

            if self._command_handler.is_command(line):
                await self._dispatch_command(line)
            else:
                await self._handle_chat(line)

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def _dispatch_command(self, line: str) -> None:
        """Parse and execute a slash-command."""
        ctx = CommandContext(
            user_manager=self._user_manager,
            prompt_manager=self._prompt_manager,
            session_manager=self._session_manager,
            model_manager=self._model_manager,
            engine=self._engine,
            view=self._view,
            current_user_name=self._user_name,
            session=self._session,
            on_user_change=self._on_user_change,
            on_preset_change=self._on_preset_change,
            on_session_open=self._on_session_open,
            on_model_change=self._on_model_change,
        )
        result = await self._command_handler.handle(line, ctx)
        if result is EXIT:
            self._running = False

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def _handle_chat(self, message: str) -> None:
        """Stream a response and persist both messages to the database.

        After the round-trip completes, the user message and assistant
        reply are saved via SessionManager so every conversation turn is
        durable from Step 7 onward.
        """
        # Input length check (Step 16a).
        if len(message) > self._max_input_length:
            self._view.show_error(
                f"Input too long ({len(message)} chars). Maximum: {self._max_input_length}"
            )
            return

        self._view.show_user_message(message)

        # Stream response token by token, with context trimming.
        full_response = ""
        try:
            self._view.show_assistant_prefix()
            async for token in self._engine.stream_chat(
                message, self._system_prompt, max_tokens=self._context_max_tokens
            ):
                if token:
                    full_response += token
                    self._view.stream_token(token)
            self._view.show_assistant_end()
        except Exception as exc:
            self._view.console.print()  # newline after streaming prefix
            error_msg = str(exc)
            if "api_key" in error_msg.lower() or "apikey" in error_msg.lower():
                self._view.show_error(
                    "API key not configured. Set OPENAI_API_KEY in your .env file."
                )
            elif "timeout" in error_msg.lower():
                self._view.show_error("Request timed out. Check your network or try again.")
            elif "connection" in error_msg.lower():
                self._view.show_error("Network error. Check your internet connection.")
            else:
                self._view.show_error(f"Chat error: {error_msg}")
            return

        # Persist both sides of the conversation immediately.
        if self._session is not None:
            try:
                await self._session_manager.add_message(self._session.id, "user", message)
                await self._session_manager.add_message(
                    self._session.id, "assistant", full_response
                )
            except Exception as exc:
                self._view.show_error(f"Failed to save message: {exc}")

    # ------------------------------------------------------------------
    # Callbacks (triggered by commands)
    # ------------------------------------------------------------------

    async def _on_user_change(self, new_name: str) -> None:
        """Switch to a different user and create a fresh session."""
        self._user_name = new_name
        user = await self._user_manager.get_user_by_name(new_name)
        self._session = await self._session_manager.create_session(
            user.id, title=f"Chat with {new_name}"
        )
        self._engine.clear_memory()
        self._view.show_info(f"New session created for user: {new_name}")

    async def _on_preset_change(self, preset_info: dict) -> None:
        """Load a preset's content as the system prompt."""
        self._system_prompt = preset_info["content"]
        self._view.set_system_prompt_active(True)

    async def _on_session_open(self, session_id: int) -> None:
        """Reopen a historical session and restore conversation context.

        If *session_id* is -1 (sentinel), a fresh session is created
        instead (used when the current session was deleted).
        """
        if session_id == -1:
            user = await self._user_manager.get_user_by_name(self._user_name)
            self._session = await self._session_manager.create_session(
                user.id, title=f"Chat with {self._user_name}"
            )
            self._engine.clear_memory()
            self._view.show_info("Fresh session created.")
            return

        try:
            session, messages = await self._session_manager.reopen_session(session_id)
        except Exception as exc:
            self._view.show_error(str(exc))
            return

        # Convert stored Message models → LangChain BaseMessage objects.
        langchain_messages: list[BaseMessage] = []
        for msg in messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))

        self._engine.load_messages(langchain_messages)
        self._session = session
        self._view.show_success(
            f"Reopened session {session.id}: {session.title!r} ({len(messages)} messages restored)"
        )

    async def _on_model_change(self, provider_name: str, model_name: str) -> None:
        """Switch the active model and push it to ChatEngine."""
        try:
            self._model_manager.switch_model(provider_name, model_name)
            new_model = self._model_manager.get_current_model()
            self._engine.set_model(new_model)
            self._view.show_success(f"Switched to [bold]{provider_name}[/] / {model_name}")
            self._view.show_welcome(
                self._user_name,
                session_id=self._session.id if self._session else None,
                system_prompt=self._system_prompt is not None,
                provider=self._model_manager.current_provider,
                model=self._model_manager.current_model,
            )
        except Exception as exc:
            self._view.show_error(str(exc))

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        """Install a handler for SIGINT (Ctrl+C) so we exit cleanly."""
        loop = asyncio.get_event_loop()

        def _on_sigint() -> None:
            self._running = False

        try:
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGINT.
            pass
