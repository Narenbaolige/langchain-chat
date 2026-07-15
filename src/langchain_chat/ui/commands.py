"""Command handler for the TUI.

Parses slash-commands, dispatches to registered handlers, and returns
actionable results back to the app loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from langchain_chat.core.chat_engine import ChatEngine
from langchain_chat.core.model_manager import ModelManager
from langchain_chat.core.prompt_manager import PromptManager
from langchain_chat.core.session_manager import SessionManager
from langchain_chat.core.user_manager import UserManager
from langchain_chat.models.session import Session
from langchain_chat.ui.chat_view import ChatView

# ------------------------------------------------------------------
# Command result types
# ------------------------------------------------------------------


class ActionResult:
    """Sentinel returned by command handlers to signal app behaviour."""


CONTINUE = ActionResult()  # keep running, nothing special
EXIT = ActionResult()  # shut down the app


# ------------------------------------------------------------------
# Command context — passed to every handler
# ------------------------------------------------------------------


@dataclass
class CommandContext:
    """Dependencies available to command handlers.

    Handlers receive this context and may call managers, mutate state,
    or signal the app loop via their return value.
    """

    user_manager: UserManager
    prompt_manager: PromptManager
    session_manager: SessionManager
    model_manager: ModelManager
    engine: ChatEngine
    view: ChatView
    current_user_name: str
    session: Session | None

    # Callbacks for state changes the app needs to know about.
    on_user_change: Callable[[str], Coroutine[Any, Any, None]] | None = None
    on_preset_change: Callable[[dict], Coroutine[Any, Any, None]] | None = None
    on_session_open: Callable[[int], Coroutine[Any, Any, None]] | None = None
    on_model_change: Callable[[str, str], Coroutine[Any, Any, None]] | None = None


# ------------------------------------------------------------------
# Handler registry
# ------------------------------------------------------------------

HandlerFunc = Callable[[CommandContext, str], Coroutine[Any, Any, ActionResult | None]]


class CommandHandler:
    """Parses and dispatches slash-commands.

    Usage::

        ctx = CommandContext(...)
        handler = CommandHandler()
        result = await handler.handle("/help", ctx)
        if result is EXIT:
            ...
    """

    def __init__(self) -> None:
        self._registry: dict[str, HandlerFunc] = {}

        # Register built-in commands.
        for name, func in _BUILTIN_COMMANDS.items():
            self._registry[name] = func

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def is_command(self, line: str) -> bool:
        """Return True if *line* looks like a slash command."""
        return bool(line) and line.strip().startswith("/")

    async def handle(self, line: str, ctx: CommandContext) -> ActionResult | None:
        """Parse and execute a single command line.

        Args:
            line: Full input line (e.g. ``"/user alice"``).
            ctx: Dependencies for the handler.

        Returns:
            An :class:`ActionResult` sentinel (e.g. ``EXIT``) or ``None``.
        """
        cmd, args = self._parse(line)
        handler = self._registry.get(cmd)
        if handler is None:
            ctx.view.show_error(f"Unknown command: {cmd!r}. Type /help for available commands.")
            return CONTINUE
        try:
            return await handler(ctx, args)
        except Exception as exc:
            ctx.view.show_error(str(exc))
            return CONTINUE

    # --------------------------------------------------------------
    # Internal
    # --------------------------------------------------------------

    @staticmethod
    def _parse(line: str) -> tuple[str, str]:
        """Split ``/cmd args...`` into (cmd, args)."""
        stripped = line.strip()
        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        return cmd, args


# ------------------------------------------------------------------
# Built-in command handlers
# ------------------------------------------------------------------
# Each handler is async and receives (ctx, args).  It returns
# CONTINUE, EXIT, or None (treated as CONTINUE).
# ------------------------------------------------------------------


async def _cmd_help(ctx: CommandContext, _args: str) -> ActionResult:
    """Show command reference."""
    ctx.view.show_help()
    return CONTINUE


async def _cmd_quit(ctx: CommandContext, _args: str) -> ActionResult:
    """Exit the application."""
    ctx.view.show_info("Goodbye!")
    return EXIT


async def _cmd_exit(ctx: CommandContext, _args: str) -> ActionResult:
    """Alias for /quit."""
    return await _cmd_quit(ctx, _args)


async def _cmd_clear(ctx: CommandContext, _args: str) -> ActionResult:
    """Clear conversation memory."""
    ctx.engine.clear_memory()
    ctx.view.show_success("Conversation memory cleared.")
    return CONTINUE


async def _cmd_users(ctx: CommandContext, args: str) -> ActionResult:
    """List all users."""
    users = await ctx.user_manager.list_users()
    user_dicts = [
        {
            "id": u.id,
            "username": u.username,
            "created_at": str(u.created_at),
        }
        for u in users
    ]
    ctx.view.show_users(user_dicts)
    return CONTINUE


async def _cmd_user(ctx: CommandContext, args: str) -> ActionResult:
    """Switch to a different user."""
    name = args.strip()
    if not name:
        ctx.view.show_error("Usage: /user <username>  — switch to a different user")
        return CONTINUE
    try:
        user = await ctx.user_manager.get_user_by_name(name)
        if ctx.on_user_change:
            await ctx.on_user_change(user.username)
        ctx.view.show_success(f"Switched to user: {user.username}")
    except Exception as exc:
        ctx.view.show_error(str(exc))
    return CONTINUE


async def _cmd_presets(ctx: CommandContext, _args: str) -> ActionResult:
    """List available presets."""
    presets = await ctx.prompt_manager.list_presets()
    preset_dicts = [
        {
            "id": p.id,
            "name": p.name,
            "content": p.content,
            "prompt_type": p.prompt_type,
        }
        for p in presets
    ]
    ctx.view.show_presets(preset_dicts)
    return CONTINUE


async def _cmd_preset(ctx: CommandContext, args: str) -> ActionResult:
    """Load a preset as the system prompt."""
    name = args.strip()
    if not name:
        ctx.view.show_error("Usage: /preset <name>  — load a preset as system prompt")
        return CONTINUE
    try:
        preset = await ctx.prompt_manager.get_preset_by_name(name)
        if ctx.on_preset_change:
            await ctx.on_preset_change({"name": preset.name, "content": preset.content})
        ctx.view.show_success(f"Loaded preset: {preset.name} ({preset.prompt_type})")
    except Exception as exc:
        ctx.view.show_error(str(exc))
    return CONTINUE


async def _cmd_system(ctx: CommandContext, args: str) -> ActionResult:
    """Set a raw system prompt directly."""
    text = args.strip()
    if not text:
        ctx.view.show_error("Usage: /system <text>  — set a raw system prompt")
        return CONTINUE
    if ctx.on_preset_change:
        await ctx.on_preset_change({"name": "(inline)", "content": text})
    ctx.view.show_success("System prompt set.")
    return CONTINUE


async def _cmd_stats(ctx: CommandContext, args: str) -> ActionResult:
    """Show conversation statistics."""
    session_id = ctx.session.id if ctx.session else -1
    ctx.view.show_stats(
        message_count=ctx.engine.message_count,
        session_id=session_id,
        user_name=ctx.current_user_name,
    )
    return CONTINUE


# ------------------------------------------------------------------
# Step 8 — session management commands
# ------------------------------------------------------------------


async def _cmd_sessions(ctx: CommandContext, _args: str) -> ActionResult:
    """List recent sessions for the current user."""
    user = await ctx.user_manager.get_user_by_name(ctx.current_user_name)
    sessions = await ctx.session_manager.list_sessions(user_id=user.id)
    if not sessions:
        ctx.view.show_info("No sessions found. Start chatting to create one.")
        return CONTINUE

    session_dicts: list[dict] = []
    for s in sessions:
        msgs = await ctx.session_manager.get_messages(s.id)
        session_dicts.append(
            {
                "id": s.id,
                "title": s.title,
                "updated_at": str(s.updated_at),
                "message_count": str(len(msgs)),
            }
        )
    ctx.view.show_sessions(session_dicts)
    return CONTINUE


async def _cmd_search(ctx: CommandContext, args: str) -> ActionResult:
    """Search session titles."""
    query = args.strip()
    if not query:
        ctx.view.show_error("Usage: /search <query>  — search sessions by title")
        return CONTINUE
    user = await ctx.user_manager.get_user_by_name(ctx.current_user_name)
    sessions = await ctx.session_manager.search_sessions(user.id, query)
    if not sessions:
        ctx.view.show_info(f"No sessions matching {query!r}.")
        return CONTINUE
    session_dicts: list[dict] = []
    for s in sessions:
        msgs = await ctx.session_manager.get_messages(s.id)
        session_dicts.append(
            {
                "id": s.id,
                "title": s.title,
                "updated_at": str(s.updated_at),
                "message_count": str(len(msgs)),
            }
        )
    ctx.view.show_sessions(session_dicts)
    return CONTINUE


async def _cmd_rename(ctx: CommandContext, args: str) -> ActionResult:
    """Rename the current session."""
    new_title = args.strip()
    if not new_title:
        ctx.view.show_error("Usage: /rename <new title>  — rename current session")
        return CONTINUE
    if ctx.session is None:
        ctx.view.show_error("No active session.")
        return CONTINUE
    try:
        updated = await ctx.session_manager.update_session(ctx.session.id, new_title)
        ctx.view.show_success(f"Session renamed to: {updated.title}")
    except Exception as exc:
        ctx.view.show_error(str(exc))
    return CONTINUE


async def _cmd_open(ctx: CommandContext, args: str) -> ActionResult:
    """Reopen a historical session by ID."""
    sid_str = args.strip()
    if not sid_str:
        ctx.view.show_error("Usage: /open <session_id>")
        return CONTINUE
    try:
        sid = int(sid_str)
    except ValueError:
        ctx.view.show_error(f"Invalid session ID: {sid_str!r}")
        return CONTINUE

    if ctx.on_session_open:
        await ctx.on_session_open(sid)
    return CONTINUE


async def _cmd_delete_session(ctx: CommandContext, args: str) -> ActionResult:
    """Delete a session by ID."""
    sid_str = args.strip()
    if not sid_str:
        ctx.view.show_error("Usage: /delete-session <session_id>")
        return CONTINUE
    try:
        sid = int(sid_str)
    except ValueError:
        ctx.view.show_error(f"Invalid session ID: {sid_str!r}")
        return CONTINUE

    deleted = await ctx.session_manager.delete_session(sid)
    if deleted:
        ctx.view.show_success(f"Session {sid} deleted.")
        if ctx.session and ctx.session.id == sid:
            # Current session was deleted — create a fresh one.
            if ctx.on_session_open:
                await ctx.on_session_open(-1)  # signal to create new
    else:
        ctx.view.show_error(f"Session {sid} not found.")
    return CONTINUE


# ------------------------------------------------------------------
# Step 10 — Model management commands
# ------------------------------------------------------------------


async def _cmd_providers(ctx: CommandContext, _args: str) -> ActionResult:
    """List available providers."""
    providers = ctx.model_manager.list_providers()
    current = ctx.model_manager.current_provider
    lines = []
    for p in providers:
        marker = " →" if p == current else "  "
        info = ctx.model_manager.get_provider_info(p)
        lines.append(f"{marker} [bold]{p}[/] — {len(info['models'])} models")
        if p == current:
            lines.append(f"        current: [yellow]{ctx.model_manager.current_model}[/]")
    ctx.view.console.print("\n".join(lines))
    return CONTINUE


async def _cmd_models(ctx: CommandContext, args: str) -> ActionResult:
    """List models for a provider (defaults to current)."""
    provider = args.strip() or ctx.model_manager.current_provider
    try:
        info = ctx.model_manager.get_provider_info(provider)
    except Exception as exc:
        ctx.view.show_error(str(exc))
        return CONTINUE

    current = ctx.model_manager.current_model
    lines = [f"Provider: [bold]{provider}[/]"]
    for m in info["models"]:
        marker = " →" if provider == ctx.model_manager.current_provider and m == current else "  "
        lines.append(f"{marker} {m}")
    ctx.view.console.print("\n".join(lines))
    return CONTINUE


async def _cmd_model(ctx: CommandContext, args: str) -> ActionResult:
    """Switch model within the current provider."""
    model_name = args.strip()
    if not model_name:
        ctx.view.show_error("Usage: /model <name>  — switch model in current provider")
        return CONTINUE
    if ctx.on_model_change:
        await ctx.on_model_change(ctx.model_manager.current_provider, model_name)
    return CONTINUE


async def _cmd_provider(ctx: CommandContext, args: str) -> ActionResult:
    """Switch provider (auto-selects its default model)."""
    name = args.strip()
    if not name:
        ctx.view.show_error("Usage: /provider <name>  — switch provider")
        return CONTINUE
    try:
        info = ctx.model_manager.get_provider_info(name)
        default = info["default_model"]
        if ctx.on_model_change:
            await ctx.on_model_change(name, default)
    except Exception as exc:
        ctx.view.show_error(str(exc))
    return CONTINUE


# Registry of all built-in commands.
_BUILTIN_COMMANDS: dict[str, HandlerFunc] = {
    "help": _cmd_help,
    "quit": _cmd_quit,
    "exit": _cmd_exit,
    "clear": _cmd_clear,
    "user": _cmd_user,
    "users": _cmd_users,
    "preset": _cmd_preset,
    "presets": _cmd_presets,
    "system": _cmd_system,
    "stats": _cmd_stats,
    "sessions": _cmd_sessions,
    "search": _cmd_search,
    "rename": _cmd_rename,
    "open": _cmd_open,
    "delete-session": _cmd_delete_session,
    "providers": _cmd_providers,
    "models": _cmd_models,
    "model": _cmd_model,
    "provider": _cmd_provider,
}
