"""Chat view — terminal rendering via Rich.

ChatView is a pure display layer.  It owns all Rich calls so that the
rest of the TUI never imports ``rich`` directly.  This keeps rendering
concerns in one place and simplifies future UI toolkit swaps.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class ChatView:
    """Terminal display manager backed by Rich.

    All methods are synchronous — they just print to the terminal.
    Async coordination lives in :class:`TuiChatApp`.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    # ------------------------------------------------------------------
    # Welcome & branding
    # ------------------------------------------------------------------

    def show_welcome(self, user_name: str) -> None:
        """Display the welcome banner with current user."""
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[bold cyan]langchain-chat[/] — Enterprise AI ChatBot\n"
                f"User: [green]{user_name}[/] | "
                f"Type [yellow]/help[/] for commands, [yellow]/quit[/] to exit",
                title="Welcome",
                border_style="cyan",
            )
        )
        self.console.print()

    # ------------------------------------------------------------------
    # Chat messages
    # ------------------------------------------------------------------

    def show_user_message(self, text: str) -> None:
        """Render a user message (right-aligned, blue)."""
        msg = Text(text, style="bold blue")
        self.console.print(Text("You ", style="bold blue"), end="")
        self.console.print(msg)

    def show_assistant_prefix(self) -> None:
        """Print the assistant label before streaming begins."""
        self.console.print(Text("🤖 ", style="bold green"), end="")

    def stream_token(self, token: str) -> None:
        """Print a single token with no trailing newline."""
        self.console.print(token, end="")

    def show_assistant_end(self) -> None:
        """Print a newline after the assistant response finishes."""
        self.console.print()

    # ------------------------------------------------------------------
    # Status messages
    # ------------------------------------------------------------------

    def show_info(self, text: str) -> None:
        """Display an informational message."""
        self.console.print(Text(text, style="dim italic"))

    def show_error(self, text: str) -> None:
        """Display an error message."""
        self.console.print(Text(f"Error: {text}", style="bold red"))

    def show_success(self, text: str) -> None:
        """Display a success message."""
        self.console.print(Text(text, style="bold green"))

    # ------------------------------------------------------------------
    # Tables — users, presets, commands
    # ------------------------------------------------------------------

    def show_users(self, users: list[dict]) -> None:
        """Display a numbered table of users."""
        table = Table(title="Users", border_style="blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("ID", style="dim")
        table.add_column("Username")
        table.add_column("Created")
        for i, u in enumerate(users, 1):
            created = u["created_at"].split("T")[0] if hasattr(u, "created_at") else ""
            table.add_row(str(i), str(u["id"]), u["username"], str(created))
        self.console.print(table)

    def show_sessions(self, sessions: list[dict]) -> None:
        """Display a numbered table of sessions."""
        if not sessions:
            self.show_info("No sessions found.")
            return
        table = Table(title="Sessions", border_style="cyan")
        table.add_column("ID", style="dim")
        table.add_column("Title")
        table.add_column("Updated")
        for s in sessions:
            updated = s.get("updated_at", "") or ""
            updated_short = updated.rsplit(".", 1)[0] if "." in str(updated) else str(updated)
            table.add_row(str(s["id"]), s.get("title", ""), str(updated_short))
        self.console.print(table)

    def show_presets(self, presets: list[dict]) -> None:
        """Display a numbered table of presets."""
        if not presets:
            self.show_info("No presets available.")
            return
        table = Table(title="Presets", border_style="magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Preview")
        for i, p in enumerate(presets, 1):
            ptype = p.get("prompt_type", "user")
            preview = (p.get("content", "") or "")[:60]
            table.add_row(str(i), p["name"], ptype, preview)
        self.console.print(table)

    def show_help(self) -> None:
        """Display the command reference."""
        table = Table(title="Commands", border_style="yellow")
        table.add_column("Command", style="bold cyan")
        table.add_column("Description")
        commands = [
            ("/help", "Show this help"),
            ("/quit, /exit", "Exit the application"),
            ("/clear", "Clear the current conversation memory"),
            ("/user <name>", "Switch to a different user"),
            ("/users", "List all users"),
            ("/preset <name>", "Load a preset as system prompt"),
            ("/presets", "List available presets"),
            ("/system <text>", "Set a raw system prompt"),
            ("/stats", "Show conversation statistics"),
            ("/sessions", "List recent sessions (Step 8)"),
            ("/search <query>", "Search sessions by title (Step 8)"),
            ("/rename <title>", "Rename current session (Step 8)"),
            ("/open <id>", "Reopen a historical session (Step 8)"),
            ("/delete-session <id>", "Delete a session by ID (Step 8)"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self.console.print(table)

    def show_stats(self, message_count: int, session_id: int, user_name: str) -> None:
        """Display conversation statistics."""
        self.console.print(
            Panel.fit(
                f"Session ID:    [dim]{session_id}[/]\n"
                f"User:          [green]{user_name}[/]\n"
                f"Messages:      [yellow]{message_count}[/]\n"
                f"System prompt: [dim]{'active' if self._system_prompt_active else 'none'}[/]",
                title="Stats",
                border_style="green",
            )
        )

    def set_system_prompt_active(self, active: bool) -> None:
        """Track whether a system prompt is active (for /stats)."""
        self._system_prompt_active = active

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def prompt_input(self, prompt_text: str = "> ") -> str:
        """Read a line of input (synchronous, wrapped by caller)."""
        return input(prompt_text)

    def show_prompt(self, text: str) -> None:
        """Show prompt text without reading (used for custom prompts)."""
        self.console.print(Text(text, style="bold"), end="")
