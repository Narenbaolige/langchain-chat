"""Chat view — terminal rendering via Rich.

ChatView is a pure display layer.  It owns all Rich calls so that the
rest of the TUI never imports ``rich`` directly.  This keeps rendering
concerns in one place and simplifies future UI toolkit swaps.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


class ChatView:
    """Terminal display manager backed by Rich.

    All methods are synchronous — they just print to the terminal.
    Async coordination lives in :class:`TuiChatApp`.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._system_prompt_active: bool = False

    # ------------------------------------------------------------------
    # Welcome & branding
    # ------------------------------------------------------------------

    def show_welcome(
        self, user_name: str, session_id: int | None = None, system_prompt: bool = False
    ) -> None:
        """Display the welcome banner with session context."""
        self.console.print()
        parts = [
            "[bold cyan]langchain-chat[/] — Enterprise AI ChatBot",
            f"User:    [green]{user_name}[/]",
        ]
        if session_id is not None:
            parts.append(f"Session: [dim]#{session_id}[/]")
        parts.append(
            f"System prompt: [{'green' if system_prompt else 'dim'}]{'active' if system_prompt else 'none'}[/]"
        )
        parts.append("Type [yellow]/help[/] for commands, [yellow]/quit[/] to exit")

        self.console.print(Panel.fit("\n".join(parts), title="Welcome", border_style="cyan"))
        self.console.print()

    # ------------------------------------------------------------------
    # Chat messages
    # ------------------------------------------------------------------

    def show_user_message(self, text: str) -> None:
        """Render a user message in a right-aligned blue panel."""
        self.console.print(
            Panel(text, title="You", border_style="blue", title_align="left"),
            justify="right",
        )

    def show_assistant_prefix(self) -> None:
        """Print the assistant label before streaming begins."""
        self.console.print(Text("🤖 ", style="bold green"), end="")

    def stream_token(self, token: str) -> None:
        """Print a single token in green with no trailing newline."""
        self.console.print(token, end="", style="green")

    def show_assistant_end(self) -> None:
        """Finish the assistant response with a subtle visual separator."""
        self.console.print()
        self.console.print(Rule(style="dim"))

    # ------------------------------------------------------------------
    # Status messages
    # ------------------------------------------------------------------

    def show_info(self, text: str) -> None:
        """Display an informational message."""
        self.console.print(Text(f" ℹ  {text}", style="dim"))

    def show_error(self, text: str) -> None:
        """Display an error message in a visible red panel."""
        self.console.print(
            Panel(f"[bold red]{text}[/]", border_style="red", title="Error", title_align="left")
        )

    def show_success(self, text: str) -> None:
        """Display a success message."""
        self.console.print(Text(f" ✓  {text}", style="bold green"))

    def show_warning(self, text: str) -> None:
        """Display a warning message."""
        self.console.print(Text(f" ⚠  {text}", style="bold yellow"))

    # ------------------------------------------------------------------
    # Tables — users, presets, sessions, commands
    # ------------------------------------------------------------------

    def show_users(self, users: list[dict]) -> None:
        """Display a numbered table of users."""
        if not users:
            self.show_info("No users found.")
            return
        table = Table(title="Users", border_style="blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("ID", style="dim")
        table.add_column("Username")
        table.add_column("Created")
        for i, u in enumerate(users, 1):
            created = self._short_ts(u.get("created_at", ""))
            table.add_row(str(i), str(u.get("id", "")), u.get("username", ""), created)
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
        table.add_column("Messages", justify="right")
        for s in sessions:
            updated = self._short_ts(s.get("updated_at", ""))
            msg_count = str(s.get("message_count", "?"))
            table.add_row(str(s.get("id", "")), s.get("title", ""), updated, msg_count)
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
            table.add_row(str(i), p.get("name", ""), ptype, preview)
        self.console.print(table)

    def show_help(self) -> None:
        """Display grouped command reference."""
        self.console.print()
        self.console.print(Rule("[bold yellow]Commands", style="yellow"))

        groups = [
            (
                "Chat",
                [
                    ("/clear", "Clear conversation memory"),
                    ("/stats", "Show conversation statistics"),
                ],
            ),
            (
                "Session",
                [
                    ("/sessions", "List recent sessions"),
                    ("/search <q>", "Search sessions by title"),
                    ("/rename <title>", "Rename current session"),
                    ("/open <id>", "Reopen a historical session"),
                    ("/delete-session <id>", "Delete a session"),
                ],
            ),
            (
                "User & Preset",
                [
                    ("/users", "List all users"),
                    ("/user <name>", "Switch user"),
                    ("/presets", "List available presets"),
                    ("/preset <name>", "Load a preset as system prompt"),
                    ("/system <text>", "Set a raw system prompt"),
                ],
            ),
            (
                "System",
                [
                    ("/help", "Show this help"),
                    ("/quit, /exit", "Exit the application"),
                ],
            ),
        ]

        for group_name, cmds in groups:
            table = Table(title=group_name, border_style="yellow", show_header=False)
            table.add_column("Command", style="bold cyan", width=24)
            table.add_column("Description")
            for cmd, desc in cmds:
                table.add_row(cmd, desc)
            self.console.print(table)
            self.console.print()

    def show_stats(self, message_count: int, session_id: int, user_name: str) -> None:
        """Display conversation statistics."""
        self.console.print(
            Panel.fit(
                f"Session ID:    [dim]{session_id}[/]\n"
                f"User:          [green]{user_name}[/]\n"
                f"Messages:      [yellow]{message_count}[/]\n"
                f"System prompt: [{'green' if self._system_prompt_active else 'dim'}]{'active' if self._system_prompt_active else 'none'}[/]",
                title="Stats",
                border_style="green",
            )
        )

    def set_system_prompt_active(self, active: bool) -> None:
        """Track whether a system prompt is active (for /stats & prompt)."""
        self._system_prompt_active = active

    @property
    def system_prompt_active(self) -> bool:
        """Whether a system prompt is currently active."""
        return self._system_prompt_active

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def build_prompt(self, user_name: str, session_id: int | None = None) -> str:
        """Build a dynamic prompt showing user and session context."""
        parts = [user_name]
        if session_id is not None:
            parts.append(f"#{session_id}")
        if self._system_prompt_active:
            parts.append("⚡")
        return f"[{':'.join(parts)}] > "

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _short_ts(timestamp: str | object) -> str:
        """Return a shortened timestamp for table display."""
        s = str(timestamp)
        return s.rsplit(".", 1)[0] if "." in s else s
