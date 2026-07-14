"""SQLite storage backend.

Implements :class:`StorageBackend` using aiosqlite for async-friendly
database access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.storage.base import StorageBackend
from langchain_chat.storage.database import init_database


class SQLiteBackend(StorageBackend):
    """Async SQLite backend implementing the StorageBackend contract.

    Usage::

        config = StorageConfig(database="data/chat.db")
        backend = SQLiteBackend(config)
        await backend.initialize()
        user = await backend.create_user("alice")
        await backend.close()
    """

    def __init__(self, config: StorageConfig) -> None:
        self._db_path: str = config.database
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create tables and open a persistent connection."""
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await init_database(self._db_path, conn=self._conn)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    async def create_user(self, username: str) -> dict[str, Any]:
        cursor = await self._db.execute("INSERT INTO users (username) VALUES (?)", (username,))
        await self._db.commit()
        row = await self._db.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (cursor.lastrowid,),
        )
        return dict(await row.fetchone())

    async def get_user(self, user_id: int) -> dict[str, Any] | None:
        cursor = await self._db.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def delete_user(self, user_id: int) -> bool:
        cursor = await self._db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_user_by_name(self, username: str) -> dict[str, Any] | None:
        cursor = await self._db.execute(
            "SELECT id, username, created_at FROM users WHERE username = ?", (username,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_users(self) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT id, username, created_at FROM users ORDER BY created_at DESC"
        )
        return [dict(row) for row in await cursor.fetchall()]

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    async def create_session(self, user_id: int, title: str = "") -> dict[str, Any]:
        cursor = await self._db.execute(
            "INSERT INTO sessions (user_id, title) VALUES (?, ?)", (user_id, title)
        )
        await self._db.commit()
        row = await self._db.execute(
            "SELECT id, user_id, title, created_at FROM sessions WHERE id = ?",
            (cursor.lastrowid,),
        )
        return dict(await row.fetchone())

    async def get_session(self, session_id: int) -> dict[str, Any] | None:
        cursor = await self._db.execute(
            "SELECT id, user_id, title, created_at FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_sessions(self, user_id: int | None = None) -> list[dict[str, Any]]:
        if user_id is not None:
            cursor = await self._db.execute(
                "SELECT id, user_id, title, created_at FROM sessions WHERE user_id = ? "
                "ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT id, user_id, title, created_at FROM sessions ORDER BY created_at DESC"
            )
        return [dict(row) for row in await cursor.fetchall()]

    async def delete_session(self, session_id: int) -> bool:
        cursor = await self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def add_message(self, session_id: int, role: str, content: str) -> dict[str, Any]:
        cursor = await self._db.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        await self._db.commit()
        row = await self._db.execute(
            "SELECT id, session_id, role, content, created_at FROM messages WHERE id = ?",
            (cursor.lastrowid,),
        )
        return dict(await row.fetchone())

    async def get_messages(self, session_id: int) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT id, session_id, role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        return [dict(row) for row in await cursor.fetchall()]

    # ------------------------------------------------------------------
    # Preset operations
    # ------------------------------------------------------------------

    async def create_preset(self, name: str, content: str) -> dict[str, Any]:
        cursor = await self._db.execute(
            "INSERT INTO presets (name, content) VALUES (?, ?)", (name, content)
        )
        await self._db.commit()
        row = await self._db.execute(
            "SELECT id, name, content, created_at FROM presets WHERE id = ?",
            (cursor.lastrowid,),
        )
        return dict(await row.fetchone())

    async def get_presets(self) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT id, name, content, created_at FROM presets ORDER BY created_at DESC"
        )
        return [dict(row) for row in await cursor.fetchall()]

    # ------------------------------------------------------------------
    # Config operations
    # ------------------------------------------------------------------

    async def get_config(self, key: str) -> str | None:
        cursor = await self._db.execute("SELECT value FROM configs WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return row["value"] if row else None

    async def save_config(self, key: str, value: str) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO configs (key, value) VALUES (?, ?)",
            (key, value),
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _db(self) -> aiosqlite.Connection:
        """Return the active connection, raising if not initialized."""
        if self._conn is None:
            raise RuntimeError(
                "SQLiteBackend is not initialized. Call await backend.initialize() first."
            )
        return self._conn
