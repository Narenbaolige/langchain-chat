"""MySQL storage backend.

Implements :class:`StorageBackend` using aiomysql for async-friendly
database access.  Suitable for production deployments where SQLite's
concurrency limitations are a concern.
"""

from __future__ import annotations

from typing import Any

import aiomysql

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.storage.base import StorageBackend

# ------------------------------------------------------------------
# MySQL DDL — mirrors the SQLite schema using MySQL types
# ------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(128) NOT NULL UNIQUE,
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sessions (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT          NOT NULL,
    preset_id  INT          NULL,
    title      VARCHAR(512) NOT NULL DEFAULT '',
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS messages (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT          NOT NULL,
    role       VARCHAR(32)  NOT NULL,
    content    TEXT         NOT NULL,
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS presets (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(128) NOT NULL,
    content     TEXT         NOT NULL,
    prompt_type VARCHAR(16)  NOT NULL DEFAULT 'user',
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS configs (
    `key`   VARCHAR(128) PRIMARY KEY,
    `value` TEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def _row_to_dict(row: tuple, columns: tuple[str, ...]) -> dict[str, Any]:
    """Convert a plain-tuple row + column names into a dict."""
    return dict(zip(columns, row, strict=True))


class MySQLBackend(StorageBackend):
    """Async MySQL backend implementing the StorageBackend contract.

    Uses a connection pool (``aiomysql.create_pool``) for efficient
    concurrent access.

    Usage::

        config = StorageConfig(type="mysql", mysql=MySQLConfig(...))
        backend = MySQLBackend(config)
        await backend.initialize()
        user = await backend.create_user("alice")
        await backend.close()
    """

    def __init__(self, config: StorageConfig) -> None:
        if config.mysql is None:
            raise ValueError("MySQL configuration is required for MySQLBackend")
        self._cfg = config.mysql
        self._pool: aiomysql.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the connection pool and initialise tables."""
        self._pool = await aiomysql.create_pool(
            host=self._cfg.host,
            port=self._cfg.port,
            db=self._cfg.database,
            user=self._cfg.user,
            password=self._cfg.password,
            minsize=1,
            maxsize=self._cfg.pool_size,
            autocommit=True,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                # MySQL does not support multi-statement in a single execute
                # unless CLIENT_MULTI_STATEMENTS is set.  We split the script.
                for stmt in _split_statements(_SCHEMA_SQL):
                    if stmt.strip():
                        await cur.execute(stmt)

    async def close(self) -> None:
        """Release the connection pool."""
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    async def create_user(self, username: str) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO users (username) VALUES (%s)", (username,))
                user_id = cur.lastrowid
                await cur.execute(
                    "SELECT id, username, created_at FROM users WHERE id = %s", (user_id,)
                )
                return _row_to_dict(await cur.fetchone(), ("id", "username", "created_at"))

    async def get_user(self, user_id: int) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, username, created_at FROM users WHERE id = %s", (user_id,)
                )
                row = await cur.fetchone()
                return _row_to_dict(row, ("id", "username", "created_at")) if row else None

    async def delete_user(self, user_id: int) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
                return cur.rowcount > 0

    async def get_user_by_name(self, username: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, username, created_at FROM users WHERE username = %s", (username,)
                )
                row = await cur.fetchone()
                return _row_to_dict(row, ("id", "username", "created_at")) if row else None

    async def list_users(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, username, created_at FROM users ORDER BY created_at DESC"
                )
                cols = ("id", "username", "created_at")
                return [_row_to_dict(r, cols) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    async def create_session(
        self, user_id: int, title: str = "", preset_id: int | None = None
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO sessions (user_id, title, preset_id) VALUES (%s, %s, %s)",
                    (user_id, title, preset_id),
                )
                sid = cur.lastrowid
                await cur.execute(
                    "SELECT id, user_id, preset_id, title, created_at, updated_at "
                    "FROM sessions WHERE id = %s",
                    (sid,),
                )
                cols = ("id", "user_id", "preset_id", "title", "created_at", "updated_at")
                return _row_to_dict(await cur.fetchone(), cols)

    async def get_session(self, session_id: int) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, user_id, preset_id, title, created_at, updated_at "
                    "FROM sessions WHERE id = %s",
                    (session_id,),
                )
                row = await cur.fetchone()
                cols = ("id", "user_id", "preset_id", "title", "created_at", "updated_at")
                return _row_to_dict(row, cols) if row else None

    async def list_sessions(self, user_id: int | None = None) -> list[dict[str, Any]]:
        cols = ("id", "user_id", "preset_id", "title", "created_at", "updated_at")
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                if user_id is not None:
                    await cur.execute(
                        "SELECT id, user_id, preset_id, title, created_at, updated_at "
                        "FROM sessions WHERE user_id = %s ORDER BY updated_at DESC",
                        (user_id,),
                    )
                else:
                    await cur.execute(
                        "SELECT id, user_id, preset_id, title, created_at, updated_at "
                        "FROM sessions ORDER BY updated_at DESC"
                    )
                return [_row_to_dict(r, cols) for r in await cur.fetchall()]

    async def delete_session(self, session_id: int) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
                return cur.rowcount > 0

    async def update_session(self, session_id: int, title: str) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE sessions SET title = %s, updated_at = NOW() WHERE id = %s",
                    (title, session_id),
                )
                if cur.rowcount == 0:
                    raise ValueError(f"Session id={session_id} not found")
                await cur.execute(
                    "SELECT id, user_id, preset_id, title, created_at, updated_at "
                    "FROM sessions WHERE id = %s",
                    (session_id,),
                )
                cols = ("id", "user_id", "preset_id", "title", "created_at", "updated_at")
                return _row_to_dict(await cur.fetchone(), cols)

    async def search_sessions(self, user_id: int, query: str) -> list[dict[str, Any]]:
        cols = ("id", "user_id", "preset_id", "title", "created_at", "updated_at")
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, user_id, preset_id, title, created_at, updated_at "
                    "FROM sessions WHERE user_id = %s AND title LIKE %s ORDER BY updated_at DESC",
                    (user_id, f"%{query}%"),
                )
                return [_row_to_dict(r, cols) for r in await cur.fetchall()]

    async def search_messages(self, session_id: int, query: str) -> list[dict[str, Any]]:
        cols = ("id", "session_id", "role", "content", "created_at")
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, session_id, role, content, created_at FROM messages "
                    "WHERE session_id = %s AND content LIKE %s ORDER BY created_at ASC",
                    (session_id, f"%{query}%"),
                )
                return [_row_to_dict(r, cols) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def add_message(self, session_id: int, role: str, content: str) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                    (session_id, role, content),
                )
                mid = cur.lastrowid
                # Bump session updated_at.
                await cur.execute(
                    "UPDATE sessions SET updated_at = NOW() WHERE id = %s", (session_id,)
                )
                await cur.execute(
                    "SELECT id, session_id, role, content, created_at FROM messages WHERE id = %s",
                    (mid,),
                )
                cols = ("id", "session_id", "role", "content", "created_at")
                return _row_to_dict(await cur.fetchone(), cols)

    async def get_messages(self, session_id: int) -> list[dict[str, Any]]:
        cols = ("id", "session_id", "role", "content", "created_at")
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, session_id, role, content, created_at FROM messages "
                    "WHERE session_id = %s ORDER BY created_at ASC",
                    (session_id,),
                )
                return [_row_to_dict(r, cols) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # Preset operations
    # ------------------------------------------------------------------

    async def create_preset(
        self, name: str, content: str, prompt_type: str = "user"
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO presets (name, content, prompt_type) VALUES (%s, %s, %s)",
                    (name, content, prompt_type),
                )
                pid = cur.lastrowid
                await cur.execute(
                    "SELECT id, name, content, prompt_type, created_at FROM presets WHERE id = %s",
                    (pid,),
                )
                cols = ("id", "name", "content", "prompt_type", "created_at")
                return _row_to_dict(await cur.fetchone(), cols)

    async def get_preset(self, preset_id: int) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, name, content, prompt_type, created_at FROM presets WHERE id = %s",
                    (preset_id,),
                )
                row = await cur.fetchone()
                cols = ("id", "name", "content", "prompt_type", "created_at")
                return _row_to_dict(row, cols) if row else None

    async def get_preset_by_name(self, name: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, name, content, prompt_type, created_at FROM presets WHERE name = %s",
                    (name,),
                )
                row = await cur.fetchone()
                cols = ("id", "name", "content", "prompt_type", "created_at")
                return _row_to_dict(row, cols) if row else None

    async def get_presets(self) -> list[dict[str, Any]]:
        cols = ("id", "name", "content", "prompt_type", "created_at")
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, name, content, prompt_type, created_at FROM presets "
                    "ORDER BY created_at DESC"
                )
                return [_row_to_dict(r, cols) for r in await cur.fetchall()]

    async def delete_preset(self, preset_id: int) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM presets WHERE id = %s", (preset_id,))
                return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Config operations
    # ------------------------------------------------------------------

    async def get_config(self, key: str) -> str | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT `value` FROM configs WHERE `key` = %s", (key,))
                row = await cur.fetchone()
                return row[0] if row else None

    async def save_config(self, key: str, value: str) -> None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO configs (`key`, `value`) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE `value` = VALUES(`value`)",
                    (key, value),
                )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def _db(self) -> aiomysql.Pool:
        """Return the active pool, raising if not initialised."""
        if self._pool is None:
            raise RuntimeError(
                "MySQLBackend is not initialized. Call await backend.initialize() first."
            )
        return self._pool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _split_statements(sql: str) -> list[str]:
    """Split a multi-statement SQL string on ``;`` for MySQL compat."""
    return [s.strip() for s in sql.split(";") if s.strip()]
