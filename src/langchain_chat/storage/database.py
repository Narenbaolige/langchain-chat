"""Database initialization for langchain-chat.

Handles creating the database file, tables, and any required directory
structure. All DDL lives here so the SQLite backend stays focused on CRUD.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

# ------------------------------------------------------------------
# DDL — table creation statements
# ------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT    NOT NULL UNIQUE,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    title      TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS presets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    prompt_type TEXT    NOT NULL DEFAULT 'user',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS configs (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


async def init_database(db_path: str, *, conn: aiosqlite.Connection | None = None) -> None:
    """Create the database file (and parent directory) plus all tables.

    Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS`` so
    existing tables are never dropped.

    Args:
        db_path: Path to the SQLite database file (e.g. ``data/chat.db``
            or ``:memory:`` for testing).
        conn: Optional existing connection. When provided, tables are
            created on *conn* (the caller keeps it alive). When omitted,
            a temporary connection is opened and closed — only suitable
            for file-backed databases, **not** ``:memory:``.
    """
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def _init(connection: aiosqlite.Connection) -> None:
        if db_path != ":memory:":
            await connection.execute("PRAGMA journal_mode=WAL")
        await connection.executescript(_SCHEMA_SQL)
        await connection.commit()

    if conn is not None:
        await _init(conn)
    else:
        async with aiosqlite.connect(db_path) as db:
            await _init(db)
