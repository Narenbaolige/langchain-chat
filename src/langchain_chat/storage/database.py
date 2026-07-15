"""Database initialization for langchain-chat.

Handles creating the database file, tables, schema migrations, and any
required directory structure. All DDL lives here so the SQLite backend
stays focused on CRUD.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

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
    preset_id  INTEGER,
    title      TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT    NOT NULL DEFAULT (datetime('now')),
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

# ------------------------------------------------------------------
# Schema migrations
# ------------------------------------------------------------------
# Each entry is a (migration_name, sql) pair.  Migrations must be
# idempotent — they catch duplicate-column errors so they are safe to
# run against a database that has already been upgraded.
# ------------------------------------------------------------------

_MIGRATIONS: list[tuple[str, str]] = [
    (
        "step7_add_preset_id_to_sessions",
        "ALTER TABLE sessions ADD COLUMN preset_id INTEGER",
    ),
    (
        "step7_add_updated_at_to_sessions",
        "ALTER TABLE sessions ADD COLUMN updated_at TEXT NOT NULL DEFAULT (datetime('now'))",
    ),
]


async def _run_migrations(conn: aiosqlite.Connection) -> None:
    """Apply any pending schema migrations to *conn*.

    Each migration is run inside its own try/except so that a migration
    that has already been applied (e.g. duplicate column) is silently
    skipped rather than aborting the entire init sequence.
    """
    for name, sql in _MIGRATIONS:
        try:
            await conn.execute(sql)
            await conn.commit()
            logger.info("Migration %s applied.", name)
        except aiosqlite.OperationalError as exc:
            # SQLite does not support "IF NOT EXISTS" for ALTER TABLE,
            # so we rely on exception swallowing for idempotency.
            logger.debug("Migration %s skipped: %s", name, exc)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


async def init_database(db_path: str, *, conn: aiosqlite.Connection | None = None) -> None:
    """Create the database file (and parent directory) plus all tables.

    Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS`` so
    existing tables are never dropped.  Pending schema migrations are
    applied after table creation.

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
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.executescript(_SCHEMA_SQL)
        await connection.commit()
        await _run_migrations(connection)

    if conn is not None:
        await _init(conn)
    else:
        async with aiosqlite.connect(db_path) as db:
            await _init(db)
