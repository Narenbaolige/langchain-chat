"""File-based storage backend.

Implements :class:`StorageBackend` using JSON files on disk.  Each table
is a separate ``.json`` file inside a dedicated directory.  Suitable for
lightweight, zero-dependency deployments and as a reference implementation.

Thread safety within a single event loop is guaranteed via :class:`asyncio.Lock`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------
# FileBackend
# ------------------------------------------------------------------


class FileBackend(StorageBackend):
    """Async file-backed storage using JSON.

    Each logical table lives in its own ``.json`` file under the
    configured ``database`` directory.  All I/O is async (via a thread
    executor) and writes are protected by an asyncio lock.

    Usage::

        config = StorageConfig(type="file", database="data/file_storage")
        backend = FileBackend(config)
        await backend.initialize()
        user = await backend.create_user("alice")
        await backend.close()
    """

    def __init__(self, config: StorageConfig) -> None:
        self._dir = Path(config.database)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Ensure the storage directory and table files exist."""
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("FileBackend initialised at %s", self._dir)
        for name in ("users", "sessions", "messages", "presets", "configs"):
            path = self._dir / f"{name}.json"
            if not path.exists():
                await self._write_json(path, [])

    async def close(self) -> None:
        """No-op — files are flushed on every write."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _read_json(self, table: str) -> list[dict[str, Any]]:
        path = self._dir / f"{table}.json"
        loop = asyncio.get_event_loop()

        def _read() -> list[dict[str, Any]]:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)

        return await loop.run_in_executor(None, _read)

    async def _write_json(self, path: Path, data: list[dict[str, Any]]) -> None:
        loop = asyncio.get_event_loop()

        def _write() -> None:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, default=str)

        await loop.run_in_executor(None, _write)

    async def _table_path(self, table: str) -> Path:
        return self._dir / f"{table}.json"

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    async def create_user(self, username: str) -> dict[str, Any]:
        async with self._lock:
            rows = await self._read_json("users")
            new_id = max((r["id"] for r in rows), default=0) + 1
            now = _now_iso()
            record = {"id": new_id, "username": username, "created_at": now}
            rows.append(record)
            await self._write_json(self._dir / "users.json", rows)
            return dict(record)

    async def get_user(self, user_id: int) -> dict[str, Any] | None:
        rows = await self._read_json("users")
        for r in rows:
            if r["id"] == user_id:
                return dict(r)
        return None

    async def delete_user(self, user_id: int) -> bool:
        async with self._lock:
            rows = await self._read_json("users")
            new_rows = [r for r in rows if r["id"] != user_id]
            if len(new_rows) == len(rows):
                return False
            await self._write_json(self._dir / "users.json", new_rows)
            return True

    async def get_user_by_name(self, username: str) -> dict[str, Any] | None:
        rows = await self._read_json("users")
        for r in rows:
            if r["username"] == username:
                return dict(r)
        return None

    async def list_users(self) -> list[dict[str, Any]]:
        rows = await self._read_json("users")
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    async def create_session(
        self, user_id: int, title: str = "", preset_id: int | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            rows = await self._read_json("sessions")
            new_id = max((r["id"] for r in rows), default=0) + 1
            now = _now_iso()
            record = {
                "id": new_id,
                "user_id": user_id,
                "preset_id": preset_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
            }
            rows.append(record)
            await self._write_json(self._dir / "sessions.json", rows)
            return dict(record)

    async def get_session(self, session_id: int) -> dict[str, Any] | None:
        rows = await self._read_json("sessions")
        for r in rows:
            if r["id"] == session_id:
                return dict(r)
        return None

    async def list_sessions(
        self, user_id: int | None = None, limit: int = 0, offset: int = 0
    ) -> list[dict[str, Any]]:
        rows = await self._read_json("sessions")
        if user_id is not None:
            rows = [r for r in rows if r.get("user_id") == user_id]
        rows.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        if limit > 0:
            rows = rows[offset : offset + limit]
        return [dict(r) for r in rows]

    async def delete_session(self, session_id: int) -> bool:
        async with self._lock:
            rows = await self._read_json("sessions")
            # Also cascade-delete messages for this session.
            msgs = await self._read_json("messages")
            msgs = [m for m in msgs if m.get("session_id") != session_id]
            await self._write_json(self._dir / "messages.json", msgs)

            new_rows = [r for r in rows if r["id"] != session_id]
            if len(new_rows) == len(rows):
                return False
            await self._write_json(self._dir / "sessions.json", new_rows)
            return True

    async def update_session(self, session_id: int, title: str) -> dict[str, Any]:
        async with self._lock:
            rows = await self._read_json("sessions")
            for r in rows:
                if r["id"] == session_id:
                    r["title"] = title
                    r["updated_at"] = _now_iso()
                    await self._write_json(self._dir / "sessions.json", rows)
                    return dict(r)
            raise ValueError(f"Session id={session_id} not found")

    async def search_sessions(self, user_id: int, query: str) -> list[dict[str, Any]]:
        rows = await self._read_json("sessions")
        results = [
            r
            for r in rows
            if r.get("user_id") == user_id and query.lower() in r.get("title", "").lower()
        ]
        results.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        return [dict(r) for r in results]

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def add_message(self, session_id: int, role: str, content: str) -> dict[str, Any]:
        async with self._lock:
            rows = await self._read_json("messages")
            new_id = max((r["id"] for r in rows), default=0) + 1
            now = _now_iso()
            record = {
                "id": new_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "created_at": now,
            }
            rows.append(record)
            await self._write_json(self._dir / "messages.json", rows)

            # Bump session updated_at.
            sessions = await self._read_json("sessions")
            for s in sessions:
                if s["id"] == session_id:
                    s["updated_at"] = now
                    break
            await self._write_json(self._dir / "sessions.json", sessions)

            return dict(record)

    async def get_messages(self, session_id: int) -> list[dict[str, Any]]:
        rows = await self._read_json("messages")
        rows = [r for r in rows if r.get("session_id") == session_id]
        rows.sort(key=lambda r: r.get("created_at", ""))
        return [dict(r) for r in rows]

    async def search_messages(self, session_id: int, query: str) -> list[dict[str, Any]]:
        rows = await self._read_json("messages")
        results = [
            r
            for r in rows
            if r.get("session_id") == session_id and query.lower() in r.get("content", "").lower()
        ]
        results.sort(key=lambda r: r.get("created_at", ""))
        return [dict(r) for r in results]

    # ------------------------------------------------------------------
    # Preset operations
    # ------------------------------------------------------------------

    async def create_preset(
        self, name: str, content: str, prompt_type: str = "user"
    ) -> dict[str, Any]:
        async with self._lock:
            rows = await self._read_json("presets")
            new_id = max((r["id"] for r in rows), default=0) + 1
            now = _now_iso()
            record = {
                "id": new_id,
                "name": name,
                "content": content,
                "prompt_type": prompt_type,
                "created_at": now,
            }
            rows.append(record)
            await self._write_json(self._dir / "presets.json", rows)
            return dict(record)

    async def get_preset(self, preset_id: int) -> dict[str, Any] | None:
        rows = await self._read_json("presets")
        for r in rows:
            if r["id"] == preset_id:
                return dict(r)
        return None

    async def get_preset_by_name(self, name: str) -> dict[str, Any] | None:
        rows = await self._read_json("presets")
        for r in rows:
            if r["name"] == name:
                return dict(r)
        return None

    async def get_presets(self) -> list[dict[str, Any]]:
        rows = await self._read_json("presets")
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return [dict(r) for r in rows]

    async def delete_preset(self, preset_id: int) -> bool:
        async with self._lock:
            rows = await self._read_json("presets")
            new_rows = [r for r in rows if r["id"] != preset_id]
            if len(new_rows) == len(rows):
                return False
            await self._write_json(self._dir / "presets.json", new_rows)
            return True

    # ------------------------------------------------------------------
    # Config operations
    # ------------------------------------------------------------------

    async def get_config(self, key: str) -> str | None:
        rows = await self._read_json("configs")
        for r in rows:
            if r.get("key") == key:
                return r["value"]
        return None

    async def save_config(self, key: str, value: str) -> None:
        async with self._lock:
            rows = await self._read_json("configs")
            for r in rows:
                if r.get("key") == key:
                    r["value"] = value
                    await self._write_json(self._dir / "configs.json", rows)
                    return
            rows.append({"key": key, "value": value})
            await self._write_json(self._dir / "configs.json", rows)
