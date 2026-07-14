"""Prompt management — Core Business Layer module.

PromptManager orchestrates preset (prompt template) operations through
the StorageBackend interface. It handles business logic such as duplicate
name detection, but never touches the database directly.
"""

from __future__ import annotations

from langchain_chat.models.preset import Preset, PromptType
from langchain_chat.storage.base import StorageBackend


class DuplicatePresetError(Exception):
    """Raised when attempting to create a preset with an existing name."""


class PresetNotFoundError(Exception):
    """Raised when a requested preset does not exist."""


class PromptManager:
    """Business-layer prompt / preset management.

    Depends on StorageBackend (not SQLiteBackend!) so it works with
    any backend the factory provides.

    Usage::

        storage = StorageFactory.create(config.storage)
        await storage.initialize()
        manager = PromptManager(storage)
        preset = await manager.create_preset("greeting", "Hello, {name}!")
    """

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_preset(
        self, name: str, content: str, prompt_type: PromptType = "user"
    ) -> Preset:
        """Create a new prompt preset.

        Args:
            name: Unique preset name.
            content: Prompt template text.
            prompt_type: ``"system"`` or ``"user"`` (default).

        Returns:
            The newly created Preset.

        Raises:
            DuplicatePresetError: If *name* is already taken.
        """
        if not name or not name.strip():
            raise ValueError("Preset name must not be empty")
        if not content or not content.strip():
            raise ValueError("Preset content must not be empty")

        name = name.strip()
        content = content.strip()

        existing = await self._storage.get_preset_by_name(name)
        if existing is not None:
            raise DuplicatePresetError(f"Preset {name!r} already exists")

        data = await self._storage.create_preset(name, content, prompt_type)
        return Preset.model_validate(data)

    async def get_preset(self, preset_id: int) -> Preset:
        """Retrieve a preset by ID.

        Raises:
            PresetNotFoundError: If no preset has that ID.
        """
        data = await self._storage.get_preset(preset_id)
        if data is None:
            raise PresetNotFoundError(f"Preset id={preset_id} not found")
        return Preset.model_validate(data)

    async def get_preset_by_name(self, name: str) -> Preset:
        """Retrieve a preset by name.

        Raises:
            PresetNotFoundError: If no preset has that name.
        """
        data = await self._storage.get_preset_by_name(name)
        if data is None:
            raise PresetNotFoundError(f"Preset {name!r} not found")
        return Preset.model_validate(data)

    async def list_presets(self) -> list[Preset]:
        """List all presets.

        Returns:
            List of Presets (may be empty).
        """
        rows = await self._storage.get_presets()
        return [Preset.model_validate(r) for r in rows]

    async def delete_preset(self, preset_id: int) -> bool:
        """Delete a preset by ID (idempotent — no error if missing).

        Returns:
            True if a preset was deleted, False if already gone.
        """
        return await self._storage.delete_preset(preset_id)
