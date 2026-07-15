"""Storage factory — creates backend instances from configuration."""

from __future__ import annotations

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.storage.base import StorageBackend
from langchain_chat.storage.mysql_backend import MySQLBackend
from langchain_chat.storage.sqlite_backend import SQLiteBackend


class StorageFactory:
    """Factory that instantiates the correct StorageBackend based on config.

    Usage::

        config = get_config()
        backend = StorageFactory.create(config.storage)
        await backend.initialize()
    """

    @staticmethod
    def create(storage_config: StorageConfig) -> StorageBackend:
        """Return a StorageBackend instance matching *storage_config.type*.

        Args:
            storage_config: Storage configuration from ProjectConfig.

        Returns:
            A concrete StorageBackend instance.

        Raises:
            ValueError: If the storage type is unknown.
        """
        backend_type = storage_config.type.lower()

        if backend_type == "sqlite":
            return SQLiteBackend(storage_config)

        if backend_type == "mysql":
            return MySQLBackend(storage_config)

        raise ValueError(f"Unknown storage type: {backend_type!r}. Supported types: sqlite, mysql")
