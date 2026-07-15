"""Storage factory — creates backend instances from configuration."""

from __future__ import annotations

from langchain_chat.core.config_models import StorageConfig
from langchain_chat.storage.base import StorageBackend
from langchain_chat.storage.file_backend import FileBackend
from langchain_chat.storage.sqlite_backend import SQLiteBackend


class StorageFactory:
    """Factory that instantiates the correct StorageBackend based on config.

    Usage::

        config = get_config()
        backend = StorageFactory.create(config.storage)
        await backend.initialize()

    MySQL backend is imported lazily — you only need ``aiomysql`` installed
    when ``storage.type`` is ``"mysql"``.
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
            ImportError: If ``type=mysql`` is requested but ``aiomysql`` is
                not installed.
        """
        backend_type = storage_config.type.lower()

        if backend_type == "sqlite":
            return SQLiteBackend(storage_config)

        if backend_type == "mysql":
            from langchain_chat.storage.mysql_backend import MySQLBackend  # noqa: PLC0415

            return MySQLBackend(storage_config)

        if backend_type == "file":
            return FileBackend(storage_config)

        raise ValueError(
            f"Unknown storage type: {backend_type!r}. Supported types: sqlite, mysql, file"
        )
