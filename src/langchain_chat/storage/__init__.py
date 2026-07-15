"""Storage abstraction layer for langchain-chat.

Provides:
- StorageBackend: abstract interface that all backends must implement
- SQLiteBackend: concrete SQLite implementation
- MySQLBackend: concrete MySQL implementation (optional, requires aiomysql)
- StorageFactory: factory to create backend instances from config
"""

from langchain_chat.storage.base import StorageBackend
from langchain_chat.storage.database import init_database
from langchain_chat.storage.factory import StorageFactory
from langchain_chat.storage.file_backend import FileBackend
from langchain_chat.storage.sqlite_backend import SQLiteBackend

# MySQLBackend is optional — only usable when aiomysql is installed.
try:
    from langchain_chat.storage.mysql_backend import MySQLBackend
except ImportError:  # pragma: no cover — optional dependency
    MySQLBackend = None  # type: ignore[assignment]

__all__ = [
    "FileBackend",
    "MySQLBackend",
    "SQLiteBackend",
    "StorageBackend",
    "StorageFactory",
    "init_database",
]
