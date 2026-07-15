"""Storage abstraction layer for langchain-chat.

Provides:
- StorageBackend: abstract interface that all backends must implement
- SQLiteBackend: concrete SQLite implementation
- MySQLBackend: concrete MySQL implementation
- StorageFactory: factory to create backend instances from config
"""

from langchain_chat.storage.base import StorageBackend
from langchain_chat.storage.database import init_database
from langchain_chat.storage.factory import StorageFactory
from langchain_chat.storage.mysql_backend import MySQLBackend
from langchain_chat.storage.sqlite_backend import SQLiteBackend

__all__ = [
    "MySQLBackend",
    "SQLiteBackend",
    "StorageBackend",
    "StorageFactory",
    "init_database",
]
