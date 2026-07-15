"""Tests for MySQLBackend — structure, factory, and DDL validation.

Full integration tests require a running MySQL server and are skipped by
default.  The tests below verify the class contract, SQL validity, and
factory wiring without a live database.
"""

from __future__ import annotations

import re

import pytest

from langchain_chat.core.config_models import MySQLConfig, StorageConfig
from langchain_chat.storage.base import StorageBackend
from langchain_chat.storage.factory import StorageFactory
from langchain_chat.storage.mysql_backend import _SCHEMA_SQL, MySQLBackend, _row_to_dict

# ------------------------------------------------------------------
# Structure tests
# ------------------------------------------------------------------


class TestMySQLBackendStructure:
    """Verify MySQLBackend satisfies the StorageBackend contract."""

    def test_extends_storage_backend(self) -> None:
        assert issubclass(MySQLBackend, StorageBackend)

    def test_all_abstract_methods_implemented(self) -> None:
        """Every abstract method in StorageBackend must have a concrete impl."""
        abstract = set(StorageBackend.__abstractmethods__)
        concrete = set(dir(MySQLBackend))
        missing = abstract - concrete
        assert not missing, f"MySQLBackend missing methods: {missing}"

    def test_method_count_matches_interface(self) -> None:
        """Sanity check: backend implements the expected number of methods."""
        abstract = StorageBackend.__abstractmethods__
        # 23 abstract methods in StorageBackend
        assert len(abstract) == 23, f"Expected 23 abstract methods, got {len(abstract)}"


class TestMySQLBackendConfig:
    """Tests for configuration handling."""

    def test_constructor_requires_mysql_config(self) -> None:
        """MySQLBackend raises if mysql section is missing from config."""
        cfg = StorageConfig(type="mysql")
        with pytest.raises(ValueError, match="MySQL configuration is required"):
            MySQLBackend(cfg)

    def test_constructor_accepts_valid_config(self) -> None:
        cfg = StorageConfig(
            type="mysql",
            mysql=MySQLConfig(host="127.0.0.1", port=3307, database="test_db", user="tester"),
        )
        be = MySQLBackend(cfg)
        assert be is not None


class TestRowToDict:
    """Tests for the _row_to_dict helper."""

    def test_converts_row(self) -> None:
        result = _row_to_dict((1, "alice"), ("id", "username"))
        assert result == {"id": 1, "username": "alice"}

    def test_single_column(self) -> None:
        result = _row_to_dict((42,), ("answer",))
        assert result == {"answer": 42}


# ------------------------------------------------------------------
# DDL validation
# ------------------------------------------------------------------


class TestDDL:
    """Validate the MySQL DDL SQL structure without executing it."""

    def test_all_tables_present(self) -> None:
        for table in ["users", "sessions", "messages", "presets", "configs"]:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in _SCHEMA_SQL, (
                f"Missing DDL for table: {table}"
            )

    def test_uses_mysql_syntax(self) -> None:
        """No SQLite-isms in the MySQL DDL."""
        assert "AUTOINCREMENT" not in _SCHEMA_SQL
        assert "datetime('now')" not in _SCHEMA_SQL.lower()

    def test_has_foreign_keys(self) -> None:
        assert "FOREIGN KEY" in _SCHEMA_SQL

    def test_statement_count(self) -> None:
        """DDL should contain 5 CREATE TABLE statements."""
        count = len(re.findall(r"CREATE TABLE", _SCHEMA_SQL, re.IGNORECASE))
        assert count == 5

    def test_no_sqlite_pragmas(self) -> None:
        assert "PRAGMA" not in _SCHEMA_SQL.upper()


# ------------------------------------------------------------------
# Factory tests
# ------------------------------------------------------------------


class TestFactoryMySQL:
    """Verify StorageFactory creates MySQLBackend correctly."""

    def test_factory_creates_mysql_backend(self) -> None:
        cfg = StorageConfig(
            type="mysql",
            mysql=MySQLConfig(host="127.0.0.1", database="test"),
        )
        backend = StorageFactory.create(cfg)
        assert isinstance(backend, MySQLBackend)

    def test_factory_creates_mysql_backend_case_insensitive(self) -> None:
        cfg = StorageConfig(
            type="MySQL",
            mysql=MySQLConfig(host="127.0.0.1", database="test"),
        )
        backend = StorageFactory.create(cfg)
        assert isinstance(backend, MySQLBackend)

    def test_factory_mysql_returns_storage_backend(self) -> None:
        cfg = StorageConfig(
            type="mysql",
            mysql=MySQLConfig(host="127.0.0.1", database="test"),
        )
        backend = StorageFactory.create(cfg)
        assert isinstance(backend, StorageBackend)

    def test_factory_still_creates_sqlite(self) -> None:
        """MySQL support does not break SQLite."""
        cfg = StorageConfig(type="sqlite", database=":memory:")
        backend = StorageFactory.create(cfg)
        from langchain_chat.storage.sqlite_backend import SQLiteBackend

        assert isinstance(backend, SQLiteBackend)

    def test_factory_unknown_type_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown storage type"):
            StorageFactory.create(StorageConfig(type="postgres"))


# ------------------------------------------------------------------
# Config model tests
# ------------------------------------------------------------------


class TestMySQLConfigModel:
    """Tests for the MySQLConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = MySQLConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 3306
        assert cfg.database == "langchain_chat"
        assert cfg.user == "root"
        assert cfg.password == ""
        assert cfg.pool_size == 5

    def test_custom_values(self) -> None:
        cfg = MySQLConfig(
            host="db.example.com",
            port=3307,
            database="mydb",
            user="admin",
            password="s3cret",
            pool_size=10,
        )
        assert cfg.host == "db.example.com"
        assert cfg.port == 3307
        assert cfg.database == "mydb"
        assert cfg.user == "admin"
        assert cfg.password == "s3cret"
        assert cfg.pool_size == 10
