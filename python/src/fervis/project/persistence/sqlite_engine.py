"""SQLite engine construction for Fervis-owned persistence."""

from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine


def create_sqlite_engine(url: str) -> Engine:
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    return engine
