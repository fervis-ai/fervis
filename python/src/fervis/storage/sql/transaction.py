"""Shared SQL transaction context for storage adapters."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy.engine import Connection, Engine

_CONNECTION: ContextVar[Connection | None] = ContextVar(
    "fervis_sql_storage_connection",
    default=None,
)


@contextmanager
def sql_transaction(engine: Engine):
    active = _CONNECTION.get()
    if active is not None:
        yield active
        return
    with engine.begin() as connection:
        token = _CONNECTION.set(connection)
        try:
            yield connection
        finally:
            _CONNECTION.reset(token)


@contextmanager
def rollback_sql_transaction(engine: Engine):
    active = _CONNECTION.get()
    if active is not None:
        yield active
        return
    with engine.connect() as connection:
        transaction = connection.begin()
        token = _CONNECTION.set(connection)
        try:
            yield connection
        finally:
            _CONNECTION.reset(token)
            transaction.rollback()


@contextmanager
def sql_connection(engine: Engine):
    active = _CONNECTION.get()
    if active is not None:
        yield active
        return
    with engine.begin() as connection:
        yield connection
