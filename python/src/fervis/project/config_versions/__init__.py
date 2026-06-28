"""Versioned Fervis JSON config schema helpers."""

from .auth import AUTH_CONFIG_SCHEMA_VERSION, normalize_auth_schema
from .main import PROJECT_CONFIG_SCHEMA_VERSION, normalize_project_schema

__all__ = [
    "AUTH_CONFIG_SCHEMA_VERSION",
    "PROJECT_CONFIG_SCHEMA_VERSION",
    "normalize_auth_schema",
    "normalize_project_schema",
]
