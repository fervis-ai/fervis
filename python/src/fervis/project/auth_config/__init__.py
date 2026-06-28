"""Schema-backed host auth configuration."""

from .commands import AuthConfigureResult, configure_auth
from fervis.project.config_versions.auth import (
    AUTH_CONFIG_SCHEMA_VERSION,
    validate_auth_schema,
)

__all__ = [
    "AUTH_CONFIG_SCHEMA_VERSION",
    "AuthConfigureResult",
    "configure_auth",
    "validate_auth_schema",
]
