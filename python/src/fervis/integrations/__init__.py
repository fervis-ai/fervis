"""Framework integration contracts and registration."""

from .registry import (
    FrameworkIntegration,
    FrameworkIntegrationSpec,
    create_framework_integration,
    framework_integration_spec,
    register_framework_integration,
)

__all__ = [
    "FrameworkIntegration",
    "FrameworkIntegrationSpec",
    "create_framework_integration",
    "framework_integration_spec",
    "register_framework_integration",
]
