from __future__ import annotations

from fervis import configured_fervis
from fervis.interfaces.fastapi import fervis_fastapi_router
from fervis.project import FastAPIIntegration

__all__ = [
    "FastAPIIntegration",
    "configured_fervis",
    "fervis_fastapi_router",
]
