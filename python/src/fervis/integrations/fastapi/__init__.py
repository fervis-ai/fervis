"""Public FastAPI integration surface."""

from fervis.interfaces.fastapi import fervis_fastapi_router

from .integration import FastAPIIntegration

__all__ = [
    "FastAPIIntegration",
    "fervis_fastapi_router",
]
