"""Version-neutral access to effective FastAPI routes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import cast

from fastapi import FastAPI
from fastapi import routing as fastapi_routing
from fastapi.routing import APIRoute
from starlette.routing import BaseRoute


_RouteIterator = Callable[[Sequence[BaseRoute]], Iterable[APIRoute]]


def effective_api_routes(app: FastAPI) -> tuple[APIRoute, ...]:
    route_contexts = cast(
        _RouteIterator | None,
        getattr(fastapi_routing, "iter_route_contexts", None),
    )
    candidates: Iterable[APIRoute]
    if route_contexts is not None:
        candidates = route_contexts(app.routes)
    else:
        candidates = cast(Iterable[APIRoute], app.routes)
    return tuple(
        candidate
        for candidate in candidates
        if _is_schema_declared_api_route(candidate)
    )


def _is_schema_declared_api_route(candidate: APIRoute) -> bool:
    original = getattr(candidate, "original_route", candidate)
    return (
        isinstance(original, APIRoute)
        and bool(candidate.include_in_schema)
        and bool(original.include_in_schema)
    )
