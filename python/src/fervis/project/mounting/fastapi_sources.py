"""FastAPI source-prefix defaults derived from the runtime app."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from fervis.host_api.adapters.fastapi.loading import import_fastapi_app
from fervis.host_api.adapters.fastapi.routes import effective_api_routes

from .common import BlockedPatch


def fastapi_source_path_prefixes(
    root: Path,
    import_path: str,
) -> tuple[str, ...] | BlockedPatch:
    try:
        app = import_fastapi_app(import_path, project_root=root)
    except Exception as exc:
        return BlockedPatch(
            "config/fervis.json",
            (
                "Could not load the configured FastAPI app to derive source "
                "prefixes. Retry with explicit `--path-prefixes`. "
                f"Underlying error: {exc}"
            ),
        )

    prefixes = {
        prefix
        for path in _get_paths(app)
        if (prefix := _source_prefix_from_path(path)) != "/"
    }
    if not prefixes:
        return BlockedPatch(
            "config/fervis.json",
            (
                "Could not identify FastAPI GET source prefixes from the "
                "runtime route registry. Retry with explicit `--path-prefixes`."
            ),
        )
    return tuple(sorted(prefixes))


def _get_paths(app: FastAPI) -> tuple[str, ...]:
    return tuple(
        route.path
        for route in effective_api_routes(app)
        if "GET" in (route.methods or set())
    )


def _source_prefix_from_path(path: str) -> str:
    text = str(path).strip()
    if not text.startswith("/"):
        text = f"/{text}"
    parts: list[str] = []
    for part in text.strip("/").split("/"):
        if not part:
            continue
        if "{" in part:
            break
        parts.append(part)
    if not parts:
        return "/"
    return "/" + "/".join(parts) + "/"
