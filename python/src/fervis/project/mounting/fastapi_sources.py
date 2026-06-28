"""FastAPI source-prefix defaults derived from the runtime app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fervis.host_api.adapters.fastapi.loading import (
    fastapi_openapi_schema,
    import_fastapi_app,
)

from .common import BlockedPatch


def fastapi_source_path_prefixes(
    root: Path,
    import_path: str,
) -> tuple[str, ...] | BlockedPatch:
    try:
        app = import_fastapi_app(import_path, project_root=root)
        schema = fastapi_openapi_schema(app, import_path=import_path)
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
        for path in _get_paths(schema)
        if (prefix := _source_prefix_from_path(path)) != "/"
    }
    if not prefixes:
        return BlockedPatch(
            "config/fervis.json",
            (
                "Could not identify FastAPI GET source prefixes from the "
                "runtime OpenAPI schema. Retry with explicit `--path-prefixes`."
            ),
        )
    return tuple(sorted(prefixes))


def _get_paths(schema: dict[str, Any]) -> tuple[str, ...]:
    paths = schema.get("paths")
    if not isinstance(paths, dict):
        return ()
    return tuple(
        str(path)
        for path, methods in paths.items()
        if isinstance(methods, dict) and "get" in methods
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
