"""FastAPI app loading from configured project import paths."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
import warnings

from fervis.project.importing import project_import_context
from fervis.host_api.adapters.runtime_output import suppress_host_output


def import_fastapi_app(import_path: str, *, project_root: Path) -> object:
    with project_import_context(project_root), suppress_host_output():
        return load_fastapi_app(import_path)


def load_fastapi_app(import_path: str) -> object:
    if ":" not in import_path:
        raise ValueError(f"FastAPI import path must use module:object: {import_path}")
    module_name, object_name = import_path.split(":", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
        value: object = module
        for part in object_name.split("."):
            value = getattr(value, part)
        return _resolve_fastapi_app(value, import_path=import_path)


def _resolve_fastapi_app(value: object, *, import_path: str) -> object:
    if callable(getattr(value, "openapi", None)):
        return value
    if not callable(value):
        return value
    app = value()
    if not callable(getattr(app, "openapi", None)):
        raise ValueError(f"{import_path} did not return a FastAPI app.")
    return app


def fastapi_openapi_schema(app: object, *, import_path: str) -> dict[str, Any]:
    openapi = getattr(app, "openapi", None)
    if not callable(openapi):
        raise ValueError(f"{import_path} does not expose callable openapi().")
    with suppress_host_output():
        schema = openapi()
    if not isinstance(schema, dict):
        raise ValueError(f"{import_path}.openapi() must return a dict.")
    return schema
