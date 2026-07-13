"""FastAPI app loading from configured project import paths."""

from __future__ import annotations

import importlib
from pathlib import Path
import warnings

from fastapi import FastAPI

from fervis.project.importing import project_import_context
from fervis.host_api.adapters.runtime_output import suppress_host_output


def import_fastapi_app(import_path: str, *, project_root: Path) -> FastAPI:
    with project_import_context(project_root), suppress_host_output():
        return load_fastapi_app(import_path)


def load_fastapi_app(import_path: str) -> FastAPI:
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


def _resolve_fastapi_app(value: object, *, import_path: str) -> FastAPI:
    if isinstance(value, FastAPI):
        return value
    if not callable(value):
        raise ValueError(f"{import_path} is not a FastAPI app or app factory.")
    app = value()
    if not isinstance(app, FastAPI):
        raise ValueError(f"{import_path} did not return a FastAPI app.")
    return app
