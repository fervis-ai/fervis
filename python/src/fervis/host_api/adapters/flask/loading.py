"""Load configured Flask app targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fervis.project.importing import import_object, project_import_context
from fervis.host_api.adapters.runtime_output import suppress_host_output


def import_flask_app(
    app_target: str,
    *,
    project_root: Path,
    app_args: tuple[object, ...] = (),
    app_kwargs: dict[str, object] | None = None,
) -> Any:
    if ":" not in app_target:
        raise ValueError("Flask app target must use module:object syntax.")
    with project_import_context(project_root), suppress_host_output():
        target = import_object(app_target)
        app = _resolve_flask_app_target(
            target,
            app_args=app_args,
            app_kwargs=dict(app_kwargs or {}),
        )
    _assert_flask_app(app, app_target=app_target)
    return app


def _resolve_flask_app_target(
    target: Any,
    *,
    app_args: tuple[object, ...],
    app_kwargs: dict[str, object],
) -> Any:
    app = _flask_app_from_target(target)
    if app is not None:
        return app
    if callable(target):
        return _flask_app_from_target(target(*app_args, **app_kwargs))
    return target


def _flask_app_from_target(target: Any) -> Any | None:
    if _is_flask_app(target):
        return target
    wrapped_app = getattr(target, "app", None)
    if _is_flask_app(wrapped_app):
        return wrapped_app
    return None


def _is_flask_app(value: Any) -> bool:
    try:
        from flask import Flask
    except ImportError:
        return False
    return isinstance(value, Flask)


def _assert_flask_app(app: Any, *, app_target: str) -> None:
    try:
        from flask import Flask
    except ImportError as exc:
        raise RuntimeError("Flask app loading requires flask to be installed.") from exc
    if not isinstance(app, Flask):
        raise TypeError(f"{app_target} did not resolve to a flask.Flask app.")
