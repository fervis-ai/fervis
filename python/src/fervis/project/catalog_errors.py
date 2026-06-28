"""Agent actions for host source catalog failures."""

from __future__ import annotations

from fervis.interfaces.agent.actions import (
    edit_config_action,
    install_dependencies_action,
    run_migrate_action,
)
from fervis.storage.sql.engine import FervisPersistenceNotReady

from .configuration import LoadedFervisConfig
from .integration import DjangoAppSource, FastAPIAppSource, FlaskAppSource


def catalog_failure_action(
    exc: Exception,
    *,
    loaded: LoadedFervisConfig,
) -> dict[str, object]:
    if isinstance(exc, FervisPersistenceNotReady):
        return run_migrate_action()
    module_name = _import_error_module(exc)
    if module_name and _module_root(module_name) not in _configured_source_roots(
        loaded
    ):
        return install_dependencies_action(module_name)
    return edit_config_action()


def _import_error_module(exc: Exception) -> str:
    if not isinstance(exc, ImportError):
        return ""
    name = getattr(exc, "name", None)
    return str(name or "")


def _module_root(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _configured_source_roots(loaded: LoadedFervisConfig) -> set[str]:
    roots: set[str] = set()
    for source in loaded.config.sources:
        if isinstance(source, DjangoAppSource):
            roots.update(module.split(".", 1)[0] for module in source.app_modules)
        elif isinstance(source, FastAPIAppSource):
            roots.update(
                import_path.split(":", 1)[0].split(".", 1)[0]
                for import_path in source.import_paths
            )
        elif isinstance(source, FlaskAppSource):
            roots.add(source.app.split(":", 1)[0].split(".", 1)[0])
    return roots
