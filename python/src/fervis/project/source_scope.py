"""Source exposure scopes derived from the public Fervis config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .configuration import ConfigProblem, load_fervis_project_config
from .discovery import ProjectInspection
from .integration import DjangoAppSource, FastAPIAppSource, FlaskAppSource, FervisConfig


@dataclass(frozen=True)
class DjangoSourceScope:
    name: str
    app_modules: tuple[str, ...]
    path_prefixes: tuple[str, ...]


def django_source_scopes(config: FervisConfig) -> tuple[DjangoSourceScope, ...]:
    return tuple(
        DjangoSourceScope(
            name=source.name,
            app_modules=tuple(source.app_modules),
            path_prefixes=tuple(source.path_prefixes),
        )
        for source in config.sources
        if isinstance(source, DjangoAppSource)
    )


def fastapi_sources(config: FervisConfig) -> tuple[FastAPIAppSource, ...]:
    return tuple(
        source for source in config.sources if isinstance(source, FastAPIAppSource)
    )


def flask_sources(config: FervisConfig) -> tuple[FlaskAppSource, ...]:
    return tuple(
        source for source in config.sources if isinstance(source, FlaskAppSource)
    )


def configured_fervis_config() -> FervisConfig:
    from django.conf import settings

    root_path = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    config_path = Path(getattr(settings, "FERVIS_CONFIG_PATH", "config/fervis.json"))
    loaded = load_fervis_project_config(
        ProjectInspection(
            framework="django",
            root_path=root_path,
            config_path=config_path,
            expected_config_path=config_path,
            confidence="high",
        )
    )
    if isinstance(loaded, ConfigProblem):
        raise RuntimeError(loaded.message)
    return loaded.config


def configured_django_source_scopes() -> tuple[DjangoSourceScope, ...]:
    return django_source_scopes(configured_fervis_config())
