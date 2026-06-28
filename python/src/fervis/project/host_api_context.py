"""Host API context composition from project config."""

from __future__ import annotations

from fervis.host_api.context import (
    HostApiContext,
    HostContext,
)

from .configuration import LoadedFervisConfig
from .configuration import ConfigProblem
from .discovery import ProjectInspection
from .source_scope import django_source_scopes, fastapi_sources, flask_sources
from .auth_config.loading import load_auth_project_schema


def host_api_context_from_config(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
) -> HostApiContext:
    host = loaded_config.config.host
    adapter = _host_api_adapter(project=project, loaded_config=loaded_config)
    return HostApiContext(
        adapter=adapter,
        host_context=HostContext(
            organization_name=host.organization_name,
            about_api=host.about_api,
        ),
    )


def _host_api_adapter(*, project: ProjectInspection, loaded_config: LoadedFervisConfig):
    if project.framework == "django":
        from fervis.host_api.adapters.django.adapter import (
            DjangoHostApiAdapter,
        )

        return DjangoHostApiAdapter(
            sources=django_source_scopes(loaded_config.config),
            auth_schema=_optional_auth_schema(project, loaded_config=loaded_config),
        )
    if project.framework == "fastapi":
        from fervis.host_api.adapters.fastapi.adapter import (
            FastAPIHostApiAdapter,
        )

        return FastAPIHostApiAdapter(
            sources=fastapi_sources(loaded_config.config),
            project_root=project.root_path,
            auth_schema=_optional_auth_schema(project, loaded_config=loaded_config),
        )
    if project.framework == "flask":
        from fervis.host_api.adapters.flask.adapter import FlaskHostApiAdapter

        return FlaskHostApiAdapter(
            sources=flask_sources(loaded_config.config),
            project_root=project.root_path,
            auth_schema=_optional_auth_schema(project, loaded_config=loaded_config),
        )
    raise ValueError(f"Unsupported Fervis project framework: {project.framework}")


def _optional_auth_schema(
    project: ProjectInspection,
    *,
    loaded_config: LoadedFervisConfig,
) -> dict[str, object] | None:
    loaded = load_auth_project_schema(
        project,
        active_environment=loaded_config.active_environment,
    )
    if isinstance(loaded, ConfigProblem):
        return None
    return loaded.schema
