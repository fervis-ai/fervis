"""Load and validate the host project's Fervis integration object."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fervis.interfaces.common.questions import QuestionInterface

from .discovery import ProjectInspection
from .config_io import ActiveEnvironment, ConfigIOError, load_project_json_config
from .config_schema import config_from_schema
from .integration import (
    DatabaseUrlPersistence,
    DjangoDatabasePersistence,
    DjangoAppSource,
    DjangoIntegration,
    FastAPIAppSource,
    FastAPIIntegration,
    FlaskAppSource,
    FlaskIntegration,
    FervisConfig,
    HostConfig,
    ModelConfig,
    PersistenceTarget,
    ProviderConfig,
    RuntimeRoutes,
    SQLitePersistence,
)
from fervis.model_io.models import ModelRef
from fervis.model_io.providers.specs import supported_provider_specs


@dataclass(frozen=True)
class ConfigProblem:
    code: str
    message: str


@dataclass(frozen=True)
class LoadedFervisConfig:
    integration: object
    config: FervisConfig
    schema: dict[str, object]
    config_path: Path
    active_environment: ActiveEnvironment


@dataclass(frozen=True)
class LoadedFervisSchema:
    schema: dict[str, object]
    config_path: Path
    active_environment: ActiveEnvironment


@dataclass(frozen=True)
class _FastAPIPrincipalDependency:
    factory: Callable[[], Callable[..., object]]
    id_attr: str


def load_fervis_project_schema(
    project: ProjectInspection,
    *,
    explicit_env: str | None = None,
) -> LoadedFervisSchema | ConfigProblem:
    if project.config_path is None:
        return ConfigProblem(
            code="config_missing",
            message="Fervis config was not found at config/fervis.json.",
        )

    try:
        loaded = load_project_json_config(
            project.root_path,
            config_path=project.config_path,
            explicit_env=explicit_env,
        )
    except ConfigIOError as exc:
        return ConfigProblem(
            code="config_schema_invalid",
            message=str(exc) or exc.__class__.__name__,
        )
    try:
        config_from_schema(loaded.active_schema)
    except ValueError as exc:
        return ConfigProblem(
            code="config_schema_invalid",
            message=str(exc),
        )
    return LoadedFervisSchema(
        schema=loaded.active_schema,
        config_path=loaded.config_path,
        active_environment=loaded.active_environment,
    )


def load_fervis_project_config(
    project: ProjectInspection,
    *,
    explicit_env: str | None = None,
) -> LoadedFervisConfig | ConfigProblem:
    loaded_schema = load_fervis_project_schema(project, explicit_env=explicit_env)
    if isinstance(loaded_schema, ConfigProblem):
        return loaded_schema
    schema = loaded_schema.schema
    config = config_from_schema(schema)
    integration = _integration_from_schema(
        schema,
        config,
        project=project,
        config_path=loaded_schema.config_path,
        active_environment=loaded_schema.active_environment,
    )

    problem = validate_fervis_integration(integration, framework=project.framework)
    if problem is not None:
        return problem

    return LoadedFervisConfig(
        integration=integration,
        config=config,
        schema=schema,
        config_path=loaded_schema.config_path,
        active_environment=loaded_schema.active_environment,
    )


def _integration_from_schema(
    schema: dict[str, object],
    config: FervisConfig,
    *,
    project: ProjectInspection,
    config_path: Path,
    active_environment: ActiveEnvironment,
) -> DjangoIntegration | FastAPIIntegration | FlaskIntegration:
    framework = str(schema.get("framework") or "")
    if framework == "django":
        return DjangoIntegration(config=config)
    if framework == "flask":
        return _flask_integration_from_schema(
            schema,
            config,
            project=project,
            config_path=config_path,
            active_environment=active_environment,
        )
    return _fastapi_integration_from_schema(
        schema,
        config,
        project=project,
        config_path=config_path,
        active_environment=active_environment,
    )


def _flask_integration_from_schema(
    schema: dict[str, object],
    config: FervisConfig,
    *,
    project: ProjectInspection,
    config_path: Path,
    active_environment: ActiveEnvironment,
) -> FlaskIntegration:
    integration: FlaskIntegration

    def question_interface_factory() -> QuestionInterface:
        loaded = LoadedFervisConfig(
            integration=integration,
            config=config,
            schema=schema,
            config_path=config_path,
            active_environment=active_environment,
        )
        return _flask_question_interface(project=project, loaded_config=loaded)

    host_api_adapter = _configured_host_api_adapter(
        project=project,
        config=config,
        schema=schema,
        config_path=config_path,
        active_environment=active_environment,
    )
    integration = FlaskIntegration(
        config=config,
        question_interface_factory=question_interface_factory,
        read_context_capture=host_api_adapter.capture_read_context,
        delegated_credential_capture=host_api_adapter.capture_delegated_credential,
        require_read_context=_has_auth_schema(
            project,
            active_environment=active_environment,
        ),
    )
    return integration


def _fastapi_integration_from_schema(
    schema: dict[str, object],
    config: FervisConfig,
    *,
    project: ProjectInspection,
    config_path: Path,
    active_environment: ActiveEnvironment,
) -> FastAPIIntegration:
    integration: FastAPIIntegration

    def question_interface_factory() -> QuestionInterface:
        loaded = LoadedFervisConfig(
            integration=integration,
            config=config,
            schema=schema,
            config_path=config_path,
            active_environment=active_environment,
        )
        return _fastapi_question_interface(project=project, loaded_config=loaded)

    host_api_adapter = _configured_host_api_adapter(
        project=project,
        config=config,
        schema=schema,
        config_path=config_path,
        active_environment=active_environment,
    )
    principal = _fastapi_principal_dependency(
        project,
        active_environment=active_environment,
    )
    integration = FastAPIIntegration(
        config=config,
        question_interface_factory=question_interface_factory,
        read_context_capture=host_api_adapter.capture_read_context,
        delegated_credential_capture=host_api_adapter.capture_delegated_credential,
        principal_dependency_factory=(principal.factory if principal else None),
        principal_id_attr=principal.id_attr if principal else "id",
        require_read_context=principal is not None,
    )
    return integration


def _fastapi_principal_dependency(
    project: ProjectInspection,
    *,
    active_environment: ActiveEnvironment,
) -> _FastAPIPrincipalDependency | None:
    from fervis.project.auth_config.loading import load_auth_project_schema
    from fervis.project.importing import import_object, project_import_context

    loaded_auth = load_auth_project_schema(
        project,
        active_environment=active_environment,
    )
    if isinstance(loaded_auth, ConfigProblem):
        return None
    principal = loaded_auth.schema.get("principal")
    if not isinstance(principal, dict):
        return None
    if principal.get("source") != "fastapi_dependency":
        return None
    dependency_path = str(principal.get("dependency") or "")

    def dependency_factory() -> Callable[..., object]:
        with project_import_context(project.root_path):
            dependency = import_object(dependency_path)
        if not callable(dependency):
            raise TypeError("FastAPI principal dependency must be callable")
        return dependency

    return _FastAPIPrincipalDependency(
        factory=dependency_factory,
        id_attr=str(principal.get("id_attr") or "id"),
    )


def _has_auth_schema(
    project: ProjectInspection,
    *,
    active_environment: ActiveEnvironment,
) -> bool:
    from fervis.project.auth_config.loading import load_auth_project_schema

    loaded_auth = load_auth_project_schema(
        project,
        active_environment=active_environment,
    )
    return not isinstance(loaded_auth, ConfigProblem)


def _configured_host_api_adapter(
    *,
    project: ProjectInspection,
    config: FervisConfig,
    schema: dict[str, object],
    config_path: Path,
    active_environment: ActiveEnvironment,
):
    from fervis.project.host_api_context import host_api_context_from_config

    context = host_api_context_from_config(
        project=project,
        loaded_config=LoadedFervisConfig(
            integration=None,
            config=config,
            schema=schema,
            config_path=config_path,
            active_environment=active_environment,
        ),
    )
    return context.adapter


def _fastapi_question_interface(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
) -> QuestionInterface:
    from fervis.storage.sql.question_interface import sql_question_interface

    return sql_question_interface(project=project, loaded_config=loaded_config)


def _flask_question_interface(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
) -> QuestionInterface:
    from fervis.storage.sql.question_interface import sql_question_interface

    return sql_question_interface(project=project, loaded_config=loaded_config)


def validate_fervis_integration(
    integration: object,
    *,
    framework: str,
) -> ConfigProblem | None:
    config = getattr(integration, "config", None)
    if not isinstance(config, FervisConfig):
        return ConfigProblem(
            code="config_type_invalid",
            message="fervis.config must be a FervisConfig instance.",
        )
    integration_framework = str(getattr(integration, "framework", "") or "")
    if integration_framework != framework:
        return ConfigProblem(
            code="config_framework_mismatch",
            message=(
                f"config framework {integration_framework or '<missing>'} does not "
                f"match detected project framework {framework}."
            ),
        )
    return _validate_config(config, framework=framework)


def _validate_config(config: FervisConfig, *, framework: str) -> ConfigProblem | None:
    type_problem = _validate_config_types(config)
    if type_problem is not None:
        return type_problem
    if not _valid_route_prefix(config.routes.prefix):
        return ConfigProblem(
            code="routes_prefix_invalid",
            message="routes.prefix must start with '/'.",
        )
    model_problem = _validate_model(config)
    if model_problem is not None:
        return model_problem
    return _validate_sources(config.sources, framework=framework)


def _validate_config_types(config: FervisConfig) -> ConfigProblem | None:
    expected = (
        ("host", HostConfig),
        ("routes", RuntimeRoutes),
        ("model", ModelConfig),
    )
    for field_name, field_type in expected:
        value = getattr(config, field_name, None)
        if not isinstance(value, field_type):
            type_name = (
                field_type.__name__
                if hasattr(field_type, "__name__")
                else str(field_type)
            )
            return ConfigProblem(
                code=f"{field_name}_type_invalid",
                message=f"{field_name} must be a {type_name} instance.",
            )
    if not isinstance(config.sources, list):
        return ConfigProblem(
            code="sources_type_invalid",
            message="sources must be a list.",
        )
    persistence_types = (
        SQLitePersistence,
        DjangoDatabasePersistence,
        DatabaseUrlPersistence,
    )
    if not isinstance(config.persistence, persistence_types):
        return ConfigProblem(
            code="persistence_type_invalid",
            message=(
                "persistence must be a SQLitePersistence, "
                "DjangoDatabasePersistence, or DatabaseUrlPersistence instance."
            ),
        )
    persistence_problem = _validate_persistence(config.persistence)
    if persistence_problem is not None:
        return persistence_problem
    return None


def _validate_persistence(persistence: PersistenceTarget) -> ConfigProblem | None:
    if isinstance(persistence, SQLitePersistence):
        if not persistence.path.strip():
            return ConfigProblem(
                code="persistence_sqlite_path_missing",
                message="SQLitePersistence.path must not be empty.",
            )
        return None
    if isinstance(persistence, DjangoDatabasePersistence):
        if not persistence.database.strip():
            return ConfigProblem(
                code="persistence_django_database_missing",
                message="DjangoDatabasePersistence.database must not be empty.",
            )
        return None
    if isinstance(persistence, DatabaseUrlPersistence):
        if not persistence.url_env.strip():
            return ConfigProblem(
                code="persistence_database_url_env_missing",
                message="DatabaseUrlPersistence.url_env must not be empty.",
            )
    return None


def _validate_model(config: FervisConfig) -> ConfigProblem | None:
    default_provider = config.model.default_provider.strip()
    default_model_key = config.model.default_model_key.strip()
    default_ref = ModelRef(provider=default_provider, model_id=default_model_key)
    provider_specs = supported_provider_specs()
    if default_ref.provider not in provider_specs:
        supported = ", ".join(sorted(provider_specs))
        return ConfigProblem(
            code="model_provider_unsupported",
            message=(
                f"model.default provider {default_ref.provider!r} is not "
                f"supported. Supported providers: {supported}."
            ),
        )
    if not isinstance(config.model.providers, list):
        return ConfigProblem(
            code="model_providers_type_invalid",
            message="model.providers must be a list.",
        )
    for provider in config.model.providers:
        if not isinstance(provider, ProviderConfig):
            return ConfigProblem(
                code="model_provider_invalid",
                message="model.providers must contain ProviderConfig instances.",
            )
        if not provider.name.strip():
            return ConfigProblem(
                code="model_provider_name_missing",
                message="providers must declare a non-empty name.",
            )
        if provider.name not in provider_specs:
            supported = ", ".join(sorted(provider_specs))
            return ConfigProblem(
                code="model_provider_unsupported",
                message=(
                    f"model provider {provider.name!r} is not supported. "
                    f"Supported providers: {supported}."
                ),
            )
        if not provider.allowed_model_keys:
            return ConfigProblem(
                code="model_provider_models_missing",
                message=f"provider {provider.name!r} must allow at least one model.",
            )
    provider_names = {provider.name for provider in config.model.providers}
    if default_ref.provider not in provider_names:
        return ConfigProblem(
            code="model_provider_missing",
            message=(
                f"model.default provider {default_ref.provider!r} is not declared "
                "in model.providers."
            ),
        )
    default_provider_config = next(
        provider
        for provider in config.model.providers
        if provider.name == default_ref.provider
    )
    if default_ref.model_id not in default_provider_config.allowed_model_keys:
        return ConfigProblem(
            code="model_default_not_allowed",
            message=(
                f"model.default model_key {default_ref.model_id!r} is not listed "
                f"in provider {default_ref.provider!r} allowed_model_keys."
            ),
        )
    return None


def _validate_sources(
    sources: list[DjangoAppSource | FastAPIAppSource | FlaskAppSource],
    *,
    framework: str,
) -> ConfigProblem | None:
    if not sources:
        return ConfigProblem(
            code="sources_missing",
            message="sources must include at least one explicit source declaration.",
        )
    expected_type: type[DjangoAppSource] | type[FastAPIAppSource] | type[FlaskAppSource]
    if framework == "django":
        expected_type = DjangoAppSource
    elif framework == "fastapi":
        expected_type = FastAPIAppSource
    else:
        expected_type = FlaskAppSource
    for source in sources:
        if not isinstance(source, expected_type):
            return ConfigProblem(
                code="source_framework_mismatch",
                message=f"source {source!r} does not match framework {framework}.",
            )
        if not source.name.strip():
            return ConfigProblem(
                code="source_name_missing",
                message="source.name must not be empty.",
            )
        if not _non_empty_string_list(source.path_prefixes):
            return ConfigProblem(
                code="source_path_prefixes_missing",
                message=(
                    f"source {source.name!r} must declare non-empty path_prefixes."
                ),
            )
        if isinstance(source, DjangoAppSource) and not _non_empty_string_list(
            source.app_modules
        ):
            return ConfigProblem(
                code="source_app_modules_missing",
                message=f"source {source.name!r} must declare non-empty app_modules.",
            )
        if isinstance(source, FastAPIAppSource) and not _non_empty_string_list(
            source.import_paths
        ):
            return ConfigProblem(
                code="source_import_paths_missing",
                message=f"source {source.name!r} must declare non-empty import_paths.",
            )
        if isinstance(source, FlaskAppSource) and not source.app.strip():
            return ConfigProblem(
                code="source_flask_app_missing",
                message=f"source {source.name!r} must declare a Flask app target.",
            )
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        return ConfigProblem(
            code="source_name_duplicate",
            message="source names must be unique.",
        )
    return None


def _non_empty_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _valid_route_prefix(value: str) -> bool:
    return value.startswith("/") and bool(value.strip("/"))
