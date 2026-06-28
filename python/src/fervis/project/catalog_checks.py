"""Doctor checks for source declarations that produce endpoint catalogs."""

from __future__ import annotations

from fervis.interfaces.agent.actions import edit_config_action

from .configuration import LoadedFervisConfig
from .django_runtime import django_project_runtime
from .discovery import ProjectInspection
from .doctor import DoctorCheck
from .integration import FastAPIAppSource
from .catalog_errors import catalog_failure_action
from .source_scope import django_source_scopes, flask_sources


def catalog_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> list[DoctorCheck]:
    if project.framework == "django":
        return _django_catalog_checks(project, loaded)
    if project.framework == "fastapi":
        return _fastapi_catalog_checks(project, loaded)
    if project.framework == "flask":
        return _flask_catalog_checks(project, loaded)
    return []


def _django_catalog_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> list[DoctorCheck]:
    try:
        with django_project_runtime(project):
            from fervis.host_api.adapters.django.catalog import (
                get_endpoint_contracts,
            )

            contracts = get_endpoint_contracts(
                sources=django_source_scopes(loaded.config),
            )
    except Exception as exc:
        return [
            DoctorCheck(
                id="source.catalog",
                status="failed",
                message=f"Django source catalog could not be built: {exc}",
                fix=catalog_failure_action(exc, loaded=loaded),
            )
        ]
    return _catalog_contract_checks("Django", contracts)


def _fastapi_catalog_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> list[DoctorCheck]:
    from fervis.host_api.adapters.fastapi.catalog import (
        get_fastapi_endpoint_contracts,
    )

    sources = tuple(
        source
        for source in loaded.config.sources
        if isinstance(source, FastAPIAppSource)
    )
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=sources,
            project_root=project.root_path,
        )
    except Exception as exc:
        return [
            DoctorCheck(
                id="source.catalog",
                status="failed",
                message=f"FastAPI source catalog could not be built: {exc}",
                fix=catalog_failure_action(exc, loaded=loaded),
            )
        ]
    return _catalog_contract_checks("FastAPI", contracts)


def _flask_catalog_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> list[DoctorCheck]:
    from fervis.host_api.adapters.flask.catalog import (
        get_flask_endpoint_contracts,
    )

    try:
        contracts = get_flask_endpoint_contracts(
            sources=flask_sources(loaded.config),
            project_root=project.root_path,
        )
    except Exception as exc:
        return [
            DoctorCheck(
                id="source.catalog",
                status="failed",
                message=f"Flask source catalog could not be built: {exc}",
                fix=catalog_failure_action(exc, loaded=loaded),
            )
        ]
    return _catalog_contract_checks(
        "Flask",
        contracts,
        require_all_exposed_response_fields=True,
    )


def _catalog_contract_checks(
    framework_label: str,
    contracts: tuple[object, ...],
    *,
    require_all_exposed_response_fields: bool = False,
) -> list[DoctorCheck]:
    read_contracts = tuple(
        contract
        for contract in contracts
        if getattr(contract, "supports_lookup_read", lambda: False)()
    )
    if not read_contracts:
        return [
            DoctorCheck(
                id="source.catalog",
                status="failed",
                message=(
                    f"{framework_label} sources produced no lookup-readable GET "
                    "endpoints with response fields."
                ),
                fix=edit_config_action(),
            )
        ]
    missing_response_schema_candidates = (
        contracts if require_all_exposed_response_fields else read_contracts
    )
    missing_response_schema = [
        contract
        for contract in missing_response_schema_candidates
        if not getattr(contract, "response_fields", ())
    ]
    checks = [
        DoctorCheck(
            id="source.catalog",
            status="passed",
            message=(
                f"Found {len(read_contracts)} lookup-readable GET endpoint"
                f"{'' if len(read_contracts) == 1 else 's'}."
            ),
        )
    ]
    if missing_response_schema:
        count = len(missing_response_schema)
        checks.append(
            DoctorCheck(
                id="source.response_schema",
                status="failed",
                message=(
                    f"{count} exposed endpoint"
                    f"{'' if count == 1 else 's'} "
                    f"{'has' if count == 1 else 'have'} no response fields."
                ),
                fix=edit_config_action(),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                id="source.response_schema",
                status="passed",
                message="Exposed endpoints provide response fields.",
            )
        )
    return checks
