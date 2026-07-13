"""Doctor checks for source declarations that produce endpoint catalogs."""

from __future__ import annotations

from fervis.host_api.contracts.response_conformance import ResponseConformanceResult
from fervis.interfaces.agent.actions import (
    configure_auth_action,
    edit_config_action,
    fix_schema_cardinality_action,
)

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
    from fervis.host_api.adapters.flask.response_conformance import (
        check_flask_response_conformance,
    )

    sources = flask_sources(loaded.config)
    try:
        contracts = get_flask_endpoint_contracts(
            sources=sources,
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
    checks = _catalog_contract_checks(
        "Flask",
        contracts,
        require_all_exposed_response_fields=True,
    )
    if any(check.status == "failed" for check in checks):
        return checks
    conformance_results = check_flask_response_conformance(
        sources=sources,
        project_root=project.root_path,
        contracts=contracts,
    )
    checks.append(_response_conformance_check(conformance_results))
    return checks


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


def _response_conformance_check(
    results: tuple[ResponseConformanceResult, ...],
) -> DoctorCheck:
    failures = tuple(result for result in results if result.status == "failed")
    if failures:
        first = failures[0]
        suffix = "" if len(failures) == 1 else f" {len(failures) - 1} more failed."
        return DoctorCheck(
            id="source.response_conformance",
            status="failed",
            message=f"{first.message}{suffix}",
            fix=fix_schema_cardinality_action(first.endpoint_name),
        )
    auth_skips = tuple(result for result in results if result.reason == "auth_required")
    if auth_skips:
        first = auth_skips[0]
        suffix = (
            ""
            if len(auth_skips) == 1
            else f" {len(auth_skips) - 1} more auth-protected."
        )
        return DoctorCheck(
            id="source.response_conformance",
            status="skipped",
            message=f"{first.message}{suffix}",
            fix=configure_auth_action(framework="flask"),
        )
    verified_count = sum(1 for result in results if result.status == "passed")
    skipped = tuple(result for result in results if result.status == "skipped")
    if skipped:
        return DoctorCheck(
            id="source.response_conformance",
            status="passed" if verified_count else "skipped",
            message=_response_conformance_skipped_message(
                verified_count=verified_count,
                skipped_count=len(skipped),
            ),
        )
    return DoctorCheck(
        id="source.response_conformance",
        status="passed",
        message="Probeable response shapes match declared schemas.",
    )


def _response_conformance_skipped_message(
    *,
    verified_count: int,
    skipped_count: int,
) -> str:
    skipped_label = f"{skipped_count} endpoint{'' if skipped_count == 1 else 's'}"
    if verified_count:
        return (
            f"Verified {verified_count} response shape"
            f"{'' if verified_count == 1 else 's'}; {skipped_label} skipped "
            "because they require params or host auth."
        )
    return (
        f"No response shape probes could run; {skipped_label} skipped because "
        "they require params or host auth."
    )
