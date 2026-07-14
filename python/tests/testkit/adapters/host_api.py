from __future__ import annotations

from typing import Any

from fervis.host_api.contracts import (
    CandidateKeyComponentContract,
    CandidateKeyContract,
    EndpointContract,
    EntityKeyComponentTargetContract,
    EntityReferenceComponentContract,
    EntityReferenceContract,
    PaginationContract,
    PaginationKind,
    ParameterContract,
    ResponseFieldContract,
)
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.lookup.relation_catalog import CatalogValidationError, parse_relation_catalog
from fervis.lookup.relation_catalog.model import PaginationMetadata

from tests.testkit.assertions import (
    expects_rejection,
    rejection_mismatches,
    status_mismatches,
    subset_mismatches,
)


def run_host_api_projection_case(payload: dict[str, Any]) -> list[str]:
    try:
        catalog = relation_catalog_from_endpoint_contracts(
            tuple(_endpoint_contract(item) for item in payload["input"]["contracts"])
        )
        parse_relation_catalog(catalog)
    except CatalogValidationError as exc:
        if expects_rejection(payload["expect"]):
            return rejection_mismatches(
                actual_code="invalid_relation_catalog",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    result = {
        "reads": {
            read.id: {
                "endpoint_name": read.endpoint_name,
                "resource_names": list(read.resource_names),
                "params": {
                    param.name: {
                        "choices": list(param.choices),
                        "choice_labels": dict(param.choice_labels or {}),
                        "description": param.description,
                    }
                    for param in read.params
                },
                "response_envelope": {
                    "results_path": read.response_envelope.results_path,
                    "count_path": read.response_envelope.count_path,
                },
                "pagination": _projected_pagination(
                    read.pagination or PaginationMetadata()
                ),
                "row_paths": [
                    {
                        "id": item.id,
                        "path": item.path,
                        "cardinality": item.cardinality.value,
                    }
                    for item in read.row_paths
                ],
                "candidate_keys": [
                    {
                        "key_id": key.id,
                        "entity_kind": key.entity_kind,
                        "components": [
                            {
                                "component_id": component.id,
                                "field_ref": component.field_ref,
                            }
                            for component in key.components
                        ],
                        "primary": key.primary,
                        "stable": key.stable,
                        "context_field_refs": list(key.context_field_refs),
                    }
                    for key in read.candidate_keys
                ],
                "entity_references": [
                    {
                        "reference_id": reference.id,
                        "target_entity_kind": reference.target_entity_kind,
                        "target_key_id": reference.target_key_id,
                        "components": [
                            {
                                "target_component_id": component.target_component_id,
                                "local_field_ref": component.local_field_ref,
                            }
                            for component in reference.components
                        ],
                        "context_field_refs": list(reference.context_field_refs),
                    }
                    for reference in read.entity_references
                ],
                "fields": {
                    path: {
                        "row_path_id": field.row_path_id,
                        "choices": list(field.choices),
                        "requirements": [
                            {
                                "param_ref": item.param_ref,
                                "value": item.value,
                            }
                            for item in field.requirements
                        ],
                        "metadata": dict(field.metadata or {}),
                    }
                    for path, field in read.fields_by_path.items()
                },
                "source_metadata": dict(read.source_metadata or {}),
            }
            for read in catalog.reads
        }
    }
    return subset_mismatches(
        actual=result,
        expected_subset=payload["expect"]["result_contains"],
    )


def _projected_pagination(pagination: PaginationMetadata) -> dict[str, str]:
    return {
        "mode": pagination.mode.value,
        "completeness_policy": pagination.completeness_policy.value,
    }


def _endpoint_contract(payload: dict[str, Any]) -> EndpointContract:
    return EndpointContract(
        endpoint_name=str(payload["endpoint_name"]),
        url_name=str(payload.get("url_name") or payload["endpoint_name"]),
        method=str(payload.get("method") or "GET"),
        path_template=str(payload.get("path_template") or ""),
        docstring=str(payload.get("docstring") or ""),
        view_class=str(payload.get("view_class") or ""),
        path_params=tuple(_parameter(item) for item in payload.get("path_params", ())),
        query_params=tuple(
            _parameter(item) for item in payload.get("query_params", ())
        ),
        response_fields=tuple(
            _response_field(item) for item in payload.get("response_fields", ())
        ),
        pagination=_pagination(payload.get("pagination")),
        resource_names=tuple(payload.get("resource_names") or ()),
        candidate_keys=tuple(
            _candidate_key(item) for item in payload.get("candidate_keys") or ()
        ),
        entity_references=tuple(
            _entity_reference(item) for item in payload.get("entity_references") or ()
        ),
    )


def _pagination(payload: object) -> PaginationContract | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("pagination must be an object")
    return PaginationContract(
        kind=PaginationKind(str(payload["kind"])),
        position_query_param=str(payload["position_query_param"]),
        page_size_query_param=str(payload["page_size_query_param"]),
        results_path=str(payload["results_path"]),
        page_size=int(payload["page_size"]),
        max_page_size=int(payload["max_page_size"]),
        total_path=str(payload.get("total_path") or ""),
        continuation_path=str(payload.get("continuation_path") or ""),
    )


def _parameter(payload: dict[str, Any]) -> ParameterContract:
    return ParameterContract(
        name=str(payload["name"]),
        type=str(payload.get("type") or "string"),
        required=bool(payload.get("required") or False),
        description=str(payload.get("description") or ""),
        choices=tuple(payload.get("choices") or ()),
        choice_labels=dict(payload.get("choice_labels") or {}),
        default=payload.get("default"),
        source=str(payload.get("source", "query")),
        entity_target=_entity_target(payload.get("entity_target")),
        semantics=str(payload.get("semantics") or ""),
    )


def _response_field(payload: dict[str, Any]) -> ResponseFieldContract:
    return ResponseFieldContract(
        name=str(payload["name"]),
        type=str(payload.get("type") or "string"),
        path=str(payload["path"]),
        description=str(payload.get("description") or ""),
        choices=tuple(payload.get("choices") or ()),
        requires=dict(payload.get("requires") or {}),
    )


def _entity_target(payload: object) -> EntityKeyComponentTargetContract | None:
    if not isinstance(payload, dict):
        return None
    return EntityKeyComponentTargetContract(
        entity_kind=str(payload["entity_kind"]),
        key_id=str(payload["key_id"]),
        component_id=str(payload["component_id"]),
    )


def _candidate_key(payload: dict[str, Any]) -> CandidateKeyContract:
    components = tuple(
        CandidateKeyComponentContract(
            component_id=str(item["component_id"]),
            field_path=str(item["field_path"]),
        )
        for item in payload["components"]
    )
    return CandidateKeyContract(
        key_id=str(payload["key_id"]),
        entity_kind=str(payload["entity_kind"]),
        components=components,
        primary=bool(payload.get("primary", False)),
        stable=bool(payload.get("stable", True)),
        context_field_paths=tuple(payload.get("context_field_paths") or ()),
    )


def _entity_reference(payload: dict[str, Any]) -> EntityReferenceContract:
    components = tuple(
        EntityReferenceComponentContract(
            target_component_id=str(item["target_component_id"]),
            local_field_path=str(item["local_field_path"]),
        )
        for item in payload["components"]
    )
    return EntityReferenceContract(
        reference_id=str(payload["reference_id"]),
        target_entity_kind=str(payload["target_entity_kind"]),
        target_key_id=str(payload["target_key_id"]),
        components=components,
        context_field_paths=tuple(payload.get("context_field_paths") or ()),
    )
