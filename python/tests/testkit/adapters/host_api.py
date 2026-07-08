from __future__ import annotations

from typing import Any

from fervis.host_api.contracts import (
    EndpointContract,
    ParameterContract,
    ResponseFieldContract,
)
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.lookup.relation_catalog import validate_relation_catalog

from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)


def run_host_api_projection_case(payload: dict[str, Any]) -> list[str]:
    try:
        catalog = relation_catalog_from_endpoint_contracts(
            tuple(_endpoint_contract(item) for item in payload["input"]["contracts"])
        )
        validate_relation_catalog(catalog)
    except Exception as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
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
                "pagination": {
                    "mode": read.pagination.mode.value,
                    "completeness_policy": read.pagination.completeness_policy.value,
                },
                "row_paths": [
                    {
                        "id": item.id,
                        "path": item.path,
                        "cardinality": item.cardinality.value,
                    }
                    for item in read.row_paths
                ],
                "fields": {
                    path: {
                        "row_path_id": field.row_path_id,
                        "choices": list(field.choices),
                        "identity": (
                            {
                                "primary_key": field.identity.primary_key,
                                "identity_field": field.identity.identity_field,
                            }
                            if field.identity
                            else None
                        ),
                        "requirements": [
                            {
                                "param_ref": item.param_ref,
                                "value": item.value,
                            }
                            for item in field.requirements
                        ],
                        "metadata": dict(field.metadata),
                    }
                    for path, field in read.fields_by_path.items()
                },
                "source_metadata": dict(read.source_metadata),
            }
            for read in catalog.reads
        }
    }
    return subset_mismatches(
        actual=result,
        expected_subset=payload["expect"]["result_contains"],
    )


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
        paginated=bool(payload.get("paginated") or False),
        resource_names=tuple(payload.get("resource_names") or ()),
        primary_key_fields=tuple(payload.get("primary_key_fields") or ()),
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
        identity=dict(payload.get("identity") or {}),
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
        identity=dict(payload.get("identity") or {}),
    )
