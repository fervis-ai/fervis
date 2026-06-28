from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import (
    CatalogFact,
    CatalogFactAvailability,
    CatalogField,
    CatalogParam,
    FieldRequirement,
    EndpointRead,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)


def catalog_from_payload(payload: dict[str, Any]) -> RelationCatalog:
    return RelationCatalog(
        reads=tuple(_read(item) for item in payload.get("reads") or ()),
        facts=tuple(_fact(item) for item in payload.get("facts") or ()),
    )


def catalog_selection_from_payload(
    payload: dict[str, Any],
    *,
    catalog: RelationCatalog,
) -> CatalogSelectionResult:
    requested_facts = tuple(
        RequestedFactCatalogSelection(
            requested_fact_id=str(item["requested_fact_id"]),
            query_terms=tuple(item.get("query_terms") or ()),
            rankings=tuple(
                CatalogSelectionRanking(
                    read_id=str(read_id),
                    score=1,
                    matched_terms=tuple(item.get("query_terms") or ()),
                )
                for read_id in item.get("selected_read_ids") or ()
            ),
            selected_read_ids=tuple(item.get("selected_read_ids") or ()),
        )
        for item in payload.get("requested_facts") or ()
    )
    return CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=requested_facts,
        selected_read_ids=tuple(
            read_id
            for fact in requested_facts
            for read_id in fact.selected_read_ids
        ),
    )


def _read(payload: dict[str, Any]) -> EndpointRead:
    return EndpointRead(
        id=str(payload["id"]),
        endpoint_name=str(payload.get("endpoint_name") or payload["id"]),
        method=str(payload.get("method") or "GET"),
        path=str(payload.get("path") or ""),
        resource_names=tuple(payload.get("resource_names") or ()),
        params=tuple(_param(item) for item in payload.get("params") or ()),
        row_paths=tuple(_row_path(item) for item in payload.get("row_paths") or ()),
        fields=tuple(_field(item) for item in payload.get("fields") or ()),
        facts=tuple(_fact(item) for item in payload.get("facts") or ()),
        source_metadata=dict(payload.get("source_metadata") or {}),
    )


def _param(payload: dict[str, Any]) -> CatalogParam:
    return CatalogParam(
        ref=str(payload["ref"]),
        name=str(payload["name"]),
        source=ParamSource(str(payload.get("source") or ParamSource.QUERY)),
        type=str(payload.get("type") or "string"),
        required=bool(payload.get("required") or False),
        choices=tuple(payload.get("choices") or ()),
        identity=_identity(payload.get("identity")),
        default=payload.get("default"),
        semantics=str(payload.get("semantics") or ""),
    )


def _row_path(payload: dict[str, Any]) -> RowPath:
    return RowPath(
        id=str(payload["id"]),
        path=str(payload["path"]),
        cardinality=RowCardinality(str(payload["cardinality"])),
        parent_path=str(payload.get("parent_path") or ""),
    )


def _field(payload: dict[str, Any]) -> CatalogField:
    return CatalogField(
        ref=str(payload["ref"]),
        path=str(payload.get("path") or ""),
        row_path_id=str(payload.get("row_path_id") or ""),
        type=str(payload.get("type") or "string"),
        choices=tuple(payload.get("choices") or ()),
        identity=_identity(payload.get("identity")),
        requirements=tuple(
            _field_requirement(item) for item in payload.get("requirements") or ()
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def _fact(payload: dict[str, Any]) -> CatalogFact:
    return CatalogFact(
        ref=str(payload["ref"]),
        availability=CatalogFactAvailability(
            str(payload.get("availability") or CatalogFactAvailability.AVAILABLE)
        ),
        field_ref=str(payload.get("field_ref") or ""),
        read_id=str(payload.get("read_id") or ""),
        proof_refs=tuple(payload.get("proof_refs") or ()),
    )


def _field_requirement(payload: dict[str, Any]) -> FieldRequirement:
    return FieldRequirement(
        param_ref=str(payload["param_ref"]),
        value=payload["value"],
    )


def _identity(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    from fervis.lookup.relation_catalog import IdentityMetadata

    return IdentityMetadata(
        entity_ref=str(payload.get("entity_ref") or ""),
        identity_field=str(payload.get("identity_field") or ""),
        primary_key=bool(payload.get("primary_key") or False),
        stable=bool(payload.get("stable", True)),
    )
