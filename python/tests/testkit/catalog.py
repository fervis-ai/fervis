from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import (
    CatalogFact,
    CatalogFactAvailability,
    CatalogField,
    CatalogParam,
    CandidateKey,
    CandidateKeyAuthority,
    CandidateKeyAuthorityComponent,
    CandidateKeyComponent,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
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
        candidate_key_authorities=tuple(
            _candidate_key_authority(item)
            for item in payload.get("candidate_key_authorities") or ()
        ),
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
            read_id for fact in requested_facts for read_id in fact.selected_read_ids
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
        candidate_keys=tuple(
            _candidate_key(item) for item in payload.get("candidate_keys") or ()
        ),
        entity_references=tuple(
            _entity_reference(item) for item in payload.get("entity_references") or ()
        ),
        facts=tuple(_fact(item) for item in payload.get("facts") or ()),
        source_metadata=(
            dict(payload["source_metadata"])
            if payload.get("source_metadata") is not None
            else None
        ),
    )


def _param(payload: dict[str, Any]) -> CatalogParam:
    return CatalogParam(
        ref=str(payload["ref"]),
        name=str(payload["name"]),
        source=ParamSource(str(payload.get("source") or ParamSource.QUERY)),
        type=str(payload.get("type") or "string"),
        required=bool(payload.get("required") or False),
        choices=tuple(payload.get("choices") or ()),
        entity_target=_entity_target(payload.get("entity_target")),
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
        requirements=tuple(
            _field_requirement(item) for item in payload.get("requirements") or ()
        ),
        metadata=(
            dict(payload["metadata"]) if payload.get("metadata") is not None else None
        ),
    )


def _candidate_key(payload: dict[str, Any]) -> CandidateKey:
    return CandidateKey(
        id=str(payload["id"]),
        entity_kind=str(payload["entity_kind"]),
        components=tuple(
            CandidateKeyComponent(
                id=str(item["id"]),
                field_ref=str(item["field_ref"]),
            )
            for item in payload["components"]
        ),
        primary=bool(payload.get("primary", False)),
        stable=bool(payload.get("stable", True)),
        context_field_refs=tuple(
            str(item) for item in payload.get("context_field_refs") or ()
        ),
    )


def _candidate_key_authority(payload: dict[str, Any]) -> CandidateKeyAuthority:
    return CandidateKeyAuthority(
        id=str(payload["id"]),
        entity_kind=str(payload["entity_kind"]),
        components=tuple(
            CandidateKeyAuthorityComponent(
                id=str(item["id"]),
                type=str(item["type"]),
            )
            for item in payload["components"]
        ),
        primary=bool(payload.get("primary", False)),
        stable=bool(payload.get("stable", True)),
    )


def _entity_reference(payload: dict[str, Any]) -> EntityReference:
    return EntityReference(
        id=str(payload["id"]),
        target_entity_kind=str(payload["target_entity_kind"]),
        target_key_id=str(payload["target_key_id"]),
        components=tuple(
            EntityReferenceComponent(
                target_component_id=str(item["target_component_id"]),
                local_field_ref=str(item["local_field_ref"]),
            )
            for item in payload["components"]
        ),
        context_field_refs=tuple(
            str(item) for item in payload.get("context_field_refs") or ()
        ),
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


def _entity_target(payload: object) -> EntityKeyComponentTarget | None:
    if not isinstance(payload, dict):
        return None
    return EntityKeyComponentTarget(
        entity_kind=str(payload["entity_kind"]),
        key_id=str(payload["key_id"]),
        component_id=str(payload["component_id"]),
    )
