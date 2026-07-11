"""Compact API read cards for retention review."""

from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import (
    CatalogFactAvailability,
    CatalogFact,
    EndpointRead,
    RelationCatalog,
)
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.turn_prompts.projections import ApiReadResponseShapeProjector
from fervis.lookup.fact_planning.row_set_filters import (
    row_set_filters_for_sources_payload,
)
from fervis.lookup.fact_plan.row_sources import RowSource
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.read_eligibility.source_groups import (
    read_card_source_groups_by_read,
)


def read_eligibility_cards_payload(
    *,
    requested_facts: tuple[RequestedFact, ...],
    catalog_selection: CatalogSelectionResult,
    available_values: tuple[FactValue, ...] = (),
) -> dict[str, object]:
    facts_by_read = _facts_by_read(catalog_selection.relation_catalog)
    reads_by_id = {read.id: read for read in catalog_selection.relation_catalog.reads}
    source_groups_by_read = read_card_source_groups_by_read(
        catalog_selection.relation_catalog
    )
    next_source_index = 1
    requested_fact_read_candidates: list[dict[str, object]] = []
    row_source_counts_by_read = {
        read_id: len(source_groups)
        for read_id, source_groups in source_groups_by_read.items()
    }
    for selection in catalog_selection.requested_fact_selections:
        if not any(fact.id == selection.requested_fact_id for fact in requested_facts):
            continue
        read_candidates = []
        for read_id in selection.selected_read_ids:
            if read_id not in reads_by_id:
                continue
            for source_group in source_groups_by_read.get(read_id, ()):
                read_candidates.append(
                    _read_card(
                        reads_by_id[read_id],
                        requested_fact_id=selection.requested_fact_id,
                        source_candidate_id=f"source_{next_source_index}",
                        facts_by_read=facts_by_read,
                        source_group=source_group,
                        read_row_source_count=row_source_counts_by_read.get(read_id, 1),
                        available_values=available_values,
                    )
                )
                next_source_index += 1
        requested_fact_read_candidates.append(
            {
                "requested_fact_id": selection.requested_fact_id,
                "read_candidates": read_candidates,
            }
        )
    return {"requested_fact_read_candidates": requested_fact_read_candidates}


def _read_card(
    read: EndpointRead,
    *,
    requested_fact_id: str,
    source_candidate_id: str,
    facts_by_read: dict[str, tuple[CatalogFact, ...]],
    source_group: tuple[RowSource, ...],
    read_row_source_count: int,
    available_values: tuple[FactValue, ...],
) -> dict[str, object]:
    read_shape = ApiReadResponseShapeProjector(read)
    read_facts = (*read.facts, *facts_by_read.get(read.id, ()))
    payload: dict[str, object] = {
        "source_candidate_id": source_candidate_id,
        "read_id": read.id,
        "row_path_id": source_group[0].row_path_id if source_group else "",
        "read_row_source_count": read_row_source_count,
        "endpoint_name": read.endpoint_name,
        "resource_names": list(read.resource_names),
        "input_params": read_shape.input_params(include_param_tokens=True),
        "response_rows": read_shape.response_rows(
            row_path_ids=tuple(source.row_path_id or "root" for source in source_group),
            source_candidate_id=source_candidate_id,
            include_evidence_tokens=True,
        ),
        "catalog_facts": [_catalog_fact_payload(fact) for fact in read_facts],
    }
    docstring = _read_docstring(read, facts=read_facts)
    if docstring:
        payload["docstring"] = docstring
    if len(source_group) == 1:
        payload["row_source_id"] = source_group[0].id
    bound_params = _source_group_bound_params(source_group)
    if bound_params:
        payload["bound_params"] = bound_params
    known_inputs = _applicable_known_inputs(
        _applicable_filter_payloads(
            source_group,
            requested_fact_id=requested_fact_id,
            available_values=available_values,
        ),
        source_group=source_group,
    )
    if known_inputs:
        payload["applicable_known_inputs"] = known_inputs
    return payload


def _applicable_filter_payloads(
    source_group: tuple[RowSource, ...],
    *,
    requested_fact_id: str,
    available_values: tuple[FactValue, ...],
) -> list[dict[str, Any]]:
    if not available_values:
        return []
    return row_set_filters_for_sources_payload(
        source_group,
        grounded_params={},
        values_by_id={value.id: value for value in available_values},
        requested_fact_id=requested_fact_id,
    )


def _applicable_known_inputs(
    filters: list[dict[str, Any]],
    *,
    source_group: tuple[RowSource, ...],
) -> list[dict[str, object]]:
    field_paths_by_id = {
        field.id: field.path
        for source in source_group
        for field in source.fields
        if field.path
    }
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in filters:
        known_input_id = str(item.get("known_input_id") or "")
        if not known_input_id:
            continue
        field_ids = tuple(str(field_id) for field_id in item.get("field_ids") or ())
        for field_id in field_ids:
            field_path = field_paths_by_id.get(field_id)
            if not field_path:
                continue
            key = (known_input_id, field_path)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "known_input_id": known_input_id,
                    "display_value": str(item.get("display_value") or ""),
                    "identity_type": str(item.get("identity_type") or ""),
                    "applies_via_field": field_path,
                    "role": "POPULATION_SCOPE",
                }
            )
    return output


def _catalog_fact_payload(fact: CatalogFact) -> dict[str, str]:
    output = {
        "fact_ref": fact.ref,
        "availability": fact.availability.value,
    }
    if fact.availability == CatalogFactAvailability.AVAILABLE and fact.field_ref:
        output["field_ref"] = fact.field_ref
    return output


def _read_docstring(read: EndpointRead, *, facts: tuple[CatalogFact, ...]) -> str:
    if any(fact.availability != CatalogFactAvailability.AVAILABLE for fact in facts):
        return ""
    metadata = read.source_metadata if isinstance(read.source_metadata, dict) else {}
    return str(metadata.get("description") or "").strip()


def _source_group_bound_params(sources: tuple[RowSource, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for param in source.params:
            if param.default_source != "source_variant":
                continue
            key = (param.param_ref, str(param.default))
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "param_id": param.id,
                    "param_ref": param.param_ref,
                    "name": param.name,
                    "value": param.default,
                    "semantics": param.semantics.value,
                }
            )
    return output


def _facts_by_read(catalog: RelationCatalog) -> dict[str, tuple[CatalogFact, ...]]:
    output: dict[str, list[CatalogFact]] = {}
    for fact in catalog.facts:
        if not fact.read_id:
            continue
        output.setdefault(fact.read_id, []).append(fact)
    return {key: tuple(value) for key, value in output.items()}
