from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import (
    CompletenessPolicy,
    EndpointRead,
    PaginationMetadata,
    PaginationMode,
)
from fervis.lookup.plan_execution.relations import api_read_completeness_proof
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    PopulationChoiceControllerKind,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_planning.pattern_plan.shared import (
    _relation_fields_with_source_requirements,
)
from fervis.lookup.source_binding.compiler_ir import (
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.fact_plan.row_sources import (
    build_row_source_catalog,
    CALENDAR_DATE_FIELD_ID,
    CALENDAR_END_PARAM_ID,
    CALENDAR_ROW_SOURCE_ID,
    CALENDAR_START_PARAM_ID,
)
from fervis.lookup.fact_plan.row_sources import row_source_for_relation

from tests.testkit.assertions import exact_mismatches, subset_mismatches
from tests.testkit.catalog import catalog_from_payload


def run_relation_contract_case(payload: dict[str, Any]) -> list[str]:
    mode = str(payload["input"]["mode"])
    if mode == "relation":
        relation = _relation(payload["input"]["relation"])
        actual = {
            "source_read_id": relation.source.read_id,
            "grain_keys": list(relation.grain_keys),
            "fields": {
                field.field_id: {
                    "roles": [role.value for role in field.roles],
                }
                for field in relation.fields
            },
        }
        return subset_mismatches(
            actual=actual,
            expected_subset=payload["expect"]["result_contains"],
        )
    if mode == "calendar_row_source":
        source = build_row_source_catalog(catalog_from_payload({"reads": []})).source(
            CALENDAR_ROW_SOURCE_ID
        )
        actual = {
            "id": source.id,
            "date_field_label": source.field(CALENDAR_DATE_FIELD_ID).label,
            "start_param_required": source.param(CALENDAR_START_PARAM_ID).required,
            "end_param_required": source.param(CALENDAR_END_PARAM_ID).required,
        }
        return subset_mismatches(
            actual=actual,
            expected_subset=payload["expect"]["result_contains"],
        )
    if mode == "row_source_for_relation":
        catalog = catalog_from_payload(payload["input"]["catalog"])
        row_sources = build_row_source_catalog(catalog)
        relation_payload = dict(payload["input"]["relation"])
        source_payload = dict(relation_payload["source"])
        if not source_payload.get("row_source_id"):
            expected_row_path_id = str(source_payload.pop("row_path_id"))
            source_payload["row_source_id"] = next(
                source.id
                for source in row_sources.sources
                if source.read_id == source_payload.get("read_id")
                and source.row_path_id == expected_row_path_id
            )
            relation_payload["source"] = source_payload
        relation = _relation(relation_payload)
        source = row_source_for_relation(relation, row_sources=row_sources)
        return subset_mismatches(
            actual={"row_source_id": source.id, "row_path_id": source.row_path_id},
            expected_subset=payload["expect"]["result_contains"],
        )
    if mode == "relation_fields_for_source_requirements":
        relation_fields = tuple(
            RelationField(
                field_id=str(item["field_id"]),
                roles=tuple(
                    FieldBindingRole(str(role)) for role in item.get("roles") or ()
                ),
            )
            for item in payload["input"].get("fields") or ()
        )
        population_choices = tuple(
            DraftRelationSourcePopulationChoice(
                controller_kind=PopulationChoiceControllerKind(
                    str(item["controller_kind"])
                ),
                controller_id=str(item["controller_id"]),
                field_id=str(item["field_id"]),
                requested_fact_ids=tuple(item["requested_fact_ids"]),
                included_values=tuple(item["included_values"]),
                excluded_values=tuple(item.get("excluded_values") or ()),
            )
            for item in payload["input"].get("population_choices") or ()
        )
        fields = _relation_fields_with_source_requirements(
            relation_fields,
            source_filters=(),
            population_choices=population_choices,
        )
        return exact_mismatches(
            actual={"field_ids": [field.field_id for field in fields]},
            expected=payload["expect"]["result_equals"],
        )
    if mode == "api_read_completeness":
        read_payload = payload["input"]["read"]
        read = EndpointRead(
            id=str(read_payload["id"]),
            endpoint_name=str(read_payload.get("endpoint_name") or read_payload["id"]),
            pagination=(
                PaginationMetadata(
                    mode=PaginationMode(str(read_payload["pagination"]["mode"])),
                    completeness_policy=CompletenessPolicy(
                        str(read_payload["pagination"]["completeness_policy"])
                    ),
                )
                if isinstance(read_payload.get("pagination"), dict)
                else None
            ),
        )
        actual = {
            name: {
                "status": proof.status.value,
                "pagination": proof.pagination.value,
            }
            for name, proof in (
                (
                    item["id"],
                    api_read_completeness_proof(
                        read,
                        row_count=int(item.get("row_count") or 0),
                        reached_terminal_page=bool(item.get("reached_terminal_page")),
                        page_cap_reached=bool(item.get("page_cap_reached")),
                    ),
                )
                for item in payload["input"]["proof_requests"]
            )
        }
        return subset_mismatches(
            actual=actual,
            expected_subset=payload["expect"]["result_contains"],
        )
    return [f"unsupported relation contract mode: {mode}"]


def _relation(payload: dict[str, Any]) -> Relation:
    return Relation(
        id=str(payload["id"]),
        source=RelationSource(
            kind=SourceKind(str(payload["source"]["kind"])),
            read_id=str(payload["source"].get("read_id") or ""),
            row_source_id=str(payload["source"].get("row_source_id") or ""),
        ),
        fields=tuple(
            RelationField(
                field_id=str(item["field_id"]),
                roles=tuple(
                    FieldBindingRole(str(role)) for role in item.get("roles") or ()
                ),
            )
            for item in payload.get("fields") or ()
        ),
    )
