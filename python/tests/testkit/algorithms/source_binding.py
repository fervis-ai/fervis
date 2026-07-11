from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from typing import Any
from xml.etree import ElementTree

from jsonschema import ValidationError, validate

from fervis.lookup.relation_catalog import (
    CatalogField,
    CatalogParam,
    EndpointRead,
    IdentityMetadata,
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
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.plan_selection import (
    PlanSelectionSet,
    SelectedSourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.fact_plan.row_sources import api_row_source_id
from fervis.lookup.answer_program.values import (
    FactValue,
    TimeComponent,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.question_contract import (
    GroupKeyDomainKind,
    KnownInputSource,
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    LiteralInputRole,
    NormalInstanceExcludedStateRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactGroupKey,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactLiteralInput,
)
from fervis.lookup.read_eligibility import (
    ReadAssessment,
    ReadEligibilityRequest,
    ReadEligibilityResult,
)
from fervis.lookup.read_eligibility.surface import (
    read_eligibility_candidate_surface,
)
from fervis.lookup.source_binding import (
    SourceBindingRequest,
    SourceBindingTurnPrompt,
    parse_source_binding,
)
from fervis.lookup.source_binding.plan_targets import (
    source_binding_fact_field_id,
    source_binding_fact_id_from_field,
    source_binding_targets_for_plan_selection,
)
from fervis.lookup.source_binding.candidates.compact import (
    _compact_prompt_payload,
)
from fervis.lookup.source_binding.candidates.evidence import (
    _candidate_with_evidence_items,
)
from fervis.lookup.source_binding.candidates.fulfillment_slots import (
    _candidate_with_fulfillment_slots,
)
from fervis.lookup.source_binding.candidates.row_predicates import (
    candidate_with_row_predicates,
)
from fervis.lookup.source_binding.candidates.registry_builder import (
    _source_candidates_from_cards,
)
from fervis.lookup.source_binding.candidates.registry import source_candidate_registry
from fervis.lookup.source_binding.review_scope import source_binding_review_scope
from fervis.lookup.source_binding.review_surface import (
    SourceBindingReviewAxisKind,
    source_binding_review_surface,
)
from fervis.lookup.source_binding.plan_targets import source_binding_target_index

from tests.testkit.assertions import (
    exact_mismatches,
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)
from tests.testkit.catalog import catalog_from_payload
from tests.lookup.prompt_sections import prompt_section_text


def run_source_binding_plan_families_case(payload: dict[str, Any]) -> list[str]:
    request = _plan_family_request()
    projected = SourceBindingTurnPrompt(request).binding_plan_families_payload()
    facts = projected["bindings_by_requested_fact"]
    actual = {
        requested_fact_id: {
            plan_shape: {
                "member_constraint": shape["member_constraint"],
                "required_answer_output_ids": shape["required_answer_output_ids"],
                "roles": list(shape["role_targets"]),
                "role_target_counts": {
                    role_id: len(targets)
                    for role_id, targets in shape["role_targets"].items()
                },
            }
            for plan_shape, shape in fact["plan_shapes"].items()
        }
        for requested_fact_id, fact in facts.items()
    }
    return exact_mismatches(
        actual=actual,
        expected=payload["expect"]["result_equals"],
    )


def run_source_binding_row_predicates_case(payload: dict[str, Any]) -> list[str]:
    candidate = candidate_with_row_predicates(
        dict(payload["input"]["candidate"]),
        relation_catalog=catalog_from_payload(payload["input"]["catalog"]),
    )
    row_predicates = list(candidate.get("row_predicates") or ())
    return subset_mismatches(
        actual={
            "row_predicates": row_predicates,
            "field_ids": [item["field_id"] for item in row_predicates],
            "predicate_ids": [item["predicate_id"] for item in row_predicates],
            "predicate_id_count": len(row_predicates),
            "unique_predicate_id_count": len(
                {item["predicate_id"] for item in row_predicates}
            ),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_source_binding_bound_params_case(payload: dict[str, Any]) -> list[str]:
    candidates = _source_candidates_from_cards(
        _candidate_cards_with_runtime_ids(payload["input"]["candidate_cards"]),
        model_visible=False,
    )
    candidate = candidates[str(payload["input"]["candidate_id"])]
    return subset_mismatches(
        actual={
            "applied_param_bindings": [
                {
                    "param_id": binding.param_id,
                    "value": binding.value,
                    "proof_refs": list(binding.proof_refs),
                }
                for binding in candidate.applied_param_bindings
            ],
            "applied_param_binding_sets": [
                [
                    {
                        "param_id": binding.param_id,
                        "value": binding.value,
                        "proof_refs": list(binding.proof_refs),
                    }
                    for binding in binding_set
                ]
                for binding_set in candidate.applied_param_binding_sets
            ],
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _candidate_cards_with_runtime_ids(payload: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(payload)
    for fact_sources in output.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for context in fact_sources.get("source_contexts") or ():
            if not isinstance(context, dict):
                continue
            for candidate in context.get("source_options") or ():
                if not isinstance(candidate, dict):
                    continue
                candidate_id = str(candidate.pop("candidate_id", "") or "")
                if candidate_id:
                    candidate["source_candidate_id"] = candidate_id
    return output


def run_source_binding_fulfillment_support_case(payload: dict[str, Any]) -> list[str]:
    requested_fact = _requested_fact(payload["input"].get("requested_fact"))
    candidate = _candidate_with_fulfillment_slots(
        _candidate_with_evidence_items(dict(payload["input"]["candidate"])),
        requested_facts=(requested_fact,),
    )
    if payload["input"].get("project_model_visible"):
        candidate = _only_candidate(
            _compact_prompt_payload(
                {
                    "requested_fact_sources": [
                        {
                            "requested_fact_id": requested_fact.id,
                            "source_contexts": [{"source_options": [candidate]}],
                        }
                    ]
                },
                relation_catalog=catalog_from_payload(payload["input"]["catalog"]),
                requested_facts=(requested_fact,),
            )
        )
    slots = [
        slot
        for support_set in _binding_surface(candidate).get("fulfillment_support_sets")
        or ()
        if isinstance(support_set, dict)
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
    ]
    expected = payload["expect"]["result_contains"]
    group_key_fields = {
        str(item["field_id"]): True
        for slot in slots
        for item in slot.get("group_key_evidence") or ()
        if isinstance(item, dict)
    }
    metric_fields = {
        str(item["field_id"]): True
        for slot in slots
        for item in slot.get("metric_measure_evidence") or ()
        if isinstance(item, dict)
    }
    group_key_fields.update(
        {
            field_id: False
            for field_id, expected_value in (
                (expected.get("group_key_fields") or {}).items()
            )
            if expected_value is False and field_id not in group_key_fields
        }
    )
    metric_fields.update(
        {
            field_id: False
            for field_id, expected_value in (
                (expected.get("metric_fields") or {}).items()
            )
            if expected_value is False and field_id not in metric_fields
        }
    )
    return subset_mismatches(
        actual={
            "group_key_field_ids": sorted(
                {
                    str(item["field_id"])
                    for slot in slots
                    for item in slot.get("group_key_evidence") or ()
                    if isinstance(item, dict)
                }
            ),
            "group_key_fields": group_key_fields,
            "metric_field_ids": sorted(
                {
                    str(item["field_id"])
                    for slot in slots
                    for item in slot.get("metric_measure_evidence") or ()
                    if isinstance(item, dict)
                }
            ),
            "metric_fields": metric_fields,
            "row_count_evidence_ids": sorted(
                {
                    str(item["evidence_id"])
                    for slot in slots
                    for item in slot.get("row_count_basis_evidence") or ()
                    if isinstance(item, dict)
                }
            ),
            "metric_evidence_ids": sorted(
                {
                    str(item["evidence_id"])
                    for slot in slots
                    for item in slot.get("metric_measure_evidence") or ()
                    if isinstance(item, dict)
                }
            ),
            "row_count_evidence": sorted(
                [
                    str(item.get("field_id") or ""),
                    str(item.get("row_source_id") or ""),
                ]
                for slot in slots
                for item in slot.get("row_count_basis_evidence") or ()
                if isinstance(item, dict)
            ),
            "row_count_slot_sizes": [
                len(slot.get("row_count_basis_evidence") or ())
                for slot in slots
                if slot.get("row_count_basis_evidence")
            ],
            "slot_role_presence_by_answer_output": (
                _slot_role_presence_by_answer_output(slots)
            ),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_source_binding_prompt_surface_case(payload: dict[str, Any]) -> list[str]:
    if _request_mode(payload["input"].get("request") or {}) == (
        "custom_supported_evidence_projection"
    ):
        return _run_custom_supported_evidence_projection_case(payload)
    request = _prompt_surface_request(payload["input"].get("request") or {})
    prompt = SourceBindingTurnPrompt(request)
    prompt_payload = prompt.source_invocation_candidate_payload()
    candidates = _source_candidates(prompt_payload)
    candidate = (
        candidates[0]
        if len(candidates) == 1
        else (
            {}
            if _prompt_surface_allows_multiple_candidates(payload)
            else _only_candidate(prompt_payload)
        )
    )
    surface = _binding_surface(candidate)
    candidate_keys = set(candidate)
    invocation = prompt.to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context={},
            memory_payload={},
        )
    )
    schema = prompt.response_contract().provider_schema
    expected = payload["expect"].get("result_contains", {})
    expected_prompt_terms = [
        str(item) for item in expected.get("prompt_text_contains") or ()
    ]
    actual = {
        "tool_name": invocation.tool_specs[0].name,
        "prompt_text_contains": [
            term for term in expected_prompt_terms if term in invocation.prompt_text
        ],
        "rendered_fulfillment_evidence": _rendered_fulfillment_evidence(
            invocation.prompt_text
        ),
        "source_candidate_ids": [
            str(item.get("source_candidate_id") or "") for item in candidates
        ],
        "candidate_count": len(candidates),
        "read_ids": [str(item.get("read_id") or "") for item in candidates],
        "read_id_presence": {
            read_id: read_id in {str(item.get("read_id") or "") for item in candidates}
            for read_id in payload["expect"]
            .get("result_contains", {})
            .get("read_id_presence", {})
        },
        "read_id": candidate.get("read_id"),
        "response_rows": candidate.get("response_rows") or [],
        "response_row_paths": [
            str(item.get("path") or "")
            for item in candidate.get("response_rows") or ()
            if isinstance(item, dict)
        ],
        "response_row_cardinalities": {
            str(item.get("path") or ""): str(item.get("cardinality") or "")
            for item in candidate.get("response_rows") or ()
            if isinstance(item, dict)
        },
        "response_row_fields_by_path": {
            str(item.get("path") or ""): [
                _response_row_field_id(field)
                for field in item.get("fields") or ()
                if isinstance(field, (dict, str))
            ]
            for item in candidate.get("response_rows") or ()
            if isinstance(item, dict)
        },
        "input_param_names": [
            str(item.get("name") or "")
            for item in candidate.get("input_params") or ()
            if isinstance(item, dict)
        ],
        "bound_params": surface.get("bound_params") or [],
        "candidate_keys": sorted(candidate),
        "candidate_key_presence": {
            key: key in candidate
            for key in payload["expect"]
            .get("result_contains", {})
            .get("candidate_key_presence", {})
        },
        "surface_keys": sorted(surface),
        "surface_key_presence": {
            key: key in surface
            for key in payload["expect"]
            .get("result_contains", {})
            .get("surface_key_presence", {})
        },
        "slot_answer_output_ids": sorted(
            {
                str(slot.get("answer_output_id") or "")
                for support_set in surface.get("fulfillment_support_sets") or ()
                if isinstance(support_set, dict)
                for slot in support_set.get("fulfillment_slots") or ()
                if isinstance(slot, dict)
            }
        ),
        "param_ids": [
            str(item.get("param_id") or "")
            for item in surface.get("params") or ()
            if isinstance(item, dict)
        ],
        "param_required_by_id": {
            str(item.get("param_id") or ""): item.get("required") is True
            for item in surface.get("params") or ()
            if isinstance(item, dict)
        },
        "finite_choice_review_param_ids": [
            str(item.get("param_id") or "")
            for item in surface.get("params") or ()
            if isinstance(item, dict)
            and isinstance(item.get("population_contract"), dict)
        ],
        "row_predicate_field_ids": [
            str(item.get("field_id") or "")
            for item in surface.get("row_predicates") or ()
            if isinstance(item, dict)
        ],
        "population_roles": candidate.get("population_roles") or [],
        "has_fulfillment_support": bool(surface.get("fulfillment_support_sets")),
        "excluded_state_role_names": sorted(
            {
                str(item.get("role") or "")
                for param in surface.get("params") or ()
                if isinstance(param, dict)
                for profile in param.get("normal_instance_role_profiles") or ()
                if isinstance(profile, dict)
                for item in profile.get("excluded_state_roles") or ()
                if isinstance(item, dict) and str(item.get("role") or "")
            }
        ),
        "excluded_state_role_definitions": {
            str(item.get("role") or ""): str(item.get("role_definition") or "")
            for param in surface.get("params") or ()
            if isinstance(param, dict)
            for profile in param.get("normal_instance_role_profiles") or ()
            if isinstance(profile, dict)
            for item in profile.get("excluded_state_roles") or ()
            if isinstance(item, dict) and str(item.get("role") or "")
        },
        "absent_candidate_keys": {
            key: key not in candidate_keys
            for key in payload["input"].get("absent_candidate_keys") or ()
        },
        "schema_property_order": {
            "finite_choice_param_review": _schema_property_order(
                schema,
                markers=(
                    "controlled_population_role_id",
                    "population_test_basis",
                    "choice_reviews",
                ),
            ),
            "population_test_result": _schema_property_order(
                schema,
                markers=("test_basis", "population_consequence", "test_effect"),
            ),
            "normal_instance_test_result": _schema_property_order(
                schema,
                markers=(
                    "role_match_basis",
                    "explicit_user_override_applies",
                    "population_consequence",
                    "disposition",
                ),
            ),
        },
        "schema_review_test_ids": _schema_review_test_ids(schema),
        "schema_review_test_id_counts": _schema_review_test_id_counts(schema),
    }
    errors = subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"].get("result_contains") or {},
    )
    for field, expected_value in (
        payload["expect"].get("result_exact_fields") or {}
    ).items():
        errors.extend(
            exact_mismatches(
                actual=actual.get(field),
                expected=expected_value,
                path=field,
            )
        )
    return errors


def _prompt_surface_allows_multiple_candidates(payload: dict[str, Any]) -> bool:
    expected = payload["expect"].get("result_contains", {})
    return any(key in expected for key in ("candidate_count", "read_id_presence"))


def run_source_binding_review_scope_case(payload: dict[str, Any]) -> list[str]:
    request = _source_binding_request(payload["input"]["request"])
    registry = source_candidate_registry(request)
    target_index = source_binding_target_index(request)
    candidates = dict(registry.candidates_by_id)
    candidate = next(iter(candidates.values()))
    candidate = _candidate_with_axis_owner_edges(
        candidate,
        finite_choice_axis_owners=payload["input"].get("finite_choice_axis_owners")
        or {},
        row_predicate_axis_owners=payload["input"].get("row_predicate_axis_owners")
        or {},
    )
    candidates[candidate.id] = candidate
    scope = source_binding_review_scope(
        request,
        candidates_by_id=candidates,
        target_index=target_index,
    )
    target = target_index.targets[0]
    surface = source_binding_review_surface(candidate)
    actual = {
        "finite_choice": {
            axis_id: list(
                scope.finite_choice_param_test_ids(
                    target.binding_target_id,
                    axis_id,
                )
            )
            for axis_id in surface.finite_choice_params
        },
        "row_predicate": {
            axis_id: list(
                scope.row_predicate_test_ids(
                    target.binding_target_id,
                    axis_id,
                )
            )
            for axis_id in surface.row_predicates
        },
        "decisions": {
            f"{axis_kind.value}:{axis_id}": _scope_decisions(
                scope,
                binding_target_id=target.binding_target_id,
                axis_kind=axis_kind,
                axis_id=axis_id,
            )
            for axis_kind, axis_ids in (
                (
                    SourceBindingReviewAxisKind.FINITE_CHOICE_PARAM,
                    tuple(surface.finite_choice_params),
                ),
                (
                    SourceBindingReviewAxisKind.ROW_PREDICATE,
                    tuple(surface.row_predicates),
                ),
            )
            for axis_id in axis_ids
        },
    }
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual,
            expected=payload["expect"]["result_equals"],
        )
    errors = subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"].get("result_contains") or {},
    )
    for field, expected_value in (
        payload["expect"].get("result_exact_fields") or {}
    ).items():
        errors.extend(
            exact_mismatches(
                actual=actual.get(field),
                expected=expected_value,
                path=field,
            )
        )
    return errors


def _scope_decisions(
    scope: Any,
    *,
    binding_target_id: str,
    axis_kind: SourceBindingReviewAxisKind,
    axis_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            "membership_test_id": decision.membership_test_id,
            "decision": decision.decision.value,
            "owner_surface_id": decision.owner_surface_id,
            "owner_surface_ids": list(decision.owner_surface_ids),
            "proof_refs": list(decision.proof_refs),
        }
        for decision in scope.axis_scope(
            binding_target_id,
            axis_kind,
            axis_id,
        ).test_scope_decisions
    ]


def _candidate_with_axis_owner_edges(
    candidate: Any,
    *,
    finite_choice_axis_owners: dict[str, Any],
    row_predicate_axis_owners: dict[str, Any],
) -> Any:
    params = tuple(
        _param_with_axis_owner_edge(param, finite_choice_axis_owners)
        for param in candidate.params
    )
    payload = dict(candidate.payload or {})
    payload["row_predicates"] = [
        _row_predicate_with_axis_owner_edge(item, row_predicate_axis_owners)
        for item in payload.get("row_predicates") or ()
        if isinstance(item, dict)
    ]
    return replace(candidate, params=params, payload=payload)


def _param_with_axis_owner_edge(
    param: Any,
    owners_by_param: dict[str, Any],
) -> Any:
    if not isinstance(param, dict):
        return param
    param_id = str(param.get("param_id") or "")
    owners = tuple(
        str(item) for item in owners_by_param.get(param_id) or () if str(item)
    )
    if not owners:
        return param
    output = dict(param)
    contract = dict(output.get("population_contract") or {})
    contract["owned_membership_test_ids"] = list(owners)
    output["population_contract"] = contract
    return output


def _row_predicate_with_axis_owner_edge(
    item: dict[str, Any],
    owners_by_predicate: dict[str, Any],
) -> dict[str, Any]:
    predicate_id = str(item.get("predicate_id") or "")
    owners = tuple(
        str(owner)
        for owner in owners_by_predicate.get(predicate_id) or ()
        if str(owner)
    )
    if not owners:
        return dict(item)
    output = dict(item)
    output["owned_membership_test_ids"] = list(owners)
    return output


def _rendered_fulfillment_evidence(prompt_text: str) -> list[dict[str, str]]:
    raw = prompt_section_text(prompt_text, "Candidate evidence sources")
    root = ElementTree.fromstring(raw)
    evidence_items: list[dict[str, str]] = []
    for choice in root.findall(".//choice"):
        for child in choice:
            if child.tag != "evidence":
                continue
            item = {
                "kind": str(child.get("kind") or ""),
                "field": str(child.get("field") or ""),
                "label": str(child.get("label") or ""),
                "row_path": str(child.get("row_path") or ""),
                "type": str(child.get("type") or ""),
                "evidence_id": str(child.get("evidence_id") or ""),
            }
            meaning = str(child.get("meaning") or "")
            if meaning:
                item["meaning"] = meaning
            evidence_items.append({key: value for key, value in item.items() if value})
    return sorted(
        evidence_items,
        key=lambda item: (
            item.get("field") not in {"sale_id", "sale_type"},
            item.get("field", ""),
            item.get("kind", ""),
        ),
    )


def _run_custom_supported_evidence_projection_case(
    payload: dict[str, Any],
) -> list[str]:
    compact = _compact_prompt_payload(
        _custom_supported_evidence_payload(),
        relation_catalog=_custom_supported_evidence_catalog(),
    )
    source = compact["requested_fact_sources"][0]["source_contexts"][0][
        "source_options"
    ][0]
    slot = source["fulfillment_choices"][0]["fulfillment_slots"][0]
    parser_slot = _source_candidates_from_cards(compact)["source_1"].payload[
        "fulfillment_support_sets"
    ][0]["fulfillment_slots"][0]
    return subset_mismatches(
        actual={
            "response_rows": source["response_rows"],
            "prompt_slot_keys": sorted(slot),
            "prompt_slot_group_key_evidence": slot.get("group_key_evidence") or [],
            "parser_slot_keys": sorted(parser_slot),
            "parser_slot_group_key_evidence": (
                parser_slot.get("group_key_evidence") or []
            ),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _response_row_field_id(field: dict[str, Any] | str) -> str:
    if isinstance(field, str):
        return field
    return str(field.get("field_id") or field.get("name") or "")


def _custom_supported_evidence_payload() -> dict[str, Any]:
    return {
        "requested_fact_sources": [
            {
                "requested_fact_id": "fact_1",
                "source_contexts": [
                    {
                        "source_options": [
                            {
                                "source_candidate_id": "source_1",
                                "kind": "new_api_read",
                                "read_id": "read_1",
                                "result_grains": [
                                    {
                                        "grain_id": "root",
                                        "row_path_id": "root",
                                        "cardinality": "one",
                                        "evidence_items": [
                                            {
                                                "evidence_id": "visible_name",
                                                "field_id": "name",
                                            }
                                        ],
                                    }
                                ],
                                "fulfillment_slots": [
                                    {
                                        "fulfillment_slot_id": "slot_1",
                                        "answer_output_id": "answer_1",
                                        "group_key_evidence": [
                                            {
                                                "evidence_id": "visible_name",
                                                "field_id": "name",
                                            }
                                        ],
                                        "unknown_evidence_key": [
                                            {
                                                "evidence_id": "hidden_label",
                                                "field_id": "label",
                                            }
                                        ],
                                    }
                                ],
                                "fulfillment_support_sets": [
                                    {
                                        "fulfillment_support_set_id": "support_1",
                                        "answer_output_id": "answer_1",
                                        "fulfillment_slots": [
                                            {"fulfillment_slot_id": "slot_1"}
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _custom_supported_evidence_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="read_1",
                endpoint_name="read_1",
                row_paths=(
                    RowPath(
                        id="root",
                        path="root",
                        cardinality=RowCardinality.ONE,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="read_1.name",
                        path="root.name",
                        type="string",
                        row_path_id="root",
                    ),
                    CatalogField(
                        ref="read_1.label",
                        path="root.label",
                        type="string",
                        row_path_id="root",
                    ),
                ),
            ),
        ),
    )


def run_source_binding_metric_fit_surface_case(payload: dict[str, Any]) -> list[str]:
    request = _source_binding_request(payload["input"].get("request") or {})
    surface = SourceBindingTurnPrompt(request).metric_fit_surface_payload()
    metric_surfaces = surface["requested_fact_metric_fit_surface"]
    actual = {
        "surface_keys": sorted(surface),
        "metric_ids_by_fact": {
            item["requested_fact_id"]: [
                candidate["metric_evidence_id"]
                for candidate in item.get("metric_candidates") or ()
            ]
            for item in metric_surfaces
        },
        "metric_id_count_by_fact": {
            item["requested_fact_id"]: len(item.get("metric_candidates") or ())
            for item in metric_surfaces
        },
        "metric_contexts": surface.get("metric_contexts") or [],
        "stable_metric_contexts": [
            {
                "read_id": context.get("read_id"),
                "row_path_id": context.get("row_path_id"),
                "same_row_field_paths": context.get("same_row_field_paths"),
            }
            for context in surface.get("metric_contexts") or ()
            if isinstance(context, dict)
        ],
        "metric_candidates": [
            candidate
            for item in metric_surfaces
            for candidate in item.get("metric_candidates") or ()
        ],
        "stable_metric_candidates": [
            {
                "read_id": candidate.get("read_id"),
                "field_path": candidate.get("field_path"),
                "field_type": candidate.get("field_type"),
                "resource_names": candidate.get("resource_names"),
            }
            for item in metric_surfaces
            for candidate in item.get("metric_candidates") or ()
            if isinstance(candidate, dict)
        ],
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_source_binding_schema_surface_case(payload: dict[str, Any]) -> list[str]:
    request = _prompt_surface_request(payload["input"].get("request") or {})
    prompt = SourceBindingTurnPrompt(request)
    candidate = _only_candidate(prompt.source_invocation_candidate_payload())
    schema = prompt.response_contract().provider_schema
    outcome = _schema_surface_outcome(
        request,
        candidate,
        include_default_decision=bool(payload["input"].get("include_default_decision")),
        include_response_shape_decision=bool(
            payload["input"].get("include_response_shape_decision")
        ),
    )
    if payload["input"].get("empty_fulfillment_decisions"):
        next(
            invocation
            for field_id, fact in outcome.items()
            if source_binding_fact_id_from_field(field_id) is not None
            for role_id, invocation in fact.items()
            if role_id != "plan_shape"
        )["fulfillment_decisions"] = {}
    try:
        validate(instance={"outcome": outcome}, schema=schema)
        validation_result = "valid"
    except ValidationError:
        validation_result = "invalid"
    parser_validation = "not_run"
    try:
        if validation_result == "valid":
            parse_source_binding({"outcome": outcome}, request=request)
            parser_validation = "valid"
    except ValueError:
        parser_validation = "invalid"
    fulfillment_schema = _first_fulfillment_decisions_schema(schema)
    return subset_mismatches(
        actual={
            "validation": validation_result,
            "parser_validation": parser_validation,
            "schema_excludes": {
                "minProperties": "minProperties" not in repr(schema),
            },
            "fulfillment_anyof_branches_are_strict": bool(
                fulfillment_schema.get("anyOf")
            )
            and all(
                isinstance(branch, dict)
                and branch.get("type") == "object"
                and branch.get("additionalProperties") is False
                for branch in fulfillment_schema.get("anyOf") or ()
            ),
            "compact_invocation_shape": not bool(fulfillment_schema.get("anyOf")),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_source_binding_row_predicate_parse_case(payload: dict[str, Any]) -> list[str]:
    request = _source_binding_request(
        payload["input"].get("request") or {"mode": "boolean_row_predicate"}
    )
    prompt = SourceBindingTurnPrompt(request)
    candidate = _only_candidate(prompt.source_invocation_candidate_payload())
    outcome = _source_binding_outcome(
        request,
        candidate,
        row_predicate_reviews=(
            {}
            if payload["input"].get("omit_row_predicate_reviews")
            else _row_predicate_reviews_from_case(
                candidate,
                payload["input"].get("row_predicates") or {},
            )
        ),
    )
    model_payload = {"outcome": outcome}
    schema = prompt.response_contract().provider_schema
    try:
        validate(instance=model_payload, schema=schema)
        result = parse_source_binding(model_payload, request=request)
    except (ValidationError, ValueError) as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    bound_source = result.outcome.bound_sources[0]
    return subset_mismatches(
        actual={
            "population_choices": _population_choices(bound_source.source),
            "available_field_ids": list(bound_source.available_field_ids),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_source_binding_row_predicate_schema_case(payload: dict[str, Any]) -> list[str]:
    request = _boolean_row_predicate_request()
    prompt = SourceBindingTurnPrompt(request)
    candidate = _only_candidate(prompt.source_invocation_candidate_payload())
    outcome = _source_binding_outcome(
        request,
        candidate,
        row_predicate_reviews=(
            {}
            if payload["input"].get("omit_row_predicate_reviews")
            else _row_predicate_reviews_from_case(
                candidate,
                payload["input"].get("row_predicates") or {},
            )
        ),
    )
    try:
        validate(
            instance={"outcome": outcome},
            schema=prompt.response_contract().provider_schema,
        )
    except ValidationError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected validation error: {exc.message}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    return []


def run_source_binding_finite_choice_parse_case(payload: dict[str, Any]) -> list[str]:
    request = _source_binding_request(payload["input"].get("request") or {})
    prompt = SourceBindingTurnPrompt(request)
    candidate = _only_candidate(prompt.source_invocation_candidate_payload())
    outcome = _source_binding_outcome(
        request,
        candidate,
        finite_choice_param_reviews={
            "status": _finite_choice_review_from_case(
                payload["input"],
            )
        },
    )
    model_payload = {"outcome": outcome}
    try:
        validate(
            instance=model_payload,
            schema=prompt.response_contract().provider_schema,
        )
        result = parse_source_binding(model_payload, request=request)
    except ValidationError as exc:
        error_text = f"schema validation failed: {exc.message}"
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {error_text}"]
    except ValueError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    bound_source = result.outcome.bound_sources[0]
    param_values = [
        {
            binding.param_id: binding.compiler_value
            for binding in source_invocation.param_bindings
        }
        for source_invocation in bound_source.source_invocations
    ]
    return subset_mismatches(
        actual={
            "source_invocation_param_values": param_values,
            "source_invocation_param_names": [sorted(item) for item in param_values],
            "population_choices": _population_choices(bound_source.source),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _population_choices(source: object) -> list[dict[str, object]]:
    return [
        {
            "controller_kind": item.controller_kind,
            "controller_id": item.controller_id,
            "field_id": item.field_id,
            "requested_fact_ids": list(item.requested_fact_ids),
            "included_values": list(item.included_values),
            "excluded_values": list(item.excluded_values),
            "review_scope_decisions": [
                {
                    "membership_test_id": decision.membership_test_id,
                    "decision": decision.decision.value,
                    "axis_kind": decision.axis_kind,
                    "axis_id": decision.axis_id,
                    "owner_surface_ids": list(decision.owner_surface_ids),
                    "proof_refs": list(decision.proof_refs),
                }
                for decision in item.review_scope_decisions
            ],
        }
        for item in getattr(source, "population_choices", ())
    ]


def run_source_binding_metric_fit_parse_case(payload: dict[str, Any]) -> list[str]:
    request = _source_binding_request(
        payload["input"].get("request") or {},
        selected_basis=str(payload["input"].get("selected_basis") or ""),
        include_secondary_metric=bool(payload["input"].get("include_secondary_metric")),
    )
    prompt = SourceBindingTurnPrompt(request)
    if (
        _request_mode(payload["input"].get("request") or {})
        == "reused_answer_output_metric_support"
    ):
        return _run_reused_answer_output_metric_fit_parse(
            request=request,
            payload=payload,
        )
    candidate = _only_candidate(prompt.source_invocation_candidate_payload())
    selected_metric_field = str(
        payload["input"].get("selected_metric_field") or "amount"
    )
    outcome = _source_binding_outcome(
        request,
        candidate,
        fulfillment_decisions=_fulfillment_decisions(
            candidate,
            field_id=selected_metric_field,
        ),
        finite_choice_param_reviews=_finite_choice_reviews_for_candidate(
            candidate,
            payload["input"].get("choices") or {},
        ),
    )
    _set_metric_fit_from_case(
        outcome,
        candidate,
        payload["input"].get("metric_decisions") or {},
    )
    _set_raw_metric_fit_from_case(
        outcome,
        payload["input"].get("raw_metric_evidence_decisions") or {},
    )
    model_payload = {"outcome": outcome}
    try:
        validate(
            instance=model_payload,
            schema=prompt.response_contract().provider_schema,
        )
        result = parse_source_binding(model_payload, request=request)
    except (ValidationError, ValueError) as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    bound_source = result.outcome.bound_sources[0]
    fulfillment = bound_source.fulfillments[0]
    return subset_mismatches(
        actual={
            "metric_measure_field_ids": _field_ids_for_evidence_ids(
                candidate,
                fulfillment.metric_measure_evidence_ids,
            ),
            "row_count_basis_evidence_ids": list(
                fulfillment.row_count_basis_evidence_ids
            ),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_source_binding_parse_case(payload: dict[str, Any]) -> list[str]:
    request = _source_binding_request(payload["input"].get("request") or {})
    prompt = SourceBindingTurnPrompt(request)
    candidate = _only_candidate(prompt.source_invocation_candidate_payload())
    selected_field_id = str(payload["input"].get("selected_field_id") or "amount")
    if payload["input"].get("invalid_impossible_requested_fact"):
        model_payload = _invalid_impossible_payload(
            requested_fact_id=str(
                payload["input"]["invalid_impossible_requested_fact"]
            ),
        )
        return _parse_invalid_source_binding_payload(
            model_payload,
            request=request,
            schema=prompt.response_contract().provider_schema,
            expected_subset=payload["expect"]["result_contains"],
        )
    model_payload = {
        "outcome": _source_binding_outcome(
            request,
            candidate,
            fulfillment_decisions=_fulfillment_decisions(
                candidate,
                field_id=selected_field_id,
            ),
            finite_choice_param_reviews=_finite_choice_reviews_for_candidate(
                candidate,
                payload["input"].get("choices") or {},
            ),
        )
    }
    if payload["input"].get("duplicate_source_invocation") is True:
        fact_binding = next(
            fact_binding
            for field_id, fact_binding in model_payload["outcome"].items()
            if source_binding_fact_id_from_field(field_id) is not None
        )
        invocation = next(
            invocation
            for role_id, invocation in fact_binding.items()
            if role_id != "plan_shape"
        )
        fact_binding["duplicate"] = dict(invocation)
    try:
        validate(
            instance=model_payload,
            schema=prompt.response_contract().provider_schema,
        )
        result = parse_source_binding(model_payload, request=request)
    except (ValidationError, ValueError) as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    bound_source = result.outcome.bound_sources[0]
    return subset_mismatches(
        actual={
            "source_row_source_id": (
                bound_source.source.row_source_id
                if bound_source.source is not None
                else ""
            ),
            "applied_filters": list(bound_source.applied_filters),
            "source_param_bindings": [
                {
                    "param_id": binding.param_id,
                    "value": binding.value,
                    "origin_kind": binding.origin_kind.value,
                    "value_id": binding.value_id,
                }
                for binding in (
                    bound_source.source.param_bindings
                    if bound_source.source is not None
                    else ()
                )
            ],
            "available_field_ids": list(bound_source.available_field_ids),
            "fulfillment_evidence_ids": (
                list(bound_source.fulfillments[0].all_evidence_ids())
                if bound_source.fulfillments
                else []
            ),
            "transport_grounded_values": prompt.transport_context_payload().get(
                "grounded_values",
                {},
            ),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _invalid_impossible_payload(*, requested_fact_id: str) -> dict[str, Any]:
    return {
        "outcome": {
            "kind": "impossible",
            "blocked_facts": [
                {
                    "requested_fact_id": requested_fact_id,
                    "basis": "policy_access",
                    "evidence_refs": ["source_1.amount"],
                }
            ],
        }
    }


def _parse_invalid_source_binding_payload(
    model_payload: dict[str, Any],
    *,
    request: SourceBindingRequest,
    schema: dict[str, Any],
    expected_subset: dict[str, Any],
) -> list[str]:
    try:
        validate(instance=model_payload, schema=schema)
        schema_validation = "valid"
        schema_error = ""
    except ValidationError as exc:
        schema_validation = "invalid"
        schema_error = exc.message
    try:
        parse_source_binding(model_payload, request=request)
        parser_validation = "valid"
        parser_error = ""
    except ValueError as exc:
        parser_validation = "invalid"
        parser_error = str(exc)
    return subset_mismatches(
        actual={
            "schema_validation": schema_validation,
            "schema_error": schema_error,
            "parser_validation": parser_validation,
            "parser_error": parser_error,
        },
        expected_subset=expected_subset,
    )


def _request_mode(payload: object) -> str:
    data = payload if isinstance(payload, dict) else {}
    return str(data.get("mode") or "choice_param")


def _run_reused_answer_output_metric_fit_parse(
    *,
    request: SourceBindingRequest,
    payload: dict[str, Any],
) -> list[str]:
    prompt = SourceBindingTurnPrompt(request)
    prompt_payload = prompt.source_invocation_candidate_payload()
    candidates = {
        (requested_fact_id, candidate["read_id"]): candidate
        for requested_fact_id, candidate in _source_candidates_with_fact(prompt_payload)
    }
    sales_candidate = candidates[("fact_sales", "sales")]
    payments_candidate = candidates[("fact_payments", "payments")]
    outcome = {
        "kind": "source_bindings",
        "metric_fit_bases": {
            "fact_sales": {
                "source_1.data.amount": {
                    "metric_meaning": "amount is sales amount.",
                    "fit_basis": "amount fits sales amount.",
                }
            },
            "fact_payments": {
                "source_2.data.amount": {
                    "metric_meaning": "amount is payment amount.",
                    "fit_basis": "amount fits payment amount.",
                }
            },
        },
        "fit_basis_interpretations": {
            "fact_sales": {
                "source_1.data.amount": {
                    "interpretation": "FITS_REQUESTED_ANSWER",
                }
            },
            "fact_payments": {
                "source_2.data.amount": {
                    "interpretation": "FITS_REQUESTED_ANSWER",
                }
            },
        },
        **{
            source_binding_fact_field_id(requested_fact_id): {
                "plan_shape": _binding_target_for_candidate(
                    request,
                    requested_fact_id=requested_fact_id,
                    source_candidate_id=str(candidate["source_candidate_id"]),
                ).plan_shape,
                _binding_target_for_candidate(
                        request,
                        requested_fact_id=requested_fact_id,
                        source_candidate_id=str(candidate["source_candidate_id"]),
                    ).requirement_id: _source_invocation_for_metric_candidate(
                        request,
                        candidate,
                        requested_fact_id=requested_fact_id,
                    )
            }
            for requested_fact_id, candidate in (
                ("fact_sales", sales_candidate),
                ("fact_payments", payments_candidate),
            )
        },
    }
    model_payload = {"outcome": outcome}
    try:
        validate(
            instance=model_payload,
            schema=prompt.response_contract().provider_schema,
        )
        result = parse_source_binding(model_payload, request=request)
    except (ValidationError, ValueError) as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    return subset_mismatches(
        actual={
            "metric_measure_evidence_ids_by_fact": {
                source.requested_fact_id: list(
                    source.fulfillments[0].metric_measure_evidence_ids
                )
                for source in result.outcome.bound_sources
            }
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _source_invocation_for_metric_candidate(
    request: SourceBindingRequest,
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
) -> dict[str, Any]:
    return {
        "binding_target_id": _binding_target_id_for_candidate(
            request,
            requested_fact_id=requested_fact_id,
            source_candidate_id=str(candidate["source_candidate_id"]),
        ),
        "answer_population": {
            "population_binding_id": _binding_surface(candidate)["population_bindings"][
                0
            ]["population_binding_id"],
            "intent_text": f"{requested_fact_id} population",
            "match_basis_explanation": "The selected source matches this requested fact.",
        },
        "fulfillment_decisions": _fulfillment_decisions(candidate, field_id="amount"),
        "param_decisions": {},
        "row_predicate_reviews": {},
        "finite_choice_param_reviews": {},
    }


def _boolean_row_predicate_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="active sales",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sale",),
                fields=(
                    CatalogField(
                        ref="sales.field.amount",
                        path="amount",
                        type="decimal",
                    ),
                    CatalogField(
                        ref="sales.field.is_active",
                        path="is_active",
                        type="boolean",
                    ),
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales",),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
        ),
        selected_read_ids=("sales",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many active sales happened?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="How many active sales happened?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.direct_field_value.1",
                    plan_shape="direct_field_value",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        SourceStrategyMember(
                            source_candidate_id=scope.source_candidate_id,
                        ),
                    ),
                    basis="Selected by conformance fixture.",
                ),
            )
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="sales",
                    relevant_row_path_ids=("root",),
                ),
            )
        ),
    )


def _prompt_surface_request(payload: object) -> SourceBindingRequest:
    return _source_binding_request(payload)


def _plan_family_request() -> SourceBindingRequest:
    base = _choice_param_request()
    fact = replace(
        base.requested_facts[0],
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.LIST_ROWS,
        ),
    )
    member = base.plan_selection.plan_selections[0].source_members[0]
    plans = (
        SelectedSourceStrategy(
            plan_selection_id="plan.fact_1.list_rows",
            requested_fact_id="fact_1",
            source_strategy_id="strategy.fact_1.list_rows",
            plan_shape="list_rows",
            required_answer_output_ids=("answer_1",),
            source_members=(replace(member, requirement_ids=("primary",)),),
            basis="One-role conformance plan.",
        ),
        SelectedSourceStrategy(
            plan_selection_id="plan.fact_1.grouped_rows",
            requested_fact_id="fact_1",
            source_strategy_id="strategy.fact_1.grouped_rows",
            plan_shape="grouped_rows",
            required_answer_output_ids=("answer_1",),
            source_members=(
                replace(
                    member,
                    requirement_ids=("primary", "group_identity"),
                ),
            ),
            basis="Two-role conformance plan.",
        ),
    )
    return replace(
        base,
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        plan_selection=PlanSelectionSet(plan_selections=plans),
    )


def _source_binding_request(
    payload: object,
    *,
    selected_basis: str = "",
    include_secondary_metric: bool = False,
) -> SourceBindingRequest:
    data = payload if isinstance(payload, dict) else {}
    mode = str(data.get("mode") or "choice_param")
    if mode == "boolean_row_predicate":
        return _boolean_row_predicate_request()
    if mode == "nested_population_roles":
        return _nested_population_roles_request()
    if mode == "selected_source_members":
        return _source_filter_request(filter_basis="plan_selection")
    if mode == "read_eligibility_filter":
        return _source_filter_request(filter_basis="read_eligibility")
    if mode == "multi_answer_outputs":
        return _choice_param_request(answer_output_ids=("answer_1", "answer_2"))
    if mode == "reused_answer_output_metric_support":
        return _reused_answer_output_metric_support_request()
    if mode == "filtered_response_shape_variant":
        return _filtered_response_shape_variant_request()
    if mode == "source_default_param_after_read_eligibility":
        return _source_default_param_after_read_eligibility_request()
    if mode == "multi_row_summary_metric":
        return _multi_row_summary_metric_request()
    if mode == "multi_row_summary_ranked_metric":
        return _multi_row_summary_metric_request(plan_shape="ranked_aggregate")
    if mode == "optional_population_params":
        return _optional_population_params_request()
    if mode == "grounded_time_filter":
        return _grounded_time_filter_request(
            start_param_name=str(data.get("raw_start_param_name") or "start_date")
        )
    if mode == "scoped_review_owned_input":
        return _scoped_review_owned_input_request()
    if mode == "scoped_review_owned_input_unbound":
        return _scoped_review_owned_input_request(
            bind_owned_input=False,
            include_available_staff_value=False,
        )
    if mode == "scoped_review_selectable_owned_inputs":
        return _scoped_review_owned_input_request(
            bind_owned_input=False,
            include_time_input=True,
        )
    if mode == "scoped_review_owned_input_choice_only":
        return _scoped_review_owned_input_request(include_row_predicate=False)
    if mode == "scoped_review_owned_input_row_only":
        return _scoped_review_owned_input_request(include_finite_choice=False)
    if mode == "scoped_review_axis_owned_test":
        return _scoped_review_owned_input_request(include_axis_owned_test=True)
    if mode == "identity_field_filter":
        return _identity_field_filter_request()
    if mode == "same_scope_memory":
        return _same_scope_memory_request()
    if mode == "yaml_prompt_surface":
        return _yaml_prompt_surface_request(data)
    if selected_basis == "row_count":
        return _row_count_request()
    return _choice_param_request(
        default_choice_param=bool(data.get("default_choice_param")),
        include_extra_field=bool(data.get("include_extra_field")),
        include_boolean_response_field=bool(data.get("include_boolean_response_field")),
        include_secondary_metric=include_secondary_metric,
        response_shape_choice_param=bool(data.get("response_shape_choice_param")),
    )


def _nested_population_roles_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="orders placed today",
        answer_subject=RequestedFactAnswerSubject(subject_text="orders"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="orders",
                endpoint_name="list_order_list",
                resource_names=("order",),
                row_paths=(
                    RowPath(
                        id="orders",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                    RowPath(
                        id="items",
                        path="data.items",
                        cardinality=RowCardinality.MANY,
                        parent_path="data",
                    ),
                ),
                params=(
                    CatalogParam(
                        ref="orders.query.status",
                        name="status",
                        source=ParamSource.QUERY,
                        type="choice",
                        choices=("OPEN", "CLOSED"),
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="orders.field.order_id",
                        path="data.order_id",
                        row_path_id="orders",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="orders.field.status",
                        path="data.status",
                        row_path_id="orders",
                        type="choice",
                        choices=("OPEN", "CLOSED"),
                    ),
                    CatalogField(
                        ref="orders.field.line_item_id",
                        path="data.items.line_item_id",
                        row_path_id="items",
                        type="uuid",
                    ),
                ),
            ),
        )
    )
    catalog_selection = _single_fact_catalog_selection(catalog, read_ids=("orders",))
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many orders happened today?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="How many orders happened today?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(scope.source_candidate_id,),
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="orders",
                    relevant_row_path_ids=("orders",),
                ),
            )
        ),
    )


def _same_scope_memory_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sales in the same scoped location",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="sales_read",
                resource_names=("sale",),
                params=(
                    CatalogParam(
                        ref="sales_read.query.location_id",
                        name="location_id",
                        source=ParamSource.QUERY,
                        type="uuid",
                    ),
                ),
                row_paths=(
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sales.field.sale_id",
                        path="data.sale_id",
                        row_path_id="data",
                        type="uuid",
                        identity=IdentityMetadata(
                            entity_ref="sale",
                            identity_field="sale_id",
                            primary_key=True,
                            stable=True,
                        ),
                    ),
                    CatalogField(
                        ref="sales.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            ),
        )
    )
    source_candidate_id = "source_1"
    return SourceBindingRequest(
        question="Show sales for the same location.",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=_single_fact_catalog_selection(catalog, read_ids=()),
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(source_candidate_id,),
        ),
        memory_inputs={
            "memoryRelations": [
                {
                    "id": "prior_sales",
                    "fields": [
                        {"field_id": "amount", "type": "decimal"},
                    ],
                    "completeness": {
                        "status": "complete",
                        "proofRefs": ["read:sales_read"],
                        "scopeFingerprint": json.dumps(
                            {
                                "endpointArgs": {
                                    "sales_read.query.location_id": "loc_westlands"
                                },
                                "endpointArgProofRefs": {
                                    "sales_read.query.location_id": [
                                        "known_input:location_1"
                                    ]
                                },
                                "rowFilters": [],
                            },
                            sort_keys=True,
                        ),
                    },
                }
            ]
        },
    )


def _yaml_prompt_surface_request(payload: dict[str, Any]) -> SourceBindingRequest:
    fact = _requested_fact_from_payload(payload["requested_fact"])
    catalog = catalog_from_payload(payload["relation_catalog"])
    same_scope_catalog = (
        catalog_from_payload(payload["same_scope_relation_catalog"])
        if payload.get("same_scope_relation_catalog")
        else None
    )
    catalog_selection = _single_fact_catalog_selection(
        catalog,
        read_ids=tuple(payload.get("selected_read_ids") or ()),
    )
    scopes_by_read_id = {
        scope.read_id: scope
        for scope in read_eligibility_candidate_surface(
            ReadEligibilityRequest(
                question=str(payload["question"]),
                question_contract=QuestionContract(requested_facts=(fact,)),
                requested_facts=(fact,),
                catalog_selection=catalog_selection,
                conversation_context={},
                available_values=(),
            )
        ).candidate_scopes
    }
    retained_read_ids = tuple(payload.get("retained_read_ids") or ())
    retained_field_refs_by_read_id = {
        str(read_id): tuple(str(ref) for ref in refs if str(ref))
        for read_id, refs in (
            payload.get("retained_field_refs_by_read_id") or {}
        ).items()
        if isinstance(refs, list)
    }
    retained_row_path_ids_by_read_id = {
        str(read_id): tuple(
            str(row_path_id) for row_path_id in row_path_ids if str(row_path_id)
        )
        for read_id, row_path_ids in (
            payload.get("retained_row_path_ids_by_read_id") or {}
        ).items()
        if isinstance(row_path_ids, list)
    }
    return SourceBindingRequest(
        question=str(payload["question"]),
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        same_scope_relation_catalog=same_scope_catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id=fact.id,
            source_candidate_ids=tuple(
                f"source_{index}"
                for index in range(
                    1,
                    int(payload.get("selected_source_candidate_count") or 1) + 1,
                )
            ),
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=tuple(
                _read_assessment(
                    scope=scopes_by_read_id[read_id],
                    requested_fact_id=fact.id,
                    read_id=read_id,
                    relevant_row_path_ids=retained_row_path_ids_by_read_id.get(
                        read_id,
                        ("root",),
                    ),
                    relevant_field_refs=retained_field_refs_by_read_id.get(read_id),
                )
                for read_id in retained_read_ids
            )
        ),
        memory_inputs=dict(payload.get("memory_inputs") or {}),
    )


def _requested_fact_from_payload(payload: dict[str, Any]) -> RequestedFact:
    return RequestedFact(
        id=str(payload.get("id") or "fact_1"),
        description=str(payload["description"]),
        answer_subject=RequestedFactAnswerSubject(
            subject_text=str(payload.get("answer_subject") or "")
        ),
        answer_outputs=tuple(
            RequestedFactAnswerOutput(id=str(answer_output_id))
            for answer_output_id in payload.get("answer_outputs") or ("answer_1",)
        ),
    )


def _source_filter_request(*, filter_basis: str) -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sales",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    catalog = RelationCatalog(
        reads=(
            _single_metric_read(read_id="sales", resource_name="sale"),
            _single_metric_read(read_id="refunds", resource_name="refund"),
        )
    )
    catalog_selection = _single_fact_catalog_selection(
        catalog,
        read_ids=("sales", "refunds"),
    )
    scopes = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales happened?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes
    sales_scope = scopes[0]
    refund_scope = scopes[1]
    selected_source_ids = (
        (sales_scope.source_candidate_id,)
        if filter_basis == "plan_selection"
        else (sales_scope.source_candidate_id, refund_scope.source_candidate_id)
    )
    read_assessments = (
        _read_assessment(
            scope=sales_scope,
            requested_fact_id="fact_1",
            read_id="sales",
        ),
        (
            _read_assessment(
                scope=refund_scope,
                requested_fact_id="fact_1",
                read_id="refunds",
            )
            if filter_basis == "plan_selection"
            else ReadAssessment(
                source_candidate_id=refund_scope.source_candidate_id,
                source_candidate_signature=refund_scope.source_candidate_signature,
                requested_fact_id="fact_1",
                read_id="refunds",
                relevant_row_path_ids=(),
                relevant_field_refs=(),
                retention_decision="DROP",
                retention_basis="Refunds are not sales.",
            )
        ),
    )
    return SourceBindingRequest(
        question="How many sales happened?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=selected_source_ids,
        ),
        read_eligibility=ReadEligibilityResult(read_assessments=read_assessments),
    )


def _single_fact_catalog_selection(
    catalog: RelationCatalog,
    *,
    read_ids: tuple[str, ...],
) -> CatalogSelectionResult:
    return CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales",),
                rankings=tuple(
                    CatalogSelectionRanking(read_id=read_id, score=10)
                    for read_id in read_ids
                ),
                selected_read_ids=read_ids,
            ),
        ),
        selected_read_ids=read_ids,
    )


def _single_metric_read(*, read_id: str, resource_name: str) -> EndpointRead:
    return EndpointRead(
        id=read_id,
        endpoint_name=f"list_{resource_name}_list",
        resource_names=(resource_name,),
        fields=(
            CatalogField(
                ref=f"{read_id}.field.amount",
                path="amount",
                type="decimal",
            ),
        ),
    )


def _read_assessment(
    *,
    scope: Any,
    requested_fact_id: str,
    read_id: str,
    relevant_row_path_ids: tuple[str, ...] = ("root",),
    relevant_field_refs: tuple[str, ...] | None = None,
) -> ReadAssessment:
    return ReadAssessment(
        source_candidate_id=scope.source_candidate_id,
        source_candidate_signature=scope.source_candidate_signature,
        requested_fact_id=requested_fact_id,
        read_id=read_id,
        relevant_row_path_ids=relevant_row_path_ids,
        relevant_field_refs=(
            relevant_field_refs
            if relevant_field_refs is not None
            else tuple(scope.field_refs_by_evidence_token.values())
        ),
        retention_decision="RETAIN",
        retention_basis=f"{read_id} rows can answer the requested fact.",
    )


def _plan_for_sources(
    *,
    requested_fact_id: str,
    source_candidate_ids: tuple[str, ...],
    plan_shape: str = "direct_field_value",
) -> PlanSelectionSet:
    return PlanSelectionSet(
        plan_selections=(
            SelectedSourceStrategy(
                plan_selection_id=f"plan.{requested_fact_id}",
                requested_fact_id=requested_fact_id,
                source_strategy_id=(
                    f"source_strategy.{requested_fact_id}.{plan_shape}.1"
                ),
                plan_shape=plan_shape,
                required_answer_output_ids=("answer_1",),
                source_members=tuple(
                    SourceStrategyMember(source_candidate_id=source_candidate_id)
                    for source_candidate_id in source_candidate_ids
                ),
                basis="Selected by conformance fixture.",
            ),
        )
    )


def _choice_param_request(
    *,
    answer_output_ids: tuple[str, ...] = ("answer_1",),
    include_secondary_metric: bool = False,
    include_extra_field: bool = False,
    default_choice_param: bool = False,
    include_boolean_response_field: bool = False,
    response_shape_choice_param: bool = False,
) -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sales",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=tuple(
            RequestedFactAnswerOutput(id=answer_output_id)
            for answer_output_id in answer_output_ids
        ),
    )
    metric_fields = [
        CatalogField(
            ref="sales.field.amount",
            path="amount",
            type="decimal",
        ),
    ]
    if include_secondary_metric:
        metric_fields.append(
            CatalogField(
                ref="sales.field.secondary_amount",
                path="secondary_amount",
                type="decimal",
            )
        )
    if include_extra_field:
        metric_fields.append(
            CatalogField(
                ref="sales.field.unrelated",
                path="unrelated",
                type="string",
            )
        )
    if include_boolean_response_field:
        metric_fields.append(
            CatalogField(
                ref="sales.field.is_active",
                path="is_active",
                type="boolean",
            )
        )
    params = [
        CatalogParam(
            ref="sales.query.status",
            name="status",
            source=ParamSource.QUERY,
            type="choice",
            choices=("OPEN", "CLOSED"),
            choice_labels={"OPEN": "Open", "CLOSED": "Closed"},
        ),
    ]
    if include_boolean_response_field:
        params.append(
            CatalogParam(
                ref="sales.query.is_active",
                name="is_active",
                source=ParamSource.QUERY,
                type="boolean",
            )
        )
    if default_choice_param:
        params.append(
            CatalogParam(
                ref="sales.query.granularity",
                name="granularity",
                source=ParamSource.QUERY,
                type="choice",
                choices=("day", "month"),
                choice_labels={"day": "Day", "month": "Month"},
                default="day",
            )
        )
    if response_shape_choice_param:
        params.append(
            CatalogParam(
                ref="sales.query.ordering",
                name="ordering",
                source=ParamSource.QUERY,
                type="choice",
                choices=("created_at", "-created_at"),
                choice_labels={
                    "created_at": "Created At",
                    "-created_at": "Created At",
                },
            )
        )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sale",),
                params=tuple(params),
                fields=(
                    *metric_fields,
                    CatalogField(
                        ref="sales.field.status",
                        path="status",
                        type="choice",
                        choices=("OPEN", "CLOSED"),
                    ),
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales",),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
        ),
        selected_read_ids=("sales",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales happened?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="How many sales happened?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.direct_field_value.1",
                    plan_shape="direct_field_value",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        SourceStrategyMember(
                            source_candidate_id=scope.source_candidate_id,
                        ),
                    ),
                    basis="Selected by conformance fixture.",
                ),
            )
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="sales",
                    relevant_row_path_ids=("root",),
                ),
            )
        ),
    )


def _scoped_review_owned_input_request(
    *,
    include_finite_choice: bool = True,
    include_row_predicate: bool = True,
    bind_owned_input: bool = True,
    include_axis_owned_test: bool = False,
    include_time_input: bool = False,
    include_reference_param: bool = True,
    include_available_staff_value: bool = True,
) -> SourceBindingRequest:
    staff_input = RequestedFactLiteralInput(
        id="staff_id_1",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="staff_id: 51515151-0000-0000-0002-000000000001",
        resolved_value_text="51515151-0000-0000-0002-000000000001",
        field_label_text="staff_id",
        value_meaning_hint="staff member",
        role=LiteralInputRole.REFERENCE_VALUE,
    )
    time_input = RequestedFactLiteralInput(
        id="time_1",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="today",
        resolved_value_text="today",
        role=LiteralInputRole.TIME_VALUE,
    )
    membership_tests: list[RequestedFactAnswerPopulationMembershipTest] = [
        RequestedFactAnswerPopulationMembershipTest(
            id="subject_identity",
            kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
            polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
            test_question="Does the row/value represent sales?",
        ),
        RequestedFactAnswerPopulationMembershipTest(
            id="specified_staff",
            kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
            polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
            test_question=(
                "Does the sale belong to the staff member identified by the "
                "question input?"
            ),
            owned_question_input_refs=("staff_id_1",),
        ),
    ]
    if include_time_input:
        membership_tests.append(
            RequestedFactAnswerPopulationMembershipTest(
                id="today_constraint",
                kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
                polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                test_question="Did the sale happen during the requested time span?",
                owned_question_input_refs=("time_1",),
            )
        )
    if include_axis_owned_test:
        membership_tests.append(
            RequestedFactAnswerPopulationMembershipTest(
                id="sale_type_constraint",
                kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
                polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                test_question="Does the sale have the requested sale type?",
            )
        )
    membership_tests.append(
        RequestedFactAnswerPopulationMembershipTest(
            id="normal_instance_guard",
            kind=AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD,
            polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
            test_question="Is this an ordinary business instance of sales?",
        )
    )
    fact = RequestedFact(
        id="fact_1",
        description="sales for one staff member",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_population=RequestedFactAnswerPopulation(
            population_label="sales for one staff member",
            counted_unit="sales",
            membership_tests=tuple(membership_tests),
        ),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
        known_inputs=(staff_input, *((time_input,) if include_time_input else ())),
        input_refs=("staff_id_1", *(("time_1",) if include_time_input else ())),
    )
    available_values = (
        *(
            (
                FactValue.identity(
                    id="staff_identity_1",
                    known_input_id="staff_id_1",
                    identity_type="staff",
                    identity_field="staff_id",
                    value="51515151-0000-0000-0002-000000000001",
                    display_value="51515151-0000-0000-0002-000000000001",
                    proof_refs=("known_input:staff_id_1", "read:staff_lookup"),
                    applies_to_requested_fact_ids=("fact_1",),
                ),
            )
            if include_available_staff_value
            else ()
        ),
        *(
            (
                FactValue.time(
                    id="time_1",
                    known_input_id="time_1",
                    expression="today",
                    resolved_start="2026-07-06",
                    resolved_end="2026-07-07",
                    granularity="day",
                    proof_refs=("known_input:time_1",),
                    applies_to_requested_fact_ids=("fact_1",),
                ),
            )
            if include_time_input
            else ()
        ),
    )
    params = []
    if include_reference_param:
        params.append(
            CatalogParam(
                ref="sales.query.staff_id",
                name="staff_id",
                source=ParamSource.QUERY,
                type="uuid",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            )
        )
    if include_time_input:
        params.extend(
            (
                CatalogParam(
                    ref="sales.query.start_date",
                    name="start_date",
                    source=ParamSource.QUERY,
                    type="date",
                ),
                CatalogParam(
                    ref="sales.query.end_date",
                    name="end_date",
                    source=ParamSource.QUERY,
                    type="date",
                ),
            )
        )
    if include_finite_choice:
        params.append(
            CatalogParam(
                ref="sales.query.status",
                name="status",
                source=ParamSource.QUERY,
                type="choice",
                choices=("OPEN", "CLOSED"),
                choice_labels={"OPEN": "Open", "CLOSED": "Closed"},
            )
        )
    if include_axis_owned_test:
        params.append(
            CatalogParam(
                ref="sales.query.sale_type",
                name="sale_type",
                source=ParamSource.QUERY,
                type="choice",
                choices=("STORE", "ONLINE"),
                choice_labels={"STORE": "Store", "ONLINE": "Online"},
            )
        )
    fields: list[CatalogField] = [
        CatalogField(
            ref="sales.field.amount",
            path="amount",
            type="decimal",
        )
    ]
    if bind_owned_input:
        fields.append(
            CatalogField(
                ref="sales.field.staff_id",
                path="staff_id",
                type="uuid",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            )
        )
    if include_finite_choice:
        fields.append(
            CatalogField(
                ref="sales.field.status",
                path="status",
                type="choice",
                choices=("OPEN", "CLOSED"),
            )
        )
    if include_axis_owned_test:
        fields.append(
            CatalogField(
                ref="sales.field.sale_type",
                path="sale_type",
                type="choice",
                choices=("STORE", "ONLINE"),
            )
        )
    if include_row_predicate:
        fields.append(
            CatalogField(
                ref="sales.field.is_deleted",
                path="is_deleted",
                type="boolean",
            )
        )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sale",),
                params=tuple(params),
                fields=tuple(fields),
            ),
        )
    )
    catalog_selection = _single_fact_catalog_selection(catalog, read_ids=("sales",))
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales did the staff member sell?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=available_values,
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="How many sales did the staff member sell?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(scope.source_candidate_id,),
        ),
        available_values=available_values,
        available_value_uses=(
            (
                GroundedInputUse(
                    id="grounded_staff_id",
                    value_id="staff_identity_1",
                    row_source_id=api_row_source_id("sales", "root"),
                    param_id="staff_id",
                ),
            )
            if bind_owned_input
            else ()
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="sales",
                    relevant_row_path_ids=("root",),
                ),
            )
        ),
    )


def _row_count_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sale count",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sale",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="sales.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales",),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
        ),
        selected_read_ids=("sales",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales happened?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="How many sales happened?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                    plan_shape="aggregate_scalar",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        SourceStrategyMember(
                            source_candidate_id=scope.source_candidate_id,
                        ),
                    ),
                    basis="Selected by conformance fixture.",
                ),
            )
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="sales",
                    relevant_row_path_ids=("data",),
                ),
            )
        ),
    )


def _reused_answer_output_metric_support_request() -> SourceBindingRequest:
    facts = (
        RequestedFact(
            id="fact_sales",
            description="total sales amount",
            answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
            answer_outputs=(
                RequestedFactAnswerOutput(
                    id="answer_1",
                    description="sales amount",
                ),
            ),
        ),
        RequestedFact(
            id="fact_payments",
            description="total payment amount",
            answer_subject=RequestedFactAnswerSubject(subject_text="payments"),
            answer_outputs=(
                RequestedFactAnswerOutput(
                    id="answer_1",
                    description="payment amount",
                ),
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sales",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="sales.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            ),
            EndpointRead(
                id="payments",
                endpoint_name="list_payment_list",
                resource_names=("payments",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="payments.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_sales",
                query_terms=("sales",),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
            RequestedFactCatalogSelection(
                requested_fact_id="fact_payments",
                query_terms=("payments",),
                rankings=(CatalogSelectionRanking(read_id="payments", score=10),),
                selected_read_ids=("payments",),
            ),
        ),
        selected_read_ids=("sales", "payments"),
    )
    scopes_by_fact_read = {
        (scope.requested_fact_id, scope.read_id): scope
        for scope in read_eligibility_candidate_surface(
            ReadEligibilityRequest(
                question="Compare sales and payment totals.",
                question_contract=QuestionContract(requested_facts=facts),
                requested_facts=facts,
                catalog_selection=catalog_selection,
                conversation_context={},
                available_values=(),
            )
        ).candidate_scopes
    }
    sales_scope = scopes_by_fact_read[("fact_sales", "sales")]
    payments_scope = scopes_by_fact_read[("fact_payments", "payments")]
    return SourceBindingRequest(
        question="Compare sales and payment totals.",
        question_contract=QuestionContract(requested_facts=facts),
        requested_facts=facts,
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                _selected_source_strategy(
                    requested_fact_id="fact_sales",
                    source_candidate_id=sales_scope.source_candidate_id,
                ),
                _selected_source_strategy(
                    requested_fact_id="fact_payments",
                    source_candidate_id=payments_scope.source_candidate_id,
                ),
            )
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=sales_scope,
                    requested_fact_id="fact_sales",
                    read_id="sales",
                ),
                _read_assessment(
                    scope=payments_scope,
                    requested_fact_id="fact_payments",
                    read_id="payments",
                ),
            )
        ),
    )


def _filtered_response_shape_variant_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sales count by status",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(
            RequestedFactAnswerOutput(id="answer_1", description="sales count"),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="irrelevant_read",
                endpoint_name="irrelevant_read",
                resource_names=("irrelevant",),
                fields=(
                    CatalogField(
                        ref="irrelevant.field.id",
                        path="id",
                        type="string",
                    ),
                ),
            ),
            EndpointRead(
                id="sales_summary",
                endpoint_name="sales_summary",
                resource_names=("sales",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                params=(
                    CatalogParam(
                        ref="sales_summary.query.group_by",
                        name="group_by",
                        source=ParamSource.QUERY,
                        type="choice",
                        required=True,
                        choices=("location", "status"),
                        semantics="response_shape",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sales_summary.field.label",
                        path="data.label",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="sales_summary.field.count",
                        path="data.count",
                        row_path_id="data",
                        type="integer",
                    ),
                ),
            ),
        )
    )
    original_selection = _single_fact_catalog_selection(
        catalog,
        read_ids=("irrelevant_read", "sales_summary"),
    )
    original_surface = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales by status?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=original_selection,
            conversation_context={},
            available_values=(),
        )
    )
    cards = original_surface.card_payload
    status_card = next(
        card
        for group in cards["requested_fact_read_candidates"]
        for card in group["read_candidates"]
        if card["read_id"] == "sales_summary"
        and card.get("bound_params")
        and card["bound_params"][0]["param_id"] == "group_by"
        and card["bound_params"][0]["value"] == "status"
    )
    status_scope = next(
        scope
        for scope in original_surface.candidate_scopes
        if scope.source_candidate_id == status_card["source_candidate_id"]
    )
    filtered_catalog = RelationCatalog(reads=(catalog.read("sales_summary"),))
    filtered_selection = _single_fact_catalog_selection(
        filtered_catalog,
        read_ids=("sales_summary",),
    )
    return SourceBindingRequest(
        question="How many sales by status?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=filtered_catalog,
        catalog_selection=filtered_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(status_scope.source_candidate_id,),
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=status_scope,
                    requested_fact_id="fact_1",
                    read_id="sales_summary",
                ),
            )
        ),
    )


def _source_default_param_after_read_eligibility_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sales summary",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(
            RequestedFactAnswerOutput(id="answer_1", description="sales summary"),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales_summary",
                endpoint_name="sales_summary",
                resource_names=("sales",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                params=(
                    CatalogParam(
                        ref="sales_summary.query.group_by",
                        name="group_by",
                        source=ParamSource.QUERY,
                        type="choice",
                        required=True,
                        choices=("date", "location"),
                        default="date",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sales_summary.field.label",
                        path="data.label",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="sales_summary.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            ),
        )
    )
    catalog_selection = _single_fact_catalog_selection(
        catalog,
        read_ids=("sales_summary",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="Show sales summary.",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="Show sales summary.",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(scope.source_candidate_id,),
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="sales_summary",
                ),
            )
        ),
    )


def _multi_row_summary_metric_request(
    *,
    plan_shape: str = "direct_field_value",
) -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="location with highest sales",
        answer_subject=RequestedFactAnswerSubject(subject_text="location"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales_summary_read",
                endpoint_name="sales_summary_read",
                resource_names=("sales summary",),
                row_paths=(
                    RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                    RowPath(
                        id="summary",
                        path="summary",
                        cardinality=RowCardinality.ONE,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.data.location_id",
                        path="data.location_id",
                        row_path_id="data",
                        type="string",
                        identity=IdentityMetadata(
                            entity_ref="location",
                            identity_field="location_id",
                            primary_key=True,
                            display_fields=("field.data.label",),
                        ),
                    ),
                    CatalogField(
                        ref="field.data.label",
                        path="data.label",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.data.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                    CatalogField(
                        ref="field.summary.total_amount",
                        path="summary.total_amount",
                        row_path_id="summary",
                        type="decimal",
                    ),
                ),
            ),
        )
    )
    catalog_selection = _single_fact_catalog_selection(
        catalog,
        read_ids=("sales_summary_read",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="Which location has the highest sales?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="Which location has the highest sales?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(scope.source_candidate_id,),
            plan_shape=plan_shape,
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                ReadAssessment(
                    source_candidate_id=scope.source_candidate_id,
                    source_candidate_signature=scope.source_candidate_signature,
                    requested_fact_id="fact_1",
                    read_id="sales_summary_read",
                    relevant_row_path_ids=("data",),
                    relevant_field_refs=(
                        "field.data.location_id",
                        "field.data.amount",
                    ),
                    retention_decision="RETAIN",
                    retention_basis="Sales summary rows can rank locations by sales.",
                ),
            )
        ),
    )


def _optional_population_params_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="count of completed store records",
        answer_subject=RequestedFactAnswerSubject(subject_text="records"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records_read",
                endpoint_name="records_read",
                resource_names=("record",),
                params=(
                    CatalogParam(
                        ref="records_read.query.channel",
                        name="channel",
                        source=ParamSource.QUERY,
                        type="choice",
                        choices=("STORE", "ONLINE"),
                    ),
                    CatalogParam(
                        ref="records_read.query.status",
                        name="status",
                        source=ParamSource.QUERY,
                        type="choice",
                        choices=("PLACED", "COMPLETED"),
                    ),
                    CatalogParam(
                        ref="records_read.query.start_date",
                        name="start_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.record_id",
                        path="data.record_id",
                        row_path_id="data",
                        type="uuid",
                        identity=IdentityMetadata(
                            entity_ref="record",
                            identity_field="record_id",
                            primary_key=True,
                        ),
                    ),
                ),
            ),
        )
    )
    catalog_selection = _single_fact_catalog_selection(
        catalog,
        read_ids=("records_read",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many completed store records are there?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="How many completed store records are there?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(scope.source_candidate_id,),
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="records_read",
                    relevant_row_path_ids=("data",),
                ),
            )
        ),
    )


def _grounded_time_filter_request(
    *,
    start_param_name: str = "start_date",
) -> SourceBindingRequest:
    root_row_source_id = api_row_source_id("sales", "root")
    fact = RequestedFact(
        id="fact_1",
        description="Sales that happened today.",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
        known_inputs=(
            RequestedFactLiteralInput(
                id="time_1",
                source=KnownInputSource.QUESTION_CONTEXT,
                text="today",
                resolved_value_text="today",
                role=LiteralInputRole.TIME_VALUE,
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sale",),
                params=(
                    CatalogParam(
                        ref="sales.query.start_date",
                        name=start_param_name,
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                    CatalogParam(
                        ref="sales.query.end_date",
                        name="end_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                    CatalogParam(
                        ref="sales.query.status",
                        name="status",
                        source=ParamSource.QUERY,
                        type="choice",
                        choices=("OPEN", "CLOSED"),
                        choice_labels={"OPEN": "Open", "CLOSED": "Closed"},
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sales.field.amount",
                        path="amount",
                        type="decimal",
                    ),
                    CatalogField(
                        ref="sales.field.status",
                        path="status",
                        type="choice",
                        choices=("OPEN", "CLOSED"),
                    ),
                ),
            ),
        )
    )
    catalog_selection = _single_fact_catalog_selection(catalog, read_ids=("sales",))
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales happened today?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="How many sales happened today?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(scope.source_candidate_id,),
        ),
        available_values=(
            FactValue.time(
                id="time_1",
                known_input_id="time_1",
                expression="today",
                resolved_start="2026-05-22",
                resolved_end="2026-05-22",
                granularity="day",
                proof_refs=("known_input:time_1",),
                applies_to_requested_fact_ids=("fact_1",),
            ),
        ),
        available_value_uses=(
            GroundedInputUse(
                id="grounded_start",
                value_id="time_1",
                row_source_id=root_row_source_id,
                param_id="start_date",
                value_component=TimeComponent.START,
            ),
            GroundedInputUse(
                id="grounded_end",
                value_id="time_1",
                row_source_id=root_row_source_id,
                param_id="end_date",
                value_component=TimeComponent.END,
            ),
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="sales",
                ),
            )
        ),
    )


def _identity_field_filter_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="Locations in London.",
        answer_subject=RequestedFactAnswerSubject(subject_text="locations"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
        known_inputs=(
            RequestedFactLiteralInput(
                id="area_1",
                source=KnownInputSource.QUESTION_CONTEXT,
                text="London",
                resolved_value_text="London",
                value_meaning_hint="area",
                role=LiteralInputRole.REFERENCE_VALUE,
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="locations",
                endpoint_name="list_locations",
                resource_names=("location",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="locations.field.location_id",
                        path="data.location_id",
                        row_path_id="data",
                        type="uuid",
                        identity=IdentityMetadata(
                            entity_ref="location",
                            identity_field="location_id",
                            primary_key=True,
                            stable=True,
                        ),
                    ),
                    CatalogField(
                        ref="locations.field.area_id",
                        path="data.area_id",
                        row_path_id="data",
                        type="uuid",
                        identity=IdentityMetadata(
                            entity_ref="area",
                            identity_field="area_id",
                            primary_key=True,
                            stable=True,
                        ),
                    ),
                ),
            ),
        )
    )
    catalog_selection = _single_fact_catalog_selection(catalog, read_ids=("locations",))
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many locations are in London?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    return SourceBindingRequest(
        question="How many locations are in London?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_plan_for_sources(
            requested_fact_id="fact_1",
            source_candidate_ids=(scope.source_candidate_id,),
        ),
        available_values=(
            FactValue.identity(
                id="nairobi_area",
                known_input_id="area_1",
                identity_type="area",
                identity_field="area_id",
                value="area_nairobi",
                display_value="London",
                matched_field_ref="field.data.name",
                matched_field_path="data.name",
                proof_refs=("known_input:area_1",),
                applies_to_requested_fact_ids=("fact_1",),
            ),
        ),
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _read_assessment(
                    scope=scope,
                    requested_fact_id="fact_1",
                    read_id="locations",
                    relevant_row_path_ids=("data",),
                ),
            )
        ),
    )


def _selected_source_strategy(
    *,
    requested_fact_id: str,
    source_candidate_id: str,
) -> SelectedSourceStrategy:
    return SelectedSourceStrategy(
        plan_selection_id=f"plan.{requested_fact_id}",
        requested_fact_id=requested_fact_id,
        source_strategy_id=f"source_strategy.{requested_fact_id}.direct_field_value.1",
        plan_shape="direct_field_value",
        required_answer_output_ids=("answer_1",),
        source_members=(SourceStrategyMember(source_candidate_id=source_candidate_id),),
        basis="Selected by conformance fixture.",
    )


def _requested_fact(payload: object) -> RequestedFact:
    data = payload if isinstance(payload, dict) else {}
    known_inputs = tuple(
        _literal_input(raw_input) for raw_input in data.get("known_inputs") or ()
    )
    return RequestedFact(
        id=str(data.get("id") or "fact_1"),
        description=str(data.get("description") or "requested fact"),
        answer_expression=_answer_expression(data),
        answer_subject=RequestedFactAnswerSubject(
            subject_text=str(data.get("subject_text") or "records")
        ),
        answer_outputs=_answer_outputs(data),
        known_inputs=known_inputs,
        input_refs=tuple(input_item.id for input_item in known_inputs),
    )


def _answer_outputs(data: dict[str, Any]) -> tuple[RequestedFactAnswerOutput, ...]:
    raw_outputs = data.get("answer_outputs")
    if raw_outputs:
        return tuple(_answer_output(raw_output) for raw_output in raw_outputs)
    return (
        RequestedFactAnswerOutput(id=str(data.get("answer_output_id") or "answer_1")),
    )


def _answer_output(raw_output: object) -> RequestedFactAnswerOutput:
    if isinstance(raw_output, dict):
        return RequestedFactAnswerOutput(
            id=str(raw_output.get("id") or raw_output.get("answer_output_id") or ""),
            description=str(raw_output.get("description") or ""),
            role=str(raw_output.get("role") or ""),
        )
    return RequestedFactAnswerOutput(id=str(raw_output))


def _answer_expression(data: dict[str, Any]) -> RequestedFactAnswerExpression | None:
    family = str(data.get("answer_expression_family") or "")
    if not family:
        return None
    return RequestedFactAnswerExpression(
        family=RequestedFactAnswerExpressionFamily(family),
        group_key=_group_key(data.get("group_key")),
    )


def _group_key(raw_value: object) -> RequestedFactGroupKey | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("group_key must be an object")
    return RequestedFactGroupKey(
        id=str(raw_value.get("id") or "group_key"),
        description=str(raw_value.get("description") or "group key"),
        domain=GroupKeyDomainKind(str(raw_value.get("domain") or "")),
        question_input_refs=tuple(
            str(item) for item in raw_value.get("question_input_refs") or ()
        ),
    )


def _literal_input(raw_input: object) -> RequestedFactLiteralInput:
    data = raw_input if isinstance(raw_input, dict) else {}
    return RequestedFactLiteralInput(
        id=str(data.get("id") or ""),
        source=KnownInputSource(str(data.get("source") or "question_context")),
        role=LiteralInputRole(str(data.get("role") or "reference_value")),
        text=str(data.get("text") or ""),
        resolved_value_text=str(
            data.get("resolved_value_text") or data.get("text") or ""
        ),
        field_label_text=str(data.get("field_label_text") or ""),
        value_meaning_hint=str(data.get("value_meaning_hint") or ""),
    )


def _slot_role_presence_by_answer_output(
    slots: list[dict[str, Any]],
) -> dict[str, dict[str, bool]]:
    output_ids = sorted(
        {
            str(slot.get("answer_output_id") or "")
            for slot in slots
            if str(slot.get("answer_output_id") or "")
        }
    )
    output = {
        answer_output_id: {
            "GROUP_KEY": False,
            "ROW_POPULATION": False,
            "MEASURED_VALUE": False,
            "POPULATION_SCOPE": False,
        }
        for answer_output_id in output_ids
    }
    for slot in slots:
        answer_output_id = str(slot.get("answer_output_id") or "")
        if not answer_output_id:
            continue
        roles = output.setdefault(
            answer_output_id,
            {
                "GROUP_KEY": False,
                "ROW_POPULATION": False,
                "MEASURED_VALUE": False,
                "POPULATION_SCOPE": False,
            },
        )
        if slot.get("group_key_evidence"):
            roles["GROUP_KEY"] = True
        if slot.get("row_count_basis_evidence"):
            roles["ROW_POPULATION"] = True
        if slot.get("metric_measure_evidence"):
            roles["MEASURED_VALUE"] = True
        if slot.get("scope_evidence"):
            roles["POPULATION_SCOPE"] = True
    return output


def _schema_property_order(
    schema: object,
    *,
    markers: tuple[str, ...],
) -> list[str]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    if isinstance(properties, dict) and all(marker in properties for marker in markers):
        return list(properties)
    for value in schema.values():
        if isinstance(value, dict):
            result = _schema_property_order(value, markers=markers)
            if result:
                return result
        if isinstance(value, list):
            for item in value:
                result = _schema_property_order(item, markers=markers)
                if result:
                    return result
    return []


def _schema_review_test_ids(schema: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    finite_choice: dict[str, list[str]] = {}
    row_predicate: dict[str, list[str]] = {}
    for item_schema in _source_invocation_item_schemas(schema):
        properties = _schema_properties(item_schema)
        finite_choice_reviews = _schema_properties(
            properties.get("finite_choice_param_reviews")
        )
        for param_id, review_schema in finite_choice_reviews.items():
            finite_choice[param_id] = _finite_choice_review_test_ids(review_schema)
        row_predicate_reviews = _schema_properties(
            properties.get("row_predicate_reviews")
        )
        for predicate_id, review_schema in row_predicate_reviews.items():
            row_predicate[predicate_id] = _row_predicate_review_test_ids(review_schema)
    return {
        "finite_choice": finite_choice,
        "row_predicate": row_predicate,
    }


def _schema_review_test_id_counts(
    schema: dict[str, Any],
) -> dict[str, dict[str, int]]:
    test_ids = _schema_review_test_ids(schema)
    return {
        group: {key: len(values) for key, values in ids_by_key.items()}
        for group, ids_by_key in test_ids.items()
    }


def _source_invocation_item_schemas(schema: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(schema, dict):
        return ()
    properties = schema.get("properties")
    if isinstance(properties, dict):
        target_id = properties.get("binding_target_id")
        if isinstance(target_id, dict):
            return (schema,)
    for value in schema.values():
        if isinstance(value, dict):
            found = _source_invocation_item_schemas(value)
            if found:
                return found
        if isinstance(value, list):
            for item in value:
                found = _source_invocation_item_schemas(item)
                if found:
                    return found
    return ()


def _expand_schema_variants(schema: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(schema, dict):
        return ()
    variants = schema.get("oneOf")
    if isinstance(variants, list):
        return tuple(item for item in variants if isinstance(item, dict))
    return (schema,)


def _schema_properties(schema: object) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    variants = _expand_schema_variants(schema)
    if variants != (schema,):
        return _schema_properties(variants[0])
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _finite_choice_review_test_ids(schema: object) -> list[str]:
    properties = _schema_properties(schema)
    basis = _schema_properties(properties.get("population_test_basis"))
    return list(basis)


def _row_predicate_review_test_ids(schema: object) -> list[str]:
    properties = _schema_properties(schema)
    choice_reviews = properties.get("choice_reviews")
    choice_item = (
        choice_reviews.get("items") if isinstance(choice_reviews, dict) else {}
    )
    item_properties = _schema_properties(choice_item)
    results = _schema_properties(item_properties.get("population_test_results"))
    return list(results)


def _schema_surface_outcome(
    request: SourceBindingRequest,
    candidate: dict[str, Any],
    *,
    include_default_decision: bool,
    include_response_shape_decision: bool,
) -> dict[str, Any]:
    param_decisions: dict[str, Any] = {}
    if include_default_decision:
        param_decisions["granularity"] = _param_decision(
            _first_param_decision(candidate, "granularity")
        )
    if include_response_shape_decision:
        param_decisions["ordering"] = _param_decision(
            _first_param_decision(candidate, "ordering")
        )
    return _source_binding_outcome(
        request,
        candidate,
        param_decisions=param_decisions,
        finite_choice_param_reviews=_finite_choice_reviews_for_candidate(
            candidate,
            {
                "OPEN": "SATISFIES_TEST",
                "CLOSED": "CONFLICTS_WITH_TEST",
            },
        ),
    )


def _first_fulfillment_decisions_schema(schema: dict[str, Any]) -> dict[str, Any]:
    found = _find_fulfillment_decisions_schema(schema)
    if found is None:
        raise AssertionError("missing fulfillment_decisions schema")
    return found


def _find_fulfillment_decisions_schema(
    schema: dict[str, Any],
) -> dict[str, Any] | None:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            fulfillment = properties.get("fulfillment_decisions")
            if isinstance(fulfillment, dict):
                return fulfillment
        for value in schema.values():
            if isinstance(value, dict):
                found = _find_fulfillment_decisions_schema(value)
                if found is not None:
                    return found
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        found = _find_fulfillment_decisions_schema(item)
                        if found is not None:
                            return found
    return None


def _source_binding_outcome(
    request: SourceBindingRequest,
    candidate: dict[str, Any],
    *,
    fulfillment_decisions: dict[str, Any] | None = None,
    param_decisions: dict[str, Any] | None = None,
    row_predicate_reviews: dict[str, Any] | None = None,
    finite_choice_param_reviews: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_fulfillment_decisions = fulfillment_decisions or _fulfillment_decisions(
        candidate,
        field_id="amount",
    )
    metric_fit_contract = _metric_fit_contract(
        candidate,
        fulfillment_decisions=selected_fulfillment_decisions,
    )
    target = next(
        target
        for target in source_binding_targets_for_plan_selection(
            request.plan_selection,
            requested_facts=request.requested_facts,
        )
        if target.requested_fact_id == "fact_1"
        and target.source_candidate_id == str(candidate["source_candidate_id"])
    )
    return {
        "kind": "source_bindings",
        **metric_fit_contract,
        "bindings_for_fact_1": {
            "plan_shape": target.plan_shape,
            target.requirement_id: {
                        "binding_target_id": target.binding_target_id,
                        "answer_population": {
                            "population_binding_id": _binding_surface(candidate)[
                                "population_bindings"
                            ][0]["population_binding_id"],
                            "intent_text": "active sales",
                            "match_basis_explanation": "The requested fact asks for active sales.",
                        },
                        "fulfillment_decisions": selected_fulfillment_decisions,
                        "param_decisions": param_decisions or {},
                        "row_predicate_reviews": row_predicate_reviews or {},
                        "finite_choice_param_reviews": finite_choice_param_reviews
                        or {},
            }
        },
    }


def _binding_target_id_for_candidate(
    request: SourceBindingRequest,
    *,
    requested_fact_id: str,
    source_candidate_id: str,
) -> str:
    return _binding_target_for_candidate(
        request,
        requested_fact_id=requested_fact_id,
        source_candidate_id=source_candidate_id,
    ).binding_target_id


def _binding_target_for_candidate(
    request: SourceBindingRequest,
    *,
    requested_fact_id: str,
    source_candidate_id: str,
):
    targets = tuple(
        target
        for target in source_binding_targets_for_plan_selection(
            request.plan_selection,
            requested_facts=request.requested_facts,
        )
        if target.requested_fact_id == requested_fact_id
        and target.source_candidate_id == source_candidate_id
    )
    if len(targets) != 1:
        raise AssertionError(
            "source binding conformance fixture must identify exactly one "
            f"binding target for {(requested_fact_id, source_candidate_id)}"
        )
    return targets[0]


def _fulfillment_decisions(
    candidate: dict[str, Any],
    *,
    field_id: str,
) -> dict[str, Any]:
    support_set = _support_set_for_field(candidate, field_id=field_id)
    answer_output_id = str(support_set["answer_output_id"])
    return {
        answer_output_id: {
            "match_basis_explanation": (
                f"{answer_output_id} is fulfilled by {field_id}."
            ),
            "fulfillment_choice_id": support_set["fulfillment_choice_id"],
        }
    }


def _metric_fit_contract(
    candidate: dict[str, Any],
    *,
    fulfillment_decisions: dict[str, Any],
) -> dict[str, Any]:
    selected_choice_ids = {
        str(item["fulfillment_choice_id"]) for item in fulfillment_decisions.values()
    }
    metric_fit_bases: dict[str, Any] = {}
    fit_basis_interpretations: dict[str, Any] = {}
    for support_set in (
        _binding_surface(candidate).get("fulfillment_support_sets") or ()
    ):
        if (
            str(support_set.get("fulfillment_choice_id") or "")
            not in selected_choice_ids
        ):
            continue
        for slot in support_set.get("fulfillment_slots") or ():
            for item in slot.get("metric_measure_evidence") or ():
                evidence_id = str(item.get("evidence_id") or "")
                if not evidence_id:
                    continue
                metric_fit_bases.setdefault("fact_1", {})[evidence_id] = {
                    "metric_meaning": f"{evidence_id} is a numeric measure.",
                    "fit_basis": "The selected metric evidence fits this fixture.",
                }
                fit_basis_interpretations.setdefault("fact_1", {})[evidence_id] = {
                    "interpretation": "FITS_REQUESTED_ANSWER",
                }
    return {
        "metric_fit_bases": metric_fit_bases,
        "fit_basis_interpretations": fit_basis_interpretations,
    }


def _set_metric_fit_from_case(
    outcome: dict[str, Any],
    candidate: dict[str, Any],
    metric_decisions: dict[str, str],
) -> None:
    bases = outcome.setdefault("metric_fit_bases", {}).setdefault("fact_1", {})
    interpretations = outcome.setdefault("fit_basis_interpretations", {}).setdefault(
        "fact_1", {}
    )
    for field_id, decision in metric_decisions.items():
        evidence_id = _metric_evidence_id_for_field(candidate, field_id=str(field_id))
        bases[evidence_id] = {
            "metric_meaning": f"{field_id} is a numeric measure.",
            "fit_basis": f"{field_id} was reviewed for this answer.",
        }
        interpretations[evidence_id] = {
            "interpretation": str(decision),
        }


def _set_raw_metric_fit_from_case(
    outcome: dict[str, Any],
    metric_decisions: dict[str, str],
) -> None:
    bases = outcome.setdefault("metric_fit_bases", {}).setdefault("fact_1", {})
    interpretations = outcome.setdefault("fit_basis_interpretations", {}).setdefault(
        "fact_1", {}
    )
    for evidence_id, decision in metric_decisions.items():
        bases[str(evidence_id)] = {
            "metric_meaning": f"{evidence_id} was provided by the test fixture.",
            "fit_basis": f"{evidence_id} was reviewed for this answer.",
        }
        interpretations[str(evidence_id)] = {
            "interpretation": str(decision),
        }


def _field_ids_for_evidence_ids(
    candidate: dict[str, Any],
    evidence_ids: tuple[str, ...],
) -> list[str]:
    field_ids_by_evidence_id = {
        str(item.get("evidence_id") or ""): str(item.get("field_id") or "")
        for support_set in _binding_surface(candidate).get("fulfillment_support_sets")
        or ()
        if isinstance(support_set, dict)
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
        for key in ("metric_measure_evidence", "row_count_basis_evidence")
        for item in slot.get(key) or ()
        if isinstance(item, dict)
    }
    return [
        field_ids_by_evidence_id[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id in field_ids_by_evidence_id
    ]


def _metric_evidence_id_for_field(
    candidate: dict[str, Any],
    *,
    field_id: str,
) -> str:
    support_set = _support_set_for_field(candidate, field_id=field_id)
    for slot in support_set.get("fulfillment_slots") or ():
        if not isinstance(slot, dict):
            continue
        for key in ("metric_measure_evidence", "row_count_basis_evidence"):
            for item in slot.get(key) or ():
                if isinstance(item, dict) and item.get("field_id") == field_id:
                    return str(item["evidence_id"])
    raise AssertionError(f"missing metric or row-count evidence for {field_id}")


def _param_decision(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "population_intent": "sales for this conformance case",
        "match_basis_explanation": "Selected source-binding param decision.",
        "param_decision_id": decision["param_decision_id"],
    }


def _first_param_decision(candidate: dict[str, Any], param_id: str) -> dict[str, Any]:
    param = _param(candidate, param_id)
    for decision in param.get("decision_options") or ():
        if isinstance(decision, dict) and decision.get("decision") == "bind":
            return decision
    raise AssertionError(f"missing bind decision for {param_id}")


def _param(candidate: dict[str, Any], param_id: str) -> dict[str, Any]:
    for param in _binding_surface(candidate).get("params") or ():
        if isinstance(param, dict) and param.get("param_id") == param_id:
            return param
    raise AssertionError(f"missing param {param_id}")


def _row_predicate_reviews_from_case(
    candidate: dict[str, Any],
    row_predicates: dict[str, Any],
) -> dict[str, Any]:
    by_field_id = {
        str(item["field_id"]): item
        for item in candidate.get("row_predicates") or ()
        if isinstance(item, dict)
    }
    output: dict[str, Any] = {}
    for field_id, value_effects in row_predicates.items():
        predicate = by_field_id[str(field_id)]
        output[str(predicate["predicate_id"])] = {
            "choice_reviews": [
                _row_predicate_choice_review(value=value, effect=effect)
                for value, effect in value_effects.items()
            ]
        }
    return output


def _row_predicate_choice_review(*, value: str, effect: object) -> dict[str, Any]:
    membership_tests = _membership_tests_from_effect(effect)
    effects = _effects_by_membership_test(effect, membership_tests)
    return {
        "choice_option_id": str(value),
        "choice_domain_meaning": f"is_active={value}",
        "population_test_results": {
            test["test_id"]: _row_predicate_population_test_result(
                test_id=test["test_id"],
                question=test["test_question"],
                effect=effects[test["test_id"]],
            )
            for test in membership_tests
        },
    }


def _finite_choice_review_from_case(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload["choices"]
    membership_tests = _membership_tests_from_choice_effects(choices)
    return {
        "controlled_population_role_id": str(
            payload.get("controlled_population_role_id") or "role_1"
        ),
        "role_selection_basis": "status controls sales rows being counted.",
        "population_test_basis": _population_test_basis(membership_tests),
        "choice_reviews": [
            _finite_choice_option_review(
                value=value,
                effect=effect,
                membership_tests=membership_tests,
            )
            for value, effect in choices.items()
        ],
    }


def _finite_choice_reviews_for_candidate(
    candidate: dict[str, Any],
    choices: dict[str, Any],
) -> dict[str, Any]:
    param_ids = {
        str(item.get("param_id") or "")
        for item in _binding_surface(candidate).get("params") or ()
        if isinstance(item, dict)
    }
    if "status" not in param_ids:
        return {}
    return {"status": _finite_choice_review_from_case({"choices": choices})}


def _finite_choice_option_review(
    *,
    value: str,
    effect: object,
    membership_tests: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    effect_spec = {} if isinstance(effect, str) else dict(effect)
    effects = _effects_by_membership_test(effect, membership_tests)
    omitted_tests = {
        str(item) for item in effect_spec.get("omit_tests", ()) if str(item)
    }
    population_test_results = {
        test["test_id"]: (
            _finite_choice_normal_guard_result(
                value=str(value),
                effect=effects[test["test_id"]],
                include_review=bool(
                    effect_spec.get("include_normal_guard_result", True)
                ),
                review_payload=effect_spec.get("normal_instance_match"),
            )
            if test["test_id"] == "normal_instance_guard"
            else _compact_population_test_result(
                test=test,
                effect=effects[test["test_id"]],
            )
        )
        for test in membership_tests
    }
    return {
        "choice_option_id": str(value),
        "choice_domain_meaning": f"{str(value).lower()} sales",
        "choice_inclusion_basis": f"{value} is reviewed for inclusion.",
        "choice_inclusion": effect_spec.get("choice_inclusion")
        or (
            "EXCLUDE"
            if any(v == "CONFLICTS_WITH_TEST" for v in effects.values())
            else "INCLUDE"
        ),
        "population_test_results": {
            test_id: result
            for test_id, result in population_test_results.items()
            if test_id not in omitted_tests
        },
    }


def _finite_choice_normal_guard_result(
    *,
    value: str,
    effect: str,
    include_review: bool,
    review_payload: object,
) -> dict[str, Any]:
    if not include_review:
        return {
            "population_consequence": (
                f"{value} has no normal-instance review in this conformance case."
            ),
            "disposition": {
                "test_effect": effect,
            },
        }
    guard_fields = _normal_instance_guard_fields_from_case(
        value=value,
        effect=effect,
        payload=review_payload,
    )
    return {
        **guard_fields,
        "population_consequence": f"{effect} for this conformance case.",
    }


def _normal_instance_guard_fields_from_case(
    *,
    value: str,
    effect: str,
    payload: object,
) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {
            "role_match_basis": f"{value} was compared to excluded normal-instance roles.",
            "explicit_user_override_evidence": list(
                payload.get("explicit_user_override_evidence") or ()
            ),
            "explicit_user_override_applies": bool(
                payload.get("explicit_user_override_applies")
            ),
            "population_consequence": f"{value} effect for normal_instance_guard is {effect}.",
            "disposition": {
                "matched_excluded_role": str(
                    payload.get("matched_excluded_role") or "NONE"
                ),
                "test_effect": effect,
            },
        }
    return _normal_instance_guard_fields_from_effect(value=value, effect=effect)


def _normal_instance_guard_fields_from_effect(
    *,
    value: str,
    effect: str,
) -> dict[str, Any]:
    matched_role = (
        NormalInstanceExcludedStateRole.CANCELED_OR_VOIDED.value
        if effect == "CONFLICTS_WITH_TEST"
        else "NONE"
    )
    return {
        "role_match_basis": f"{value} was compared to excluded normal-instance roles.",
        "explicit_user_override_evidence": [],
        "explicit_user_override_applies": False,
        "population_consequence": f"{value} effect for normal_instance_guard is {effect}.",
        "disposition": {
            "matched_excluded_role": matched_role,
            "test_effect": effect,
        },
    }


def _compact_population_test_result(
    *,
    test: dict[str, str],
    effect: str,
) -> dict[str, str]:
    return {
        "test_basis": f"{effect} for {test['test_id']} in this conformance case.",
        "population_consequence": f"{effect} for this conformance case.",
        "test_effect": effect,
    }


def _row_predicate_population_test_result(
    *,
    test_id: str,
    question: str,
    effect: str,
) -> dict[str, str]:
    return {
        "test_id": test_id,
        "test_question": question,
        "role_scoped_test_question": question,
        "because": f"{effect} for this conformance case.",
        "test_effect": effect,
    }


def _finite_choice_membership_tests() -> tuple[dict[str, str], ...]:
    return (
        {
            "test_id": "subject_identity",
            "test_question": "Does the row/value represent sales?",
        },
        {
            "test_id": "normal_instance_guard",
            "test_question": "Is this an ordinary business instance of sales?",
        },
    )


def _membership_tests_from_choice_effects(
    choices: dict[str, Any],
) -> tuple[dict[str, str], ...]:
    test_ids: list[str] = []
    for effect in choices.values():
        for test_id in _effect_membership_test_ids(effect):
            if test_id not in test_ids:
                test_ids.append(test_id)
    if not test_ids:
        return _finite_choice_membership_tests()
    return tuple(_membership_test_definition(test_id) for test_id in test_ids)


def _membership_tests_from_effect(effect: object) -> tuple[dict[str, str], ...]:
    test_ids = _effect_membership_test_ids(effect)
    if not test_ids:
        return _finite_choice_membership_tests()
    return tuple(_membership_test_definition(test_id) for test_id in test_ids)


def _effect_membership_test_ids(effect: object) -> tuple[str, ...]:
    if isinstance(effect, str):
        return ()
    effect_spec = dict(effect)
    special_keys = {
        "omit_tests",
        "include_normal_guard_result",
        "normal_instance_match",
        "choice_inclusion",
    }
    return tuple(str(key) for key in effect_spec if str(key) not in special_keys)


def _effects_by_membership_test(
    effect: object,
    membership_tests: tuple[dict[str, str], ...],
) -> dict[str, str]:
    if isinstance(effect, str):
        return {test["test_id"]: str(effect) for test in membership_tests}
    effect_spec = dict(effect)
    return {
        test["test_id"]: str(effect_spec[test["test_id"]]) for test in membership_tests
    }


def _membership_test_definition(test_id: str) -> dict[str, str]:
    questions = {
        "subject_identity": "Does the row/value represent sales?",
        "normal_instance_guard": "Is this an ordinary business instance of sales?",
        "explicit_user_constraint:specified_staff": (
            "Does the sale belong to the specified staff member?"
        ),
    }
    return {
        "test_id": test_id,
        "test_question": questions.get(test_id, f"Does this satisfy {test_id}?"),
    }


def _membership_test_by_id(
    tests: tuple[dict[str, str], ...],
    test_id: str,
) -> dict[str, str]:
    for test in tests:
        if test["test_id"] == test_id:
            return test
    raise AssertionError(f"unknown finite-choice membership test: {test_id}")


def _population_test_basis(
    tests: tuple[dict[str, str], ...],
) -> dict[str, dict[str, str]]:
    return {
        test["test_id"]: {
            "test_question": test["test_question"],
            "role_scoped_test_question": (
                f"For sales rows being counted, {test['test_question']}"
            ),
        }
        for test in tests
    }


def _support_set_for_field(
    candidate: dict[str, Any],
    *,
    field_id: str,
) -> dict[str, Any]:
    for support_set in (
        _binding_surface(candidate).get("fulfillment_support_sets") or ()
    ):
        for slot in support_set.get("fulfillment_slots") or ():
            for key in (
                "metric_measure_evidence",
                "row_count_basis_evidence",
                "group_key_evidence",
            ):
                for item in slot.get(key) or ():
                    if item.get("field_id") == field_id:
                        return support_set
    raise AssertionError(f"missing fulfillment support set for {field_id}")


def _only_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = _source_candidates(payload)
    if len(candidates) != 1:
        raise AssertionError(f"expected one source candidate, got {candidates!r}")
    return candidates[0]


def _source_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for fact_sources in payload.get("requested_fact_sources") or ()
        if isinstance(fact_sources, dict)
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict)
    ]


def _source_candidates_with_fact(
    payload: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(fact_sources.get("requested_fact_id") or ""), candidate)
        for fact_sources in payload.get("requested_fact_sources") or ()
        if isinstance(fact_sources, dict)
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict)
    ]


def _binding_surface(candidate: dict[str, Any]) -> dict[str, Any]:
    surface = candidate.get("binding_surface")
    if isinstance(surface, dict):
        return surface
    output = {
        key: candidate[key]
        for key in ("population_bindings", "params", "row_predicates", "bound_params")
        if key in candidate
    }
    if "fulfillment_support_sets" in candidate:
        output["fulfillment_support_sets"] = candidate["fulfillment_support_sets"]
    if "fulfillment_choices" in candidate:
        output["fulfillment_support_sets"] = candidate["fulfillment_choices"]
    return output
