from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    EndpointRead,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
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
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    GroupKeyDomainKind,
    KnownInputSource,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactGroupKey,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactLiteralInput,
)
from fervis.lookup.fact_plan.row_sources import api_row_source_id
from fervis.lookup.answer_program.values import FactValue, TimeComponent
from fervis.lookup.canonical_data import entity_key_value
from fervis.lookup.fact_planning.pattern_plan import compile_pattern_answer_program
from fervis.lookup.fact_planning.provider_contract import parse_pattern_answer
from fervis.lookup.provider_contract import ProviderObject
from fervis.lookup.answer_program import compiler_input_context
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.question_inputs import LiteralInputRole
from fervis.lookup.read_eligibility import (
    ReadAssessment,
    ReadEligibilityRequest,
    ReadEligibilityResult,
)
from fervis.lookup.read_eligibility.surface import read_eligibility_candidate_surface
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.turn_prompts.projections import source_binding_candidates_xml
from fervis.lookup.plan_selection import (
    PlanSelectionSet,
    SelectedSourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.source_binding import (
    BoundSource,
    SourceBindingRequest,
    SourceBindingTurnPrompt,
)
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.candidates.contracts import (
    parse_evidence_item,
    parse_fulfillment_support_set,
)
from fervis.lookup.source_binding.parser.fulfillment import parse_source_fulfillments
from fervis.lookup.source_binding.provider_contract import FulfillmentDecisionOutput
from fervis.lookup.source_binding.schema import build_source_binding_schema
from fervis.lookup.plan_selection.family_specs import SourceMemberConstraint
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingPlanFamily,
    SourceBindingTarget,
    SourceBindingTargetCompatibility,
    source_binding_fact_field_id,
    source_binding_target_index_for_plan_selection,
)
from fervis.lookup.source_binding.role_selection import value_only_source_binding_plan
from fervis.lookup.source_binding.parser import parse_source_binding
from fervis.lookup.source_binding.model import SourceBindingPlan
from fervis.lookup.orchestration.pipeline import (
    _bound_plan_selection_from_plan_selection,
)
from fervis.lookup.operation_families.source_binding_registry import (
    source_binding_metric_evidence_ids_by_requested_fact,
)
from tests.lookup.source_binding_helpers import (
    source_binding_request,
    source_binding_target_id_for_candidate,
    source_fulfills_by_row_population_for_candidate,
    source_fulfills_fields_for_candidate,
    source_fulfills_keys_for_candidate,
)


def test_source_binding_schema_uses_compact_role_target_handles():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    invocation_schemas = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )

    target_ids = {target["binding_target_id"] for target in targets}

    assert set(invocation_schemas) == target_ids
    for invocation_schema in invocation_schemas.values():
        assert "source_candidate_id" not in invocation_schema["properties"]
    candidate_target_schema = invocation_schemas[
        "target.fact_1.set_difference.source_1.candidate_set"
    ]
    fulfillment_schema = candidate_target_schema["properties"]["fulfillment_decisions"]
    assert "answer_1" in fulfillment_schema["properties"]
    assert "anyOf" not in fulfillment_schema


def test_source_binding_projects_fact_shape_role_plan_families():
    prompt = SourceBindingTurnPrompt(_set_difference_request())

    payload = prompt.binding_plan_families_payload()

    fact = payload["bindings_by_requested_fact"]["fact_1"]
    shape = fact["plan_shapes"]["set_difference"]
    assert shape["member_constraint"] == "DISTINCT_SOURCE_CANDIDATES"
    assert set(shape["role_targets"]) == {"candidate_set", "observed_set"}
    assert all(
        target["requirement_id"] == role_id
        for role_id, targets in shape["role_targets"].items()
        for target in targets
    )


def test_source_binding_schema_is_fact_shape_role_keyed():
    schema = (
        SourceBindingTurnPrompt(_set_difference_request())
        .response_contract()
        .provider_schema
    )

    outcome = next(
        variant
        for variant in schema["properties"]["outcome"]["oneOf"]
        if variant["properties"]["kind"].get("enum") == ["source_bindings"]
    )
    fact = outcome["properties"]["bindings_for_fact_1"]

    assert "source_invocations" not in outcome["properties"]
    assert fact["properties"]["plan_shape"]["enum"] == ["set_difference"]
    assert set(fact["properties"]) - {"plan_shape"} == {
        "candidate_set",
        "observed_set",
    }


def test_source_binding_schema_scopes_param_surfaces_to_binding_target():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={
            "target.source_1": {
                "start_date": ("param_decision.source_1.start_date.bind.month",),
            },
            "target.source_2": {
                "status": ("param_decision.source_2.status.use_default.completed",),
                "group_by": ("param_decision.source_2.group_by.bind.location",),
            },
        },
        target_finite_choice_values={
            "target.source_1": {"status": ("DRAFT", "COMPLETED")},
            "target.source_2": {"sale_type": ("STORE", "ONLINE")},
        },
        target_row_predicate_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_test_ids={
            "target.source_1": {"status": ("subject_identity",)},
            "target.source_2": {"sale_type": ("subject_identity",)},
        },
        target_finite_choice_normal_instance_test_ids={
            "target.source_1": {"status": ()},
            "target.source_2": {"sale_type": ()},
        },
        target_row_predicate_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_population_roles={
            "target.source_1": ({"role_id": "role.source_1.rows"},),
            "target.source_2": ({"role_id": "role.source_2.rows"},),
        },
        target_requested_fact_ids={
            "target.source_1": "fact_1",
            "target.source_2": "fact_1",
        },
        metric_evidence_ids_by_requested_fact={"fact_1": ("source_2.data.amount",)},
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {"answer_1": ("support.source_1.answer_1",)},
            "target.source_2": {"answer_1": ("support.source_2.answer_1",)},
        },
        target_required_fulfillment_answer_output_ids={
            "target.source_1": ("answer_1",),
            "target.source_2": ("answer_1",),
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1.candidate_population",),
            "target.source_2": ("pop.source_2.candidate_population",),
        },
        plan_families=_test_plan_families("target.source_1", "target.source_2"),
    )

    variants = _source_invocation_variants_by_target(schema)
    source_1 = variants["target.source_1"]
    source_2 = variants["target.source_2"]

    assert set(source_1["properties"]["param_decisions"]["properties"]) == {
        "start_date"
    }
    assert set(source_1["properties"]["finite_choice_param_reviews"]["properties"]) == {
        "status"
    }
    assert set(source_2["properties"]["param_decisions"]["properties"]) == {
        "group_by",
        "status",
    }
    assert set(source_2["properties"]["finite_choice_param_reviews"]["properties"]) == {
        "sale_type"
    }


def test_source_binding_schema_requires_only_selectable_fulfillment_outputs():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={"target.source_1": {}},
        target_finite_choice_values={"target.source_1": {}},
        target_row_predicate_values={"target.source_1": {}},
        target_finite_choice_test_ids={"target.source_1": {}},
        target_finite_choice_normal_instance_test_ids={"target.source_1": {}},
        target_row_predicate_test_ids={"target.source_1": {}},
        target_population_roles={"target.source_1": ()},
        target_requested_fact_ids={"target.source_1": "fact_1"},
        metric_evidence_ids_by_requested_fact={
            "fact_1": ("source_1.data.total",),
        },
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {
                "answer_group": ("support.source_1.answer_group",),
            },
        },
        target_required_fulfillment_answer_output_ids={
            "target.source_1": ("answer_group", "answer_metric"),
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1.candidate_population",),
        },
        plan_families=_test_plan_families("target.source_1"),
    )
    invocation_schema = _source_invocation_variants_by_target(schema)["target.source_1"]
    fulfillment_schema = invocation_schema["properties"]["fulfillment_decisions"]

    assert fulfillment_schema["required"] == ["answer_group"]
    assert set(fulfillment_schema["properties"]) == {"answer_group"}
    validate(
        instance={
            "outcome": {
                "kind": "source_bindings",
                "metric_fit_bases": {
                    "fact_1": {
                        "source_1.data.total": {
                            "metric_meaning": "sales total",
                            "fit_basis": "The selected metric answers the fact.",
                        }
                    }
                },
                "fit_basis_interpretations": {
                    "fact_1": {
                        "source_1.data.total": {
                            "interpretation": "FITS_REQUESTED_ANSWER",
                        }
                    }
                },
                "bindings_for_fact_1": {
                    "plan_shape": "test_shape",
                    "primary": {
                        **_minimal_source_invocation(
                            "target.source_1",
                            "pop.source_1.candidate_population",
                        ),
                        "fulfillment_decisions": {
                            "answer_group": {
                                "fulfillment_choice_id": (
                                    "support.source_1.answer_group"
                                ),
                                "match_basis_explanation": "The selected group key.",
                            },
                        },
                    },
                },
            },
        },
        schema=schema,
    )


def test_source_binding_schema_requires_exposed_row_predicate_reviews():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={"target.source_1": {}},
        target_finite_choice_values={"target.source_1": {}},
        target_row_predicate_values={
            "target.source_1": {
                "rp.source_1.row.data.is_deposited": ("true", "false"),
            },
        },
        target_finite_choice_test_ids={"target.source_1": {}},
        target_finite_choice_normal_instance_test_ids={"target.source_1": {}},
        target_row_predicate_test_ids={
            "target.source_1": {
                "rp.source_1.row.data.is_deposited": ("membership_test_1",),
            },
        },
        target_population_roles={"target.source_1": ()},
        target_requested_fact_ids={"target.source_1": "fact_1"},
        metric_evidence_ids_by_requested_fact={},
        target_fulfillment_support_set_ids_by_answer_output={"target.source_1": {}},
        target_required_fulfillment_answer_output_ids={"target.source_1": ()},
        target_population_binding_ids={
            "target.source_1": ("pop.source_1.candidate_population",),
        },
        plan_families=_test_plan_families("target.source_1"),
    )
    invocation_schema = _source_invocation_variants_by_target(schema)["target.source_1"]
    row_reviews_schema = invocation_schema["properties"]["row_predicate_reviews"]

    assert row_reviews_schema["required"] == ["rp.source_1.row.data.is_deposited"]
    with pytest.raises(ValidationError):
        validate(
            instance={
                "outcome": {
                    "kind": "source_bindings",
                    "metric_fit_bases": {},
                    "fit_basis_interpretations": {},
                    **_test_fact_binding(
                        requested_fact_id="fact_1",
                        plan_shape="test_shape",
                        requirement_id="primary",
                        invocation=_minimal_source_invocation(
                            "target.source_1",
                            "pop.source_1.candidate_population",
                        ),
                    ),
                },
            },
            schema=schema,
        )


def test_source_binding_schema_accepts_known_target_arrays_without_enumeration():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_row_predicate_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_normal_instance_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_row_predicate_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_population_roles={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_requested_fact_ids={
            "target.source_1": "fact_1",
            "target.source_2": "fact_1",
        },
        metric_evidence_ids_by_requested_fact={},
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_required_fulfillment_answer_output_ids={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1",),
            "target.source_2": ("pop.source_2",),
        },
        plan_families=_test_plan_families("target.source_1", "target.source_2"),
    )
    source_1 = _minimal_source_invocation("target.source_1", "pop.source_1")
    source_2 = _minimal_source_invocation("target.source_2", "pop.source_2")

    validate(
        instance=_source_binding_plan_payload(source_1),
        schema=schema,
    )
    validate(instance=_source_binding_plan_payload(source_2), schema=schema)


def test_source_binding_schema_exposes_alternatives_as_one_role_choice():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_row_predicate_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_normal_instance_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_row_predicate_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_population_roles={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_requested_fact_ids={
            "target.source_1": "fact_1",
            "target.source_2": "fact_1",
        },
        metric_evidence_ids_by_requested_fact={},
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_required_fulfillment_answer_output_ids={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1",),
            "target.source_2": ("pop.source_2",),
        },
        plan_families=_test_plan_families("target.source_1", "target.source_2"),
    )
    source_1 = _minimal_source_invocation("target.source_1", "pop.source_1")
    source_2 = _minimal_source_invocation("target.source_2", "pop.source_2")

    validate(instance=_source_binding_plan_payload(source_1), schema=schema)
    validate(instance=_source_binding_plan_payload(source_2), schema=schema)


def test_source_binding_prompt_requires_each_shape_role_once():
    schema = (
        SourceBindingTurnPrompt(_set_difference_request())
        .response_contract()
        .provider_schema
    )
    outcome = _source_binding_outcome_schema(schema)
    fact = outcome["properties"]["bindings_for_fact_1"]

    assert fact["required"] == ["plan_shape", "candidate_set", "observed_set"]


def test_source_binding_prompt_scopes_plan_selection_basis_to_api_read():
    request = _closed_key_grouped_staff_sales_request()
    note = "Uses row-level sales records; grouping and counting still happen later."
    plan = request.plan_selection.plan_selections[0]
    request = replace(
        request,
        plan_selection=PlanSelectionSet(plan_selections=(replace(plan, basis=note),)),
    )
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    xml = source_binding_candidates_xml(prompt.source_invocation_candidate_payload())

    api_read = f'<api_read id="{target["source_candidate_id"]}"'
    note_node = f"<selection_note>{note}</selection_note>"

    assert api_read in xml
    assert note_node in xml
    api_read_start = xml.index(api_read)
    note_start = xml.index(note_node)
    input_params_start = xml.index("<input_params>", api_read_start)
    assert api_read_start < note_start < input_params_start


def test_closed_key_grouped_identity_param_is_backend_owned_not_model_authored():
    request = _closed_key_grouped_staff_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]

    param_properties = invocation_schema["properties"]["param_decisions"].get(
        "properties",
        {},
    )

    assert "staff_id" not in param_properties


def test_closed_key_grouped_identity_param_is_backend_owned_without_grounding_uses():
    request = replace(
        _closed_key_grouped_staff_sales_request(),
        available_value_uses=(),
    )
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]

    param_properties = invocation_schema["properties"]["param_decisions"].get(
        "properties",
        {},
    )
    candidate_prompt = source_binding_candidates_xml(
        prompt.source_invocation_candidate_payload()
    )

    assert "backend_owned_param_bindings" in str(target)
    assert "staff_id" not in param_properties
    assert '<param param_id="staff_id"' not in candidate_prompt


def test_closed_key_grouped_identity_param_can_use_group_key_field_id():
    request = _closed_key_grouped_staff_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]

    param_properties = invocation_schema["properties"]["param_decisions"].get(
        "properties",
        {},
    )

    assert "backend_owned_param_bindings" in str(target)
    assert "staff_id" not in param_properties


def test_grounded_endpoint_params_are_backend_owned_not_model_authored():
    request = _closed_key_grouped_staff_sales_today_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]

    param_properties = invocation_schema["properties"]["param_decisions"].get(
        "properties",
        {},
    )
    backend_bindings = target["backend_owned_param_bindings"]
    direct_param_ids = {
        binding["param_id"] for binding in backend_bindings if "param_id" in binding
    }
    key_param_ids = {
        param_id
        for binding in backend_bindings
        for param_id in binding.get("params_by_component_id", {}).values()
    }
    backend_param_ids = direct_param_ids | key_param_ids

    assert {"staff_id", "start_date", "end_date"}.isdisjoint(param_properties)
    assert backend_param_ids == {"staff_id", "start_date", "end_date"}


def test_closed_key_grouped_identity_param_scopes_group_key_fulfillment_choices():
    request = _closed_key_grouped_staff_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]
    candidate = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())[
        target["source_candidate_id"]
    ]

    choice_ids = invocation_schema["properties"]["fulfillment_decisions"]["properties"][
        "answer_staff"
    ]["properties"]["fulfillment_choice_id"]["enum"]
    staff_id_choice = source_fulfills_keys_for_candidate(
        candidate,
        key_ids_by_answer_output={"answer_staff": "staff_key"},
    )["answer_staff"]["fulfillment_choice_id"]

    assert choice_ids == ["source_1.data.reference.sale_staff"]
    assert staff_id_choice == "source_1.data.reference.sale_staff"
    assert "source_1.data.staff_name" not in choice_ids


def test_source_binding_prompt_marks_canonical_group_entity_choice():
    request = _closed_key_grouped_staff_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    prompt_text = prompt.to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context={},
        )
    ).prompt_text

    candidate_prompt = source_binding_candidates_xml(
        prompt.source_invocation_candidate_payload()
    )

    assert (
        "Entity outputs use a declared source candidate key or entity reference"
        in prompt_text
    )
    assert "Context labels are not selectable computation evidence" in prompt_text
    assert '<choice id="source_1.data.reference.sale_staff"' in candidate_prompt
    assert '<choice id="fulfillment_' not in candidate_prompt


def test_parse_source_binding_rejects_closed_key_param_with_mismatched_group_key():
    request = _closed_key_grouped_staff_sales_request()
    row_population_evidence_id = _only_metric_evidence_id(request)
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    candidate = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())[
        target["source_candidate_id"]
    ]

    with pytest.raises(
        ValueError, match="source fulfillment references unknown choice"
    ):
        parse_source_binding(
            {
                "outcome": {
                    "kind": "source_bindings",
                    "metric_fit_bases": {
                        "fact_1": {
                            row_population_evidence_id: {
                                "metric_meaning": "count of sales rows",
                                "fit_basis": (
                                    "The requested sales count is row cardinality."
                                ),
                            }
                        }
                    },
                    "fit_basis_interpretations": {
                        "fact_1": {
                            row_population_evidence_id: {
                                "interpretation": "FITS_REQUESTED_ANSWER",
                            }
                        }
                    },
                    **_test_fact_binding(
                        requested_fact_id=target["requested_fact_id"],
                        plan_shape=target["plan_shape"],
                        requirement_id=target["requirement_id"],
                        invocation={
                            "binding_target_id": target["binding_target_id"],
                            "answer_population": {
                                "population_binding_id": _population_binding_id(
                                    candidate
                                ),
                                "intent_text": "sales by specified staff member",
                                "match_basis_explanation": (
                                    "Use the sales row population for the grouped count."
                                ),
                            },
                            "fulfillment_decisions": {
                                "answer_staff": {
                                    "fulfillment_choice_id": "source_1.data.staff_name",
                                    "match_basis_explanation": (
                                        "Stale non-canonical display field choice."
                                    ),
                                },
                                **source_fulfills_by_row_population_for_candidate(
                                    candidate,
                                    answer_output_ids=("answer_count",),
                                    row_path_id="data",
                                ),
                            },
                            "param_decisions": {},
                            "row_predicate_reviews": {},
                            "finite_choice_param_reviews": {},
                        },
                    ),
                },
            },
            request=request,
        )


def test_parse_source_binding_expands_backend_owned_closed_key_param_bindings():
    request = _closed_key_grouped_staff_sales_request()
    row_population_evidence_id = _only_metric_evidence_id(request)
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    candidate = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())[
        target["source_candidate_id"]
    ]

    result = parse_source_binding(
        {
            "outcome": {
                "kind": "source_bindings",
                "metric_fit_bases": {
                    "fact_1": {
                        row_population_evidence_id: {
                            "metric_meaning": "count of sales rows",
                            "fit_basis": "The requested sales count is row cardinality.",
                        }
                    }
                },
                "fit_basis_interpretations": {
                    "fact_1": {
                        row_population_evidence_id: {
                            "interpretation": "FITS_REQUESTED_ANSWER",
                        }
                    }
                },
                **_test_fact_binding(
                    requested_fact_id=target["requested_fact_id"],
                    plan_shape=target["plan_shape"],
                    requirement_id=target["requirement_id"],
                    invocation={
                        "binding_target_id": target["binding_target_id"],
                        "answer_population": {
                            "population_binding_id": _population_binding_id(candidate),
                            "intent_text": "sales by specified staff member",
                            "match_basis_explanation": (
                                "Use the sales row population for the grouped count."
                            ),
                        },
                        "fulfillment_decisions": {
                            **source_fulfills_fields_for_candidate(
                                candidate,
                                field_ids_by_answer_output={
                                    "answer_staff": ("staff_id",),
                                },
                            ),
                            **source_fulfills_by_row_population_for_candidate(
                                candidate,
                                answer_output_ids=("answer_count",),
                                row_path_id="data",
                            ),
                        },
                        "param_decisions": {},
                        "row_predicate_reviews": {},
                        "finite_choice_param_reviews": {},
                    },
                ),
            },
        },
        request=request,
    )

    assert isinstance(result.outcome, SourceBindingPlan)
    bound_source = result.outcome.bound_sources[0]
    assert tuple(
        binding.value
        for invocation in bound_source.source_invocations
        for binding in invocation.param_bindings
        if binding.param_id == "staff_id"
    ) == (
        "51515151-0000-0000-0002-000000000001",
        "51515151-0000-0000-0002-000000000002",
    )


def test_parse_source_fulfillment_derives_required_row_count_without_selectable_choice():
    evidence_items = tuple(
        parse_evidence_item(item)
        for item in (
            {
                "evidence_id": "source_1.data.staff_id",
                "field_id": "staff_id",
                "type": "string",
            },
            {
                "evidence_id": "source_1.data.key.staff_key",
                "type": "candidate_key",
                "key_id": "staff_key",
                "entity_kind": "staff",
                "components": [
                    {
                        "component_id": "staff_id",
                        "field_evidence_id": "source_1.data.staff_id",
                        "field_id": "staff_id",
                    }
                ],
                "row_source_id": "read.sales.data",
                "row_path_id": "data",
            },
            {
                "evidence_id": "row_population.data",
                "type": "row_population",
                "row_source_id": "read.sales.data",
                "row_path_id": "data",
                "row_cardinality": "many",
            },
        )
    )
    support_set = parse_fulfillment_support_set(
        {
            "answer_output_id": "answer_staff",
            "fulfillment_choice_id": "fulfillment_staff",
            "fulfillment_support_set_id": "support_staff",
            "fulfillment_slots": [
                {
                    "fulfillment_slot_id": "slot_staff",
                    "entity_evidence": [evidence_items[1].payload()],
                }
            ],
        }
    )
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        evidence_items=evidence_items,
        fulfillment_support_sets=(support_set,),
    )

    fulfillments = parse_source_fulfillments(
        {
            "answer_staff": FulfillmentDecisionOutput(
                fulfillment_choice_id="fulfillment_staff",
                match_basis_explanation="Staff id is the grouped result key.",
            )
        },
        requested_fact_id="fact_1",
        answer_output_ids={"answer_staff", "answer_count"},
        required_answer_output_ids={"answer_staff", "answer_count"},
        metric_answer_output_ids={"answer_count"},
        candidate=candidate,
        plan_shape="aggregate_by_group",
        metric_fit_reviews_by_requested_output={
            "fact_1": {
                "row_population.data": {
                    "interpretation": "FITS_REQUESTED_ANSWER",
                    "metric_meaning": "count of sales rows",
                    "fit_basis": "The requested count is row cardinality.",
                }
            }
        },
    )
    count_fulfillment = next(
        fulfillment
        for fulfillment in fulfillments
        if fulfillment.answer_output_id == "answer_count"
    )
    assert count_fulfillment.row_count_basis_evidence_ids == ("row_population.data",)


def test_closed_key_source_binding_retains_input_proofs_through_grouped_count_plan():
    request = _closed_key_grouped_staff_sales_today_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    candidate = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())[
        target["source_candidate_id"]
    ]

    result = parse_source_binding(
        _closed_key_model_output_with_single_staff_param(
            binding_target_id=target["binding_target_id"],
            candidate=candidate,
            row_population_evidence_id=_only_metric_evidence_id(request),
        ),
        request=request,
    )

    assert isinstance(result.outcome, SourceBindingPlan)
    bound_source = result.outcome.bound_sources[0]
    assert _param_proofs_by_invocation(bound_source, "staff_id") == (
        (
            "51515151-0000-0000-0002-000000000001",
            ("known_input:staff_id_1",),
        ),
        (
            "51515151-0000-0000-0002-000000000002",
            ("known_input:staff_id_2",),
        ),
    )
    assert _param_proofs_by_invocation(bound_source, "start_date") == (
        ("2026-07-06", ("known_input:today",)),
        ("2026-07-06", ("known_input:today",)),
    )
    assert _param_proofs_by_invocation(bound_source, "end_date") == (
        ("2026-07-06", ("known_input:today",)),
        ("2026-07-06", ("known_input:today",)),
    )
    assert {
        applied_filter.known_input_id for applied_filter in bound_source.applied_filters
    } == {"today"}

    answer_plan, answer_bindings = compile_pattern_answer_program(
        (
            parse_pattern_answer(
                ProviderObject(
                    {
                        "requested_fact_id": "fact_1",
                        "pattern": "aggregate_by_group",
                        "source_binding_id": bound_source.id,
                        "metric": {
                            "selection_basis": "Count matching sales rows.",
                            "id": "metric_1",
                            "kind": "count_records",
                        },
                        "function": {
                            "selection_basis": "A row count uses count.",
                            "id": "function_count",
                            "value": "count",
                        },
                    }
                )
            ),
        ),
        bound_sources=result.outcome.bound_sources,
        source_binding_ids_by_requested_fact_id={"fact_1": (bound_source.id,)},
        source_binding_ids_by_requirement_by_requested_fact_id={
            "fact_1": {"operation": (bound_source.id,)}
        },
        input_context=compiler_input_context(
            values=request.available_values,
            question_contract=request.question_contract,
        ),
    )

    from fervis.lookup.answer_program.compilation import compile_answer_program
    from fervis.lookup.answer_program.instantiation import (
        ExecutionEnvironment,
        instantiate_answer_program,
    )

    program, bindings = compile_answer_program(
        answer_plan,
        question_contract=request.question_contract,
        catalog=request.relation_catalog,
        bindings=answer_bindings,
    )
    instantiate_answer_program(
        program,
        bindings,
        ExecutionEnvironment(
            catalog=request.relation_catalog,
            catalog_selection=request.catalog_selection,
        ),
    )


def test_source_binding_prompt_distinguishes_role_targets_from_fulfillment_outputs():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    prompt_text = prompt.to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context={},
        )
    ).prompt_text
    observed_target = _target_for(
        _binding_targets(prompt),
        "source_2",
        "observed_set",
    )

    assert observed_target["answer_output_ids"] == []
    assert "bind every role shown for that shape exactly once" in prompt_text
    assert "including roles with no answer outputs" in prompt_text


def test_parse_source_binding_binds_observed_target_without_answer_fulfillment():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    candidate_target = _target_for(targets, "source_1", "candidate_set")
    observed_target = _target_for(targets, "source_2", "observed_set")

    result = parse_source_binding(
        {
            "outcome": _source_binding_outcome(
                prompt,
                targets=(candidate_target, observed_target),
            )
        },
        request=request,
    )

    assert isinstance(result.outcome, SourceBindingPlan)
    bound_by_requirement = {
        bound.requirement_id: bound for bound in result.outcome.bound_sources
    }
    assert set(bound_by_requirement) == {"candidate_set", "observed_set"}
    assert bound_by_requirement["candidate_set"].fulfillments
    assert bound_by_requirement["observed_set"].fulfillments == ()


def test_parse_source_binding_rejects_incomplete_compact_role_targets():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    candidate_target = _target_for(targets, "source_1", "candidate_set")

    with pytest.raises(ValueError, match="complete source binding role set"):
        parse_source_binding(
            {
                "outcome": _source_binding_outcome(
                    prompt,
                    targets=(candidate_target,),
                )
            },
            request=request,
        )


def test_bound_plan_assembly_rejects_mixed_strategy_targets():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    mixed_targets = (
        _target_for(targets, "source_1", "candidate_set"),
        _target_for(targets, "source_1", "observed_set"),
    )
    with pytest.raises(ValueError, match="complete source binding role set"):
        parse_source_binding(
            {"outcome": _source_binding_outcome(prompt, targets=mixed_targets)},
            request=request,
        )


def test_bound_plan_assembly_rejects_untargeted_bound_source():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    complete_targets = (
        _target_for(targets, "source_1", "candidate_set"),
        _target_for(targets, "source_2", "observed_set"),
    )
    result = parse_source_binding(
        {"outcome": _source_binding_outcome(prompt, targets=complete_targets)},
        request=request,
    )
    assert isinstance(result.outcome, SourceBindingPlan)
    valid_bound = _bound_plan_selection_from_plan_selection(
        SimpleNamespace(
            plan_selection_outcome=request.plan_selection,
            source_binding_outcome=result.outcome,
            question_contract=request.question_contract,
        )
    )
    assert valid_bound is not None

    legacy_bound = replace(
        result.outcome.bound_sources[0],
        id="source_binding.legacy",
        binding_target_id="",
        requirement_id="",
    )
    bound = _bound_plan_selection_from_plan_selection(
        SimpleNamespace(
            plan_selection_outcome=request.plan_selection,
            source_binding_outcome=SourceBindingPlan(
                bound_sources=(legacy_bound, *result.outcome.bound_sources),
            ),
            question_contract=request.question_contract,
        )
    )

    assert bound is None


def test_bound_plan_assembly_keeps_auxiliary_values_out_of_target_matching():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    complete_targets = (
        _target_for(targets, "source_1", "candidate_set"),
        _target_for(targets, "source_2", "observed_set"),
    )
    result = parse_source_binding(
        {"outcome": _source_binding_outcome(prompt, targets=complete_targets)},
        request=request,
    )
    assert isinstance(result.outcome, SourceBindingPlan)
    auxiliary_value = replace(
        result.outcome.bound_sources[0],
        id="source_binding.aux_value",
        binding_target_id="",
        requirement_id="",
        source=None,
        source_invocations=(),
        value_id="value.prior_total_sales",
        source_candidate_id="value.prior_total_sales",
        fulfillments=(),
        available_field_ids=(),
        available_fields=(),
    )

    bound = _bound_plan_selection_from_plan_selection(
        SimpleNamespace(
            plan_selection_outcome=request.plan_selection,
            source_binding_outcome=SourceBindingPlan(
                bound_sources=(*result.outcome.bound_sources, auxiliary_value),
            ),
            question_contract=request.question_contract,
        )
    )

    assert bound is not None
    selected_source_binding_ids = {
        source_binding_id
        for plan in bound.plan_selections
        for member in plan.source_members
        for source_binding_id in member.source_binding_ids
    }
    assert "source_binding.aux_value" not in selected_source_binding_ids


def test_source_binding_target_compatibility_does_not_carry_evidence_selection():
    compatibility_fields = {
        field.name for field in fields(SourceBindingTargetCompatibility)
    }

    assert "fulfillment_support_set_ids" not in compatibility_fields
    assert "answer_output_ids" not in compatibility_fields


def test_value_only_bypass_does_not_choose_non_equivalent_plan_by_tuple_order():
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="percentage increase",
                answer_subject=RequestedFactAnswerSubject(subject_text="increase"),
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer_1", description="increase", role="ANSWER_VALUE"
                    ),
                ),
                answer_expression=RequestedFactAnswerExpression(
                    family=RequestedFactAnswerExpressionFamily.COMPUTED_SCALAR,
                ),
            ),
        )
    )
    plan_selection = PlanSelectionSet(
        plan_selections=(
            _value_only_plan(
                plan_id="plan.fact_1.values.a",
                value_1_candidate_id="value.current",
                value_2_candidate_id="value.previous",
            ),
            _value_only_plan(
                plan_id="plan.fact_1.values.b",
                value_1_candidate_id="value.previous",
                value_2_candidate_id="value.current",
            ),
        )
    )
    forward = value_only_source_binding_plan(
        plan_selection,
        requested_facts=question_contract.requested_facts,
    )
    reversed_order = value_only_source_binding_plan(
        PlanSelectionSet(
            plan_selections=tuple(reversed(plan_selection.plan_selections))
        ),
        requested_facts=question_contract.requested_facts,
    )

    assert forward is None
    assert reversed_order is None


def test_source_binding_target_construction_dedupes_duplicate_private_targets():
    request = _set_difference_request()
    plan = request.plan_selection.plan_selections[0]
    duplicated_member_request = replace(
        request,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                replace(
                    plan,
                    source_members=(plan.source_members[0], plan.source_members[0]),
                ),
            ),
        ),
    )

    targets = source_binding_target_index_for_plan_selection(
        duplicated_member_request.plan_selection,
        requested_facts=duplicated_member_request.requested_facts,
    ).targets

    assert len(targets) == 1
    assert targets[0].binding_target_id == (
        "target.fact_1.set_difference.source_1.candidate_set"
    )


def test_source_binding_fixture_selector_returns_compact_role_target():
    request = _set_difference_request()
    plan = request.plan_selection.plan_selections[0]
    ambiguous_request = replace(
        request,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                plan,
                replace(
                    plan,
                    plan_selection_id="plan.fact_1.c",
                    source_strategy_id="source_strategy.plan.fact_1.c",
                ),
            ),
        ),
    )
    prompt_text = (
        SourceBindingTurnPrompt(ambiguous_request)
        .to_model_invocation(
            build_turn_prompt_context(
                current_question=ambiguous_request.question,
                conversation_context={},
            )
        )
        .prompt_text
    )

    assert (
        source_binding_target_id_for_candidate(
            prompt_text,
            requested_fact_id="fact_1",
            source_candidate_id=plan.source_members[0].source_candidate_id,
            source_role="candidate",
            plan_shape="set_difference",
        )
        == "target.fact_1.set_difference.source_1.candidate_set"
    )


def _set_difference_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="Staff who have not made a sale this month.",
        answer_subject=RequestedFactAnswerSubject(subject_text="staff"),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1", description="staff", role="ANSWER_VALUE"
            ),
        ),
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SET_DIFFERENCE,
        ),
    )
    catalog = RelationCatalog(
        reads=(
            _staff_read(),
            _sales_read(),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("staff", "sales"),
                rankings=(
                    CatalogSelectionRanking(read_id="staff", score=10),
                    CatalogSelectionRanking(read_id="sales", score=9),
                ),
                selected_read_ids=("staff", "sales"),
            ),
        ),
        selected_read_ids=("staff", "sales"),
    )
    scopes = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="Which staff have not made a sale this month?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes
    scopes_by_read = {scope.read_id: scope for scope in scopes}
    read_eligibility = ReadEligibilityResult(
        read_assessments=tuple(
            ReadAssessment(
                source_candidate_id=scopes_by_read[read_id].source_candidate_id,
                source_candidate_signature=(
                    scopes_by_read[read_id].source_candidate_signature
                ),
                requested_fact_id="fact_1",
                read_id=read_id,
                relevant_row_path_ids=("data",),
                relevant_field_refs=tuple(
                    scopes_by_read[read_id].field_refs_by_evidence_token.values()
                ),
                retention_basis=f"{read_id} is needed for the set-difference answer.",
                retention_decision="RETAIN",
            )
            for read_id in ("staff", "sales")
        )
    )
    staff_candidate_id = scopes_by_read["staff"].source_candidate_id
    sales_candidate_id = scopes_by_read["sales"].source_candidate_id
    return source_binding_request(
        question="Which staff have not made a sale this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                _set_difference_plan(
                    plan_id="plan.fact_1.a",
                    candidate_source_id=staff_candidate_id,
                    observed_source_id=sales_candidate_id,
                ),
                _set_difference_plan(
                    plan_id="plan.fact_1.b",
                    candidate_source_id=sales_candidate_id,
                    observed_source_id=staff_candidate_id,
                ),
            )
        ),
        read_eligibility=read_eligibility,
    )


def _closed_key_grouped_staff_sales_request() -> SourceBindingRequest:
    staff_1 = RequestedFactLiteralInput(
        id="staff_id_1",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="51515151-0000-0000-0002-000000000001",
        resolved_value_text="51515151-0000-0000-0002-000000000001",
        field_label_text="staff_id",
        value_meaning_hint="staff member",
        role=LiteralInputRole.REFERENCE_VALUE,
    )
    staff_2 = RequestedFactLiteralInput(
        id="staff_id_2",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="51515151-0000-0000-0002-000000000002",
        resolved_value_text="51515151-0000-0000-0002-000000000002",
        field_label_text="staff_id",
        value_meaning_hint="staff member",
        role=LiteralInputRole.REFERENCE_VALUE,
    )
    fact = RequestedFact(
        id="fact_1",
        description="sales count for each specified staff member today",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_population=RequestedFactAnswerPopulation(
            population_label="sales by specified staff member",
            counted_unit="sales",
            membership_tests=(
                RequestedFactAnswerPopulationMembershipTest(
                    id="subject_identity",
                    kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question="Does the row represent a sale?",
                ),
                RequestedFactAnswerPopulationMembershipTest(
                    id="specified_staff",
                    kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question=(
                        "Does the sale belong to one of the staff members specified "
                        "by the question inputs?"
                    ),
                    owned_question_input_refs=("staff_id_1", "staff_id_2"),
                ),
                RequestedFactAnswerPopulationMembershipTest(
                    id="normal_instance_guard",
                    kind=AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD,
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question="Is this an ordinary business instance of sales?",
                ),
            ),
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_count",
                description="sales count",
                role="ROW_COUNT",
            ),
        ),
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
            group_key=RequestedFactGroupKey(
                id="answer_staff",
                description="specified staff member",
                domain=GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS,
                question_input_refs=("staff_id_1", "staff_id_2"),
            ),
        ),
        known_inputs=(staff_1, staff_2),
        input_refs=("staff_id_1", "staff_id_2"),
    )
    question_contract = QuestionContract(requested_facts=(fact,))
    available_values = (
        FactValue.identity(
            id="staff_identity_1",
            known_input_id="staff_id_1",
            key=entity_key_value(
                "staff",
                "staff_key",
                {"staff_id": "51515151-0000-0000-0002-000000000001"},
            ),
            display_value="51515151-0000-0000-0002-000000000001",
            proof_refs=("known_input:staff_id_1",),
            applies_to_requested_fact_ids=("fact_1",),
        ),
        FactValue.identity(
            id="staff_identity_2",
            known_input_id="staff_id_2",
            key=entity_key_value(
                "staff",
                "staff_key",
                {"staff_id": "51515151-0000-0000-0002-000000000002"},
            ),
            display_value="51515151-0000-0000-0002-000000000002",
            proof_refs=("known_input:staff_id_2",),
            applies_to_requested_fact_ids=("fact_1",),
        ),
    )
    catalog = RelationCatalog(reads=(_staff_sales_read(), _staff_read()))
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales", "staff"),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
        ),
        selected_read_ids=("sales",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales did the specified staff members sell each today?",
            question_contract=question_contract,
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=available_values,
        )
    ).candidate_scopes[0]
    read_eligibility = ReadEligibilityResult(
        read_assessments=(
            ReadAssessment(
                source_candidate_id=scope.source_candidate_id,
                source_candidate_signature=scope.source_candidate_signature,
                requested_fact_id="fact_1",
                read_id="sales",
                relevant_row_path_ids=("data",),
                relevant_field_refs=tuple(
                    dict.fromkeys(
                        (
                            *scope.field_refs_by_evidence_token.values(),
                            "sales.field.staff_name",
                        )
                    )
                ),
                retention_decision="RETAIN",
                retention_basis="The sales read can count sales by staff_id.",
            ),
        )
    )
    return source_binding_request(
        question="How many sales did the specified staff members sell each today?",
        question_contract=question_contract,
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.grouped_staff_sales_count",
                    requested_fact_id="fact_1",
                    source_strategy_id=(
                        "source_strategy.fact_1.grouped_staff_sales_count"
                    ),
                    plan_shape="aggregate_by_group",
                    required_answer_output_ids=("answer_staff", "answer_count"),
                    source_members=(
                        SourceStrategyMember(
                            source_candidate_id=scope.source_candidate_id,
                            requirement_ids=("operation",),
                        ),
                    ),
                    basis="Fixture-selected grouped aggregate strategy.",
                ),
            )
        ),
        available_values=available_values,
        available_value_uses=(
            GroundedInputUse(
                id="grounded_staff_1",
                value_id="staff_identity_1",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="staff_id",
            ),
            GroundedInputUse(
                id="grounded_staff_2",
                value_id="staff_identity_2",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="staff_id",
            ),
        ),
        read_eligibility=read_eligibility,
    )


def _closed_key_grouped_staff_sales_today_request() -> SourceBindingRequest:
    base = _closed_key_grouped_staff_sales_request()
    today = RequestedFactLiteralInput(
        id="today",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="today",
        resolved_value_text="today",
        field_label_text="time period",
        value_meaning_hint="current day",
        role=LiteralInputRole.TIME_VALUE,
    )
    fact = replace(
        base.requested_facts[0],
        known_inputs=(*base.requested_facts[0].known_inputs, today),
        input_refs=("staff_id_1", "staff_id_2", "today"),
    )
    question_contract = QuestionContract(requested_facts=(fact,))
    today_value = FactValue.time(
        id="today_value",
        known_input_id="today",
        expression="today",
        resolved_start="2026-07-06",
        resolved_end="2026-07-06",
        granularity="day",
        proof_refs=("known_input:today",),
        applies_to_requested_fact_ids=("fact_1",),
    )
    available_values = (*base.available_values, today_value)
    catalog = RelationCatalog(reads=(_staff_sales_today_read(), _staff_read()))
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales", "staff"),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
        ),
        selected_read_ids=("sales",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales did the specified staff members sell each today?",
            question_contract=question_contract,
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=available_values,
        )
    ).candidate_scopes[0]
    read_eligibility = ReadEligibilityResult(
        read_assessments=(
            ReadAssessment(
                source_candidate_id=scope.source_candidate_id,
                source_candidate_signature=scope.source_candidate_signature,
                requested_fact_id="fact_1",
                read_id="sales",
                relevant_row_path_ids=("data",),
                relevant_field_refs=tuple(scope.field_refs_by_evidence_token.values()),
                retention_decision="RETAIN",
                retention_basis="The sales read can count sales by staff_id and day.",
            ),
        )
    )
    return source_binding_request(
        question=base.question,
        question_contract=question_contract,
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.grouped_staff_sales_count",
                    requested_fact_id="fact_1",
                    source_strategy_id=(
                        "source_strategy.fact_1.grouped_staff_sales_count"
                    ),
                    plan_shape="aggregate_by_group",
                    required_answer_output_ids=("answer_staff", "answer_count"),
                    source_members=(
                        SourceStrategyMember(
                            source_candidate_id=scope.source_candidate_id,
                            requirement_ids=("operation",),
                        ),
                    ),
                    basis="Fixture-selected grouped aggregate strategy.",
                ),
            )
        ),
        available_values=available_values,
        available_value_uses=(
            GroundedInputUse(
                id="grounded_staff_1",
                value_id="staff_identity_1",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="staff_id",
            ),
            GroundedInputUse(
                id="grounded_staff_2",
                value_id="staff_identity_2",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="staff_id",
            ),
            GroundedInputUse(
                id="grounded_today_start",
                value_id="today_value",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="start_date",
                value_component=TimeComponent.START,
            ),
            GroundedInputUse(
                id="grounded_today_end",
                value_id="today_value",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="end_date",
                value_component=TimeComponent.END,
            ),
        ),
        read_eligibility=read_eligibility,
        same_scope_relation_catalog=base.same_scope_relation_catalog,
        memory_inputs=base.memory_inputs,
        active_memory_ids=base.active_memory_ids,
        conversation_context=base.conversation_context,
        conversation_resolution=base.conversation_resolution,
        host=base.host,
    )


def _closed_key_model_output_with_single_staff_param(
    *,
    binding_target_id: str,
    candidate: dict[str, Any],
    row_population_evidence_id: str,
) -> dict[str, Any]:
    return {
        "outcome": {
            "kind": "source_bindings",
            "metric_fit_bases": {
                "fact_1": {
                    row_population_evidence_id: {
                        "metric_meaning": "count of sales rows",
                        "fit_basis": "The requested sales count is row cardinality.",
                    }
                }
            },
            "fit_basis_interpretations": {
                "fact_1": {
                    row_population_evidence_id: {
                        "interpretation": "FITS_REQUESTED_ANSWER",
                    }
                }
            },
            **_test_fact_binding(
                requested_fact_id="fact_1",
                plan_shape="aggregate_by_group",
                requirement_id="operation",
                invocation={
                    "binding_target_id": binding_target_id,
                    "answer_population": {
                        "population_binding_id": _population_binding_id(candidate),
                        "intent_text": "sales by specified staff member today",
                        "match_basis_explanation": (
                            "Use the sales row population for the grouped count."
                        ),
                    },
                    "fulfillment_decisions": {
                        **source_fulfills_fields_for_candidate(
                            candidate,
                            field_ids_by_answer_output={
                                "answer_staff": ("staff_id",),
                            },
                        ),
                        **source_fulfills_by_row_population_for_candidate(
                            candidate,
                            answer_output_ids=("answer_count",),
                            row_path_id="data",
                        ),
                    },
                    "param_decisions": {
                        "staff_id": {
                            "param_decision_id": "ignored_backend_owned_staff_id",
                            "match_basis_explanation": (
                                "This model-authored closed-key param is ignored."
                            ),
                            "population_intent": "Do not use this decision.",
                        },
                    },
                    "row_predicate_reviews": {},
                    "finite_choice_param_reviews": {},
                },
            ),
        },
    }


def _single_param_decision(
    candidate: dict[str, Any],
    *,
    param_id: str,
    value: str,
    value_component: str = "",
) -> dict[str, str]:
    param = next(
        param for param in candidate["params"] if param["param_id"] == param_id
    )
    option = next(
        option
        for option in param["decision_options"]
        if option.get("decision") == "bind"
        and option.get("value") == value
        and str(option.get("value_component") or "") == value_component
    )
    return {
        "param_decision_id": option["param_decision_id"],
        "match_basis_explanation": f"Bind {param_id} from the grounded question input.",
        "population_intent": f"Filter by {param_id}.",
    }


def _param_proofs_by_invocation(
    bound_source: BoundSource,
    param_id: str,
) -> tuple[tuple[object, tuple[str, ...]], ...]:
    return tuple(
        (binding.value, binding.proof_refs)
        for invocation in bound_source.source_invocations
        for binding in invocation.param_bindings
        if binding.param_id == param_id
    )


def _staff_sales_read() -> EndpointRead:
    return EndpointRead(
        id="sales",
        endpoint_name="get_staff_sales",
        resource_names=("sale",),
        params=(
            CatalogParam(
                ref="sales.query.staff_id",
                name="staff_id",
                source=ParamSource.QUERY,
                type="uuid",
                required=True,
                entity_target=EntityKeyComponentTarget(
                    entity_kind="staff",
                    key_id="staff_key",
                    component_id="staff_id",
                ),
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="sales.field.sale_id",
                path="data.sale_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="sales.field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="sales.field.staff_name",
                path="data.staff_name",
                row_path_id="data",
                type="string",
            ),
        ),
        entity_references=(
            EntityReference(
                id="sale_staff",
                target_entity_kind="staff",
                target_key_id="staff_key",
                components=(
                    EntityReferenceComponent(
                        target_component_id="staff_id",
                        local_field_ref="sales.field.staff_id",
                    ),
                ),
                context_field_refs=("sales.field.staff_name",),
            ),
        ),
    )


def _staff_sales_today_read() -> EndpointRead:
    base = _staff_sales_read()
    return replace(
        base,
        params=(
            *base.params,
            CatalogParam(
                ref="sales.query.start_date",
                name="start_date",
                source=ParamSource.QUERY,
                type="date",
                required=True,
            ),
            CatalogParam(
                ref="sales.query.end_date",
                name="end_date",
                source=ParamSource.QUERY,
                type="date",
                required=True,
            ),
        ),
    )


def _set_difference_plan(
    *,
    plan_id: str,
    candidate_source_id: str,
    observed_source_id: str,
) -> SelectedSourceStrategy:
    support_set_ids_by_candidate = {
        "source_1": (
            "support.source_1.answer_1.slot.source_1.answer_1.entity."
            "source_1.data.key.staff_key"
        ),
        "source_2": (
            "support.source_2.answer_1.slot.source_2.answer_1.entity."
            "source_2.data.reference.sale_staff"
        ),
    }
    candidate_support_set_id = support_set_ids_by_candidate[candidate_source_id]
    return SelectedSourceStrategy(
        plan_selection_id=plan_id,
        requested_fact_id="fact_1",
        source_strategy_id=f"source_strategy.{plan_id}",
        plan_shape="set_difference",
        required_answer_output_ids=("answer_1",),
        source_members=(
            SourceStrategyMember(
                source_candidate_id=candidate_source_id,
                requirement_ids=("candidate_set",),
                fulfillment_support_set_ids=(candidate_support_set_id,),
            ),
            SourceStrategyMember(
                source_candidate_id=observed_source_id,
                requirement_ids=("observed_set",),
            ),
        ),
        basis="Fixture-selected set-difference strategy.",
    )


def _value_only_plan(
    *,
    plan_id: str,
    value_1_candidate_id: str,
    value_2_candidate_id: str,
) -> SelectedSourceStrategy:
    return SelectedSourceStrategy(
        plan_selection_id=plan_id,
        requested_fact_id="fact_1",
        source_strategy_id=f"source_strategy.{plan_id}",
        plan_shape="computed_scalar",
        required_answer_output_ids=("answer_1",),
        source_members=(
            SourceStrategyMember(
                source_candidate_id=value_1_candidate_id,
                requirement_ids=("value_1",),
                value_id=value_1_candidate_id,
            ),
            SourceStrategyMember(
                source_candidate_id=value_2_candidate_id,
                requirement_ids=("value_2",),
                value_id=value_2_candidate_id,
            ),
        ),
        basis="Fixture-selected value-only strategy.",
    )


def _staff_read() -> EndpointRead:
    return EndpointRead(
        id="staff",
        endpoint_name="list_staff",
        resource_names=("staff",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="staff.field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="staff.field.staff_name",
                path="data.staff_name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                id="staff_key",
                entity_kind="staff",
                components=(
                    CandidateKeyComponent(
                        id="staff_id",
                        field_ref="staff.field.staff_id",
                    ),
                ),
                primary=True,
            ),
        ),
    )


def _sales_read() -> EndpointRead:
    return EndpointRead(
        id="sales",
        endpoint_name="list_sales",
        resource_names=("sale",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="sales.field.sale_id",
                path="data.sale_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="sales.field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
            ),
        ),
        entity_references=(
            EntityReference(
                id="sale_staff",
                target_entity_kind="staff",
                target_key_id="staff_key",
                components=(
                    EntityReferenceComponent(
                        target_component_id="staff_id",
                        local_field_ref="sales.field.staff_id",
                    ),
                ),
            ),
        ),
    )


def _binding_targets(prompt: SourceBindingTurnPrompt) -> tuple[dict[str, Any], ...]:
    families = prompt.transport_context_payload()["binding_plan_families"]
    return tuple(
        target
        for fact in families["bindings_by_requested_fact"].values()
        for shape in fact["plan_shapes"].values()
        for targets in shape["role_targets"].values()
        for target in targets
    )


def _only_binding_target(prompt: SourceBindingTurnPrompt) -> dict[str, Any]:
    targets = _binding_targets(prompt)
    assert len(targets) == 1
    return targets[0]


def _only_metric_evidence_id(request: SourceBindingRequest) -> str:
    evidence_ids_by_fact = source_binding_metric_evidence_ids_by_requested_fact(request)
    evidence_ids = evidence_ids_by_fact["fact_1"]
    assert len(evidence_ids) == 1
    return evidence_ids[0]


def _target_for(
    targets: tuple[dict[str, Any], ...],
    source_candidate_id: str,
    requirement_id: str,
) -> dict[str, Any]:
    matches = [
        target
        for target in targets
        if target["source_candidate_id"] == source_candidate_id
        and target["requirement_id"] == requirement_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"target not found: {source_candidate_id}/{requirement_id}"
        )
    return matches[0]


def _source_binding_outcome(
    prompt: SourceBindingTurnPrompt,
    *,
    targets: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    candidates = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())
    invocations = []
    for target in targets:
        candidate = candidates[str(target["source_candidate_id"])]
        fulfillment_decisions = (
            source_fulfills_keys_for_candidate(
                candidate,
                key_ids_by_answer_output={"answer_1": "staff_key"},
            )
            if target["requirement_id"] == "candidate_set"
            else {}
        )
        invocations.append(
            {
                "binding_target_id": target["binding_target_id"],
                "answer_population": {
                    "population_binding_id": _population_binding_id(candidate),
                    "intent_text": f"{target['requirement_id']} rows",
                    "match_basis_explanation": (
                        f"The {target['requirement_id']} target uses this source."
                    ),
                },
                "fulfillment_decisions": fulfillment_decisions,
                "param_decisions": {},
                "row_predicate_reviews": {},
                "finite_choice_param_reviews": {},
            }
        )
    fact_bindings: dict[str, dict[str, Any]] = {}
    for target, invocation in zip(targets, invocations, strict=True):
        fact_id = str(target["requested_fact_id"])
        plan_shape = str(target["plan_shape"])
        field_id = source_binding_fact_field_id(fact_id)
        fact_binding = fact_bindings.setdefault(
            field_id,
            {"plan_shape": plan_shape},
        )
        if fact_binding["plan_shape"] != plan_shape:
            raise AssertionError("test outcome mixes source-binding plan shapes")
        fact_binding[str(target["requirement_id"])] = invocation
    return {
        "kind": "source_bindings",
        "metric_fit_bases": {},
        "fit_basis_interpretations": {},
        **fact_bindings,
    }


def _source_binding_plan_payload(*invocations: dict[str, Any]) -> dict[str, Any]:
    assert len(invocations) == 1
    return {
        "outcome": {
            "kind": "source_bindings",
            "metric_fit_bases": {},
            "fit_basis_interpretations": {},
            "bindings_for_fact_1": {
                "plan_shape": "test_shape",
                "primary": invocations[0],
            },
        }
    }


def _test_fact_binding(
    *,
    requested_fact_id: str,
    plan_shape: str,
    requirement_id: str,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    return {
        source_binding_fact_field_id(requested_fact_id): {
            "plan_shape": plan_shape,
            requirement_id: invocation,
        }
    }


def _test_plan_families(*target_ids: str) -> tuple[SourceBindingPlanFamily, ...]:
    return (
        SourceBindingPlanFamily(
            requested_fact_id="fact_1",
            plan_shape="test_shape",
            member_constraint=SourceMemberConstraint.ANY,
            required_answer_output_ids=(),
            role_targets=(
                (
                    "primary",
                    tuple(
                        SourceBindingTarget(
                            binding_target_id=target_id,
                            requested_fact_id="fact_1",
                            plan_shape="test_shape",
                            source_candidate_id=target_id.removeprefix("target."),
                            requirement_id="primary",
                        )
                        for target_id in target_ids
                    ),
                ),
            ),
        ),
    )


def _minimal_source_invocation(
    binding_target_id: str,
    population_binding_id: str,
) -> dict[str, Any]:
    return {
        "binding_target_id": binding_target_id,
        "answer_population": {
            "population_binding_id": population_binding_id,
            "intent_text": "selected rows",
            "match_basis_explanation": "This target uses the selected rows.",
        },
        "fulfillment_decisions": {},
        "param_decisions": {},
        "row_predicate_reviews": {},
        "finite_choice_param_reviews": {},
    }


def _prompt_candidates_by_id(
    payload: dict[str, object],
) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["source_candidate_id"]): candidate
        for group in payload.get("requested_fact_sources") or ()
        if isinstance(group, dict)
        for context in group.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict) and candidate.get("source_candidate_id")
    }


def _population_binding_id(candidate: dict[str, Any]) -> str:
    bindings = candidate.get("population_bindings")
    if not bindings:
        bindings = (candidate.get("binding_surface") or {}).get("population_bindings")
    return str(bindings[0]["population_binding_id"])


def _source_invocation_variants_by_target(
    schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        target_schema = properties.get("binding_target_id")
        if not isinstance(target_schema, dict):
            continue
        target_ids = target_schema.get("enum")
        if isinstance(target_ids, list) and len(target_ids) == 1:
            output[str(target_ids[0])] = node
    return output


def _schema_nodes(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _schema_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _schema_nodes(child)


def _source_binding_outcome_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return next(
        variant
        for variant in schema["properties"]["outcome"]["oneOf"]
        if variant["properties"]["kind"].get("enum") == ["source_bindings"]
    )


def _flatten_source_invocation_item_variants(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    properties = schema.get("properties")
    if isinstance(properties, dict) and "binding_target_id" in properties:
        return (schema,)
    variants = schema.get("oneOf")
    if not isinstance(variants, list):
        return ()
    return tuple(
        item
        for variant in variants
        if isinstance(variant, dict)
        for item in _flatten_source_invocation_item_variants(variant)
    )
