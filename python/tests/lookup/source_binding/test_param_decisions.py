from __future__ import annotations

from dataclasses import asdict, replace

import pytest
from jsonschema import ValidationError, validate

from fervis.lookup.source_binding.candidates import (
    SourceCandidate,
    source_candidate_required_param_decision_ids,
)
from fervis.lookup.source_binding.candidates.model import (
    CandidateBindingValue,
    CandidateParameter,
    CandidateParamDecision,
)
from fervis.lookup.source_binding.candidates.contracts import (
    EntityTarget,
    EntityReferenceEvidence,
    EvidenceComponent,
    FieldEvidence,
)
from fervis.lookup.source_binding.candidates.params import (
    _candidate_with_param_decision_options,
    _candidate_with_param_population_contracts,
    _param_bind_options,
    _param_binding_values,
    _param_omit_option,
)
from fervis.lookup.source_binding.candidates.api_sources import (
    _api_candidate_payload,
)
from fervis.lookup.answer_program.inputs import resolve_value_expression
from fervis.lookup.answer_program.values import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    FactValue,
    LiteralType,
    ParameterBinding,
    ParameterRef,
)
from fervis.lookup.canonical_data import entity_key_value
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    GroupKeyDomainKind,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    ResultSelectionKind,
    RequestedFactAnswerOutput,
    RequestedFactGroupKey,
    RequestedFactAnswerPopulationMembershipTest,
)
from fervis.lookup.answer_program.operations import PredicateOperator
from fervis.lookup.answer_program.relations import PopulationCoverageRole
from fervis.lookup.source_binding.model import AnswerPopulation
from fervis.lookup.source_binding.compiler_ir import RelationInputOrigin
from fervis.lookup.source_binding.input_applications import (
    ResolvedInputApplicationTargetKind,
    parse_resolved_input_applications,
    resolved_input_application_surface,
)
from fervis.lookup.source_binding.parser.params import parse_param_decision_binding_sets
from fervis.lookup.source_binding.parser.types import NormalizedParamDecision
from fervis.lookup.source_binding.provider_contract import (
    ResolvedInputApplicationOutput,
    ResolvedInputTargetApplicationOutput,
)


def test_choice_parameter_requires_a_decision_even_when_it_has_a_default() -> None:
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="api",
        params=(
            CandidateParameter(
                id="status",
                type="string",
                required=False,
                choices=("OPEN", "CLOSED"),
                decision_options=(
                    CandidateParamDecision(
                        id="status.use_default",
                        decision="use_default",
                    ),
                ),
                has_default=True,
                default="OPEN",
            ),
        ),
    )

    assert source_candidate_required_param_decision_ids(candidate) == ("status",)


def test_optional_static_boolean_param_exposes_omit_decision():
    param = {
        "param_id": "is_open",
        "source": "query",
        "type": "boolean",
        "required": False,
        "binding_values": [
            {"value": "true", "label": "true", "source": "static_choice"},
            {"value": "false", "label": "false", "source": "static_choice"},
        ],
    }
    bind_options = _param_bind_options(param)

    candidate = _candidate_with_param_decision_options(
        {
            "source_candidate_id": "source_1",
            "params": [
                {
                    **param,
                    "bind_options": bind_options,
                    "omit_option": _param_omit_option(
                        param,
                        bind_options=bind_options,
                    ),
                }
            ],
        }
    )

    decision_options = candidate["params"][0]["decision_options"]
    omit_options = [
        option for option in decision_options if option.get("decision") == "omit"
    ]

    assert len(omit_options) == 1
    assert (
        omit_options[0]["param_decision_id"] == "param_decision.source_1.is_open.omit"
    )
    assert "true and false" in omit_options[0]["meaning"]

    candidate = _candidate_with_param_population_contracts(candidate)

    assert "population_contract" not in candidate["params"][0]


def test_resolved_identity_param_is_not_a_param_decision():
    identity = FactValue.identity(
        id="customer_1",
        key=entity_key_value(
            "customer",
            "primary_key",
            {"customer_id": "customer-1"},
        ),
        display_value="Alice",
    )
    param = {
        "param_id": "customer_id",
        "source": "query",
        "type": "uuid",
        "required": False,
        "entity_target": {
            "entity_kind": "customer",
            "key_id": "primary_key",
            "component_id": "customer_id",
        },
    }
    param["binding_values"] = _param_binding_values(
        param,
        available_values=(identity,),
    )
    candidate = _candidate_with_param_decision_options(
        {
            "source_candidate_id": "source_1",
            "params": [param],
        }
    )

    assert "decision_options" not in candidate["params"][0]


def test_api_candidate_preserves_required_parameter_without_entity_target():
    candidate = _api_candidate_payload(
        {
            "read_id": "get_staff_sales",
            "row_source_id": "api:get_staff_sales:root",
            "row_path_id": "root",
            "read_row_source_count": 1,
            "params": [
                {
                    "param_id": "staff_id",
                    "source": "path",
                    "type": "uuid",
                    "required": True,
                }
            ],
        },
        available_values=(),
    )

    assert candidate["params"] == [
        {
            "param_id": "staff_id",
            "source": "path",
            "type": "uuid",
            "required": True,
        }
    ]


def test_api_candidate_exposes_only_compatible_optional_identity_parameter():
    identity = FactValue.identity(
        id="staff_1",
        key=entity_key_value(
            "staff",
            "primary_key",
            {"staff_id": "51515151-0000-0000-0002-000000000001"},
        ),
    )
    candidate = _api_candidate_payload(
        {
            "read_id": "list_staff_sales",
            "row_source_id": "api:list_staff_sales:root",
            "row_path_id": "root",
            "read_row_source_count": 1,
            "params": [
                {
                    "param_id": "staff_id",
                    "source": "query",
                    "type": "uuid",
                    "required": False,
                },
                {
                    "param_id": "page",
                    "source": "query",
                    "type": "integer",
                    "required": False,
                },
            ],
        },
        available_values=(identity,),
    )

    assert [param["param_id"] for param in candidate["params"]] == ["staff_id"]


def test_resolved_identity_component_compiles_to_compatible_raw_parameter():
    identity = FactValue.identity(
        id="staff_1",
        key=entity_key_value(
            "staff",
            "primary_key",
            {"staff_id": "51515151-0000-0000-0002-000000000001"},
        ),
        known_input_id="staff_member_id_1",
        applies_to_requested_fact_ids=("fact_1",),
    )
    param = CandidateParameter(
        id="staff_id",
        type="uuid",
        required=True,
        choices=(),
        decision_options=(),
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="get_staff_sales",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=(param,),
        ),
        requested_fact_id="fact_1",
        resolved_values=(identity,),
    )

    assert surface.prompt_payload()["resolved_values"] == [
            {
                "value_id": "staff_1",
                "kind": "identity",
                "value_meaning": (
                    "{'staff_id': '51515151-0000-0000-0002-000000000001'}"
                ),
            "components_by_target_kind": {"request_parameter": ["staff_id"]},
            "population_test_basis": {},
        }
    ]
    parsed = parse_resolved_input_applications(
        (
            _application(
                identity.id,
                param.id,
                value_component="staff_id",
            ),
        ),
        surface=surface,
    )

    binding = parsed.param_binding_sets[0][0]
    assert binding.param_id == "staff_id"
    assert binding.value == "51515151-0000-0000-0002-000000000001"
    assert binding.value_component == "key_component:staff_id"


def test_returned_identity_supersedes_optional_targetless_identity_parameter():
    identity = FactValue.identity(
        id="area_1",
        key=entity_key_value(
            "area",
            "primary_key",
            {"area_id": "1607d0c3-c237-427d-9d2b-f9ef100bcf9d"},
        ),
        known_input_id="area_name",
        applies_to_requested_fact_ids=("fact_1",),
    )
    reference = EntityReferenceEvidence(
        evidence_id="source_1.data.reference.area",
        reference_id="area_reference",
        target_key_id="primary_key",
        target_entity_kind="area",
        components=(
            EvidenceComponent(
                component_id="area_id",
                field_id="area_id",
                field_evidence_id="field.area_id",
            ),
        ),
        row_path_id="data",
        row_source_id="locations",
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="list_locations",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=(
                CandidateParameter(
                    id="name",
                    type="string",
                    required=False,
                    choices=(),
                    decision_options=(),
                ),
            ),
            evidence_items=(reference,),
        ),
        requested_fact_id="fact_1",
        resolved_values=(identity,),
    )

    assert surface.prompt_payload()["targets_by_kind"]["request_parameter"] == []
    assert surface.prompt_payload()["resolved_values"] == [
        {
            "value_id": "area_1",
            "kind": "identity",
            "value_meaning": (
                "{'area_id': '1607d0c3-c237-427d-9d2b-f9ef100bcf9d'}"
            ),
            "components_by_target_kind": {"returned_identity": ["canonical_key"]},
            "population_test_basis": {},
        }
    ]


def test_grouped_identity_inputs_compile_as_parameter_alternatives() -> None:
    fact = RequestedFact(
        id="fact_1",
        description="sales count for each specified staff member",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
            group_key=RequestedFactGroupKey(
                id="answer_staff",
                description="specified staff member",
                domain=GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS,
                question_input_refs=("staff_id_1", "staff_id_2"),
            ),
            selection_kind=ResultSelectionKind.ALL_RESULTS,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_count",
                description="sales count",
                role="ROW_COUNT",
            ),
        ),
        input_refs=("staff_id_1", "staff_id_2"),
    )
    identities = tuple(
        FactValue.identity(
            id=f"staff_{index}",
            key=entity_key_value(
                "staff",
                "primary_key",
                {"staff_id": staff_id},
            ),
            known_input_id=f"staff_id_{index}",
            proof_refs=(f"known_input:staff_id_{index}",),
            applies_to_requested_fact_ids=("fact_1",),
        )
        for index, staff_id in enumerate(
            (
                "51515151-0000-0000-0002-000000000001",
                "51515151-0000-0000-0002-000000000002",
            ),
            start=1,
        )
    )
    param = CandidateParameter(
        id="staff_id",
        type="uuid",
        required=True,
        choices=(),
        decision_options=(),
    )
    alternative_group = fact.specified_group_key()
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="get_staff_sales",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=(param,),
        ),
        requested_fact_id="fact_1",
        resolved_values=identities,
        parameter_alternative_group=alternative_group,
    )

    assert alternative_group is fact.answer_expression.group_key
    assert surface.provider_schema()["maxItems"] == 2
    assert [
        value["request_parameter_alternative_group"]
        for value in surface.prompt_payload()["resolved_values"]
    ] == ["answer_staff", "answer_staff"]

    applications = tuple(
        _application(value.id, param.id, value_component="staff_id")
        for value in identities
    )
    parsed = parse_resolved_input_applications(
        applications,
        surface=surface,
    )

    assert [
        tuple((binding.param_id, binding.compiler_value) for binding in binding_set)
        for binding_set in parsed.param_binding_sets
    ] == [
        (("staff_id", "51515151-0000-0000-0002-000000000001"),),
        (("staff_id", "51515151-0000-0000-0002-000000000002"),),
    ]
    assert parsed.population_coverage_claims == ()
    with pytest.raises(ValueError, match="repeats a target"):
        parse_resolved_input_applications(
            applications,
            surface=replace(surface, parameter_alternative_group=None),
        )
    with pytest.raises(ValueError, match="alternatives must apply together"):
        parse_resolved_input_applications(
            applications[:1],
            surface=surface,
        )


def test_raw_parameter_rejects_incompatible_identity_component_type():
    identity = FactValue.identity(
        id="staff_1",
        key=entity_key_value(
            "staff",
            "primary_key",
            {"staff_id": "not-a-uuid"},
        ),
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="get_staff_sales",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=(
                CandidateParameter(
                    id="staff_id",
                    type="uuid",
                    required=True,
                    choices=(),
                    decision_options=(),
                ),
            ),
        ),
        requested_fact_id="fact_1",
        resolved_values=(identity,),
    )

    assert surface.parameter_targets_by_id == {}
    assert surface.prompt_payload()["resolved_values"] == []


def test_resolved_values_compile_through_explicit_input_applications():
    identity = FactValue.identity(
        id="customer_1",
        key=entity_key_value(
            "customer",
            "primary_key",
            {"customer_id": "customer-1"},
        ),
        proof_refs=("known_input:customer",),
    )
    period = FactValue.time(
        id="period_1",
        expression="this month",
        resolved_start="2026-07-01",
        resolved_end="2026-07-31",
        granularity="month",
        proof_refs=("known_input:period",),
    )
    threshold = FactValue.literal(
        id="threshold_1",
        literal_type=LiteralType.NUMBER,
        value="25",
        proof_refs=("known_input:threshold",),
    )
    params = (
        _candidate_param(
            param_id="customer_id",
            value_id=identity.id,
        ),
        _candidate_param(
            param_id="start_date",
            value_id=period.id,
            value_component="start",
        ),
        _candidate_param(
            param_id="minimum_total",
            value_id=threshold.id,
        ),
    )
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        params=params,
    )

    surface = resolved_input_application_surface(
        candidate=candidate,
        requested_fact_id="fact_1",
        resolved_values=(identity, period, threshold),
    )
    parsed = parse_resolved_input_applications(
        (
            _application(
                identity.id,
                "customer_id",
                value_component="customer_id",
            ),
            _application(period.id, "start_date", value_component="start"),
            _application(threshold.id, "minimum_total"),
        ),
        surface=surface,
    )

    assert len(parsed.param_binding_sets) == 1
    bindings = {binding.param_id: binding for binding in parsed.param_binding_sets[0]}
    assert bindings["customer_id"].value_id == identity.id
    assert bindings["customer_id"].value == "customer-1"
    assert bindings["start_date"].value_id == period.id
    assert bindings["start_date"].value_component == "start"
    assert bindings["start_date"].value == "2026-07-01"
    assert bindings["minimum_total"].value_id == threshold.id
    assert bindings["minimum_total"].value == "25"


def test_one_time_input_proves_its_constraint_through_both_bounds() -> None:
    period = FactValue.time(
        id="period_1",
        expression="March 2026",
        resolved_start="2026-03-01",
        resolved_end="2026-03-31",
        granularity="month",
        known_input_id="qi_period",
        proof_refs=("known_input:qi_period",),
    )
    test = _predicate_test(input_id="qi_period")
    params = tuple(
        _candidate_param(
            param_id=param_id,
            value_id=period.id,
            value_component=component,
        )
        for param_id, component in (("start_date", "start"), ("end_date", "end"))
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=params,
        ),
        requested_fact_id="fact_1",
        resolved_values=(period,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )

    parsed = parse_resolved_input_applications(
        (
            _application_group(
                period.id,
                (("start_date", "start"), ("end_date", "end")),
                population_test_results=_satisfies_test_result(),
            ),
        ),
        surface=surface,
    )

    assert [binding.param_id for binding in parsed.param_binding_sets[0]] == [
        "start_date",
        "end_date",
    ]
    assert parsed.population_coverage_claims[0].proof_refs == (
        "known_input:qi_period",
        "source_param:start_date",
        "source_param:end_date",
    )


def test_one_compound_predicate_value_applies_through_two_parameters() -> None:
    predicate = FactValue.literal(
        id="completed_in_person",
        known_input_id="qi_predicate",
        literal_type=LiteralType.STRING,
        value="completed in-person",
        proof_refs=("known_input:qi_predicate",),
    )
    test = replace(
        _predicate_test(input_id="qi_predicate"),
        owned_question_input_refs=("qi_predicate", "qi_period"),
    )
    params = (
        CandidateParameter(
            id="status",
            type="choice",
            required=False,
            choices=("COMPLETED", "PENDING"),
            decision_options=(),
        ),
        CandidateParameter(
            id="channel",
            type="choice",
            required=False,
            choices=("IN_PERSON", "ONLINE"),
            decision_options=(),
        ),
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=params,
        ),
        requested_fact_id="fact_1",
        resolved_values=(predicate,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )
    application_value_ids = {
        (target_id, raw_value): application_value_id
        for application_value_id, (_, target_id, raw_value) in (
            surface.application_values_by_id.items()
        )
    }

    parsed = parse_resolved_input_applications(
        (
            ResolvedInputApplicationOutput(
                value_id=predicate.id,
                applications=(
                    ResolvedInputTargetApplicationOutput(
                        application_target_id=application_value_ids[
                            ("status", "COMPLETED")
                        ],
                        value_component="value",
                        match_basis_explanation=(
                            "Completed selects the completed status."
                        ),
                    ),
                    ResolvedInputTargetApplicationOutput(
                        application_target_id=application_value_ids[
                            ("channel", "IN_PERSON")
                        ],
                        value_component="value",
                        match_basis_explanation=(
                            "In-person selects the in-person channel."
                        ),
                    ),
                ),
                population_test_results=_satisfies_test_result(),
            ),
        ),
        surface=surface,
    )

    assert [
        (binding.param_id, binding.compiler_value)
        for binding in parsed.param_binding_sets[0]
    ] == [("status", "COMPLETED"), ("channel", "IN_PERSON")]
    assert all(
        binding.origin_kind is RelationInputOrigin.CONTEXT_CONSTANT
        and not binding.value_id
        for binding in parsed.param_binding_sets[0]
    )
    assert parsed.population_coverage_claims[0].proof_refs == (
        "known_input:qi_predicate",
        "source_param:status",
        "source_param:channel",
        "known_input:qi_period",
    )


def test_resolved_identity_set_expands_scalar_entity_parameter_bindings():
    identities = FactValue.identity_set(
        id="staff_ids",
        keys=(
            entity_key_value(
                "staff",
                "primary_key",
                {"staff_id": "51515151-0000-0000-0002-000000000001"},
            ),
            entity_key_value(
                "staff",
                "primary_key",
                {"staff_id": "51515151-0000-0000-0002-000000000002"},
            ),
        ),
        proof_refs=("known_input:staff",),
    )
    param = CandidateParameter(
        id="staff_id",
        type="uuid",
        required=True,
        choices=(),
        binding_values=(
            CandidateBindingValue(
                value=identities.id,
                source="available_value",
            ),
        ),
        decision_options=(),
        entity_target=EntityTarget(
            entity_kind="staff",
            key_id="primary_key",
            component_id="staff_id",
        ),
    )
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        params=(param,),
    )
    surface = resolved_input_application_surface(
        candidate=candidate,
        requested_fact_id="fact_1",
        resolved_values=(identities,),
    )

    parsed = parse_resolved_input_applications(
        (_application(identities.id, param.id),),
        surface=surface,
    )

    assert [
        (
            binding.compiler_value,
            binding.value_component,
            binding.value_item_index,
        )
        for binding_set in parsed.param_binding_sets
        for binding in binding_set
    ] == [
        (
            "51515151-0000-0000-0002-000000000001",
            "key_component:staff_id",
            0,
        ),
        (
            "51515151-0000-0000-0002-000000000002",
            "key_component:staff_id",
            1,
        ),
    ]
    program_parameter_id = "question.staff_ids"
    program_bindings = BindingSet.from_bindings(
        (
            ParameterBinding(
                parameter_id=program_parameter_id,
                value=identities,
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.QUESTION_INPUT,
                ),
            ),
        )
    )
    assert [
        resolve_value_expression(
            ParameterRef(
                parameter_id=program_parameter_id,
                component=binding.value_component,
                item_index=binding.value_item_index,
            ),
            bindings=program_bindings,
        ).value
        for binding_set in parsed.param_binding_sets
        for binding in binding_set
    ] == [
        "51515151-0000-0000-0002-000000000001",
        "51515151-0000-0000-0002-000000000002",
    ]


def test_composite_identity_set_preserves_key_component_correlation():
    identities = FactValue.identity_set(
        id="staff_ids",
        keys=(
            entity_key_value(
                "staff",
                "composite_key",
                {
                    "organization_id": "51515151-0000-0000-0001-000000000001",
                    "staff_id": "51515151-0000-0000-0002-000000000001",
                },
            ),
            entity_key_value(
                "staff",
                "composite_key",
                {
                    "organization_id": "51515151-0000-0000-0001-000000000002",
                    "staff_id": "51515151-0000-0000-0002-000000000002",
                },
            ),
        ),
        proof_refs=("known_input:staff",),
    )
    params = tuple(
        CandidateParameter(
            id=component_id,
            type="uuid",
            required=True,
            choices=(),
            binding_values=(
                CandidateBindingValue(
                    value=identities.id,
                    source="available_value",
                ),
            ),
            decision_options=(),
            entity_target=EntityTarget(
                entity_kind="staff",
                key_id="composite_key",
                component_id=component_id,
            ),
        )
        for component_id in ("organization_id", "staff_id")
    )
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        params=params,
    )
    surface = resolved_input_application_surface(
        candidate=candidate,
        requested_fact_id="fact_1",
        resolved_values=(identities,),
    )

    parsed = parse_resolved_input_applications(
        (
            _application_group(
                identities.id,
                tuple((param.id, "value") for param in params),
            ),
        ),
        surface=surface,
    )

    assert [
        tuple((binding.param_id, binding.compiler_value) for binding in binding_set)
        for binding_set in parsed.param_binding_sets
    ] == [
        (
            (
                "organization_id",
                "51515151-0000-0000-0001-000000000001",
            ),
            ("staff_id", "51515151-0000-0000-0002-000000000001"),
        ),
        (
            (
                "organization_id",
                "51515151-0000-0000-0001-000000000002",
            ),
            ("staff_id", "51515151-0000-0000-0002-000000000002"),
        ),
    ]


def test_resolved_input_contract_lists_values_and_targets_without_pair_product():
    values = (
        FactValue.literal(
            id="minimum_amount",
            literal_type=LiteralType.NUMBER,
            value="10",
        ),
        FactValue.literal(
            id="maximum_amount",
            literal_type=LiteralType.NUMBER,
            value="20",
        ),
    )
    params = tuple(
        CandidateParameter(
            id=param_id,
            type="number",
            required=False,
            choices=(),
            binding_values=tuple(
                CandidateBindingValue(
                    value=value.id,
                    source="available_value",
                )
                for value in values
            ),
            decision_options=(),
        )
        for param_id in ("lower_bound", "upper_bound")
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=params,
        ),
        requested_fact_id="fact_1",
        resolved_values=values,
    )

    assert surface.prompt_payload() == {
        "resolved_values": [
                {
                    "value_id": "minimum_amount",
                    "kind": "literal",
                    "value_meaning": "minimum_amount",
                    "literal_type": "number",
                    "literal_value": "10",
                "components_by_target_kind": {
                    "request_parameter": ["value"],
                },
                "population_test_basis": {},
            },
                {
                    "value_id": "maximum_amount",
                    "kind": "literal",
                    "value_meaning": "maximum_amount",
                    "literal_type": "number",
                    "literal_value": "20",
                "components_by_target_kind": {
                    "request_parameter": ["value"],
                },
                "population_test_basis": {},
            },
        ],
        "targets_by_kind": {
            "request_parameter": [
                {
                    "target_id": "lower_bound",
                    "application_target_id": "request_parameter.lower_bound",
                    "type": "number",
                },
                {
                    "target_id": "upper_bound",
                    "application_target_id": "request_parameter.upper_bound",
                    "type": "number",
                },
            ],
            "returned_identity": [],
            "returned_field": [],
        },
        "predicate_requirements": [],
    }
    item_variants = surface.provider_schema()["items"]["oneOf"]

    assert [variant["properties"]["value_id"]["enum"] for variant in item_variants] == [
        ["minimum_amount"],
        ["maximum_amount"],
    ]
    assert all(
        variant["properties"]["applications"]["items"]["properties"][
            "application_target_id"
        ]["enum"]
        == ["request_parameter.lower_bound", "request_parameter.upper_bound"]
        for variant in item_variants
    )


def test_resolved_input_contract_exposes_only_each_values_owned_tests() -> None:
    values = tuple(
        FactValue.literal(
            id=f"value_{index}",
            known_input_id=f"input_{index}",
            literal_type=LiteralType.STRING,
            value=value,
        )
        for index, value in enumerate(("ready", "finished"), start=1)
    )
    tests = tuple(
        RequestedFactAnswerPopulationMembershipTest(
            id=f"test_{index}",
            kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
            polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
            test_question=f"Does the row satisfy predicate {index}?",
            owned_question_input_refs=(f"input_{index}",),
        )
        for index in (1, 2)
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=(
                CandidateParameter(
                    id="state",
                    type="string",
                    required=False,
                    choices=(),
                    decision_options=(),
                ),
            ),
        ),
        requested_fact_id="fact_1",
        resolved_values=values,
        membership_tests=tests,
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )
    schema = surface.provider_schema()["items"]
    result = {
        "test_id": "explicit_user_constraint:test_1",
        "test_question": tests[0].test_question,
        "role_scoped_test_question": (
            f"For source rows, {tests[0].test_question}"
        ),
        "because": "The selected target applies this predicate.",
        "test_effect": "SATISFIES_TEST",
    }
    application = {
        "value_id": "value_1",
        "applications": [
            {
                "application_target_id": "request_parameter.state",
                "value_component": "value",
                "match_basis_explanation": "Apply the first predicate.",
            }
        ],
        "population_test_results": {
            "explicit_user_constraint:test_1": result,
        },
    }

    validate(instance=application, schema=schema)
    application["population_test_results"] = {
        "explicit_user_constraint:test_2": {
            **result,
            "test_id": "explicit_user_constraint:test_2",
        }
    }
    with pytest.raises(ValidationError):
        validate(instance=application, schema=schema)


def test_resolved_identity_compiles_against_one_declared_entity_reference():
    identity = FactValue.identity(
        id="area_1",
        key=entity_key_value(
            "area",
            "primary_key",
            {"country_code": "KE", "area_code": "NRB"},
        ),
        known_input_id="area_name",
        proof_refs=("known_input:area_name",),
        applies_to_requested_fact_ids=("fact_1",),
    )
    reference = EntityReferenceEvidence(
        evidence_id="source_1.root.reference.area",
        reference_id="area_reference",
        target_key_id="primary_key",
        target_entity_kind="area",
        components=(
            EvidenceComponent(
                component_id="country_code",
                field_id="country_code",
                field_evidence_id="field.country_code",
            ),
            EvidenceComponent(
                component_id="area_code",
                field_id="area_code",
                field_evidence_id="field.area_code",
            ),
        ),
        row_path_id="root",
        row_source_id="locations",
    )
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        evidence_items=(reference,),
    )
    surface = resolved_input_application_surface(
        candidate=candidate,
        requested_fact_id="fact_1",
        resolved_values=(identity,),
    )

    parsed = parse_resolved_input_applications(
        (
            _application(
                identity.id,
                reference.evidence_id,
                target_kind=(
                    ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value
                ),
                value_component="canonical_key",
            ),
        ),
        surface=surface,
    )

    assert [item.predicate_field_ids for item in parsed.applied_filters] == [
        ("country_code",),
        ("area_code",),
    ]
    assert [item.value_component for item in parsed.applied_filters] == [
        "key_component:country_code",
        "key_component:area_code",
    ]


def _candidate_param(
    *,
    param_id: str,
    value_id: str,
    value_component: str = "",
) -> CandidateParameter:
    return CandidateParameter(
        id=param_id,
        type="string",
        required=True,
        choices=(),
        binding_values=(
            CandidateBindingValue(
                value=value_id,
                source="available_value",
                value_component=value_component,
            ),
        ),
        decision_options=(),
    )


def _application(
    value_id: str,
    target_id: str,
    *,
    value_component: str = "value",
    target_kind: str = ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value,
    application_target_id: str | None = None,
    population_test_results: dict[str, object] | None = None,
) -> ResolvedInputApplicationOutput:
    return _application_group(
        value_id,
        ((target_id, value_component),),
        target_kind=target_kind,
        application_target_id=application_target_id,
        population_test_results=population_test_results,
    )


def _application_group(
    value_id: str,
    targets: tuple[tuple[str, str], ...],
    *,
    target_kind: str = ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value,
    application_target_id: str | None = None,
    population_test_results: dict[str, object] | None = None,
) -> ResolvedInputApplicationOutput:
    return ResolvedInputApplicationOutput(
        value_id=value_id,
        applications=tuple(
            ResolvedInputTargetApplicationOutput(
                application_target_id=(
                    application_target_id or f"{target_kind}.{target_id}"
                ),
                value_component=value_component,
                match_basis_explanation=(
                    "The resolved input restricts this source."
                ),
            )
            for target_id, value_component in targets
        ),
        population_test_results=population_test_results or {},
    )


def _predicate_test(
    *,
    input_id: str,
    polarity: AnswerPopulationMembershipTestPolarity = (
        AnswerPopulationMembershipTestPolarity.MUST_PASS
    ),
    operator: PredicateOperator | None = None,
) -> RequestedFactAnswerPopulationMembershipTest:
    return RequestedFactAnswerPopulationMembershipTest(
        id="test_operand",
        kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
        polarity=polarity,
        test_question="Does the row satisfy the requested value predicate?",
        owned_question_input_refs=(input_id,),
        comparison_operator=operator,
    )


def _satisfies_test_result(
    question: str = "Does the row satisfy the requested value predicate?",
    *,
    test_effect: str = "SATISFIES_TEST",
):
    from fervis.lookup.source_binding.provider_contract import (
        RowPredicatePopulationTestResultOutput,
    )

    return {
        "explicit_user_constraint:test_operand": RowPredicatePopulationTestResultOutput(
            test_id="explicit_user_constraint:test_operand",
            test_question=question,
            role_scoped_test_question=(
                f"For source rows, {question}"
            ),
            because="The selected physical target enforces the fixed predicate.",
            test_effect=test_effect,
        )
    }


def test_threshold_predicate_can_bind_one_declared_request_parameter() -> None:
    value = FactValue.literal(
        id="minimum_amount",
        known_input_id="qi_amount",
        literal_type=LiteralType.NUMBER,
        value="1000",
        proof_refs=("known_input:qi_amount",),
    )
    test = _predicate_test(input_id="qi_amount", operator=PredicateOperator.GT)
    param = CandidateParameter(
        id="minimum_amount",
        type="number",
        required=False,
        choices=(),
        decision_options=(),
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=(param,),
        ),
        requested_fact_id="fact_1",
        resolved_values=(value,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )

    parsed = parse_resolved_input_applications(
        (
            _application(
                value.id,
                param.id,
                population_test_results=_satisfies_test_result(),
            ),
        ),
        surface=surface,
    )

    assert parsed.param_binding_sets[0][0].param_id == "minimum_amount"
    assert parsed.param_binding_sets[0][0].compiler_value == "1000"
    assert [claim.test_ref.membership_test_id for claim in parsed.population_coverage_claims] == [
        "test_operand"
    ]


def test_threshold_predicate_can_filter_one_declared_returned_field() -> None:
    value = FactValue.literal(
        id="minimum_amount",
        known_input_id="qi_amount",
        literal_type=LiteralType.NUMBER,
        value="1000",
        proof_refs=("known_input:qi_amount",),
    )
    test = _predicate_test(input_id="qi_amount", operator=PredicateOperator.GTE)
    field = FieldEvidence(
        evidence_id="field.amount",
        field_id="data.amount",
        type="number",
        row_path_id="data",
        row_source_id="sales",
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            evidence_items=(field,),
        ),
        requested_fact_id="fact_1",
        resolved_values=(value,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )

    parsed = parse_resolved_input_applications(
        (
            _application(
                value.id,
                field.evidence_id,
                target_kind=ResolvedInputApplicationTargetKind.RETURNED_FIELD.value,
                population_test_results=_satisfies_test_result(),
            ),
        ),
        surface=surface,
    )

    assert len(parsed.applied_filters) == 1
    applied = parsed.applied_filters[0]
    assert applied.predicate_field_ids == ("data.amount",)
    assert applied.operator == PredicateOperator.GTE.value
    assert applied.value_id == value.id


def test_predicate_choice_requires_one_target_local_application_value() -> None:
    value = FactValue.literal(
        id="requested_state",
        known_input_id="qi_state",
        literal_type=LiteralType.STRING,
        value="finished",
        proof_refs=("known_input:qi_state",),
    )
    test = _predicate_test(input_id="qi_state")
    param = CandidateParameter(
        id="state",
        type="choice",
        required=False,
        choices=("DONE", "OPEN"),
        decision_options=(),
    )
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        params=(param,),
    )
    surface = resolved_input_application_surface(
        candidate=candidate,
        requested_fact_id="fact_1",
        resolved_values=(value,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )
    selected_id = next(
        application_value_id
        for application_value_id, (_, _, raw_value) in (
            surface.application_values_by_id.items()
        )
        if raw_value == "DONE"
    )

    with pytest.raises(ValueError, match="unknown application target"):
        parse_resolved_input_applications(
            (
                _application(
                    value.id,
                    param.id,
                    population_test_results=_satisfies_test_result(),
                ),
            ),
            surface=surface,
        )

    parsed = parse_resolved_input_applications(
        (
            _application(
                value.id,
                param.id,
                application_target_id=selected_id,
                population_test_results=_satisfies_test_result(),
            ),
        ),
        surface=surface,
    )
    assert parsed.param_binding_sets[0][0].compiler_value == "DONE"


def test_resolved_input_schema_uses_one_target_bound_application_id() -> None:
    value = FactValue.literal(
        id="requested_state",
        known_input_id="qi_state",
        literal_type=LiteralType.STRING,
        value="finished",
    )
    test = _predicate_test(input_id="qi_state")
    param = CandidateParameter(
        id="state",
        type="choice",
        required=False,
        choices=("DONE", "OPEN"),
        decision_options=(),
    )
    field = FieldEvidence(
        evidence_id="field.state",
        field_id="data.state",
        type="string",
        row_path_id="data",
        row_source_id="records",
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            params=(param,),
            evidence_items=(field,),
        ),
        requested_fact_id="fact_1",
        resolved_values=(value,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )
    selected_id = next(iter(surface.application_values_by_id))
    population_test_results = {
        test_id: asdict(result)
        for test_id, result in _satisfies_test_result().items()
    }
    common = {
        "value_id": value.id,
        "population_test_results": population_test_results,
    }

    validate(
        instance=[
            {
                **common,
                "applications": [
                    {
                        "application_target_id": (
                            f"returned_field.{field.evidence_id}"
                        ),
                        "value_component": "value",
                        "match_basis_explanation": (
                            "The returned state applies the requested predicate."
                        ),
                    }
                ],
            }
        ],
        schema=surface.provider_schema(),
    )
    validate(
        instance=[
            {
                **common,
                "applications": [
                    {
                        "application_target_id": selected_id,
                        "value_component": "value",
                        "match_basis_explanation": (
                            "The request state applies the requested predicate."
                        ),
                    }
                ],
            }
        ],
        schema=surface.provider_schema(),
    )

    with pytest.raises(ValidationError):
        validate(
            instance=[
                {
                    **common,
                    "applications": [
                        {
                            "application_target_id": (
                                f"returned_field.{field.evidence_id}"
                            ),
                            "value_component": "value",
                            "match_basis_explanation": (
                                "The returned state applies the requested predicate."
                            ),
                            "application_value_id": selected_id,
                        }
                    ],
                }
            ],
            schema=surface.provider_schema(),
        )


def test_returned_field_predicate_rejects_incompatible_declared_type() -> None:
    value = FactValue.literal(
        id="minimum_amount",
        known_input_id="qi_amount",
        literal_type=LiteralType.NUMBER,
        value="1000",
    )
    test = _predicate_test(input_id="qi_amount", operator=PredicateOperator.LT)
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            evidence_items=(
                FieldEvidence(
                    evidence_id="field.description",
                    field_id="data.description",
                    type="string",
                    row_path_id="data",
                    row_source_id="sales",
                ),
            ),
        ),
        requested_fact_id="fact_1",
        resolved_values=(value,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )

    assert surface.returned_field_targets_by_id == {}


def test_identity_value_is_not_exposed_as_a_returned_field_scalar() -> None:
    identity = FactValue.identity(
        id="resource_identity",
        known_input_id="qi_resource",
        key=entity_key_value(
            "resource",
            "primary_key",
            {"resource_id": "resource_1"},
        ),
    )
    test = _predicate_test(input_id="qi_resource")
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            evidence_items=(
                FieldEvidence(
                    evidence_id="field.resource_id",
                    field_id="data.resource_id",
                    type="string",
                    row_path_id="data",
                    row_source_id="records",
                ),
            ),
        ),
        requested_fact_id="fact_1",
        resolved_values=(identity,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )

    assert surface.returned_field_targets_by_id == {}


def test_excluded_predicate_value_compiles_to_not_equals_filter() -> None:
    value = FactValue.literal(
        id="excluded_state",
        known_input_id="qi_state",
        literal_type=LiteralType.STRING,
        value="archived",
    )
    test = _predicate_test(
        input_id="qi_state",
        polarity=AnswerPopulationMembershipTestPolarity.MUST_FAIL,
    )
    field = FieldEvidence(
        evidence_id="field.state",
        field_id="data.state",
        type="string",
        row_path_id="data",
        row_source_id="records",
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            evidence_items=(field,),
        ),
        requested_fact_id="fact_1",
        resolved_values=(value,),
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )

    parsed = parse_resolved_input_applications(
        (
            _application(
                value.id,
                field.evidence_id,
                target_kind=ResolvedInputApplicationTargetKind.RETURNED_FIELD.value,
                population_test_results=_satisfies_test_result(
                    test_effect="CONFLICTS_WITH_TEST"
                ),
            ),
        ),
        surface=surface,
    )

    assert parsed.applied_filters[0].operator == PredicateOperator.NOT_EQUALS.value


def test_multiple_predicate_values_compile_to_one_returned_field_membership() -> None:
    values = tuple(
        FactValue.literal(
            id=f"state_{index}",
            known_input_id=f"qi_state_{index}",
            literal_type=LiteralType.STRING,
            value=value,
            proof_refs=(f"known_input:qi_state_{index}",),
        )
        for index, value in enumerate(("ready", "finished"), start=1)
    )
    test = RequestedFactAnswerPopulationMembershipTest(
        id="test_operand",
        kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
        polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
        test_question="Is the row in a requested state?",
        owned_question_input_refs=tuple(value.known_input_id for value in values),
    )
    field = FieldEvidence(
        evidence_id="field.state",
        field_id="data.state",
        type="string",
        row_path_id="data",
        row_source_id="records",
    )
    surface = resolved_input_application_surface(
        candidate=SourceCandidate(
            id="source_1",
            applies_to_requested_fact_ids=("fact_1",),
            kind="read",
            evidence_items=(field,),
        ),
        requested_fact_id="fact_1",
        resolved_values=values,
        membership_tests=(test,),
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
    )
    result = _satisfies_test_result(test.test_question)

    parsed = parse_resolved_input_applications(
        tuple(
            _application(
                value.id,
                field.evidence_id,
                target_kind=ResolvedInputApplicationTargetKind.RETURNED_FIELD.value,
                population_test_results=result,
            )
            for value in values
        ),
        surface=surface,
    )

    assert len(parsed.applied_filters) == 1
    assert parsed.applied_filters[0].operator == PredicateOperator.IN.value
    assert set(parsed.applied_filters[0].application_values) == {
        "ready",
        "finished",
    }


def test_boolean_finite_choice_param_keeps_population_contract_review_surface():
    param = {
        "param_id": "is_active",
        "source": "query",
        "type": "boolean",
        "required": False,
        "choices": ["true", "false"],
        "binding_values": [
            {"value": "true", "label": "true", "source": "static_choice"},
            {"value": "false", "label": "false", "source": "static_choice"},
        ],
    }
    bind_options = _param_bind_options(param)
    candidate = {
        "source_candidate_id": "source_1",
        "params": [
            {
                **param,
                "bind_options": bind_options,
                "omit_option": _param_omit_option(param, bind_options=bind_options),
            }
        ],
    }

    candidate = _candidate_with_param_decision_options(candidate)
    candidate = _candidate_with_param_population_contracts(candidate)

    reviewed_param = candidate["params"][0]
    decisions = {
        option["decision"] for option in reviewed_param.get("decision_options") or ()
    }

    assert decisions == {"bind"}
    assert isinstance(reviewed_param.get("population_contract"), dict)


def test_omit_param_decision_compiles_to_no_endpoint_binding():
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        params=(
            CandidateParameter(
                id="is_open",
                type="boolean",
                required=False,
                choices=(),
                decision_options=(
                    CandidateParamDecision(
                        id="param_decision.source_1.is_open.omit",
                        decision="omit",
                    ),
                ),
            ),
        ),
    )

    parsed = parse_param_decision_binding_sets(
        {
            "is_open": NormalizedParamDecision(
                match_basis_explanation="No open-status filter was requested.",
                param_decision_id="param_decision.source_1.is_open.omit",
                population_intent="All open-status values.",
            )
        },
        candidate=candidate,
        available_values=(),
        answer_population=AnswerPopulation(
            population_binding_id="pop.source_1.candidate_population",
            intent_text="all stores",
            match_basis_explanation="Use the full store population.",
        ),
        parameter_namespace="fixture.source_1",
    )

    assert parsed.binding_sets == ((),)


def test_prebound_parameter_requires_no_second_param_decision() -> None:
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        params=(
            CandidateParameter(
                id="status",
                type="choice",
                required=False,
                choices=("OPEN", "CLOSED"),
                decision_options=(
                    CandidateParamDecision(
                        id="param_decision.source_1.status.bind.open",
                        decision="bind",
                        value="OPEN",
                    ),
                ),
            ),
        ),
    )

    parsed = parse_param_decision_binding_sets(
        {},
        candidate=candidate,
        available_values=(),
        answer_population=AnswerPopulation(
            population_binding_id="pop.source_1.candidate_population",
            intent_text="open records",
            match_basis_explanation="Use open records.",
        ),
        parameter_namespace="fixture.source_1",
        effective_param_ids=("status",),
        prebound_param_ids=("status",),
    )

    assert parsed.binding_sets == ((),)
