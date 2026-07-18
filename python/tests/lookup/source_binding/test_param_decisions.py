from __future__ import annotations

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
from fervis.lookup.source_binding.model import AnswerPopulation
from fervis.lookup.source_binding.input_applications import (
    ResolvedInputApplicationTargetKind,
    parse_resolved_input_applications,
    resolved_input_application_surface,
)
from fervis.lookup.source_binding.parser.params import parse_param_decision_binding_sets
from fervis.lookup.source_binding.parser.types import NormalizedParamDecision
from fervis.lookup.source_binding.provider_contract import (
    ResolvedInputApplicationOutput,
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
        tuple(_application(identities.id, param.id) for param in params),
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
                "components_by_target_kind": {
                    "request_parameter": ["value"],
                },
                "population_test_basis": {},
            },
            {
                "value_id": "maximum_amount",
                "components_by_target_kind": {
                    "request_parameter": ["value"],
                },
                "population_test_basis": {},
            },
        ],
        "targets_by_kind": {
            "request_parameter": ["lower_bound", "upper_bound"],
            "returned_identity": [],
        },
    }
    variants = surface.provider_schema()["items"]["oneOf"]

    assert len(variants) == 2
    assert all(
        variant["properties"]["target_id"]["enum"]
        == ["lower_bound", "upper_bound"]
        for variant in variants
    )


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
            ResolvedInputApplicationOutput(
                target_kind=(
                    ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value
                ),
                target_id=reference.evidence_id,
                value_id=identity.id,
                value_component="canonical_key",
                    match_basis_explanation=(
                        "The returned locations are restricted by the resolved area."
                    ),
                    population_test_results={},
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
) -> ResolvedInputApplicationOutput:
    return ResolvedInputApplicationOutput(
        target_kind=ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value,
        target_id=target_id,
        value_id=value_id,
        value_component=value_component,
        match_basis_explanation="The resolved input restricts this source.",
        population_test_results={},
    )


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
