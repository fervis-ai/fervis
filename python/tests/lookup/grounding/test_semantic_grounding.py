from copy import deepcopy

from jsonschema import ValidationError, validate
import pytest

from fervis.lookup.answer_program.values import LiteralType, LiteralValuePayload
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.relation_catalog.row_sources import RowSource, RowSourceKind
from fervis.lookup.grounding.identity import (
    IdentifierKind,
    InputBindingPurpose,
    InputBindingKeyComponent,
    InputBindingOption,
    ResolverCandidate,
)
from fervis.lookup.grounding.semantic import (
    CompatibleIdentityRoute,
    deterministic_scalar_values,
    GroundingPartition,
    reference_grounding_tasks,
    identity_resolution_tasks,
    SemanticGroundingRequest,
    time_grounding_tasks,
)
from fervis.lookup.grounding.identity import reference_binding_options
from fervis.lookup.grounding.semantic_parser import parse_semantic_grounding
from fervis.lookup.grounding.semantic_prompt import SemanticGroundingTurnPrompt
from fervis.lookup.grounding.semantic_schema import build_semantic_grounding_schema
from fervis.lookup.question_contract.model import FactLocalKind, FactLocalRef
from fervis.lookup.question_contract.model import InputTerm
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.semantic_types import (
    CollectionType,
    DecimalType,
    IdentifierType,
    SourceOrigin,
    SourceOriginKind,
    TemporalScopeType,
    TextType,
    UnitlessMeasure,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from tests.lookup.grounding._fixtures import _staff_detail_read, _staff_read


def test_identity_routes_are_grouped_by_canonical_meaning() -> None:
    expected_set_ref = FactLocalRef("fact_1", FactLocalKind.SET, "s1")
    partition = GroundingPartition(
        input_ref="input_1",
        use_refs=("fact_1:input_use:e1:1",),
        expected_value_type=IdentifierType("s1"),
        expected_set_ref=expected_set_ref,
        operand_meaning="the staff member being identified",
    )
    detail = _option("detail_staff", entity_kind="staff")
    listing = _option("list_staff", entity_kind="staff")
    grounding_task = reference_grounding_tasks(
        (partition,),
        resolver_options_by_use_ref={
            partition.use_refs[0]: (detail, listing),
        },
        denoted_instance_kinds_by_input_ref={"input_1": "staff member"},
    )[0]

    result = identity_resolution_tasks(
        (grounding_task,),
        compatible_bindings_by_task_ref={
            grounding_task.task_ref: (_binding(detail), _binding(listing))
        },
    )[0]

    assert len(result.canonical_options) == 1
    assert result.canonical_options[0].identity_ref == "staff:primary_key"
    assert result.canonical_options[0].resolver_route_refs == (
        detail.id,
        listing.id,
    )
    assert tuple(route.route_ref for route in result.resolver_routes) == (
        detail.id,
        listing.id,
    )


def test_identity_use_partitions_keep_distinct_expected_sets_separate() -> None:
    use_ref = "fact_1:input_use:e1:1"
    staff = _option("detail_staff", entity_kind="staff")
    location = _option("detail_location", entity_kind="location")
    tasks = reference_grounding_tasks(
        (
            GroundingPartition(
                input_ref="input_1",
                use_refs=(use_ref,),
                expected_value_type=IdentifierType("s1"),
                expected_set_ref=FactLocalRef("fact_1", FactLocalKind.SET, "s1"),
                operand_meaning="the staff member being identified",
            ),
            GroundingPartition(
                input_ref="input_1",
                use_refs=("fact_2:input_use:e1:1",),
                expected_value_type=IdentifierType("s2"),
                expected_set_ref=FactLocalRef("fact_2", FactLocalKind.SET, "s2"),
                operand_meaning="the location being identified",
            ),
        ),
        resolver_options_by_use_ref={
            use_ref: (staff,),
            "fact_2:input_use:e1:1": (location,),
        },
        denoted_instance_kinds_by_input_ref={"input_1": "staff member"},
    )

    assert tuple(task.expected_set_ref.local_id for task in tasks) == ("s1", "s2")
    assert tuple(option.id for option in tasks[0].options) == (staff.id,)
    assert tuple(option.id for option in tasks[1].options) == (location.id,)


def test_semantic_grounding_preserves_compatible_routes_for_read_eligibility() -> None:
    catalog = RelationCatalog(reads=(_staff_read(),))
    [option] = reference_binding_options(
        input_id="input_1",
        resolver_catalog=catalog,
        resolver_row_sources=build_row_source_catalog(catalog),
        expected_identity=None,
    )
    expected_set_ref = FactLocalRef("fact_1", FactLocalKind.SET, "s1")
    input_term = InputTerm(
        id="input_1",
        origin=SourceOrigin(
            SourceOriginKind.QUESTION_CONTEXT,
            "one staff member named Ada",
        ),
        operand="Ada",
        value_type=IdentifierType("s1"),
    )
    task = reference_grounding_tasks(
        (
            GroundingPartition(
                input_ref=input_term.id,
                use_refs=("fact_1:input_use:e1:1",),
                expected_value_type=input_term.value_type,
                expected_set_ref=expected_set_ref,
                operand_meaning="the staff member whose sales are counted",
            ),
        ),
        resolver_options_by_use_ref={"fact_1:input_use:e1:1": (option,)},
        denoted_instance_kinds_by_input_ref={input_term.id: "staff member"},
    )[0]
    request = SemanticGroundingRequest(
        question="How many sales did Ada make?",
        inputs=(input_term,),
        tasks=(task,),
        set_origins={
            expected_set_ref: SourceOrigin(
                SourceOriginKind.QUESTION_CONTEXT,
                "staff members",
            )
        },
        resolver_catalog=catalog,
    )
    payload = {
        "time_resolutions": {},
        "reference_reviews": {
            task.task_ref: {
                "identifier_kind_basis": "Ada is a descriptive name.",
                "identifier_kind": "DESCRIPTIVE",
                "purpose": "reference_grounding",
                "resource_type_reviews": {
                    "staff": {
                        "compatibility_basis": (
                            "A staff record could be the staff member named Ada."
                        ),
                        "compatibility": "POSSIBLE_DENOTED_KIND",
                        "route_reviews": {
                            option.id: {
                                "assessment_basis": "The route searches staff names and returns staff identities.",
                                "resolution": {
                                    "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                                    "lookup_request_params": [
                                        "list_staff_list.query.name"
                                    ],
                                    "returned_identity_verification_fields": [
                                        "data.full_name"
                                    ],
                                },
                            }
                        },
                    }
                },
            }
        },
    }

    validate(payload, build_semantic_grounding_schema(request))
    result = parse_semantic_grounding(payload, request=request)
    [resolution_task] = result.identity_tasks
    prompt = (
        SemanticGroundingTurnPrompt(request)
        .to_model_invocation(
            build_turn_prompt_context(
                current_question=request.question,
                conversation_context={},
            )
        )
        .prompt_text
    )

    assert resolution_task.use_refs == task.use_refs
    assert resolution_task.canonical_options[0].identity_ref == "staff:primary_key"
    assert resolution_task.canonical_options[0].resolver_route_refs == (option.id,)
    assert '<expected_set ref="fact_1:set:s1">' in prompt
    assert "staff members" in prompt


def test_primary_key_grounding_exposes_only_identity_validation_routes() -> None:
    catalog = RelationCatalog(reads=(_staff_read(), _staff_detail_read()))
    options = reference_binding_options(
        input_id="input_1",
        resolver_catalog=catalog,
        resolver_row_sources=build_row_source_catalog(catalog),
        expected_identity=None,
    )
    assert {option.purpose for option in options} == {
        InputBindingPurpose.IDENTITY_VALIDATION,
        InputBindingPurpose.REFERENCE_GROUNDING,
    }
    expected_set_ref = FactLocalRef("fact_1", FactLocalKind.SET, "s1")
    input_term = InputTerm(
        id="input_1",
        origin=SourceOrigin(
            SourceOriginKind.QUESTION_CONTEXT,
            "one staff member identified by staff id",
        ),
        operand="staff-1",
        value_type=IdentifierType("s1"),
    )
    [task] = reference_grounding_tasks(
        (
            GroundingPartition(
                input_ref=input_term.id,
                use_refs=("fact_1:input_use:e1:1",),
                expected_value_type=input_term.value_type,
                expected_set_ref=expected_set_ref,
                operand_meaning="the staff member identified by staff id",
            ),
        ),
        resolver_options_by_use_ref={"fact_1:input_use:e1:1": options},
        denoted_instance_kinds_by_input_ref={input_term.id: "staff member"},
    )
    request = SemanticGroundingRequest(
        question="Which staff member has staff id staff-1?",
        inputs=(input_term,),
        tasks=(task,),
        set_origins={
            expected_set_ref: SourceOrigin(
                SourceOriginKind.QUESTION_CONTEXT,
                "staff member",
            )
        },
        resolver_catalog=catalog,
    )
    detail = next(
        option
        for option in options
        if option.purpose is InputBindingPurpose.IDENTITY_VALIDATION
    )
    listing = next(
        option
        for option in options
        if option.purpose is InputBindingPurpose.REFERENCE_GROUNDING
    )
    payload = {
        "time_resolutions": {},
        "reference_reviews": {
            task.task_ref: {
                "identifier_kind_basis": "The supplied value is the full staff key.",
                "identifier_kind": "PRIMARY_KEY",
                "purpose": "identity_validation",
                "resource_type_reviews": {
                    "staff": {
                        "compatibility_basis": "The route returns a Staff instance.",
                        "compatibility": "POSSIBLE_DENOTED_KIND",
                        "route_reviews": {
                            detail.id: {
                                "assessment_basis": "The path key validates the Staff identity.",
                                "resolution": {
                                    "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                                    "lookup_request_params": [
                                        "get_staff_detail.path.staff_id"
                                    ],
                                    "returned_identity_verification_fields": [
                                        "data.staff_id"
                                    ],
                                },
                            }
                        },
                    }
                },
            }
        },
    }

    validate(payload, build_semantic_grounding_schema(request))
    wrong_purpose = deepcopy(payload)
    wrong_review = wrong_purpose["reference_reviews"][task.task_ref]
    wrong_review["purpose"] = "reference_grounding"
    wrong_review["resource_type_reviews"]["staff"]["route_reviews"] = {
        listing.id: {
            "assessment_basis": "The list route searches descriptive names.",
            "resolution": {
                "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                "lookup_request_params": ["list_staff_list.query.name"],
                "returned_identity_verification_fields": ["data.full_name"],
            },
        }
    }
    with pytest.raises(ValidationError):
        validate(wrong_purpose, build_semantic_grounding_schema(request))
    result = parse_semantic_grounding(payload, request=request)
    [resolution_task] = result.identity_tasks
    prompt = (
        SemanticGroundingTurnPrompt(request)
        .to_model_invocation(
            build_turn_prompt_context(
                current_question=request.question,
                conversation_context={},
            )
        )
        .prompt_text
    )
    assert [route.option.purpose for route in resolution_task.resolver_routes] == [
        InputBindingPurpose.IDENTITY_VALIDATION
    ]
    assert 'purpose="identity_validation"' in prompt
    assert 'purpose="reference_grounding"' in prompt


def test_semantic_grounding_resolves_time_into_one_canonical_value() -> None:
    input_term = InputTerm(
        id="input_1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "March 2026"),
        operand="March 2026",
        value_type=TemporalScopeType(),
    )
    partition = GroundingPartition(
        input_ref=input_term.id,
        use_refs=("fact_1:input_use:e1:1",),
        expected_value_type=input_term.value_type,
        expected_set_ref=None,
        operand_meaning="the reporting period",
    )
    [task] = time_grounding_tasks((partition,), inputs={input_term.id: input_term})
    request = SemanticGroundingRequest(
        question="How many events occurred during March 2026?",
        inputs=(input_term,),
        tasks=(),
        time_tasks=(task,),
        set_origins={},
        resolver_catalog=RelationCatalog(),
        runtime_date="2026-07-21",
        timezone="Africa/Nairobi",
    )
    payload = {
        "time_resolutions": {
            task.task_ref: {
                "date_intent": {
                    "expression": task.expression,
                    "intent": {
                        "time_shape": "period_named",
                        "unit": "month",
                        "mode": "full",
                        "year": 2026,
                        "month": 0,
                        "day": 0,
                        "year_policy": "none",
                        "relative_offset": 0,
                        "named_value": 3,
                        "end_year": 0,
                        "end_month": 0,
                        "end_day": 0,
                        "end_year_policy": "none",
                        "count": 0,
                        "direction": "none",
                    },
                }
            }
        },
        "reference_reviews": {},
    }

    validate(payload, build_semantic_grounding_schema(request))
    conflicting_policy = deepcopy(payload)
    conflicting_policy["time_resolutions"][task.task_ref]["date_intent"]["intent"][
        "year_policy"
    ] = "most_recent"
    with pytest.raises(ValidationError):
        validate(conflicting_policy, build_semantic_grounding_schema(request))

    result = parse_semantic_grounding(payload, request=request)
    [value] = result.canonical_values
    assert value.use_refs == partition.use_refs
    assert value.typed_value.payload.resolved_start == "2026-03-01"
    assert value.typed_value.payload.resolved_end == "2026-03-31"


@pytest.mark.parametrize(
    ("expression", "runtime_date", "intent", "expected_bounds"),
    (
        (
            "Q1",
            "2026-05-12",
            {
                "time_shape": "period_named",
                "unit": "quarter",
                "mode": "full",
                "year": 0,
                "month": 0,
                "day": 0,
                "year_policy": "most_recent",
                "relative_offset": 0,
                "named_value": 1,
                "end_year": 0,
                "end_month": 0,
                "end_day": 0,
                "end_year_policy": "none",
                "count": 0,
                "direction": "none",
            },
            ("2026-01-01", "2026-03-31"),
        ),
        (
            "this month",
            "2026-06-01",
            {
                "time_shape": "period_relative",
                "unit": "month",
                "mode": "full",
                "year": 0,
                "month": 0,
                "day": 0,
                "year_policy": "none",
                "relative_offset": 0,
                "named_value": 0,
                "end_year": 0,
                "end_month": 0,
                "end_day": 0,
                "end_year_policy": "none",
                "count": 0,
                "direction": "none",
            },
            ("2026-06-01", "2026-06-30"),
        ),
        (
            "this week so far",
            "2026-06-04",
            {
                "time_shape": "period_relative",
                "unit": "week",
                "mode": "to_date",
                "year": 0,
                "month": 0,
                "day": 0,
                "year_policy": "none",
                "relative_offset": 0,
                "named_value": 0,
                "end_year": 0,
                "end_month": 0,
                "end_day": 0,
                "end_year_policy": "none",
                "count": 0,
                "direction": "none",
            },
            ("2026-06-01", "2026-06-04"),
        ),
    ),
)
def test_semantic_time_resolution_preserves_business_period_intent(
    expression: str,
    runtime_date: str,
    intent: dict[str, object],
    expected_bounds: tuple[str, str],
) -> None:
    input_term = InputTerm(
        id="input_1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, expression),
        operand=expression,
        value_type=TemporalScopeType(),
    )
    [task] = time_grounding_tasks(
        (
            GroundingPartition(
                input_ref=input_term.id,
                use_refs=("fact_1:input_use:e1:1",),
                expected_value_type=input_term.value_type,
                expected_set_ref=None,
                operand_meaning="the reporting period",
            ),
        ),
        inputs={input_term.id: input_term},
    )
    request = SemanticGroundingRequest(
        question="What happened during the reporting period?",
        inputs=(input_term,),
        tasks=(),
        time_tasks=(task,),
        set_origins={},
        resolver_catalog=RelationCatalog(),
        runtime_date=runtime_date,
        timezone="Africa/Nairobi",
    )

    result = parse_semantic_grounding(
        {
            "time_resolutions": {
                task.task_ref: {
                    "date_intent": {"expression": expression, "intent": intent}
                }
            },
            "reference_reviews": {},
        },
        request=request,
    )

    [value] = result.canonical_values
    assert (
        value.typed_value.payload.resolved_start,
        value.typed_value.payload.resolved_end,
    ) == expected_bounds


def test_collection_input_waits_for_typed_identity_grounding() -> None:
    input_term = InputTerm(
        id="input_1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "two identifiers"),
        operand=("identifier-1", "identifier-2"),
        value_type=CollectionType(TextType()),
    )
    partition = GroundingPartition(
        input_ref=input_term.id,
        use_refs=("fact_1:input_use:e1:1",),
        expected_value_type=CollectionType(IdentifierType("s1")),
        expected_set_ref=FactLocalRef("fact_1", FactLocalKind.SET, "s1"),
        operand_meaning="the staff members whose sales are counted",
    )

    assert (
        deterministic_scalar_values((partition,), inputs={input_term.id: input_term})
        == ()
    )


def test_non_identity_scalars_are_typed_without_identity_resolution() -> None:
    state = InputTerm(
        id="state",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "finished"),
        operand="finished",
        value_type=TextType(),
    )
    amount = InputTerm(
        id="amount",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "1000.00"),
        operand="1000.00",
        value_type=DecimalType(UnitlessMeasure()),
    )
    values = deterministic_scalar_values(
        (
            GroundingPartition("state", ("use:state",), TextType(), None, "state"),
            GroundingPartition(
                "amount",
                ("use:amount",),
                DecimalType(UnitlessMeasure()),
                None,
                "threshold",
            ),
        ),
        inputs={"state": state, "amount": amount},
    )

    state_payload, amount_payload = (value.typed_value.payload for value in values)
    assert isinstance(state_payload, LiteralValuePayload)
    assert state_payload.literal_type is LiteralType.STRING
    assert state_payload.value == "finished"
    assert isinstance(amount_payload, LiteralValuePayload)
    assert amount_payload.literal_type is LiteralType.NUMBER
    assert amount_payload.value == "1000"


def _option(option_id: str, *, entity_kind: str) -> InputBindingOption:
    source = RowSource(
        id=f"source_{option_id}",
        kind=RowSourceKind.API_READ,
        label=option_id,
        read_id=option_id,
        endpoint_name=option_id,
        row_path_id="rows",
    )
    candidate = ResolverCandidate(
        known_input_id="input_1",
        resolver_source=source,
        entity_kind=entity_kind,
        key_id="primary_key",
        key_components=(
            InputBindingKeyComponent(
                component_id=f"{entity_kind}_id",
                field_id=f"{entity_kind}_id",
                field_ref=f"{option_id}.{entity_kind}_id",
            ),
        ),
    )
    return InputBindingOption(
        id=option_id,
        known_input_id="input_1",
        candidate=candidate,
    )


def _binding(option: InputBindingOption) -> CompatibleIdentityRoute:
    return CompatibleIdentityRoute(
        option_id=option.id,
        identifier_kind=IdentifierKind.DESCRIPTIVE,
        lookup_request_param_refs=(f"{option.id}.query",),
        returned_identity_verification_field_paths=("data.id",),
    )
