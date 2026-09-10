from dataclasses import replace

import pytest
from jsonschema import validate

from fervis.lookup.answer_program.values import IdentityValuePayload
from fervis.lookup.grounding.identity import IdentifierKind, reference_binding_options
from fervis.lookup.grounding.semantic import (
    CanonicalIdentityOption,
    CompatibleIdentityRoute,
    GroundingPartition,
    IdentityResolutionTask,
    IdentityResolverRoute,
    ResolvedIdentity,
    identity_resolution_tasks,
    reference_grounding_tasks,
)
from fervis.lookup.question_contract.model import (
    FactLocalKind,
    FactLocalRef,
    InputTerm,
)
from fervis.lookup.question_contract.parser import (
    ParsedSemanticQuestionContract,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_contract,
    parse_semantic_question_frame,
)
from fervis.lookup.read_eligibility.semantic import (
    IdentityRouteSelection,
    SemanticReadDecision,
    SemanticReadEligibilityRequest,
)
from fervis.lookup.read_eligibility.semantic_parser import (
    parse_semantic_read_eligibility,
)
from fervis.lookup.read_eligibility.semantic_prompt import (
    SemanticReadEligibilityTurnPrompt,
)
from fervis.lookup.read_eligibility.semantic_resolution import (
    execute_identity_selection,
)
from fervis.lookup.read_eligibility.semantic_schema import (
    build_semantic_read_eligibility_schema,
)
from fervis.lookup.relation_catalog import RelationCatalog, RowCardinality, RowPath
from fervis.lookup.relation_catalog.row_sources import (
    RowSourceCatalog,
    build_row_source_catalog,
)
from fervis.lookup.semantic_types import (
    CollectionType,
    IdentifierType,
    SourceOrigin,
    SourceOriginKind,
    TextType,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from tests.lookup.grounding._fixtures import (
    _area_read,
    _location_with_area_read,
    _staff_read,
)


def test_read_eligibility_assesses_one_api_read_instead_of_its_row_paths() -> None:
    [index] = _semantic_contract().semantic_indexes
    read = replace(
        _staff_read(),
        row_paths=(
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
        ),
    )
    catalog = RelationCatalog(reads=(read,))
    request = _request(index=index, catalog=catalog)
    candidate = next(item for item in request.read_candidates if item.read_id)
    invocation = SemanticReadEligibilityTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question="Which staff member is named Ada?",
            conversation_context={},
        )
    )

    assert candidate.read_id == "list_staff_list"
    assert invocation.prompt_text.count('read="list_staff_list"') == 1
    assert all(source.row_path_id != "root" for source in candidate.sources)


def test_read_eligibility_sees_declared_cross_read_relations() -> None:
    [index] = _semantic_contract().semantic_indexes
    catalog = RelationCatalog(reads=(_location_with_area_read(), _area_read()))
    request = _request(index=index, catalog=catalog)

    prompt = (
        SemanticReadEligibilityTurnPrompt(request)
        .to_model_invocation(
            build_turn_prompt_context(
                current_question="How many locations are in an area?",
                conversation_context={},
            )
        )
        .prompt_text
    )

    assert (
        '<relation left_source="list_location_list" '
        'left_field_refs="field.data.area.area_id" '
        'right_source="list_area_list" '
        'right_field_refs="field.data.area_id" />'
    ) in prompt


def test_read_retention_is_fact_local_and_does_not_author_bindings() -> None:
    [index] = _semantic_contract().semantic_indexes
    catalog = _catalog()
    request = _request(index=index, catalog=catalog)
    candidate = next(item for item in request.read_candidates if item.read_id)
    field_ref = candidate.fields[0].field_ref
    payload = _retention_payload(
        request,
        index.requested_fact_id,
        retained_candidate_ref=candidate.candidate_ref,
        relevant_field_refs=(field_ref,),
    )

    schema = build_semantic_read_eligibility_schema(request)
    validate(payload, schema)
    result = parse_semantic_read_eligibility(payload, request=request)
    assessment = next(
        item
        for item in result.read_assessments
        if item.candidate_ref == candidate.candidate_ref
    )

    assert assessment.requested_fact_id == index.requested_fact_id
    assert assessment.decision is SemanticReadDecision.RETAIN
    assert assessment.relevant_field_refs == (field_ref,)
    assert "set_reviews" not in str(schema)
    assert "identifier_reviews" not in str(schema)
    assert "supported_terms" not in str(schema)

    invalid = _retention_payload(
        request,
        index.requested_fact_id,
        retained_candidate_ref=None,
    )
    invalid["read_assessments_by_requested_fact"][index.requested_fact_id][
        candidate.candidate_ref
    ]["relevant_field_refs"] = [field_ref]
    with pytest.raises(ValueError, match="dropped read retains fields"):
        parse_semantic_read_eligibility(invalid, request=request)


def test_identity_selection_precedes_read_assessment_and_resolves() -> None:
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    [input_term] = parsed.contract.inputs
    catalog = _catalog()
    source_catalog = build_row_source_catalog(catalog)
    [option] = reference_binding_options(
        input_id=input_term.id,
        resolver_catalog=catalog,
        resolver_row_sources=source_catalog,
        expected_identity=None,
    )
    partition = GroundingPartition(
        input_ref=input_term.id,
        use_refs=tuple(item.use_ref for item in index.input_use_sites),
        expected_value_type=input_term.value_type,
        expected_set_ref=index.input_use_sites[0].identity_set_ref,
        operand_meaning=index.input_use_sites[0].operand_meaning,
    )
    [grounding_task] = reference_grounding_tasks(
        (partition,),
        resolver_options_by_use_ref={
            use_ref: (option,) for use_ref in partition.use_refs
        },
        denoted_instance_kinds_by_input_ref={input_term.id: "staff member"},
    )
    binding = CompatibleIdentityRoute(
        option_id=option.id,
        identifier_kind=IdentifierKind.DESCRIPTIVE,
        lookup_request_param_refs=("list_staff_list.query.name",),
        returned_identity_verification_field_paths=("data.full_name",),
    )
    [identity_task] = identity_resolution_tasks(
        (grounding_task,),
        compatible_bindings_by_task_ref={grounding_task.task_ref: (binding,)},
    )
    request = SemanticReadEligibilityRequest(
        indexes=(index,),
        source_catalog=source_catalog,
        answer_catalog=catalog,
        identity_tasks=(identity_task,),
        resolver_catalog=catalog,
    )
    candidate = next(item for item in request.read_candidates if item.read_id)
    [canonical_option] = identity_task.canonical_options
    payload = _retention_payload(
        request,
        index.requested_fact_id,
        retained_candidate_ref=candidate.candidate_ref,
        relevant_field_refs=(candidate.fields[0].field_ref,),
    )
    payload["identity_outcomes"] = {
        identity_task.task_ref: _selected_identity_payload(
            task=identity_task,
            canonical_option_id=canonical_option.canonical_option_id,
            route_id=option.id,
        )
    }

    validate(payload, build_semantic_read_eligibility_schema(request))
    result = parse_semantic_read_eligibility(payload, request=request)
    [selection] = result.identity_outcomes
    resolution = execute_identity_selection(
        task=identity_task,
        selection=selection,
        input_term=input_term,
        full_catalog=catalog,
        data_access_port=_DataAccess(),
        source_read_key_prefix="test",
    )
    prompt = (
        SemanticReadEligibilityTurnPrompt(request)
        .to_model_invocation(
            build_turn_prompt_context(
                current_question="Which staff member is named Ada?",
                conversation_context={},
            )
        )
        .prompt_text
    )

    assert isinstance(selection, IdentityRouteSelection)
    assert isinstance(resolution, ResolvedIdentity)
    assert isinstance(
        resolution.canonical_value.typed_value.payload, IdentityValuePayload
    )
    assert prompt.index(
        "Available identity resolver routes (separate from answer reads):"
    ) < prompt.index("Declared canonical identity uses in answer reads:")
    assert prompt.index(
        "Declared canonical identity uses in answer reads:"
    ) < prompt.index("Answer read candidates:")
    assert prompt.index("Identity selection") < prompt.index("Read assessment")
    assert (
        '<parameter param_ref="list_staff_list.query.name" source="query" value="Ada" />'
        in prompt
    )
    assert "regardless of whether another read could also answer" in prompt
    assert "set_reviews" not in prompt
    assert "identifier_reviews" not in prompt


def test_identity_route_assessments_cover_every_meaning_before_selection() -> None:
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    [input_term] = parsed.contract.inputs
    catalog = _catalog()
    [option] = reference_binding_options(
        input_id=input_term.id,
        resolver_catalog=catalog,
        resolver_row_sources=build_row_source_catalog(catalog),
        expected_identity=None,
    )
    other_option = replace(
        option,
        id=f"{option.id}_other",
        candidate=replace(option.candidate, entity_kind="other_staff_meaning"),
    )
    route = IdentityResolverRoute(
        route_ref=option.id,
        option=option,
        compatibility=CompatibleIdentityRoute(
            option_id=option.id,
            identifier_kind=IdentifierKind.DESCRIPTIVE,
            lookup_request_param_refs=("list_staff_list.query.name",),
            returned_identity_verification_field_paths=("data.full_name",),
        ),
    )
    other_route = IdentityResolverRoute(
        route_ref=other_option.id,
        option=other_option,
        compatibility=replace(route.compatibility, option_id=other_option.id),
    )
    task = IdentityResolutionTask(
        task_ref="fact_1:identity:input_1",
        input_ref=input_term.id,
        use_refs=tuple(item.use_ref for item in index.input_use_sites),
        expected_set_ref=index.input_use_sites[0].identity_set_ref,
        canonical_options=(
            CanonicalIdentityOption(
                canonical_option_id="staff_meaning",
                identity_ref="staff:primary_key",
                resolver_route_refs=(route.route_ref,),
            ),
            CanonicalIdentityOption(
                canonical_option_id="other_staff_meaning",
                identity_ref="other_staff_meaning:primary_key",
                resolver_route_refs=(other_route.route_ref,),
            ),
        ),
        resolver_routes=(route, other_route),
    )
    request = SemanticReadEligibilityRequest(
        indexes=(index,),
        source_catalog=RowSourceCatalog(),
        answer_catalog=RelationCatalog(),
        identity_tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = {
        "identity_outcomes": {
            task.task_ref: {
                "canonical_option_assessments": [
                    {
                        "canonical_option_id": "staff_meaning",
                        "assessment": "This is the requested staff meaning.",
                        "decision": "FITS",
                    },
                    {
                        "canonical_option_id": "other_staff_meaning",
                        "assessment": "This is another identity meaning.",
                        "decision": "DOES_NOT_FIT",
                    },
                ],
                "canonical_option_basis": "The input denotes staff.",
                "canonical_option_id": "staff_meaning",
                "resolver_route_assessments": [
                    {
                        "resolver_route_id": route.route_ref,
                        "assessment": "This route resolves staff.",
                        "decision": "FITS",
                    },
                    {
                        "resolver_route_id": other_route.route_ref,
                        "assessment": "This route resolves another meaning.",
                        "decision": "DOES_NOT_FIT",
                    },
                ],
                "resolver_route_basis": "Only the staff route fits.",
                "resolver_route_id": route.route_ref,
                "evidence_refs": [
                    task.task_ref,
                    "staff_meaning",
                    route.route_ref,
                ],
                "outcome": "SELECTED",
            }
        },
        "read_assessments_by_requested_fact": {"fact_1": {}},
    }

    validate(payload, build_semantic_read_eligibility_schema(request))
    [selection] = parse_semantic_read_eligibility(
        payload,
        request=request,
    ).identity_outcomes
    assert selection.resolver_route_id == route.route_ref

    payload["identity_outcomes"][task.task_ref]["resolver_route_id"] = (
        other_route.route_ref
    )
    validate(payload, build_semantic_read_eligibility_schema(request))
    with pytest.raises(
        ValueError,
        match="identity selection references an unknown resolver route",
    ):
        parse_semantic_read_eligibility(payload, request=request)


def test_one_selected_identity_route_resolves_every_collection_operand() -> None:
    catalog = _catalog()
    input_term = InputTerm(
        id="input_1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "two staff names"),
        operand=("Ada", "Grace"),
        value_type=CollectionType(TextType()),
    )
    set_ref = FactLocalRef("fact_1", FactLocalKind.SET, "s1")
    [option] = reference_binding_options(
        input_id=input_term.id,
        resolver_catalog=catalog,
        resolver_row_sources=build_row_source_catalog(catalog),
        expected_identity=None,
    )
    [grounding_task] = reference_grounding_tasks(
        (
            GroundingPartition(
                input_ref=input_term.id,
                use_refs=("fact_1:input_use:e1:1",),
                expected_value_type=CollectionType(IdentifierType("s1")),
                expected_set_ref=set_ref,
                operand_meaning="the staff members whose sales are counted",
            ),
        ),
        resolver_options_by_use_ref={"fact_1:input_use:e1:1": (option,)},
        denoted_instance_kinds_by_input_ref={input_term.id: "staff member"},
    )
    route = CompatibleIdentityRoute(
        option_id=option.id,
        identifier_kind=IdentifierKind.DESCRIPTIVE,
        lookup_request_param_refs=("list_staff_list.query.name",),
        returned_identity_verification_field_paths=("data.full_name",),
    )
    [task] = identity_resolution_tasks(
        (grounding_task,),
        compatible_bindings_by_task_ref={grounding_task.task_ref: (route,)},
    )
    [canonical_option] = task.canonical_options
    selection = IdentityRouteSelection(
        task_ref=task.task_ref,
        canonical_option_assessments=(),
        canonical_option_basis="Both operands denote staff identities.",
        canonical_option_id=canonical_option.canonical_option_id,
        resolver_route_assessments=(),
        resolver_route_basis="The route resolves each supplied staff name.",
        resolver_route_id=option.id,
    )
    data_access = _CollectionDataAccess()

    result = execute_identity_selection(
        task=task,
        selection=selection,
        input_term=input_term,
        full_catalog=catalog,
        data_access_port=data_access,
        source_read_key_prefix="test",
    )

    assert isinstance(result, ResolvedIdentity)
    assert result.canonical_value.typed_value.payload.canonical_value() == [
        "staff_ada",
        "staff_grace",
    ]
    assert data_access.names == ["Ada", "Grace"]


def _request(*, index, catalog: RelationCatalog) -> SemanticReadEligibilityRequest:
    return SemanticReadEligibilityRequest(
        indexes=(index,),
        source_catalog=build_row_source_catalog(catalog),
        answer_catalog=catalog,
        identity_tasks=(),
        resolver_catalog=RelationCatalog(),
    )


def _retention_payload(
    request: SemanticReadEligibilityRequest,
    requested_fact_id: str,
    *,
    retained_candidate_ref: str | None,
    relevant_field_refs: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "identity_outcomes": {},
        "read_assessments_by_requested_fact": {
            requested_fact_id: {
                candidate.candidate_ref: {
                    "assessment_basis": (
                        "This read can contribute answer-changing rows or values."
                        if candidate.candidate_ref == retained_candidate_ref
                        else "This read does not contribute to this requested fact."
                    ),
                    "relevant_field_refs": (
                        list(relevant_field_refs)
                        if candidate.candidate_ref == retained_candidate_ref
                        else []
                    ),
                    "decision": (
                        "RETAIN"
                        if candidate.candidate_ref == retained_candidate_ref
                        else "DROP"
                    ),
                }
                for candidate in request.read_candidates
            }
        },
    }


def _selected_identity_payload(
    *,
    task: IdentityResolutionTask,
    canonical_option_id: str,
    route_id: str,
) -> dict[str, object]:
    return {
        "canonical_option_assessments": [
            {
                "canonical_option_id": canonical_option_id,
                "assessment": "This is the identity used by the requested fact.",
                "decision": "FITS",
            }
        ],
        "canonical_option_basis": "The input denotes this identity.",
        "canonical_option_id": canonical_option_id,
        "resolver_route_assessments": [
            {
                "resolver_route_id": route_id,
                "assessment": "The route resolves the supplied identifier.",
                "decision": "FITS",
            }
        ],
        "resolver_route_basis": "The route resolves the selected identity.",
        "resolver_route_id": route_id,
        "evidence_refs": [task.task_ref, route_id],
        "outcome": "SELECTED",
    }


def _semantic_contract() -> ParsedSemanticQuestionContract:
    question = "Which staff member is named Ada?"
    meaning = parse_semantic_question_frame(
        {
            "decision_basis": "The request asks for staff members named Ada.",
            "outcome": {
                "kind": "question_meaning",
                "answer_requests": [
                    {
                        "return_request_basis": "Return the matching staff identity.",
                        "relational_shape_basis": (
                            "This is an ordinary identity-qualified row request."
                        ),
                        "request": {
                            "relational_shape": "ordinary",
                            "result_grain_basis": (
                                "One result row represents one qualifying staff member."
                            ),
                            "result": {
                                "kind": "one_result_per_qualifying_row",
                                "result_candidates": {
                                    "instance_kind": "staff members",
                                    "origin": {"kind": "question"},
                                },
                                "projection": {
                                    "projection_basis": (
                                        "Return the matching staff identity."
                                    ),
                                    "candidate_identity": "returned",
                                    "explicitly_requested_values": [],
                                },
                                "result_order": {
                                    "ordering_request_basis": (
                                        "The question requests no ordering."
                                    ),
                                    "ordering": {"kind": "no_ordering_requested"},
                                    "selection": {"kind": "all_results"},
                                },
                            },
                        },
                    }
                ],
                "supplied_values": {
                    "operands": [
                        {
                            "meaning": "the staff member being listed",
                            "denotation_basis": (
                                "Ada names one particular staff member."
                            ),
                            "answer_request_numbers": [1],
                            "entity_reference": {
                                "instance_kind": "staff members",
                                "value": {
                                    "kind": "single_identity",
                                    "identity_value": {"kind": "literal", "value": "Ada"},
                                    "origin": {"kind": "question"},
                                },
                            },
                        }
                    ],
                    "selection_limits": [],
                },
                "question_input_inventory_check": {
                    "all_input_like_phrases_declared": True,
                },
            },
        },
        question_context_texts=(question,),
    )
    assert isinstance(meaning, ParsedSemanticQuestionMeaning)
    parsed = parse_semantic_question_contract(
        {
            "decision_basis": "The request returns staff members named Ada.",
            "outcome": {
                "kind": "question_contract",
                "answer_requests": [
                    {
                        "requested_fact_ref": "fact_1",
                        "origin": _origin("staff members named Ada"),
                        "candidate_set": {
                            "instance_kind": "staff members",
                            "instance_interpretation": "resource_population",
                        },
                        "set_graph": {
                            "identity_input_relations": {"i1": None},
                            "requested_output_relations": {},
                            "other_related_sets": [],
                        },
                        "qualification": {
                            "kind": "input_comparison",
                            "input": {
                                "kind": "input_ref",
                                "input_ref": "i1",
                                "operand_meaning": "the staff member being listed",
                                "instance_kind": "staff members",
                            },
                            "operator": "equals",
                            "fact": {
                                "kind": "fact",
                                "identity_path": {
                                    "kind": "candidate_instance",
                                },
                                "origin": _origin("staff member identity"),
                            },
                        },
                        "grouping": [],
                        "ordering": [],
                        "selection": None,
                        "distinct_by": [],
                        "outputs": {
                            "result_key_outputs": [
                                {
                                    "expression": {
                                        "kind": "set_ref",
                                        "set_ref": "s1",
                                    }
                                }
                            ],
                            "requested_value_outputs": [],
                        },
                    }
                ],
            },
        },
        meaning=meaning,
        question_context_texts=(question,),
    )
    assert isinstance(parsed, ParsedSemanticQuestionContract)
    return parsed


def _origin(meaning: str) -> dict[str, object]:
    return {
        "source": "question_context",
        "meaning": meaning,
        "resolved_input_ref": None,
    }


def _frame_origin(meaning: str) -> dict[str, object]:
    return {"meaning": meaning, "origin": {"kind": "question"}}


def _catalog() -> RelationCatalog:
    return RelationCatalog(reads=(_staff_read(),))


class _DataAccess:
    def read(self, *, endpoint_name, args):
        assert endpoint_name == "list_staff_list"
        assert args == {"list_staff_list.query.name": "Ada"}
        return {
            "responseStatus": 200,
            "responseBody": {"data": [{"staff_id": "staff_1", "full_name": "Ada"}]},
        }


class _CollectionDataAccess:
    def __init__(self) -> None:
        self.names: list[str] = []

    def read(self, *, endpoint_name, args):
        assert endpoint_name == "list_staff_list"
        name = args["list_staff_list.query.name"]
        self.names.append(name)
        return {
            "responseStatus": 200,
            "responseBody": {
                "data": [{"staff_id": f"staff_{name.casefold()}", "full_name": name}]
            },
        }
