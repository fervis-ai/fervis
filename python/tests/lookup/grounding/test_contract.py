from tests.lookup.grounding._support import (
    ApiReadResponseShapeProjector,
    GroundingRequest,
    GroundingTurnPrompt,
    IdentifierKind,
    InputBindingCompatibility,
    NO_SHOWN_RESOURCE_TYPE,
    RelationCatalog,
    ResourceTypeMatch,
    ValidationError,
    _grounding_prompt,
    _grounding_review_arguments,
    _json_payload_from_prompt_section,
    build_row_source_catalog,
    parse_grounding_compatibility,
    pytest,
    reference_input_binding_tasks,
    replace,
    resolver_option_surface,
    validate,
)
from tests.lookup.grounding._fixtures import (
    _flow_read,
    _location_detail_read,
    _location_read,
    _location_with_area_read,
    _question_contract,
    _staff_detail_read,
    _staff_question_contract,
    _staff_read,
    _uuid_person_read,
    _variant_person_read,
)


def test_grounding_prompt_instructs_binding_id_copying_verbatim():
    [task] = reference_input_binding_tasks(
        _question_contract("Shipment Tracker", description="flow"),
        resolver_catalog=RelationCatalog(reads=(_flow_read(),)),
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(
                RelationCatalog(reads=(_flow_read(),))
            )
        },
    )
    request = GroundingRequest(
        question="What does Shipment Tracker do?",
        tasks=(task,),
        resolver_catalog=RelationCatalog(reads=(_flow_read(),)),
    )
    prompt = _grounding_prompt(request)

    assert "Known input binding tasks:" in prompt
    assert "<known_input_binding_tasks>" in prompt
    assert '<known_input id="input_location"' in prompt
    assert "<shown_resource_types>" in prompt
    assert "<resource_type>flow</resource_type>" in prompt
    assert "<binding_option" in prompt
    assert "<api_read" in prompt
    assert '"binding_options":' not in prompt
    assert "Write resource_type_basis first" in prompt
    assert "set resource_type_x to exactly one shown_resource_type" in prompt
    assert "SAME_RESOURCE_TYPE means it exactly equals resource_type_x" in prompt
    assert "Within each known-input review, write fields in this order: resource_type_basis, resource_type_x, identifier_kind_basis, identifier_kind, option_reviews." in prompt
    assert "Within each option review, write fields in this order: resource_type, resource_type_match, resolver_fit_question, because, resolution." in prompt
    assert "returned_identity_verification_fields are returned-resource fields that may exactly equal lookup_text" in prompt
    assert "Include each selected field exactly once." not in prompt
    assert "can/cannot identify the returned" not in prompt
    assert "because briefly explains the capability decision" not in prompt
    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    bindings_schema = schema["properties"]["known_input_binding_reviews"]
    assert bindings_schema["type"] == "object"
    assert bindings_schema["required"] == [task.known_input_id]
    known_input_review = bindings_schema["properties"][task.known_input_id]
    assert list(known_input_review["properties"]) == [
        "resource_type_basis",
        "resource_type_x",
        "identifier_kind_basis",
        "identifier_kind",
        "option_reviews",
    ]
    assert known_input_review["properties"]["resource_type_x"]["enum"] == [
        "flow",
        NO_SHOWN_RESOURCE_TYPE,
    ]
    option_reviews = known_input_review["properties"]["option_reviews"]
    assert option_reviews["required"] == [option.id for option in task.options]
    first_review = option_reviews["properties"][task.options[0].id]
    assert "oneOf" not in first_review
    assert list(first_review["properties"]) == [
        "resource_type",
        "resource_type_match",
        "resolver_fit_question",
        "because",
        "resolution",
    ]
    resolution = first_review["properties"]["resolution"]
    assert resolution["properties"]["decision"]["enum"] == [
        "CANNOT_RESOLVE_LOOKUP_TEXT"
    ]
    assert list(resolution["properties"]) == [
        "decision",
        "lookup_request_params",
        "returned_identity_verification_fields",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda review: review.update(resource_type_x="staff"),
            "resource_type_x was not shown",
        ),
        (
            lambda review: next(iter(review["option_reviews"].values())).update(
                resource_type="staff"
            ),
            "option resource_type mismatch",
        ),
        (
            lambda review: next(iter(review["option_reviews"].values())).update(
                resource_type_match=ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value
            ),
            "resource_type_match contradicts resource types",
        ),
    ),
)
def test_grounding_parser_rejects_inconsistent_resource_type_contract(
    mutate,
    message: str,
) -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="location"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    request = GroundingRequest(
        question="How many stores are in Nairobi?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = _grounding_review_arguments(
        _grounding_prompt(request),
        selected_by_input={},
    )
    review = payload["known_input_binding_reviews"][task.known_input_id]
    mutate(review)

    with pytest.raises(ValueError, match=message):
        parse_grounding_compatibility(payload, request=request)


def test_grounding_parser_accepts_no_shown_resource_type_only_with_negative_options(
) -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    request = GroundingRequest(
        question="How many stores are in Nairobi?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = _grounding_review_arguments(
        _grounding_prompt(request),
        selected_by_input={},
    )
    review = payload["known_input_binding_reviews"][task.known_input_id]
    review["resource_type_x"] = NO_SHOWN_RESOURCE_TYPE
    for option_review in review["option_reviews"].values():
        option_review["resource_type_match"] = (
            ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value
        )

    result = parse_grounding_compatibility(payload, request=request)

    assert result.compatibilities == (
        InputBindingCompatibility(known_input_id=task.known_input_id, bindings=()),
    )


def test_grounding_selects_one_resource_type_before_reviewing_resolver_mechanics(
) -> None:
    catalog = RelationCatalog(
        reads=(_staff_read(), _staff_detail_read(), _location_detail_read())
    )
    [task] = reference_input_binding_tasks(
        _staff_question_contract(
            "staff-1",
            description="staff identifier",
            field_label_text="staff_id",
        ),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    option_by_read = {
        option.candidate.resolver_read_id: option for option in task.options
    }
    request = GroundingRequest(
        question="How many sales did the staff with staff_id staff-1 sell?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = _grounding_review_arguments(
        _grounding_prompt(request),
        selected_by_input={
            task.known_input_id: option_by_read["get_staff_detail"].id,
        },
    )
    review = payload["known_input_binding_reviews"][task.known_input_id]
    review["identifier_kind_basis"] = "staff-1 is the supplied primary key."
    review["identifier_kind"] = IdentifierKind.PRIMARY_KEY.value

    result = parse_grounding_compatibility(payload, request=request)

    assert review["resource_type_x"] == "staff"
    assert {
        option_by_read[read_id].candidate.entity_kind: option_review[
            "resource_type_match"
        ]
        for read_id, option_review in (
            (
                option.candidate.resolver_read_id,
                review["option_reviews"][option.id],
            )
            for option in task.options
        )
    } == {
        "staff": ResourceTypeMatch.SAME_RESOURCE_TYPE.value,
        "location": ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value,
    }
    [binding] = result.compatibilities[0].bindings
    assert binding.option_id == option_by_read["get_staff_detail"].id


@pytest.mark.parametrize(
    ("parameter_type", "lookup_text"),
    (
        ("boolean", "not-a-boolean"),
        ("integer", "not-an-integer"),
        ("uuid", "not-a-uuid"),
    ),
)
def test_grounding_schema_rejects_lookup_value_for_incompatible_parameter_type(
    parameter_type: str,
    lookup_text: str,
) -> None:
    read = _staff_detail_read(param_type=parameter_type)
    catalog = RelationCatalog(reads=(read,))
    [task] = reference_input_binding_tasks(
        _staff_question_contract(lookup_text, description="staff member"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question=f"Show the staff member {lookup_text}",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = _grounding_review_arguments(
        _grounding_prompt(request),
        selected_by_input={task.known_input_id: option.id},
    )

    with pytest.raises(ValidationError):
        validate(
            payload,
            GroundingTurnPrompt(request).response_contract().provider_schema,
        )


def test_grounding_schema_rejects_repeated_response_match_field() -> None:
    catalog = RelationCatalog(reads=(_uuid_person_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract(
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
            description="person",
        ),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question="Which person?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    option_schema = schema["properties"]["known_input_binding_reviews"][
        "properties"
    ][task.known_input_id]["properties"]["option_reviews"]["properties"][option.id]
    match_schema = option_schema["properties"]["resolution"]["oneOf"][1][
        "properties"
    ]["returned_identity_verification_fields"]

    with pytest.raises(ValidationError):
        validate(["data.person_id", "data.person_id"], match_schema)


def test_grounding_schema_requires_no_default_resolver_parameters() -> None:
    read = _staff_detail_read()
    catalog = RelationCatalog(reads=(read,))
    [task] = reference_input_binding_tasks(
        _staff_question_contract("staff-1", description="staff member"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question="Show staff member staff-1",
        tasks=(task,),
        resolver_catalog=catalog,
    )

    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    option_schema = schema["properties"]["known_input_binding_reviews"][
        "properties"
    ]["input_staff"]["properties"]["option_reviews"]["properties"][option.id]
    resolution_schema = option_schema["properties"]["resolution"]
    assert [
        branch["properties"]["decision"]["enum"]
        for branch in resolution_schema["oneOf"]
    ] == [
        ["CANNOT_RESOLVE_LOOKUP_TEXT"],
        ["CAN_RESOLVE_LOOKUP_TEXT"],
    ]
    positive_resolution = resolution_schema["oneOf"][1]
    lookup_params_schema = positive_resolution["properties"][
        "lookup_request_params"
    ]

    assert lookup_params_schema["type"] == "array"
    assert lookup_params_schema["minItems"] == 1
    assert lookup_params_schema["maxItems"] == 1
    assert lookup_params_schema["items"]["properties"]["param_ref"]["enum"] == [
        "get_staff_detail.path.staff_id"
    ]
    verification_fields = positive_resolution["properties"][
        "returned_identity_verification_fields"
    ]
    assert verification_fields["type"] == "array"
    assert verification_fields["maxItems"] == len(
        verification_fields["items"]["enum"]
    )
    assert verification_fields["uniqueItems"] is True


@pytest.mark.parametrize(
    "resolution",
    (
        {
            "decision": "CAN_RESOLVE_LOOKUP_TEXT",
            "lookup_request_params": [],
            "returned_identity_verification_fields": [],
        },
        {
            "decision": "CANNOT_RESOLVE_LOOKUP_TEXT",
            "lookup_request_params": [
                {
                    "param_ref": "get_staff_detail.path.staff_id",
                    "value": "staff-1",
                }
            ],
            "returned_identity_verification_fields": ["data.staff_id"],
        },
    ),
)
def test_grounding_schema_correlates_resolution_decision_with_selected_mechanics(
    resolution: dict[str, object],
) -> None:
    catalog = RelationCatalog(reads=(_staff_detail_read(),))
    [task] = reference_input_binding_tasks(
        _staff_question_contract("staff-1", description="staff member"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question="Show staff member staff-1",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    resolution_schema = schema["properties"]["known_input_binding_reviews"][
        "properties"
    ]["input_staff"]["properties"]["option_reviews"]["properties"][option.id][
        "properties"
    ]["resolution"]

    with pytest.raises(ValidationError):
        validate(resolution, resolution_schema)


def test_named_reference_options_return_canonical_entity_keys():
    catalog = RelationCatalog(reads=(_flow_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("operations", description="tag label"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog)
        },
    )

    surfaces = tuple(option.candidate.result_surface for option in task.options)

    assert surfaces
    assert set(surfaces) == {"entity flow:primary_key"}
    assert len(task.options) == 1


def test_grounding_shows_one_shared_endpoint_projection_per_canonical_result() -> None:
    catalog = RelationCatalog(reads=(_staff_read(),))
    [task] = reference_input_binding_tasks(
        _staff_question_contract("Azraah", description="staff member or seller"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    request = GroundingRequest(
        question="How much did Azraah make in sales?",
        tasks=(task,),
        resolver_catalog=catalog,
    )

    [input_task] = _json_payload_from_prompt_section(
        _grounding_prompt(request),
        "Known input binding tasks:",
    )["known_input_binding_tasks"]
    [option] = input_task["binding_options"]

    assert option["api_read"] == ApiReadResponseShapeProjector(
        catalog.read("list_staff_list")
    ).prompt_payload(row_path_ids=("data",))
    assert option["canonical_result"] == {
        "entity_kind": "staff",
        "key_id": "primary_key",
        "components": [
            {
                "component_id": "staff_id",
                "field_path": "data.staff_id",
            }
        ],
    }


def test_grounding_preserves_shared_parameter_context_and_source_overlay() -> None:
    base_read = _variant_person_read()
    name_param, shape_param = base_read.params
    read = replace(
        base_read,
        params=(
            replace(name_param, description="Match a person by name."),
            replace(
                shape_param,
                description="Select the returned person representation.",
                choice_labels={"SUMMARY": "Summary", "DETAIL": "Detail"},
            ),
        ),
    )
    catalog = RelationCatalog(reads=(read,))
    [task] = reference_input_binding_tasks(
        _question_contract("Nadia", description="person"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    option = task.options[0]
    request = GroundingRequest(
        question="What is Nadia's status?",
        tasks=(task,),
        resolver_catalog=catalog,
    )

    input_params = resolver_option_surface(request, option).prompt_payload()[
        "api_read"
    ]["input_params"]

    assert input_params[0]["description"] == "Match a person by name."
    assert input_params[1]["description"] == (
        "Select the returned person representation."
    )
    assert input_params[1]["choice_labels"] == {
        "SUMMARY": "Summary",
        "DETAIL": "Detail",
    }
    assert input_params[1]["default"] in {"SUMMARY", "DETAIL"}
    assert input_params[1]["default_source"] == "source_variant"
    assert input_params[1]["semantics"] == "response_shape"


def test_shared_endpoint_projection_keeps_distinct_fields_with_same_leaf_name() -> None:
    read = _location_with_area_read()

    [row] = ApiReadResponseShapeProjector(read).prompt_payload(row_path_ids=("data",))[
        "response_rows"
    ]

    assert [field["path"] for field in row["fields"]] == [
        "data.location_id",
        "data.name",
        "data.type",
        "data.area.area_id",
        "data.area.name",
    ]


def test_grounding_does_not_offer_related_resource_fields_as_match_fields() -> None:
    catalog = RelationCatalog(reads=(_location_with_area_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question="How many stores are in Nairobi?",
        tasks=(task,),
        resolver_catalog=catalog,
    )

    surface = resolver_option_surface(request, option)
    match_field_paths = {field.path for field in surface.response_match_fields}
    assert "data.name" in match_field_paths
    assert "data.area.area_id" not in match_field_paths
    assert "data.area.name" not in match_field_paths

    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    option_review = schema["properties"]["known_input_binding_reviews"][
        "properties"
    ]["input_location"]["properties"]["option_reviews"]["properties"][
        option.id
    ]
    legal_match_paths = option_review["properties"]["resolution"]["oneOf"][1][
        "properties"
    ]["returned_identity_verification_fields"]["items"]["enum"]
    assert "data.name" in legal_match_paths
    assert "data.area.area_id" not in legal_match_paths
    assert "data.area.name" not in legal_match_paths
