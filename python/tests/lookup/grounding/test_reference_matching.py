from tests.lookup.grounding._support import (
    CompatibleInputBinding,
    GroundingRequest,
    GroundingTerminalKind,
    GroundingTurnPrompt,
    IdentifierKind,
    LookupRequestParameter,
    RelationCatalog,
    _EndpointDataAccess,
    _compatible_binding,
    _grounding_prompt,
    _json_payload_from_prompt_section,
    build_row_source_catalog,
    execute_compatible_reference_bindings,
    parse_grounding_compatibility,
    reference_binding_issue,
    reference_input_binding_tasks,
    resolver_fit_question_for_option,
)
from tests.lookup.grounding._fixtures import (
    _location_read,
    _question_contract,
    _staff_question_contract,
    _staff_read,
    _uuid_person_read,
    _variant_person_read,
)


def test_selected_staff_lookup_fields_produce_one_canonical_staff_key() -> None:
    catalog = RelationCatalog(reads=(_staff_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _staff_question_contract("Azraah", description="staff member or seller"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_staff": row_sources},
    )
    [option] = task.options
    request = GroundingRequest(
        question="How much did Azraah make in sales?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    [input_task] = _json_payload_from_prompt_section(
        _grounding_prompt(request),
        "Known input binding tasks:",
    )["known_input_binding_tasks"]
    [prompt_option] = input_task["binding_options"]
    shown_field_paths = {
        field["path"]
        for row in prompt_option["api_read"]["response_rows"]
        for field in row["fields"]
    }
    assert "data.phone_number" in shown_field_paths

    compatibility = parse_grounding_compatibility(
        {
            "known_time_resolutions": {},
            "known_input_binding_reviews": {
                "input_staff": {
                    "resource_type_basis": "Azraah identifies a staff member.",
                    "resource_type_compatibility": {
                        "staff": "SAME_RESOURCE_TYPE"
                    },
                    "identifier_kind_basis": "Azraah is a descriptive name.",
                    "identifier_kind": "DESCRIPTIVE",
                    "option_reviews": {
                        option.id: {
                            "resource_type": "staff",
                            "resolver_fit_question": resolver_fit_question_for_option(
                                task=task,
                                option=option,
                            ),
                            "because": (
                                "The read accepts a name and returns staff-name fields."
                            ),
                            "resolution": {
                                "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                                "lookup_request_params": [
                                    {
                                        "param_ref": "list_staff_list.query.name",
                                        "value": "Azraah",
                                    }
                                ],
                                "returned_identity_verification_fields": [
                                    "data.first_name",
                                    "data.last_name",
                                    "data.full_name",
                                ],
                            },
                        }
                    }
                }
            },
        },
        request=request,
    )
    [binding] = compatibility.compatibilities[0].bindings
    data_access = _EndpointDataAccess(
        {
            "list_staff_list": {
                "data": [
                    {
                        "staff_id": "staff_azraah",
                        "first_name": "Azraah",
                        "last_name": "Ahmed",
                        "full_name": "Azraah Ahmed",
                        "phone_number": "+254700000001",
                    },
                    {
                        "staff_id": "staff_other",
                        "first_name": "Other",
                        "last_name": "Staff",
                        "full_name": "Other Staff",
                        "phone_number": "Azraah",
                    },
                ]
            }
        }
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )[option.id]

    assert binding.returned_identity_verification_field_paths == (
        "data.first_name",
        "data.last_name",
        "data.full_name",
    )
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": "Azraah"})
    ]
    assert execution.ledger is not None
    [value] = execution.ledger.values
    assert value.payload.key.component_values() == {"staff_id": "staff_azraah"}
    assert value.payload.matched_field_path == "data.first_name"
    assert value.payload.matched_value == "Azraah"


def test_two_exact_staff_matches_produce_typed_clarification_options() -> None:
    catalog = RelationCatalog(reads=(_staff_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _staff_question_contract("Azraah", description="staff member or seller"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_staff": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(
        catalog,
        option,
        lookup_text="Azraah",
        match_paths=("data.first_name", "data.last_name", "data.full_name"),
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess(
            {
                "list_staff_list": {
                    "data": [
                        {
                            "staff_id": "staff_1",
                            "first_name": "Azraah",
                            "last_name": "One",
                            "full_name": "Azraah One",
                        },
                        {
                            "staff_id": "staff_2",
                            "first_name": "Azraah",
                            "last_name": "Two",
                            "full_name": "Azraah Two",
                        },
                    ]
                }
            }
        ),
    )[option.id]

    assert execution.ledger is not None
    issue = reference_binding_issue(
        task,
        candidate=option.candidate,
        values=execution.ledger.values,
    )

    assert issue is not None
    assert issue.kind is GroundingTerminalKind.AMBIGUOUS_REFERENCE
    assert tuple(
        candidate.key.component_values() for candidate in issue.candidate_options
    ) == (
        {"staff_id": "staff_1"},
        {"staff_id": "staff_2"},
    )
    assert tuple(candidate.matched_field for candidate in issue.candidate_options) == (
        "field.data.first_name",
        "field.data.first_name",
    )
    assert tuple(candidate.matched_value for candidate in issue.candidate_options) == (
        "Azraah",
        "Azraah",
    )


def test_resolver_row_source_variants_keep_distinct_identity_and_defaults() -> None:
    catalog = RelationCatalog(reads=(_variant_person_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Azraah", description="person"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )

    assert len(task.options) == 2
    assert len({option.id for option in task.options}) == 2
    prompt_options = GroundingTurnPrompt(
        GroundingRequest(
            question="Which person is named Azraah?",
            tasks=(task,),
            resolver_catalog=catalog,
        )
    ).known_input_binding_tasks_payload()["known_input_binding_tasks"][0][
        "binding_options"
    ]
    shown_shape_defaults = {
        parameter["default"]
        for option in prompt_options
        for parameter in option["api_read"]["input_params"]
        if parameter["param_ref"] == "list_people.query.shape"
    }
    assert shown_shape_defaults == {"SUMMARY", "DETAIL"}

    bindings = tuple(
        CompatibleInputBinding(
            option_id=option.id,
            lookup_value="Azraah",
            identifier_kind=IdentifierKind.DESCRIPTIVE,
            lookup_request_parameters=(
                LookupRequestParameter(
                    param_ref="list_people.query.name",
                    value="Azraah",
                ),
            ),
            returned_identity_verification_field_paths=("data.name",),
        )
        for option in task.options
    )
    data_access = _EndpointDataAccess(
        {"list_people": {"data": [{"person_id": "person_1", "name": "Azraah"}]}}
    )

    executions = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=bindings,
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )

    assert set(executions) == {option.id for option in task.options}
    assert data_access.calls == [
        (
            "list_people",
            {"list_people.query.name": "Azraah", "list_people.query.shape": "SUMMARY"},
        ),
        (
            "list_people",
            {"list_people.query.name": "Azraah", "list_people.query.shape": "DETAIL"},
        ),
    ]


def test_resolver_verifies_the_typed_request_value_against_typed_response_fields() -> (
    None
):
    uppercase_uuid = "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"
    canonical_uuid = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    catalog = RelationCatalog(reads=(_uuid_person_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract(uppercase_uuid, description="person"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    request = GroundingRequest(
        question="Which person?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    compatibility = parse_grounding_compatibility(
        {
            "known_time_resolutions": {},
            "known_input_binding_reviews": {
                "input_location": {
                    "resource_type_basis": "The UUID identifies a person.",
                    "resource_type_compatibility": {
                        "person": "SAME_RESOURCE_TYPE"
                    },
                    "identifier_kind_basis": "The lookup is a primary key.",
                    "identifier_kind": "PRIMARY_KEY",
                    "option_reviews": {
                        option.id: {
                            "resource_type": "person",
                            "resolver_fit_question": resolver_fit_question_for_option(
                                task=task,
                                option=option,
                            ),
                            "because": "The exact UUID identifies the returned person.",
                            "resolution": {
                                "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                                "lookup_request_params": [
                                    {
                                        "param_ref": "get_person.query.person_id",
                                        "value": canonical_uuid,
                                    }
                                ],
                                "returned_identity_verification_fields": [
                                    "data.person_id"
                                ],
                            },
                        }
                    },
                }
            },
        },
        request=request,
    )
    [binding] = compatibility.compatibilities[0].bindings
    data_access = _EndpointDataAccess(
        {"get_person": {"data": {"person_id": canonical_uuid, "name": "Azraah"}}}
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )[option.id]

    assert data_access.calls == [
        ("get_person", {"get_person.query.person_id": canonical_uuid})
    ]
    assert execution.ledger is not None
    [value] = execution.ledger.values
    assert value.payload.key.component_values() == {"person_id": canonical_uuid}


def test_resolver_string_verification_keeps_only_the_exact_case_match() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="location"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog)
        },
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess(
            {
                "list_location_list": {
                    "data": [
                        {"location_id": "lowercase", "name": "nairobi"},
                        {"location_id": "exact", "name": "Nairobi"},
                    ]
                }
            }
        ),
    )[option.id]

    assert execution.ledger is not None
    [value] = execution.ledger.values
    assert value.payload.key.component_values() == {"location_id": "exact"}
