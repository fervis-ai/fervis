from tests.lookup.grounding._support import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    EntityTargetResolverSelection,
    GroundingTerminalKind,
    QuestionContract,
    RelationCatalog,
    RequestedFact,
    RequestedFactAnswerOutput,
    RuntimeValueContext,
    _DataAccess,
    _EndpointDataAccess,
    _NoCompatibleResolverGroundingModel,
    _NoGroundingModel,
    _NoShownResourceTypeGroundingModel,
    _compatible_binding,
    build_row_source_catalog,
    execute_compatible_reference_bindings,
    ground_question_inputs,
    reference_binding_issue,
    reference_binding_sources_by_known_input,
    reference_input_binding_tasks,
    replace,
)
from tests.lookup.grounding._fixtures import (
    _endpoint_result,
    _location_read,
    _location_with_area_read,
    _question_contract,
    _reference_input,
    _staff_detail_read,
    _staff_question_contract,
    _staff_read,
    _store_read,
)


def test_named_reference_option_does_not_preselect_identity_match_fields() -> None:
    catalog = RelationCatalog(reads=(_location_with_area_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="city"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog)
        },
    )

    assert len(task.options) == 1
    assert task.options[0].candidate.entity_kind == "location"


def test_named_reference_without_a_catalog_resolver_requires_clarification() -> None:
    output = ground_question_inputs(
        question="How many stores are in Nairobi?",
        question_contract=_question_contract("Nairobi", description="area"),
        full_catalog=RelationCatalog(),
        resolver_selections=(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-07-15",
            timezone="Africa/Nairobi",
        ),
        model_port=_NoGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    [issue] = output.ledger.issues
    assert issue.kind is GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert issue.known_input_id == "input_location"


def test_named_reference_with_no_positive_resolver_requires_clarification() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    output = ground_question_inputs(
        question="How many stores are in Nairobi?",
        question_contract=_question_contract("Nairobi", description="area"),
        full_catalog=catalog,
        resolver_selections=(
            EntityTargetResolverSelection(
                target_id="input_location",
                catalog_search_terms=("location",),
                selected_read_ids=("list_location_list",),
            ),
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-07-15",
            timezone="Africa/Nairobi",
        ),
        model_port=_NoCompatibleResolverGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    [issue] = output.ledger.issues
    assert issue.kind is GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert issue.known_input_id == "input_location"


def test_no_shown_resource_type_requires_clarification() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    output = ground_question_inputs(
        question="How many stores are in Nairobi?",
        question_contract=_question_contract("Nairobi", description="place"),
        full_catalog=catalog,
        resolver_selections=(
            EntityTargetResolverSelection(
                target_id="input_location",
                catalog_search_terms=("location",),
                selected_read_ids=("list_location_list",),
            ),
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-07-15",
            timezone="Africa/Nairobi",
        ),
        model_port=_NoShownResourceTypeGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    [issue] = output.ledger.issues
    assert issue.kind is GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert issue.known_input_id == "input_location"


def test_binding_tasks_use_catalog_for_recalled_and_validation_sources() -> None:
    full_catalog = RelationCatalog(reads=(_staff_read(), _staff_detail_read()))
    binding_sources = reference_binding_sources_by_known_input(
        full_row_sources=build_row_source_catalog(full_catalog),
        resolver_selections=(
            EntityTargetResolverSelection(
                target_id="input_staff",
                catalog_search_terms=("staff",),
                selected_read_ids=("list_staff_list",),
            ),
        ),
    )

    [task] = reference_input_binding_tasks(
        _staff_question_contract("staff_1", description="staff identifier"),
        resolver_catalog=full_catalog,
        resolver_sources_by_known_input=binding_sources,
    )

    assert {option.candidate.resolver_read_id for option in task.options} == {
        "list_staff_list",
        "get_staff_detail",
    }


def test_binding_tasks_preserve_resolver_selection_per_known_input() -> None:
    catalog = RelationCatalog(reads=(_location_read(), _store_read()))
    sources_by_input = reference_binding_sources_by_known_input(
        full_row_sources=build_row_source_catalog(catalog),
        resolver_selections=(
            EntityTargetResolverSelection(
                target_id="input_location",
                catalog_search_terms=("location",),
                selected_read_ids=("list_location_list",),
            ),
            EntityTargetResolverSelection(
                target_id="input_store",
                catalog_search_terms=("store",),
                selected_read_ids=("list_store_list",),
            ),
        ),
    )
    contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales between two places",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        role="ANSWER_VALUE",
                        description="total sales",
                    ),
                ),
                known_inputs=(
                    _reference_input("input_location", "Nairobi"),
                    _reference_input("input_store", "Pivot Mall"),
                ),
            ),
        )
    )

    tasks = reference_input_binding_tasks(
        contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input=sources_by_input,
    )
    read_ids_by_input = {
        task.known_input_id: {
            option.candidate.resolver_read_id for option in task.options
        }
        for task in tasks
    }

    assert read_ids_by_input == {
        "input_location": {"list_location_list"},
        "input_store": {"list_store_list"},
    }


def test_reference_option_read_failure_remains_scoped_to_its_binding() -> None:
    catalog = RelationCatalog(reads=(_location_read(), _store_read()))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Pivot Mall", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    entity_options = task.options

    class _PartiallyFailingDataAccess:
        def read(self, *, endpoint_name, args):
            del args
            if endpoint_name == "list_store_list":
                return {
                    "responseStatus": 404,
                    "responseBody": {"detail": "not found"},
                }
            return _endpoint_result(
                {"data": [{"location_id": "location_1", "name": "Pivot Mall"}]}
            )

    bindings = tuple(
        _compatible_binding(
            catalog,
            option,
            lookup_text="Pivot Mall",
        )
        for option in entity_options
    )
    executions = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=bindings,
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_PartiallyFailingDataAccess(),
    )
    executions_by_read = {
        option.candidate.resolver_read_id: executions[option.id]
        for option in entity_options
    }

    assert executions_by_read["list_location_list"].ledger is not None
    assert executions_by_read["list_location_list"].failure is None
    assert executions_by_read["list_store_list"].ledger is None
    assert executions_by_read["list_store_list"].failure is not None


def test_selected_reference_option_with_no_exact_match_is_unresolved() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess({"list_location_list": {"data": []}}),
    )[option.id]

    assert execution.ledger is not None
    issue = reference_binding_issue(
        task,
        candidate=option.candidate,
        values=execution.ledger.values,
    )
    assert issue is not None
    assert issue.kind is GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert issue.resolver_read_id == "list_location_list"


def test_truncated_reference_response_cannot_prove_a_unique_name_match() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")
    data_access = _DataAccess(
        {
            "responseStatus": 200,
            "responseBody": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}]
            },
            "truncated": True,
        }
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )[option.id]

    assert execution.ledger is not None
    issue = reference_binding_issue(
        task,
        candidate=option.candidate,
        values=execution.ledger.values,
        truncated=execution.truncated,
        matched_field_is_stable_unique=execution.matched_field_is_stable_unique,
    )
    assert issue is not None
    assert issue.kind is GroundingTerminalKind.INCOMPLETE_REFERENCE


def test_truncated_reference_response_can_prove_an_exact_stable_unique_match() -> None:
    base_read = _location_read()
    location_read = replace(
        base_read,
        candidate_keys=(
            *base_read.candidate_keys,
            CandidateKey(
                id="unique_name",
                entity_kind="location",
                components=(
                    CandidateKeyComponent(
                        id="name",
                        field_ref="field.data.name",
                    ),
                ),
                stable=True,
            ),
        ),
    )
    catalog = RelationCatalog(reads=(location_read,))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")
    data_access = _DataAccess(
        {
            "responseStatus": 200,
            "responseBody": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}]
            },
            "truncated": True,
        }
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )[option.id]

    assert execution.matched_field_is_stable_unique is True
    assert execution.ledger is not None
    assert (
        reference_binding_issue(
            task,
            candidate=option.candidate,
            values=execution.ledger.values,
            truncated=execution.truncated,
            matched_field_is_stable_unique=execution.matched_field_is_stable_unique,
        )
        is None
    )


def test_reference_match_requires_case_sensitive_exact_equality() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
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
                        {"location_id": "location_1", "name": "NAIROBI"},
                        {
                            "location_id": "location_2",
                            "name": "Greater Nairobi",
                        },
                    ]
                }
            }
        ),
    )[option.id]

    assert execution.ledger is not None
    assert execution.ledger.values == ()


def test_reference_match_does_not_use_another_response_field() -> None:
    base_read = _location_read()
    location_read = replace(
        base_read,
        fields=(
            *base_read.fields,
            CatalogField(
                ref="field.data.county",
                path="data.county",
                row_path_id="data",
                type="string",
            ),
        ),
    )
    catalog = RelationCatalog(reads=(location_read,))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(
        catalog,
        option,
        lookup_text="Nairobi",
        match_paths=("data.name",),
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess(
            {
                "list_location_list": {
                    "data": [
                        {
                            "location_id": "location_1",
                            "name": "Goldset Nairobi Store",
                            "county": "Nairobi",
                        }
                    ]
                }
            }
        ),
    )[option.id]

    assert execution.ledger is not None
    assert execution.ledger.values == ()


def test_identical_selected_resolver_requests_execute_once_for_distinct_inputs() -> (
    None
):
    catalog = RelationCatalog(reads=(_location_read(),))
    row_sources = build_row_source_catalog(catalog)
    contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="route between two locations",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="distance",
                        role="ANSWER_VALUE",
                        description="distance",
                    ),
                ),
                known_inputs=(
                    _reference_input("origin", "Nairobi"),
                    _reference_input("destination", "Nairobi"),
                ),
            ),
        )
    )
    tasks = reference_input_binding_tasks(
        contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "origin": row_sources,
            "destination": row_sources,
        },
    )
    selected_options = tuple(task.options[0] for task in tasks)
    selected_bindings = tuple(
        _compatible_binding(catalog, option, lookup_text="Nairobi")
        for option in selected_options
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}]
            }
        }
    )

    executions = execute_compatible_reference_bindings(
        tasks=tasks,
        bindings=selected_bindings,
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )

    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "Nairobi"})
    ]
    assert {
        execution.ledger.values[0].known_input_id
        for execution in executions.values()
        if execution.ledger is not None
    } == {"origin", "destination"}
