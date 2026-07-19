from tests.lookup.grounding._support import (
    CanonicalInputSelection,
    CatalogSelectionResult,
    GroundingCompatibilityResult,
    InputBindingCompatibility,
    ReadEligibilityRequest,
    ReadEligibilityResult,
    ReadEligibilityTurnPrompt,
    RelationCatalog,
    RequestedFactCatalogSelection,
    _EndpointDataAccess,
    _compatible_binding,
    build_row_source_catalog,
    parse_read_eligibility,
    read_eligibility_candidate_surface,
    reference_input_binding_tasks,
    resolve_read_eligibility,
    validate,
)
from tests.lookup.grounding._fixtures import (
    _location_alias_read,
    _location_read,
    _question_contract,
)


def test_selected_canonical_identity_resolves_without_a_read_target() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    contract = _question_contract("Nairobi", description="location")
    tasks = reference_input_binding_tasks(
        contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    compatibility = GroundingCompatibilityResult(
        compatibilities=(
            InputBindingCompatibility(
                known_input_id="input_location",
                bindings=tuple(
                    _compatible_binding(catalog, option, lookup_text="Nairobi")
                    for option in tasks[0].options
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=(),
                rankings=(),
                selected_read_ids=(),
            ),
        ),
        selected_read_ids=(),
    )
    request = ReadEligibilityRequest(
        question="How much did we sell in Nairobi?",
        question_contract=contract,
        requested_facts=contract.requested_facts,
        catalog_selection=catalog_selection,
        conversation_context={},
        binding_tasks=tasks,
        compatible_reference_bindings=compatibility.compatibilities[0].bindings,
        resolver_catalog=catalog,
    )
    surface = read_eligibility_candidate_surface(request)
    [canonical_option] = surface.canonical_options
    result = ReadEligibilityResult(
        read_assessments=(),
        canonical_inputs=(
            CanonicalInputSelection(
                option=canonical_option,
                selected_resolver_binding=canonical_option.resolver_bindings[0],
                interpretation_question="Which location?",
                canonical_option_assessments=(
                    (canonical_option.id, "The location read exposes this identity."),
                ),
                because="Nairobi denotes the returned location.",
                resolver_option_assessments=(
                    (
                        canonical_option.resolver_bindings[0].option_id,
                        "The name parameter retrieves the returned location.",
                    ),
                ),
            ),
        ),
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}],
            }
        }
    )
    resolved = resolve_read_eligibility(
        request=request,
        result=result,
        full_catalog=catalog,
        data_access_port=data_access,
        source_read_key_prefix="test",
    )

    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "Nairobi"})
    ]
    assert len(resolved.ledger.values) == 1
    [value] = resolved.ledger.values
    assert value.known_input_id == "input_location"
    assert value.payload.matched_field_ref == "field.data.name"
    assert value.payload.matched_field_path == "data.name"
    assert value.payload.matched_value == "Nairobi"
    assert resolved.ledger.uses == ()


def test_read_eligibility_executes_only_the_selected_reference_option() -> None:
    catalog = RelationCatalog(reads=(_location_read(), _location_alias_read()))
    contract = _question_contract("Nairobi", description="location")
    tasks = reference_input_binding_tasks(
        contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    compatibility = GroundingCompatibilityResult(
        compatibilities=(
            InputBindingCompatibility(
                known_input_id="input_location",
                bindings=tuple(
                    _compatible_binding(catalog, option, lookup_text="Nairobi")
                    for option in tasks[0].options
                ),
            ),
        )
    )
    request = ReadEligibilityRequest(
        question="How much did we sell in Nairobi?",
        question_contract=contract,
        requested_facts=contract.requested_facts,
        catalog_selection=CatalogSelectionResult(
            relation_catalog=catalog,
            requested_fact_selections=(
                RequestedFactCatalogSelection(
                    requested_fact_id="fact_1",
                    query_terms=(),
                    rankings=(),
                    selected_read_ids=(),
                ),
            ),
            selected_read_ids=(),
        ),
        conversation_context={},
        binding_tasks=tasks,
        compatible_reference_bindings=compatibility.compatibilities[0].bindings,
        resolver_catalog=catalog,
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}],
            },
            "list_location_alias_list": {
                "data": [{"location_id": "location_2", "display_name": "Nairobi"}],
            },
        }
    )

    surface = read_eligibility_candidate_surface(request)

    assert data_access.calls == []
    [canonical_option] = surface.canonical_options
    resolver_options_by_id = {
        option.id: option for task in tasks for option in task.options
    }
    assert {
        binding.option_id for binding in canonical_option.resolver_bindings
    } == set(resolver_options_by_id)
    selected_resolver_binding = next(
        binding
        for binding in canonical_option.resolver_bindings
        if resolver_options_by_id[
            binding.option_id
        ].candidate.resolver_read_id
        == "list_location_list"
    )
    interpretation_question = (
        surface.card_payload["requested_fact_read_candidates"][0]["known_inputs"][0][
            "interpretation_question"
        ]
    )
    payload = {
        "requested_fact_assessments": {
            "fact_1": {
                "read_candidate_reviews": {},
                "canonical_inputs": {
                    canonical_option.known_input_token: {
                        "interpretation_question": interpretation_question,
                        "canonical_option_assessments": {
                            canonical_option.id: (
                                "The candidate reads were assessed under this identity."
                            )
                        },
                        "because": (
                            "Nairobi denotes the location returned by this read."
                        ),
                        "canonical_option_id": canonical_option.id,
                        "resolver_option_assessments": {
                            binding.option_id: (
                                "The shown lookup parameters retrieve the location "
                                "and its returned fields verify that location."
                            )
                            for binding in canonical_option.resolver_bindings
                        },
                        "resolver_option_id": selected_resolver_binding.option_id,
                    }
                },
            }
        }
    }
    schema = ReadEligibilityTurnPrompt(request).response_contract().provider_schema
    validate(instance=payload, schema=schema)
    result = parse_read_eligibility(payload, request=request)
    resolved = resolve_read_eligibility(
        request=request,
        result=result,
        full_catalog=catalog,
        data_access_port=data_access,
        source_read_key_prefix="test",
    )

    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "Nairobi"})
    ]
    assert len(resolved.ledger.values) == 1
