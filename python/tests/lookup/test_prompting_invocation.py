import json

from tests.lookup.source_binding_helpers import source_binding_request

from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    EndpointRead,
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
from fervis.lookup.grounding.model import (
    GroundingRequest,
    InputBindingOption,
    InputBindingKeyComponent,
    KnownInputBindingTask,
    ResolverCandidate,
)
from fervis.lookup.grounding.prompt import GroundingTurnPrompt
from fervis.lookup.fact_plan.row_sources import build_row_source_catalog
from fervis.lookup.answer_program.relations import (
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_planning.request import (
    FactPlanRequest,
    PatternFactPlanTurnPrompt,
)
from fervis.lookup.turn_prompts import (
    ApprovedPromptChars,
    PromptApprovalManifest,
    build_turn_prompt_context,
)
from fervis.lookup.query_enrichment import (
    QueryEnrichmentRequest,
    QueryEnrichmentTurnPrompt,
)
from fervis.lookup.question_contract import (
    QuestionContract,
    QuestionContractRequest,
    QuestionContractTurnPrompt,
    RequestedFact,
    RequestedFactAnswerOutput,
)
from fervis.lookup.source_binding import (
    AnswerPopulation,
    BoundSource,
    SourceFulfillment,
    SourceBindingTurnPrompt,
)
from fervis.lookup.plan_selection import (
    SourceStrategyMember,
    BoundRoleTarget,
    BoundSourceStrategyMember,
    BoundSelectedSourceStrategy,
    BoundPlanSelectionSet,
    PlanSelectionSet,
    SelectedSourceStrategy,
)


_APPROVED_CHARS = {
    "question contract": (364, 18074, 27161),
    "query enrichment": (364, 5185, 7408),
    "grounding": (364, 6105, 8964),
    "source binding": (364, 16267, 20138),
    "pattern fact planning": (364, 3497, 5311),
}


def test_model_turn_invocations_preserve_provider_payload_shape():
    for invocation in _turn_invocations():
        payload = invocation.to_provider_payload()

        assert set(payload) == {
            "system_prompt",
            "prompt",
            "output_mode",
            "tool_specs",
        }
        assert payload["system_prompt"] == invocation.system_prompt
        assert payload["prompt"] == invocation.prompt_text
        assert payload["tool_specs"] == invocation.tool_specs
        assert len(invocation.tool_specs) >= 1
        assert all(tool_spec.strict is True for tool_spec in invocation.tool_specs)


def test_model_turn_invocations_match_approved_prompt_chars():
    manifest = PromptApprovalManifest(
        approved_chars=tuple(
            ApprovedPromptChars(
                turn_name=turn_name,
                fixture_name="minimal",
                system_prompt_chars=chars[0],
                prompt_text_chars=chars[1],
                provider_payload_chars=chars[2],
            )
            for turn_name, chars in _APPROVED_CHARS.items()
        )
    )

    for invocation in _turn_invocations():
        approved = manifest.maximum_approved_chars_for(
            turn_name=invocation.turn_name,
            fixture_name="minimal",
        )

        assert len(invocation.system_prompt) == approved.system_prompt_chars
        assert len(invocation.prompt_text) == approved.prompt_text_chars
        assert (
            len(_serialized_provider_payload(invocation.to_provider_payload()))
            == approved.provider_payload_chars
        )


def test_question_contract_prompt_states_relational_ownership_together():
    invocation = next(
        item for item in _turn_invocations() if item.turn_name == "question contract"
    )
    ownership = "\n".join(
        (
            "Relational Ownership",
            "answer_subject: Kind of candidate instance to which answer_expression applies.",
            "answer_population: Candidate instances qualifying independently, before cross-instance operations.",
            "answer_expression: Operation over candidates: list, order, compare, rank, limit, or aggregate.",
            "answer_outputs: Values or facts projected from the result.",
        )
    )

    assert ownership in invocation.prompt_text


def test_source_binding_prompt_distinguishes_ranked_physical_operations():
    invocation = next(
        item for item in _turn_invocations() if item.turn_name == "source binding"
    )

    assert (
        "ranked_rows ranks individual source rows without grouping or aggregation."
        in invocation.prompt_text
    )
    assert (
        "ranked_aggregate groups source rows by an entity key, aggregates a measure "
        "within each group, and ranks the resulting groups." in invocation.prompt_text
    )
    assert (
        "A metric fits when it is the correct measure input to the requested "
        "computation. Do not reject it merely because aggregation or another later "
        "operation produces the final answer value." in invocation.prompt_text
    )
    assert "directly yields the requested measure" not in invocation.prompt_text
    assert "candidate's metric_operation" not in invocation.prompt_text


def test_model_turn_invocations_render_expected_shared_frame():
    for invocation in _turn_invocations():
        assert invocation.prompt_text.startswith("Current question:\n")
        assert f"We are currently on the {invocation.turn_name} step." in (
            invocation.prompt_text
        )


def _turn_invocations():
    question = "How many sales happened today?"
    context = {}
    contract = _question_contract()
    catalog = _catalog()
    selection = _catalog_selection(catalog)

    question_request = QuestionContractRequest(
        current_question=question,
        conversation_context=context,
    )
    yield QuestionContractTurnPrompt(question_request).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context=context,
        )
    )

    query_request = QueryEnrichmentRequest(
        question=question,
        conversation_context=context,
        requested_facts=contract.requested_facts,
        relation_catalog=catalog,
    )
    yield QueryEnrichmentTurnPrompt(query_request).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context=context,
        )
    )

    resolver_catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="read_today",
                endpoint_name="read_today",
                resource_names=("calendar_value",),
                params=(
                    CatalogParam(
                        ref="read_today.query.value",
                        name="value",
                        source=ParamSource.QUERY,
                        type="string",
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
                        ref="read_today.id",
                        path="data.id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="read_today.value",
                        path="data.value",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="calendar_value",
                        components=(
                            CandidateKeyComponent(
                                id="id",
                                field_ref="read_today.id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
            ),
        ),
    )
    grounding_request = GroundingRequest(
        question=question,
        conversation_context=context,
        resolver_catalog=resolver_catalog,
        tasks=(
            KnownInputBindingTask(
                known_input_id="input_today",
                known_input_text="today",
                known_input_kind="literal_text",
                requested_fact_id="fact_1",
                lookup_text="today",
                options=(
                    InputBindingOption(
                        id="bind_input_today_1",
                        known_input_id="input_today",
                        candidate=ResolverCandidate(
                            known_input_id="input_today",
                            resolver_source=next(
                                source
                                for source in build_row_source_catalog(
                                    resolver_catalog
                                ).sources
                                if source.read_id == "read_today"
                            ),
                            entity_kind="calendar_value",
                            key_id="primary_key",
                            key_components=(
                                InputBindingKeyComponent(
                                    component_id="id",
                                    field_id="field.id",
                                    field_ref="read_today.id",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    yield GroundingTurnPrompt(grounding_request).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context=context,
        )
    )

    source_request = source_binding_request(
        question=question,
        question_contract=contract,
        requested_facts=contract.requested_facts,
        relation_catalog=catalog,
        catalog_selection=selection,
        plan_selection=_selected_plan(),
        memory_inputs={},
        conversation_context=context,
    )
    yield SourceBindingTurnPrompt(source_request).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context=context,
            memory_payload={},
        )
    )

    fact_request = FactPlanRequest(
        question=question,
        question_contract=contract,
        relation_catalog=catalog,
        catalog_selection=selection,
        bound_sources=(
            BoundSource(
                id="source_fact_1",
                requested_fact_id="fact_1",
                answer_population=AnswerPopulation(
                    population_binding_id="pop.source_fact_1.candidate_population",
                    intent_text="sales",
                    match_basis_explanation="sales defines the source population",
                ),
                source=RelationSource(
                    kind=SourceKind.API_READ,
                    read_id="sales",
                ),
                available_field_ids=("amount",),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        metric_measure_evidence_ids=("amount",),
                        match_basis_explanation=(
                            "metric_measure is fulfilled by amount because "
                            "amount provides the answer output evidence."
                        ),
                    ),
                ),
            ),
        ),
        memory_inputs={},
        conversation_context=context,
    )
    plan_selection = BoundPlanSelectionSet(
        plan_selections=(
            BoundSelectedSourceStrategy(
                requested_fact_id="fact_1",
                plan_selection_id="fact_1.aggregate_scalar.source_fact_1",
                source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                plan_shape="aggregate_scalar",
                required_answer_output_ids=("answer_1",),
                source_members=(
                    BoundSourceStrategyMember(
                        source_candidate_id="source_fact_1",
                        role_targets=(
                            BoundRoleTarget(
                                requirement_id="source",
                                source_candidate_id="source_fact_1",
                                source_binding_ids=("source_fact_1",),
                            ),
                        ),
                        field_ids=("amount",),
                    ),
                ),
            ),
        )
    )
    yield PatternFactPlanTurnPrompt(
        fact_request,
        plan_selection=plan_selection,
    ).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context=context,
            memory_payload={},
        )
    )


def _selected_plan() -> PlanSelectionSet:
    return PlanSelectionSet(
        plan_selections=(
            SelectedSourceStrategy(
                plan_selection_id="plan.fact_1",
                requested_fact_id="fact_1",
                source_strategy_id="source_strategy.fact_1.direct_field_value.1",
                plan_shape="direct_field_value",
                required_answer_output_ids=("answer_1",),
                source_members=(SourceStrategyMember(source_candidate_id="source_1"),),
                basis="Selected by test fixture.",
            ),
        )
    )


def _question_contract() -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="Count of sales that happened today.",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer_1",
                        role="ANSWER_VALUE",
                        description="Count of sales",
                    ),
                ),
            ),
        )
    )


def _catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sale",),
                row_paths=(
                    RowPath(
                        id="root",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sale.id",
                        type="uuid",
                        path="id",
                        row_path_id="root",
                    ),
                ),
            ),
        )
    )


def _catalog_selection(catalog: RelationCatalog) -> CatalogSelectionResult:
    return CatalogSelectionResult(
        relation_catalog=catalog,
        selected_read_ids=("sales",),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales",),
                rankings=(
                    CatalogSelectionRanking(
                        read_id="sales",
                        score=10,
                        matched_terms=("sales",),
                    ),
                ),
                selected_read_ids=("sales",),
            ),
        ),
    )


def _serialized_provider_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, default=str, sort_keys=True)
