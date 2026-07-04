import json

from fervis.lookup.relation_catalog import (
    CatalogField,
    EndpointRead,
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
    KnownInputBindingTask,
)
from fervis.lookup.grounding.prompt import GroundingTurnPrompt
from fervis.lookup.fact_plan.relations import (
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
    SourceBindingRequest,
    SourceBindingTurnPrompt,
)
from fervis.lookup.plan_selection import (
    SourceStrategyMember,
    BoundSourceStrategyMember,
    BoundSelectedSourceStrategy,
    BoundPlanSelectionSet,
    PlanSelectionSet,
    SelectedSourceStrategy,
)
from fervis.memory.addresses import FactAddress
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)


_APPROVED_CHARS = {
    "question contract": (364, 12139, 21044),
    "query enrichment": (364, 5985, 8214),
    "grounding": (364, 4949, 6766),
    "source binding": (364, 13384, 20765),
    "pattern fact planning": (364, 3363, 6093),
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


def test_model_turn_invocations_render_expected_shared_frame():
    for invocation in _turn_invocations():
        assert invocation.prompt_text.startswith("Current question:\n")
        assert f"We are currently on the {invocation.turn_name} step." in (
            invocation.prompt_text
        )


def test_active_clarification_context_is_question_contract_only():
    question = "ABC Mall"
    clarification = build_fact_artifact(
        artifact_id="turn_clarification",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        source_question="How much did we make yesterday?",
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_clarification",
                terminal="needs_clarification",
                clarification_questions=("Which store?",),
            ),
        ),
    )
    context = {"factArtifacts": [clarification.to_dict()]}
    contract = _question_contract()
    catalog = _catalog()

    question_prompt = QuestionContractTurnPrompt(
        QuestionContractRequest(
            current_question=question,
            conversation_context=context,
        )
    ).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context=context,
        )
    )
    query_prompt = QueryEnrichmentTurnPrompt(
        QueryEnrichmentRequest(
            question=question,
            conversation_context=context,
            requested_facts=contract.requested_facts,
            relation_catalog=catalog,
        )
    ).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context=context,
        )
    )

    assert "Active clarification context:" in question_prompt.prompt_text
    assert "Which store?" in question_prompt.prompt_text
    assert "Active clarification context:" not in query_prompt.prompt_text
    assert "Which store?" not in query_prompt.prompt_text


def test_planning_request_does_not_accept_conversation_overlay():
    assert "conversation_resolution_overlay" not in FactPlanRequest.__dataclass_fields__


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

    grounding_request = GroundingRequest(
        question=question,
        conversation_context=context,
        tasks=(
            KnownInputBindingTask(
                known_input_id="input_today",
                known_input_text="today",
                known_input_kind="time_text",
                requested_fact_id="fact_1",
                options=(
                    InputBindingOption(
                        id="bind_input_today_1",
                        known_input_id="input_today",
                        path="placeholder resolver",
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

    source_request = SourceBindingRequest(
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
                        source_binding_ids=("source_fact_1",),
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
