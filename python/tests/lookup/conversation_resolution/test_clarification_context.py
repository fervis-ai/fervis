from fervis.lookup.clarification.model import (
    ClarificationAnnotation,
    ClarificationResponseSource,
    ConversationResolutionResponse,
)
from fervis.lookup.conversation_resolution.model import ConversationResolutionRequest
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    compile_conversation_resolution,
    parse_conversation_resolution,
)
from fervis.lookup.conversation_resolution.prompt import (
    ConversationResolutionTurnPrompt,
    conversation_resolution_context_sources,
)
from fervis.memory.conversation_context import ConversationMemoryCardProjection


def test_grounding_prose_becomes_one_active_clarification_context_source() -> None:
    response = ConversationResolutionResponse(
        source=ClarificationResponseSource(
            response_id="response_1",
            clarification_id="clarification_1",
            exact_user_text="Area: Nairobi",
        ),
        annotation=ClarificationAnnotation(
            suspended_question_text="How many stores are in Nairobi?",
            clarification_question_text=(
                'I could not find place "Nairobi". Which place should I use?'
            ),
        ),
    )
    request = ConversationResolutionRequest(
        question="Area: Nairobi",
        conversation_context={},
        clarification_responses=(response,),
    )

    sources = conversation_resolution_context_sources(request)
    invocation = ConversationResolutionTurnPrompt(request).to_model_invocation()

    assert len(sources) == 1
    assert sources[0].kind == "active_clarification"
    assert sources[0].source_id == "active_clarification:response_1"
    assert tuple(anchor.text for anchor in sources[0].meaning_anchors) == (
        "How many stores are in Nairobi?",
        'I could not find place "Nairobi". Which place should I use?',
        "Area: Nairobi",
    )
    assert "active_clarification" in invocation.prompt_text
    assert (
        "An active_clarification context source contains the original question and "
        "every clarification question and answer in order."
    ) in invocation.prompt_text
    assert (
        "Later exchanges extend the established question context; they do not "
        "replace earlier exchanges."
    ) in invocation.prompt_text
    assert "active_clarification:response_1" in str(
        invocation.tool_specs[0].input_schema
    )
    schema = invocation.tool_specs[0].input_schema
    outcome_schema = schema["properties"]["outcome"]["oneOf"][0]
    source_variants = outcome_schema["properties"]["clauses"]["items"][
        "properties"
    ]["values"]["items"]["properties"]["sources"]["items"]["oneOf"]
    context_anchor_variants = tuple(
        variant
        for variant in source_variants
        if variant["properties"]["kind"].get("enum") == ["context_anchor"]
    )

    assert context_anchor_variants
    assert all(
        "source_text" not in variant["properties"]
        for variant in context_anchor_variants
    )
    schema_text = str(schema)
    assert 'I could not find place "Nairobi"' not in schema_text


def test_active_clarification_anchor_compiles_without_memory_activation() -> None:
    response = ConversationResolutionResponse(
        source=ClarificationResponseSource(
            response_id="response_1",
            clarification_id="clarification_1",
            exact_user_text="Area: Nairobi",
        ),
        annotation=ClarificationAnnotation(
            suspended_question_text="How many stores are in Nairobi?",
            clarification_question_text=(
                'I could not find place "Nairobi". Which place should I use?'
            ),
        ),
    )
    request = ConversationResolutionRequest(
        question="Area: Nairobi",
        conversation_context={},
        clarification_responses=(response,),
    )
    sources = conversation_resolution_context_sources(request)
    anchor = sources[0].meaning_anchors[1]
    payload = {
        "kind": "conversation_resolution",
        "current_question_text": "Area: Nairobi",
        "outcome": {
            "kind": "resolved",
            "resolution_basis": "The response supplies the requested area.",
            "contextualized_question": "How many stores are in Nairobi?",
            "clauses": [
                {
                    "current_clause_text": "Area: Nairobi",
                    "occurrence": 1,
                    "resolved_text": "How many stores are in Nairobi?",
                    "retained_frame_parts": [],
                    "values": [
                        {
                            "value_id": "area",
                            "resolved_text": "Nairobi",
                            "frame_parameter": {"kind": "none"},
                            "sources": [
                                {
                                    "kind": "current_span",
                                    "text": "Nairobi",
                                    "occurrence": 1,
                                },
                                {
                                    "kind": "context_anchor",
                                    "source_id": sources[0].source_id,
                                    "anchor_id": anchor.anchor_id,
                                },
                            ],
                        }
                    ],
                }
            ],
        },
    }

    resolution = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload=payload,
        current_question=request.question,
        context_sources=sources,
    ).outcome
    compiled = compile_conversation_resolution(
        resolution,
        memory_projection=ConversationMemoryCardProjection(
            context_sources=(),
            context_frames=(),
            private_cards={},
        ),
        context_sources=sources,
    )

    assert compiled.contextualized_question == "How many stores are in Nairobi?"
    assert compiled.used_memory_ids == ()
    prompt_payload = compiled.to_prompt_payload()

    assert prompt_payload["active_clarification"] == {
        "original_question": "How many stores are in Nairobi?",
        "exchanges": [
            {
                "response_id": "response_1",
                "clarification_questions": [
                    'I could not find place "Nairobi". Which place should I use?'
                ],
                "answer": "Area: Nairobi",
            }
        ],
    }
    assert "integrated_question" not in str(prompt_payload)
    assert compiled.clarification_lineage_refs == (
        "clarification_response:response_1",
    )


def test_consecutive_clarifications_compile_as_one_ordered_chain() -> None:
    first = ConversationResolutionResponse(
        source=ClarificationResponseSource(
            response_id="response_1",
            clarification_id="clarification_1",
            exact_user_text="Nairobi",
        ),
        annotation=ClarificationAnnotation(
            suspended_question_text="How many stores are there?",
            clarification_question_text="Which place should I use?",
        ),
    )
    second = ConversationResolutionResponse(
        source=ClarificationResponseSource(
            response_id="response_2",
            clarification_id="clarification_2",
            exact_user_text="Area",
        ),
        annotation=ClarificationAnnotation(
            suspended_question_text="Nairobi",
            clarification_question_text="Which kind of place should I use?",
        ),
    )
    request = ConversationResolutionRequest(
        question="Area",
        conversation_context={},
        clarification_responses=(first, second),
    )
    sources = conversation_resolution_context_sources(request)
    answer_anchor = sources[0].meaning_anchors[-1]
    payload = {
        "kind": "conversation_resolution",
        "current_question_text": "Area",
        "outcome": {
            "kind": "resolved",
            "resolution_basis": "The two responses identify the requested area.",
            "contextualized_question": "How many stores are in the Nairobi area?",
            "clauses": [
                {
                    "current_clause_text": "Area",
                    "occurrence": 1,
                    "resolved_text": "How many stores are in the Nairobi area?",
                    "retained_frame_parts": [],
                    "values": [
                        {
                            "value_id": "place_kind",
                            "resolved_text": "Area",
                            "frame_parameter": {"kind": "none"},
                            "sources": [
                                {
                                    "kind": "current_span",
                                    "text": "Area",
                                    "occurrence": 1,
                                },
                                {
                                    "kind": "context_anchor",
                                    "source_id": sources[0].source_id,
                                    "anchor_id": answer_anchor.anchor_id,
                                },
                            ],
                        }
                    ],
                }
            ],
        },
    }
    resolution = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload=payload,
        current_question=request.question,
        context_sources=sources,
    ).outcome
    compiled = compile_conversation_resolution(
        resolution,
        memory_projection=ConversationMemoryCardProjection(
            context_sources=(),
            context_frames=(),
            private_cards={},
        ),
        context_sources=sources,
    )

    assert compiled.to_prompt_payload()["active_clarification"] == {
        "original_question": "How many stores are there?",
        "exchanges": [
            {
                "response_id": "response_1",
                "clarification_questions": ["Which place should I use?"],
                "answer": "Nairobi",
            },
            {
                "response_id": "response_2",
                "clarification_questions": [
                    "Which kind of place should I use?"
                ],
                "answer": "Area",
            },
        ],
    }
    assert compiled.clarification_lineage_refs == (
        "clarification_response:response_1",
        "clarification_response:response_2",
    )
