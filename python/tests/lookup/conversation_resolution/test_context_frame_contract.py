from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationMemoryCard,
    ConversationMemoryCardProjection,
    ConversationContextSource,
    ConversationMeaningAnchor,
    ConversationReplaceablePart,
)
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    ConversationResolutionKind,
    ConversationResolutionRequest,
    ConversationResolutionTurnPrompt,
    conversation_resolution_question_contract_context_texts,
    conversation_resolution_overlay_from,
    conversation_resolution_source_binding_prompt_payload,
    generate_conversation_resolution,
    parse_conversation_resolution,
)
from fervis.model_io.backbone.dto import ToolSpec, ProviderOutputMode
from fervis.lookup.memory.projection import project_conversation_memory_cards
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)
from fervis.memory.addresses import FactAddress


def _anchor(
    memory_id: str,
    text: str,
    *,
    kind: str = "other",
    label: str = "prior meaning",
) -> ConversationMeaningAnchor:
    return ConversationMeaningAnchor(
        memory_id=memory_id,
        text=text,
        occurrence=1,
        kind=kind,
        label=label,
    )


@dataclass
class _ConversationResolutionModelPort:
    arguments: dict[str, object]

    def generate(
        self,
        *,
        provider: str,
        system_prompt: str,
        prompt: str,
        max_thinking_tokens: int,
        output_mode: ProviderOutputMode,
        tool_specs: tuple[ToolSpec, ...],
    ) -> dict[str, object]:
        del (
            provider,
            system_prompt,
            prompt,
            max_thinking_tokens,
            output_mode,
            tool_specs,
        )
        return {
            "answer": json.dumps(
                {
                    "tool": CONVERSATION_RESOLUTION_TOOL_NAME,
                    "arguments": self.arguments,
                }
            ),
            "usage": {},
        }


def test_conversation_resolution_turn_artifact_carries_value_frame_projection():
    turn = generate_conversation_resolution(
        request=ConversationResolutionRequest(
            question="what about last month?",
            conversation_context={},
            context_sources=(
                ConversationContextSource(
                    source_id="prior_question_1",
                    kind="prior_user_question",
                    text="how many completed in-person sales this month?",
                ),
            ),
            context_frames=(
                ConversationContextFrame(
                    frame_id="context_frame_1",
                    source_ids=("prior_question_1",),
                    requested_frame="count of completed in-person sales",
                    prior_answer_fact="count of completed in-person sales",
                ),
            ),
        ),
        model_port=_ConversationResolutionModelPort(
            arguments={
                "kind": "conversation_resolution",
                "current_question_text": "what about last month?",
                "clause_resolutions": [
                    {
                        "current_clause_text": "what about last month?",
                        "occurrence": 1,
                        "requested_value_frame": {
                            "current_value_surface": {
                                "text": "what about last month?",
                                "kind": "broad_current_value",
                            },
                            "context_frame_choices": [
                                {
                                    "frame_id": "context_frame_1",
                                    "choice": "use_frame",
                                    "current_conflict_quotes": [],
                                }
                            ],
                        },
                        "dependencies": [],
                        "resolved_clause_text": (
                            "how many completed in-person sales last month?"
                        ),
                    }
                ],
                "unresolved": {
                    "unresolved_kind": "none",
                    "why_unresolved": "",
                    "candidate_interpretations": [],
                },
            }
        ),
        provider="fake",
        model_key="fake-model",
        max_thinking_tokens=0,
    )

    assert turn.artifact.derived_payload is not None
    assert turn.artifact.derived_payload["value_frames"][0]["resolved_frame_text"] == (
        "count of completed in-person sales"
    )


def test_conversation_resolution_dependency_separates_anchor_from_inherited_components():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": "How much is that in total sales?",
            "clause_resolutions": [
                {
                    "current_clause_text": "How much is that in total sales?",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "total sales",
                            "kind": "self_sufficient_current_value",
                        },
                        "context_frame_choices": [
                            {
                                "frame_id": "context_frame_1",
                                "choice": "not_for_this_clause",
                                "current_conflict_quotes": [],
                            },
                            {
                                "frame_id": "context_frame_2",
                                "choice": "use_frame",
                                "current_conflict_quotes": [],
                            },
                        ],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "that",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "entity",
                                    "source_id": "prior_1",
                                    "source_text": "Alice",
                                    "memory_id": "memory_entity",
                                    "resolved_text": "Alice",
                                },
                                {
                                    "kind": "scope",
                                    "source_id": "prior_1",
                                    "source_text": "today",
                                    "memory_id": "memory_today",
                                    "resolved_text": "today",
                                },
                                {
                                    "kind": "row_set",
                                    "source_id": "prior_2",
                                    "source_text": "Velvet Lip Gloss",
                                    "memory_id": "memory_rows",
                                    "resolved_text": "products sold",
                                },
                            ],
                            "resolved_text": "products Alice sold today",
                            "must_preserve_terms": ["products", "Alice", "today"],
                        }
                    ],
                    "resolved_clause_text": (
                        "How much is the total sales for products Alice sold today?"
                    ),
                }
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question="How much is that in total sales?",
        context_sources=(
            ConversationContextSource(
                source_id="prior_1",
                kind="prior_user_question",
                text="Which products did Alice sell today? Group them by sale.",
                meaning_anchors=(
                    _anchor(
                        "memory_entity",
                        "Alice",
                        kind="entity_identity",
                        label="staff identity",
                    ),
                    _anchor(
                        "memory_today", "today", kind="time_scope", label="time scope"
                    ),
                ),
            ),
            ConversationContextSource(
                source_id="prior_2",
                kind="prior_fervis_answer",
                text="Velvet Lip Gloss\nHydra Glow Serum",
                meaning_anchors=(
                    _anchor(
                        "memory_rows",
                        "Velvet Lip Gloss",
                        kind="row_set",
                        label="row set",
                    ),
                ),
            ),
        ),
        context_frames=(
            ConversationContextFrame(
                frame_id="context_frame_1",
                source_ids=("prior_1", "prior_2"),
                requested_frame="products",
                prior_answer_fact="products Alice sold today",
            ),
            ConversationContextFrame(
                frame_id="context_frame_2",
                source_ids=("prior_1", "prior_2"),
                requested_frame="sale",
                prior_answer_fact="products Alice sold today",
            ),
        ),
    )

    [dependency] = result.outcome.clause_resolutions[0].dependencies
    assert dependency.anchor_text == "that"
    assert [component.kind.value for component in dependency.meaning_components] == [
        "entity",
        "scope",
        "row_set",
    ]


def test_context_frame_contract_preserves_prior_requested_value_frame():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": (
                "And how much did she make yesterday? and where did she work?"
            ),
            "clause_resolutions": [
                {
                    "current_clause_text": "how much did she make yesterday",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "how much did she make",
                            "kind": "broad_current_value",
                        },
                        "context_frame_choices": [
                            {
                                "frame_id": "context_frame_1",
                                "choice": "use_frame",
                                "current_conflict_quotes": [],
                            },
                            {
                                "frame_id": "context_frame_2",
                                "choice": "not_for_this_clause",
                                "current_conflict_quotes": [],
                            },
                        ],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "she",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "other",
                                    "source_id": "prior_question_2",
                                    "source_text": "Alice",
                                    "memory_id": "memory_entity",
                                    "resolved_text": "Alice",
                                },
                            ],
                            "resolved_text": "Alice Smith",
                            "must_preserve_terms": ["Alice Smith"],
                        },
                    ],
                    "resolved_clause_text": (
                        "what was Alice Smith's total sales amount yesterday"
                    ),
                },
                {
                    "current_clause_text": "where did she work",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "where did she work",
                            "kind": "self_sufficient_current_value",
                        },
                        "context_frame_choices": [
                            {
                                "frame_id": "context_frame_1",
                                "choice": "not_for_this_clause",
                                "current_conflict_quotes": [],
                            },
                            {
                                "frame_id": "context_frame_2",
                                "choice": "not_for_this_clause",
                                "current_conflict_quotes": [],
                            },
                        ],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "she",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "other",
                                    "source_id": "prior_question_2",
                                    "source_text": "Alice",
                                    "memory_id": "memory_entity",
                                    "resolved_text": "Alice",
                                },
                            ],
                            "resolved_text": "Alice Smith",
                            "must_preserve_terms": ["Alice Smith"],
                        },
                    ],
                    "resolved_clause_text": "where did Alice Smith work",
                },
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question=(
            "And how much did she make yesterday? and where did she work?"
        ),
        context_sources=(
            ConversationContextSource(
                source_id="prior_question_1",
                kind="prior_user_question",
                text=(
                    "How much is that in total sales for products sold by "
                    "Alice today, grouped by sale?"
                ),
                source_card_ids=("card_prior_total_sales",),
                source_memory_ids=("mem_prior_total_sales",),
            ),
            ConversationContextSource(
                source_id="prior_question_2",
                kind="prior_user_question",
                text="Which products did Alice Smith sell today?",
                source_card_ids=("card_prior_products",),
                source_memory_ids=("mem_prior_products",),
                meaning_anchors=(
                    _anchor(
                        "memory_entity",
                        "Alice",
                        kind="entity_identity",
                        label="staff identity",
                    ),
                ),
            ),
        ),
        context_frames=(
            ConversationContextFrame(
                frame_id="context_frame_1",
                source_ids=("prior_question_1",),
                requested_frame="total sales amount",
                prior_answer_fact=(
                    "total sales for products sold by Alice today, grouped by sale"
                ),
            ),
            ConversationContextFrame(
                frame_id="context_frame_2",
                source_ids=("prior_question_2",),
                requested_frame="products sold",
                prior_answer_fact="products sold by Alice today, grouped by sale",
            ),
        ),
    )

    assert result.outcome.resolution is ConversationResolutionKind.RESOLVED
    assert result.outcome.used_source_card_ids == (
        "card_prior_total_sales",
        "card_prior_products",
    )
    assert result.outcome.used_memory_ids == (
        "mem_prior_total_sales",
        "mem_prior_products",
        "memory_entity",
    )


def test_conversation_resolution_projects_first_class_context_overlay():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": (
                "And how much did she make yesterday? and where did she work?"
            ),
            "clause_resolutions": [
                {
                    "current_clause_text": "how much did she make yesterday",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "how much did she make",
                            "kind": "broad_current_value",
                        },
                        "context_frame_choices": [
                            {
                                "frame_id": "context_frame_1",
                                "choice": "use_frame",
                                "current_conflict_quotes": [],
                            }
                        ],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "she",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "other",
                                    "source_id": "prior_question_1",
                                    "source_text": "Alice",
                                    "memory_id": "memory_entity",
                                    "resolved_text": "Alice",
                                },
                            ],
                            "resolved_text": "Alice Smith",
                            "must_preserve_terms": ["Alice Smith"],
                        },
                    ],
                    "resolved_clause_text": (
                        "what was Alice Smith's total sales amount yesterday"
                    ),
                },
                {
                    "current_clause_text": "where did she work",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "where did she work",
                            "kind": "self_sufficient_current_value",
                        },
                        "context_frame_choices": [
                            {
                                "frame_id": "context_frame_1",
                                "choice": "not_for_this_clause",
                                "current_conflict_quotes": [],
                            }
                        ],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "she",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "other",
                                    "source_id": "prior_question_1",
                                    "source_text": "Alice",
                                    "memory_id": "memory_entity",
                                    "resolved_text": "Alice",
                                },
                            ],
                            "resolved_text": "Alice Smith",
                            "must_preserve_terms": ["Alice Smith"],
                        },
                    ],
                    "resolved_clause_text": "where did Alice Smith work",
                },
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question=(
            "And how much did she make yesterday? and where did she work?"
        ),
        context_sources=(
            ConversationContextSource(
                source_id="prior_question_1",
                kind="prior_user_question",
                text="How much is that in total sales for Alice?",
                source_card_ids=("card_prior_total_sales",),
                source_memory_ids=("mem_prior_total_sales",),
                meaning_anchors=(
                    _anchor(
                        "memory_entity",
                        "Alice",
                        kind="entity_identity",
                        label="staff identity",
                    ),
                ),
            ),
        ),
        context_frames=(
            ConversationContextFrame(
                frame_id="context_frame_1",
                source_ids=("prior_question_1",),
                requested_frame="total sales amount",
                prior_answer_fact="total sales for prior rows",
            ),
        ),
    )

    overlay = conversation_resolution_overlay_from(result.outcome)
    payload = overlay.to_prompt_payload()

    assert payload == {
        "current_question": (
            "And how much did she make yesterday? and where did she work?"
        ),
        "value_frames": [
            {
                "current_clause_text": "how much did she make yesterday",
                "current_value_text": "how much did she make",
                "current_value_kind": "broad_current_value",
                "resolved_frame_text": "total sales amount",
                "must_preserve_terms": ["total sales amount"],
                "used_context_frame_ids": ["context_frame_1"],
            },
            {
                "current_clause_text": "where did she work",
                "current_value_text": "where did she work",
                "current_value_kind": "self_sufficient_current_value",
                "resolved_frame_text": "where did she work",
                "must_preserve_terms": [],
                "used_context_frame_ids": [],
            },
        ],
        "references": [
            {
                "current_clause_text": "how much did she make yesterday",
                "anchor_text": "she",
                "occurrence": 1,
                "resolved_text": "Alice Smith",
                "must_preserve_terms": ["Alice Smith"],
                "source_ids": ["prior_question_1"],
                "memory_ids": ["memory_entity"],
            },
            {
                "current_clause_text": "where did she work",
                "anchor_text": "she",
                "occurrence": 1,
                "resolved_text": "Alice Smith",
                "must_preserve_terms": ["Alice Smith"],
                "source_ids": ["prior_question_1"],
                "memory_ids": ["memory_entity"],
            },
        ],
        "scopes": [],
        "activated_memory_ids": ["mem_prior_total_sales", "memory_entity"],
        "used_source_card_ids": ["card_prior_total_sales"],
    }
    assert "integrated_question" not in payload
    assert "resolved_clause_text" not in str(payload)


def test_conversation_resolution_overlay_question_contract_context_uses_value_frames_not_dependency_text():
    overlay = conversation_resolution_overlay_from(
        parse_conversation_resolution(
            tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
            payload={
                "kind": "conversation_resolution",
                "current_question_text": "How much is that in total sales?",
                "clause_resolutions": [
                    {
                        "current_clause_text": "How much is that in total sales?",
                        "occurrence": 1,
                        "requested_value_frame": {
                            "current_value_surface": {
                                "text": "total sales",
                                "kind": "self_sufficient_current_value",
                            },
                            "context_frame_choices": [],
                        },
                        "dependencies": [
                            {
                                "anchor_text": "that",
                                "occurrence": 1,
                                "kind": "reference",
                                "meaning_components": [
                                    {
                                        "kind": "other",
                                        "source_id": "prior_1",
                                        "source_text": "Sale rows",
                                        "memory_id": "memory_rows",
                                        "resolved_text": "Sale rows",
                                    },
                                ],
                                "resolved_text": "sales listed in the prior answer",
                                "must_preserve_terms": ["sales"],
                            }
                        ],
                        "resolved_clause_text": (
                            "How much is the total sales for sales listed "
                            "in the prior answer?"
                        ),
                    }
                ],
                "unresolved": {
                    "unresolved_kind": "none",
                    "why_unresolved": "",
                    "candidate_interpretations": [],
                },
            },
            current_question="How much is that in total sales?",
            context_sources=(
                ConversationContextSource(
                    source_id="prior_1",
                    kind="prior_fervis_answer",
                    text="Sale rows",
                    meaning_anchors=(
                        _anchor(
                            "memory_rows", "Sale rows", kind="row_set", label="row set"
                        ),
                    ),
                ),
            ),
        ).outcome
    )

    context_texts = conversation_resolution_question_contract_context_texts(overlay)

    assert "total sales" in context_texts
    assert "sales listed in the prior answer" not in context_texts
    assert "Sale rows" not in context_texts


def test_conversation_resolution_source_binding_payload_exposes_stable_annotation_ids():
    overlay = conversation_resolution_overlay_from(
        parse_conversation_resolution(
            tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
            payload={
                "kind": "conversation_resolution",
                "current_question_text": "How many were open today?",
                "clause_resolutions": [
                    {
                        "current_clause_text": "How many were open today?",
                        "occurrence": 1,
                        "requested_value_frame": {
                            "current_value_surface": {
                                "text": "open today",
                                "kind": "broad_current_value",
                            },
                            "context_frame_choices": [
                                {
                                    "frame_id": "context_frame_1",
                                    "choice": "use_frame",
                                    "current_conflict_quotes": [],
                                }
                            ],
                        },
                        "dependencies": [],
                        "resolved_clause_text": "How many open sales today?",
                    }
                ],
                "unresolved": {
                    "unresolved_kind": "none",
                    "why_unresolved": "",
                    "candidate_interpretations": [],
                },
            },
            current_question="How many were open today?",
            context_sources=(
                ConversationContextSource(
                    source_id="prior_question_1",
                    kind="prior_user_question",
                    text="open sales today",
                ),
            ),
            context_frames=(
                ConversationContextFrame(
                    frame_id="context_frame_1",
                    source_ids=("prior_question_1",),
                    requested_frame="open sales today",
                    prior_answer_fact="open sales today",
                ),
            ),
        ).outcome
    )

    payload = conversation_resolution_source_binding_prompt_payload(overlay)

    assert payload["value_frames"][0]["annotation_id"] == "value_frame_1"
    assert payload["value_frames"][0]["resolved_frame_text"] == "open sales today"


def test_conversation_resolution_overlay_projects_entity_references_as_question_inputs():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": "And how much did she make for that yesterday?",
            "clause_resolutions": [
                {
                    "current_clause_text": "how much did she make for that yesterday",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "how much did she make",
                            "kind": "broad_current_value",
                        },
                        "context_frame_choices": [],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "she",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "other",
                                    "source_id": "prior_1",
                                    "source_text": "Alice",
                                    "memory_id": "memory_entity",
                                    "resolved_text": "Alice",
                                },
                            ],
                            "resolved_text": "Alice Smith",
                            "must_preserve_terms": ["Alice Smith"],
                        },
                        {
                            "anchor_text": "that",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "other",
                                    "source_id": "prior_2",
                                    "source_text": "sale rows",
                                    "memory_id": "memory_rows",
                                    "resolved_text": "sale rows",
                                },
                            ],
                            "resolved_text": "the sales items listed in the prior answer",
                            "must_preserve_terms": ["sales items"],
                        },
                    ],
                    "resolved_clause_text": (
                        "how much total sales did Alice Smith make for the "
                        "prior sale rows yesterday"
                    ),
                }
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question="And how much did she make for that yesterday?",
        context_sources=(
            ConversationContextSource(
                source_id="prior_1",
                kind="prior_fervis_answer",
                text="Alice",
                source_card_ids=("card_entity",),
                source_memory_ids=("memory_entity",),
                meaning_anchors=(
                    _anchor(
                        "memory_entity",
                        "Alice",
                        kind="entity_identity",
                        label="entity identity",
                    ),
                ),
            ),
            ConversationContextSource(
                source_id="prior_2",
                kind="prior_fervis_answer",
                text="sale rows",
                source_card_ids=("card_rows",),
                source_memory_ids=("memory_rows", "memory_entity"),
                meaning_anchors=(
                    _anchor(
                        "memory_rows", "sale rows", kind="row_set", label="row set"
                    ),
                ),
            ),
        ),
    ).outcome

    overlay = conversation_resolution_overlay_from(
        result,
        memory_projection=ConversationMemoryCardProjection(
            context_sources=(
                ConversationContextSource(
                    source_id="prior_1",
                    kind="prior_fervis_answer",
                    text="Alice",
                    source_card_ids=("card_entity",),
                    source_memory_ids=("memory_entity",),
                    meaning_anchors=(
                        _anchor(
                            "memory_entity",
                            "Alice",
                            kind="entity_identity",
                            label="entity identity",
                        ),
                    ),
                ),
                ConversationContextSource(
                    source_id="prior_2",
                    kind="prior_fervis_answer",
                    text="sale rows",
                    source_card_ids=("card_rows",),
                    source_memory_ids=("memory_rows", "memory_entity"),
                    meaning_anchors=(
                        _anchor(
                            "memory_rows", "sale rows", kind="row_set", label="row set"
                        ),
                    ),
                ),
            ),
            cards=(
                ConversationMemoryCard(
                    card_id="card_entity",
                    memory_id="memory_entity",
                    kind="entity_identity",
                    display="Alice Smith",
                ),
                ConversationMemoryCard(
                    card_id="card_rows",
                    memory_id="memory_rows",
                    kind="row_set",
                    display="sale rows",
                ),
            ),
            private_cards={
                "memory_entity": {
                    "kind": "entity_identity",
                    "identity_type": "staff",
                    "canonical_values": {"staff_id": "staff_alice"},
                    "proof_refs": ("prior_source_read:staff:alice",),
                    "display": "Alice Smith",
                }
            },
        ),
    )

    payload = overlay.to_prompt_payload()
    assert payload["resolved_question_inputs"] == [
        {
            "kind": "row_set_reference",
            "reference_text": "that",
            "occurrence": 1,
            "resolved_input_ref": "cr_input_1",
        },
        {
            "kind": "literal_text",
            "source_text": "she",
            "occurrence": 1,
            "resolved_input_ref": "cr_input_2",
            "resolved_value_text": "Alice Smith",
            "value_meaning_hint": "staff identity",
            "role": "reference_value",
        }
    ]


def test_conversation_resolution_derives_resolved_input_from_selected_memory_anchor():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": "How much did she make yesterday?",
            "clause_resolutions": [
                {
                    "current_clause_text": "How much did she make yesterday",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "How much did she make",
                            "kind": "broad_current_value",
                        },
                        "context_frame_choices": [],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "she",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "entity",
                                    "source_id": "prior_1",
                                    "source_text": "Alice",
                                    "memory_id": "turn_1.entity.staff.alice",
                                    "resolved_text": "Alice",
                                }
                            ],
                            "resolved_text": "Alice",
                            "must_preserve_terms": ["Alice"],
                        }
                    ],
                    "resolved_clause_text": "How much did Alice make yesterday?",
                }
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question="How much did she make yesterday?",
        context_sources=(
            ConversationContextSource(
                source_id="prior_1",
                kind="prior_user_question",
                text="Which products did Alice sell today?",
                meaning_anchors=(
                    ConversationMeaningAnchor(
                        memory_id="turn_1.entity.staff.alice",
                        text="Alice",
                        occurrence=1,
                        kind="entity_identity",
                        label="staff identity",
                    ),
                ),
            ),
        ),
    ).outcome

    overlay = conversation_resolution_overlay_from(
        result,
        memory_projection=ConversationMemoryCardProjection(
            context_sources=(
                ConversationContextSource(
                    source_id="prior_1",
                    kind="prior_user_question",
                    text="Which products did Alice sell today?",
                    meaning_anchors=(
                        ConversationMeaningAnchor(
                            memory_id="turn_1.entity.staff.alice",
                            text="Alice",
                            occurrence=1,
                            kind="entity_identity",
                            label="staff identity",
                        ),
                    ),
                ),
            ),
            cards=(
                ConversationMemoryCard(
                    card_id="card_entity",
                    memory_id="turn_1.entity.staff.alice",
                    kind="entity_identity",
                    display="Alice Smith",
                ),
            ),
            private_cards={
                "turn_1.entity.staff.alice": {
                    "kind": "entity_identity",
                    "identity_type": "staff",
                    "canonical_values": {
                        "staff_id": "51515151-0000-0000-0002-000000000001",
                    },
                    "proof_refs": ("prior_source_read:staff:alice",),
                    "display": "Alice Smith",
                }
            },
        ),
    )

    assert result.used_memory_ids == ("turn_1.entity.staff.alice",)
    assert overlay.to_prompt_payload()["resolved_question_inputs"] == [
        {
            "kind": "literal_text",
            "source_text": "she",
            "occurrence": 1,
            "resolved_input_ref": "cr_input_1",
            "resolved_value_text": "Alice Smith",
            "value_meaning_hint": "staff identity",
            "role": "reference_value",
        }
    ]


def test_conversation_resolution_handoff_uses_selected_memory_not_preserve_terms():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": "What about her for then?",
            "clause_resolutions": [
                {
                    "current_clause_text": "What about her for then",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "What about her",
                            "kind": "broad_current_value",
                        },
                        "context_frame_choices": [],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "her",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "entity",
                                    "source_id": "prior_entity",
                                    "source_text": "Alice",
                                    "memory_id": "turn_1.entity.staff.alice",
                                    "resolved_text": "Alice",
                                }
                            ],
                            "resolved_text": "Alice Smith",
                            "must_preserve_terms": ["Alice"],
                        },
                        {
                            "anchor_text": "then",
                            "occurrence": 1,
                            "kind": "scope",
                            "meaning_components": [
                                {
                                    "kind": "scope",
                                    "source_id": "prior_time",
                                    "source_text": "last month",
                                    "memory_id": "turn_1.scope.time.last_month",
                                    "resolved_text": "last month",
                                }
                            ],
                            "resolved_text": "last month",
                            "must_preserve_terms": ["month"],
                        },
                    ],
                    "resolved_clause_text": "What about Alice for last month?",
                }
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question="What about her for then?",
        context_sources=(
            ConversationContextSource(
                source_id="prior_entity",
                kind="prior_user_question",
                text="How much did Alice sell?",
                meaning_anchors=(
                    ConversationMeaningAnchor(
                        memory_id="turn_1.entity.staff.alice",
                        text="Alice",
                        occurrence=1,
                        kind="entity_identity",
                        label="staff identity",
                    ),
                ),
            ),
            ConversationContextSource(
                source_id="prior_time",
                kind="prior_user_question",
                text="How much did we sell last month?",
                meaning_anchors=(
                    ConversationMeaningAnchor(
                        memory_id="turn_1.scope.time.last_month",
                        text="last month",
                        occurrence=1,
                        kind="time_scope",
                        label="time scope",
                    ),
                ),
            ),
        ),
    ).outcome

    overlay = conversation_resolution_overlay_from(
        result,
        memory_projection=ConversationMemoryCardProjection(
            cards=(
                ConversationMemoryCard(
                    card_id="card_entity",
                    memory_id="turn_1.entity.staff.alice",
                    kind="entity_identity",
                    display="Alice Smith",
                ),
                ConversationMemoryCard(
                    card_id="card_time",
                    memory_id="turn_1.scope.time.last_month",
                    kind="time_scope",
                    display="last month",
                ),
            ),
            private_cards={
                "turn_1.entity.staff.alice": {
                    "kind": "entity_identity",
                    "identity_type": "staff",
                    "canonical_values": {"staff_id": "staff_alice"},
                    "proof_refs": ("prior_source_read:staff:alice",),
                    "display": "Alice Smith",
                },
                "turn_1.scope.time.last_month": {
                    "kind": "time_scope",
                    "expression": "last month",
                },
            },
        ),
    )

    assert overlay.to_prompt_payload()["resolved_question_inputs"] == [
        {
            "kind": "literal_text",
            "source_text": "her",
            "occurrence": 1,
            "resolved_input_ref": "cr_input_1",
            "resolved_value_text": "Alice Smith",
            "value_meaning_hint": "staff identity",
            "role": "reference_value",
        },
        {
            "kind": "literal_text",
            "source_text": "then",
            "occurrence": 1,
            "resolved_input_ref": "cr_input_2",
            "resolved_value_text": "last month",
            "value_meaning_hint": "time scope",
            "role": "time_value",
        },
    ]


def test_conversation_resolution_entity_handoff_is_text_only_without_authority():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": "How much did she make yesterday?",
            "clause_resolutions": [
                {
                    "current_clause_text": "How much did she make yesterday",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "How much did she make",
                            "kind": "broad_current_value",
                        },
                        "context_frame_choices": [],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "she",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "entity",
                                    "source_id": "prior_1",
                                    "source_text": "Alice",
                                    "memory_id": "turn_1.entity.staff.alice",
                                    "resolved_text": "Alice",
                                }
                            ],
                            "resolved_text": "Alice Smith",
                            "must_preserve_terms": ["Alice Smith"],
                        }
                    ],
                    "resolved_clause_text": "How much did Alice Smith make yesterday?",
                }
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question="How much did she make yesterday?",
        context_sources=(
            ConversationContextSource(
                source_id="prior_1",
                kind="prior_user_question",
                text="Which products did Alice sell today?",
                meaning_anchors=(
                    ConversationMeaningAnchor(
                        memory_id="turn_1.entity.staff.alice",
                        text="Alice",
                        occurrence=1,
                        kind="entity_identity",
                        label="staff identity",
                    ),
                ),
            ),
        ),
    ).outcome

    overlay = conversation_resolution_overlay_from(
        result,
        memory_projection=ConversationMemoryCardProjection(
            cards=(
                ConversationMemoryCard(
                    card_id="card_entity",
                    memory_id="turn_1.entity.staff.alice",
                    kind="entity_identity",
                    display="Alice Smith",
                ),
            ),
            private_cards={
                "turn_1.entity.staff.alice": {
                    "kind": "entity_identity",
                    "identity_type": "staff",
                    "canonical_values": {
                        "staff_id": "51515151-0000-0000-0002-000000000001",
                    },
                    "proof_refs": ("known_input:prior_staff",),
                    "display": "Alice Smith",
                }
            },
        ),
    )

    assert overlay.to_prompt_payload()["resolved_question_inputs"] == [
        {
            "kind": "literal_text",
            "source_text": "she",
            "occurrence": 1,
            "resolved_input_ref": "cr_input_1",
            "resolved_value_text": "Alice Smith",
            "value_meaning_hint": "staff identity",
            "role": "reference_value",
        }
    ]
    assert overlay.to_backend_payload()["resolved_question_inputs"] == [
        {
            "kind": "literal_text",
            "source_text": "she",
            "occurrence": 1,
            "resolved_input_ref": "cr_input_1",
            "resolved_value_text": "Alice Smith",
            "value_meaning_hint": "staff identity",
            "role": "reference_value",
            "evidence_refs": ["turn_1.entity.staff.alice"],
        }
    ]


def test_conversation_resolution_derives_time_value_input_from_scope_memory_anchor():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": "ABC Mall",
            "clause_resolutions": [
                {
                    "current_clause_text": "ABC Mall",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "ABC Mall",
                            "kind": "no_value_request",
                        },
                        "context_frame_choices": [],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "ABC Mall",
                            "occurrence": 1,
                            "kind": "scope",
                            "meaning_components": [
                                {
                                    "kind": "scope",
                                    "source_id": "prior_1",
                                    "source_text": "yesterday",
                                    "memory_id": "turn_1.scope.time.yesterday",
                                    "resolved_text": "yesterday",
                                }
                            ],
                            "resolved_text": "yesterday",
                            "must_preserve_terms": ["yesterday"],
                        }
                    ],
                    "resolved_clause_text": (
                        "How much sales did we make at ABC Mall yesterday?"
                    ),
                }
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question="ABC Mall",
        context_sources=(
            ConversationContextSource(
                source_id="prior_1",
                kind="prior_user_question",
                text="How much sales did we make yesterday?",
                meaning_anchors=(
                    ConversationMeaningAnchor(
                        memory_id="turn_1.scope.time.yesterday",
                        text="yesterday",
                        occurrence=1,
                        kind="time_scope",
                        label="time scope",
                    ),
                ),
            ),
        ),
    ).outcome

    overlay = conversation_resolution_overlay_from(
        result,
        memory_projection=ConversationMemoryCardProjection(
            context_sources=(),
            cards=(
                ConversationMemoryCard(
                    card_id="card_time",
                    memory_id="turn_1.scope.time.yesterday",
                    kind="time_scope",
                    display="yesterday",
                ),
            ),
            private_cards={
                "turn_1.scope.time.yesterday": {
                    "kind": "time_scope",
                    "expression": "yesterday",
                }
            },
        ),
    )

    assert overlay.to_prompt_payload()["resolved_question_inputs"] == [
        {
            "kind": "literal_text",
            "source_text": "ABC Mall",
            "occurrence": 1,
            "resolved_input_ref": "cr_input_1",
            "resolved_value_text": "yesterday",
            "value_meaning_hint": "time scope",
            "role": "time_value",
        }
    ]


def test_parser_accepts_contextual_frame_when_preserve_terms_are_kept():
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": "How much did she make yesterday?",
            "clause_resolutions": [
                {
                    "current_clause_text": "How much did she make yesterday?",
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": "How much did she make",
                            "kind": "broad_current_value",
                        },
                        "context_frame_choices": [
                            {
                                "frame_id": "context_frame_1",
                                "choice": "use_frame",
                                "current_conflict_quotes": [],
                            },
                        ],
                    },
                    "dependencies": [
                        {
                            "anchor_text": "she",
                            "occurrence": 1,
                            "kind": "reference",
                            "meaning_components": [
                                {
                                    "kind": "other",
                                    "source_id": "prior_question_1",
                                    "source_text": "Alice",
                                    "memory_id": "memory_entity",
                                    "resolved_text": "Alice",
                                },
                            ],
                            "resolved_text": "Alice Smith",
                            "must_preserve_terms": ["Alice Smith"],
                        },
                    ],
                    "resolved_clause_text": (
                        "What was the total monetary value of completed sale rows by "
                        "Alice Smith yesterday?"
                    ),
                },
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question="How much did she make yesterday?",
        context_sources=(
            ConversationContextSource(
                source_id="prior_question_1",
                kind="prior_user_question",
                text=(
                    "How much total monetary value of completed sale rows for Alice?"
                ),
                meaning_anchors=(
                    _anchor(
                        "memory_entity",
                        "Alice",
                        kind="entity_identity",
                        label="staff identity",
                    ),
                ),
            ),
        ),
        context_frames=(
            ConversationContextFrame(
                frame_id="context_frame_1",
                source_ids=("prior_question_1",),
                requested_frame="total monetary value of completed sale rows",
                prior_answer_fact="completed sale rows for Alice",
            ),
        ),
    )

    assert result.outcome.clause_resolutions[0].resolved_clause_text == (
        "What was the total monetary value of completed sale rows by "
        "Alice Smith yesterday?"
    )


def test_conversation_resolution_prompt_exposes_available_context_frames():
    prompt = ConversationResolutionTurnPrompt(
        question="And how much did she make yesterday?",
        conversation_context={},
        context_sources=(
            ConversationContextSource(
                source_id="prior_question_1",
                kind="prior_user_question",
                text="How much is that in total sales?",
            ),
        ),
        context_frames=(
            ConversationContextFrame(
                frame_id="context_frame_1",
                source_ids=("prior_question_1",),
                requested_frame="total sales amount",
                prior_answer_fact="total sales for prior rows",
            ),
        ),
    )

    invocation = prompt.to_model_invocation()

    assert "Available context frames:" in invocation.prompt_text
    assert '"prior_answer_fact": "total sales for prior rows"' in invocation.prompt_text
    assert "requested_value_frame.context_frame_choices" in invocation.prompt_text
    assert "status=standalone" not in invocation.prompt_text
    assert "status=resolved" not in invocation.prompt_text
    assert "status=needs_clarification" not in invocation.prompt_text
    assert 'unresolved.why_unresolved=""' in invocation.prompt_text
    assert "unresolved.candidate_interpretations=[]" in invocation.prompt_text
    schema = invocation.provider_schema[CONVERSATION_RESOLUTION_TOOL_NAME]
    assert "status" not in schema["properties"]
    assert "status" not in schema["required"]
    clause_schema = schema["properties"]["clause_resolutions"]["items"]
    frame_schema = clause_schema["properties"]["requested_value_frame"]
    assert "literal_frame_status" not in frame_schema["properties"]
    assert "literal_frame_status" not in frame_schema["required"]
    assert "literal_frame" not in frame_schema["properties"]
    assert "resolved_frame_text" not in frame_schema["properties"]
    choices = frame_schema["properties"]["context_frame_choices"]
    assert choices["items"]["properties"]["frame_id"]["enum"] == ["context_frame_1"]
    assert "contains" not in choices
    assert "minContains" not in choices
    assert "maxContains" not in choices


def _sales_count_time_context_source() -> ConversationContextSource:
    return ConversationContextSource(
        source_id="prior_1",
        kind="prior_user_question",
        text="How many sales did we make on July 9th?",
    )


def _sales_count_time_context_frame() -> ConversationContextFrame:
    return ConversationContextFrame(
        frame_id="context_frame_1",
        source_ids=("prior_1",),
        requested_frame="count",
        prior_answer_fact="sales count on July 9th",
        replaceable_parts=(
            ConversationReplaceablePart(
                part_id="q_time",
                kind="time_scope",
                text="July 9th",
            ),
        ),
    )


def _sales_count_time_continuation_payload(
    *,
    part_id: str = "q_time",
    current_text: str = "July 8th",
    replacement_text: str = "July 8th",
) -> dict[str, object]:
    return {
        "kind": "conversation_resolution",
        "current_question_text": "sorry, meant July 8th",
        "clause_resolutions": [
            {
                "current_clause_text": "sorry, meant July 8th",
                "occurrence": 1,
                "requested_value_frame": {
                    "current_value_surface": {
                        "text": current_text,
                        "kind": "broad_current_value",
                    },
                    "context_frame_choices": [
                        {
                            "frame_id": "context_frame_1",
                            "choice": "use_frame",
                            "current_conflict_quotes": [],
                        }
                    ],
                },
                "continuation": {
                    "kind": "continue_prior_question",
                    "frame_id": "context_frame_1",
                    "replacements": [
                        {
                            "part_id": part_id,
                            "current_text": replacement_text,
                        }
                    ],
                },
                "dependencies": [],
                "resolved_clause_text": "How many sales did we make on July 8th?",
            }
        ],
        "unresolved": {
            "unresolved_kind": "none",
            "why_unresolved": "",
            "candidate_interpretations": [],
        },
    }


def _parse_sales_count_time_continuation(
    payload: dict[str, object] | None = None,
):
    return parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload=payload or _sales_count_time_continuation_payload(),
        current_question="sorry, meant July 8th",
        context_frames=(_sales_count_time_context_frame(),),
        context_sources=(_sales_count_time_context_source(),),
    )


def test_schema_exposes_continuation_with_bounded_replaceable_part_ids():
    prompt = ConversationResolutionTurnPrompt(
        question="sorry, meant July 8th",
        conversation_context={},
        context_sources=(_sales_count_time_context_source(),),
        context_frames=(_sales_count_time_context_frame(),),
    )

    schema = prompt.response_contract().provider_schema[
        CONVERSATION_RESOLUTION_TOOL_NAME
    ]
    clause_schema = schema["properties"]["clause_resolutions"]["items"]
    continuation_schema = clause_schema["properties"]["continuation"]
    replacement_schema = continuation_schema["properties"]["replacements"]["items"]

    assert "continuation" not in clause_schema["required"]
    assert continuation_schema["properties"]["kind"]["enum"] == [
        "continue_prior_question"
    ]
    assert continuation_schema["properties"]["frame_id"]["enum"] == [
        "context_frame_1"
    ]
    assert replacement_schema["properties"]["part_id"]["enum"] == ["q_time"]


def test_prior_answer_outputs_project_to_context_frames_without_backend_semantics():
    artifact = build_fact_artifact(
        artifact_id="turn_sales_total",
        outcome=FactOutcome.ANSWERED,
        source_question="How much is that in total sales?",
        source_answer="2468",
        provenance={
            "question_contract": {
                "answer_requests": [
                    {
                        "id": "fact_1",
                        "answer_fact": "total sales for products sold by Alice",
                        "answer_outputs": [
                            {
                                "id": "answer_output_1",
                                "description": "total sales amount",
                            }
                        ],
                        "used_question_inputs": [],
                    }
                ],
                "question_inputs": [],
            }
        },
        addresses=(
            FactAddress.value(
                address="value.total_sales",
                value={"type": "decimal", "value": "2468.00"},
            ),
        ),
    )

    projection = project_conversation_memory_cards(
        {"factArtifacts": [artifact.to_dict()]},
        current_question="And how much did she make yesterday?",
    )

    assert projection.context_frames
    frame = projection.context_frames[0]
    assert frame.requested_frame == "total sales amount"
    assert frame.prior_answer_fact == "total sales for products sold by Alice"
    assert frame.source_ids


def test_prior_answer_request_context_frame_exposes_replaceable_parts_from_contract():
    artifact = build_fact_artifact(
        artifact_id="turn_sales_count",
        outcome=FactOutcome.ANSWERED,
        source_question="How many sales did we make on July 9th?",
        source_answer="8",
        provenance={
            "question_contract": {
                "question_inputs": [
                    {
                        "id": "q_time",
                        "kind": "literal_text",
                        "role": "time_value",
                        "text": "July 9th",
                        "resolved_value_text": "July 9th",
                    }
                ],
                "answer_requests": [
                    {
                        "id": "fact_1",
                        "answer_fact": "sales count on July 9th",
                        "answer_subject": {
                            "subject_text": "sales",
                        },
                        "answer_outputs": [
                            {
                                "id": "answer_1",
                                "description": "sales count",
                            }
                        ],
                        "used_question_inputs": ["q_time"],
                    }
                ],
            }
        },
        addresses=(
            FactAddress.value(
                address="value.answer_1",
                value={"type": "integer", "value": 8},
            ),
        ),
    )

    projection = project_conversation_memory_cards(
        {"factArtifacts": [artifact.to_dict()]},
        current_question="sorry, meant July 8th",
    )

    frame = projection.context_frames[0]
    assert [part.to_model_dict() for part in frame.replaceable_parts] == [
        {
            "part_id": "answer_subject",
            "kind": "answer_subject",
            "text": "sales",
        },
        {
            "part_id": "q_time",
            "kind": "time_scope",
            "text": "July 9th",
        },
    ]


def test_parser_accepts_prior_question_continuation_replacement():
    result = _parse_sales_count_time_continuation()

    continuation = result.outcome.clause_resolutions[0].continuation
    assert continuation is not None
    assert continuation.frame_id == "context_frame_1"
    assert [
        replacement.to_model_dict() for replacement in continuation.replacements
    ] == [
        {
            "part_id": "q_time",
            "current_text": "July 8th",
        }
    ]


def test_parser_rejects_unknown_continuation_replacement_part():
    with pytest.raises(ValueError, match="part_id is not replaceable on frame"):
        _parse_sales_count_time_continuation(
            _sales_count_time_continuation_payload(part_id="q_missing")
        )


def test_parser_rejects_continuation_text_not_copied_from_clause():
    with pytest.raises(ValueError, match="current_text does not appear"):
        _parse_sales_count_time_continuation(
            _sales_count_time_continuation_payload(replacement_text="July 10th")
        )


def test_prior_answer_outputs_project_unique_context_frames_for_same_prior_frame():
    first = _sale_ids_artifact("first_turn_sale_ids")
    second = _sale_ids_artifact("second_turn_sale_ids")

    projection = project_conversation_memory_cards(
        {"factArtifacts": [first.to_dict(), second.to_dict()]},
        current_question="What products are included in those sales?",
    )

    frames = projection.context_frames
    assert [
        (frame.requested_frame, frame.source_ids, frame.prior_answer_fact)
        for frame in frames
    ] == [
        (
            "sale IDs",
            ("prior_1", "prior_2"),
            "sale IDs for Amani's sales yesterday",
        )
    ]


def _sale_ids_artifact(artifact_id: str):
    return build_fact_artifact(
        artifact_id=artifact_id,
        outcome=FactOutcome.ANSWERED,
        source_question="List sale IDs of Amani's yesterday sales.",
        source_answer="sale-1\nsale-2",
        provenance={
            "question_contract": {
                "answer_requests": [
                    {
                        "id": "fact_1",
                        "answer_fact": "sale IDs for Amani's sales yesterday",
                        "answer_outputs": [
                            {
                                "id": "answer_1",
                                "description": "sale IDs",
                            }
                        ],
                        "used_question_inputs": [],
                    }
                ],
                "question_inputs": [],
            }
        },
        addresses=(
            FactAddress.relation(
                address="relation.answer_1_rows",
                source={
                    "kind": "operation_output",
                    "relationId": "answer_1_rows",
                },
                grain_keys=("sale_id",),
                field_coverage={"sale_id": "answer_1_rows.sale_id"},
                completeness={"status": "complete", "rowCount": 2},
                row_addresses=(
                    "row.answer_1_rows.1",
                    "row.answer_1_rows.2",
                ),
            ),
            FactAddress.row(
                address="row.answer_1_rows.1",
                relation="relation.answer_1_rows",
                grain={"sale_id": "sale-1"},
                values={
                    "sale_id": {
                        "type": "uuid",
                        "value": "sale-1",
                        "answer_output_ids": ["answer_1"],
                    }
                },
            ),
            FactAddress.row(
                address="row.answer_1_rows.2",
                relation="relation.answer_1_rows",
                grain={"sale_id": "sale-2"},
                values={
                    "sale_id": {
                        "type": "uuid",
                        "value": "sale-2",
                        "answer_output_ids": ["answer_1"],
                    }
                },
            ),
        ),
    )
