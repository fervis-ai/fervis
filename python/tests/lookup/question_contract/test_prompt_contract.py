from __future__ import annotations

import json

from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.conversation_resolution import (
    ConversationDependencyOverlay,
    ConversationResolutionOverlay,
    ConversationValueFrameOverlay,
    LiteralQuestionInputOverlay,
    conversation_resolution_question_contract_prompt_payload,
)
from fervis.lookup.question_contract import QuestionContractRequest
from fervis.lookup.question_contract.prompt import QuestionContractTurnPrompt
from fervis.lookup.question_contract.tools import (
    ANSWER_REQUEST_CONTRACT_TOOL_NAME,
)
from fervis.lookup.question_contract.turn import generate_question_contract
from fervis.lookup.question_inputs import LiteralInputRole
from fervis.memory.addresses import FactAddress
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)


def test_question_contract_prompt_uses_raw_question_with_conversation_overlay():
    assert "integrated_question" not in QuestionContractRequest.__dataclass_fields__
    overlay = ConversationResolutionOverlay(
        current_question="And how much did she make yesterday? and where did she work?",
        value_frames=(
            ConversationValueFrameOverlay(
                current_clause_text="how much did she make yesterday",
                current_value_text="how much did she make",
                current_value_kind="broad_current_value",
                resolved_frame_text="total sales amount",
                must_preserve_terms=("total sales amount",),
                used_context_frame_ids=("context_frame_1",),
            ),
        ),
        references=(
            ConversationDependencyOverlay(
                current_clause_text="how much did she make yesterday",
                anchor_text="she",
                occurrence=1,
                resolved_text="Alice Smith",
                must_preserve_terms=("Alice Smith",),
                source_ids=("prior_question_1",),
            ),
        ),
        scopes=(),
        activated_memory_ids=("mem_prior_total_sales",),
        used_source_card_ids=("card_prior_total_sales",),
        resolved_question_inputs=(
            LiteralQuestionInputOverlay(
                source_text="she",
                occurrence=1,
                resolved_input_ref="cr_input_1",
                resolved_value_text="Alice Smith",
                value_meaning_hint="staff identity",
                role=LiteralInputRole.REFERENCE_VALUE,
            ),
        ),
    )
    request = QuestionContractRequest(
        current_question=(
            "And how much did she make yesterday? and where did she work?"
        ),
        conversation_resolution_overlay=overlay,
        conversation_context={},
    )

    prompt = (
        QuestionContractTurnPrompt(request)
        .to_model_invocation(
            build_turn_prompt_context(
                current_question=request.current_question,
                conversation_context=request.conversation_context,
                conversation_resolution_overlay=conversation_resolution_question_contract_prompt_payload(
                    overlay
                ),
            )
        )
        .prompt_text
    )

    assert "Current question:" in prompt
    assert "And how much did she make yesterday? and where did she work?" in prompt
    assert "Conversation resolution annotations:" in prompt
    assert "resolved_question_inputs" in prompt
    assert "literal_text" in prompt
    assert "Alice Smith" in prompt
    assert "question_context" in prompt
    assert '"value_frames"' in prompt
    assert '"resolved_frame_text"' in prompt
    assert '"must_preserve_terms"' in prompt
    assert "total sales amount" in prompt
    assert (
        "When a time input constrains an answer_request, include its input_ref "
        "in that answer_request's used_question_inputs."
    ) in prompt
    assert "integrated_question" not in prompt
    assert "Integrated question:" not in prompt
    assert "Activated memory:" not in prompt
    assert "memory.entity.alice" not in prompt
    assert '"references"' not in prompt
    assert '"scopes"' not in prompt


def test_question_contract_turn_does_not_recompose_resolved_active_clarification():
    overlay = ConversationResolutionOverlay(
        current_question="ABC Mall",
        value_frames=(
            ConversationValueFrameOverlay(
                current_clause_text="ABC Mall",
                current_value_text="ABC Mall",
                current_value_kind="self_sufficient_current_value",
                resolved_frame_text="money made at ABC Mall yesterday",
                must_preserve_terms=("ABC Mall", "yesterday"),
                used_context_frame_ids=(),
            ),
        ),
        references=(),
        scopes=(),
        activated_memory_ids=(),
        used_source_card_ids=(),
    )
    clarification_artifact = build_fact_artifact(
        artifact_id="turn_clarification",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        source_question="How much money did we make yesterday?",
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_clarification",
                terminal="needs_clarification",
                clarification_questions=("Which location?",),
            ),
        ),
    )
    conversation_context = {"factArtifacts": [clarification_artifact.to_dict()]}
    model_port = _CapturingQuestionContractModelPort()

    generate_question_contract(
        request=QuestionContractRequest(
            current_question="ABC Mall",
            conversation_context=conversation_context,
            conversation_resolution_overlay=overlay,
        ),
        model_port=model_port,
        provider="fake",
        model_key="fake",
        max_thinking_tokens=0,
    )

    assert "Integrated question:" not in model_port.prompt
    assert "Current question:\nABC Mall" in model_port.prompt
    assert "Conversation resolution annotations:" in model_port.prompt
    assert "money made at ABC Mall yesterday" in model_port.prompt
    assert "How much money did we make yesterday?" in model_port.prompt


def test_question_contract_prompt_excludes_dependency_resolved_text():
    overlay = ConversationResolutionOverlay(
        current_question="How much is that in total sales?",
        value_frames=(
            ConversationValueFrameOverlay(
                current_clause_text="How much is that in total sales?",
                current_value_text="How much is that in total sales",
                current_value_kind="broad_current_value",
                resolved_frame_text="total sales",
                must_preserve_terms=(),
                used_context_frame_ids=(),
            ),
        ),
        references=(
            ConversationDependencyOverlay(
                current_clause_text="How much is that in total sales?",
                anchor_text="that",
                occurrence=1,
                resolved_text="the sales items listed in the prior answer",
                must_preserve_terms=("sales items",),
                source_ids=("prior_2",),
            ),
        ),
        scopes=(),
        activated_memory_ids=(),
        used_source_card_ids=(),
    )
    request = QuestionContractRequest(
        current_question="How much is that in total sales?",
        conversation_resolution_overlay=overlay,
        conversation_context={},
    )

    prompt = (
        QuestionContractTurnPrompt(request)
        .to_model_invocation(
            build_turn_prompt_context(
                current_question=request.current_question,
                conversation_context=request.conversation_context,
                conversation_resolution_overlay=conversation_resolution_question_contract_prompt_payload(
                    overlay
                ),
            )
        )
        .prompt_text
    )

    assert "total sales" in prompt
    assert "the sales items listed in the prior answer" not in prompt
    assert "sales items" not in prompt
    assert '"resolved_question_inputs"' not in prompt


def test_question_contract_turn_accepts_subject_from_conversation_value_frame():
    overlay = ConversationResolutionOverlay(
        current_question="what about last month?",
        value_frames=(
            ConversationValueFrameOverlay(
                current_clause_text="what about last month?",
                current_value_text="what about last month?",
                current_value_kind="broad_current_value",
                resolved_frame_text="count of completed in-person sales",
                must_preserve_terms=("completed", "in-person", "sales"),
                used_context_frame_ids=("context_frame_1",),
            ),
        ),
        references=(),
        scopes=(),
        activated_memory_ids=("mem_prior_sales_count",),
        used_source_card_ids=("card_prior_sales_count",),
    )

    result = generate_question_contract(
        request=QuestionContractRequest(
            current_question="what about last month?",
            conversation_context={},
            conversation_resolution_overlay=overlay,
        ),
        model_port=_FollowUpQuestionContractModelPort(),
        provider="fake",
        model_key="fake",
        max_thinking_tokens=0,
    )

    fact = result.result.outcome.requested_facts[0]
    assert fact.description == "Count of completed in-person sales for last month"
    assert fact.answer_subject is not None
    assert fact.answer_subject.subject_text == "sales"


class _CapturingQuestionContractModelPort:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode=None,
        tool_specs=(),
    ):
        del provider, max_thinking_tokens, system_prompt, output_mode, tool_specs
        self.prompt = prompt
        return {
            "answer": json.dumps(
                {
                    "tool": ANSWER_REQUEST_CONTRACT_TOOL_NAME,
                    "arguments": {
                        "kind": "question_contract",
                        "answer_requests_count": 1,
                        "question_inputs": [],
                        "answer_requests": [
                            {
                                "answer_fact": "money made at ABC Mall yesterday",
                                "answer_expression": {"family": "scalar_aggregate"},
                                "answer_subject": {
                                    "subject_text": "money",
                                    "instance_interpretation": {
                                        "kind": "NORMAL_BUSINESS_INSTANCE"
                                    },
                                },
                                "answer_population": {
                                    "population_label": "money made at ABC Mall yesterday",
                                    "counted_unit": "money",
                                    "membership_tests": [
                                        {
                                            "test_id": "pop_test_1",
                                            "kind": "SUBJECT_IDENTITY",
                                            "polarity": "MUST_PASS",
                                            "test_question": (
                                                "Does the row/value represent money?"
                                            ),
                                            "owned_question_input_refs": [],
                                        }
                                    ],
                                },
                                "answer_outputs": [
                                    {
                                        "description": "total money",
                                    }
                                ],
                                "used_question_inputs": [],
                            }
                        ],
                        "question_input_inventory_check": {
                            "all_input_like_phrases_declared": True,
                        },
                    },
                }
            ),
            "usage": {},
        }


class _FollowUpQuestionContractModelPort:
    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode=None,
        tool_specs=(),
    ):
        del (
            provider,
            prompt,
            max_thinking_tokens,
            system_prompt,
            output_mode,
            tool_specs,
        )
        return {
            "answer": json.dumps(
                {
                    "tool": ANSWER_REQUEST_CONTRACT_TOOL_NAME,
                    "arguments": {
                        "kind": "question_contract",
                        "answer_requests_count": 1,
                        "question_inputs": [
                            {
                                "source": "question_context",
                                "kind": "literal_text",
                                "input_ref": "time_1",
                                "value_source_text": "last month",
                                "resolved_value_text": "last month",
                                "role": "time_value",
                                "inventory_check": {
                                    "why_this_is_an_input": (
                                        "This calendar period constrains the count."
                                    )
                                },
                            }
                        ],
                        "answer_requests": [
                            {
                                "answer_fact": (
                                    "Count of completed in-person sales for last month"
                                ),
                                "answer_expression": {"family": "scalar_aggregate"},
                                "answer_subject": {
                                    "subject_text": "sales",
                                    "instance_interpretation": {
                                        "kind": "NORMAL_BUSINESS_INSTANCE"
                                    },
                                },
                                "answer_population": {
                                    "population_label": (
                                        "completed in-person sales last month"
                                    ),
                                    "counted_unit": "sale",
                                    "membership_tests": [
                                        {
                                            "test_id": "test_1",
                                            "kind": "SUBJECT_IDENTITY",
                                            "polarity": "MUST_PASS",
                                            "test_question": (
                                                "Is the instance a sale?"
                                            ),
                                            "owned_question_input_refs": [],
                                        },
                                        {
                                            "test_id": "test_2",
                                            "kind": "EXPLICIT_USER_CONSTRAINT",
                                            "polarity": "MUST_PASS",
                                            "test_question": ("Is the sale completed?"),
                                            "owned_question_input_refs": [],
                                        },
                                        {
                                            "test_id": "test_3",
                                            "kind": "EXPLICIT_USER_CONSTRAINT",
                                            "polarity": "MUST_PASS",
                                            "test_question": (
                                                "Was the sale in-person?"
                                            ),
                                            "owned_question_input_refs": [],
                                        },
                                    ],
                                },
                                "answer_outputs": [
                                    {
                                        "description": (
                                            "The count of completed in-person sales"
                                        )
                                    }
                                ],
                                "used_question_inputs": ["time_1"],
                            }
                        ],
                        "question_input_inventory_check": {
                            "all_input_like_phrases_declared": True,
                        },
                    },
                }
            ),
            "usage": {},
        }
