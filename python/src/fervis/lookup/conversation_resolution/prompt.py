"""Prompt for conversation resolution."""

from __future__ import annotations

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
)
from fervis.lookup.conversation_resolution.clarifications import (
    active_clarification_contract,
)
from fervis.lookup.conversation_resolution.model import (
    ConversationResolutionRequest,
)
from fervis.lookup.conversation_resolution.schema import (
    build_conversation_resolution_tool_schemas,
)
from fervis.lookup.conversation_resolution.tools import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
)
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
    build_turn_prompt_context,
)
from fervis.model_io.structured_output.specs import required_tool_spec


class ConversationResolutionTurnPrompt(TurnPromptBase):
    turn_name = "conversation resolution"
    turn_task = "resolve how the current utterance depends on prior conversation context"

    def __init__(
        self,
        request: ConversationResolutionRequest | None = None,
        *,
        question: str = "",
        context_sources: tuple[ConversationContextSource, ...] = (),
        context_frames: tuple[ConversationContextFrame, ...] = (),
        conversation_context: dict[str, object] | None = None,
    ) -> None:
        self.request = request or ConversationResolutionRequest(
            question=question,
            conversation_context=dict(conversation_context or {}),
            context_sources=tuple(context_sources),
            context_frames=tuple(context_frames),
        )

    def data_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.json_section(
                "Context sources:",
                {
                    "current_question_text": self.request.question,
                    "context_sources": [
                        item.to_model_dict()
                        for item in conversation_resolution_context_sources(
                            self.request
                        )
                    ],
                },
                indent=2,
            ),
            builder.json_section(
                "Available context frames:",
                {
                    "available_context_frames": [
                        item.to_model_dict()
                        for item in conversation_resolution_context_frames(self.request)
                    ],
                },
                indent=2,
            ),
        )

    def instruction_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Task",
                (
                    "Return a conversation resolution result for the current user utterance.",
                    "If the current question can already be answered without prior turns, "
                    "set clause_resolutions=[] and unresolved.unresolved_kind=none.",
                    "If context is needed and sufficient, produce one "
                    "clause_resolution for each answerable clause and set "
                    "unresolved.unresolved_kind=none.",
                    "When unresolved.unresolved_kind=none, set "
                    'unresolved.why_unresolved="" and '
                    "unresolved.candidate_interpretations=[].",
                    "A standalone factual question can be answered without reading prior "
                    "turns, memory cards, or transcript context.",
                    "Use visible context sources only when the current question is not "
                    "self-sufficient.",
                    "An active_clarification context source is sufficient context when "
                    "it contains the prior question plus the user's clarification answer.",
                    "In that case, resolve the current utterance into the standalone "
                    "question answered by applying the clarification to the prior question.",
                ),
            ),
            builder.instruction_block(
                "Clause Resolution",
                (
                    "current_clause_text is exact text copied from the current user "
                    "question: one answerable part of the current question.",
                    "Each answerable clause has a requested_value_frame: the measure, "
                    "attribute, relation, or comparison whose value the clause asks for.",
                    "Copy current_value_surface.text from the current clause. Do not "
                    "paraphrase it.",
                    "If the current clause contains exact words that name the "
                    "requested value, current_value_surface.text must include "
                    "those words.",
                    "Question words alone are not enough when the current clause "
                    "also names the requested value.",
                    "Set current_value_surface.kind before choosing frames.",
                    "Use self_sufficient_current_value when current text already "
                    "identifies the requested value frame for this clause.",
                    "Use broad_current_value when current text asks for a value but "
                    "does not identify the value frame without context.",
                    "Use no_value_request only when the clause does not ask for a "
                    "value frame.",
                    "requested_value_frame.context_frame_choices must include one "
                    "choice item for each available_context_frames item.",
                    "For one clause, at most one context_frame_choices item may "
                    "have choice=use_frame.",
                    "If exactly one prior frame supplies the requested value frame, "
                    "mark that frame use_frame and mark every other frame "
                    "not_for_this_clause or current_text_names_different_value.",
                    "If multiple prior frames seem necessary for the requested "
                    "value frame, the clause is not resolved; use unresolved "
                    "with multiple_meanings and explain the competing frames.",
                    "Use use_frame when the current clause continues that context "
                    "frame as the requested value frame.",
                    "Use current_text_names_different_value only when the current "
                    "clause contains exact words that explicitly name a different "
                    "requested value frame. Put those exact words in "
                    "current_conflict_quotes.",
                    "Use not_for_this_clause when the clause asks for a different "
                    "kind of answer and the context frame is not a candidate for "
                    "that clause.",
                    "Use ambiguous when the frame could be the intended value frame "
                    "but visible context is insufficient to choose it.",
                    "Use dependencies only for context-dependent references or "
                    "scopes that are not replacement of a visible prior part.",
                    "Broad value wording is not conflict evidence. Broad value "
                    "wording cannot be used in current_conflict_quotes.",
                    "Do not create an unsupported domain-specific value frame from broad "
                    "current wording.",
                    "When current_value_surface.kind=broad_current_value, do not use "
                    "current_text_names_different_value.",
                    "For resolved clauses, do not use ambiguous. If a clause cannot "
                    "choose one value frame from current text and available context, "
                    "return unresolved with clause_resolutions=[].",
                ),
            ),
            builder.instruction_block(
                "Prior Question Continuation",
                (
                    "Use continuation only when the current clause is a follow-up to exactly one selected context frame, and the clause provides new text that replaces one or more visible replaceable_parts on that frame.",
                    "List only changed parts. Each replacement must copy a visible replaceable_parts[].part_id and exact current_text from current_clause_text.",
                    "Omitted replaceable_parts are carried by the backend.",
                    "Replacement current_text is current-turn evidence, not a dependency or meaning_component.",
                    "If the continued frame or replaced part is not clear, return unresolved.",
                    "When continuation is present, resolved_clause_text must state the compact resolved factual request after applying replacements.",
                ),
            ),
            builder.instruction_block(
                "Other Dependencies",
                (
                    "A dependency is an exact word or phrase in current_clause_text "
                    "whose standalone meaning comes from context and that "
                    "is not the requested value frame itself.",
                    "Dependencies include references and scopes.",
                    "A reference names an entity, row, set, or value indirectly.",
                    "A scope limits the question by time, place, grouping, row set, "
                    "or other boundary.",
                    "dependencies lists context-dependent references and scopes in "
                    "that clause. Do not duplicate the requested value frame as a "
                    "dependency. Use dependencies=[] when there are no such references "
                    "or scopes.",
                    "For each dependency, anchor_text is the exact current-question "
                    "text being resolved.",
                    "meaning_components lists the inherited pieces that make up "
                    "the meaning of anchor_text.",
                    "Each meaning component has source_text: exact text copied from "
                    "the cited source, and resolved_text: what that source text means "
                    "inside this dependency.",
                    "When source_text comes from a context source meaning_anchors "
                    "item, copy that item's memory_id exactly into the meaning "
                    "component.",
                    "Use one meaning component for each important inherited entity, "
                    "scope, row set, value, or other meaning piece.",
                    "Use kind=entity for a named person, place, organization, or "
                    "item; kind=scope for time, place, filter, grouping, or other "
                    "boundary; kind=row_set for prior returned rows; kind=value for "
                    "one prior scalar value; use kind=other only when none of those "
                    "fit.",
                    "If a current anchor like that or those carries a prior time, "
                    "put the time in meaning_components with kind=scope; do not create "
                    "a separate dependency whose anchor_text is absent from the "
                    "current clause.",
                    "For each dependency, resolved_text is the standalone "
                    "meaning that must be represented in "
                    "resolved_clause_text.",
                    "For each dependency, must_preserve_terms are exact words or "
                    "phrases from resolved_text that must appear verbatim in "
                    "resolved_clause_text when such terms are "
                    "necessary to prevent meaning loss. Use an empty array only when no "
                    "exact term is necessary.",
                    "Do not simplify or generalize must_preserve_terms.",
                ),
            ),
            builder.instruction_block(
                "Resolved Clauses",
                (
                    "resolved_clause_text rewrites current_clause_text as a standalone "
                    "clause for audit and downstream annotations.",
                    "It must include every required term from dependencies and "
                    "any selected context frame.",
                    "Each resolved_clause_text must state the factual meaning of "
                    "that clause without losing required terms.",
                    "Do not simplify or generalize a resolved measure, attribute, "
                    "relation, comparison, action, entity, row set, or scope.",
                    "For self-sufficient questions, clause_resolutions is empty because "
                    "current_question_text already carries its own meaning.",
                ),
            ),
            builder.instruction_block(
                "Needs Clarification",
                (
                    "Use unresolved only when the current question "
                    "plus visible context cannot determine one standalone factual "
                    "question.",
                    "When unresolved is used, set clause_resolutions=[].",
                    "If there are multiple plausible meanings, set "
                    "unresolved_kind=multiple_meanings and provide at least two "
                    "candidate_interpretations with complete integrated_question values.",
                    "If a required input is missing, set unresolved_kind=missing_input "
                    "and explain the missing input in why_unresolved.",
                    "Do not use needs_clarification when visible context is sufficient "
                    "to form a standalone question.",
                ),
            ),
            builder.instruction_block(
                "Boundaries",
                (
                    "Do not produce requested facts, endpoint IDs, field IDs, "
                    "source candidates, plans, or final answer text.",
                    "Conversation resolution only produces the standalone question "
                    "and clause-level context resolution that justifies it.",
                    "Later turns decide requested facts, grounding, source binding, "
                    "planning, and execution.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    f"Return exactly one {CONVERSATION_RESOLUTION_TOOL_NAME} tool call.",
                    "Return only valid JSON arguments for that tool call.",
                    "Always include kind, current_question_text, "
                    "clause_resolutions, and unresolved.",
                ),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        schemas = build_conversation_resolution_tool_schemas(
            context_sources=conversation_resolution_context_sources(self.request),
            context_frames=conversation_resolution_context_frames(self.request),
        )
        return ProviderResponseContract(provider_schema=schemas)

    def tool_contract(self) -> ProviderToolContract:
        schemas = build_conversation_resolution_tool_schemas(
            context_sources=conversation_resolution_context_sources(self.request),
            context_frames=conversation_resolution_context_frames(self.request),
        )
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
                    tool_description=(
                        "Submit a conversation-resolution decision with clause resolutions."
                    ),
                    input_schema=schemas[CONVERSATION_RESOLUTION_TOOL_NAME],
                ),
            )
        )

    def to_model_invocation(self, context=None):  # type: ignore[override]
        return super().to_model_invocation(
            context
            or build_turn_prompt_context(
                current_question=self.request.question,
                conversation_context=self.request.conversation_context,
                host=self.request.host,
            )
        )


def conversation_resolution_context_sources(
    request: ConversationResolutionRequest,
) -> tuple[ConversationContextSource, ...]:
    active_clarification = active_clarification_contract(
        request.conversation_context,
        current_question=request.question,
    )
    if active_clarification is None:
        return request.context_sources
    return (
        *request.context_sources,
        ConversationContextSource(
            source_id="active_clarification_1",
            kind="active_clarification",
            text=_active_clarification_text(active_clarification),
        ),
    )


def conversation_resolution_context_frames(
    request: ConversationResolutionRequest,
) -> tuple[ConversationContextFrame, ...]:
    return request.context_frames


def _active_clarification_text(active_clarification: dict[str, object]) -> str:
    exchanges = active_clarification.get("exchanges")
    if not isinstance(exchanges, list):
        return "Active clarification is available."
    parts: list[str] = []
    for exchange in exchanges:
        if not isinstance(exchange, dict):
            continue
        questions = exchange.get("clarification_questions")
        if isinstance(questions, list):
            parts.extend(str(item) for item in questions if str(item).strip())
        answer = str(exchange.get("answer") or "").strip()
        if answer:
            parts.append(answer)
    return " ".join(parts).strip() or "Active clarification is available."
