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
                    "Produce the complete standalone factual question the user means.",
                    "Resolve every answerable clause completely. Do not describe how "
                    "the wording changed.",
                    "Treat meaning stated explicitly in the current question as "
                    "authoritative. Prior context fills what the current wording leaves "
                    "implicit; it does not overwrite explicit current meaning unless "
                    "the current wording supports that interpretation.",
                    "If the current wording needs no prior context, copy it exactly as "
                    "the resolved question and copy each clause exactly as resolved_text.",
                    "Use only the current question, visible context sources, and visible "
                    "context frames.",
                ),
            ),
            builder.instruction_block(
                "Resolved Clauses",
                (
                    "Write resolution_basis first. State which explicit current "
                    "meanings replace prior meanings, which prior meanings remain, and "
                    "why the resulting question is coherent.",
                    "contextualized_question is the complete question formed from all "
                    "resolved clauses.",
                    "For every answerable clause, copy current_clause_text exactly and "
                    "write resolved_text as a complete standalone factual clause.",
                    "A resolved clause must retain every current constraint and every "
                    "prior meaning needed to answer it.",
                    "Every meaning added to the resolved clause but absent from the "
                    "current clause must be represented by a resolved value with a "
                    "visible prior source.",
                    "Writing the complete text is part of the resolution task: use it "
                    "to check that the selected meanings form one coherent question.",
                ),
            ),
            builder.instruction_block(
                "Retained Frame Shape",
                (
                    "retained_frame_parts records fixed prior question meaning that "
                    "the resolved clause still uses but the current clause omits.",
                    "Select only retained subject, output, population, or grouping "
                    "parts. Do not select a part that explicit current meaning replaces.",
                    "Fixed question shape belongs here, not in resolved values.",
                ),
            ),
            builder.instruction_block(
                "Resolved Values",
                (
                    "Create a resolved value for every meaning needed to interpret the "
                    "current clause that is established by a current span, context "
                    "anchor, or prior frame part.",
                    "resolved_text states the value's standalone meaning.",
                    "A current_span source copies one exact occurrence from the current "
                    "clause. A context_anchor source copies one shown typed anchor. A "
                    "frame_part source carries one shown part of a prior question.",
                    "Do not emit a resolved value for prior meaning that explicit "
                    "current wording replaces. When only part of a prior frame part "
                    "remains relevant, resolved_text states only that retained meaning.",
                    "A prior frame part is a source only when its own meaning is "
                    "retained. Do not cite a replaced frame part as support for a new "
                    "current value merely because both occupy the same structural "
                    "position, and do not blend the new value into the retained part.",
                    "Use every source that materially establishes the resolved value. "
                    "Current text and prior verified evidence may both support the same "
                    "value.",
                    "Every contextual meaning integrated into the complete clause must "
                    "also be represented as a resolved value; resolved values are the "
                    "structured handoff to later interpretation.",
                    "Do not classify values by linguistic operation or by whether they "
                    "are self-sufficient.",
                ),
            ),
            builder.instruction_block(
                "Callable Frames",
                (
                    "Use frame_call=call when and only when the complete resolved "
                    "question is exactly another invocation of one shown callable "
                    "frame's fixed factual function. A matching call is required, not "
                    "optional.",
                    "Write exactly one argument for every shown parameter. Use carry "
                    "when its prior value remains unchanged and resolved_value when a "
                    "resolved value supplies the new argument.",
                    "Before writing frame_call=none, compare the resolved question to "
                    "each frame as a function signature. If its fixed subject, outputs, "
                    "population, grouping, and computation are unchanged and every "
                    "changed value maps to a shown parameter, the frame matches and "
                    "must be called.",
                    "Use frame_call=none only when no callable frame matches, including "
                    "when the resolved question changes fixed subject, outputs, "
                    "population meaning, grouping, computation, or otherwise asks a "
                    "new factual function.",
                    "Do not force a new question into a frame merely because it refers "
                    "to the same entities or prior result.",
                ),
            ),
            builder.instruction_block(
                "Outcome",
                (
                    "Report ambiguity or missing information only when deciding how "
                    "the current utterance depends on prior conversation context. "
                    "Preserve coherent explicit current wording unchanged for "
                    "downstream question interpretation.",
                    "Use resolved when the visible evidence supports one complete "
                    "factual question.",
                    "Use multiple_meanings when visible evidence supports competing "
                    "complete interpretations, explain why, and include at least two "
                    "candidates.",
                    "Each competing interpretation must be a complete factual question "
                    "that retains all explicit current meaning and is directly supported "
                    "by visible evidence. A hypothetical alternative without such "
                    "evidence is not an ambiguity.",
                    "For each candidate, context_evidence cites the exact prior-context "
                    "evidence that produces that resolution. Competing candidates must "
                    "cite different context evidence; the same context evidence with "
                    "different imagined readings is not a conversation ambiguity.",
                    "Use missing_input when required information is absent and explain "
                    "what is missing.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    f"Return exactly one {CONVERSATION_RESOLUTION_TOOL_NAME} tool call.",
                    "Return only valid JSON arguments for that tool call.",
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
