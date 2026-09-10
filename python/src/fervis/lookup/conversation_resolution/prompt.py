"""Prompt for conversation resolution."""

from __future__ import annotations

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
)
from fervis.lookup.clarification.context import (
    active_clarification,
    clarification_context_source,
)
from fervis.lookup.conversation_resolution.model import ConversationResolutionRequest
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


def _context_source_model_payload(source: ConversationContextSource, frames: tuple[ConversationContextFrame, ...]) -> dict[str, object]:
    payload = source.to_model_dict()
    frame_ids = [frame.frame_id for frame in frames if source.source_id in frame.source_ids]
    if source.kind == "prior_fervis_answer" and not source.meaning_anchors and frame_ids:
        payload.pop("text", None)
        payload["request_frame_ids"] = frame_ids
        payload["value_authority"] = "Use the typed request frames to evaluate the earlier facts again; their previous result values are not new literal inputs."
    return payload


class ConversationResolutionTurnPrompt(TurnPromptBase):
    turn_name = "conversation resolution"
    turn_task = (
        "resolve how the current utterance depends on prior conversation context"
    )

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
        sections = [
            builder.json_section(
                "Context sources:",
                {
                    "current_question_text": self.request.question,
                    "context_sources": [
                        _context_source_model_payload(item, conversation_resolution_context_frames(self.request))
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
        ]
        responses = tuple(
            response
            for response in self.request.clarification_responses
            if response.candidate is not None
        )
        if responses:
            sections.append(
                builder.json_section(
                    "Attributed clarification responses:",
                    {
                        "responses": [
                            {
                                "response_id": response.source.response_id,
                                "clarification_id": response.source.clarification_id,
                                "exact_user_text": response.source.exact_user_text,
                                "selected_candidate": (
                                    None
                                    if response.candidate is None
                                    else {
                                        "id": response.candidate.id,
                                        "contextualized_question": response.candidate.contextualized_question,
                                        "source_evidence": [
                                            {
                                                "source_id": item.source_id,
                                                "exact_source_texts": list(
                                                    item.exact_source_texts
                                                ),
                                            }
                                            for item in response.candidate.source_evidence
                                        ],
                                    }
                                ),
                            }
                            for response in responses
                        ]
                    },
                )
            )
        return tuple(sections)

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
                    "When the current wording states the complete factual request, "
                    "copy it exactly as the resolved question and copy each clause "
                    "exactly as resolved_text.",
                    "Use only the current question, visible context sources, and visible "
                    "context frames.",
                    "An active_clarification context source contains the original question and every clarification question and answer in order. Resolve the current utterance using the complete chain.",
                    "Each answer resolves the clarification question it follows. Later exchanges extend the established question context; they do not replace earlier exchanges.",
                ),
            ),
            builder.instruction_block(
                "Resolved Clauses",
                (
                    "Write resolution_basis first. State which explicit current "
                    "meanings replace prior meanings, which prior meanings remain, and "
                    "why the resulting question is coherent.",
                    "For every answerable clause, copy current_clause_text exactly and "
                    "write resolved_text as a complete standalone factual clause.",
                    "Each resolved_text is the complete factual request for its current-clause span. The engine combines these spans into the resolved question; do not author a second copy.",
                    "A resolved clause must retain every current constraint and every "
                    "prior meaning needed to answer it.",
                    "Every value added to the resolved clause but absent from the "
                    "current clause must be represented by a resolved value with a "
                    "visible prior source.",
                    "Writing the complete text is part of the resolution task: use it "
                    "to check that the selected meanings form one coherent question.",
                ),
            ),
            builder.instruction_block(
                "Retained Frame Shape",
                (
                    "Write request_shape_basis before request_shape_source. "
                    "request_shape_source records where the current factual request "
                    "gets its request shape.",
                    "current_clause_supplies_request means current_clause_text itself "
                    "states the complete factual request. A complete request containing "
                    "a pronoun or contextual reference keeps the current request shape "
                    "while prior evidence resolves that value. Canonical identity or "
                    "type evidence for a value in that request belongs in resolved "
                    "values. retained_frame_parts is empty.",
                    "active_clarification_supplies_request means current_clause_text "
                    "answers or continues the shown active clarification, whose "
                    "original question supplies the factual request. "
                    "retained_frame_parts is empty.",
                    "prior_frame_supplies_omitted_request_parts means "
                    "current_clause_text supplies only a changed or reaffirmed value "
                    "while a shown prior frame supplies omitted request parts. "
                    "Conversational scaffolding such as 'what about', 'how about', "
                    "'same question', 'and', or 'instead' supplies no request shape. "
                    "retained_frame_parts contains exactly the omitted subject, "
                    "qualification, grouping, requested output, ordering, selection, "
                    "or canonical output identity.",
                ),
            ),
            builder.instruction_block(
                "Resolved Values",
                (
                    "Create a resolved value for every value needed to interpret the "
                    "current clause that is established by a current span, context "
                    "anchor, or prior input frame part.",
                    "resolved_text states the value's standalone meaning.",
                    "A current_span source copies one exact occurrence from the current "
                    "clause. A context_anchor source copies one shown typed anchor. A "
                    "frame_part source carries one shown part of a prior question.",
                    "Do not emit a resolved value established only by current_span unless it supplies a shown callable parameter; "
                    "otherwise later interpretation already receives the current clause.",
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
                    "Resolved values are the structured handoff for values to later "
                    "interpretation; retained frame shape is already represented by "
                    "retained_frame_parts.",
                    "Do not classify values by linguistic operation or by whether they "
                    "are self-sufficient.",
                ),
            ),
            builder.instruction_block(
                "Callable Parameters",
                (
                    "For each resolved value, set frame_parameter to the shown callable "
                    "parameter that the value supplies, or none when it supplies no "
                    "shown parameter.",
                    "When a named or string callable parameter is supplied by one current_span, copy just its literal argument value in that span. Execution binds that copied text; resolved_text may describe its meaning but does not replace the literal data.",
                    "Represent unchanged callable values as resolved values too, using "
                    "their visible prior sources and parameter references.",
                    "Do not decide whether to call a prior frame. The backend derives "
                    "that result only when the retained fixed parts and parameter "
                    "bindings completely match one callable signature.",
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
                    "Use missing_input only when essential meaning is absent, so the "
                    "visible conversation does not determine one complete factual "
                    "question. Facts that must be fetched to answer a complete question "
                    "are downstream data, not missing question meaning.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    f"Return exactly one {CONVERSATION_RESOLUTION_TOOL_NAME} tool call.",
                    "Return only valid JSON arguments for that tool call.",
                ),
            ),
            builder.instruction_block('Context value authority', (
                'Resolved values cite only the shown meaning anchors or permitted frame parts. Earlier answer text is discourse context, not authority for new literal inputs. For a new calculation over earlier facts, carry their original subject and input constraints and express the new calculation, so those facts can be evaluated again.',
            )),
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
    responses = tuple(
        response
        for response in request.clarification_responses
        if response.annotation is not None
    )
    if not responses:
        return request.context_sources
    chain = active_clarification(responses)
    source = clarification_context_source(chain)
    return (*request.context_sources, source)


def conversation_resolution_context_frames(
    request: ConversationResolutionRequest,
) -> tuple[ConversationContextFrame, ...]:
    return request.context_frames
