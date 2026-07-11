from __future__ import annotations

from dataclasses import dataclass
from fervis.lineage.enums import ProgramInvocationKind

from fervis.lookup.answer_program.codec import answer_program_id
from fervis.lookup.answer_program.contracts import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRole,
    ParameterValueType,
)
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.persistence import (
    StoredProgramInvocation,
    program_invocation,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.conversation_resolution.callable_frames import (
    callable_frame_bindings,
    load_callable_frame_program,
)
from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
    CompiledResolvedClause,
    CompiledResolvedValue,
    ResolvedLiteralQuestionInput,
)
from fervis.lookup.conversation_resolution.model import (
    ConversationFrameCall,
    CurrentSpanSource,
    ResolvedValueFrameArgument,
)
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
)
from fervis.memory.conversation_context import (
    ConversationAnswerShape,
    ConversationCallableSignature,
    ConversationContextFrame,
    ConversationFrameParameter,
    ConversationFramePart,
    ConversationFramePartKind,
    ConversationMemoryCardProjection,
)


def test_callable_frame_reuses_shape_and_rebinds_only_changed_argument() -> None:
    program, bindings = _base_program()
    stored = StoredProgramInvocation(
        invocation=program_invocation(
            run_id="run_1",
            program_id=answer_program_id(program),
            bindings=bindings,
            kind=ProgramInvocationKind.COMPILED_QUESTION,
        ),
        program=program,
    )
    reader = _Reader(stored)
    prepared = load_callable_frame_program(
        resolution=_resolution(),
        memory_projection=_memory_projection(),
        reader=reader,
        conversation_id="conversation_1",
        tenant_id="tenant_1",
    )

    current_input = prepared.question_contract.question_inputs[0]
    assert current_input == RequestedFactLiteralInput(
        id="place",
        source=KnownInputSource.CONVERSATION_RESOLUTION,
        role=LiteralInputRole.REFERENCE_VALUE,
        text="Pivot Mall",
        resolved_value_text="Pivot Mall",
        field_label_text="mall",
        value_meaning_hint="mall identity",
        resolved_input_ref="conversation.place_2",
    )

    current_value = FactValue.identity(
        id="grounded_place",
        known_input_id="place",
        identity_type="mall",
        identity_field="id",
        value="mall_2",
        display_value="Pivot Mall",
        proof_refs=("source_read:mall_2",),
        applies_to_requested_fact_ids=("sales_count",),
    )
    rebound = callable_frame_bindings(
        prepared,
        grounded_values=(current_value,),
    )

    rebound_value = rebound.get("question.place").value
    assert rebound_value.known_input_id == "place"
    assert rebound_value.payload.canonical_value() == "mall_2"
    assert reader.request == ("run_1", "conversation_1", "tenant_1")


@dataclass
class _Reader:
    stored: StoredProgramInvocation
    request: tuple[str, str, str] | None = None

    def load_prior_answered_invocation(
        self,
        *,
        run_id: str,
        conversation_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation:
        self.request = (run_id, conversation_id, tenant_id)
        return self.stored


def _base_program() -> tuple[AnswerProgram, BindingSet]:
    known = RequestedFactLiteralInput(
        id="place",
        source=KnownInputSource.QUESTION_CONTEXT,
        role=LiteralInputRole.REFERENCE_VALUE,
        text="Acacia Mall",
        resolved_value_text="Acacia Mall",
        field_label_text="mall",
        value_meaning_hint="mall identity",
    )
    program = AnswerProgram(
        fact_template=(
            RequestedFact(
                id="sales_count",
                description="sales count",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer",
                        description="sales count",
                        role="ROW_POPULATION",
                    ),
                ),
                known_inputs=(known,),
                input_refs=(known.id,),
            ),
        ),
        parameters=(
            ParameterDeclaration(
                id="question.place",
                role=ParameterRole.QUESTION_INPUT,
                value_type=ParameterValueType.IDENTITY,
            ),
        ),
    )
    base_value = FactValue.identity(
        id="grounded_place",
        known_input_id="place",
        identity_type="mall",
        identity_field="id",
        value="mall_1",
        display_value="Acacia Mall",
        proof_refs=("source_read:mall_1",),
        applies_to_requested_fact_ids=("sales_count",),
    )
    return program, BindingSet.from_bindings(
        (
            ParameterBinding(
                parameter_id="question.place",
                value=base_value,
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.QUESTION_INPUT,
                    refs=("known_input:place",),
                ),
            ),
        )
    )


def _memory_projection() -> ConversationMemoryCardProjection:
    return ConversationMemoryCardProjection(
        context_frames=(
            ConversationContextFrame(
                frame_id="request:1",
                source_ids=("prior_question",),
                answer_shape=ConversationAnswerShape(
                    expression_family="scalar_aggregate",
                    output_roles=("ROW_POPULATION",),
                ),
                parts=(
                    ConversationFramePart(
                        part_id="input:entity_identity:1",
                        kind=ConversationFramePartKind.ENTITY_IDENTITY,
                        text="Acacia Mall",
                        source_ref="place",
                    ),
                ),
                callable=ConversationCallableSignature(
                    base_run_id="run_1",
                    requested_fact_id="sales_count",
                    parameters=(
                        ConversationFrameParameter(
                            parameter_id="question.place",
                            part_id="input:entity_identity:1",
                            kind=ConversationFramePartKind.ENTITY_IDENTITY,
                            current_text="Acacia Mall",
                            resolved_text="Acacia Mall",
                            field_label_text="mall",
                            value_meaning_hint="mall identity",
                        ),
                    ),
                ),
            ),
        )
    )


def _resolution() -> CompiledConversationResolution:
    source = CurrentSpanSource(text="Pivot Mall", occurrence=1)
    return CompiledConversationResolution(
        current_question_text="What about Pivot Mall?",
        contextualized_question="How many sales at Pivot Mall?",
        clauses=(
            CompiledResolvedClause(
                current_clause_text="What about Pivot Mall?",
                resolved_text="How many sales at Pivot Mall?",
                retained_frame_parts=(),
                values=(
                    CompiledResolvedValue(
                        value_id="place_2",
                        resolved_text="Pivot Mall",
                        source_kinds=("entity_identity",),
                        sources=(source,),
                        field_label_text="mall",
                        value_meaning_hint="mall identity",
                    ),
                ),
            ),
        ),
        inputs=(
            ResolvedLiteralQuestionInput(
                input_ref="conversation.place_2",
                value_source_text="Pivot Mall",
                resolved_value_text="Pivot Mall",
                role=LiteralInputRole.REFERENCE_VALUE,
                field_label_text="mall",
                value_meaning_hint="mall identity",
            ),
        ),
        frame_call=ConversationFrameCall(
            frame_id="request:1",
            arguments=(
                ResolvedValueFrameArgument(
                    parameter_id="question.place",
                    value_id="place_2",
                ),
            ),
        ),
        used_source_card_ids=("prior_card",),
        used_memory_ids=("prior_request",),
    )
