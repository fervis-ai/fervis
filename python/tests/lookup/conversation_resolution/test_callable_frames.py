from __future__ import annotations

from dataclasses import dataclass, replace
from fervis.lineage.enums import ProgramInvocationKind

import pytest

from fervis.lookup.contract_codec import answer_program_id
from fervis.lookup.contract_codec import canonical_contract_fingerprint
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
from fervis.lookup.canonical_data import entity_key_value
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
from fervis.lookup.question_contract.model import (
    AllResults,
    InputDenotation,
    InputDenotationKind,
    InputTerm,
    InstanceInterpretation,
    RequestedFact,
    SetTerm,
    Subject,
)
from fervis.lookup.semantic_types import (
    IdentifierType,
    SourceOrigin,
    SourceOriginKind,
)
from fervis.memory.conversation_context import (
    ConversationCallableParameter,
    ConversationCallableSignature,
    ConversationContextFrame,
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

    [current_input] = prepared.changed_inputs
    assert current_input.id == "place"
    assert current_input.operand == "Pivot Mall"
    assert current_input.origin.meaning == "Pivot Mall"

    current_value = FactValue.identity(
        id="grounded_place",
        known_input_id="place",
        key=entity_key_value(
            "mall",
            "tenant_mall_key",
            {"tenant_id": "tenant_1", "id": "mall_2"},
        ),
        display_value="Pivot Mall",
        proof_refs=("source_read:mall_2",),
        applies_to_requested_fact_ids=("sales_count",),
    )
    rebound = callable_frame_bindings(
        prepared,
        grounded_values=(current_value,),
    )

    rebound_value = rebound.get("value.place").value
    assert rebound_value.known_input_id == "place"
    assert rebound_value.payload.canonical_value() == entity_key_value(
        "mall",
        "tenant_mall_key",
        {"tenant_id": "tenant_1", "id": "mall_2"},
    )
    assert reader.request == (
        stored.invocation.invocation_id,
        "conversation_1",
        "tenant_1",
    )


def test_callable_frame_rejects_fact_scoped_signature_for_multi_fact_program() -> None:
    program, bindings = _base_program()
    [fact] = program.fact_template
    multi_fact_program = replace(
        program,
        fact_template=(fact, replace(fact, id="returns_count")),
    )
    stored = StoredProgramInvocation(
        invocation=program_invocation(
            run_id="run_1",
            program_id=answer_program_id(multi_fact_program),
            bindings=bindings,
            kind=ProgramInvocationKind.COMPILED_QUESTION,
        ),
        program=multi_fact_program,
    )
    projection = _memory_projection()
    [frame] = projection.context_frames
    assert frame.callable is not None
    projection = replace(
        projection,
        context_frames=(
            replace(
                frame,
                callable=replace(
                    frame.callable,
                    program_id=stored.invocation.program_id,
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="callable frame must identify the complete saved program",
    ):
        load_callable_frame_program(
            resolution=_resolution(),
            memory_projection=projection,
            reader=_Reader(stored),
            conversation_id="conversation_1",
            tenant_id="tenant_1",
        )


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

    def load_prior_invocation(
        self,
        *,
        invocation_id: str,
        conversation_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation:
        self.request = (invocation_id, conversation_id, tenant_id)
        return self.stored


def _base_program() -> tuple[AnswerProgram, BindingSet]:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "sales at Acacia Mall")
    input_term = InputTerm(
        id="place",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "Acacia Mall"),
        operand="Acacia Mall",
        value_type=IdentifierType("s1"),
    )
    fact = RequestedFact(
        id="sales_count",
        origin=origin,
        sets=(
            SetTerm(
                id="s1", origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "malls")
            ),
        ),
        associations=(),
        facts=(),
        expressions=(),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    program = AnswerProgram(
        inputs=(input_term,),
        input_denotations=(
            InputDenotation(
                id="denotation_1",
                input_ref="place",
                operand_meaning="the mall whose sales are counted",
                denotation_basis="Acacia Mall names a mall",
                denoted_instance_kind="mall",
                kind=InputDenotationKind.IDENTITY_REFERENCE,
            ),
        ),
        fact_template=(fact,),
        parameters=(
            ParameterDeclaration(
                id="value.place",
                role=ParameterRole.QUESTION_INPUT,
                value_type=ParameterValueType.IDENTITY,
                input_ref="place",
                input_use_refs=("sales_count:input_use:place",),
            ),
        ),
    )
    base_value = FactValue.identity(
        id="grounded_place",
        known_input_id="place",
        key=entity_key_value(
            "mall",
            "tenant_mall_key",
            {"tenant_id": "tenant_1", "id": "mall_1"},
        ),
        display_value="Acacia Mall",
        proof_refs=("source_read:mall_1",),
        applies_to_requested_fact_ids=("sales_count",),
    )
    return program, BindingSet.from_bindings(
        (
            ParameterBinding(
                parameter_id="value.place",
                value=base_value,
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.QUESTION_INPUT,
                    refs=("known_input:place",),
                ),
            ),
        )
    )


def _memory_projection() -> ConversationMemoryCardProjection:
    program, bindings = _base_program()
    invocation = program_invocation(
        run_id="run_1",
        program_id=answer_program_id(program),
        bindings=bindings,
        kind=ProgramInvocationKind.COMPILED_QUESTION,
    )
    fact = program.fact_template[0]
    return ConversationMemoryCardProjection(
        context_frames=(
            ConversationContextFrame(
                frame_id="request:1",
                source_ids=("prior_question",),
                parts=(
                    ConversationFramePart(
                        part_id="input:value.place",
                        kind=ConversationFramePartKind.INPUT,
                        text="Acacia Mall",
                        source_ref="place",
                        value_type="identity",
                    ),
                ),
                callable=ConversationCallableSignature(
                    base_invocation_id=invocation.invocation_id,
                    program_id=invocation.program_id,
                    requested_fact_ref="sales_count",
                    requested_fact_fingerprint=canonical_contract_fingerprint(fact),
                    parameters=(
                        ConversationCallableParameter(
                            parameter_id="value.place",
                            part_id="input:value.place",
                            value_type="identity",
                            input_ref="place",
                            input_use_refs=("sales_count:input_use:place",),
                            current_text="Acacia Mall",
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
                        value_type="identity",
                    ),
                ),
            ),
        ),
        inputs=(
            ResolvedLiteralQuestionInput(
                input_ref="conversation.place_2",
                value_source_text="Pivot Mall",
                resolved_value_text="Pivot Mall",
                value_type="identity",
            ),
        ),
        frame_call=ConversationFrameCall(
            frame_id="request:1",
            arguments=(
                ResolvedValueFrameArgument(
                    parameter_id="value.place",
                    value_id="place_2",
                ),
            ),
        ),
        used_source_card_ids=("prior_card",),
        used_memory_ids=("prior_request",),
    )
