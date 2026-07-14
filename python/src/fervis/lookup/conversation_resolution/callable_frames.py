"""Verified compilation and binding of callable prior question frames."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fervis.lookup.answer_program.contracts import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    ParameterBinding,
    ParameterValueType,
    parameter_value_type,
)
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.persistence import PriorProgramInvocationReader
from fervis.lookup.answer_program.rerun import RerunnableProgramInvocation
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
)
from fervis.lookup.question_contract import (
    KnownInputSource,
    QuestionContract,
    RequestedFactLiteralInput,
    RequestedFactRowSetReferenceInput,
)
from fervis.lookup.grounding.model import ExpectedInputIdentity
from fervis.memory.conversation_context import (
    ConversationCallableSignature,
    ConversationMemoryCardProjection,
)

from .compilation import (
    CompiledConversationResolution,
    ResolvedLiteralQuestionInput,
    ResolvedQuestionInput,
    ResolvedRowSetQuestionInput,
)


@dataclass(frozen=True)
class CallableFrameProgram:
    base: RerunnableProgramInvocation
    signature: ConversationCallableSignature
    question_contract: QuestionContract
    arguments_by_parameter_id: dict[str, str]

    @property
    def program(self) -> AnswerProgram:
        return self.base.program

    @property
    def changed_input_ids(self) -> frozenset[str]:
        return frozenset(
            _known_input_id(parameter_id)
            for parameter_id, value_id in self.arguments_by_parameter_id.items()
            if value_id
        )

    @property
    def expected_input_identities(self) -> dict[str, ExpectedInputIdentity]:
        expected: dict[str, ExpectedInputIdentity] = {}
        for parameter_id, value_id in self.arguments_by_parameter_id.items():
            if not value_id:
                continue
            binding = self.base.bindings.get(parameter_id)
            if binding is None:
                raise ValueError("callable frame parameter is not bound")
            identity = _expected_input_identity(binding.value)
            if identity is not None:
                expected[_known_input_id(parameter_id)] = identity
        return expected


def load_callable_frame_program(
    *,
    resolution: CompiledConversationResolution,
    memory_projection: ConversationMemoryCardProjection,
    reader: PriorProgramInvocationReader,
    conversation_id: str,
    tenant_id: str,
) -> CallableFrameProgram:
    call = resolution.frame_call
    if call is None:
        raise ValueError("callable frame preparation requires a frame call")
    try:
        frame = memory_projection.frame(call.frame_id)
    except KeyError as exc:
        raise ValueError("frame call references an unavailable frame") from exc
    signature = frame.callable
    if signature is None:
        raise ValueError("frame call references a non-callable frame")
    stored = reader.load_prior_answered_invocation(
        run_id=signature.base_run_id,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
    )
    if stored is None:
        raise ValueError("callable frame base invocation is unavailable")
    base = RerunnableProgramInvocation.parse(stored)
    _verify_program_signature(base.program, base.bindings, signature=signature)
    arguments = {
        argument.parameter_id: argument.resolved_value_ref()
        for argument in call.arguments
    }
    return CallableFrameProgram(
        base=base,
        signature=signature,
        question_contract=_invocation_question_contract(
            base.program,
            arguments_by_parameter_id=arguments,
            resolution=resolution,
        ),
        arguments_by_parameter_id=arguments,
    )


def callable_frame_bindings(
    prepared: CallableFrameProgram,
    *,
    grounded_values: tuple[FactValue, ...],
) -> BindingSet:
    values_by_input_id = {
        value.known_input_id: value for value in grounded_values if value.known_input_id
    }
    bindings: list[ParameterBinding] = []
    for base_binding in prepared.base.bindings.bindings:
        value_id = prepared.arguments_by_parameter_id.get(base_binding.parameter_id, "")
        if not value_id:
            bindings.append(base_binding)
            continue
        known_input_id = _known_input_id(base_binding.parameter_id)
        value = values_by_input_id.get(known_input_id)
        if value is None:
            raise ValueError("frame argument was not grounded")
        bindings.append(
            ParameterBinding(
                parameter_id=base_binding.parameter_id,
                value=_parameter_value(
                    prepared.program,
                    parameter_id=base_binding.parameter_id,
                    value=value,
                    base_value=base_binding.value,
                ),
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.QUESTION_INPUT,
                    refs=(
                        f"conversation_resolution:{value_id}",
                        f"run:{prepared.signature.base_run_id}",
                    ),
                ),
            )
        )
    return BindingSet.from_bindings(tuple(bindings))


def _parameter_value(
    program: AnswerProgram,
    *,
    parameter_id: str,
    value: FactValue,
    base_value: FactValue,
) -> FactValue:
    declaration = next(
        parameter for parameter in program.parameters if parameter.id == parameter_id
    )
    actual = parameter_value_type(value)
    _require_matching_identity(value, expected=_expected_input_identity(base_value))
    if actual is declaration.value_type:
        return value
    match declaration.value_type, value.payload:
        case ParameterValueType.IDENTITY_SET, IdentityValuePayload() as identity:
            return replace(
                value,
                payload=IdentitySetValuePayload(
                    keys=(identity.key,),
                    display_value=identity.display_value,
                ),
            )
        case _:
            raise ValueError("frame argument does not match its program parameter")


def _expected_input_identity(value: FactValue) -> ExpectedInputIdentity | None:
    match value.payload:
        case IdentityValuePayload() | IdentitySetValuePayload() as identity:
            return ExpectedInputIdentity(
                entity_kind=identity.entity_kind,
                key_id=identity.key_id,
                key_component_ids=tuple(
                    component.component_id
                    for component in (
                        identity.key.components
                        if isinstance(identity, IdentityValuePayload)
                        else identity.keys[0].components
                    )
                ),
            )
        case _:
            return None


def _require_matching_identity(
    value: FactValue,
    *,
    expected: ExpectedInputIdentity | None,
) -> None:
    if expected is None:
        return
    actual = _expected_input_identity(value)
    if actual != expected:
        raise ValueError("frame argument has the wrong canonical identity")


def _verify_program_signature(
    program: AnswerProgram,
    bindings: BindingSet,
    *,
    signature: ConversationCallableSignature,
) -> None:
    requested_fact_ids = tuple(fact.id for fact in program.fact_template)
    if requested_fact_ids != (signature.requested_fact_id,):
        raise ValueError("callable frame does not identify the saved program fact")
    signature_parameter_ids = {
        parameter.parameter_id for parameter in signature.parameters
    }
    question_parameter_ids = {
        parameter.id
        for parameter in program.parameters
        if parameter.id.startswith("question.")
    }
    if signature_parameter_ids != question_parameter_ids:
        raise ValueError(
            "callable signature does not match program question parameters"
        )
    if not signature_parameter_ids.issubset(bindings.parameter_ids):
        raise ValueError("callable program has an unbound question parameter")


def _invocation_question_contract(
    program: AnswerProgram,
    *,
    arguments_by_parameter_id: dict[str, str],
    resolution: CompiledConversationResolution,
) -> QuestionContract:
    base = QuestionContract(requested_facts=program.fact_template)
    inputs_by_ref = {item.input_ref: item for item in resolution.inputs}
    question_inputs = tuple(
        _invocation_question_input(
            known,
            value_id=arguments_by_parameter_id.get(f"question.{known.id}", ""),
            inputs_by_ref=inputs_by_ref,
        )
        for known in base.question_inputs
    )
    return QuestionContract(
        question_inputs=question_inputs,
        requested_facts=base.requested_facts,
    )


def _invocation_question_input(
    known: RequestedFactLiteralInput | RequestedFactRowSetReferenceInput,
    *,
    value_id: str,
    inputs_by_ref: dict[str, ResolvedQuestionInput],
) -> RequestedFactLiteralInput | RequestedFactRowSetReferenceInput:
    if not value_id:
        return known
    resolved = inputs_by_ref.get(f"conversation.{value_id}")
    if resolved is None:
        raise ValueError("frame argument does not identify a compiled input")
    match known:
        case RequestedFactLiteralInput():
            match resolved:
                case ResolvedLiteralQuestionInput():
                    if resolved.role is not known.role:
                        raise ValueError("frame argument has the wrong input role")
                    return replace(
                        known,
                        source=KnownInputSource.CONVERSATION_RESOLUTION,
                        text=resolved.value_source_text,
                        resolved_value_text=resolved.resolved_value_text,
                        resolved_input_ref=resolved.input_ref,
                        field_label_text=(
                            resolved.field_label_text or known.field_label_text
                        ),
                        value_meaning_hint=(
                            resolved.value_meaning_hint or known.value_meaning_hint
                        ),
                    )
                case ResolvedRowSetQuestionInput():
                    raise ValueError("callable frames cannot replace row-set inputs")
        case RequestedFactRowSetReferenceInput():
            raise ValueError("callable frames cannot replace row-set inputs")


def _known_input_id(parameter_id: str) -> str:
    prefix = "question."
    if not parameter_id.startswith(prefix):
        raise ValueError("frame argument does not bind a question parameter")
    return parameter_id[len(prefix) :]
