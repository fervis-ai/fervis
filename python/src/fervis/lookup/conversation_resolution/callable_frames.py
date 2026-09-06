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
from fervis.lookup.contract_codec import answer_program_id, canonical_contract_fingerprint
from fervis.lookup.answer_program.persistence import PriorProgramInvocationReader
from fervis.lookup.answer_program.rerun import RerunnableProgramInvocation
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
)
from fervis.lookup.question_contract import InputDenotation, InputTerm
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
from fervis.lookup.grounding.identity import ExpectedInputIdentity
from fervis.memory.conversation_context import (
    ConversationCallableSignature,
    ConversationMemoryCardProjection,
)

from .compilation import (
    CompiledConversationResolution,
    ResolvedCanonicalIdentity,
    ResolvedLiteralQuestionInput,
    ResolvedQuestionInput,
    ResolvedRowSetQuestionInput,
)


@dataclass(frozen=True)
class CallableFrameArgument:
    parameter_id: str
    input_ref: str
    input_use_refs: tuple[str, ...]
    source_text: str
    operand: str
    expected_identity: ExpectedInputIdentity | None
    canonical_identity: ResolvedCanonicalIdentity | None = None


@dataclass(frozen=True)
class CallableFrameProgram:
    base: RerunnableProgramInvocation
    signature: ConversationCallableSignature
    arguments: tuple[CallableFrameArgument, ...]

    @property
    def program(self) -> AnswerProgram:
        return self.base.program

    @property
    def changed_input_ids(self) -> frozenset[str]:
        return frozenset(argument.input_ref for argument in self.arguments)

    @property
    def expected_input_identities(self) -> dict[str, ExpectedInputIdentity]:
        expected: dict[str, ExpectedInputIdentity] = {}
        for argument in self.arguments:
            identity = argument.expected_identity
            if identity is not None:
                expected[argument.input_ref] = identity
        return expected

    @property
    def changed_inputs(self) -> tuple[InputTerm, ...]:
        inputs = {item.id: item for item in self.program.inputs}
        return tuple(
            replace(
                inputs[argument.input_ref],
                origin=_changed_input_origin(argument),
                operand=argument.operand,
            )
            for argument in self.arguments
        )

    @property
    def changed_input_denotations(self) -> tuple[InputDenotation, ...]:
        denotations = {
            item.input_ref: item for item in self.program.input_denotations
        }
        return tuple(
            replace(
                denotations[argument.input_ref],
                operand_meaning=argument.operand,
                denotation_basis=argument.source_text,
                denoted_instance_kind=(
                    argument.expected_identity.entity_kind
                    if argument.expected_identity is not None
                    else denotations[argument.input_ref].denoted_instance_kind
                ),
            )
            for argument in self.arguments
        )

    @property
    def changed_input_origins(self) -> dict[str, SourceOrigin]:
        return {
            argument.input_ref: _changed_input_origin(argument)
            for argument in self.arguments
        }

    @property
    def certified_argument_values(self) -> tuple[FactValue, ...]:
        requested_fact_ids = tuple(item.id for item in self.program.fact_template)
        return tuple(
            FactValue.identity(
                id=f"{argument.input_ref}:conversation_identity",
                known_input_id=argument.input_ref,
                key=argument.canonical_identity.key,
                display_value=argument.operand,
                proof_refs=argument.canonical_identity.authority_refs,
                source_refs=argument.canonical_identity.lineage_refs,
                applies_to_requested_fact_ids=requested_fact_ids,
            )
            for argument in self.arguments
            if argument.canonical_identity is not None
        )


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
    stored = reader.load_prior_invocation(
        invocation_id=signature.base_invocation_id,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
    )
    if stored is None:
        raise ValueError("callable frame base invocation is unavailable")
    base = RerunnableProgramInvocation.parse(stored)
    _verify_program_signature(base.program, base.bindings, signature=signature)
    argument_refs = {
        argument.parameter_id: f"conversation.{argument.resolved_value_ref()}"
        for argument in call.arguments
        if argument.resolved_value_ref()
    }
    inputs_by_ref = {item.input_ref: item for item in resolution.inputs}
    return CallableFrameProgram(
        base=base,
        signature=signature,
        arguments=tuple(
            _callable_argument(
                base,
                parameter_id=parameter_id,
                resolved_value_ref=resolved_value_ref,
                inputs_by_ref=inputs_by_ref,
            )
            for parameter_id, resolved_value_ref in argument_refs.items()
        ),
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
    arguments = {item.parameter_id: item for item in prepared.arguments}
    for base_binding in prepared.base.bindings.bindings:
        argument = arguments.get(base_binding.parameter_id)
        if argument is None:
            bindings.append(base_binding)
            continue
        value = values_by_input_id.get(argument.input_ref)
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
                        f"conversation_resolution:{argument.input_ref}",
                        f"invocation:{prepared.signature.base_invocation_id}",
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
    if signature.program_id != answer_program_id(program):
        raise ValueError("callable signature does not identify the saved program")
    if (
        len(program.fact_template) != 1
        or program.fact_template[0].id != signature.requested_fact_ref
    ):
        raise ValueError("callable frame must identify the complete saved program")
    fact = next(
        (
            item
            for item in program.fact_template
            if item.id == signature.requested_fact_ref
        ),
        None,
    )
    if fact is None:
        raise ValueError("callable frame does not identify the saved program fact")
    if canonical_contract_fingerprint(fact) != signature.requested_fact_fingerprint:
        raise ValueError("callable signature fact fingerprint is stale")
    declarations = {parameter.id: parameter for parameter in program.parameters}
    signature_parameter_ids = {item.parameter_id for item in signature.parameters}
    for parameter in signature.parameters:
        declaration = declarations.get(parameter.parameter_id)
        if declaration is None or (
            declaration.value_type.value,
            declaration.input_ref,
            declaration.input_use_refs,
        ) != (
            parameter.value_type,
            parameter.input_ref,
            parameter.input_use_refs,
        ):
            raise ValueError("callable parameter signature is stale")
    if not signature_parameter_ids.issubset(bindings.parameter_ids):
        raise ValueError("callable program has an unbound parameter")


def _callable_argument(
    base: RerunnableProgramInvocation,
    *,
    parameter_id: str,
    resolved_value_ref: str,
    inputs_by_ref: dict[str, ResolvedQuestionInput],
) -> CallableFrameArgument:
    declaration = next(
        (item for item in base.program.parameters if item.id == parameter_id),
        None,
    )
    if declaration is None or not declaration.input_ref:
        raise ValueError("callable argument does not address a question input")
    resolved = inputs_by_ref.get(resolved_value_ref)
    if resolved is None:
        raise ValueError("frame argument does not identify a compiled input")
    if isinstance(resolved, ResolvedRowSetQuestionInput):
        raise ValueError("callable frames cannot replace row-set inputs")
    if not isinstance(resolved, ResolvedLiteralQuestionInput):
        raise TypeError("unsupported callable frame argument")
    binding = base.bindings.get(parameter_id)
    if binding is None:
        raise ValueError("callable frame parameter is not bound")
    return CallableFrameArgument(
        parameter_id=parameter_id,
        input_ref=declaration.input_ref,
        input_use_refs=declaration.input_use_refs,
        source_text=resolved.value_source_text,
        operand=resolved.resolved_value_text,
        expected_identity=_expected_input_identity(binding.value),
        canonical_identity=resolved.canonical_identity,
    )


def _changed_input_origin(argument: CallableFrameArgument) -> SourceOrigin:
    return SourceOrigin(
        SourceOriginKind.CONVERSATION_RESOLUTION,
        argument.source_text,
        resolved_input_ref=argument.input_ref,
    )
