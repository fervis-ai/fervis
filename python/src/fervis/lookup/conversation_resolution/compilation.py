"""Compile resolved conversation meaning into the next runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from fervis.lookup.question_inputs import (
    KnownInputKind,
    LiteralInputRole,
)
from fervis.memory.conversation_context import (
    ConversationAnswerShape,
    ConversationContextFrame,
    ConversationFrameParameter,
    ConversationFramePart,
    ConversationFramePartKind,
    ConversationMemoryCardProjection,
)
from fervis.memory.prior_requests import PriorEntityIdentityBinding

from .model import (
    ConversationFrameCall,
    ConversationResolution,
    FramePartSource,
    ResolutionSource,
    ResolvedConversationClause,
    ResolvedConversationValue,
)

if TYPE_CHECKING:
    from fervis.lookup.question_contract.model import (
        RequestedFactKnownInput,
        RequestedFactLiteralInput,
        RequestedFactRowSetReferenceInput,
    )


@dataclass(frozen=True)
class ResolvedCanonicalIdentity:
    entity_kind: str
    key_id: str
    key_component_id: str
    value: str
    authority_refs: tuple[str, ...]
    lineage_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            (self.entity_kind, self.key_id, self.key_component_id, self.value)
        ):
            raise ValueError("resolved identity requires a complete candidate-key value")
        if not self.authority_refs or not self.lineage_refs:
            raise ValueError("resolved identity requires authority and lineage")


@dataclass(frozen=True)
class ResolvedIdentityInput:
    input_ref: str
    value_source_text: str
    resolved_value_text: str
    role: LiteralInputRole
    occurrence: int
    field_label_text: str
    value_meaning_hint: str
    canonical_identity: ResolvedCanonicalIdentity


@dataclass(frozen=True)
class ResolvedLiteralQuestionInput:
    input_ref: str
    value_source_text: str
    resolved_value_text: str
    role: LiteralInputRole
    occurrence: int = 1
    field_label_text: str = ""
    value_meaning_hint: str = ""
    evidence_refs: tuple[str, ...] = ()
    canonical_identity: ResolvedCanonicalIdentity | None = None

    @property
    def kind(self) -> KnownInputKind:
        return KnownInputKind.LITERAL

    def __post_init__(self) -> None:
        if not all((self.input_ref, self.value_source_text, self.resolved_value_text)):
            raise ValueError("resolved literal input requires identity and value")
        if self.occurrence < 1:
            raise ValueError("resolved literal input occurrence must be positive")

    def to_prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind.value,
            "input_ref": self.input_ref,
            "value_source_text": self.value_source_text,
            "resolved_value_text": self.resolved_value_text,
            "role": self.role.value,
            "occurrence": self.occurrence,
        }
        if self.field_label_text:
            payload["field_label_text"] = self.field_label_text
        if self.value_meaning_hint:
            payload["value_meaning_hint"] = self.value_meaning_hint
        return payload

    def accepts(self, known: RequestedFactLiteralInput) -> bool:
        return (
            known.kind is self.kind
            and known.text == self.value_source_text
            and known.resolved_input_ref == self.input_ref
            and known.resolved_value_text == self.resolved_value_text
            and known.role is self.role
            and known.occurrence == self.occurrence
            and known.field_label_text == self.field_label_text
            and known.value_meaning_hint == self.value_meaning_hint
        )

    def context_texts(self) -> tuple[str, ...]:
        return (self.value_source_text, self.resolved_value_text)

    def identity_input(self) -> ResolvedIdentityInput | None:
        if self.canonical_identity is None:
            return None
        return ResolvedIdentityInput(
            input_ref=self.input_ref,
            value_source_text=self.value_source_text,
            resolved_value_text=self.resolved_value_text,
            role=self.role,
            occurrence=self.occurrence,
            field_label_text=self.field_label_text,
            value_meaning_hint=self.value_meaning_hint,
            canonical_identity=self.canonical_identity,
        )

    def row_set_memory_references(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class ResolvedRowSetQuestionInput:
    input_ref: str
    reference_text: str
    memory_ids: tuple[str, ...]
    occurrence: int = 1

    @property
    def kind(self) -> KnownInputKind:
        return KnownInputKind.ROW_SET_REFERENCE

    def __post_init__(self) -> None:
        if not self.input_ref or not self.reference_text or not self.memory_ids:
            raise ValueError("resolved row-set input requires identity and memory")
        if self.occurrence < 1:
            raise ValueError("resolved row-set occurrence must be positive")

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "input_ref": self.input_ref,
            "reference_text": self.reference_text,
            "occurrence": self.occurrence,
        }

    def accepts(self, known: RequestedFactRowSetReferenceInput) -> bool:
        return (
            known.kind is self.kind
            and known.text == self.reference_text
            and known.resolved_input_ref == self.input_ref
            and known.occurrence == self.occurrence
        )

    def context_texts(self) -> tuple[str, ...]:
        return (self.reference_text,)

    def identity_input(self) -> ResolvedIdentityInput | None:
        return None

    def row_set_memory_references(self) -> tuple[str, ...]:
        return self.memory_ids


ResolvedQuestionInput: TypeAlias = (
    ResolvedLiteralQuestionInput | ResolvedRowSetQuestionInput
)


@dataclass(frozen=True)
class CompiledResolvedValue:
    value_id: str
    resolved_text: str
    source_kinds: tuple[str, ...]
    sources: tuple[ResolutionSource, ...]
    field_label_text: str = ""
    value_meaning_hint: str = ""
    canonical_identity: ResolvedCanonicalIdentity | None = None

    def to_prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "value_id": self.value_id,
            "resolved_text": self.resolved_text,
        }
        if self.source_kinds:
            payload["source_kinds"] = list(self.source_kinds)
        return payload


@dataclass(frozen=True)
class CompiledRetainedFramePart:
    kind: ConversationFramePartKind
    text: str
    answer_shape: ConversationAnswerShape | None = None

    def to_prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind.value,
            "text": self.text,
        }
        if self.answer_shape is not None:
            payload["answer_shape"] = self.answer_shape.to_model_dict()
        return payload


@dataclass(frozen=True)
class CompiledResolvedClause:
    current_clause_text: str
    resolved_text: str
    retained_frame_parts: tuple[CompiledRetainedFramePart, ...]
    values: tuple[CompiledResolvedValue, ...]

    def to_prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "current_clause_text": self.current_clause_text,
            "resolved_values": [
                value.to_prompt_payload() for value in self.values if value.source_kinds
            ],
        }
        if self.retained_frame_parts:
            payload["retained_frame_parts"] = [
                part.to_prompt_payload() for part in self.retained_frame_parts
            ]
        return payload


@dataclass(frozen=True)
class CompiledConversationResolution:
    current_question_text: str
    contextualized_question: str
    clauses: tuple[CompiledResolvedClause, ...]
    inputs: tuple[ResolvedQuestionInput, ...]
    frame_call: ConversationFrameCall | None
    used_source_card_ids: tuple[str, ...]
    used_memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.current_question_text or not self.contextualized_question:
            raise ValueError("compiled conversation resolution requires both questions")
        input_refs = tuple(item.input_ref for item in self.inputs)
        if len(input_refs) != len(set(input_refs)):
            raise ValueError("compiled conversation resolution has duplicate inputs")

    def to_prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "clauses": [clause.to_prompt_payload() for clause in self.clauses],
        }
        if self.inputs:
            payload["resolved_question_inputs"] = [
                item.to_prompt_payload() for item in self.inputs
            ]
        return payload

    def context_texts(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *(
                        value.resolved_text
                        for clause in self.clauses
                        for value in clause.values
                        if value.source_kinds
                    ),
                    *(
                        part.text
                        for clause in self.clauses
                        for part in clause.retained_frame_parts
                    ),
                    *(text for item in self.inputs for text in item.context_texts()),
                )
            )
        )

    def accepts_question_input(self, known: RequestedFactKnownInput) -> bool:
        from fervis.lookup.question_contract.model import (
            RequestedFactLiteralInput,
            RequestedFactRowSetReferenceInput,
        )

        match known:
            case RequestedFactLiteralInput():
                for item in self.inputs:
                    match item:
                        case ResolvedLiteralQuestionInput() if item.accepts(known):
                            return True
                        case _:
                            continue
                return False
            case RequestedFactRowSetReferenceInput():
                for item in self.inputs:
                    match item:
                        case ResolvedRowSetQuestionInput() if item.accepts(known):
                            return True
                        case _:
                            continue
                return False

    def identity_inputs(self) -> tuple[ResolvedIdentityInput, ...]:
        return tuple(
            identity
            for item in self.inputs
            if (identity := item.identity_input()) is not None
        )

    @property
    def uses_prior_context(self) -> bool:
        return (
            self.frame_call is not None
            or any(clause.retained_frame_parts for clause in self.clauses)
            or any(
                source.uses_prior_context()
                for clause in self.clauses
                for value in clause.values
                for source in value.sources
            )
        )


def compile_conversation_resolution(
    resolution: ConversationResolution,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> CompiledConversationResolution:
    if resolution.needs_clarification:
        raise ValueError("an unresolved conversation cannot be compiled")
    parameters_by_value_id = _resolved_value_parameters(
        resolution,
        memory_projection=memory_projection,
    )
    clauses = tuple(
        _compile_clause(
            clause,
            parameters_by_value_id=parameters_by_value_id,
            memory_projection=memory_projection,
        )
        for clause in resolution.clauses
    )
    inputs = tuple(
        compiled_input
        for compiled_value in (value for clause in clauses for value in clause.values)
        if resolution.frame_call is None
        or compiled_value.value_id in parameters_by_value_id
        if (
            compiled_input := _compile_input(
                compiled_value,
                memory_projection=memory_projection,
            )
        )
        is not None
    )
    return CompiledConversationResolution(
        current_question_text=resolution.current_question_text,
        contextualized_question=resolution.contextualized_question,
        clauses=clauses,
        inputs=inputs,
        frame_call=resolution.frame_call,
        used_source_card_ids=resolution.used_source_card_ids,
        used_memory_ids=resolution.used_memory_ids,
    )


def _compile_clause(
    clause: ResolvedConversationClause,
    *,
    parameters_by_value_id: dict[str, ConversationFrameParameter],
    memory_projection: ConversationMemoryCardProjection,
) -> CompiledResolvedClause:
    return CompiledResolvedClause(
        current_clause_text=clause.current_clause_text,
        resolved_text=clause.resolved_text,
        retained_frame_parts=tuple(
            _compile_retained_frame_part(
                retained,
                memory_projection=memory_projection,
            )
            for retained in clause.retained_frame_parts
        ),
        values=tuple(
            _compile_value(
                value,
                parameter=parameters_by_value_id.get(value.value_id),
                memory_projection=memory_projection,
            )
            for value in clause.values
        ),
    )


def _compile_retained_frame_part(
    retained: FramePartSource,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> CompiledRetainedFramePart:
    frame = _frame(retained.frame_id, memory_projection=memory_projection)
    part = _frame_part(
        retained.frame_id,
        retained.part_id,
        memory_projection=memory_projection,
    )
    return CompiledRetainedFramePart(
        kind=part.kind,
        text=part.text,
        answer_shape=(
            frame.answer_shape
            if part.kind is ConversationFramePartKind.ANSWER_OUTPUT
            else None
        ),
    )


def _resolved_value_parameters(
    resolution: ConversationResolution,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> dict[str, ConversationFrameParameter]:
    parameters: dict[str, ConversationFrameParameter] = {}
    for clause in resolution.clauses:
        for value in clause.values:
            reference = value.frame_parameter
            if reference is None:
                continue
            frame = _frame(
                reference.frame_id,
                memory_projection=memory_projection,
            )
            if frame.callable is None:
                raise ValueError("conversation frame is not callable")
            parameters_by_id = {
                parameter.parameter_id: parameter
                for parameter in frame.callable.parameters
            }
            parameters[value.value_id] = parameters_by_id[reference.parameter_id]
    return parameters


def _compile_value(
    value: ResolvedConversationValue,
    *,
    parameter: ConversationFrameParameter | None,
    memory_projection: ConversationMemoryCardProjection,
) -> CompiledResolvedValue:
    return CompiledResolvedValue(
        value_id=value.value_id,
        resolved_text=value.resolved_text,
        source_kinds=_value_source_kinds(
            value,
            parameter_kind=parameter.kind if parameter is not None else None,
            memory_projection=memory_projection,
        ),
        sources=value.sources,
        field_label_text=(parameter.field_label_text if parameter is not None else ""),
        value_meaning_hint=(
            parameter.value_meaning_hint if parameter is not None else ""
        ),
        canonical_identity=_parameter_canonical_identity(parameter),
    )


def _parameter_canonical_identity(
    parameter: ConversationFrameParameter | None,
) -> ResolvedCanonicalIdentity | None:
    if parameter is None or not isinstance(
        parameter.binding,
        PriorEntityIdentityBinding,
    ):
        return None
    components = tuple(parameter.binding.canonical_values.items())
    if len(components) != 1:
        raise ValueError("resolved identity requires one candidate-key component")
    key_component_id, value = components[0]
    return ResolvedCanonicalIdentity(
        entity_kind=parameter.binding.entity_kind,
        key_id=parameter.binding.key_id,
        key_component_id=key_component_id,
        value=value,
        authority_refs=parameter.binding.source_lineage,
        lineage_refs=parameter.binding.source_lineage,
    )


def _compile_input(
    value: CompiledResolvedValue,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> ResolvedQuestionInput | None:
    if value.canonical_identity is not None:
        return ResolvedLiteralQuestionInput(
            input_ref=_input_ref(value.value_id),
            value_source_text=value.resolved_text,
            resolved_value_text=value.resolved_text,
            role=LiteralInputRole.REFERENCE_VALUE,
            field_label_text=value.field_label_text,
            value_meaning_hint=value.value_meaning_hint,
            evidence_refs=value.canonical_identity.lineage_refs,
            canonical_identity=value.canonical_identity,
        )
    memory_ids = tuple(
        dict.fromkeys(
            memory_id
            for source in value.sources
            for memory_id in source.memory_references()
        )
    )
    input_kinds = set(value.source_kinds) & {
        ConversationFramePartKind.ENTITY_IDENTITY.value,
        ConversationFramePartKind.TIME_SCOPE.value,
        ConversationFramePartKind.LIMIT.value,
    }
    row_set_memory_ids = tuple(
        memory_id
        for memory_id in memory_ids
        if _memory_kind(memory_id, memory_projection=memory_projection) == "row_set"
    )
    canonical_identity = _canonical_identity(
        ConversationFramePartKind.ENTITY_IDENTITY,
        memory_ids=row_set_memory_ids,
        memory_projection=memory_projection,
    )
    if canonical_identity is not None:
        return ResolvedLiteralQuestionInput(
            input_ref=_input_ref(value.value_id),
            value_source_text=value.resolved_text,
            resolved_value_text=value.resolved_text,
            role=LiteralInputRole.REFERENCE_VALUE,
            value_meaning_hint=_value_meaning_hint(
                ConversationFramePartKind.ENTITY_IDENTITY,
                declared_hint=value.value_meaning_hint,
                memory_ids=row_set_memory_ids,
                memory_projection=memory_projection,
            ),
            evidence_refs=row_set_memory_ids,
            canonical_identity=canonical_identity,
        )
    if row_set_memory_ids:
        if input_kinds:
            raise ValueError("one resolved value cannot be both a row set and literal")
        return ResolvedRowSetQuestionInput(
            input_ref=_input_ref(value.value_id),
            reference_text=value.resolved_text,
            memory_ids=row_set_memory_ids,
        )
    if not input_kinds:
        return None
    if len(input_kinds) != 1:
        raise ValueError("resolved value has conflicting input meanings")
    part_kind = ConversationFramePartKind(next(iter(input_kinds)))
    return ResolvedLiteralQuestionInput(
        input_ref=_input_ref(value.value_id),
        value_source_text=value.resolved_text,
        resolved_value_text=value.resolved_text,
        role=_literal_role(part_kind),
        field_label_text=value.field_label_text,
        value_meaning_hint=_value_meaning_hint(
            part_kind,
            declared_hint=value.value_meaning_hint,
            memory_ids=memory_ids,
            memory_projection=memory_projection,
        ),
        evidence_refs=memory_ids,
        canonical_identity=_canonical_identity(
            part_kind,
            memory_ids=memory_ids,
            memory_projection=memory_projection,
        ),
    )


def _value_source_kinds(
    value: ResolvedConversationValue,
    *,
    parameter_kind: ConversationFramePartKind | None,
    memory_projection: ConversationMemoryCardProjection,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            kind
            for kind in (
                *(
                    _memory_kind(memory_id, memory_projection=memory_projection)
                    for source in value.sources
                    for memory_id in source.memory_references()
                ),
                *(
                    _frame_part(
                        frame_id,
                        part_id,
                        memory_projection=memory_projection,
                    ).kind.value
                    for source in value.sources
                    for frame_id, part_id in source.frame_part_references()
                ),
                parameter_kind.value if parameter_kind is not None else "",
            )
            if kind
        )
    )


def _input_ref(value_id: str) -> str:
    return f"conversation.{value_id}"


def _frame(
    frame_id: str,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> ConversationContextFrame:
    try:
        return memory_projection.frame(frame_id)
    except KeyError as exc:
        raise ValueError(
            "conversation resolution references an unavailable frame"
        ) from exc


def _frame_part(
    frame_id: str,
    part_id: str,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> ConversationFramePart:
    frame = _frame(frame_id, memory_projection=memory_projection)
    for part in frame.parts:
        if part.part_id == part_id:
            return part
    raise ValueError("conversation resolution references an unavailable frame part")


def _memory_kind(
    memory_id: str,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> str:
    return str(memory_projection.private_card(memory_id).get("kind") or "").strip()


def _literal_role(kind: ConversationFramePartKind) -> LiteralInputRole:
    roles = {
        ConversationFramePartKind.ENTITY_IDENTITY: LiteralInputRole.REFERENCE_VALUE,
        ConversationFramePartKind.TIME_SCOPE: LiteralInputRole.TIME_VALUE,
        ConversationFramePartKind.LIMIT: LiteralInputRole.RESULT_LIMIT,
    }
    return roles[kind]


def _value_meaning_hint(
    kind: ConversationFramePartKind,
    *,
    declared_hint: str,
    memory_ids: tuple[str, ...],
    memory_projection: ConversationMemoryCardProjection,
) -> str:
    if declared_hint:
        return declared_hint
    if kind is ConversationFramePartKind.TIME_SCOPE:
        return "time scope"
    if kind is ConversationFramePartKind.LIMIT:
        return "result limit"
    entity_kinds = tuple(
        dict.fromkeys(
            entity_kind
            for memory_id in memory_ids
            if (
                entity_kind := _private_entity_kind(
                    memory_projection.private_card(memory_id)
                )
            )
        )
    )
    return f"{entity_kinds[0]} identity" if len(entity_kinds) == 1 else ""


def _private_entity_kind(private_card: dict[str, object]) -> str:
    entity_key = private_card.get("entity_key")
    if not isinstance(entity_key, dict):
        return ""
    return str(entity_key.get("entity_kind") or "").strip()


def _canonical_identity(
    kind: ConversationFramePartKind,
    *,
    memory_ids: tuple[str, ...],
    memory_projection: ConversationMemoryCardProjection,
) -> ResolvedCanonicalIdentity | None:
    if kind is not ConversationFramePartKind.ENTITY_IDENTITY:
        return None
    candidates: list[ResolvedCanonicalIdentity] = []
    for memory_id in memory_ids:
        private = memory_projection.private_card(memory_id)
        entity_key = private.get("entity_key")
        if not isinstance(entity_key, dict):
            continue
        components = entity_key.get("components")
        if not isinstance(components, dict) or len(components) != 1:
            continue
        key_component_id, value = next(iter(components.items()))
        entity_kind = str(entity_key.get("entity_kind") or "").strip()
        key_id = str(entity_key.get("key_id") or "").strip()
        authority_refs = tuple(
            _prior_authority_ref(ref)
            for raw_ref in private.get("proof_refs") or ()
            if (ref := str(raw_ref).strip())
            and (ref.startswith("source_read:") or ref.startswith("prior_source_read:"))
        )
        if not entity_kind or not key_id or not authority_refs:
            continue
        candidates.append(
            ResolvedCanonicalIdentity(
                entity_kind=entity_kind,
                key_id=key_id,
                key_component_id=str(key_component_id),
                value=str(value),
                authority_refs=authority_refs,
                lineage_refs=(f"memory:{memory_id}",),
            )
        )
    if len(candidates) > 1 and len(set(candidates)) != 1:
        raise ValueError("resolved value has conflicting canonical identities")
    return candidates[0] if candidates else None


def _prior_authority_ref(proof_ref: str) -> str:
    if proof_ref.startswith("source_read:"):
        return f"prior_source_read:{proof_ref.removeprefix('source_read:')}"
    return proof_ref
