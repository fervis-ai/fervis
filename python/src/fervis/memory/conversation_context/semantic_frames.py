"""Semantic prior-request frames projected from canonical program invocations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fervis.lookup.answer_program.persistence import StoredProgramInvocation
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.contract_codec import canonical_contract_fingerprint
from fervis.lookup.question_contract import (
    AssociationTerm,
    ExpressionNode,
    FactTerm,
    RequestedFact,
    SetTerm,
    analyze_requested_fact,
)
from fervis.lookup.question_contract.semantic_model import (
    Aggregate,
    AggregateFunction,
    AllResults,
    FirstRankWithTies,
    RequestedOutput,
    TakeWithBoundaryTies,
)
from fervis.lookup.semantic_types import IdentifierType, ValueType, value_type_kind
from fervis.memory.artifacts import FactArtifact, FactOutcome

from .model import (
    ConversationCallableParameter,
    ConversationCallableSignature,
    ConversationContextFrame,
    ConversationFramePart,
    ConversationFramePartKind,
)


@dataclass(frozen=True)
class PriorRequestFrame:
    memory_id: str
    artifact_id: str
    run_id: str
    requested_fact_ref: str
    display: str
    frame: ConversationContextFrame


@dataclass(frozen=True)
class _SemanticFrameProjection:
    parts: tuple[ConversationFramePart, ...]
    callable: ConversationCallableSignature | None


def prior_request_frames(
    artifacts: tuple[FactArtifact, ...],
    *,
    invocations_by_run_id: dict[str, StoredProgramInvocation],
) -> tuple[PriorRequestFrame, ...]:
    """Project one prior request per answered run/fact pair."""

    output: list[PriorRequestFrame] = []
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        run_id = str(artifact.provenance.get("runId") or "").strip()
        requested_fact_ref = str(
            artifact.provenance.get("requestedFactKey") or ""
        ).strip()
        key = (run_id, requested_fact_ref)
        if (
            artifact.outcome is not FactOutcome.ANSWERED
            or not all(key)
            or key in seen
        ):
            continue
        stored = invocations_by_run_id.get(run_id)
        if stored is None:
            continue
        fact = next(
            (
                item
                for item in stored.program.fact_template
                if item.id == requested_fact_ref
            ),
            None,
        )
        if fact is None:
            raise ValueError("memory artifact references an unavailable program fact")
        seen.add(key)
        memory_id = f"prior_request:{run_id}:{requested_fact_ref}"
        projection = _semantic_frame_projection(fact, stored=stored)
        output.append(
            PriorRequestFrame(
                memory_id=memory_id,
                artifact_id=artifact.artifact_id,
                run_id=run_id,
                requested_fact_ref=requested_fact_ref,
                display=artifact.source_question.strip() or fact.origin.meaning,
                frame=ConversationContextFrame(
                    frame_id=memory_id,
                    source_ids=(artifact.artifact_id,),
                    parts=projection.parts,
                    callable=projection.callable,
                ),
            )
        )
    return tuple(output)


def _semantic_frame_projection(
    fact: RequestedFact,
    *,
    stored: StoredProgramInvocation,
) -> _SemanticFrameProjection:
    analysis = _semantic_analysis(fact, stored=stored)
    types_by_local_ref = {
        local_id: analysis.inferred_type_by_ref[ref]
        for local_id, ref in analysis.fact_local_ref_by_local_id.items()
        if ref in analysis.inferred_type_by_ref
    }
    values = {
        item.id: item
        for item in (*fact.sets, *fact.associations, *fact.facts, *fact.expressions)
    }
    input_parts = _input_parts(
        stored=stored,
        used_input_refs={item.input_ref for item in analysis.input_use_sites},
    )
    parts: list[ConversationFramePart] = [
        ConversationFramePart(
            part_id="subject",
            kind=ConversationFramePartKind.SUBJECT,
            text=next(item for item in fact.sets if item.id == fact.subject.set_ref).origin.meaning,
            source_ref=fact.subject.set_ref,
        )
    ]
    if fact.qualification_ref is not None:
        parts.append(
            _value_part(
                "qualification",
                ConversationFramePartKind.QUALIFICATION,
                fact.qualification_ref,
                types_by_local_ref=types_by_local_ref,
                values=values,
            )
        )
    parts.extend(
        _value_part(
            f"grouping:{index}",
            ConversationFramePartKind.GROUPING,
            ref,
            types_by_local_ref=types_by_local_ref,
            values=values,
        )
        for index, ref in enumerate(fact.grouping_refs, start=1)
    )
    parts.extend(
        _requested_output_part(
            fact,
            output,
            index=index,
            types_by_local_ref=types_by_local_ref,
            values=values,
        )
        for index, output in enumerate(fact.outputs, start=1)
    )
    parts.extend(
        ConversationFramePart(
            part_id=f"ordering:{index}",
            kind=ConversationFramePartKind.ORDERING,
            text=ordering.origin.meaning,
            source_ref=ordering.expression_ref,
            value_type=value_type_kind(types_by_local_ref[ordering.expression_ref]),
        )
        for index, ordering in enumerate(fact.ordering, start=1)
    )
    parts.append(
        ConversationFramePart(
            part_id="selection",
            kind=ConversationFramePartKind.SELECTION,
            text=_selection_text(fact),
        )
    )
    parts.extend(_canonical_output_identity_parts(fact, values=values))
    parts.extend(input_parts)
    return _SemanticFrameProjection(
        parts=tuple(parts),
        callable=(
            _callable_signature(
                fact,
                stored=stored,
                input_parts=input_parts,
            )
            if len(stored.program.fact_template) == 1
            else None
        ),
    )


def _requested_output_part(
    fact: RequestedFact,
    output: RequestedOutput,
    *,
    index: int,
    types_by_local_ref: dict[str, ValueType],
    values: Mapping[str, SetTerm | AssociationTerm | FactTerm | ExpressionNode],
) -> ConversationFramePart:
    expression = values[output.expression_ref]
    text = output.origin.meaning
    if (
        isinstance(expression, Aggregate)
        and expression.function is AggregateFunction.COUNT
        and expression.argument_ref == fact.subject.set_ref
    ):
        text = "row count"
    return ConversationFramePart(
        part_id=f"output:{index}",
        kind=ConversationFramePartKind.REQUESTED_OUTPUT,
        text=text,
        source_ref=output.id,
        value_type=value_type_kind(types_by_local_ref[output.expression_ref]),
    )


def _value_part(
    part_id: str,
    kind: ConversationFramePartKind,
    ref: str,
    *,
    types_by_local_ref: dict[str, ValueType],
    values: Mapping[str, SetTerm | AssociationTerm | FactTerm | ExpressionNode],
) -> ConversationFramePart:
    item = values[ref]
    return ConversationFramePart(
        part_id=part_id,
        kind=kind,
        text=item.origin.meaning,
        source_ref=ref,
        value_type=value_type_kind(types_by_local_ref[ref]),
    )


def _canonical_output_identity_parts(
    fact: RequestedFact,
    *,
    values: Mapping[str, SetTerm | AssociationTerm | FactTerm | ExpressionNode],
) -> tuple[ConversationFramePart, ...]:
    output: list[ConversationFramePart] = []
    for index, requested_output in enumerate(fact.outputs, start=1):
        item = values.get(requested_output.expression_ref)
        if not isinstance(item, FactTerm) or not isinstance(item.value_type, IdentifierType):
            continue
        output.append(
            ConversationFramePart(
                part_id=f"canonical_output_identity:{index}",
                kind=ConversationFramePartKind.CANONICAL_OUTPUT_IDENTITY,
                text=requested_output.origin.meaning,
                source_ref=requested_output.id,
                value_type=value_type_kind(item.value_type),
            )
        )
    return tuple(output)


def _input_parts(
    *,
    stored: StoredProgramInvocation,
    used_input_refs: set[str],
) -> tuple[ConversationFramePart, ...]:
    inputs = {item.id: item for item in stored.program.inputs}
    return tuple(
        ConversationFramePart(
            part_id=f"input:{parameter.id}",
            kind=ConversationFramePartKind.INPUT,
            text=_fact_value_text(binding.value),
            source_ref=parameter.input_ref,
            value_type=parameter.value_type.value,
        )
        for parameter in stored.program.parameters
        if parameter.input_ref in used_input_refs
        if (binding := stored.bindings.get(parameter.id)) is not None
        if parameter.input_ref in inputs
    )


def _semantic_analysis(
    fact: RequestedFact,
    *,
    stored: StoredProgramInvocation,
):
    return analyze_requested_fact(
        fact,
        inputs={item.id: item for item in stored.program.inputs},
        input_denotations={
            item.input_ref: item for item in stored.program.input_denotations
        },
    )


def _callable_signature(
    fact: RequestedFact,
    *,
    stored: StoredProgramInvocation,
    input_parts: tuple[ConversationFramePart, ...],
) -> ConversationCallableSignature:
    parts_by_input_ref = {part.source_ref: part for part in input_parts}
    parameters = tuple(
        ConversationCallableParameter(
            parameter_id=parameter.id,
            part_id=parts_by_input_ref[parameter.input_ref].part_id,
            value_type=parameter.value_type.value,
            input_ref=parameter.input_ref,
            input_use_refs=parameter.input_use_refs,
            current_text=_fact_value_text(binding.value),
        )
        for parameter in stored.program.parameters
        if parameter.input_ref in parts_by_input_ref
        if (binding := stored.bindings.get(parameter.id)) is not None
    )
    return ConversationCallableSignature(
        base_invocation_id=stored.invocation.invocation_id,
        program_id=stored.invocation.program_id,
        requested_fact_ref=fact.id,
        requested_fact_fingerprint=canonical_contract_fingerprint(fact),
        parameters=parameters,
    )


def _fact_value_text(value: FactValue) -> str:
    display = str(getattr(value.payload, "display_value", "") or "").strip()
    if display:
        return display
    return str(value.payload.canonical_value())


def _selection_text(fact: RequestedFact) -> str:
    match fact.selection:
        case AllResults():
            return "all qualifying results"
        case FirstRankWithTies():
            return "first ranked results, including ties"
        case TakeWithBoundaryTies(limit_input_ref=limit):
            return f"the requested number of ranked results, including boundary ties ({limit})"
    raise TypeError("unsupported result selection")


__all__ = ["PriorRequestFrame", "prior_request_frames"]
