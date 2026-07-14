"""Project prior fact artifacts into Lookup-readable memory relations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.canonical_data import (
    EntityKeyValue,
    RuntimeValue,
    parse_runtime_value,
)

from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessSourceKind,
    CompletenessStatus,
    PaginationCompleteness,
    RelationRows,
    RelationSetKind,
)
from fervis.memory.prior_requests import (
    PriorRequestMemory,
    PriorRequestSlot,
    prior_requests_from_artifact,
)
from fervis.memory.artifacts import FactArtifact, FactOutcome
from fervis.memory.addresses import FactAddress, FactAddressKind, FactAddressValue
from fervis.memory.identities import (
    MemoryIdentitySet,
    MemoryIdentityValue,
    project_memory_identity_values,
)
from fervis.memory.projection import fact_artifacts_from_context
from fervis.memory.conversation_context import (
    ConversationAnswerShape,
    ConversationCallableSignature,
    ConversationContextFrame,
    ConversationContextSource,
    ConversationFrameParameter,
    ConversationFramePart,
    ConversationFramePartKind,
    ConversationMeaningAnchor,
    ConversationMemoryActivation,
    ConversationMemoryActivationKind,
    ConversationMemoryCard,
    ConversationMemoryCardProjection,
)

_BACKING_MEMORY_CARDS = "backing_cards"


class ConversationMemoryProjectionOverflow(ValueError):
    """Raised when required conversation memory cannot fit in the prompt budget."""


@dataclass(frozen=True)
class MemoryValue:
    id: str
    value: object
    value_type: str = ""
    proof_refs: tuple[str, ...] = ()
    source_relation_id: str = ""
    source_row_id: str = ""
    source_row_grain: dict[str, object] | None = None
    source_field_id: str = ""
    answer_output_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LookupMemory:
    values: tuple[MemoryValue, ...] = ()
    identity_values: tuple[MemoryIdentityValue, ...] = ()
    identity_sets: tuple[MemoryIdentitySet, ...] = ()
    relations: tuple[RelationRows, ...] = ()
    prompt_context: dict[str, Any] | None = None

    def relation(self, ref: str) -> RelationRows:
        for relation in self.relations:
            if relation.id == ref:
                return relation
        raise KeyError(ref)


@dataclass
class _ContextSourceGroup:
    card_ids: list[str]
    memory_ids: list[str]
    meaning_anchors: list[ConversationMeaningAnchor]


def project_lookup_memory(
    conversation_context: dict[str, Any],
) -> LookupMemory:
    artifacts = fact_artifacts_from_context(conversation_context)
    identity_projection = project_memory_identity_values(artifacts)
    values: list[MemoryValue] = []
    relations: list[RelationRows] = []
    relation_prompts: list[dict[str, Any]] = []
    outcome_prompts: list[dict[str, Any]] = []
    for artifact in artifacts:
        projected = _project_artifact_memory(artifact)
        values.extend(projected.values)
        relations.extend(projected.relations)
        relation_prompts.extend(projected.relation_prompts)
        outcome_prompts.extend(projected.outcome_prompts)
    return LookupMemory(
        values=tuple(values),
        identity_values=identity_projection.identity_values,
        identity_sets=identity_projection.identity_sets,
        relations=tuple(relations),
        prompt_context=_prompt_context(
            values=tuple(values),
            relation_prompts=tuple(relation_prompts),
            outcome_prompts=tuple(outcome_prompts),
        ),
    )


@dataclass(frozen=True)
class _ProjectedArtifactMemory:
    values: tuple[MemoryValue, ...] = ()
    relations: tuple[RelationRows, ...] = ()
    relation_prompts: tuple[dict[str, Any], ...] = ()
    outcome_prompts: tuple[dict[str, Any], ...] = ()


def _project_artifact_memory(artifact: Any) -> _ProjectedArtifactMemory:
    rows_by_address = {
        address.address: address
        for address in artifact.addresses
        if address.kind == FactAddressKind.ROW
    }
    values: list[MemoryValue] = []
    relations: list[RelationRows] = []
    relation_prompts: list[dict[str, Any]] = []
    outcome_prompts: list[dict[str, Any]] = []
    for address in artifact.addresses:
        if address.kind == FactAddressKind.OUTCOME:
            outcome_prompts.append(
                _memory_outcome_prompt(artifact=artifact, address=address)
            )
        elif address.kind == FactAddressKind.VALUE:
            values.append(_memory_scalar_value(artifact=artifact, address=address))
        elif address.kind == FactAddressKind.RELATION:
            projected_relation = _project_memory_relation(
                address,
                artifact_id=artifact.artifact_id,
                rows_by_address=rows_by_address,
            )
            values.extend(projected_relation.values)
            relations.append(projected_relation.relation)
            relation_prompts.append(projected_relation.prompt)
    return _ProjectedArtifactMemory(
        values=tuple(values),
        relations=tuple(relations),
        relation_prompts=tuple(relation_prompts),
        outcome_prompts=tuple(outcome_prompts),
    )


def _memory_scalar_value(*, artifact: Any, address: Any) -> MemoryValue:
    return MemoryValue(
        id=f"{artifact.artifact_id}.{address.address}",
        value=address.scalar_value.get("value", ""),
        value_type=str(address.scalar_value.get("type") or ""),
        proof_refs=tuple(address.evidence.step_ids if address.evidence else ()),
    )


@dataclass(frozen=True)
class _ProjectedMemoryRelation:
    relation: RelationRows
    values: tuple[MemoryValue, ...]
    prompt: dict[str, Any]


def _project_memory_relation(
    address: Any,
    *,
    artifact_id: str,
    rows_by_address: dict[str, Any],
) -> _ProjectedMemoryRelation:
    relation_id = f"{artifact_id}.{address.address}"
    relation_rows = _relation_rows(address, rows_by_address=rows_by_address)
    rows = tuple(_row_values(row) for row in relation_rows)
    completeness = _memory_completeness(address)
    return _ProjectedMemoryRelation(
        relation=RelationRows(
            id=relation_id,
            rows=rows,
            grain_keys=tuple(address.grain_keys),
            field_types={
                field_id: _memory_field_type(field_id, rows=relation_rows)
                for field_id in _memory_field_ids(
                    address=address,
                    rows=relation_rows,
                )
            },
            field_answer_output_ids={
                field_id: output_ids
                for field_id in _memory_field_ids(
                    address=address,
                    rows=relation_rows,
                )
                for output_ids in (
                    _memory_field_answer_output_ids(field_id, relation_rows),
                )
                if output_ids
            },
            completeness=completeness,
        ),
        values=_relation_cell_values(
            relation_id=relation_id,
            address=address,
            rows=relation_rows,
            relation_proof_refs=tuple(completeness.proof_refs),
        ),
        prompt=_memory_relation_prompt(
            relation_id=relation_id,
            address=address,
            rows=relation_rows,
            completeness=completeness,
        ),
    )


def project_conversation_memory_cards(
    conversation_context: dict[str, Any],
    *,
    current_question: str,
    max_cards: int = 12,
) -> ConversationMemoryCardProjection:
    artifacts = fact_artifacts_from_context(conversation_context)
    prior_requests_by_artifact_id = {
        artifact.artifact_id: prior_requests_from_artifact(artifact)
        for artifact in artifacts
    }
    prior_requests = tuple(
        request
        for requests in prior_requests_by_artifact_id.values()
        for request in requests
    )
    ranked = _ranked_memory_card_records(
        artifacts,
        current_question=current_question,
        prior_requests_by_artifact_id=prior_requests_by_artifact_id,
    )
    must_include_ids = _must_include_memory_ids(artifacts)
    must_include = tuple(
        record for record in ranked if _record_has_memory_id(record, must_include_ids)
    )
    if len(must_include) > max_cards:
        raise ConversationMemoryProjectionOverflow(
            "required conversation memory exceeds projection budget"
        )
    quota = tuple(
        record
        for record in ranked
        if not _record_has_memory_id(record, must_include_ids)
    )
    visible = must_include + quota[: max_cards - len(must_include)]
    visible_memory_ids = {
        memory_id for record in visible for memory_id in _record_memory_ids(record)
    }
    omitted = tuple(
        record
        for record in ranked
        if not _record_has_memory_id(record, visible_memory_ids)
    )
    omitted_counts: dict[str, int] = {}
    for card, _private in omitted:
        omitted_counts[card.kind] = omitted_counts.get(card.kind, 0) + 1
    visible_cards = tuple(
        _with_prompt_card_id(card, index=index)
        for index, (card, _private) in enumerate(visible, start=1)
    )
    private_cards = _private_cards_by_memory_id(ranked)
    context_sources = _context_sources(
        cards=visible_cards,
        private_cards=private_cards,
        artifacts_by_id={
            str(getattr(artifact, "artifact_id", "") or ""): artifact
            for artifact in artifacts
        },
    )
    return ConversationMemoryCardProjection(
        cards=visible_cards,
        activations=_memory_activations(
            artifacts,
            prior_requests_by_artifact_id=prior_requests_by_artifact_id,
        ),
        context_sources=context_sources,
        context_frames=_context_frames(
            cards=visible_cards,
            context_sources=context_sources,
            prior_requests_by_memory_id={
                request.memory_id: request for request in prior_requests
            },
        ),
        prior_requests=prior_requests,
        private_cards=private_cards,
        omitted_counts_by_kind=omitted_counts,
    )


def _memory_activations(
    artifacts: tuple[FactArtifact, ...],
    *,
    prior_requests_by_artifact_id: dict[str, tuple[PriorRequestMemory, ...]],
) -> tuple[ConversationMemoryActivation, ...]:
    activations: list[ConversationMemoryActivation] = []
    for artifact in artifacts:
        for prior_request in prior_requests_by_artifact_id[artifact.artifact_id]:
            card, _private = _prior_request_memory_card(
                artifact=artifact,
                prior_request=prior_request,
                use_fact_scoped_display=(
                    len(prior_requests_by_artifact_id[artifact.artifact_id]) > 1
                ),
            )
            activations.append(
                ConversationMemoryActivation(
                    card=card,
                    kind=ConversationMemoryActivationKind.PRIOR_REQUEST,
                    artifact_id=artifact.artifact_id,
                    prior_request=prior_request,
                )
            )
        for address in artifact.addresses:
            projected = _memory_card_for_address(artifact=artifact, address=address)
            if projected is None:
                continue
            card, _private = projected
            activations.append(
                ConversationMemoryActivation(
                    card=card,
                    kind=ConversationMemoryActivationKind(card.kind),
                    artifact_id=artifact.artifact_id,
                    address_id=address.address,
                )
            )
    return tuple(activations)


def _must_include_memory_ids(artifacts: tuple[Any, ...]) -> frozenset[str]:
    if not artifacts:
        return frozenset()
    output: set[str] = set()
    for artifact in _must_include_artifacts(artifacts):
        output.update(_direct_artifact_memory_ids(artifact))
        output.update(_activated_memory_ids_from_artifact(artifact))
    return frozenset(output)


def _context_sources(
    *,
    cards: tuple[ConversationMemoryCard, ...],
    private_cards: dict[str, dict[str, Any]],
    artifacts_by_id: dict[str, Any],
) -> tuple[ConversationContextSource, ...]:
    grouped: dict[tuple[str, str], _ContextSourceGroup] = {}
    for card in cards:
        private = private_cards.get(card.memory_id) or {}
        artifact_id = str(private.get("artifact_id") or "").strip()
        artifact = artifacts_by_id.get(artifact_id)
        source_memory_ids = _context_source_memory_ids(card=card, private=private)
        for kind, text in _artifact_context_source_texts(
            artifact=artifact,
            card=card,
            private=private,
        ):
            key = (kind, text)
            item = grouped.setdefault(
                key,
                _ContextSourceGroup(
                    card_ids=[],
                    memory_ids=[],
                    meaning_anchors=[],
                ),
            )
            _append_unique(item.card_ids, card.card_id)
            for memory_id in source_memory_ids:
                _append_unique(item.memory_ids, memory_id)
            for anchor in _meaning_anchors_for_source(
                source_kind=kind,
                text=text,
                source_memory_ids=source_memory_ids,
                private_cards=private_cards,
            ):
                if anchor not in item.meaning_anchors:
                    item.meaning_anchors.append(anchor)
    return tuple(
        ConversationContextSource(
            source_id=f"prior_{index}",
            kind=kind,
            text=text,
            source_card_ids=tuple(item.card_ids),
            source_memory_ids=tuple(item.memory_ids),
            meaning_anchors=tuple(item.meaning_anchors),
        )
        for index, ((kind, text), item) in enumerate(grouped.items(), start=1)
    )


def _append_unique(output: list[str], value: str) -> None:
    if value and value not in output:
        output.append(value)


def _meaning_anchors_for_source(
    *,
    source_kind: str,
    text: str,
    source_memory_ids: tuple[str, ...],
    private_cards: dict[str, dict[str, Any]],
) -> tuple[ConversationMeaningAnchor, ...]:
    output: list[ConversationMeaningAnchor] = []
    for memory_id in source_memory_ids:
        private = private_cards.get(memory_id) or {}
        kind = str(private.get("kind") or "").strip()
        if kind not in {
            "entity_identity",
            "time_scope",
            "scalar_value",
            "row_set",
        }:
            continue
        for anchor_text in _meaning_anchor_text_candidates(private):
            occurrence = _first_occurrence(anchor_text, text)
            if occurrence < 1:
                continue
            anchor = ConversationMeaningAnchor(
                anchor_id=memory_id,
                text=anchor_text,
                occurrence=occurrence,
                kind=kind,
                label=_meaning_anchor_label(kind=kind, private=private),
            )
            if anchor not in output:
                output.append(anchor)
            break
        else:
            if (
                kind == "row_set"
                and source_kind == "prior_fervis_answer"
                and text.strip()
            ):
                anchor = ConversationMeaningAnchor(
                    anchor_id=memory_id,
                    text=text.strip(),
                    occurrence=1,
                    kind=kind,
                    label=_meaning_anchor_label(kind=kind, private=private),
                )
                if anchor not in output:
                    output.append(anchor)
    return tuple(output)


def _meaning_anchor_text_candidates(private: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for raw in (
        private.get("reference_text"),
        private.get("question_being_clarified"),
        private.get("clarification_question"),
        private.get("display"),
        private.get("expression"),
        private.get("value"),
    ):
        value = str(raw or "").strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _first_occurrence(needle: str, haystack: str) -> int:
    needle = str(needle or "").strip()
    haystack = str(haystack or "")
    if not needle:
        return 0
    start = haystack.find(needle)
    if start < 0:
        return 0
    return haystack[:start].count(needle) + 1


def _meaning_anchor_label(*, kind: str, private: dict[str, Any]) -> str:
    if kind == "entity_identity":
        entity_key = private.get("entity_key")
        if isinstance(entity_key, dict):
            entity_kind = str(entity_key.get("entity_kind") or "").strip()
            if entity_kind:
                return f"{entity_kind} identity"
    if kind == "time_scope":
        return "time scope"
    if kind == "row_set":
        return "row set"
    if kind == "scalar_value":
        return "scalar value"
    if kind == "prior_answer_request":
        return "prior answer request"
    return kind.replace("_", " ")


def _context_frames(
    *,
    cards: tuple[ConversationMemoryCard, ...],
    context_sources: tuple[ConversationContextSource, ...],
    prior_requests_by_memory_id: dict[str, PriorRequestMemory],
) -> tuple[ConversationContextFrame, ...]:
    output: list[ConversationContextFrame] = []
    seen: set[tuple[object, ...]] = set()
    for card in cards:
        prior_request = prior_requests_by_memory_id.get(card.memory_id)
        if prior_request is None:
            continue
        if prior_request.answer_shape is None:
            continue
        source_ids = _context_frame_source_ids(
            card=card,
            context_sources=context_sources,
        )
        if not source_ids:
            continue
        answer_shape = ConversationAnswerShape(
            expression_family=prior_request.answer_shape.expression_family,
            output_roles=prior_request.answer_shape.output_roles,
        )
        parts = _context_frame_parts(prior_request)
        candidate = ConversationContextFrame(
            frame_id="candidate",
            source_ids=source_ids,
            answer_shape=answer_shape,
            parts=parts,
            callable=_callable_signature(
                prior_request,
                parts=parts,
            ),
        )
        key = (frozenset(source_ids), *candidate.control_key())
        if key in seen:
            continue
        seen.add(key)
        output.append(replace(candidate, frame_id=f"request:{len(output) + 1}"))
    return tuple(output)


def _context_frame_parts(
    prior_request: PriorRequestMemory,
) -> tuple[ConversationFramePart, ...]:
    parts: list[ConversationFramePart] = []
    answer_subject = prior_request.answer_subject_text
    if answer_subject:
        parts.append(
            ConversationFramePart(
                part_id="subject",
                kind=ConversationFramePartKind.ANSWER_SUBJECT,
                text=answer_subject,
            )
        )
    for index, output in enumerate(prior_request.output_frames, start=1):
        output_text = "row count" if output.role == "ROW_COUNT" else output.description
        parts.append(
            ConversationFramePart(
                part_id=f"output:{index}",
                kind=ConversationFramePartKind.ANSWER_OUTPUT,
                text=output_text,
                source_ref=output.output_id,
            )
        )
    parts.extend(_input_frame_parts(prior_request.slots))
    role_counts: dict[tuple[str, str], int] = {}
    for part in prior_request.semantic_parts:
        count_key = (part.kind.value, part.role)
        role_counts[count_key] = role_counts.get(count_key, 0) + 1
        index = role_counts[count_key]
        if part.kind.value == "grouping":
            part_id = f"grouping:{index}"
        else:
            part_id = f"population:{part.role}:{index}"
        parts.append(
            ConversationFramePart(
                part_id=part_id,
                kind=ConversationFramePartKind(part.kind.value),
                text=part.text,
            )
        )
    return tuple(parts)


def _input_frame_parts(
    slots: tuple[PriorRequestSlot, ...],
) -> tuple[ConversationFramePart, ...]:
    kind_counts: dict[str, int] = {}
    parts: list[ConversationFramePart] = []
    for slot in slots:
        kind_counts[slot.kind.value] = kind_counts.get(slot.kind.value, 0) + 1
        parts.append(
            ConversationFramePart(
                part_id=f"input:{slot.kind.value}:{kind_counts[slot.kind.value]}",
                kind=ConversationFramePartKind(slot.kind.value),
                text=slot.text,
                source_ref=slot.slot_id,
            )
        )
    return tuple(parts)


def _callable_signature(
    prior_request: PriorRequestMemory,
    *,
    parts: tuple[ConversationFramePart, ...],
) -> ConversationCallableSignature | None:
    if not prior_request.run_id or prior_request.program_request_ids != (
        prior_request.request_id,
    ):
        return None
    parts_by_source_ref = {part.source_ref: part for part in parts if part.source_ref}
    parameters = tuple(
        ConversationFrameParameter(
            parameter_id=f"question.{slot.slot_id}",
            part_id=parts_by_source_ref[slot.slot_id].part_id,
            kind=ConversationFramePartKind(slot.kind.value),
            current_text=slot.text,
            resolved_text=slot.resolved_value_text,
            field_label_text=slot.field_label_text,
            value_meaning_hint=slot.value_meaning_hint,
            binding=prior_request.binding(slot.slot_id),
        )
        for slot in prior_request.slots
    )
    return ConversationCallableSignature(
        base_run_id=prior_request.run_id,
        requested_fact_id=prior_request.request_id,
        parameters=parameters,
    )


def _context_frame_source_ids(
    *,
    card: ConversationMemoryCard,
    context_sources: tuple[ConversationContextSource, ...],
) -> tuple[str, ...]:
    return tuple(
        source.source_id
        for source in context_sources
        if card.card_id in source.source_card_ids
    )


def _context_source_memory_ids(
    *,
    card: ConversationMemoryCard,
    private: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            memory_id
            for memory_id in (
                card.memory_id,
                *(
                    _private_memory_id(backing)
                    for backing in private.get(_BACKING_MEMORY_CARDS) or ()
                    if isinstance(backing, dict)
                ),
            )
            if memory_id
        )
    )


def _artifact_context_source_texts(
    *,
    artifact: Any,
    card: ConversationMemoryCard,
    private: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    output: list[tuple[str, str]] = []
    source_question = str(getattr(artifact, "source_question", "") or "").strip()
    if source_question:
        output.append(("prior_user_question", source_question))
    source_answer = str(getattr(artifact, "source_answer", "") or "").strip()
    if source_answer:
        output.append(("prior_fervis_answer", source_answer))
    return tuple(output)


def _record_has_memory_id(
    record: tuple[ConversationMemoryCard, dict[str, Any]],
    memory_ids: Collection[str],
) -> bool:
    return any(memory_id in memory_ids for memory_id in _record_memory_ids(record))


def _record_memory_ids(
    record: tuple[ConversationMemoryCard, dict[str, Any]],
) -> tuple[str, ...]:
    card, private = record
    output: list[str] = [card.memory_id]
    for backing in private.get(_BACKING_MEMORY_CARDS) or ():
        if not isinstance(backing, dict):
            continue
        memory_id = _private_memory_id(backing)
        if memory_id:
            output.append(memory_id)
    return tuple(dict.fromkeys(output))


def _must_include_artifacts(artifacts: tuple[Any, ...]) -> tuple[Any, ...]:
    return (artifacts[-1],)


def _direct_artifact_memory_ids(artifact: FactArtifact) -> tuple[str, ...]:
    output: list[str] = []
    artifact_id = artifact.artifact_id
    if artifact.outcome is FactOutcome.ANSWERED:
        output.extend(
            request.memory_id for request in prior_requests_from_artifact(artifact)
        )
    for address in artifact.addresses:
        if _memory_card_for_address(artifact=artifact, address=address) is None:
            continue
        output.append(f"{artifact_id}.{address.address}")
    return tuple(output)


def _activated_memory_ids_from_artifact(artifact: Any) -> tuple[str, ...]:
    provenance = getattr(artifact, "provenance", {}) or {}
    if not isinstance(provenance, dict):
        return ()
    activation = provenance.get("conversation_resolution_activation")
    if not isinstance(activation, dict):
        return ()
    return tuple(
        memory_id
        for raw_memory_id in activation.get("activated_memory_ids") or ()
        if (memory_id := str(raw_memory_id or "").strip())
    )


def _with_prompt_card_id(
    card: ConversationMemoryCard,
    *,
    index: int,
) -> ConversationMemoryCard:
    return ConversationMemoryCard(
        card_id=f"card_{index}",
        memory_id=card.memory_id,
        kind=card.kind,
        display=card.display,
        details=card.details,
    )


def _private_cards_by_memory_id(
    ranked: tuple[tuple[ConversationMemoryCard, dict[str, Any]], ...],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for card, private in ranked:
        output[card.memory_id] = private
        for backing in private.get(_BACKING_MEMORY_CARDS) or ():
            if not isinstance(backing, dict):
                continue
            memory_id = _private_memory_id(backing)
            if memory_id:
                output.setdefault(memory_id, backing)
    return output


def _private_memory_id(private: dict[str, Any]) -> str:
    artifact_id = str(private.get("artifact_id") or "").strip()
    address = str(private.get("address") or "").strip()
    if not artifact_id or not address:
        return ""
    return f"{artifact_id}.{address}"


def _coalesce_memory_card_records(
    records: tuple[
        tuple[
            int,
            int,
            ConversationMemoryCard,
            dict[str, Any],
            tuple[object, ...] | None,
        ],
        ...,
    ],
) -> tuple[tuple[ConversationMemoryCard, dict[str, Any]], ...]:
    coalesced: list[tuple[int, int, ConversationMemoryCard, dict[str, Any]]] = []
    key_positions: dict[tuple[object, ...], int] = {}
    for rank, position, card, private, key in records:
        if key is None:
            coalesced.append((rank, position, card, private))
            continue
        existing_index = key_positions.get(key)
        if existing_index is None:
            key_positions[key] = len(coalesced)
            coalesced.append(
                (
                    rank,
                    position,
                    card,
                    private,
                )
            )
            continue
        existing_rank, existing_position, existing_card, existing_private = coalesced[
            existing_index
        ]
        coalesced[existing_index] = (
            existing_rank,
            existing_position,
            existing_card,
            _merge_backing_memory_cards(existing_private, private),
        )
    return tuple((card, private) for _rank, _position, card, private in coalesced)


def _ranked_memory_card_records(
    artifacts: tuple[Any, ...],
    *,
    current_question: str,
    prior_requests_by_artifact_id: dict[str, tuple[PriorRequestMemory, ...]],
) -> tuple[tuple[ConversationMemoryCard, dict[str, Any]], ...]:
    records: list[
        tuple[
            int,
            int,
            ConversationMemoryCard,
            dict[str, Any],
            tuple[object, ...] | None,
        ]
    ] = []
    position = 0
    ownership = _MemoryOwnershipIndex.from_artifacts(
        artifacts,
        current_question=current_question,
        prior_requests_by_artifact_id=prior_requests_by_artifact_id,
    )
    for artifact in reversed(artifacts):
        for prior_request in ownership.prior_requests_for_artifact(artifact):
            card, private = prior_request
            if not ownership.is_visible_prior_request(card):
                continue
            records.append(
                (1, position, card, ownership.with_owned_backing(private), None)
            )
            position += 1
        for address in artifact.addresses:
            projected = _memory_card_for_address(artifact=artifact, address=address)
            if projected is None:
                continue
            card, private = projected
            if not ownership.is_visible_address_card(card):
                continue
            records.append(
                (
                    1,
                    position,
                    card,
                    private,
                    _memory_card_coalesce_key(address),
                )
            )
            position += 1
    return _coalesce_memory_card_records(tuple(sorted(records)))


@dataclass(frozen=True)
class _MemoryOwnershipIndex:
    prior_requests_by_artifact_id: dict[
        str, tuple[tuple[ConversationMemoryCard, dict[str, Any]], ...]
    ]
    continued_prior_request_ids: frozenset[str]
    owned_backing_ids: frozenset[str]
    current_question: str

    @classmethod
    def from_artifacts(
        cls,
        artifacts: tuple[Any, ...],
        *,
        current_question: str,
        prior_requests_by_artifact_id: dict[str, tuple[PriorRequestMemory, ...]],
    ) -> "_MemoryOwnershipIndex":
        prior_request_cards_by_artifact_id = {
            artifact_id: _memory_cards_for_prior_requests(
                artifact=next(
                    artifact
                    for artifact in artifacts
                    if str(artifact.artifact_id) == artifact_id
                ),
                prior_requests=requests,
            )
            for artifact_id, requests in prior_requests_by_artifact_id.items()
        }
        all_prior_requests = tuple(
            prior_request
            for prior_requests in prior_request_cards_by_artifact_id.values()
            for prior_request in prior_requests
        )
        return cls(
            prior_requests_by_artifact_id=prior_request_cards_by_artifact_id,
            continued_prior_request_ids=_continued_prior_request_memory_ids(artifacts),
            owned_backing_ids=_owned_backing_memory_ids(all_prior_requests),
            current_question=current_question,
        )

    def prior_requests_for_artifact(
        self,
        artifact: Any,
    ) -> tuple[tuple[ConversationMemoryCard, dict[str, Any]], ...]:
        return self.prior_requests_by_artifact_id.get(
            str(getattr(artifact, "artifact_id", "") or ""),
            (),
        )

    def is_visible_prior_request(self, card: ConversationMemoryCard) -> bool:
        return card.memory_id not in self.continued_prior_request_ids

    def is_visible_address_card(self, card: ConversationMemoryCard) -> bool:
        if card.memory_id in self.continued_prior_request_ids:
            return False
        return card.memory_id not in self.owned_backing_ids

    def with_owned_backing(self, private: dict[str, Any]) -> dict[str, Any]:
        continued = tuple(
            backing
            for memory_id in _activated_prior_request_memory_ids(private)
            for backing in (self.private_prior_request_card(memory_id),)
            if backing is not None
        )
        if not continued:
            return private
        return {
            **private,
            _BACKING_MEMORY_CARDS: (
                *tuple(private.get(_BACKING_MEMORY_CARDS) or ()),
                *continued,
            ),
        }

    def private_prior_request_card(self, memory_id: str) -> dict[str, Any] | None:
        artifact_id = memory_id.split(".prior_request.", 1)[0]
        for card, private in self.prior_requests_by_artifact_id.get(artifact_id, ()):
            if card.memory_id == memory_id:
                return private
        return None


def _continued_prior_request_memory_ids(artifacts: tuple[Any, ...]) -> frozenset[str]:
    return frozenset(
        memory_id
        for artifact in artifacts
        for memory_id in _activated_memory_ids_from_artifact(artifact)
        if _is_prior_request_memory_id(memory_id)
    )


def _activated_prior_request_memory_ids(private: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(memory_id)
        for memory_id in private.get("activated_memory_ids") or ()
        if _is_prior_request_memory_id(str(memory_id))
    )


def _is_prior_request_memory_id(memory_id: str) -> bool:
    return ".prior_request." in str(memory_id)


def _owned_backing_memory_ids(
    prior_requests: tuple[tuple[ConversationMemoryCard, dict[str, Any]], ...],
) -> frozenset[str]:
    return frozenset(
        memory_id
        for _card, private in prior_requests
        for backing in private.get(_BACKING_MEMORY_CARDS) or ()
        if isinstance(backing, dict)
        if _should_suppress_backing_card(backing)
        for memory_id in (_private_memory_id(backing),)
        if memory_id
    )


def _should_suppress_backing_card(backing: dict[str, Any]) -> bool:
    kind = str(backing.get("kind") or "")
    return kind in {"entity_identity", "time_scope", "scalar_value"}


def _memory_cards_for_prior_requests(
    *,
    artifact: FactArtifact,
    prior_requests: tuple[PriorRequestMemory, ...],
) -> tuple[tuple[ConversationMemoryCard, dict[str, Any]], ...]:
    if artifact.outcome is not FactOutcome.ANSWERED:
        return ()
    output: list[tuple[ConversationMemoryCard, dict[str, Any]]] = []
    use_fact_scoped_display = len(prior_requests) > 1
    for prior_request in prior_requests:
        output.append(
            _prior_request_memory_card(
                artifact=artifact,
                prior_request=prior_request,
                use_fact_scoped_display=use_fact_scoped_display,
            )
        )
    return tuple(output)


def _prior_request_memory_card(
    *,
    artifact: FactArtifact,
    prior_request: PriorRequestMemory,
    use_fact_scoped_display: bool,
) -> tuple[ConversationMemoryCard, dict[str, Any]]:
    artifact_id = artifact.artifact_id
    display = _prior_request_card_display(
        artifact,
        answer_request=prior_request,
        use_fact_scoped_display=use_fact_scoped_display,
    )
    memory_id = prior_request.memory_id
    details = _prior_request_details(prior_request)
    private = {
        "kind": "prior_answer_request",
        "artifact_id": artifact_id,
        "address": f"prior_request.{prior_request.request_id}",
        "display": display,
        "activated_memory_ids": _activated_memory_ids_from_artifact(artifact),
        "answer_output_frames": prior_request.output_frames,
        **details,
    }
    private_cards_by_memory_id = _address_private_cards_by_memory_id(artifact)
    backing_cards = tuple(
        private_cards_by_memory_id[source_id]
        for source_id in prior_request.source_lineage
        if source_id in private_cards_by_memory_id
    )
    if backing_cards:
        private[_BACKING_MEMORY_CARDS] = backing_cards
    return (
        ConversationMemoryCard(
            card_id=memory_id,
            memory_id=memory_id,
            kind="prior_answer_request",
            display=display,
            details=details,
        ),
        private,
    )


def _address_private_cards_by_memory_id(
    artifact: FactArtifact,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for address in artifact.addresses:
        projected = _memory_card_for_address(artifact=artifact, address=address)
        if projected is None:
            continue
        card, private = projected
        output[card.memory_id] = private
    return output


def _prior_request_details(prior_request: PriorRequestMemory) -> dict[str, Any]:
    return {
        "request_shape": prior_request.request_shape_payload(),
        "prior_slot_bindings": prior_request.slot_bindings_payload(),
    }


def _prior_request_card_display(
    artifact: FactArtifact,
    *,
    answer_request: PriorRequestMemory,
    use_fact_scoped_display: bool,
) -> str:
    if use_fact_scoped_display:
        return answer_request.answer_fact
    return artifact.source_question.strip() or answer_request.answer_fact


def _memory_card_coalesce_key(address: Any) -> tuple[object, ...] | None:
    if getattr(address, "kind", None) != FactAddressKind.ENTITY:
        return None
    resource = str(getattr(address, "resource", "") or "").strip()
    identity = {
        str(key).strip(): str(value).strip()
        for key, value in (getattr(address, "identity", {}) or {}).items()
        if str(key).strip() and str(value).strip()
    }
    if not resource or not identity:
        return None
    return (
        "entity_identity",
        resource,
        tuple(sorted(identity.items())),
    )


def _merge_backing_memory_cards(
    existing: dict[str, Any],
    private: dict[str, Any],
) -> dict[str, Any]:
    output = dict(existing)
    backing_cards = list(output.get(_BACKING_MEMORY_CARDS) or (existing,))
    backing_cards.append(dict(private))
    output[_BACKING_MEMORY_CARDS] = tuple(backing_cards)
    return output


def _memory_card_for_address(
    *,
    artifact: FactArtifact,
    address: FactAddress,
) -> tuple[ConversationMemoryCard, dict[str, Any]] | None:
    builder = _MEMORY_CARD_BUILDERS.get(address.kind)
    if builder is None:
        return None
    return builder(artifact=artifact, address=address)


def _entity_memory_card(
    *,
    artifact: Any,
    address: Any,
) -> tuple[ConversationMemoryCard, dict[str, Any]]:
    card, private = _memory_card_pair(
        artifact=artifact,
        address=address,
        kind="entity_identity",
    )
    private["entity_key"] = {
        "entity_kind": str(getattr(address, "resource", "") or "").strip(),
        "key_id": str(getattr(address, "key_id", "") or "").strip(),
        "components": dict(getattr(address, "identity", {}) or {}),
    }
    private["reference_text"] = str(
        getattr(address, "reference_text", "") or ""
    ).strip()
    return card, private


def _value_memory_card(
    *,
    artifact: Any,
    address: Any,
) -> tuple[ConversationMemoryCard, dict[str, Any]]:
    kind = (
        "time_scope"
        if str((getattr(address, "scalar_value", {}) or {}).get("type") or "")
        == "time_scope"
        else "scalar_value"
    )
    card, private = _memory_card_pair(artifact=artifact, address=address, kind=kind)
    scalar_value = getattr(address, "scalar_value", {}) or {}
    private["value"] = scalar_value.get("value")
    private["expression"] = scalar_value.get("expression")
    return card, private


def _relation_memory_card(
    *,
    artifact: Any,
    address: Any,
) -> tuple[ConversationMemoryCard, dict[str, Any]]:
    subject = _single_answer_request_display(artifact)
    card, private = _memory_card_pair(
        artifact=artifact,
        address=address,
        kind="row_set",
        display=subject,
    )
    identity = _singleton_answer_entity_key(artifact, address=address)
    if identity is not None:
        entity_kind, key_id, components = identity
        private["entity_key"] = {
            "entity_kind": entity_kind,
            "key_id": key_id,
            "components": components,
        }
    return card, private


def _singleton_answer_entity_key(
    artifact: FactArtifact,
    *,
    address: FactAddress,
) -> tuple[str, str, dict[str, str]] | None:
    if len(address.row_addresses) != 1:
        return None
    row = artifact.address(address.row_addresses[0])
    if row is None:
        return None
    identities = tuple(
        value.value
        for value in row.values.values()
        if isinstance(value.value, EntityKeyValue)
    )
    unique = tuple(dict.fromkeys(_entity_key_signature(item) for item in identities))
    if len(unique) != 1:
        return None
    entity_kind, key_id, components = unique[0]
    return entity_kind, key_id, dict(components)


def _entity_key_signature(
    identity: EntityKeyValue,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    components = tuple(
        sorted(
            (component.component_id, str(component.value))
            for component in identity.components
        )
    )
    return identity.entity_kind, identity.key_id, components


def _memory_card_pair(
    *,
    artifact: Any,
    address: Any,
    kind: str,
    display: str = "",
    details: dict[str, Any] | None = None,
) -> tuple[ConversationMemoryCard, dict[str, Any]]:
    memory_id = f"{artifact.artifact_id}.{address.address}"
    private = {
        "kind": kind,
        "artifact_id": artifact.artifact_id,
        "address": address.address,
    }
    proof_refs = tuple(
        str(item).strip()
        for item in getattr(getattr(address, "evidence", None), "step_ids", ()) or ()
        if str(item).strip()
    )
    if proof_refs:
        private["proof_refs"] = proof_refs
    if details:
        private.update(details)
    card_kwargs: dict[str, Any] = {
        "card_id": memory_id,
        "memory_id": memory_id,
        "kind": kind,
        "display": display or _card_display(artifact=artifact, address=address),
    }
    private["display"] = card_kwargs["display"]
    if details:
        card_kwargs["details"] = details
    return (
        ConversationMemoryCard(**card_kwargs),
        private,
    )


_MEMORY_CARD_BUILDERS = {
    FactAddressKind.ENTITY: _entity_memory_card,
    FactAddressKind.VALUE: _value_memory_card,
    FactAddressKind.RELATION: _relation_memory_card,
}


def _card_display(*, artifact: Any, address: Any) -> str:
    display = str(getattr(address, "display", "") or "").strip()
    if display:
        return display
    reference_text = str(getattr(address, "reference_text", "") or "").strip()
    if reference_text:
        return reference_text
    clarification_questions = tuple(
        question
        for raw in getattr(address, "clarification_questions", ()) or ()
        if (question := str(raw or "").strip())
    )
    source_question = str(getattr(artifact, "source_question", "") or "").strip()
    if clarification_questions and source_question:
        return f"{source_question} Clarification needed: {' '.join(clarification_questions)}"
    if clarification_questions:
        return " ".join(clarification_questions)
    if source_question:
        return source_question
    return str(getattr(address, "address", "") or "").strip()


def _single_answer_request_display(artifact: FactArtifact) -> str:
    requests = prior_requests_from_artifact(artifact)
    if len(requests) != 1:
        return ""
    return requests[0].answer_fact


def _without_empty_strings(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if not (isinstance(value, str) and not value.strip())
    }


def _relation_rows(
    address: Any,
    *,
    rows_by_address: dict[str, Any],
) -> tuple[Any, ...]:
    if address.row_addresses:
        rows: list[Any] = []
        for row_address in address.row_addresses:
            row = rows_by_address.get(row_address)
            if row is None or row.source_relation != address.address:
                raise ValueError(
                    f"memory relation row address not found: {row_address}"
                )
            rows.append(row)
        return tuple(rows)
    return tuple(
        row
        for row in rows_by_address.values()
        if row.source_relation == address.address
    )


def _prompt_context(
    *,
    values: tuple[MemoryValue, ...],
    relation_prompts: tuple[dict[str, Any], ...],
    outcome_prompts: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if values:
        context["memoryValues"] = [
            _without_empty_strings(
                {
                    "id": value.id,
                    "type": value.value_type,
                    "value": value.value,
                    "proofRefs": list(value.proof_refs),
                    "sourceRelationId": value.source_relation_id,
                    "sourceRowId": value.source_row_id,
                    "sourceRowGrain": dict(value.source_row_grain or {}),
                    "sourceFieldId": value.source_field_id,
                    "priorAnswerOutputIds": list(value.answer_output_ids),
                }
            )
            for value in values
        ]
    if relation_prompts:
        context["memoryRelations"] = list(relation_prompts)
    if outcome_prompts:
        context["memoryOutcomes"] = list(outcome_prompts)
    return context


def _memory_outcome_prompt(*, artifact: Any, address: Any) -> dict[str, Any]:
    return {
        "id": f"{artifact.artifact_id}.{address.address}",
        "artifactId": artifact.artifact_id,
        "terminal": address.terminal,
        "clarificationQuestions": list(address.clarification_questions),
        "scope": dict(address.scope),
        "proof": dict(address.proof),
        "sourceQuestion": artifact.source_question,
        "sourceAnswer": artifact.source_answer,
        "proofRefs": list(address.evidence.step_ids if address.evidence else ()),
    }


def _memory_relation_prompt(
    *,
    relation_id: str,
    address: Any,
    rows: tuple[Any, ...],
    completeness: CompletenessProof,
) -> dict[str, Any]:
    return {
        "id": relation_id,
        "source": dict(address.source),
        "grainKeys": list(address.grain_keys),
        "rowCount": len(rows),
        "fields": [
            _memory_field_prompt(field_id, address=address, rows=rows)
            for field_id in _memory_field_ids(address=address, rows=rows)
        ],
        "completeness": _memory_completeness_prompt(completeness),
    }


def _relation_cell_values(
    *,
    relation_id: str,
    address: Any,
    rows: tuple[Any, ...],
    relation_proof_refs: tuple[str, ...],
) -> tuple[MemoryValue, ...]:
    output: list[MemoryValue] = []
    grain_keys = set(address.grain_keys or ())
    for row_index, row in enumerate(rows, start=1):
        row_id = str(getattr(row, "address", "") or f"row_{row_index}")
        row_suffix = "value" if len(rows) == 1 else f"value.{row_id}"
        for field_id in _memory_field_ids(address=address, rows=rows):
            if field_id in grain_keys:
                continue
            scalar = _row_scalar_value(row, field_id=field_id)
            if scalar is None:
                continue
            value_type, value = scalar
            output.append(
                MemoryValue(
                    id=f"{relation_id}.{row_suffix}.{field_id}",
                    value=value,
                    value_type=value_type,
                    proof_refs=_row_proof_refs(row) or relation_proof_refs,
                    source_relation_id=relation_id,
                    source_row_id=row_id,
                    source_row_grain=dict(getattr(row, "grain", {}) or {}),
                    source_field_id=field_id,
                    answer_output_ids=_row_answer_output_ids(row, field_id=field_id),
                )
            )
    return tuple(output)


def _row_scalar_value(
    row: FactAddress,
    *,
    field_id: str,
) -> tuple[str, RuntimeValue] | None:
    address_value = row.values.get(field_id)
    if address_value is None or isinstance(address_value.value, EntityKeyValue):
        return None
    value = address_value.value
    if value in ("", None) or isinstance(value, (dict, list)):
        return None
    return address_value.type, value


def _row_answer_output_ids(row: FactAddress, *, field_id: str) -> tuple[str, ...]:
    value = row.values.get(field_id)
    return value.answer_output_ids if value is not None else ()


def _row_proof_refs(row: FactAddress) -> tuple[str, ...]:
    return row.evidence.step_ids if row.evidence is not None else ()


def _memory_field_ids(
    *,
    address: FactAddress,
    rows: tuple[FactAddress, ...],
) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for field_id in (
        *tuple(address.grain_keys),
        *tuple(address.field_coverage),
        *tuple(key for row in rows for key in row.values),
    ):
        normalized = str(field_id)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ids.append(normalized)
    return tuple(ids)


def _memory_field_prompt(
    field_id: str,
    *,
    address: FactAddress,
    rows: tuple[FactAddress, ...],
) -> dict[str, Any]:
    field: dict[str, Any] = {
        "id": field_id,
        "type": _memory_field_type(field_id, rows),
        "grain": field_id in set(address.grain_keys),
    }
    source_field = address.field_coverage.get(field_id)
    if source_field:
        field["sourceField"] = str(source_field)
    answer_output_ids = _memory_field_answer_output_ids(field_id, rows)
    if answer_output_ids:
        field["prior_answer_output_ids"] = list(answer_output_ids)
    return field


def _memory_field_type(field_id: str, rows: tuple[FactAddress, ...]) -> str:
    for row in rows:
        value = row.values.get(field_id)
        if value is not None:
            return value.type
    return "unknown"


def _memory_field_answer_output_ids(
    field_id: str,
    rows: tuple[FactAddress, ...],
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.values.get(field_id)
        if value is None:
            continue
        for raw in value.answer_output_ids:
            normalized = str(raw or "").strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(normalized)
    return tuple(output)


def _memory_completeness_prompt(completeness: CompletenessProof) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": completeness.status.value,
        "setKind": completeness.set_kind.value,
        "pagination": completeness.pagination.value,
        "scopeFingerprint": completeness.scope_fingerprint,
        "proofRefs": list(completeness.proof_refs),
    }
    if completeness.row_count is not None:
        payload["rowCount"] = completeness.row_count
    return payload


def _memory_completeness(address: Any) -> CompletenessProof:
    payload = dict(address.completeness or {})
    truncated = payload.get("truncated") is True
    status = (
        CompletenessStatus.INCOMPLETE
        if truncated
        else _completeness_status(payload.get("status"))
    )
    return CompletenessProof(
        status=status,
        source_kind=CompletenessSourceKind.MEMORY_READ,
        set_kind=_set_kind(payload.get("setKind")),
        scope_fingerprint=str(payload.get("scopeFingerprint") or address.scope or {}),
        proof_refs=tuple(address.evidence.step_ids if address.evidence else ()),
        row_count=_int_or_none(payload.get("rowCount")),
        pagination=(
            PaginationCompleteness.TRUNCATED
            if truncated
            else _pagination(payload.get("pagination"))
        ),
    )


def _completeness_status(value: Any) -> CompletenessStatus:
    if value in {item.value for item in CompletenessStatus}:
        return CompletenessStatus(str(value))
    return CompletenessStatus.UNKNOWN


def _set_kind(value: Any) -> RelationSetKind:
    if value in {item.value for item in RelationSetKind}:
        return RelationSetKind(str(value))
    return RelationSetKind.UNKNOWN


def _pagination(value: Any) -> PaginationCompleteness:
    if value in {item.value for item in PaginationCompleteness}:
        return PaginationCompleteness(str(value))
    return PaginationCompleteness.UNKNOWN


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _row_values(row: FactAddress) -> dict[str, RuntimeValue]:
    output: dict[str, RuntimeValue] = {}
    for values in (row.grain, row.identity, row.values):
        for key, value in values.items():
            _assign_row_value(output, str(key), _memory_value(value))
    return output


def _memory_value(value: Any) -> RuntimeValue:
    if isinstance(value, FactAddressValue):
        if isinstance(value.value, EntityKeyValue):
            return value.value.component_values()
        return value.value
    return parse_runtime_value(value)


def _assign_row_value(
    output: dict[str, RuntimeValue],
    key: str,
    value: RuntimeValue,
) -> None:
    existing = output.get(key)
    if key in output and existing != value:
        raise ValueError(f"conflicting memory row field {key}")
    output[key] = value
