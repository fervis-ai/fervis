"""Canonical conversation-resolution input provenance for question contract."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from fervis.lookup.conversation_resolution.overlay import (
    ConversationResolutionOverlay,
    ResolvedQuestionInputOverlay,
)
from fervis.lookup.continuations.model import (
    ContinuationCarriedInput,
    ContinuationPlan,
    ContinuationReplacement,
)
from fervis.lookup.question_inputs import (
    KnownInputKind,
    LiteralInputRole,
    literal_role_from_part_kind,
)

if TYPE_CHECKING:
    from fervis.lookup.question_contract.model import RequestedFactKnownInput


class ConversationInputProvenanceSourceKind(StrEnum):
    RESOLVED_QUESTION_INPUT = "resolved_question_input"
    CONTINUATION_REPLACEMENT = "continuation_replacement"
    CONTINUATION_CARRIED = "continuation_carried"


@dataclass(frozen=True)
class ConversationInputProvenanceSource:
    kind: ConversationInputProvenanceSourceKind
    current_text: str = ""
    prior_text: str = ""
    part_id: str = ""
    resolved_input_ref: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConversationInputProvenanceSourceKind):
            object.__setattr__(
                self,
                "kind",
                ConversationInputProvenanceSourceKind(str(self.kind)),
            )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"kind": self.kind.value}
        if self.current_text:
            payload["current_text"] = self.current_text
        if self.prior_text:
            payload["prior_text"] = self.prior_text
        if self.part_id:
            payload["part_id"] = self.part_id
        if self.resolved_input_ref:
            payload["resolved_input_ref"] = self.resolved_input_ref
        return payload


@dataclass(frozen=True)
class ConversationInputProvenance:
    input_ref: str
    kind: KnownInputKind
    value_source_text: str
    resolved_value_text: str = ""
    role: LiteralInputRole | None = None
    field_label_text: str = ""
    value_meaning_hint: str = ""
    sources: tuple[ConversationInputProvenanceSource, ...] = ()

    def __post_init__(self) -> None:
        if not self.input_ref.strip():
            raise ValueError("conversation input provenance requires input_ref")
        if not isinstance(self.kind, KnownInputKind):
            object.__setattr__(self, "kind", KnownInputKind(str(self.kind)))
        if not self.value_source_text.strip():
            raise ValueError("conversation input provenance requires source text")
        if self.kind == KnownInputKind.LITERAL:
            if self.role is None:
                raise ValueError("literal conversation input provenance requires role")
            if not isinstance(self.role, LiteralInputRole):
                object.__setattr__(self, "role", LiteralInputRole(str(self.role)))
            if not self.resolved_value_text.strip():
                raise ValueError(
                    "literal conversation input provenance requires resolved value"
                )
        elif self.role is not None:
            raise ValueError("row-set conversation input provenance cannot carry role")
        if not self.sources:
            raise ValueError("conversation input provenance requires sources")

    @property
    def is_conversation_resolution_input(self) -> bool:
        conversation_resolution_sources = {
            ConversationInputProvenanceSourceKind.RESOLVED_QUESTION_INPUT,
            ConversationInputProvenanceSourceKind.CONTINUATION_CARRIED,
        }
        source_kinds = {source.kind for source in self.sources}
        return bool(source_kinds & conversation_resolution_sources)

    @property
    def question_input_source(self) -> str:
        if self.is_conversation_resolution_input:
            return "conversation_resolution"
        return "question_context"

    def accepts_question_input(self, known: "RequestedFactKnownInput") -> bool:
        if self.kind != known.kind:
            return False
        if self.input_ref != known.resolved_input_ref:
            return False
        if self.value_source_text != known.text:
            return False
        if self.kind == KnownInputKind.ROW_SET_REFERENCE:
            return True
        role_matches = self.role == known.role
        resolved_value_matches = self.resolved_value_text == known.resolved_value_text
        field_label_matches = self.field_label_text == known.field_label_text
        meaning_hint_matches = self.value_meaning_hint == known.value_meaning_hint
        literal_fields_match = all(
            (
                role_matches,
                resolved_value_matches,
                field_label_matches,
                meaning_hint_matches,
            )
        )
        return literal_fields_match

    def context_texts(self) -> tuple[str, ...]:
        values = (
            self.value_source_text,
            self.resolved_value_text,
            self.field_label_text,
            self.value_meaning_hint,
            self.input_ref,
            *(source.current_text for source in self.sources),
            *(source.prior_text for source in self.sources),
        )
        return _unique_non_empty_texts(values)

    def to_prompt_payload(self) -> dict[str, object]:
        source_kinds = [source.kind.value for source in self.sources]
        payload: dict[str, object] = {
            "input_ref": self.input_ref,
            "kind": self.kind.value,
            "question_input_source": self.question_input_source,
            "value_source_text": self.value_source_text,
            "sources": source_kinds,
        }
        if self.resolved_value_text:
            payload["resolved_value_text"] = self.resolved_value_text
        if self.role is not None:
            payload["role"] = self.role.value
        if self.field_label_text:
            payload["field_label_text"] = self.field_label_text
        if self.value_meaning_hint:
            payload["value_meaning_hint"] = self.value_meaning_hint
        return payload

    def to_inspection_payload(self) -> dict[str, object]:
        payload = self.to_prompt_payload()
        source_payloads = [source.to_payload() for source in self.sources]
        payload["sources"] = source_payloads
        return payload


@dataclass(frozen=True)
class ConversationInputProvenanceSet:
    resolved_request_text: str = ""
    question_context_kind: str = ""
    inputs: tuple[ConversationInputProvenance, ...] = ()

    @property
    def has_inputs(self) -> bool:
        return bool(self.inputs)

    @property
    def has_conversation_resolution_inputs(self) -> bool:
        has_conversation_resolution_input = any(
            item.is_conversation_resolution_input for item in self.inputs
        )
        return has_conversation_resolution_input

    def accepts_question_input(self, known: "RequestedFactKnownInput") -> bool:
        for item in self.inputs:
            input_is_from_conversation_resolution = item.is_conversation_resolution_input
            input_matches_known = item.accepts_question_input(known)
            if input_is_from_conversation_resolution and input_matches_known:
                return True
        return False

    def context_texts(self) -> tuple[str, ...]:
        output = [self.resolved_request_text, self.question_context_kind]
        for item in self.inputs:
            output.extend(item.context_texts())
        return _unique_non_empty_texts(output)

    def to_prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.question_context_kind:
            payload["question_context_kind"] = self.question_context_kind
        if self.resolved_request_text:
            payload["resolved_request_text"] = self.resolved_request_text
        if self.inputs:
            input_payloads = [item.to_prompt_payload() for item in self.inputs]
            payload["inputs"] = input_payloads
        return payload

    def to_inspection_payload(self) -> dict[str, object]:
        payload = self.to_prompt_payload()
        if self.inputs:
            input_payloads = [item.to_inspection_payload() for item in self.inputs]
            payload["inputs"] = input_payloads
        return payload


def conversation_input_provenance_from(
    *,
    overlay: ConversationResolutionOverlay | None,
    continuation_plan: ContinuationPlan | None,
) -> ConversationInputProvenanceSet:
    overlay_inputs = _overlay_provenance_inputs(overlay)
    continuation_inputs = _continuation_provenance_inputs(continuation_plan)
    inputs = _merge_resolved_inputs_with_carried_inputs(
        overlay_inputs=overlay_inputs,
        continuation_inputs=continuation_inputs,
    )
    has_continuation = (
        continuation_plan is not None and continuation_plan.has_continuation
    )
    resolved_request_text = (
        continuation_plan.resolved_request_text if has_continuation else ""
    )
    question_context_kind = (
        "prior_question_continuation" if has_continuation else ""
    )
    merged_inputs = _merge_provenance_inputs(inputs)
    return ConversationInputProvenanceSet(
        resolved_request_text=resolved_request_text,
        question_context_kind=question_context_kind,
        inputs=merged_inputs,
    )


def _overlay_provenance_inputs(
    overlay: ConversationResolutionOverlay | None,
) -> tuple[ConversationInputProvenance, ...]:
    if overlay is None:
        return ()
    inputs: list[ConversationInputProvenance] = []
    for item in overlay.resolved_question_inputs:
        inputs.append(_resolved_question_input_provenance(item))
    return tuple(inputs)


def _resolved_question_input_provenance(
    item: ResolvedQuestionInputOverlay,
) -> ConversationInputProvenance:
    source = _resolved_question_input_source(item)
    if item.kind == KnownInputKind.ROW_SET_REFERENCE:
        return _row_set_reference_provenance(item, source=source)
    return _literal_resolved_question_input_provenance(item, source=source)


def _resolved_question_input_source(
    item: ResolvedQuestionInputOverlay,
) -> ConversationInputProvenanceSource:
    current_text = _resolved_question_input_current_text(item)
    return ConversationInputProvenanceSource(
        kind=ConversationInputProvenanceSourceKind.RESOLVED_QUESTION_INPUT,
        current_text=current_text,
        resolved_input_ref=item.resolved_input_ref,
    )


def _resolved_question_input_current_text(
    item: ResolvedQuestionInputOverlay,
) -> str:
    if item.kind == KnownInputKind.LITERAL:
        return item.source_text
    return item.reference_text


def _row_set_reference_provenance(
    item: ResolvedQuestionInputOverlay,
    *,
    source: ConversationInputProvenanceSource,
) -> ConversationInputProvenance:
    if item.kind == KnownInputKind.ROW_SET_REFERENCE:
        return ConversationInputProvenance(
            input_ref=item.resolved_input_ref,
            kind=KnownInputKind.ROW_SET_REFERENCE,
            value_source_text=item.reference_text,
            sources=(source,),
        )
    raise ValueError("row-set provenance requires row-set input")


def _literal_resolved_question_input_provenance(
    item: ResolvedQuestionInputOverlay,
    *,
    source: ConversationInputProvenanceSource,
) -> ConversationInputProvenance:
    return ConversationInputProvenance(
        input_ref=item.resolved_input_ref,
        kind=KnownInputKind.LITERAL,
        value_source_text=item.source_text,
        resolved_value_text=item.resolved_value_text,
        role=item.role,
        field_label_text=item.field_label_text,
        value_meaning_hint=item.value_meaning_hint,
        sources=(source,),
    )


def _continuation_provenance_inputs(
    plan: ContinuationPlan | None,
) -> tuple[ConversationInputProvenance, ...]:
    if plan is None or not plan.has_continuation:
        return ()
    inputs: list[ConversationInputProvenance] = []
    for replacement in plan.replacements:
        input_item = _replacement_provenance_input(replacement)
        if input_item is not None:
            inputs.append(input_item)
    for carried in plan.carried_inputs:
        input_item = _carried_provenance_input(carried)
        if input_item is not None:
            inputs.append(input_item)
    return tuple(inputs)


def _replacement_provenance_input(
    item: ContinuationReplacement,
) -> ConversationInputProvenance | None:
    role = literal_role_from_part_kind(item.part.kind)
    if role is None:
        return None
    source = _replacement_provenance_source(item)
    return _literal_provenance_input(
        input_ref=item.part_id,
        value_source_text=item.current_text,
        resolved_value_text=item.current_text,
        role=role,
        source=source,
    )


def _replacement_provenance_source(
    item: ContinuationReplacement,
) -> ConversationInputProvenanceSource:
    return ConversationInputProvenanceSource(
        kind=ConversationInputProvenanceSourceKind.CONTINUATION_REPLACEMENT,
        current_text=item.current_text,
        prior_text=item.part.text,
        part_id=item.part_id,
    )


def _carried_provenance_input(
    item: ContinuationCarriedInput,
) -> ConversationInputProvenance | None:
    role = literal_role_from_part_kind(item.part.kind)
    if role is None:
        return None
    source = _carried_provenance_source(item)
    return _literal_provenance_input(
        input_ref=item.part_id,
        value_source_text=item.part.text,
        resolved_value_text=item.resolved_value_text or item.part.text,
        role=role,
        source=source,
        field_label_text=item.field_label_text,
        value_meaning_hint=item.value_meaning_hint,
    )


def _carried_provenance_source(
    item: ContinuationCarriedInput,
) -> ConversationInputProvenanceSource:
    return ConversationInputProvenanceSource(
        kind=ConversationInputProvenanceSourceKind.CONTINUATION_CARRIED,
        prior_text=item.part.text,
        part_id=item.part_id,
    )


def _literal_provenance_input(
    *,
    input_ref: str,
    value_source_text: str,
    resolved_value_text: str,
    role: LiteralInputRole,
    source: ConversationInputProvenanceSource,
    field_label_text: str = "",
    value_meaning_hint: str = "",
) -> ConversationInputProvenance:
    return ConversationInputProvenance(
        input_ref=input_ref,
        kind=KnownInputKind.LITERAL,
        value_source_text=value_source_text,
        resolved_value_text=resolved_value_text,
        role=role,
        field_label_text=field_label_text,
        value_meaning_hint=value_meaning_hint,
        sources=(source,),
    )


def _merge_provenance_inputs(
    inputs: tuple[ConversationInputProvenance, ...],
) -> tuple[ConversationInputProvenance, ...]:
    merged: dict[tuple[object, ...], ConversationInputProvenance] = {}
    for item in inputs:
        key = _provenance_key(item)
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
            continue
        merged[key] = _with_merged_sources(existing, item)
    return tuple(merged.values())


def _with_merged_sources(
    existing: ConversationInputProvenance,
    item: ConversationInputProvenance,
) -> ConversationInputProvenance:
    merged_sources = _merged_sources(existing.sources, item.sources)
    return ConversationInputProvenance(
        input_ref=existing.input_ref,
        kind=existing.kind,
        value_source_text=existing.value_source_text,
        resolved_value_text=existing.resolved_value_text,
        role=existing.role,
        field_label_text=existing.field_label_text,
        value_meaning_hint=existing.value_meaning_hint,
        sources=merged_sources,
    )


def _merged_sources(
    first: tuple[ConversationInputProvenanceSource, ...],
    second: tuple[ConversationInputProvenanceSource, ...],
) -> tuple[ConversationInputProvenanceSource, ...]:
    output: list[ConversationInputProvenanceSource] = []
    for source in (*first, *second):
        if source not in output:
            output.append(source)
    return tuple(output)


def _merge_resolved_inputs_with_carried_inputs(
    *,
    overlay_inputs: tuple[ConversationInputProvenance, ...],
    continuation_inputs: tuple[ConversationInputProvenance, ...],
) -> tuple[ConversationInputProvenance, ...]:
    carried_by_signature = _unique_carried_inputs_by_signature(continuation_inputs)
    used_carried_refs: set[str] = set()
    output: list[ConversationInputProvenance] = []
    for resolved in overlay_inputs:
        signature = _resolved_equivalence_signature(resolved)
        carried = carried_by_signature.get(signature) if signature is not None else None
        if carried is None:
            output.append(resolved)
            continue
        used_carried_refs.add(carried.input_ref)
        output.append(_resolved_carried_input(resolved=resolved, carried=carried))
    for item in continuation_inputs:
        carried_input_already_used = item.input_ref in used_carried_refs
        if not carried_input_already_used:
            output.append(item)
    return tuple(output)


def _unique_carried_inputs_by_signature(
    inputs: tuple[ConversationInputProvenance, ...],
) -> dict[tuple[object, ...], ConversationInputProvenance]:
    grouped: dict[tuple[object, ...], list[ConversationInputProvenance]] = {}
    for item in inputs:
        signature = _carried_equivalence_signature(item)
        is_carried = _has_source_kind(
            item,
            ConversationInputProvenanceSourceKind.CONTINUATION_CARRIED,
        )
        if signature is not None and is_carried:
            grouped.setdefault(signature, []).append(item)
    unique_items: dict[tuple[object, ...], ConversationInputProvenance] = {}
    for signature, items in grouped.items():
        signature_has_one_match = len(items) == 1
        if signature_has_one_match:
            unique_items[signature] = items[0]
    return unique_items


def _resolved_carried_input(
    *,
    resolved: ConversationInputProvenance,
    carried: ConversationInputProvenance,
) -> ConversationInputProvenance:
    resolved_value_text = resolved.resolved_value_text or carried.resolved_value_text
    role = resolved.role or carried.role
    field_label_text = resolved.field_label_text or carried.field_label_text
    value_meaning_hint = resolved.value_meaning_hint or carried.value_meaning_hint
    sources = (*resolved.sources, *carried.sources)
    return ConversationInputProvenance(
        input_ref=carried.input_ref,
        kind=resolved.kind,
        value_source_text=resolved.value_source_text,
        resolved_value_text=resolved_value_text,
        role=role,
        field_label_text=field_label_text,
        value_meaning_hint=value_meaning_hint,
        sources=sources,
    )


def _resolved_equivalence_signature(
    item: ConversationInputProvenance,
) -> tuple[object, ...] | None:
    return _literal_role_value_signature(item)


def _carried_equivalence_signature(
    item: ConversationInputProvenance,
) -> tuple[object, ...] | None:
    return _literal_role_value_signature(item)


def _literal_role_value_signature(
    item: ConversationInputProvenance,
) -> tuple[object, ...] | None:
    if item.kind != KnownInputKind.LITERAL or item.role is None:
        return None
    return (item.kind, item.role, item.resolved_value_text)


def _has_source_kind(
    item: ConversationInputProvenance,
    source_kind: ConversationInputProvenanceSourceKind,
) -> bool:
    source_kinds = {source.kind for source in item.sources}
    return source_kind in source_kinds


def _provenance_key(item: ConversationInputProvenance) -> tuple[object, ...]:
    return (
        item.input_ref,
        item.kind,
        item.value_source_text,
        item.resolved_value_text,
        item.role,
        item.field_label_text,
        item.value_meaning_hint,
    )

def _unique_non_empty_texts(values: Iterable[object]) -> tuple[str, ...]:
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return tuple(output)
