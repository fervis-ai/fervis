"""Activation projection for selected conversation memory handles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.memory.conversation_context.model import ConversationMemoryCard
from fervis.memory.addresses import FactAddress
from fervis.memory.artifacts import FactArtifact


@dataclass(frozen=True)
class ExpandedActivatedMemory:
    by_memory_id: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"activatedMemory": dict(self.by_memory_id)}


def expand_activated_memory_cards(
    *,
    artifacts: tuple[FactArtifact, ...],
    memory_cards: dict[str, dict[str, Any]],
    used_memory_ids: tuple[str, ...],
) -> ExpandedActivatedMemory:
    context = _ActivationContext(
        artifacts_by_id={artifact.artifact_id: artifact for artifact in artifacts},
        memory_cards=memory_cards,
    )
    output: dict[str, dict[str, Any]] = {}
    for card in _selected_memory_cards(
        memory_cards=memory_cards,
        memory_ids=used_memory_ids,
    ):
        if card.memory_id in output:
            raise ValueError("duplicate used memory id")
        output[card.memory_id] = _expanded_memory_card_payload(
            selected_card=card,
            context=context,
        )
    return ExpandedActivatedMemory(by_memory_id=output)


def _selected_memory_cards(
    *,
    memory_cards: dict[str, dict[str, Any]],
    memory_ids: tuple[str, ...],
) -> tuple[ConversationMemoryCard, ...]:
    output: list[ConversationMemoryCard] = []
    for memory_id in memory_ids:
        payload = memory_cards.get(memory_id)
        if not isinstance(payload, dict):
            raise ValueError(f"unknown memory id: {memory_id}")
        output.append(
            ConversationMemoryCard(
                card_id=str(payload.get("card_id") or memory_id),
                memory_id=memory_id,
                kind=str(payload.get("kind") or ""),
                display=str(payload.get("display") or memory_id),
                details=dict(payload.get("details") or {}),
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class _ActivationContext:
    artifacts_by_id: dict[str, FactArtifact]
    memory_cards: dict[str, dict[str, Any]]


def _expanded_memory_card_payload(
    *,
    selected_card: ConversationMemoryCard,
    context: _ActivationContext,
) -> dict[str, Any]:
    card = dict(context.memory_cards.get(selected_card.memory_id) or {})
    if not card:
        raise ValueError(f"unknown memory id: {selected_card.memory_id}")
    artifact = _artifact_for_card(selected_card.memory_id, card=card, context=context)
    _validate_memory_card(selected_card=selected_card, card=card)
    if card.get("kind") == "prior_answer_request":
        return _expanded_prior_request_payload(
            artifact=artifact,
            card=card,
        )
    address = _address_for_card(selected_card.memory_id, card=card, artifact=artifact)
    return _expanded_card_payload(
        artifact=artifact,
        address=address,
        card=card,
    )


def _artifact_for_card(
    memory_id: str,
    *,
    card: dict[str, Any],
    context: _ActivationContext,
) -> FactArtifact:
    artifact_id = str(card.get("artifact_id") or "").strip()
    artifact = context.artifacts_by_id.get(artifact_id)
    if artifact is None:
        raise ValueError(f"memory {memory_id} references unknown artifact")
    return artifact


def _address_for_card(
    memory_id: str,
    *,
    card: dict[str, Any],
    artifact: FactArtifact,
) -> FactAddress:
    address_id = str(card.get("address") or "").strip()
    address = artifact.address(address_id)
    if address is None:
        raise ValueError(f"memory {memory_id} references unknown address")
    return address


def _validate_memory_card(
    *,
    selected_card: ConversationMemoryCard,
    card: dict[str, Any],
) -> None:
    kind = str(card.get("kind") or "").strip()
    if kind != selected_card.kind:
        raise ValueError("memory card kind does not match backing card")


def _expanded_prior_request_payload(
    *,
    artifact: FactArtifact,
    card: dict[str, Any],
) -> dict[str, Any]:
    request_shape = dict(card.get("request_shape") or {})
    slots = _request_slots(request_shape)
    return {
        "kind": "prior_answer_request",
        "source_question": artifact.source_question,
        "request_shape": {
            **request_shape,
            "slots": slots,
        },
    }


def _expanded_card_payload(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    card: dict[str, Any],
) -> dict[str, Any]:
    kind = str(card.get("kind") or "").strip()
    builder = _CARD_PAYLOAD_BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unsupported memory card kind: {kind}")
    return builder(
        artifact=artifact,
        address=address,
        card=card,
    )


def _row_set_payload(
    *,
    artifact: FactArtifact,
    address: FactAddress,
) -> dict[str, Any]:
    rows = tuple(
        row
        for row in (
            artifact.address(row_address) for row_address in address.row_addresses
        )
        if row is not None
    )
    return {
        "kind": "row_set",
        "grain": tuple(address.grain_keys),
        "canonical_ids": tuple(
            value
            for row in rows
            for value in row.identity.values()
            if str(value or "").strip()
        ),
        "row_fact_refs": tuple(row.address for row in rows),
        "source_lineage": _source_lineage(address),
        "proof_refs": _proof_refs(address),
    }


def _entity_identity_payload(
    *,
    address: FactAddress,
) -> dict[str, Any]:
    canonical_values = {
        str(field).strip(): str(value).strip()
        for field, value in address.identity.items()
        if str(field).strip() and str(value).strip()
    }
    if len(canonical_values) != 1:
        raise ValueError("entity identity memory requires one canonical identity")
    identity_field, canonical_id = next(iter(canonical_values.items()))
    identity_type = str(address.resource or "").strip()
    if not identity_type:
        raise ValueError("entity identity memory requires identity type")
    return {
        "kind": "entity_identity",
        "identity_type": identity_type,
        "identity_field": identity_field,
        "canonical_id": canonical_id,
        "canonical_values": canonical_values,
        "display_label": address.reference_text,
        "display_fact_refs": (address.address,),
        "proof_refs": _proof_refs(address),
    }


def _scalar_value_payload(
    *,
    address: FactAddress,
) -> dict[str, Any]:
    return {
        "kind": "scalar_value",
        "value": address.scalar_value.get("value"),
        "value_kind": str(address.scalar_value.get("type") or ""),
        "unit": address.scalar_value.get("unit"),
        "value_fact_refs": (address.address,),
        "proof_refs": _proof_refs(address),
    }


def _comparison_payload(
    *,
    address: FactAddress,
) -> dict[str, Any]:
    return {
        "kind": "comparison",
        "value": address.scalar_value.get("value"),
        "value_kind": str(address.scalar_value.get("type") or ""),
        "unit": address.scalar_value.get("unit"),
        "value_fact_refs": (address.address,),
        "proof_refs": _proof_refs(address),
    }


def _time_scope_payload(
    *,
    address: FactAddress,
) -> dict[str, Any]:
    return {
        "kind": "time_scope",
        "expression": address.scalar_value.get("expression")
        or address.scalar_value.get("value"),
        "resolved_start": address.scalar_value.get("resolvedStart"),
        "resolved_end": address.scalar_value.get("resolvedEnd"),
        "granularity": address.scalar_value.get("granularity"),
        "value_fact_refs": (address.address,),
        "proof_refs": _proof_refs(address),
    }


def _clarification_payload(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    card: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "clarification_answer",
        "clarification_chain_id": artifact.artifact_id,
        "clarification_question": card.get("clarification_question")
        or " ".join(address.clarification_questions),
        "pending_integrated_question": card.get("pending_integrated_question")
        or artifact.source_question,
        "activation_fact_refs": (address.address,),
    }


def _request_slots(request_shape: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_slots = request_shape.get("slots") or ()
    if not isinstance(raw_slots, (list, tuple)):
        return ()
    slots: list[dict[str, Any]] = []
    for item in raw_slots:
        if isinstance(item, dict):
            slots.append(dict(item))
    return tuple(slots)


def _source_lineage(address: FactAddress) -> tuple[Any, ...]:
    return tuple(
        item
        for item in (
            address.source.get("read_id"),
            address.source.get("readId"),
            address.source.get("endpointName"),
        )
        if item
    )


def _proof_refs(address: FactAddress) -> tuple[Any, ...]:
    return tuple(address.evidence.step_ids if address.evidence else ())


def _row_set_payload_from_card(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    card: dict[str, Any],
) -> dict[str, Any]:
    del card
    return _row_set_payload(artifact=artifact, address=address)


def _entity_identity_payload_from_card(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    card: dict[str, Any],
) -> dict[str, Any]:
    del artifact, card
    return _entity_identity_payload(address=address)


def _scalar_value_payload_from_card(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    card: dict[str, Any],
) -> dict[str, Any]:
    del artifact, card
    return _scalar_value_payload(address=address)


def _comparison_payload_from_card(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    card: dict[str, Any],
) -> dict[str, Any]:
    del artifact, card
    return _comparison_payload(address=address)


def _time_scope_payload_from_card(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    card: dict[str, Any],
) -> dict[str, Any]:
    del artifact, card
    return _time_scope_payload(address=address)


_CARD_PAYLOAD_BUILDERS = {
    "row_set": _row_set_payload_from_card,
    "entity_identity": _entity_identity_payload_from_card,
    "scalar_value": _scalar_value_payload_from_card,
    "comparison": _comparison_payload_from_card,
    "time_scope": _time_scope_payload_from_card,
    "clarification_answer": _clarification_payload,
}
