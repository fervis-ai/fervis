"""Activation projection for selected conversation memory handles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.memory.conversation_context.model import (
    ConversationMemoryActivation,
    ConversationMemoryActivationKind,
    ConversationMemoryCardProjection,
)
from fervis.memory.addresses import FactAddress
from fervis.memory.artifacts import FactArtifact
from fervis.memory.prior_requests import PriorRequestMemory


@dataclass(frozen=True)
class ExpandedActivatedMemory:
    by_memory_id: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"activatedMemory": dict(self.by_memory_id)}


def expand_activated_memory_cards(
    *,
    artifacts: tuple[FactArtifact, ...],
    memory_projection: ConversationMemoryCardProjection,
    used_memory_ids: tuple[str, ...],
) -> ExpandedActivatedMemory:
    context = _ActivationContext(
        artifacts_by_id={artifact.artifact_id: artifact for artifact in artifacts},
    )
    output: dict[str, dict[str, Any]] = {}
    for activation in _selected_memory_activations(
        memory_projection=memory_projection,
        memory_ids=used_memory_ids,
    ):
        if activation.memory_id in output:
            raise ValueError("duplicate used memory id")
        output[activation.memory_id] = _expanded_memory_card_payload(
            activation=activation,
            context=context,
        )
    return ExpandedActivatedMemory(by_memory_id=output)


def _selected_memory_activations(
    *,
    memory_projection: ConversationMemoryCardProjection,
    memory_ids: tuple[str, ...],
) -> tuple[ConversationMemoryActivation, ...]:
    activations_by_memory_id = {
        activation.memory_id: activation
        for activation in memory_projection.activations
    }
    if len(activations_by_memory_id) != len(memory_projection.activations):
        raise ValueError("memory projection contains duplicate memory ids")
    output: list[ConversationMemoryActivation] = []
    for memory_id in memory_ids:
        activation = activations_by_memory_id.get(memory_id)
        if activation is None:
            raise ValueError(f"unknown memory id: {memory_id}")
        output.append(activation)
    return tuple(output)


@dataclass(frozen=True)
class _ActivationContext:
    artifacts_by_id: dict[str, FactArtifact]


def _expanded_memory_card_payload(
    *,
    activation: ConversationMemoryActivation,
    context: _ActivationContext,
) -> dict[str, Any]:
    if activation.kind is ConversationMemoryActivationKind.PRIOR_REQUEST:
        prior_request = activation.prior_request
        assert prior_request is not None
        artifact = _artifact_for_id(
            activation.memory_id,
            artifact_id=activation.artifact_id,
            context=context,
        )
        return _expanded_prior_request_payload(
            artifact=artifact,
            prior_request=prior_request,
        )
    artifact = _artifact_for_id(
        activation.memory_id,
        artifact_id=activation.artifact_id,
        context=context,
    )
    address = _address_for_id(
        activation.memory_id,
        address_id=activation.address_id,
        artifact=artifact,
    )
    return _expanded_card_payload(
        artifact=artifact,
        address=address,
        kind=activation.kind,
    )


def _artifact_for_id(
    memory_id: str,
    *,
    artifact_id: str,
    context: _ActivationContext,
) -> FactArtifact:
    artifact = context.artifacts_by_id.get(artifact_id)
    if artifact is None:
        raise ValueError(f"memory {memory_id} references unknown artifact")
    return artifact


def _address_for_id(
    memory_id: str,
    *,
    address_id: str,
    artifact: FactArtifact,
) -> FactAddress:
    address = artifact.address(address_id)
    if address is None:
        raise ValueError(f"memory {memory_id} references unknown address")
    return address


def _expanded_prior_request_payload(
    *,
    artifact: FactArtifact,
    prior_request: PriorRequestMemory,
) -> dict[str, Any]:
    return {
        "kind": "prior_answer_request",
        "source_question": artifact.source_question,
        "request_shape": prior_request.request_shape_payload(),
    }


def _expanded_card_payload(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    kind: ConversationMemoryActivationKind,
) -> dict[str, Any]:
    builder = _CARD_PAYLOAD_BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unsupported memory card kind: {kind}")
    return builder(
        artifact=artifact,
        address=address,
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
    artifact: FactArtifact,
    address: FactAddress,
) -> dict[str, Any]:
    del artifact
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
    artifact: FactArtifact,
    address: FactAddress,
) -> dict[str, Any]:
    del artifact
    return {
        "kind": "scalar_value",
        "value": address.scalar_value.get("value"),
        "value_kind": str(address.scalar_value.get("type") or ""),
        "unit": address.scalar_value.get("unit"),
        "value_fact_refs": (address.address,),
        "proof_refs": _proof_refs(address),
    }


def _time_scope_payload(
    *,
    artifact: FactArtifact,
    address: FactAddress,
) -> dict[str, Any]:
    del artifact
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


def _clarification_answer_payload(
    *,
    artifact: FactArtifact,
    address: FactAddress,
) -> dict[str, Any]:
    return {
        "kind": "clarification_answer",
        "clarification_chain_id": artifact.artifact_id,
        "clarification_question": " ".join(address.clarification_questions),
        "question_being_clarified": artifact.source_question,
        "activation_fact_refs": (address.address,),
    }


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


_CARD_PAYLOAD_BUILDERS = {
    ConversationMemoryActivationKind.ROW_SET: _row_set_payload,
    ConversationMemoryActivationKind.ENTITY_IDENTITY: _entity_identity_payload,
    ConversationMemoryActivationKind.SCALAR_VALUE: _scalar_value_payload,
    ConversationMemoryActivationKind.TIME_SCOPE: _time_scope_payload,
    ConversationMemoryActivationKind.CLARIFICATION_ANSWER: (
        _clarification_answer_payload
    ),
}
