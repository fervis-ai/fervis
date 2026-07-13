"""Memory read-model projection for planner prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from fervis.memory._serialization import without_empty
from fervis.memory.addresses import FactAddress, fact_address_from_payload
from fervis.memory.artifacts import (
    FactArtifact,
    FactOutcome,
    build_fact_artifact,
)

DEFAULT_MAX_INDEX_ITEMS = 16
_REFERENCE_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class MemoryProjection:
    prompt_context: dict[str, Any] = field(default_factory=dict)
    execution_context: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[FactArtifact, ...] = ()


def project_conversation_memory(
    conversation_context: dict[str, Any],
    *,
    current_user_message: str = "",
    max_index_items: int = DEFAULT_MAX_INDEX_ITEMS,
) -> MemoryProjection:
    artifacts = _fact_artifacts(conversation_context)
    index = _memory_index(
        artifacts,
        max_items=max_index_items,
        current_user_message=current_user_message,
    )
    execution_artifacts = _artifacts_for_index(artifacts, index=index)
    prompt_context: dict[str, Any] = {}
    if index:
        prompt_context["memoryIndex"] = index
    return MemoryProjection(
        prompt_context=prompt_context,
        execution_context={
            "factArtifacts": [item.to_dict() for item in execution_artifacts]
        }
        if execution_artifacts
        else {},
        artifacts=artifacts,
    )


def fact_artifacts_from_context(
    conversation_context: dict[str, Any],
) -> tuple[FactArtifact, ...]:
    return _fact_artifacts(conversation_context)


def _fact_artifacts(conversation_context: dict[str, Any]) -> tuple[FactArtifact, ...]:
    raw_artifacts = conversation_context.get("factArtifacts")
    if isinstance(raw_artifacts, list):
        return tuple(_artifact_from_payload(raw) for raw in raw_artifacts)
    return ()


def _artifact_from_payload(payload: Any) -> FactArtifact:
    if not isinstance(payload, dict):
        raise ValueError("fact artifact payload must be an object")
    artifact_id = str(payload.get("artifactId") or "").strip()
    outcome = _fact_outcome(payload.get("outcome"))
    if not artifact_id:
        raise ValueError("fact artifact requires artifact_id")
    if outcome is None:
        raise ValueError("fact artifact requires valid outcome")
    raw_addresses = payload.get("addresses", [])
    if raw_addresses is None:
        raw_addresses = []
    if not isinstance(raw_addresses, list):
        raise ValueError("fact artifact addresses must be a list")
    addresses = tuple(fact_address_from_payload(item) for item in raw_addresses)
    return build_fact_artifact(
        artifact_id=artifact_id,
        outcome=outcome,
        addresses=addresses,
        provenance=dict(payload.get("provenance") or {}),
        source_question=str(payload.get("sourceQuestion") or ""),
        source_answer=str(payload.get("sourceAnswer") or ""),
    )


def _memory_index(
    artifacts: tuple[FactArtifact, ...],
    *,
    max_items: int,
    current_user_message: str,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for artifact, address in _ordered_addresses(
        artifacts,
        current_user_message=current_user_message,
    ):
        if len(items) >= max_items:
            return {
                "order": "explicit_match_then_newest",
                "truncated": True,
                "items": items,
            }
        items.append(_index_item(artifact, address))
    if not items:
        return {}
    return {"order": "explicit_match_then_newest", "truncated": False, "items": items}


def _artifacts_for_index(
    artifacts: tuple[FactArtifact, ...],
    *,
    index: dict[str, Any],
) -> tuple[FactArtifact, ...]:
    artifact_ids = tuple(
        dict.fromkeys(
            str(item.get("artifactId") or "").strip()
            for item in index.get("items") or ()
            if isinstance(item, dict) and str(item.get("artifactId") or "").strip()
        )
    )
    if not artifact_ids:
        return ()
    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    return tuple(
        by_id[artifact_id] for artifact_id in artifact_ids if artifact_id in by_id
    )


def _ordered_addresses(
    artifacts: tuple[FactArtifact, ...],
    *,
    current_user_message: str,
) -> tuple[tuple[FactArtifact, FactAddress], ...]:
    question_tokens = _reference_tokens(current_user_message)
    indexed: list[tuple[int, int, FactArtifact, FactAddress]] = []
    position = 0
    for artifact in artifacts:
        for address in artifact.addresses:
            indexed.append(
                (
                    0
                    if _address_matches_question(
                        address,
                        question_tokens=question_tokens,
                    )
                    else 1,
                    position,
                    artifact,
                    address,
                )
            )
            position += 1
    return tuple((artifact, address) for _, _, artifact, address in sorted(indexed))


def _reference_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).casefold() for match in _REFERENCE_TOKEN_RE.finditer(str(text))
    )


def _address_matches_question(
    address: FactAddress,
    *,
    question_tokens: tuple[str, ...],
) -> bool:
    reference_tokens = _reference_tokens(address.reference_text)
    if not reference_tokens:
        return False
    width = len(reference_tokens)
    if width > len(question_tokens):
        return False
    return any(
        question_tokens[index : index + width] == reference_tokens
        for index in range(len(question_tokens) - width + 1)
    )


def _index_item(artifact: FactArtifact, address: FactAddress) -> dict[str, Any]:
    return without_empty(
        {
            "artifactId": artifact.artifact_id,
            "address": address.address,
            "kind": address.kind.value,
            "resource": address.resource,
            "keyId": address.key_id,
            "referenceText": address.reference_text,
            "identity": dict(address.identity),
            "scope": dict(address.scope),
            "grainKeys": list(address.grain_keys),
            "terminal": address.terminal,
            "clarificationQuestions": list(address.clarification_questions),
            "sourceQuestion": artifact.source_question,
            "sourceAnswer": artifact.source_answer,
            "provenance": dict(artifact.provenance),
        }
    )


def _fact_outcome(value: Any) -> FactOutcome | None:
    try:
        return FactOutcome(str(value))
    except ValueError:
        return None
