"""Question-scoped values exposed from activated conversation memory."""

from __future__ import annotations

from typing import Any

from fervis.lookup.memory.projection import LookupMemory, MemoryValue
from fervis.lookup.answer_program.values import (
    FactValue,
    LiteralType,
    ValueDependency,
    ValueDependencyKind,
)


def active_memory_reference_values(
    *,
    memory: LookupMemory,
    active_memory_ids: frozenset[str],
) -> tuple[FactValue, ...]:
    if not active_memory_ids:
        return ()
    values = (
        *_active_identity_values(memory=memory, active_memory_ids=active_memory_ids),
        *_active_identity_sets(memory=memory, active_memory_ids=active_memory_ids),
    )
    return _dedupe_values(values)


def active_memory_operation_values(
    *,
    memory: LookupMemory,
    active_memory_ids: frozenset[str],
) -> tuple[FactValue, ...]:
    if not active_memory_ids:
        return ()
    return _dedupe_values(
        (
            *active_memory_reference_values(
                memory=memory,
                active_memory_ids=active_memory_ids,
            ),
            *_active_literal_values(
                memory=memory,
                active_memory_ids=active_memory_ids,
                include_relation_literals=True,
            ),
        )
    )


def _active_identity_values(
    *,
    memory: LookupMemory,
    active_memory_ids: frozenset[str],
) -> tuple[FactValue, ...]:
    return tuple(
        FactValue.identity(
            id=value.id,
            identity_type=value.identity_type,
            identity_field=value.identity_field,
            value=value.value,
            display_value=value.display_label or value.lookup_text or value.value,
            proof_refs=value.proof_refs,
            source_refs=_active_memory_source_refs(
                _identity_value_memory_id(value.source)
            ),
            dependencies=_active_memory_dependencies(
                _identity_value_memory_id(value.source)
            ),
        )
        for value in memory.identity_values
        if _identity_value_memory_id(value.source) in active_memory_ids
    )


def _identity_value_memory_id(source: dict[str, Any]) -> str:
    artifact_id = str(source.get("artifact_id") or "").strip()
    address = str(source.get("address") or "").strip()
    if not artifact_id or not address:
        return ""
    return f"{artifact_id}.{address}"


def _active_identity_sets(
    *,
    memory: LookupMemory,
    active_memory_ids: frozenset[str],
) -> tuple[FactValue, ...]:
    return tuple(
        FactValue.identity_set(
            id=value.id,
            identity_type=value.identity_type,
            identity_field=value.identity_field,
            values=value.values,
            display_value=value.display_label,
            source_relation_id=value.source_relation_id,
            proof_refs=value.proof_refs,
            source_refs=_active_memory_source_refs(value.source_relation_id),
            dependencies=_active_memory_dependencies(value.source_relation_id),
        )
        for value in memory.identity_sets
        if value.source_relation_id in active_memory_ids
    )


def _active_literal_values(
    *,
    memory: LookupMemory,
    active_memory_ids: frozenset[str],
    include_relation_literals: bool,
) -> tuple[FactValue, ...]:
    return tuple(
        _literal_fact_value(value)
        for value in memory.values
        if _is_literal_memory_value(value)
        and (
            value.id in active_memory_ids
            or (
                include_relation_literals
                and value.source_relation_id in active_memory_ids
            )
        )
    )


def _is_literal_memory_value(value: MemoryValue) -> bool:
    return value.value_type.strip().lower() != "time_scope"


def _literal_fact_value(value: MemoryValue) -> FactValue:
    return FactValue.literal(
        id=value.id,
        literal_type=_literal_type(value),
        value=str(value.value),
        label=value.id,
        proof_refs=value.proof_refs,
        source_refs=_active_memory_source_refs(value.id),
        dependencies=_active_memory_dependencies(value.id),
    )


def _literal_type(value: MemoryValue) -> LiteralType:
    value_type = value.value_type.lower()
    if value_type in {"number", "decimal", "integer", "float"}:
        return LiteralType.NUMBER
    if value_type == "boolean" or isinstance(value.value, bool):
        return LiteralType.BOOLEAN
    if isinstance(value.value, (int, float)):
        return LiteralType.NUMBER
    return LiteralType.STRING


def _active_memory_source_refs(memory_id: str) -> tuple[str, ...]:
    return (f"memory:{memory_id}",) if memory_id else ()


def _active_memory_dependencies(memory_id: str) -> tuple[ValueDependency, ...]:
    if not memory_id:
        return ()
    return (
        ValueDependency(
            kind=ValueDependencyKind.CONVERSATION_MEMORY,
            ref=memory_id,
        ),
    )


def _dedupe_values(values: tuple[FactValue, ...]) -> tuple[FactValue, ...]:
    output: list[FactValue] = []
    seen: set[str] = set()
    for value in values:
        if value.id in seen:
            continue
        seen.add(value.id)
        output.append(value)
    return tuple(output)
