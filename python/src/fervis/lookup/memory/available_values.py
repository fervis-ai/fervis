"""Question-scoped values exposed from activated conversation memory."""

from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.values import (
    FactValue,
    LiteralType,
    ValueDependency,
    ValueDependencyKind,
)
from fervis.lookup.canonical_data import EntityKeyComponentValue, EntityKeyValue
from fervis.lookup.memory.projection import LookupMemory, MemoryValue
from fervis.memory.identities import MemoryIdentitySet


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
    values: list[FactValue] = []
    for value in memory.identity_values:
        memory_id = _identity_value_memory_id(value.source)
        component = _single_key_component(value.value)
        if memory_id not in active_memory_ids or component is None:
            continue
        values.append(
            FactValue.identity(
                id=value.id,
                entity_kind=value.value.entity_kind,
                key_id=value.value.key_id,
                key_component_id=component.component_id,
                value=str(component.value),
                display_value=(
                    value.display_label or value.lookup_text or str(component.value)
                ),
                proof_refs=value.proof_refs,
                source_refs=_active_memory_source_refs(memory_id),
                dependencies=_active_memory_dependencies(memory_id),
            )
        )
    return tuple(values)


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
    values: list[FactValue] = []
    for value in memory.identity_sets:
        component_set = _single_component_set(value)
        if value.source_relation_id not in active_memory_ids or component_set is None:
            continue
        component_id, component_values = component_set
        values.append(
            FactValue.identity_set(
                id=value.id,
                entity_kind=value.entity_kind,
                key_id=value.key_id,
                key_component_id=component_id,
                values=component_values,
                display_value=value.display_label,
                source_relation_id=value.source_relation_id,
                proof_refs=value.proof_refs,
                source_refs=_active_memory_source_refs(value.source_relation_id),
                dependencies=_active_memory_dependencies(value.source_relation_id),
            )
        )
    return tuple(values)


def _single_key_component(value: EntityKeyValue) -> EntityKeyComponentValue | None:
    if len(value.components) != 1:
        return None
    return value.components[0]


def _single_component_set(
    value: MemoryIdentitySet,
) -> tuple[str, tuple[str, ...]] | None:
    components = tuple(_single_key_component(key) for key in value.keys)
    if any(component is None for component in components):
        return None
    present_components = tuple(
        component for component in components if component is not None
    )
    component_ids = {component.component_id for component in present_components}
    if len(component_ids) != 1:
        return None
    component_id = present_components[0].component_id
    component_values = tuple(str(component.value) for component in present_components)
    return component_id, component_values


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
