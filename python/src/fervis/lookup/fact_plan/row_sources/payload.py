"""Prompt payload projection for row sources."""

from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog.model import (
    EntityKeyComponentTarget,
    RowCardinality,
)
from fervis.lookup.answer_program.relations import FieldBindingRole
from fervis.lookup.fact_planning.required_inputs import required_input_id

from .builder import memory_row_source_id
from .evidence import (
    required_input_evidence_ref,
    row_source_description_evidence_ref,
    row_source_evidence_ref,
    row_source_field_evidence_ref,
    row_source_param_evidence_ref,
)
from .model import (
    RowSource,
    RowSourceCatalog,
    RowSourceField,
    RowSourceKind,
    RowSourceParam,
    row_source_value_type,
)


def row_source_prompt_payload(row_sources: RowSourceCatalog) -> dict[str, Any]:
    return {
        "row_sources": [
            _prompt_source_payload(source) for source in row_sources.sources
        ]
    }


def memory_row_source_prompt_payload(
    memory_inputs: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _prompt_source_payload(source)
        for source in _memory_prompt_row_sources(memory_inputs)
    )


def _memory_prompt_row_sources(memory_inputs: dict[str, Any]) -> tuple[RowSource, ...]:
    sources: list[RowSource] = []
    for relation in memory_inputs.get("memoryRelations", ()) or ():
        if not isinstance(relation, dict):
            continue
        relation_id = str(relation.get("id") or "")
        if not relation_id:
            continue
        fields = tuple(
            RowSourceField(
                id=str(field.get("id") or ""),
                field_ref=str(field.get("id") or ""),
                label=str(field.get("id") or ""),
                type=row_source_value_type(str(field.get("type") or "")),
                allowed_roles=_memory_prompt_roles(field),
            )
            for field in relation.get("fields", ()) or ()
            if isinstance(field, dict) and field.get("id")
        )
        sources.append(
            RowSource(
                id=memory_row_source_id(relation_id),
                kind=RowSourceKind.MEMORY_READ,
                label=relation_id,
                memory_ref=relation_id,
                row_cardinality=RowCardinality.MANY,
                fields=fields,
            )
        )
    return tuple(sources)


def _prompt_source_payload(source: RowSource) -> dict[str, Any]:
    return {
        "row_source_id": source.id,
        "kind": source.kind.value,
        "evidence_ref": row_source_evidence_ref(source.id),
        **({"description": source.description} if source.description else {}),
        **(
            {"description_evidence_ref": row_source_description_evidence_ref(source.id)}
            if source.description
            else {}
        ),
        "cardinality": source.row_cardinality.value,
        "fields": [
            {
                "field_id": field.id,
                "evidence_ref": row_source_field_evidence_ref(
                    row_source_id=source.id,
                    field_id=field.id,
                ),
                "label": field.label,
                "type": field.type.value,
                "allowed_roles": [role.value for role in field.allowed_roles],
                **({"description": field.description} if field.description else {}),
            }
            for field in source.fields
        ],
        **(
            {
                "blocked_facts": [
                    {
                        "fact_ref": fact.fact_ref,
                        "availability": fact.availability.value,
                        **({"field_id": fact.field_id} if fact.field_id else {}),
                        **(
                            {"proof_refs": list(fact.proof_refs)}
                            if fact.proof_refs
                            else {}
                        ),
                    }
                    for fact in source.blocked_facts
                ]
            }
            if source.blocked_facts
            else {}
        ),
        "params": [
            row_source_param_prompt_payload(source, param) for param in source.params
        ],
    }


def row_source_param_prompt_payload(
    source: RowSource, param: RowSourceParam
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "param_id": param.id,
        "evidence_ref": row_source_param_evidence_ref(
            row_source_id=source.id,
            param_id=param.id,
        ),
        "label": param.name,
        "source": _param_source_value(param.source),
        "type": param.type.value,
        **(_entity_target_prompt_payload(param.entity_target)),
    }
    if param.required:
        payload["required"] = True
    if param.required and param.default is None:
        catalog_input_id = required_input_id(
            row_source_id=source.id,
            param_id=param.id,
        )
        if param.choices:
            payload["required_catalog_choice_input_id"] = catalog_input_id
            payload["required_catalog_choice_input_evidence_ref"] = (
                required_input_evidence_ref(required_input_id=catalog_input_id)
            )
        else:
            payload["required_catalog_input_id"] = catalog_input_id
            payload["required_catalog_input_evidence_ref"] = (
                required_input_evidence_ref(required_input_id=catalog_input_id)
            )
    if param.choices:
        payload["choices"] = list(param.choices)
    if param.choice_labels:
        payload["choice_labels"] = dict(param.choice_labels)
    if param.default is not None:
        payload["default"] = param.default
    if param.default_source:
        payload["default_source"] = param.default_source
    if param.semantics:
        payload["param_semantics"] = param.semantics.value
    return payload


def _param_source_value(source: object) -> str:
    return str(getattr(source, "value", source) or "")


def _entity_target_prompt_payload(
    target: EntityKeyComponentTarget | None,
) -> dict[str, dict[str, object]]:
    if target is None:
        return {}
    return {
        "entity_target": {
            "entity_kind": target.entity_kind,
            "key_id": target.key_id,
            "component_id": target.component_id,
        }
    }


def _memory_prompt_roles(field: dict[str, Any]) -> tuple[FieldBindingRole, ...]:
    roles = [FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE]
    if field.get("grain") is True:
        roles.insert(0, FieldBindingRole.IDENTITY)
    return tuple(roles)
