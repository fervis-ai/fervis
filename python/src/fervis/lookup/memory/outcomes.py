"""Memory addresses for terminal fact outcomes."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

from fervis.lookup.outcomes.model import (
    AnswerResult,
    FactResult,
    Impossible,
    NoData,
    Undefined,
)
from fervis.lookup.outcomes.terminal_details import (
    empty_relation_payload,
    undefined_operation_payload,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    ValueKind,
)
from fervis.lookup.answer_program.result_projection import (
    EntityKeyValue,
    ProjectedResultRow,
    RelationResultOutput,
)
from fervis.lookup.answer_program.values import (
    IdentityValuePayload,
    TimeValuePayload,
)
from fervis.memory.addresses import EvidenceRef, FactAddress, FactAddressValue
from fervis.memory.artifacts import FactOutcome
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.plan_execution.relations import RelationRows

MAX_FACT_RELATION_ROWS = 50


def fact_result_answer_addresses(
    result: FactResult,
    *,
    question_contract: Any = None,
    grounded_values: tuple[FactValue, ...] = (),
) -> tuple[FactAddress, ...]:
    outcome = result.outcome
    if not isinstance(outcome, AnswerResult):
        return ()
    addresses: list[FactAddress] = []
    result_fields = _result_fields_by_relation(outcome)
    outputs_by_relation = _relation_outputs_by_relation(outcome)
    projected_rows_by_key = {
        (row.relation_id, row.row_index): row for row in outcome.projected_rows
    }
    for relation in outcome.relations:
        allowed_fields = result_fields.get(relation.id)
        if allowed_fields is None:
            continue
        projected_rows = relation.rows[:MAX_FACT_RELATION_ROWS]
        row_addresses = tuple(
            f"row.{relation.id}.{index + 1}"
            for index, _row in enumerate(projected_rows)
        )
        fields = tuple(
            dict.fromkeys(
                key
                for row in projected_rows
                for key in row
                if isinstance(key, str) and key in allowed_fields
            )
        )
        proof = relation.completeness
        truncated = len(projected_rows) < len(relation.rows)
        memory_status = "incomplete" if truncated else proof.status.value
        memory_pagination = "truncated" if truncated else proof.pagination.value
        evidence = EvidenceRef(step_ids=proof.proof_refs)
        addresses.append(
            FactAddress.relation(
                address=f"relation.{relation.id}",
                source={
                    "kind": "operation_output",
                    "relationId": relation.id,
                },
                grain_keys=relation.grain_keys,
                field_coverage={field: f"{relation.id}.{field}" for field in fields},
                completeness={
                    "status": memory_status,
                    "rowCount": proof.row_count or len(relation.rows),
                    "setKind": proof.set_kind.value,
                    "pagination": memory_pagination,
                    "scopeFingerprint": proof.scope_fingerprint,
                    **(
                        {
                            "truncated": True,
                            "projectedRowCount": len(projected_rows),
                        }
                        if truncated
                        else {}
                    ),
                },
                row_addresses=row_addresses,
                evidence=evidence,
            )
        )
        for row_index, (row_address, row) in enumerate(
            zip(row_addresses, projected_rows)
        ):
            projected_row = _projected_row(
                projected_rows_by_key,
                relation_id=relation.id,
                row_index=row_index,
            )
            answer_values = _memory_answer_values(
                projected_row,
                relation=relation,
                outputs=outputs_by_relation.get(relation.id, ()),
            )
            identity = {
                key: str(row[key])
                for key in relation.grain_keys
                if key in row
                and not isinstance(row[key], (dict, list))
                and row[key] not in ("", None)
            }
            addresses.append(
                FactAddress.row(
                    address=row_address,
                    relation=f"relation.{relation.id}",
                    grain={key: identity[key] for key in identity},
                    values=answer_values,
                    identity=identity,
                    evidence=evidence,
                )
            )
    for key, value in (outcome.scalars or {}).items():
        if isinstance(value, (dict, list)) or value in ("", None):
            continue
        scalar_output_ids = _scalar_answer_output_ids(outcome, scalar_id=str(key))
        addresses.append(
            FactAddress.value(
                address=f"value.{key}",
                value={"type": _value_type(value), "value": value},
                derivation={
                    "source": "operation_output",
                    **(
                        {"answer_output_ids": list(scalar_output_ids)}
                        if scalar_output_ids
                        else {}
                    ),
                },
                evidence=EvidenceRef(step_ids=outcome.proof_refs),
            )
        )
    return tuple(addresses)


def fact_value_memory_addresses(
    values: tuple[FactValue, ...],
) -> tuple[FactAddress, ...]:
    addresses: list[FactAddress] = []
    for value in values:
        if value.kind == ValueKind.IDENTITY and isinstance(
            value.payload, IdentityValuePayload
        ):
            reference_text = value.payload.display_value or value.label
            if not reference_text:
                continue
            addresses.append(
                FactAddress.entity(
                    address=f"entity.{value.id}",
                    resource=value.payload.entity_kind,
                    key_id=value.payload.key_id,
                    reference_text=reference_text,
                    identity={
                        value.payload.key_component_id: value.payload.value,
                    },
                    evidence=EvidenceRef(step_ids=value.proof_refs),
                )
            )
            continue
        if value.kind == ValueKind.TIME and isinstance(value.payload, TimeValuePayload):
            addresses.append(
                FactAddress.value(
                    address=f"value.{value.id}",
                    value={
                        "type": "time_scope",
                        "value": value.payload.expression,
                        "expression": value.payload.expression,
                        "resolvedStart": value.payload.resolved_start,
                        "resolvedEnd": value.payload.resolved_end,
                        "granularity": value.payload.granularity,
                    },
                    display=value.label or value.payload.expression,
                    evidence=EvidenceRef(step_ids=value.proof_refs),
                )
            )
            continue
    return tuple(addresses)


def _result_fields_by_relation(
    outcome: AnswerResult,
) -> dict[str, frozenset[str]]:
    fields: dict[str, set[str]] = {}
    for relation_output in outcome.result_projection.relation_outputs:
        fields.setdefault(relation_output.relation_id, set()).update(
            tuple(
                component.field_id
                for component in relation_output.entity_key.components
            )
            if relation_output.entity_key is not None
            else (relation_output.field_id,)
        )
    return {
        relation_id: frozenset(field_refs) for relation_id, field_refs in fields.items()
    }


def _relation_outputs_by_relation(
    outcome: AnswerResult,
) -> dict[str, tuple[RelationResultOutput, ...]]:
    output: dict[str, list[RelationResultOutput]] = {}
    for relation_output in outcome.result_projection.relation_outputs:
        output.setdefault(relation_output.relation_id, []).append(relation_output)
    return {relation_id: tuple(outputs) for relation_id, outputs in output.items()}


def _projected_row(
    rows_by_key: dict[tuple[str, int], ProjectedResultRow],
    *,
    relation_id: str,
    row_index: int,
) -> ProjectedResultRow:
    row = rows_by_key.get((relation_id, row_index))
    if row is None:
        raise ValueError("projected result row is unavailable")
    return row


def _memory_answer_values(
    row: ProjectedResultRow,
    *,
    relation: RelationRows,
    outputs: tuple[RelationResultOutput, ...],
) -> dict[str, FactAddressValue]:
    values: dict[str, FactAddressValue] = {}
    for output in outputs:
        if output.role and output.role != "answer_value":
            continue
        value = row.values[output.id]
        if isinstance(value, EntityKeyValue):
            values[output.id] = FactAddressValue(
                type="entity_key",
                value=value,
                answer_output_ids=(output.id,),
            )
            continue
        if isinstance(value, (dict, list)) or value in ("", None):
            continue
        field_id = output.field_id
        existing = values.get(field_id)
        if existing is not None:
            values[field_id] = replace(
                existing,
                answer_output_ids=(*existing.answer_output_ids, output.id),
            )
            continue
        values[field_id] = FactAddressValue(
            type=_relation_field_type(relation, field_id, value),
            value=value,
            answer_output_ids=(output.id,),
        )
    return values


def _scalar_answer_output_ids(
    outcome: AnswerResult,
    *,
    scalar_id: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item.id)
            for item in outcome.result_projection.scalar_outputs
            if str(item.scalar_id) == scalar_id and str(item.id)
        )
    )


def fact_result_outcome_address(
    result: FactResult,
    *,
    requested_fact_id: str | None = None,
) -> FactAddress | None:
    outcome = result.outcome
    if isinstance(outcome, NoData):
        empty = outcome.empty_relation
        if requested_fact_id and empty.requested_fact_ids:
            requested_fact_ids = tuple(
                item for item in empty.requested_fact_ids if item == requested_fact_id
            )
            if not requested_fact_ids:
                return None
            empty = replace(empty, requested_fact_ids=requested_fact_ids)
        return FactAddress.outcome(
            address="outcome.no_data",
            terminal=FactOutcome.NO_DATA.value,
            scope={"scopeRef": empty.scope_ref},
            proof={"emptyRelation": empty_relation_payload(empty)},
            evidence=EvidenceRef(step_ids=outcome.proof_refs),
        )
    if isinstance(outcome, Undefined):
        operation = outcome.operation
        return FactAddress.outcome(
            address="outcome.undefined",
            terminal=FactOutcome.UNDEFINED.value,
            proof={"operation": undefined_operation_payload(operation)},
            evidence=EvidenceRef(step_ids=_undefined_proof_refs(outcome)),
        )
    if isinstance(outcome, Impossible):
        if requested_fact_id:
            blocked_requirements = tuple(
                item
                for item in outcome.blocked_requirements
                if item.requested_fact_id == requested_fact_id
            )
            if not blocked_requirements:
                return None
            outcome = replace(outcome, blocked_requirements=blocked_requirements)
        return FactAddress.outcome(
            address="outcome.impossible",
            terminal=FactOutcome.IMPOSSIBLE.value,
            proof=_impossible_memory_payload(outcome),
            evidence=EvidenceRef(step_ids=_impossible_proof_refs(outcome)),
        )
    return None


def _value_type(value: RuntimeValue) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float | Decimal):
        return "decimal"
    return "string"


def _relation_field_type(
    relation: RelationRows,
    field_id: str,
    value: RuntimeValue,
) -> str:
    field_types = relation.field_types
    declared = str(field_types.get(field_id) if field_types is not None else "").strip()
    if declared:
        return declared
    return _value_type(value)


def _impossible_proof_refs(outcome: Impossible) -> tuple[str, ...]:
    refs = [*outcome.proof_refs]
    for requirement in outcome.blocked_requirements:
        refs.extend(requirement.proof_refs)
    return tuple(dict.fromkeys(refs))


def _impossible_memory_payload(outcome: Impossible) -> dict[str, object]:
    return {
        "blockedRequirements": [
            {
                "id": requirement.id,
                "kind": requirement.kind.value,
                "requestedFactId": requirement.requested_fact_id,
                "factRef": requirement.fact_ref,
                "requiredFor": requirement.required_for,
                "proofRefs": list(requirement.proof_refs),
            }
            for requirement in outcome.blocked_requirements
        ]
    }


def _undefined_proof_refs(outcome: Undefined) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*outcome.proof_refs, *outcome.operation.proof_refs)))
