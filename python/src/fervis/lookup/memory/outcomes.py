"""Memory addresses for terminal fact outcomes."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

from fervis.lookup.outcomes.model import (
    AnswerResult,
    FactResult,
    Impossible,
    NeedsClarification,
    NoData,
    Undefined,
)
from fervis.lookup.outcomes.terminal_details import (
    clarification_payload,
    empty_relation_payload,
    undefined_operation_payload,
)
from fervis.lookup.fact_plan.values import (
    FactValue,
    ValueKind,
)
from fervis.lookup.fact_plan.values import (
    IdentityValuePayload,
    TimeValuePayload,
)
from fervis.memory.addresses import EvidenceRef, FactAddress
from fervis.memory.artifacts import FactOutcome

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
    render_fields = _render_fields_by_relation(outcome)
    render_answer_output_ids = _render_answer_output_ids_by_relation_field(outcome)
    for relation in outcome.relations:
        allowed_fields = render_fields.get(relation.id)
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
        evidence = EvidenceRef(step_ids=proof.proof_refs)
        addresses.append(
            FactAddress.relation(
                address=f"relation.{relation.id}",
                source={
                    "kind": "operation_output",
                    "relationId": relation.id,
                    **(
                        {"identityType": relation.identity_type}
                        if relation.identity_type
                        else {}
                    ),
                },
                grain_keys=relation.grain_keys,
                field_coverage={field: f"{relation.id}.{field}" for field in fields},
                completeness={
                    "status": proof.status.value,
                    "rowCount": proof.row_count or len(relation.rows),
                    "setKind": proof.set_kind.value,
                    "pagination": proof.pagination.value,
                    "scopeFingerprint": proof.scope_fingerprint,
                    **(
                        {
                            "truncated": True,
                            "projectedRowCount": len(projected_rows),
                        }
                        if len(projected_rows) < len(relation.rows)
                        else {}
                    ),
                },
                row_addresses=row_addresses,
                evidence=evidence,
            )
        )
        for row_address, row in zip(row_addresses, projected_rows):
            scalar_values = {
                str(key): {
                    "type": _relation_field_type(relation, str(key), value),
                    "value": str(value),
                    **_relation_field_answer_output_payload(
                        relation,
                        str(key),
                        render_answer_output_ids=render_answer_output_ids,
                    ),
                }
                for key, value in row.items()
                if key in allowed_fields
                and not isinstance(value, (dict, list))
                and value not in ("", None)
            }
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
                    values=scalar_values,
                    identity=identity,
                    evidence=evidence,
                )
            )
    for key, value in (outcome.scalars or {}).items():
        if isinstance(value, (dict, list)) or value in ("", None):
            continue
        answer_output_ids = _scalar_answer_output_ids(outcome, scalar_id=str(key))
        addresses.append(
            FactAddress.value(
                address=f"value.{key}",
                value={"type": _value_type(value), "value": str(value)},
                derivation={
                    "source": "operation_output",
                    **(
                        {"answer_output_ids": list(answer_output_ids)}
                        if answer_output_ids
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
                    resource=value.payload.identity_type,
                    reference_text=reference_text,
                    identity={
                        value.payload.identity_field: value.payload.value,
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


def fact_value_identity_addresses(
    values: tuple[FactValue, ...],
) -> tuple[FactAddress, ...]:
    return tuple(
        address
        for address in fact_value_memory_addresses(values)
        if address.kind.value == "entity"
    )


def _render_fields_by_relation(
    outcome: AnswerResult,
) -> dict[str, frozenset[str]]:
    render_spec = outcome.render_spec
    if render_spec is None:
        return {}
    fields: dict[str, set[str]] = {}
    for relation_output in render_spec.relation_outputs:
        fields.setdefault(relation_output.relation_id, set()).add(
            relation_output.field_id
        )
    return {
        relation_id: frozenset(field_refs) for relation_id, field_refs in fields.items()
    }


def _render_answer_output_ids_by_relation_field(
    outcome: AnswerResult,
) -> dict[tuple[str, str], tuple[str, ...]]:
    render_spec = outcome.render_spec
    if render_spec is None:
        return {}
    output: dict[tuple[str, str], list[str]] = {}
    for relation_output in render_spec.relation_outputs:
        if relation_output.role and relation_output.role != "answer_value":
            continue
        output.setdefault(
            (relation_output.relation_id, relation_output.field_id),
            [],
        ).append(relation_output.id)
    return {key: tuple(dict.fromkeys(value)) for key, value in output.items() if value}


def _scalar_answer_output_ids(
    outcome: AnswerResult,
    *,
    scalar_id: str,
) -> tuple[str, ...]:
    render_spec = outcome.render_spec
    if render_spec is None:
        return ()
    return tuple(
        dict.fromkeys(
            str(item.id)
            for item in render_spec.scalar_outputs
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
    if isinstance(outcome, NeedsClarification):
        if requested_fact_id:
            clarifications = tuple(
                item
                for item in outcome.clarifications
                if item.requested_fact_id == requested_fact_id
            )
            if not clarifications:
                return None
            outcome = replace(outcome, clarifications=clarifications)
        return FactAddress.outcome(
            address="outcome.needs_clarification",
            terminal=FactOutcome.NEEDS_CLARIFICATION.value,
            clarification_questions=tuple(
                item.question for item in outcome.clarifications
            ),
            proof=clarification_payload(outcome),
            evidence=EvidenceRef(step_ids=_clarification_proof_refs(outcome)),
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


def _value_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float | Decimal):
        return "decimal"
    return "string"


def _relation_field_type(relation: object, field_id: str, value: object) -> str:
    field_types = getattr(relation, "field_types", None) or {}
    declared = str(field_types.get(field_id) or "").strip()
    if declared:
        return declared
    return _value_type(value)


def _relation_field_answer_output_payload(
    relation: object,
    field_id: str,
    *,
    render_answer_output_ids: dict[tuple[str, str], tuple[str, ...]],
) -> dict[str, object]:
    explicit_output_ids = tuple(
        str(item)
        for item in (
            (getattr(relation, "field_answer_output_ids", None) or {}).get(field_id)
            or ()
        )
        if str(item).strip()
    )
    output_ids = explicit_output_ids or render_answer_output_ids.get(
        (str(getattr(relation, "id", "") or ""), field_id),
        (),
    )
    if not output_ids:
        return {}
    return {"answer_output_ids": list(output_ids)}


def _clarification_proof_refs(outcome: NeedsClarification) -> tuple[str, ...]:
    refs = [*outcome.proof_refs]
    for item in outcome.clarifications:
        refs.extend(item.evidence_refs)
    return tuple(dict.fromkeys(refs))


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
