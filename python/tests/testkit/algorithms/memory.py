from __future__ import annotations

from typing import Any

from fervis.memory.conversation_context import expand_activated_memory_cards
from fervis.memory.activation import (
    activate_memory,
    activated_entity_id_rows,
    ActivatedInput,
    UseAs,
)
from fervis.memory.addresses import fact_address_from_payload
from fervis.memory.answer_outputs import prior_answer_request_artifacts
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)
from fervis.memory.identities import project_memory_identity_values
from fervis.memory.projection import (
    fact_artifacts_from_context,
    project_conversation_memory,
)
from fervis.memory.prompt_contract import planner_memory_contract
from fervis.lookup.memory.projection import project_lookup_memory
from fervis.lookup.memory.projection import project_conversation_memory_cards
from fervis.lookup.memory.available_values import (
    active_memory_operation_values,
)
from fervis.lookup.memory.projection import LookupMemory, MemoryValue
from fervis.lookup.outcomes.clarifications import Clarification, ClarificationBasis
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.memory.outcomes import (
    fact_result_answer_addresses,
    fact_result_outcome_address,
)
from fervis.lookup.outcomes.model import (
    AnswerResult,
    BlockedRequirement,
    BlockedRequirementKind,
    EmptyRelation,
    EmptyRelationKind,
    FactResult,
    Impossible,
    NeedsClarification,
    NoData,
    Undefined,
    UndefinedOperationRef,
    UndefinedReasonCode,
)
from fervis.lookup.fact_plan.render_spec import (
    RenderRelationOutput,
    RenderScalarOutput,
    RenderSpec,
)

from tests.testkit.assertions import exact_mismatches, subset_mismatches
from tests.testkit.serialization import portable_value


def run_memory_activate_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    try:
        memory = activate_memory(
            artifacts=fact_artifacts_from_context(
                input_payload["conversation_context"]
            ),
            requests=tuple(
                ActivatedInput(
                    id=item["id"],
                    from_artifact_id=item["from_artifact_id"],
                    from_address=item["from_address"],
                    use_as=UseAs(item["use_as"]),
                    target_binding_id=item["target_binding_id"],
                    requested_scope=dict(item.get("requested_scope") or {}),
                )
                for item in input_payload.get("requests") or ()
            ),
        )
    except ValueError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    return subset_mismatches(
        actual={
            "activated": memory.to_dict(),
            "entity_id_rows": list(activated_entity_id_rows(memory)),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_memory_build_artifact_case(payload: dict[str, Any]) -> list[str]:
    artifact = _build_artifact(payload["input"]["artifact"])
    expected = payload["expect"]
    if "result_equals" in expected:
        return exact_mismatches(
            actual=artifact.to_dict(),
            expected=expected["result_equals"],
        )
    return subset_mismatches(
        actual=artifact.to_dict(),
        expected_subset=expected["result_contains"],
    )


def run_memory_prior_answer_request_case(payload: dict[str, Any]) -> list[str]:
    [artifact] = fact_artifacts_from_context(payload["input"]["conversation_context"])
    return subset_mismatches(
        actual={
            "requests": [
                {
                    "id": item.id,
                    "answer_fact": item.answer_fact,
                    "output_frames": [
                        frame.to_request_shape() for frame in item.output_frames
                    ],
                }
                for item in prior_answer_request_artifacts(artifact)
            ]
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_conversation_memory_expand_activated_case(
    payload: dict[str, Any],
) -> list[str]:
    input_payload = payload["input"]
    activated = expand_activated_memory_cards(
        artifacts=fact_artifacts_from_context(input_payload["conversation_context"]),
        memory_cards=dict(input_payload["memory_cards"]),
        used_memory_ids=tuple(input_payload["used_memory_ids"]),
    )
    return subset_mismatches(
        actual=portable_value(activated.to_dict()),
        expected_subset=payload["expect"]["result_contains"],
    )


def run_conversation_memory_card_projection_case(
    payload: dict[str, Any],
) -> list[str]:
    input_payload = payload["input"]
    conversation_context = dict(input_payload["conversation_context"])
    try:
        projection = project_conversation_memory_cards(
            conversation_context,
            current_question=input_payload["current_question"],
            max_cards=int(input_payload.get("max_cards") or 12),
        )
    except ValueError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    actual = portable_value(
        {
            "cards": [card.to_model_dict() for card in projection.cards],
            "cards_by_memory_id": {
                card.memory_id: card.to_model_dict() for card in projection.cards
            },
            "public_cards": [
                card.to_model_dict(include_memory_id=False) for card in projection.cards
            ],
            "public_card_keys": [
                sorted(card.to_model_dict(include_memory_id=False))
                for card in projection.cards
            ],
            "card_kinds": [card.kind for card in projection.cards],
            "card_kind_counts": _counts(card.kind for card in projection.cards),
            "card_memory_ids": [card.memory_id for card in projection.cards],
            "card_count": len(projection.cards),
            "context_sources": [
                {
                    **source.to_model_dict(),
                    "source_memory_ids": list(source.source_memory_ids),
                }
                for source in projection.context_sources
            ],
            "context_frames": [
                {
                    **frame.to_model_dict(),
                    "prior_answer_fact": frame.prior_answer_fact,
                }
                for frame in projection.context_frames
            ],
            "private_cards": projection.private_cards or {},
            "private_backing_ids": _private_backing_ids(projection.private_cards or {}),
            "omitted_counts_by_kind": projection.omitted_counts_by_kind or {},
            "omitted_total": sum((projection.omitted_counts_by_kind or {}).values()),
            "has_omitted_cards": (
                sum((projection.omitted_counts_by_kind or {}).values()) > 0
            ),
        }
    )
    errors: list[str] = []
    expected = payload["expect"]
    if "result_contains" in expected:
        errors.extend(
            subset_mismatches(
                actual=actual,
                expected_subset=expected["result_contains"],
            )
        )
    serialized = repr(actual)
    for text in expected.get("text_excludes") or ():
        if text in serialized:
            errors.append(f"unexpected text present: {text!r}")
    for field, excluded_values in (expected.get("text_excludes_from") or {}).items():
        field_text = repr(actual.get(field))
        for text in excluded_values or ():
            if text in field_text:
                errors.append(f"{field}: unexpected text present: {text!r}")
    return errors


def run_memory_lineage_memory_artifacts_case(payload: dict[str, Any]) -> list[str]:
    from fervis.lineage.enums import MemoryArtifactSourceKind
    from fervis.lineage.memory_artifacts import MemoryArtifactRow
    from fervis.memory.lineage import (
        LineageMemoryArtifactService,
    )

    input_payload = payload["input"]
    rows = tuple(
        _FixtureMemoryArtifactRow(
            conversation_id=str(item["conversation_id"]),
            row=MemoryArtifactRow(
                memory_artifact_id=str(item["memory_artifact_id"]),
                run_id=str(item["run_id"]),
                produced_by_step_id=str(
                    item.get("produced_by_step_id") or f"{item['run_id']}.step"
                ),
                source_kind=MemoryArtifactSourceKind(str(item["source_kind"])),
                payload_schema=str(item["payload_schema"]),
                payload_schema_rev=int(item["payload_schema_rev"]),
                payload_json=dict(item["payload_json"]),
                requested_fact_id=(
                    str(item["requested_fact_id"])
                    if item.get("requested_fact_id")
                    else None
                ),
                fact_result_id=(
                    str(item["fact_result_id"]) if item.get("fact_result_id") else None
                ),
            ),
        )
        for item in input_payload["memory_artifacts"]
    )
    try:
        artifacts = LineageMemoryArtifactService(
            _FixtureMemoryArtifactQuery(rows)
        ).for_conversation(
            str(input_payload["conversation_id"]),
            limit=int(input_payload.get("limit") or 5),
        )
    except ValueError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    actual = {"artifacts": [artifact.to_dict() for artifact in artifacts]}
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual,
            expected=payload["expect"]["result_equals"],
        )
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


class _FixtureMemoryArtifactQuery:
    def __init__(self, rows: tuple[Any, ...]) -> None:
        self._rows = rows

    def memory_artifact_rows_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> tuple[Any, ...]:
        rows = [
            item.row for item in self._rows if item.conversation_id == conversation_id
        ]
        run_ids = tuple(
            dict.fromkeys(reversed([row.run_id for row in rows]))
        )[:limit]
        selected_run_ids = frozenset(run_ids)
        return tuple(row for row in rows if row.run_id in selected_run_ids)


class _FixtureMemoryArtifactRow:
    def __init__(self, *, conversation_id: str, row: Any) -> None:
        self.conversation_id = conversation_id
        self.row = row


def run_memory_outcome_address_case(payload: dict[str, Any]) -> list[str]:
    address = fact_result_outcome_address(_fact_result(payload["input"]["fact_result"]))
    return subset_mismatches(
        actual={"address": address.to_dict()},
        expected_subset=payload["expect"]["result_contains"],
    )


def run_memory_answer_addresses_case(payload: dict[str, Any]) -> list[str]:
    addresses = fact_result_answer_addresses(
        _fact_result(payload["input"]["fact_result"])
    )
    return subset_mismatches(
        actual={
            "address_count": len(addresses),
            "addresses": [address.to_dict() for address in addresses],
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_memory_available_values_case(payload: dict[str, Any]) -> list[str]:
    memory = LookupMemory(
        values=tuple(
            _memory_value(item) for item in payload["input"].get("values") or ()
        )
    )
    active_memory_ids = frozenset(
        str(item) for item in payload["input"].get("active_memory_ids") or ()
    )
    operation_values = active_memory_operation_values(
        memory=memory,
        active_memory_ids=active_memory_ids,
    )
    return subset_mismatches(
        actual={
            "operation_values": _fact_values(operation_values),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_memory_project_conversation_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    try:
        projection = project_conversation_memory(
            input_payload["conversation_context"],
            current_user_message=str(input_payload.get("current_user_message") or ""),
            max_index_items=int(input_payload.get("max_index_items") or 16),
        )
    except ValueError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    return subset_mismatches(
        actual={
            "prompt_context": projection.prompt_context,
            "execution_context": projection.execution_context,
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_memory_lookup_projection_case(payload: dict[str, Any]) -> list[str]:
    memory = project_lookup_memory(payload["input"]["conversation_context"])
    relation_field_types = {
        relation.id: dict(relation.field_types) for relation in memory.relations
    }
    relation_answer_outputs = {
        relation.id: {
            field_id: list(output_ids)
            for field_id, output_ids in relation.field_answer_output_ids.items()
        }
        for relation in memory.relations
    }
    actual = {
        "relations": [
            {
                "id": relation.id,
                "rows": list(relation.rows),
                "grain_keys": list(relation.grain_keys),
                "field_types": dict(relation.field_types),
                "field_answer_output_ids": {
                    field_id: list(output_ids)
                    for field_id, output_ids in relation.field_answer_output_ids.items()
                },
                "completeness": _completeness_dict(relation.completeness),
            }
            for relation in memory.relations
        ],
        "relation_field_types": relation_field_types,
        "relation_answer_outputs": relation_answer_outputs,
        "memory_values_by_field": _memory_values_by_field(memory.prompt_context),
        "identity_values": [item.to_dict() for item in memory.identity_values],
        "identity_sets": [
            {
                **item.to_dict(),
                "values": list(item.values),
            }
            for item in memory.identity_sets
        ],
        "prompt_context": memory.prompt_context,
        "prompt_context_keys": sorted((memory.prompt_context or {}).keys()),
    }
    errors = subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )
    for field, excluded_values in (
        payload["expect"].get("text_excludes_from") or {}
    ).items():
        field_text = repr(actual.get(field))
        for text in excluded_values or ():
            if text in field_text:
                errors.append(f"{field}: unexpected text present: {text!r}")
    return errors


def run_memory_identity_projection_case(payload: dict[str, Any]) -> list[str]:
    projection = project_memory_identity_values(
        fact_artifacts_from_context(payload["input"]["conversation_context"])
    )
    return subset_mismatches(
        actual={
            "identity_values": [item.to_dict() for item in projection.identity_values],
            "identity_sets": [
                {
                    **item.to_dict(),
                    "values": list(item.values),
                }
                for item in projection.identity_sets
            ],
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_memory_planner_contract_case(payload: dict[str, Any]) -> list[str]:
    return subset_mismatches(
        actual={"contract": planner_memory_contract(payload["input"]["memory_frame"])},
        expected_subset=payload["expect"]["result_contains"],
    )


def _fact_result(payload: dict[str, Any]) -> FactResult:
    outcome = payload["outcome"]
    kind = outcome["kind"]
    if kind == "no_data":
        return FactResult(outcome=_no_data(outcome))
    if kind == "undefined":
        return FactResult(outcome=_undefined(outcome))
    if kind == "needs_clarification":
        return FactResult(outcome=_needs_clarification(outcome))
    if kind == "impossible":
        return FactResult(outcome=_impossible(outcome))
    if kind == "answer":
        return FactResult(outcome=_answer(outcome))
    raise ValueError(f"unsupported fact result outcome: {kind}")


def _build_artifact(payload: dict[str, Any]):
    return build_fact_artifact(
        artifact_id=payload["artifact_id"],
        outcome=FactOutcome(payload["outcome"]),
        addresses=tuple(
            fact_address_from_payload(item) for item in payload.get("addresses") or ()
        ),
        provenance=dict(payload.get("provenance") or {}),
        source_question=payload.get("source_question") or "",
    )


def _memory_value(payload: dict[str, Any]) -> MemoryValue:
    return MemoryValue(
        id=str(payload["id"]),
        value=payload.get("value"),
        value_type=str(payload.get("value_type") or ""),
        proof_refs=tuple(str(item) for item in payload.get("proof_refs") or ()),
        source_relation_id=str(payload.get("source_relation_id") or ""),
        source_row_id=str(payload.get("source_row_id") or ""),
        source_field_id=str(payload.get("source_field_id") or ""),
    )


def _fact_values(values: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": value.id,
            "kind": value.kind.value,
            "payload": (
                portable_value(value.payload) if value.payload is not None else None
            ),
        }
        for value in values
    ]


def _completeness_dict(completeness: Any) -> dict[str, Any]:
    return {
        "status": completeness.status.value,
        "source_kind": completeness.source_kind.value,
        "set_kind": completeness.set_kind.value,
        "scope_fingerprint": completeness.scope_fingerprint,
        "proof_refs": list(completeness.proof_refs),
        "row_count": completeness.row_count,
        "pagination": completeness.pagination.value,
    }


def _memory_values_by_field(prompt_context: dict[str, Any] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in (prompt_context or {}).get("memoryValues") or ():
        relation_id = item.get("sourceRelationId")
        field_id = item.get("sourceFieldId")
        if relation_id and field_id:
            values[f"{relation_id}.{field_id}"] = item
    return values


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _private_backing_ids(
    private_cards: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for memory_id, card in private_cards.items():
        backing_ids: list[str] = []
        for item in card.get("backing_cards") or ():
            if not isinstance(item, dict):
                continue
            artifact_id = item.get("artifact_id")
            address = item.get("address")
            if artifact_id and address:
                backing_ids.append(f"{artifact_id}.{address}")
        if backing_ids:
            output[memory_id] = backing_ids
    return output


def _no_data(payload: dict[str, Any]) -> NoData:
    relation = payload["empty_relation"]
    return NoData(
        empty_relation=EmptyRelation(
            kind=EmptyRelationKind(relation["kind"]),
            relation_id=relation["relation_id"],
            grain_keys=tuple(relation.get("grain_keys") or ()),
            scope_ref=relation.get("scope_ref") or "",
            proof_refs=tuple(relation.get("proof_refs") or ()),
        ),
        proof_refs=tuple(payload.get("proof_refs") or ()),
    )


def _undefined(payload: dict[str, Any]) -> Undefined:
    operation = payload["operation"]
    return Undefined(
        operation=UndefinedOperationRef(
            operation_id=operation["operation_id"],
            reason_code=UndefinedReasonCode(operation["reason_code"]),
            input_refs=tuple(operation.get("input_refs") or ()),
            proof_refs=tuple(operation.get("proof_refs") or ()),
        ),
        proof_refs=tuple(payload.get("proof_refs") or ()),
    )


def _needs_clarification(payload: dict[str, Any]) -> NeedsClarification:
    return NeedsClarification(
        clarifications=tuple(
            Clarification(
                id=item["id"],
                requested_fact_id=item["requested_fact_id"],
                basis=ClarificationBasis(item["basis"]),
                question=item["question"],
                known_input_id=item.get("known_input_id") or "",
                candidate_refs=tuple(item.get("candidate_refs") or ()),
                evidence_refs=tuple(item.get("evidence_refs") or ()),
            )
            for item in payload.get("clarifications") or ()
        ),
        proof_refs=tuple(payload.get("proof_refs") or ()),
    )


def _impossible(payload: dict[str, Any]) -> Impossible:
    return Impossible(
        blocked_requirements=tuple(
            BlockedRequirement(
                id=item["id"],
                kind=BlockedRequirementKind(item["kind"]),
                requested_fact_id=item["requested_fact_id"],
                fact_ref=item["fact_ref"],
                proof_refs=tuple(item.get("proof_refs") or ()),
            )
            for item in payload.get("blocked_requirements") or ()
        ),
        proof_refs=tuple(payload.get("proof_refs") or ()),
    )


def _answer(payload: dict[str, Any]) -> AnswerResult:
    return AnswerResult(
        render_spec=RenderSpec(
            relation_outputs=tuple(
                RenderRelationOutput(
                    id=item["id"],
                    relation_id=item["relation_id"],
                    field_id=item["field_id"],
                )
                for item in payload.get("relation_outputs") or ()
            ),
            scalar_outputs=tuple(
                RenderScalarOutput(
                    id=item["id"],
                    scalar_id=item["scalar_id"],
                )
                for item in payload.get("scalar_outputs") or ()
            ),
        ),
        relations=tuple(
            RelationRows(
                id=item["id"],
                rows=_relation_rows(item),
                grain_keys=tuple(item.get("grain_keys") or ()),
                field_types=dict(item.get("field_types") or {}),
                field_answer_output_ids={
                    field_id: tuple(output_ids)
                    for field_id, output_ids in (
                        item.get("field_answer_output_ids") or {}
                    ).items()
                },
                identity_type=item.get("identity_type") or "",
            )
            for item in payload.get("relations") or ()
        ),
        scalars=dict(payload.get("scalars") or {}),
        proof_refs=tuple(payload.get("proof_refs") or ()),
    )


def _relation_rows(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if "rows" in payload:
        return tuple(payload["rows"] or ())
    generated = payload.get("generate_rows")
    if not generated:
        return ()
    count = int(generated["count"])
    field_templates = dict(generated["field_templates"])
    return tuple(
        {
            field_id: str(template).format(index=index)
            for field_id, template in field_templates.items()
        }
        for index in range(count)
    )
