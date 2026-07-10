"""Runtime projection from lookup results to lineage memory artifacts."""

from __future__ import annotations

from typing import Any

from fervis.lineage.enums import MemoryArtifactSourceKind
from fervis.lineage.ids import lineage_id
from fervis.lineage.recorder import (
    FactResultWrite,
    MemoryArtifactWrite,
    RequestedFactWrite,
)
from fervis.lookup.memory.outcomes import (
    fact_result_answer_addresses,
    fact_result_outcome_address,
)
from fervis.lookup.outcomes.model import FactResult
from fervis.lookup.question_contract import QuestionContract
from fervis.memory.artifacts import FactOutcome, build_fact_artifact
from fervis.memory.lineage import (
    MEMORY_ARTIFACT_SCHEMA,
    MEMORY_ARTIFACT_SCHEMA_REV,
    memory_artifact_payload,
)


def answered_memory_artifacts(
    *,
    run_id: str,
    fact_result: FactResult,
    requested_facts: tuple[RequestedFactWrite, ...],
    fact_results: tuple[FactResultWrite, ...],
    question_contract: QuestionContract,
    answer_plan: Any,
    grounded_values: tuple[Any, ...],
    extra_fact_addresses: tuple[Any, ...],
    known_input_step_id: str | None,
    source_question: str,
    source_answer: str,
    conversation_resolution_activation: dict[str, Any] | None,
) -> tuple[MemoryArtifactWrite, ...]:
    answer_addresses = fact_result_answer_addresses(
        fact_result,
        question_contract=question_contract,
        grounded_values=grounded_values,
    )
    fact_key_by_requested_fact_id = {
        fact.requested_fact_id: fact.fact_key for fact in requested_facts
    }
    output_ids_by_fact_key = _answer_output_ids_by_requested_fact(answer_plan)
    artifacts: list[MemoryArtifactWrite] = list(
        (
            *_requested_fact_memory_artifacts(
                run_id=run_id,
                requested_facts=requested_facts,
                question_contract=question_contract,
                outcome=FactOutcome.ANSWERED,
                source_question=source_question,
                source_answer=source_answer,
                conversation_resolution_activation=conversation_resolution_activation,
            ),
            *_known_input_memory_artifacts(
                run_id=run_id,
                produced_by_step_id=known_input_step_id,
                addresses=extra_fact_addresses,
                source_question=source_question,
                conversation_resolution_activation=conversation_resolution_activation,
            ),
        )
    )
    for fact in fact_results:
        fact_key = fact_key_by_requested_fact_id[fact.requested_fact_id]
        output_ids = output_ids_by_fact_key.get(fact_key, frozenset())
        addresses = _dedupe_addresses(
            _addresses_for_answer_outputs(
                answer_addresses,
                answer_output_ids=output_ids,
            )
        )
        if not addresses:
            continue
        artifacts.append(
            _memory_artifact_write(
                run_id=run_id,
                produced_by_step_id=fact.produced_by_step_id,
                source_kind=MemoryArtifactSourceKind.FACT_RESULT,
                fact_result_id=fact.fact_result_id,
                artifact=build_fact_artifact(
                    artifact_id=lineage_id(
                        "memory_artifact",
                        run_id,
                        fact.fact_result_id,
                    ),
                    outcome=FactOutcome.ANSWERED,
                    addresses=addresses,
                    provenance=_memory_provenance(
                        run_id=run_id,
                        question_contract=question_contract,
                        requested_fact_id=fact.requested_fact_id,
                        requested_fact_key=fact_key,
                        conversation_resolution_activation=(
                            conversation_resolution_activation
                        ),
                    ),
                    source_question=source_question,
                    source_answer=source_answer,
                ),
            ),
        )
    return tuple(artifacts)


def terminal_memory_artifacts(
    *,
    run_id: str,
    fact_result: FactResult,
    requested_facts: tuple[RequestedFactWrite, ...],
    fact_results: tuple[FactResultWrite, ...],
    question_contract: QuestionContract,
    produced_by_step_id: str | None,
    outcome: FactOutcome,
    source_question: str,
    conversation_resolution_activation: dict[str, Any] | None,
) -> tuple[MemoryArtifactWrite, ...]:
    fact_key_by_requested_fact_id = {
        fact.requested_fact_id: fact.fact_key for fact in requested_facts
    }
    fact_address_by_result_id = {
        fact.fact_result_id: address
        for fact in fact_results
        for address in (
            fact_result_outcome_address(
                fact_result,
                requested_fact_id=fact_key_by_requested_fact_id.get(
                    fact.requested_fact_id,
                    "",
                ),
            ),
        )
        if address is not None
    }
    if not fact_address_by_result_id:
        return _run_terminal_memory_artifacts(
            run_id=run_id,
            produced_by_step_id=produced_by_step_id,
            fact_result=fact_result,
            outcome=outcome,
            source_question=source_question,
            conversation_resolution_activation=conversation_resolution_activation,
        )
    return (
        *_requested_fact_memory_artifacts(
            run_id=run_id,
            requested_facts=requested_facts,
            question_contract=question_contract,
            outcome=outcome,
            source_question=source_question,
            conversation_resolution_activation=conversation_resolution_activation,
        ),
        *tuple(
            _memory_artifact_write(
                run_id=run_id,
                produced_by_step_id=fact.produced_by_step_id,
                source_kind=MemoryArtifactSourceKind.FACT_RESULT,
                fact_result_id=fact.fact_result_id,
                artifact=build_fact_artifact(
                    artifact_id=lineage_id(
                        "memory_artifact",
                        run_id,
                        fact.fact_result_id,
                    ),
                    outcome=outcome,
                    addresses=(fact_address_by_result_id[fact.fact_result_id],),
                    provenance=_memory_provenance(
                        run_id=run_id,
                        question_contract=None,
                        requested_fact_id=fact.requested_fact_id,
                        requested_fact_key="",
                        conversation_resolution_activation=(
                            conversation_resolution_activation
                        ),
                    ),
                    source_question=source_question,
                ),
            )
            for fact in fact_results
            if fact.fact_result_id in fact_address_by_result_id
        ),
    )


def _run_terminal_memory_artifacts(
    *,
    run_id: str,
    produced_by_step_id: str | None,
    fact_result: FactResult,
    outcome: FactOutcome,
    source_question: str,
    conversation_resolution_activation: dict[str, Any] | None,
) -> tuple[MemoryArtifactWrite, ...]:
    if produced_by_step_id is None:
        return ()
    address = fact_result_outcome_address(fact_result)
    if address is None:
        return ()
    return (
        _memory_artifact_write(
            run_id=run_id,
            produced_by_step_id=produced_by_step_id,
            source_kind=MemoryArtifactSourceKind.RUN_TERMINAL,
            artifact=build_fact_artifact(
                artifact_id=lineage_id("memory_artifact", run_id, "run_terminal"),
                outcome=outcome,
                addresses=(address,),
                provenance={
                    key: value
                    for key, value in {
                        "runId": run_id,
                        "conversation_resolution_activation": (
                            dict(conversation_resolution_activation)
                            if conversation_resolution_activation
                            else None
                        ),
                    }.items()
                    if value is not None
                },
                source_question=source_question,
            ),
        ),
    )


def _known_input_memory_artifacts(
    *,
    run_id: str,
    produced_by_step_id: str | None,
    addresses: tuple[Any, ...],
    source_question: str,
    conversation_resolution_activation: dict[str, Any] | None,
) -> tuple[MemoryArtifactWrite, ...]:
    if produced_by_step_id is None:
        return ()
    return tuple(
        _memory_artifact_write(
            run_id=run_id,
            produced_by_step_id=produced_by_step_id,
            source_kind=MemoryArtifactSourceKind.KNOWN_INPUT,
            artifact=build_fact_artifact(
                artifact_id=lineage_id(
                    "memory_artifact",
                    run_id,
                    "known_input",
                    _address_identity(address),
                ),
                outcome=FactOutcome.ANSWERED,
                addresses=(address,),
                provenance={
                    key: value
                    for key, value in {
                        "runId": run_id,
                        "conversation_resolution_activation": (
                            dict(conversation_resolution_activation)
                            if conversation_resolution_activation
                            else None
                        ),
                    }.items()
                    if value is not None
                },
                source_question=source_question,
            ),
        )
        for address in _dedupe_addresses(addresses)
    )


def _requested_fact_memory_artifacts(
    *,
    run_id: str,
    requested_facts: tuple[RequestedFactWrite, ...],
    question_contract: QuestionContract,
    outcome: FactOutcome,
    source_question: str,
    source_answer: str = "",
    conversation_resolution_activation: dict[str, Any] | None,
) -> tuple[MemoryArtifactWrite, ...]:
    return tuple(
        _memory_artifact_write(
            run_id=run_id,
            produced_by_step_id=fact.produced_by_step_id,
            source_kind=MemoryArtifactSourceKind.REQUESTED_FACT,
            requested_fact_id=fact.requested_fact_id,
            artifact=build_fact_artifact(
                artifact_id=lineage_id(
                    "memory_artifact",
                    run_id,
                    fact.requested_fact_id,
                ),
                outcome=outcome,
                provenance=_memory_provenance(
                    run_id=run_id,
                    question_contract=question_contract,
                    requested_fact_id=fact.requested_fact_id,
                    requested_fact_key=fact.fact_key,
                    conversation_resolution_activation=(
                        conversation_resolution_activation
                    ),
                ),
                source_question=source_question,
                source_answer=source_answer,
            ),
        )
        for fact in requested_facts
    )


def _memory_artifact_write(
    *,
    run_id: str,
    produced_by_step_id: str,
    source_kind: MemoryArtifactSourceKind,
    artifact: Any,
    fact_result_id: str | None = None,
    requested_fact_id: str | None = None,
) -> MemoryArtifactWrite:
    return MemoryArtifactWrite(
        memory_artifact_id=artifact.artifact_id,
        run_id=run_id,
        produced_by_step_id=produced_by_step_id,
        source_kind=source_kind,
        requested_fact_id=requested_fact_id,
        fact_result_id=fact_result_id,
        payload_schema=MEMORY_ARTIFACT_SCHEMA,
        payload_schema_rev=MEMORY_ARTIFACT_SCHEMA_REV,
        payload_json=memory_artifact_payload(
            artifact=artifact,
            source_kind=source_kind,
        ),
    )


def _memory_provenance(
    *,
    run_id: str,
    question_contract: QuestionContract | None,
    requested_fact_id: str,
    requested_fact_key: str,
    conversation_resolution_activation: dict[str, Any] | None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "runId": run_id,
        "requestedFactId": requested_fact_id,
    }
    if requested_fact_key:
        provenance["requestedFactKey"] = requested_fact_key
    if question_contract is not None:
        provenance["question_contract"] = _question_contract_memory_payload(
            question_contract,
            requested_fact_key=requested_fact_key,
        )
    if conversation_resolution_activation:
        provenance["conversation_resolution_activation"] = dict(
            conversation_resolution_activation
        )
    return provenance


def _question_contract_memory_payload(
    question_contract: QuestionContract,
    *,
    requested_fact_key: str,
) -> dict[str, Any]:
    requested_fact = next(
        (
            fact
            for fact in question_contract.requested_facts
            if fact.id == requested_fact_key
        ),
        None,
    )
    if requested_fact is None:
        raise ValueError("memory artifact references unknown requested fact")
    scoped_contract = QuestionContract(
        question_inputs=question_contract.inputs_for_fact(requested_fact_key),
        requested_facts=(requested_fact,),
    )
    return dict(scoped_contract.to_model_dict())


def _answer_output_ids_by_requested_fact(
    answer_plan: Any,
) -> dict[str, frozenset[str]]:
    output: dict[str, set[str]] = {}
    for fulfillment in answer_plan.fulfillment:
        requested_fact_id = str(fulfillment.requested_fact_id or "")
        answer_output_id = str(fulfillment.answer_output_id or "")
        if not requested_fact_id or not answer_output_id:
            continue
        output.setdefault(requested_fact_id, set()).add(answer_output_id)
    return {key: frozenset(values) for key, values in output.items()}


def _addresses_for_answer_outputs(
    addresses: tuple[Any, ...],
    *,
    answer_output_ids: frozenset[str],
) -> tuple[Any, ...]:
    selected_rows = tuple(
        address
        for address in addresses
        if _address_matches_answer_outputs(
            address,
            answer_output_ids=answer_output_ids,
        )
    )
    selected_row_relations = {
        getattr(address, "source_relation", "")
        for address in selected_rows
        if getattr(address, "source_relation", "")
    }
    selected_relations = tuple(
        address
        for address in addresses
        if getattr(address, "address", "") in selected_row_relations
    )
    return (*selected_relations, *selected_rows)


def _address_matches_answer_outputs(
    address: Any,
    *,
    answer_output_ids: frozenset[str],
) -> bool:
    if not answer_output_ids:
        return False
    derivation = getattr(address, "derivation", {}) or {}
    if answer_output_ids.intersection(derivation.get("answer_output_ids") or ()):
        return True
    values = getattr(address, "values", {}) or {}
    return any(
        answer_output_ids.intersection((value or {}).get("answer_output_ids") or ())
        for value in values.values()
        if isinstance(value, dict)
    )


def _dedupe_addresses(addresses: tuple[Any, ...]) -> tuple[Any, ...]:
    output: list[Any] = []
    seen: set[str] = set()
    for address in addresses:
        address_id = str(getattr(address, "address", "") or "")
        if not address_id or address_id in seen:
            continue
        seen.add(address_id)
        output.append(address)
    return tuple(output)


def _address_identity(address: Any) -> str:
    address_id = str(getattr(address, "address", "") or "")
    if not address_id:
        raise ValueError("memory artifact address requires a stable address id")
    return address_id
