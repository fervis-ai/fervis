"""Lookup result to answer-lineage write projection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Mapping

from fervis.lookup.errors import ErrorCode
from fervis.lookup.clarification import clarification_payload
from fervis.lineage.enums import (
    AnswerValueKind,
    FactResultKind,
    PresentationKind,
    RunResultKind,
    RuntimeErrorKind,
)
from fervis.lineage.ids import lineage_id
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.payloads.execution_proof_graph import (
    EXECUTION_PROOF_GRAPH_SCHEMA,
    EXECUTION_PROOF_GRAPH_SCHEMA_REV,
    execution_proof_graph_payload,
    read_execution_proof_graph_payload,
)
from fervis.lineage.proof_projection import project_proof_payload
from fervis.lineage.recorder import (
    AnswerOutputWrite,
    AnswerPresentationWrite,
    AnswerWrite,
    AnsweredRunResultWrite,
    ClarificationRequestWrite,
    ExecutionProofGraphWrite,
    FactualTerminalRunResultWrite,
    FactResultWrite,
    RequestedFactWrite,
    RunResultWrite,
    RuntimeErrorResultWrite,
    RuntimeErrorWrite,
)
from fervis.lookup.outcomes.model import (
    AnswerResult,
    FactResult,
    Impossible,
    NeedsClarification,
    NoData,
    Undefined,
)
from fervis.lookup.outcomes.terminal_details import fact_result_terminal_details
from fervis.lookup.answer_rendering import RenderedFact
from fervis.lookup.answer_program.result_projection import EntityKeyValue, ResultValue
from fervis.lookup.orchestration.result import LookupResult
from fervis.lookup.orchestration.request import (
    LineagePorts,
    LookupRequest,
)
from fervis.memory.artifacts import FactOutcome
from fervis.lookup.memory.lineage_artifacts import (
    answered_memory_artifacts,
    terminal_memory_artifacts,
)
from fervis.lookup.lineage.errors import LineagePersistenceUnavailable
from fervis.lookup.lineage.steps import LineageRuntimeStepSink


TERMINAL_FACT_PAYLOAD_SCHEMA = "fervis.fact_terminal"
TERMINAL_FACT_PAYLOAD_SCHEMA_REV = 1

_ERROR_KIND_BY_CODE = {
    ErrorCode.PLANNING_FAILED: RuntimeErrorKind.PLANNING_FAILED,
    ErrorCode.PLAN_VALIDATION_FAILED: RuntimeErrorKind.PLAN_VALIDATION_FAILED,
    ErrorCode.FACT_PLAN_EXECUTION_FAILED: RuntimeErrorKind.FACT_PLAN_EXECUTION_FAILED,
    ErrorCode.FRAMEWORK_ADAPTER_FAILED: RuntimeErrorKind.FRAMEWORK_ADAPTER_FAILED,
    ErrorCode.PROVIDER_RUNTIME_FAILED: RuntimeErrorKind.PROVIDER_RUNTIME_FAILED,
    ErrorCode.LINEAGE_PERSISTENCE_FAILED: RuntimeErrorKind.LINEAGE_PERSISTENCE_FAILED,
    ErrorCode.MAX_BUDGET_EXCEEDED: RuntimeErrorKind.POLICY_LIMIT_EXCEEDED,
}


@dataclass(frozen=True)
class RuntimeErrorTerminal:
    run_id: str
    error_code: str
    message: str
    failed_step_id: str | None = None
    status: str = "FAILED"
    result_data: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)

    def lookup_result(self) -> LookupResult:
        return LookupResult(
            status=self.status,
            error=self.error_code,
            result_data=dict(self.result_data),
            usage=dict(self.usage),
        )


def record_runtime_error_terminal(
    terminal: RuntimeErrorTerminal,
    *,
    recorder: LineageRecorderPort | None,
    lineage_required: bool = False,
) -> None:
    if recorder is None:
        if lineage_required:
            raise LineagePersistenceUnavailable(
                "runtime error lineage requires a lineage recorder"
            )
        return
    result = RunResultWrite(
        run_result_id=lineage_id("run_result", terminal.run_id, "runtime_error"),
        run_id=terminal.run_id,
        result_kind=RunResultKind.RUNTIME_ERROR,
    )
    recorder.record_runtime_error_result(
        RuntimeErrorResultWrite(
            result=result,
            error=RuntimeErrorWrite(
                runtime_error_detail_id=lineage_id(
                    "runtime_error",
                    terminal.run_id,
                    terminal.failed_step_id or "",
                    terminal.error_code,
                ),
                run_id=terminal.run_id,
                run_result_id=result.run_result_id,
                failed_step_id=terminal.failed_step_id,
                error_kind=_runtime_error_kind(terminal.error_code),
                message=terminal.message,
            ),
        )
    )


def runtime_error_terminal_result(
    terminal: RuntimeErrorTerminal,
    *,
    recorder: LineageRecorderPort | None,
    lineage_required: bool = False,
) -> LookupResult:
    try:
        record_runtime_error_terminal(
            terminal,
            recorder=recorder,
            lineage_required=lineage_required,
        )
    except Exception:
        return terminal.lookup_result()
    return terminal.lookup_result()


def record_runtime_error_lineage(
    *,
    request: LookupRequest,
    ports: LineagePorts,
    failed_step_id: str | None,
    error_code: str,
    message: str,
) -> None:
    sink = _lineage_sink(ports)
    if sink is None:
        return
    record_runtime_error_terminal(
        RuntimeErrorTerminal(
            run_id=request.run_id,
            failed_step_id=failed_step_id,
            error_code=error_code,
            message=message,
        ),
        recorder=sink.recorder,
        lineage_required=getattr(ports, "lineage_required", False),
    )


def record_lookup_result_lineage(
    *,
    request: LookupRequest,
    ports: LineagePorts,
    fact_result: FactResult,
    rendered: RenderedFact,
    answer: str,
    question_contract: Any,
    question_contract_step_id: str | None,
    compile_step_id: str | None,
    execute_step_id: str | None,
    render_step_id: str | None,
    proof_graph: Any,
    answer_plan: Any,
    proof_node_refs_by_result_output_id: dict[str, tuple[str, ...]],
    grounded_values: tuple[Any, ...] = (),
    extra_fact_addresses: tuple[Any, ...] = (),
    known_input_step_id: str | None = None,
    conversation_resolution_activation: dict[str, Any] | None = None,
) -> None:
    if isinstance(fact_result.outcome, NeedsClarification):
        record_clarification_wait_lineage(
            request=request,
            ports=ports,
            fact_result=fact_result,
            step_id=(
                render_step_id
                or execute_step_id
                or compile_step_id
                or question_contract_step_id
            ),
        )
        return
    if isinstance(fact_result.outcome, AnswerResult):
        record_answered_result_lineage(
            request=request,
            ports=ports,
            fact_result=fact_result,
            rendered=rendered,
            answer=answer,
            question_contract=question_contract,
            question_contract_step_id=question_contract_step_id,
            compile_step_id=compile_step_id,
            execute_step_id=execute_step_id,
            render_step_id=render_step_id,
            proof_graph=proof_graph,
            answer_plan=answer_plan,
            proof_node_refs_by_result_output_id=proof_node_refs_by_result_output_id,
            grounded_values=grounded_values,
            extra_fact_addresses=extra_fact_addresses,
            known_input_step_id=known_input_step_id,
            conversation_resolution_activation=conversation_resolution_activation,
        )
        return
    record_terminal_result_lineage(
        request=request,
        ports=ports,
        fact_result=fact_result,
        question_contract=question_contract,
        question_contract_step_id=question_contract_step_id,
        compile_step_id=compile_step_id,
        execute_step_id=execute_step_id,
        render_step_id=render_step_id,
        proof_graph=proof_graph,
        conversation_resolution_activation=conversation_resolution_activation,
    )


def record_clarification_wait_lineage(
    *,
    request: LookupRequest,
    ports: LineagePorts,
    fact_result: FactResult,
    step_id: str | None,
) -> None:
    sink = _lineage_sink(ports)
    if sink is None:
        return
    if step_id is None:
        raise ValueError("clarification wait requires a producing step")
    clarifications = _clarification_requests(
        run_id=request.run_id,
        fact_result=fact_result,
        step_id=step_id,
    )
    for clarification in clarifications:
        sink.recorder.record_clarification_request(clarification)


def record_answered_result_lineage(
    *,
    request: LookupRequest,
    ports: LineagePorts,
    fact_result: FactResult,
    rendered: RenderedFact,
    answer: str,
    question_contract: Any,
    question_contract_step_id: str | None,
    compile_step_id: str | None,
    execute_step_id: str | None,
    render_step_id: str | None,
    proof_graph: Any,
    answer_plan: Any,
    proof_node_refs_by_result_output_id: dict[str, tuple[str, ...]],
    grounded_values: tuple[Any, ...] = (),
    extra_fact_addresses: tuple[Any, ...] = (),
    known_input_step_id: str | None = None,
    conversation_resolution_activation: dict[str, Any] | None = None,
) -> None:
    sink = _lineage_sink(ports)
    if sink is None:
        return
    if not isinstance(fact_result.outcome, AnswerResult):
        raise ValueError("answered lineage requires an answer result")
    _require_answered_lineage_inputs(
        question_contract=question_contract,
        question_contract_step_id=question_contract_step_id,
        compile_step_id=compile_step_id,
        render_step_id=render_step_id,
        proof_graph=proof_graph,
        answer_plan=answer_plan,
    )
    assert question_contract_step_id is not None
    assert compile_step_id is not None
    assert render_step_id is not None

    run_id = request.run_id
    result = RunResultWrite(
        run_result_id=lineage_id("run_result", run_id, "answered"),
        run_id=run_id,
        result_kind=RunResultKind.ANSWERED,
    )
    requested_facts = tuple(
        _requested_fact_write(
            run_id=run_id,
            fact=fact,
            question_contract_step_id=question_contract_step_id,
        )
        for fact in question_contract.requested_facts
    )
    evidence_refs_by_fact_key = _evidence_refs_by_fact_key(
        answer_plan=answer_plan,
        proof_graph=proof_graph,
        proof_node_refs_by_result_output_id=proof_node_refs_by_result_output_id,
    )
    fact_results = _fact_results(
        run_id=run_id,
        requested_facts=requested_facts,
        answer_plan=answer_plan,
        produced_by_step_id=execute_step_id or compile_step_id,
        evidence_refs_by_fact_key=evidence_refs_by_fact_key,
    )
    proofs = _execution_proofs(
        run_id=run_id,
        fact_results=fact_results,
        compile_step_id=compile_step_id,
        execute_step_id=execute_step_id,
        proof_graph=proof_graph,
        proof_result_kinds=(FactResultKind.ANSWERED,),
    )
    lineage_answer = AnswerWrite(
        answer_id=lineage_id("answer", run_id),
        run_id=run_id,
        run_result_id=result.run_result_id,
    )
    outputs = _answer_outputs(
        run_id=run_id,
        answer_id=lineage_answer.answer_id,
        fact_result_id_by_fact_key={
            requested_fact.fact_key: fact_result.fact_result_id
            for requested_fact, fact_result in zip(
                _fulfilled_requested_facts(
                    requested_facts=requested_facts,
                    answer_plan=answer_plan,
                ),
                fact_results,
                strict=True,
            )
        },
        answer_plan=answer_plan,
        fact_result=fact_result,
        rendered=rendered,
        proof_node_refs_by_result_output_id=proof_node_refs_by_result_output_id,
    )
    if not outputs:
        raise ValueError("answered lineage requires answer outputs")
    presentation = AnswerPresentationWrite(
        presentation_id=lineage_id("presentation", run_id, lineage_answer.answer_id),
        run_id=run_id,
        answer_id=lineage_answer.answer_id,
        presentation_kind=PresentationKind.TEXT,
        render_step_id=render_step_id,
        rendered_value=answer,
    )
    memory_artifacts = answered_memory_artifacts(
        run_id=run_id,
        fact_result=fact_result,
        requested_facts=requested_facts,
        fact_results=fact_results,
        question_contract=question_contract,
        answer_plan=answer_plan,
        grounded_values=grounded_values,
        extra_fact_addresses=extra_fact_addresses,
        known_input_step_id=known_input_step_id,
        source_question=request.question,
        source_answer=answer,
        conversation_resolution_activation=conversation_resolution_activation,
    )
    sink.recorder.record_answered_result(
        AnsweredRunResultWrite(
            result=result,
            requested_facts=requested_facts,
            fact_results=fact_results,
            proof_graphs=proofs,
            answer=lineage_answer,
            outputs=outputs,
            presentations=(presentation,),
            memory_artifacts=memory_artifacts,
        )
    )


def record_terminal_result_lineage(
    *,
    request: LookupRequest,
    ports: LineagePorts,
    fact_result: FactResult,
    question_contract: Any,
    question_contract_step_id: str | None,
    compile_step_id: str | None,
    execute_step_id: str | None,
    render_step_id: str | None,
    proof_graph: Any = None,
    conversation_resolution_activation: dict[str, Any] | None = None,
) -> None:
    sink = _lineage_sink(ports)
    if sink is None:
        return
    run_id = request.run_id
    result = RunResultWrite(
        run_result_id=lineage_id("run_result", run_id, "factual_terminal"),
        run_id=run_id,
        result_kind=RunResultKind.FACTUAL_TERMINAL,
    )
    requested_facts = _terminal_requested_facts(
        run_id=run_id,
        question_contract=question_contract,
        question_contract_step_id=question_contract_step_id,
        fact_result=fact_result,
        render_step_id=render_step_id,
    )
    fact_results = _terminal_fact_results(
        run_id=run_id,
        requested_facts=requested_facts,
        produced_by_step_id=render_step_id or question_contract_step_id,
        fact_result=fact_result,
    )
    memory_artifacts = terminal_memory_artifacts(
        run_id=run_id,
        fact_result=fact_result,
        requested_facts=requested_facts,
        fact_results=fact_results,
        question_contract=question_contract,
        produced_by_step_id=render_step_id or question_contract_step_id,
        outcome=FactOutcome(_terminal_fact_result_kind(fact_result).value),
        source_question=request.question,
        conversation_resolution_activation=conversation_resolution_activation,
    )
    proofs = _execution_proofs(
        run_id=run_id,
        fact_results=fact_results,
        compile_step_id=compile_step_id,
        execute_step_id=execute_step_id,
        proof_graph=proof_graph,
        proof_result_kinds=(FactResultKind.NO_DATA, FactResultKind.UNDEFINED),
    )
    sink.recorder.record_factual_terminal_result(
        FactualTerminalRunResultWrite(
            result=result,
            requested_facts=requested_facts,
            fact_results=fact_results,
            proof_graphs=proofs,
            memory_artifacts=memory_artifacts,
        )
    )


def _runtime_error_kind(error_code: str) -> RuntimeErrorKind:
    return _ERROR_KIND_BY_CODE.get(error_code, RuntimeErrorKind.INFRASTRUCTURE_FAILED)


def _terminal_requested_facts(
    *,
    run_id: str,
    question_contract: Any,
    question_contract_step_id: str | None,
    fact_result: FactResult,
    render_step_id: str | None,
) -> tuple[RequestedFactWrite, ...]:
    if question_contract is None:
        return ()
    produced_by_step_id = question_contract_step_id or render_step_id
    if produced_by_step_id is None:
        raise ValueError("terminal lineage with requested facts requires a step id")
    return tuple(
        _requested_fact_write(
            run_id=run_id,
            fact=fact,
            question_contract_step_id=produced_by_step_id,
        )
        for fact in question_contract.requested_facts
        if _terminal_applies_to_requested_fact(fact_result, fact.id)
    )


def _terminal_fact_results(
    *,
    run_id: str,
    requested_facts: tuple[RequestedFactWrite, ...],
    produced_by_step_id: str | None,
    fact_result: FactResult,
) -> tuple[FactResultWrite, ...]:
    if not requested_facts:
        return ()
    if produced_by_step_id is None:
        raise ValueError("terminal fact results require a producing step id")
    return tuple(
        FactResultWrite(
            fact_result_id=lineage_id("fact_result", run_id, fact.fact_key),
            run_id=run_id,
            requested_fact_id=fact.requested_fact_id,
            produced_by_step_id=produced_by_step_id,
            result_kind=_terminal_fact_result_kind(fact_result),
            evidence_refs_json=list(_terminal_proof_refs(fact_result)),
            payload_schema=TERMINAL_FACT_PAYLOAD_SCHEMA,
            payload_schema_rev=TERMINAL_FACT_PAYLOAD_SCHEMA_REV,
            payload_json=_terminal_fact_result_payload(
                fact_result,
            ),
        )
        for fact in requested_facts
    )


def _terminal_fact_result_payload(
    fact_result: FactResult,
) -> dict[str, object]:
    return dict(fact_result_terminal_details(fact_result) or {})


def _execution_proofs(
    *,
    run_id: str,
    fact_results: tuple[FactResultWrite, ...],
    compile_step_id: str | None,
    execute_step_id: str | None,
    proof_graph: Any,
    proof_result_kinds: tuple[FactResultKind, ...],
) -> tuple[ExecutionProofGraphWrite, ...]:
    if proof_graph is None or compile_step_id is None:
        return ()
    proof_result_kind_set = set(proof_result_kinds)
    return tuple(
        _proof_graph_write(
            run_id=run_id,
            fact_result_id=fact.fact_result_id,
            compile_step_id=compile_step_id,
            execute_step_id=execute_step_id,
            proof_graph=proof_graph,
        )
        for fact in fact_results
        if fact.result_kind in proof_result_kind_set
    )


def _clarification_requests(
    *,
    run_id: str,
    fact_result: FactResult,
    step_id: str,
) -> tuple[ClarificationRequestWrite, ...]:
    outcome = fact_result.outcome
    if not isinstance(outcome, NeedsClarification):
        return ()
    clarifications: list[ClarificationRequestWrite] = []
    for item in outcome.clarifications:
        clarification_id = _clarification_request_id(run_id, item)
        clarifications.append(
            ClarificationRequestWrite(
                clarification_id=clarification_id,
                run_id=run_id,
                step_id=step_id,
                payload_json=clarification_payload(replace(item, id=clarification_id)),
            )
        )
    return tuple(clarifications)


def _clarification_request_id(run_id: str, item: Any) -> str:
    return lineage_id("clarification", run_id, item.id)


def _terminal_fact_result_kind(fact_result: FactResult) -> FactResultKind:
    outcome = fact_result.outcome
    if isinstance(outcome, Impossible):
        return FactResultKind.IMPOSSIBLE
    if isinstance(outcome, NoData):
        return FactResultKind.NO_DATA
    if isinstance(outcome, Undefined):
        return FactResultKind.UNDEFINED
    raise ValueError(f"unsupported terminal fact result {type(outcome).__name__}")


def _terminal_applies_to_requested_fact(
    fact_result: FactResult,
    requested_fact_id: str,
) -> bool:
    applicable = _terminal_requested_fact_ids(fact_result)
    return not applicable or requested_fact_id in applicable


def _terminal_requested_fact_ids(fact_result: FactResult) -> set[str]:
    outcome = fact_result.outcome
    if isinstance(outcome, Impossible):
        return {
            item.requested_fact_id
            for item in outcome.blocked_requirements
            if item.requested_fact_id
        }
    if isinstance(outcome, NoData):
        return set(outcome.empty_relation.requested_fact_ids)
    if isinstance(outcome, Undefined):
        return set()
    return set()


def _terminal_proof_refs(fact_result: FactResult) -> tuple[str, ...]:
    outcome = fact_result.outcome
    refs = list(getattr(outcome, "proof_refs", ()))
    if isinstance(outcome, Impossible):
        for requirement in outcome.blocked_requirements:
            refs.extend(requirement.proof_refs)
    if isinstance(outcome, NoData):
        refs.extend(outcome.empty_relation.proof_refs)
    if isinstance(outcome, Undefined):
        refs.extend(outcome.operation.proof_refs)
    return tuple(dict.fromkeys(ref for ref in refs if ref))


def _lineage_sink(ports: LineagePorts) -> LineageRuntimeStepSink | None:
    sink = ports.lineage_step_sink
    if sink is None and getattr(ports, "lineage_required", False):
        raise LineagePersistenceUnavailable(
            "lookup result lineage requires a lineage_step_sink"
        )
    return sink


def _require_answered_lineage_inputs(
    *,
    question_contract: Any,
    question_contract_step_id: str | None,
    compile_step_id: str | None,
    render_step_id: str | None,
    proof_graph: Any,
    answer_plan: Any,
) -> None:
    missing = [
        name
        for name, value in {
            "question_contract": question_contract,
            "question_contract_step_id": question_contract_step_id,
            "compile_step_id": compile_step_id,
            "render_step_id": render_step_id,
            "proof_graph": proof_graph,
            "answer_plan": answer_plan,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"answered lineage requires {', '.join(missing)}")


def _fact_results(
    *,
    run_id: str,
    requested_facts: tuple[RequestedFactWrite, ...],
    answer_plan: Any,
    produced_by_step_id: str,
    evidence_refs_by_fact_key: dict[str, list[str]],
) -> tuple[FactResultWrite, ...]:
    requested_facts = _fulfilled_requested_facts(
        requested_facts=requested_facts,
        answer_plan=answer_plan,
    )
    return tuple(
        FactResultWrite(
            fact_result_id=lineage_id("fact_result", run_id, fact.fact_key),
            run_id=run_id,
            requested_fact_id=fact.requested_fact_id,
            produced_by_step_id=produced_by_step_id,
            result_kind=FactResultKind.ANSWERED,
            evidence_refs_json=evidence_refs_by_fact_key.get(fact.fact_key, []),
        )
        for fact in requested_facts
    )


def _evidence_refs_by_fact_key(
    *,
    answer_plan: Any,
    proof_graph: Any,
    proof_node_refs_by_result_output_id: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    payload = read_execution_proof_graph_payload(
        payload_schema=EXECUTION_PROOF_GRAPH_SCHEMA,
        payload_schema_rev=EXECUTION_PROOF_GRAPH_SCHEMA_REV,
        payload_json=execution_proof_graph_payload(proof_graph),
    )
    output: dict[str, list[str]] = {}
    roots_by_fact_key: dict[str, list[str]] = {}
    for fulfillment in answer_plan.fulfillment:
        roots_by_fact_key.setdefault(fulfillment.requested_fact_id, []).extend(
            proof_node_refs_by_result_output_id.get(fulfillment.result_output_id, ())
        )
    for fact_key, roots in roots_by_fact_key.items():
        projected = project_proof_payload(
            payload,
            target_node_ids=tuple(dict.fromkeys(roots)),
        )
        refs = [ref for node in projected.nodes for ref in node.proof_refs if ref]
        output[fact_key] = list(dict.fromkeys(refs))
    return output


def _fulfilled_requested_facts(
    *,
    requested_facts: tuple[RequestedFactWrite, ...],
    answer_plan: Any,
) -> tuple[RequestedFactWrite, ...]:
    fact_write_by_key = {fact.fact_key: fact for fact in requested_facts}
    requested_fact_ids = tuple(
        dict.fromkeys(
            fulfillment.requested_fact_id
            for fulfillment in answer_plan.fulfillment
            if fulfillment.requested_fact_id
        )
    )
    if not requested_fact_ids:
        raise ValueError("answered lineage requires fulfilled requested facts")
    output: list[RequestedFactWrite] = []
    for requested_fact_id in requested_fact_ids:
        requested_fact = fact_write_by_key.get(requested_fact_id)
        if requested_fact is None:
            raise ValueError(
                f"answered lineage references unknown requested fact {requested_fact_id!r}"
            )
        output.append(requested_fact)
    return tuple(output)


def _proof_graph_write(
    *,
    run_id: str,
    fact_result_id: str,
    compile_step_id: str,
    execute_step_id: str | None,
    proof_graph: Any,
) -> ExecutionProofGraphWrite:
    return ExecutionProofGraphWrite(
        proof_graph_id=lineage_id("proof_graph", run_id, fact_result_id),
        run_id=run_id,
        fact_result_id=fact_result_id,
        compile_step_id=compile_step_id,
        execute_step_id=execute_step_id,
        payload_schema=EXECUTION_PROOF_GRAPH_SCHEMA,
        payload_schema_rev=EXECUTION_PROOF_GRAPH_SCHEMA_REV,
        payload_json=execution_proof_graph_payload(proof_graph),
    )


def _requested_fact_write(
    *,
    run_id: str,
    fact: Any,
    question_contract_step_id: str,
) -> RequestedFactWrite:
    answer_expression = getattr(fact, "answer_expression", None)
    answer_expression_family = getattr(answer_expression, "family", "")
    return RequestedFactWrite(
        requested_fact_id=lineage_id("requested_fact", run_id, fact.id),
        run_id=run_id,
        produced_by_step_id=question_contract_step_id,
        fact_key=fact.id,
        description=fact.description,
        answer_expression_family=str(getattr(answer_expression_family, "value", "")),
        requested_fact_json=fact.answer_request_model_dict(),
        answer_requests_json={
            "answer_outputs": [
                output.to_model_dict() for output in fact.answer_outputs
            ],
        },
    )


def _answer_outputs(
    *,
    run_id: str,
    answer_id: str,
    fact_result_id_by_fact_key: dict[str, str],
    answer_plan: Any,
    fact_result: FactResult,
    rendered: RenderedFact,
    proof_node_refs_by_result_output_id: dict[str, tuple[str, ...]],
) -> tuple[AnswerOutputWrite, ...]:
    return tuple(
        AnswerOutputWrite(
            answer_output_id=lineage_id(
                "answer_output",
                run_id,
                answer_id,
                fulfillment.requested_fact_id,
                fulfillment.answer_output_id,
            ),
            run_id=run_id,
            answer_id=answer_id,
            fact_result_id=fact_result_id_by_fact_key[fulfillment.requested_fact_id],
            output_key=fulfillment.answer_output_id,
            value_kind=value_kind,
            value_json=value_json,
            proof_node_refs_json=proof_refs,
        )
        for fulfillment, value_kind, value_json, proof_refs in _fulfilled_output_values(
            answer_plan=answer_plan,
            fact_result=fact_result,
            rendered=rendered,
            proof_node_refs_by_result_output_id=proof_node_refs_by_result_output_id,
        )
    )


def _fulfilled_output_values(
    *,
    answer_plan: Any,
    fact_result: FactResult,
    rendered: RenderedFact,
    proof_node_refs_by_result_output_id: dict[str, tuple[str, ...]],
) -> tuple[tuple[Any, AnswerValueKind, dict[str, Any], list[str]], ...]:
    values: list[tuple[Any, AnswerValueKind, dict[str, Any], list[str]]] = []
    for fulfillment in answer_plan.fulfillment:
        value_kind, value_json = _execution_lineage_value(
            fact_result,
            fulfillment.result_output_id,
        )
        if value_kind is None:
            value = _rendered_value_for_key(rendered, fulfillment.result_output_id)
            value_kind, value_json = _lineage_value(value)
        proof_refs = list(
            proof_node_refs_by_result_output_id.get(
                fulfillment.result_output_id,
                (),
            )
        )
        if not proof_refs:
            raise ValueError(
                f"answered lineage missing proof node refs for result output {fulfillment.result_output_id!r}"
            )
        values.append(
            (
                fulfillment,
                value_kind,
                value_json,
                proof_refs,
            )
        )
    return tuple(values)


def _execution_lineage_value(
    fact_result: FactResult,
    result_output_id: str,
) -> tuple[AnswerValueKind | None, dict[str, Any]]:
    outcome = fact_result.outcome
    if not isinstance(outcome, AnswerResult):
        return None, {}
    for relation_output in outcome.result_projection.relation_outputs:
        if relation_output.id != result_output_id:
            continue
        values = tuple(
            row.values[result_output_id]
            for row in outcome.projected_rows
            if result_output_id in row.values
        )
        return _projected_lineage_values(values)
    for scalar_output in outcome.result_projection.scalar_outputs:
        if scalar_output.id != result_output_id:
            continue
        if outcome.scalars and scalar_output.scalar_id in outcome.scalars:
            return _lineage_value(outcome.scalars[scalar_output.scalar_id])
    return None, {}


def _projected_lineage_values(
    values: tuple[ResultValue, ...],
) -> tuple[AnswerValueKind | None, dict[str, Any]]:
    if len(values) == 1:
        return _lineage_value(values[0])
    if values:
        return _lineage_value(values)
    return None, {}


def _rendered_value_for_key(rendered: RenderedFact, key: str) -> object:
    row_values = [row[key] for row in rendered.rows if key in row]
    if len(row_values) == 1:
        return row_values[0]
    if len(row_values) > 1:
        return row_values
    if rendered.scalars and key in rendered.scalars:
        return rendered.scalars[key]
    raise ValueError(f"answered lineage render output {key!r} is unavailable")


def _lineage_value(value: object) -> tuple[AnswerValueKind, dict[str, Any]]:
    if isinstance(value, EntityKeyValue):
        return AnswerValueKind.ENTITY, _entity_key_json(value)
    if isinstance(value, bool):
        return AnswerValueKind.BOOLEAN, {"kind": "boolean", "value": value}
    if isinstance(value, int | float | Decimal) and not isinstance(value, bool):
        return AnswerValueKind.NUMBER, {"kind": "number", "value": str(value)}
    if isinstance(value, list | tuple):
        return (
            AnswerValueKind.LIST,
            {"kind": "list", "values": [_json_safe(item) for item in value]},
        )
    if isinstance(value, Mapping):
        return AnswerValueKind.OBJECT, {"kind": "object", "value": _json_safe(value)}
    return (
        AnswerValueKind.TEXT,
        {"kind": "text", "value": "" if value is None else str(value)},
    )


def _json_safe(value: object) -> object:
    if isinstance(value, EntityKeyValue):
        return _entity_key_json(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _entity_key_json(value: EntityKeyValue) -> dict[str, object]:
    components = {
        component.component_id: _json_safe(component.value)
        for component in value.components
    }
    return {
        "kind": "entity",
        "entity_kind": value.entity_kind,
        "key_id": value.key_id,
        "components": components,
    }
