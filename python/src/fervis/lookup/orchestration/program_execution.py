"""One answer-program execution path for initial answers and deterministic reruns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.answer_program.instantiation import (
    ExecutionEnvironment,
    VerifiedExecution,
)
from fervis.lookup.answer_program.invocation import (
    RuntimePorts as AnswerProgramRuntimePorts,
    invoke_answer_program,
)
from fervis.lookup.answer_program.persistence import (
    ProgramInvocation,
    ProgramInvocationBinding,
)
from fervis.lineage.enums import ProgramInvocationKind
from fervis.lookup.errors import ErrorCode
from fervis.lookup.lineage.source_read_buffer import buffered_source_read_lineage
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.lineage.results import record_runtime_error_lineage
from fervis.lookup.lineage.steps import (
    compile_step_id,
    execution_step_id,
    lineage_error_json,
    record_compile_step,
    record_execution_step,
)
from fervis.lookup.orchestration.request import LookupRequest
from fervis.lookup.orchestration.result import LookupResult, RunStatus
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.question_contract import QuestionContract
from fervis.observability.event_contracts import EventPayloadKey

from .result_synthesis import _synthesize_result
from .terminal_results import _execution_failure_payload, _status_for_fact_result


@dataclass(frozen=True)
class ProgramExecutionPorts:
    data_access_port: Any
    memory: LookupMemory = field(default_factory=LookupMemory)
    lineage_step_sink: Any = None
    lineage_required: bool = False


@dataclass(frozen=True)
class _RecordCompileInvocationBinding:
    binding: ProgramInvocationBinding
    ports: ProgramExecutionPorts

    def bind(
        self,
        execution: VerifiedExecution,
        *,
        kind: ProgramInvocationKind,
        base_invocation_id: str | None,
    ) -> ProgramInvocation:
        invocation = self.binding.bind(
            execution,
            kind=kind,
            base_invocation_id=base_invocation_id,
        )
        record_compile_step(
            self.ports,
            program_id=invocation.program_id,
            invocation_id=invocation.invocation_id,
            proof_node_count=len(execution.proof_graph.nodes),
            proof_edge_count=len(execution.proof_graph.edges),
        )
        return invocation


def run_answer_program_execution(
    *,
    request: LookupRequest,
    ports: ProgramExecutionPorts,
    program: AnswerProgram,
    bindings: BindingSet,
    environment: ExecutionEnvironment,
    invocation_binding: ProgramInvocationBinding | None,
    question_contract_step_id: str,
    usage: dict[str, Any] | None = None,
    grounded_values: tuple[Any, ...] = (),
    extra_fact_addresses: tuple[Any, ...] = (),
    known_input_step_id: str | None = None,
    conversation_resolution_activation: dict[str, Any] | None = None,
    invocation_kind: ProgramInvocationKind = ProgramInvocationKind.COMPILED_QUESTION,
    base_invocation_id: str | None = None,
) -> LookupResult:
    execution_lineage = buffered_source_read_lineage(
        run_id=request.run_id,
        step_id=execution_step_id(ports),
    )
    try:
        execution = invoke_answer_program(
            program=program,
            bindings=bindings,
            environment=environment,
            ports=AnswerProgramRuntimePorts(
                data_access_port=ports.data_access_port,
                memory=ports.memory,
                source_read_lineage=execution_lineage.scope,
                invocation_binding=(
                    _RecordCompileInvocationBinding(
                        binding=invocation_binding,
                        ports=ports,
                    )
                    if invocation_binding is not None
                    else None
                ),
                invocation_kind=invocation_kind,
                base_invocation_id=base_invocation_id,
            ),
        )
    except VerificationError as exc:
        return _failed_execution(
            request=request,
            ports=ports,
            execution_lineage=execution_lineage,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            exc=exc,
            usage=usage or {},
        )
    except Exception as exc:
        return _failed_execution(
            request=request,
            ports=ports,
            execution_lineage=execution_lineage,
            error_code=ErrorCode.FACT_PLAN_EXECUTION_FAILED,
            exc=exc,
            usage=usage or {},
        )
    execution_payload = {
        EventPayloadKey.RUN_ID: request.run_id,
        EventPayloadKey.RELATION_COUNT: len(execution.relations),
    }
    if execution.issue is not None:
        execution_payload.update(
            {
                EventPayloadKey.ERROR_CODE: execution.issue.kind.value,
                EventPayloadKey.ERROR_CLASS: execution.issue.__class__.__name__,
                EventPayloadKey.ERROR_CONTEXT: execution.issue.message,
            }
        )
    record_execution_step(
        ports,
        program_id=execution.program_id,
        invocation_id=execution.invocation_id,
        relation_count=len(execution.relations),
        proof_refs=tuple(str(item) for item in execution.proof_refs),
        error_json=lineage_error_json(execution_payload),
        catalog_endpoints=execution_lineage.catalog_endpoints,
        source_reads=execution_lineage.source_reads,
    )
    if execution.issue is not None or execution.fact_result is None:
        error_code = (
            execution.issue.kind.value
            if execution.issue is not None
            else ErrorCode.FACT_PLAN_EXECUTION_FAILED
        )
        record_runtime_error_lineage(
            request=request,
            ports=ports,
            failed_step_id=execution_step_id(ports),
            error_code=error_code,
            message=(
                execution.issue.message
                if execution.issue is not None
                else ErrorCode.FACT_PLAN_EXECUTION_FAILED
            ),
        )
        return LookupResult(
            status=RunStatus.FAILED, error=error_code, usage=usage or {}
        )
    return _synthesize_result(
        request=request,
        ports=ports,
        fact_result=execution.fact_result,
        status=_status_for_fact_result(execution.fact_result),
        usage=usage or {},
        question_contract=QuestionContract(
            requested_facts=execution.effective_requested_facts
        ),
        grounded_values=grounded_values,
        extra_fact_addresses=extra_fact_addresses,
        known_input_step_id=known_input_step_id,
        question_contract_step_id=question_contract_step_id,
        compile_step_id=compile_step_id(ports),
        execute_step_id=execution_step_id(ports),
        proof_graph=execution.proof_graph,
        answer_plan=execution.program,
        proof_node_refs_by_result_output_id=(
            execution.proof_node_refs_by_result_output_id
        ),
        conversation_resolution_activation=conversation_resolution_activation,
    )


def _failed_execution(
    *,
    request: LookupRequest,
    ports: ProgramExecutionPorts,
    execution_lineage,
    error_code: str,
    exc: Exception,
    usage: dict[str, Any],
) -> LookupResult:
    payload = _execution_failure_payload(
        request=request,
        error_code=error_code,
        exc=exc,
    )
    failed_step = record_execution_step(
        ports,
        error_json=lineage_error_json(payload),
        catalog_endpoints=execution_lineage.catalog_endpoints,
        source_reads=execution_lineage.source_reads,
    )
    record_runtime_error_lineage(
        request=request,
        ports=ports,
        failed_step_id=failed_step.step_id if failed_step is not None else None,
        error_code=error_code,
        message=str(exc),
    )
    return LookupResult(status=RunStatus.FAILED, error=error_code, usage=usage)
