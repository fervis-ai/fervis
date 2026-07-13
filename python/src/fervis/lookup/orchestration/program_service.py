"""Model-free execution service for persisted answer-program invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.host_api.context import HostApiContext
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.enums import ProgramInvocationKind
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.persistence import (
    StoredProgramInvocation,
    StoredProgramInvocationBinding,
)
from fervis.lookup.lineage.steps import (
    LineageRuntimeStepSink,
    record_program_contract_step,
)
from fervis.lookup.memory.outcomes import fact_value_memory_addresses
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.orchestration.host_runtime import (
    HostRelationDataAccess,
    host_relation_catalog,
)
from fervis.lookup.orchestration.program_execution import (
    ProgramExecutionPorts,
    run_answer_program_execution,
)
from fervis.lookup.orchestration.request import LookupRequest
from fervis.lookup.orchestration.result import LookupResult
from fervis.lookup.plan_execution.authorized_sources import AuthorizedExecutionSources
from fervis.lookup.relation_catalog import parse_relation_catalog


@dataclass(frozen=True)
class StoredProgramRunRequest:
    run_id: str
    conversation_id: str
    tenant_id: str
    question: str
    read_context_ref: ReadContextRef
    principal: Any
    invocation: StoredProgramInvocation
    runtime_context: dict[str, Any] = field(default_factory=dict)
    active_attempt: int | None = None
    delegated_credential: DelegatedReadCredential | None = None


@dataclass(frozen=True)
class AnswerProgramService:
    host_api_context: HostApiContext
    lineage_recorder: LineageRecorderPort

    def run_program(self, request: StoredProgramRunRequest) -> LookupResult:
        authority = ReadAuthority(
            tenant_id=request.tenant_id,
            read_context_ref=request.read_context_ref,
            delegated_credential=request.delegated_credential,
        )
        full_catalog = parse_relation_catalog(
            host_relation_catalog(self.host_api_context)
        )
        program = request.invocation.program
        authorized_sources = AuthorizedExecutionSources.from_program(
            full_catalog=full_catalog,
            program=program,
        )
        lineage_sink = LineageRuntimeStepSink(
            run_id=request.run_id,
            recorder=self.lineage_recorder,
            attempt=request.active_attempt,
        )
        ports = ProgramExecutionPorts(
            data_access_port=HostRelationDataAccess(
                host_api_context=self.host_api_context,
                authority=authority,
            ),
            memory=LookupMemory(),
            lineage_step_sink=lineage_sink,
            lineage_required=True,
        )
        contract_step = record_program_contract_step(
            ports,
            program_id=request.invocation.invocation.program_id,
            invocation_id=request.invocation.invocation.invocation_id,
            requested_fact_count=len(program.fact_template),
        )
        if contract_step is None:
            raise RuntimeError("deterministic program contract lineage is required")
        bindings = request.invocation.bindings
        bound_values = tuple(binding.value for binding in bindings.bindings)
        return run_answer_program_execution(
            request=LookupRequest(
                question=request.question,
                run_id=request.run_id,
                tenant_id=request.tenant_id,
                authority_ref=authority.evidence_ref,
                user_context={
                    **dict(request.runtime_context),
                    "conversationId": request.conversation_id,
                },
                active_attempt=request.active_attempt,
            ),
            ports=ports,
            program=program,
            bindings=bindings,
            environment=ExecutionEnvironment(
                catalog=authorized_sources.relation_catalog,
                authorized_sources=authorized_sources,
                memory_relations=(),
                authority_ref=authority.evidence_ref,
            ),
            invocation_binding=StoredProgramInvocationBinding(request.invocation),
            question_contract_step_id=contract_step.step_id,
            grounded_values=bound_values,
            extra_fact_addresses=fact_value_memory_addresses(bound_values),
            known_input_step_id=contract_step.step_id,
            invocation_kind=ProgramInvocationKind.RERUN_PROGRAM,
            base_invocation_id=request.invocation.invocation.base_invocation_id,
        )
