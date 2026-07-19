"""Framework-neutral answer lineage ports."""

from __future__ import annotations

from typing import Protocol

from fervis.lineage.recorder import (
    AnswerOutputWrite,
    AnswerPresentationWrite,
    AnswerWrite,
    AnsweredRunResultWrite,
    CatalogEndpointWrite,
    ClarificationRequestWrite,
    ClarificationResponseWrite,
    ConversationWrite,
    ExecutionProofGraphWrite,
    FactualTerminalRunResultWrite,
    FactResultWrite,
    MemoryArtifactWrite,
    ModelCallAuditWrite,
    ModelCallUsageWrite,
    ModelCallWrite,
    QuestionRunWrite,
    ProgramInvocationBundleWrite,
    ProgramRevisionBundleWrite,
    QuestionWrite,
    RequestedFactWrite,
    RunArtifactWrite,
    RunResultWrite,
    RunStepWrite,
    RuntimeErrorResultWrite,
    SourceReadWrite,
)


class SourceReadRecorderPort(Protocol):
    def record_catalog_endpoint(
        self, catalog_endpoint: CatalogEndpointWrite
    ) -> CatalogEndpointWrite: ...

    def record_source_read(self, source_read: SourceReadWrite) -> SourceReadWrite: ...

    def record_artifact(self, artifact: RunArtifactWrite) -> RunArtifactWrite: ...


class LineageRecorderPort(Protocol):
    def ensure_conversation(
        self, conversation: ConversationWrite
    ) -> ConversationWrite: ...

    def record_question(self, question: QuestionWrite) -> QuestionWrite: ...

    def start_run(self, run: QuestionRunWrite) -> QuestionRunWrite: ...

    def record_program_invocation(
        self, bundle: ProgramInvocationBundleWrite
    ) -> ProgramInvocationBundleWrite: ...

    def record_program_revision(
        self, bundle: ProgramRevisionBundleWrite
    ) -> ProgramRevisionBundleWrite: ...

    def record_step(self, step: RunStepWrite) -> RunStepWrite: ...

    def record_step_with_source_context(
        self,
        step: RunStepWrite,
        catalog_endpoints: tuple[CatalogEndpointWrite, ...],
        source_reads: tuple[SourceReadWrite, ...],
        artifacts: tuple[RunArtifactWrite, ...],
    ) -> RunStepWrite: ...

    def record_model_call(self, model_call: ModelCallWrite) -> ModelCallWrite: ...

    def record_model_call_audit(
        self, audit: ModelCallAuditWrite
    ) -> ModelCallAuditWrite: ...

    def record_model_call_usage(
        self, usage: ModelCallUsageWrite
    ) -> ModelCallUsageWrite: ...

    def record_catalog_endpoint(
        self, catalog_endpoint: CatalogEndpointWrite
    ) -> CatalogEndpointWrite: ...

    def record_source_read(self, source_read: SourceReadWrite) -> SourceReadWrite: ...

    def record_artifact(self, artifact: RunArtifactWrite) -> RunArtifactWrite: ...

    def record_run_result(self, result: RunResultWrite) -> RunResultWrite: ...

    def record_runtime_error_result(
        self, runtime_error: RuntimeErrorResultWrite
    ) -> RuntimeErrorResultWrite: ...

    def record_answered_result(
        self, answered_result: AnsweredRunResultWrite
    ) -> AnsweredRunResultWrite: ...

    def record_factual_terminal_result(
        self, terminal_result: FactualTerminalRunResultWrite
    ) -> FactualTerminalRunResultWrite: ...

    def record_requested_fact(
        self, requested_fact: RequestedFactWrite
    ) -> RequestedFactWrite: ...

    def record_fact_result(self, fact_result: FactResultWrite) -> FactResultWrite: ...

    def record_memory_artifact(
        self, memory_artifact: MemoryArtifactWrite
    ) -> MemoryArtifactWrite: ...

    def record_clarification_request(
        self, clarification: ClarificationRequestWrite
    ) -> ClarificationRequestWrite: ...

    def record_clarification_response(
        self, response: ClarificationResponseWrite
    ) -> ClarificationResponseWrite: ...

    def record_answer(self, answer: AnswerWrite) -> AnswerWrite: ...

    def record_answer_output(
        self, answer_output: AnswerOutputWrite
    ) -> AnswerOutputWrite: ...

    def record_answer_presentation(
        self, presentation: AnswerPresentationWrite
    ) -> AnswerPresentationWrite: ...

    def record_execution_proof_graph(
        self, proof_graph: ExecutionProofGraphWrite
    ) -> ExecutionProofGraphWrite: ...
