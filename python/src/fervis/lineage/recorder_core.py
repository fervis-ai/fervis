"""Framework-neutral lineage recorder implementation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, TypeVar

from fervis.lineage.enums import (
    ArtifactKind,
    FactResultKind,
    ProofNodeKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    SourceReadStatus,
)
from fervis.lineage.payloads.execution_proof_graph import (
    ProofGraphPayload,
    ProofGraphPayloadNode,
    read_execution_proof_graph_payload,
)
from fervis.lineage import records
from fervis.lineage.records import LineageRow
from fervis.lineage.recorder import (
    AnswerOutputWrite,
    AnswerPresentationWrite,
    AnswerWrite,
    CatalogEndpointWrite,
    AnsweredRunResultWrite,
    ClarificationRequestWrite,
    ClarificationResponseWrite,
    ConversationWrite,
    ExecutionProofGraphWrite,
    FactualTerminalRunResultWrite,
    FactResultWrite,
    LineageRecorderConflict,
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

T = TypeVar("T")


class LineageRecorderStore(Protocol):
    def transaction(self) -> AbstractContextManager[object]: ...

    def get_or_insert_row(self, row: LineageRow) -> LineageRow: ...

    def find_row(
        self,
        *,
        key: str,
        lookup: dict[str, object],
        fields: tuple[str, ...],
    ) -> LineageRow | None: ...

    def insert_row(self, row: LineageRow) -> None: ...


class LineageRecorder:
    def __init__(self, store: LineageRecorderStore) -> None:
        self._store = store

    def ensure_conversation(self, conversation: ConversationWrite) -> ConversationWrite:
        return self._record_idempotent(
            records.CONVERSATION,
            conversation,
            "different lineage root fields",
        )

    def record_question(self, question: QuestionWrite) -> QuestionWrite:
        return self._record_idempotent(
            records.QUESTION,
            question,
            "different lineage root fields",
        )

    def start_run(self, run: QuestionRunWrite) -> QuestionRunWrite:
        self._validate_run_trigger(run)
        return self._record_idempotent(
            records.QUESTION_RUN,
            run,
            "different lineage root fields",
        )

    def record_program_invocation(
        self,
        bundle: ProgramInvocationBundleWrite,
    ) -> ProgramInvocationBundleWrite:
        self._require_row(
            records.QUESTION_RUN,
            {"run_id": bundle.invocation.run_id},
            label="program invocation run",
        )
        if bundle.invocation.program_id != bundle.program.program_id:
            raise LineageRecorderConflict(
                "invocation must reference its answer program"
            )
        if bundle.invocation.revision_id is not None:
            revision = self._require_row(
                records.PROGRAM_REVISION,
                {"revision_id": bundle.invocation.revision_id},
                label="program invocation revision",
            )
            if revision.values["revised_program_id"] != bundle.invocation.program_id:
                raise LineageRecorderConflict(
                    "program invocation revision must produce its answer program"
                )
        with self._store.transaction():
            self._record_idempotent(records.ANSWER_PROGRAM, bundle.program)
            self._record_idempotent(records.PROGRAM_INVOCATION, bundle.invocation)
        return bundle

    def record_program_revision(
        self,
        bundle: ProgramRevisionBundleWrite,
    ) -> ProgramRevisionBundleWrite:
        self._require_row(
            records.ANSWER_PROGRAM,
            {"program_id": bundle.revision.base_program_id},
            label="program revision base program",
        )
        if bundle.revision.revised_program_id != bundle.program.program_id:
            raise LineageRecorderConflict(
                "program revision must reference its revised answer program"
            )
        if bundle.revision.base_program_id == bundle.revision.revised_program_id:
            raise LineageRecorderConflict(
                "program revision must change program identity"
            )
        with self._store.transaction():
            self._record_idempotent(records.ANSWER_PROGRAM, bundle.program)
            self._record_idempotent(records.PROGRAM_REVISION, bundle.revision)
        return bundle

    def record_step(self, step: RunStepWrite) -> RunStepWrite:
        return self._record_idempotent(records.RUN_STEP, step)

    def record_step_with_source_context(
        self,
        step: RunStepWrite,
        catalog_endpoints: tuple[CatalogEndpointWrite, ...],
        source_reads: tuple[SourceReadWrite, ...],
        artifacts: tuple[RunArtifactWrite, ...],
    ) -> RunStepWrite:
        if catalog_endpoints or source_reads or artifacts:
            _require_source_read_step(
                step_key=step.step_key.value,
                kind=step.kind.value,
                label=f"source read step {step.step_id!r}",
            )
        for catalog_endpoint in catalog_endpoints:
            if catalog_endpoint.run_id != step.run_id:
                raise LineageRecorderConflict(
                    f"catalog endpoint {catalog_endpoint.catalog_endpoint_id!r} "
                    f"does not belong to run {step.run_id!r}"
                )
        for source_read in source_reads:
            _require_source_read_belongs_to_step(source_read, step)
        for artifact in artifacts:
            if artifact.run_id != step.run_id or artifact.step_id != step.step_id:
                raise LineageRecorderConflict(
                    f"artifact {artifact.artifact_id!r} does not belong to "
                    f"step {step.step_id!r}"
                )
        _validate_source_response_artifacts(source_reads, artifacts)
        with self._store.transaction():
            self._record_idempotent(records.RUN_STEP, step)
            for catalog_endpoint in catalog_endpoints:
                self._record_idempotent(records.CATALOG_ENDPOINT, catalog_endpoint)
            for artifact in artifacts:
                self._record_idempotent(records.RUN_ARTIFACT, artifact)
            for source_read in source_reads:
                self._record_idempotent(records.SOURCE_READ, source_read)
        return step

    def record_model_call(self, model_call: ModelCallWrite) -> ModelCallWrite:
        _require_model_turn_step(
            self._require_same_run_row(
                records.RUN_STEP,
                run_id=model_call.run_id,
                identity_field="step_id",
                identity_value=model_call.step_id,
                label="model call step",
            )
        )
        return self._record_idempotent(records.MODEL_CALL, model_call)

    def record_model_call_audit(
        self, audit: ModelCallAuditWrite
    ) -> ModelCallAuditWrite:
        _require_model_turn_step(
            self._require_same_run_row(
                records.RUN_STEP,
                run_id=audit.model_call.run_id,
                identity_field="step_id",
                identity_value=audit.model_call.step_id,
                label="model call step",
            )
        )
        with self._store.transaction():
            self._record_idempotent(records.MODEL_CALL, audit.model_call)
            for usage in audit.usage_rows:
                self._record_idempotent(records.MODEL_CALL_USAGE, usage)
            for artifact in audit.artifacts:
                self._record_idempotent(records.RUN_ARTIFACT, artifact)
        return audit

    def record_model_call_usage(
        self, usage: ModelCallUsageWrite
    ) -> ModelCallUsageWrite:
        return self._record_idempotent(records.MODEL_CALL_USAGE, usage)

    def record_catalog_endpoint(
        self, catalog_endpoint: CatalogEndpointWrite
    ) -> CatalogEndpointWrite:
        return self._record_idempotent(records.CATALOG_ENDPOINT, catalog_endpoint)

    def record_source_read(self, source_read: SourceReadWrite) -> SourceReadWrite:
        step = self._require_same_run_row(
            records.RUN_STEP,
            run_id=source_read.run_id,
            identity_field="step_id",
            identity_value=source_read.step_id,
            label="source read step",
        )
        _require_source_read_step(
            step_key=str(step.values["step_key"]),
            kind=str(step.values["kind"]),
            label=f"source read step {source_read.step_id!r}",
        )
        return self._record_idempotent(records.SOURCE_READ, source_read)

    def record_artifact(self, artifact: RunArtifactWrite) -> RunArtifactWrite:
        return self._record_idempotent(records.RUN_ARTIFACT, artifact)

    def record_run_result(self, result: RunResultWrite) -> RunResultWrite:
        return self._record_idempotent(records.RUN_RESULT, result)

    def record_runtime_error_result(
        self, runtime_error: RuntimeErrorResultWrite
    ) -> RuntimeErrorResultWrite:
        with self._store.transaction():
            self.record_run_result(runtime_error.result)
            self._record_idempotent(records.RUNTIME_ERROR, runtime_error.error)
        return runtime_error

    def record_answered_result(
        self,
        answered_result: AnsweredRunResultWrite,
    ) -> AnsweredRunResultWrite:
        self._require_row(
            records.PROGRAM_INVOCATION,
            {"run_id": answered_result.result.run_id},
            label="answered run program invocation",
        )
        with self._store.transaction():
            self.record_run_result(answered_result.result)
            for requested_fact in answered_result.requested_facts:
                self.record_requested_fact(requested_fact)
            for fact_result in answered_result.fact_results:
                self.record_fact_result(fact_result)
            for proof_graph in answered_result.proof_graphs:
                self.record_execution_proof_graph(proof_graph)
            self.record_answer(answered_result.answer)
            for output in answered_result.outputs:
                self.record_answer_output(output)
            for presentation in answered_result.presentations:
                self.record_answer_presentation(presentation)
            for memory_artifact in answered_result.memory_artifacts:
                self.record_memory_artifact(memory_artifact)
        return answered_result

    def record_factual_terminal_result(
        self,
        terminal_result: FactualTerminalRunResultWrite,
    ) -> FactualTerminalRunResultWrite:
        with self._store.transaction():
            self.record_run_result(terminal_result.result)
            for requested_fact in terminal_result.requested_facts:
                self.record_requested_fact(requested_fact)
            for fact_result in terminal_result.fact_results:
                self.record_fact_result(fact_result)
            for proof_graph in terminal_result.proof_graphs:
                self.record_execution_proof_graph(proof_graph)
            for memory_artifact in terminal_result.memory_artifacts:
                self.record_memory_artifact(memory_artifact)
        return terminal_result

    def record_requested_fact(
        self, requested_fact: RequestedFactWrite
    ) -> RequestedFactWrite:
        return self._record_idempotent(records.REQUESTED_FACT, requested_fact)

    def record_fact_result(self, fact_result: FactResultWrite) -> FactResultWrite:
        return self._record_idempotent(records.FACT_RESULT, fact_result)

    def record_memory_artifact(
        self, memory_artifact: MemoryArtifactWrite
    ) -> MemoryArtifactWrite:
        return self._record_idempotent(records.MEMORY_ARTIFACT, memory_artifact)

    def record_clarification_request(
        self, clarification: ClarificationRequestWrite
    ) -> ClarificationRequestWrite:
        return self._record_idempotent(records.CLARIFICATION_REQUEST, clarification)

    def record_clarification_response(
        self, response: ClarificationResponseWrite
    ) -> ClarificationResponseWrite:
        return self._record_idempotent(records.CLARIFICATION_RESPONSE, response)

    def record_answer(self, answer: AnswerWrite) -> AnswerWrite:
        self._require_row_value(
            records.RUN_RESULT,
            run_id=answer.run_id,
            identity_field="run_result_id",
            identity_value=answer.run_result_id,
            value_field="result_kind",
            expected=RunResultKind.ANSWERED.value,
            label="answer run result",
        )
        return self._record_idempotent(records.ANSWER, answer)

    def record_answer_output(
        self, answer_output: AnswerOutputWrite
    ) -> AnswerOutputWrite:
        self._require_row_value(
            records.FACT_RESULT,
            run_id=answer_output.run_id,
            identity_field="fact_result_id",
            identity_value=answer_output.fact_result_id,
            value_field="result_kind",
            expected=FactResultKind.ANSWERED.value,
            label="answer output fact result",
        )
        self._validate_answer_output_proof_refs(answer_output)
        return self._record_idempotent(records.ANSWER_OUTPUT, answer_output)

    def record_answer_presentation(
        self, presentation: AnswerPresentationWrite
    ) -> AnswerPresentationWrite:
        render_step = self._require_same_run_row(
            records.RUN_STEP,
            run_id=presentation.run_id,
            identity_field="step_id",
            identity_value=presentation.render_step_id,
            label="answer presentation render step",
        )
        if render_step.values["step_key"] not in {
            RunStepKey.RENDER.value,
            RunStepKey.ANSWER_SYNTHESIS.value,
        }:
            raise LineageRecorderConflict(
                f"answer presentation render step {presentation.render_step_id!r} must be a render or answer_synthesis step"
            )
        return self._record_idempotent(records.ANSWER_PRESENTATION, presentation)

    def record_execution_proof_graph(
        self, proof_graph: ExecutionProofGraphWrite
    ) -> ExecutionProofGraphWrite:
        self._require_proof_graph_fact_result(proof_graph)
        self._require_step_key(
            run_id=proof_graph.run_id,
            step_id=proof_graph.compile_step_id,
            expected=RunStepKey.COMPILE,
            label="proof graph compile step",
        )
        if proof_graph.execute_step_id is not None:
            self._require_step_key(
                run_id=proof_graph.run_id,
                step_id=proof_graph.execute_step_id,
                expected=RunStepKey.EXECUTE,
                label="proof graph execute step",
            )
        payload = _validate_proof_graph_payload(proof_graph)
        self._validate_proof_graph_source_reads(proof_graph.run_id, payload)
        return self._record_idempotent(records.EXECUTION_PROOF_GRAPH, proof_graph)

    def _validate_proof_graph_source_reads(
        self,
        run_id: str,
        payload: ProofGraphPayload,
    ) -> None:
        for node in payload.nodes:
            for source_read_id in _source_read_ids(node.proof_refs):
                self._require_same_run_row(
                    records.SOURCE_READ,
                    run_id=run_id,
                    identity_field="source_read_id",
                    identity_value=source_read_id,
                    label="proof graph source read",
                )

    def _require_proof_graph_fact_result(
        self, proof_graph: ExecutionProofGraphWrite
    ) -> None:
        self._require_same_run_row(
            records.FACT_RESULT,
            run_id=proof_graph.run_id,
            identity_field="fact_result_id",
            identity_value=proof_graph.fact_result_id,
            label="proof graph fact result",
        )

    def _record_idempotent(
        self,
        spec: records.LineageRowSpec[T],
        write: T,
        conflict_detail: str = "different lineage fields",
    ) -> T:
        row = spec.to_row(write)
        self._validate_row_refs(row)
        existing = self._store.get_or_insert_row(row)
        if existing.values != row.values:
            identifier = next(iter(row.identity.values()))
            raise LineageRecorderConflict(
                f"{spec.label} {identifier!r} already exists with {conflict_detail}"
            )
        return write

    def _validate_row_refs(self, row: LineageRow) -> None:
        for reference in row.same_run_refs:
            fields = tuple(
                dict.fromkeys(
                    (
                        *reference.lookup,
                        *(
                            expectation.target_field
                            for expectation in reference.field_expectations
                        ),
                    )
                )
            )
            referenced = self._store.find_row(
                key=reference.target_key,
                lookup=reference.lookup,
                fields=fields,
            )
            if referenced is None:
                run_id = str(reference.lookup.get("run_id") or "")
                raise LineageRecorderConflict(
                    f"{reference.label} must belong to run {run_id!r}"
                )
            for expectation in reference.field_expectations:
                if referenced.values[expectation.target_field] != (
                    expectation.expected_value
                ):
                    raise LineageRecorderConflict(f"{expectation.label} must match")

    def _validate_run_trigger(self, run: QuestionRunWrite) -> None:
        if run.base_run_id:
            base = self._require_row(
                records.QUESTION_RUN,
                {"run_id": run.base_run_id},
                label="base run",
            )
            if base.values["question_id"] != run.question_id:
                raise LineageRecorderConflict(
                    f"base run {run.base_run_id!r} must belong to question {run.question_id!r}"
                )

    def _validate_answer_output_proof_refs(
        self,
        answer_output: AnswerOutputWrite,
    ) -> None:
        if not answer_output.proof_node_refs_json:
            raise LineageRecorderConflict("answer output requires proof node refs")
        proof_graph = self._require_same_run_row(
            records.EXECUTION_PROOF_GRAPH,
            run_id=answer_output.run_id,
            identity_field="fact_result_id",
            identity_value=answer_output.fact_result_id,
            label="answer output proof graph",
        )
        proof_payload = _read_proof_graph_payload_from_row(proof_graph)
        proof_node_by_id = {node.id: node for node in proof_payload.nodes}
        proof_node_ids = set(proof_node_by_id)
        missing = sorted(set(answer_output.proof_node_refs_json) - proof_node_ids)
        if missing:
            raise LineageRecorderConflict(
                f"answer output proof refs are missing from proof graph: {', '.join(missing)}"
            )
        for proof_ref in answer_output.proof_node_refs_json:
            node = proof_node_by_id[proof_ref]
            if node.kind is not ProofNodeKind.ANSWER_OUTPUT:
                raise LineageRecorderConflict(
                    f"answer output proof ref {proof_ref!r} must reference an answer_output proof node"
                )
            if not _has_upstream_proof_ref(proof_payload, proof_ref):
                raise LineageRecorderConflict(
                    f"answer output proof ref {proof_ref!r} is not connected to proof-bearing evidence"
                )

    def _require_step_key(
        self,
        *,
        run_id: str,
        step_id: str,
        expected: RunStepKey,
        label: str,
    ) -> None:
        row = self._require_same_run_row(
            records.RUN_STEP,
            run_id=run_id,
            identity_field="step_id",
            identity_value=step_id,
            label=label,
        )
        if row.values["step_key"] != expected.value:
            raise LineageRecorderConflict(
                f"{label} {step_id!r} must be a {expected.value} step"
            )

    def _require_row_value(
        self,
        spec: records.LineageRowSpec[T],
        *,
        run_id: str,
        identity_field: str,
        identity_value: str,
        value_field: str,
        expected: str,
        label: str,
    ) -> None:
        row = self._require_same_run_row(
            spec,
            run_id=run_id,
            identity_field=identity_field,
            identity_value=identity_value,
            label=label,
        )
        if row.values[value_field] != expected:
            raise LineageRecorderConflict(f"{label} must be {expected}")

    def _require_same_run_row(
        self,
        spec: records.LineageRowSpec[T],
        *,
        run_id: str,
        identity_field: str,
        identity_value: str | None,
        label: str,
    ) -> LineageRow:
        return self._require_row(
            spec,
            {identity_field: identity_value, "run_id": run_id},
            label=label,
        )

    def _require_row(
        self,
        spec: records.LineageRowSpec[T],
        lookup: dict[str, object],
        *,
        label: str,
    ) -> LineageRow:
        if any(value in (None, "") for value in lookup.values()):
            raise LineageRecorderConflict(f"{label} reference is missing")
        row = self._store.find_row(
            key=spec.key,
            lookup=lookup,
            fields=spec.storage_fields,
        )
        if row is None:
            raise LineageRecorderConflict(f"{label} does not exist")
        return row


def _require_model_turn_step(step: LineageRow) -> None:
    if step.values["kind"] != RunStepKind.MODEL_TURN.value:
        raise LineageRecorderConflict(
            f"model call step {step.identity['step_id']!r} is not a model_turn step"
        )


def _require_source_read_step(*, step_key: str, kind: str, label: str) -> None:
    if kind == RunStepKind.MODEL_TURN.value and step_key == RunStepKey.GROUNDING.value:
        return
    if kind == RunStepKind.DETERMINISTIC.value and step_key == RunStepKey.EXECUTE.value:
        return
    raise LineageRecorderConflict(
        f"{label} cannot own source reads; expected grounding or execute"
    )


def _require_source_read_belongs_to_step(
    source_read: SourceReadWrite,
    step: RunStepWrite,
) -> None:
    if source_read.run_id != step.run_id:
        raise LineageRecorderConflict(
            f"source read {source_read.source_read_id!r} does not belong to run {step.run_id!r}"
        )
    if source_read.step_id != step.step_id:
        raise LineageRecorderConflict(
            f"source read {source_read.source_read_id!r} does not belong to step {step.step_id!r}"
        )


def _validate_source_response_artifacts(
    source_reads: tuple[SourceReadWrite, ...],
    artifacts: tuple[RunArtifactWrite, ...],
) -> None:
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    if len(artifacts_by_id) != len(artifacts):
        raise LineageRecorderConflict("source context has duplicate artifact ids")
    referenced_artifact_ids: set[str] = set()
    for source_read in source_reads:
        if source_read.status is not SourceReadStatus.SUCCEEDED:
            if source_read.artifact_id is not None:
                raise LineageRecorderConflict(
                    f"failed source read {source_read.source_read_id!r} cannot reference an artifact"
                )
            continue
        if source_read.artifact_id is None:
            raise LineageRecorderConflict(
                f"successful source read {source_read.source_read_id!r} requires a source response artifact"
            )
        artifact = artifacts_by_id.get(source_read.artifact_id)
        if artifact is None:
            raise LineageRecorderConflict(
                f"source read {source_read.source_read_id!r} references an unavailable artifact"
            )
        if source_read.artifact_id in referenced_artifact_ids:
            raise LineageRecorderConflict(
                f"source response artifact {source_read.artifact_id!r} must belong to one source read"
            )
        if artifact.artifact_kind is not ArtifactKind.SOURCE_RESPONSE:
            raise LineageRecorderConflict(
                f"source read {source_read.source_read_id!r} requires a source_response artifact"
            )
        if artifact.content_hash != source_read.response_hash:
            raise LineageRecorderConflict(
                f"source read {source_read.source_read_id!r} response hash does not match its artifact"
            )
        referenced_artifact_ids.add(source_read.artifact_id)
    unreferenced = set(artifacts_by_id) - referenced_artifact_ids
    if unreferenced:
        raise LineageRecorderConflict(
            "source context has unreferenced artifacts: "
            + ", ".join(sorted(unreferenced))
        )


def _validate_proof_graph_payload(
    proof_graph: ExecutionProofGraphWrite,
) -> ProofGraphPayload:
    payload = read_execution_proof_graph_payload(
        payload_schema=proof_graph.payload_schema,
        payload_schema_rev=proof_graph.payload_schema_rev,
        payload_json=proof_graph.payload_json,
    )
    node_ids = [node.id for node in payload.nodes]
    duplicate_node_ids = sorted(
        node_id for node_id in set(node_ids) if node_ids.count(node_id) > 1
    )
    if duplicate_node_ids:
        raise LineageRecorderConflict(
            f"proof graph has duplicate node ids: {', '.join(duplicate_node_ids)}"
        )
    known_node_ids = set(node_ids)
    missing_edge_endpoints = sorted(
        {
            endpoint
            for edge in payload.edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in known_node_ids
        }
    )
    if missing_edge_endpoints:
        raise LineageRecorderConflict(
            "proof graph edges reference missing nodes: "
            + ", ".join(missing_edge_endpoints)
        )
    return payload


def _read_proof_graph_payload_from_row(row: LineageRow) -> ProofGraphPayload:
    payload_schema_rev = row.values["payload_schema_rev"]
    if payload_schema_rev is None:
        raise LineageRecorderConflict("proof graph payload revision is missing")
    try:
        return read_execution_proof_graph_payload(
            payload_schema=str(row.values["payload_schema"]),
            payload_schema_rev=int(payload_schema_rev),
            payload_json=row.values["payload_json"],
        )
    except ValueError as exc:
        raise LineageRecorderConflict(str(exc)) from exc


def _has_upstream_proof_ref(payload: ProofGraphPayload, node_id: str) -> bool:
    node_by_id = {node.id: node for node in payload.nodes}
    incoming_by_target: dict[str, list[str]] = {}
    for edge in payload.edges:
        incoming_by_target.setdefault(edge.target, []).append(edge.source)
    pending = [node_id]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        node = node_by_id.get(current)
        if _node_has_proof_refs(node):
            return True
        pending.extend(incoming_by_target.get(current, ()))
    return False


def _node_has_proof_refs(node: ProofGraphPayloadNode | None) -> bool:
    return bool(node is not None and node.proof_refs)


def _source_read_ids(proof_refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        proof_ref.removeprefix("source_read:")
        for proof_ref in proof_refs
        if proof_ref.startswith("source_read:")
        and proof_ref.removeprefix("source_read:")
    )
