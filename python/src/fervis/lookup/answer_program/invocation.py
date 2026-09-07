"""The single deterministic invocation kernel for answer programs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from fervis.lookup.answer_program.instantiation import (
    ExecutionEnvironment,
    ExecutionProofGraph,
    VerifiedExecution,
    instantiate_answer_program,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    RowContextStore,
)
from fervis.lookup.lineage.source_reads import (
    SourceReadLineageScope,
)
from fervis.lookup.outcomes.model import FactResult
from fervis.lookup.outcomes.classification import classify_answer_result
from fervis.lookup.outcomes.errors import ExecutionIssue
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.answer_program.source_materialization import execute_relation_inputs
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.contract_codec import answer_program_id
from fervis.lookup.answer_program.persistence import ProgramInvocationBinding
from fervis.lineage.enums import ProgramInvocationKind
from fervis.lookup.question_contract import RequestedFact

if TYPE_CHECKING:
    from fervis.lookup.memory.projection import LookupMemory


@dataclass(frozen=True)
class AnswerExecution:
    fact_result: FactResult | None
    issue: ExecutionIssue | None = None
    program: AnswerProgram | None = None
    program_id: str = ""
    invocation_id: str = ""
    proof_node_refs_by_result_output_id: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    relations: tuple[RelationRows, ...] = ()
    proof_refs: tuple[str, ...] = ()
    proof_graph: ExecutionProofGraph = field(default_factory=ExecutionProofGraph)
    effective_requested_facts: tuple[RequestedFact, ...] = ()
    row_context: RowContextStore = field(default_factory=RowContextStore)




@dataclass(frozen=True)
class RuntimePorts:
    data_access_port: Any
    memory: LookupMemory
    source_read_lineage: SourceReadLineageScope | None = None
    invocation_binding: ProgramInvocationBinding | None = None
    invocation_kind: ProgramInvocationKind = ProgramInvocationKind.COMPILED_QUESTION
    base_invocation_id: str | None = None


def invoke_answer_program(
    *,
    program: AnswerProgram,
    bindings: BindingSet,
    environment: ExecutionEnvironment,
    ports: RuntimePorts,
) -> AnswerExecution:
    execution = instantiate_answer_program(program, bindings, environment)
    invocation_id = ""
    if ports.invocation_binding is not None:
        invocation_id = ports.invocation_binding.bind(
            execution,
            kind=ports.invocation_kind,
            base_invocation_id=ports.base_invocation_id,
        ).invocation_id
    return replace(
        execute_verified_program(execution, ports),
        invocation_id=invocation_id,
    )


def execute_verified_program(
    execution: VerifiedExecution,
    ports: RuntimePorts,
) -> AnswerExecution:
    answer = execution.answer
    materialized = execute_relation_inputs(answer,execution,catalog=execution.catalog,row_sources=execution.row_sources,
        memory=ports.memory,read_session=ApiReadSession(ports.data_access_port,ports.source_read_lineage),
        authority_ref=execution.authority_ref)
    engine_output, row_context = materialized.engine_output, materialized.row_context
    classified = classify_answer_result(
        answer,
        engine_output=engine_output,
    )
    if isinstance(classified, ExecutionIssue):
        return AnswerExecution(
            fact_result=None,
            issue=classified,
            program=answer,
            program_id=answer_program_id(answer),
            proof_node_refs_by_result_output_id=(
                execution.proof_node_refs_by_result_output_id
            ),
            relations=engine_output.relations,
            proof_refs=_proof_refs(engine_output.relations),
            proof_graph=execution.proof_graph.with_executed_relations(
                engine_output.relations
            ),
            effective_requested_facts=execution.effective_requested_facts,
            row_context=row_context,
        )
    return AnswerExecution(
        fact_result=classified,
        program=answer,
        program_id=answer_program_id(answer),
        proof_node_refs_by_result_output_id=(
            execution.proof_node_refs_by_result_output_id
        ),
        relations=engine_output.relations,
        proof_refs=_proof_refs(engine_output.relations),
        proof_graph=execution.proof_graph.with_executed_relations(
            engine_output.relations
        ),
        effective_requested_facts=execution.effective_requested_facts,
        row_context=row_context,
    )



def _proof_refs(relations: tuple[RelationRows, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for relation in relations:
        for ref in relation.completeness.proof_refs:
            if ref not in refs:
                refs.append(ref)
    return tuple(refs)
