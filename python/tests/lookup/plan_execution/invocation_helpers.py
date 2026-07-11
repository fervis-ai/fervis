from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.compilation import compile_answer_program
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
from fervis.lookup.fact_plan.fact_plan import FactPlan
from fervis.lookup.lineage.source_reads import SourceReadLineageScope
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.plan_execution.authorized_sources import AuthorizedExecutionSources
from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult


def compile_and_invoke(
    *,
    plan: FactPlan,
    question_contract: QuestionContract,
    catalog: RelationCatalog,
    data_access_port: Any,
    memory: LookupMemory,
    catalog_selection: CatalogSelectionResult | None = None,
    authorized_sources: AuthorizedExecutionSources | None = None,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> Any:
    if not isinstance(plan.outcome, AnswerProgram):
        raise ValueError("test invocation requires answer program")
    execution_catalog = (
        authorized_sources.relation_catalog
        if authorized_sources is not None
        else catalog
    )
    program, bindings = compile_answer_program(
        plan.outcome,
        question_contract=question_contract,
        catalog=execution_catalog,
        bindings=plan.bindings,
        memory_relations=memory.relations,
    )
    return invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(
            catalog=execution_catalog,
            authorized_sources=authorized_sources,
            catalog_selection=catalog_selection,
            memory_relations=memory.relations,
        ),
        ports=RuntimePorts(
            data_access_port=data_access_port,
            memory=memory,
            source_read_lineage=source_read_lineage,
        ),
    )
