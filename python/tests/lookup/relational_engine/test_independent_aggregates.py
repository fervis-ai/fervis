"""Scalar aggregate inputs retain their independent populations."""

from dataclasses import replace
from decimal import Decimal

import pytest

from tests.lookup.fact_compilation.test_compiler import _compile_memory_count
from fervis.lookup.question_contract.model import (
    Aggregate,
    Arithmetic,
    SetTerm,
    RequestedOutput,
    FactTerm,
)
from fervis.lookup.question_contract.model import AggregateFunction
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.expression_operators import ExpressionBinaryOperator
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    CompletenessProof,
    CompletenessStatus,
    CompletenessSourceKind,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.source_binding.model import (
    SetRealization,
    FactRealization,
    FactRealizationKind,
)
from fervis.lookup.semantic_types import DecimalType, UnitlessMeasure
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.answer_program.invocation import invoke_answer_program
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.invocation import RuntimePorts
from fervis.lookup.memory.projection import LookupMemory


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("left_count", [0, 2])
@pytest.mark.parametrize("right_count", [0, 1, 3])
@pytest.mark.parametrize("function", [AggregateFunction.COUNT, AggregateFunction.SUM])
def test_difference_of_independent_aggregates_uses_each_complete_population(
    left_count, right_count, function, standalone
):
    _, left_source, left_rows, _, _, original = _compile_memory_count(
        tuple(
            {"event_id": f"left_{i}", "amount": Decimal(i + 1)}
            for i in range(left_count)
        )
    )
    left_rows = replace(
        left_rows, field_types={"event_id": "string", "amount": "decimal"}
    )
    fact = original.request.index.requested_fact
    origin = fact.origin
    fact = replace(
        fact,
        sets=(*fact.sets, SetTerm("s2", origin)),
        expressions=(
            *fact.expressions,
            Aggregate("e2", AggregateFunction.COUNT, "s2", None, False, origin),
            Arithmetic("e3", ExpressionBinaryOperator.SUBTRACT, ("e1", "e2"), origin),
        ),
        outputs=(RequestedOutput("output_1", "e3", origin),),
    )
    if function is AggregateFunction.SUM:
        fact = replace(
            fact,
            facts=(
                FactTerm("left_amount", "s1", DecimalType(UnitlessMeasure()), origin),
                FactTerm("right_amount", "s2", DecimalType(UnitlessMeasure()), origin),
            ),
            expressions=(
                replace(
                    fact.expressions[0], function=function, argument_ref="left_amount"
                ),
                replace(
                    fact.expressions[1], function=function, argument_ref="right_amount"
                ),
                fact.expressions[2],
            ),
        )
    if standalone:
        fact = replace(fact, outputs=(RequestedOutput("output_1", "e2", origin),))
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    right_rows = RelationRows(
        "other_events",
        tuple(
            {"event_id": f"right_{i}", "amount": Decimal(i + 3)}
            for i in range(right_count)
        ),
        field_types={"event_id": "string", "amount": "decimal"},
        completeness=CompletenessProof(
            status=CompletenessStatus.COMPLETE,
            source_kind=CompletenessSourceKind.MEMORY_READ,
            proof_refs=("memory:other_events",),
        ),
    )
    row_sources = build_row_source_catalog(
        RelationCatalog(), memory_relations=(left_rows, right_rows)
    )
    left_source = next(
        source for source in row_sources.sources if source.memory_ref == left_rows.id
    )
    right_source = next(
        source for source in row_sources.sources if source.memory_ref == right_rows.id
    )
    branch = replace(
        original.request.strategy.branches[0],
        source_refs=(left_source.id, right_source.id),
    )
    strategy = replace(original.request.strategy, branches=(branch,))
    request = replace(
        original.request,
        index=index,
        strategy=strategy,
        source_catalog=replace(
            original.request.source_catalog, sources=row_sources.sources
        ),
    )
    plan = replace(
        original.binding_plan,
        strategy=strategy,
        set_bindings={
            **original.binding_plan.set_bindings,
            index.fact_local_ref_by_local_id["s2"].token: (
                SetRealization(
                    branch.branch_id,
                    "The independently counted population.",
                    right_source.id,
                    None,
                    (),
                    (right_source.id,),
                ),
            ),
        },
    )
    if function is AggregateFunction.SUM:
        fact_bindings = {}
        for ref, source in (
            ("left_amount", left_source),
            ("right_amount", right_source),
        ):
            field = next(field for field in source.fields if field.id == "amount")
            fact_bindings[index.fact_local_ref_by_local_id[ref].token] = (
                FactRealization(
                    branch.branch_id,
                    "The declared amount on this population.",
                    source.id,
                    FactRealizationKind.RETURNED_FIELD,
                    None,
                    (field.field_ref,),
                    (source.id, field.field_ref),
                ),
            )
        plan = replace(plan, fact_bindings=fact_bindings)
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    compilation = compile_verified_source_strategy(verified)
    memory = (left_rows, right_rows)
    execution = invoke_answer_program(
        program=compilation.answer_program,
        bindings=compilation.initial_bindings,
        environment=ExecutionEnvironment(
            catalog=RelationCatalog(), memory_relations=memory
        ),
        ports=RuntimePorts(
            data_access_port=None, memory=LookupMemory(relations=memory)
        ),
    )
    assert execution.issue is None
    expected = (
        left_count - right_count
        if function is AggregateFunction.COUNT
        else sum(range(1, left_count + 1)) - sum(range(3, right_count + 3))
    )
    if standalone:
        expected = (
            right_count
            if function is AggregateFunction.COUNT
            else sum(range(3, right_count + 3))
        )
    actual = (
        [
            value
            for row in execution.fact_result.outcome.projected_rows
            for value in row.values.values()
        ]
        if standalone
        else list(execution.fact_result.outcome.scalars.values())
    )
    assert actual == [Decimal(expected)]
    # Publishing the answer must be able to follow every aggregate operand
    # back to its actual source relation, including empty source populations.
    incoming = {}
    for edge in execution.proof_graph.edges:
        incoming.setdefault(edge.target, []).append(edge.source)
    reachable = set()
    pending = [
        node.id
        for node in execution.proof_graph.nodes
        if node.kind.value == "answer_output"
    ]
    while pending:
        node = pending.pop()
        if node not in reachable:
            reachable.add(node)
            pending.extend(incoming.get(node, ()))
    assert {
        f"relation:{relation.id}" for relation in compilation.answer_program.relations
    } <= reachable


def test_independent_scalar_domains_do_not_authorize_an_unrelated_row_projection():
    from fervis.lookup.question_contract.model import (
        RequestedFact,
        Subject,
        InstanceInterpretation,
        AllResults,
    )
    from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind

    origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT, "A row value on an unrelated population"
    )
    fact = RequestedFact(
        "fact",
        origin,
        (SetTerm("left", origin), SetTerm("right", origin)),
        (),
        (FactTerm("right_value", "right", DecimalType(UnitlessMeasure()), origin),),
        (),
        Subject("left", InstanceInterpretation.RESOURCE_POPULATION),
        None,
        (),
        (RequestedOutput("value", "right_value", origin),),
        (),
        AllResults(),
        (),
    )
    with pytest.raises(ValueError, match="row domain"):
        analyze_requested_fact(fact, inputs={}, input_denotations={})
