"""A read-level predicate must preserve all consumers of that read."""

from dataclasses import replace

import pytest

from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    Arithmetic,
    RequestedOutput,
)
from fervis.lookup.answer_program.expressions import ExpressionBinaryOperator
from fervis.lookup.source_binding.verification import (
    SourceStrategyVerificationFailure,
    verify_source_strategy,
)
from tests.lookup.fact_compilation import test_compiler as fixtures


@pytest.mark.parametrize("sibling_filter", [None, "other_period"])
def test_aggregate_local_invocation_cannot_narrow_sibling_population(
    monkeypatch, sibling_filter
):
    captured = {}
    original = fixtures.verify_source_strategy

    def capture(plan, *, request):
        captured.update(plan=plan, request=request)
        return original(plan, request=request)

    monkeypatch.setattr(fixtures, "verify_source_strategy", capture)
    fixtures.test_invocation_realized_aggregate_filter_is_not_reapplied_to_rows()
    request = captured["request"]
    fact = request.index.requested_fact
    expressions = fact.expressions
    inputs = dict(request.index.input_by_ref)
    denotations = dict(request.index.input_denotation_by_ref)
    if sibling_filter:
        comparison = next(node for node in expressions if node.id == "e1")
        inputs["i2"] = replace(inputs["i1"], id="i2")
        denotations["i2"] = denotations["i1"]
        expressions += (replace(comparison, id=sibling_filter, right_ref="i2"),)
    fact = replace(
        fact,
        expressions=(
            *expressions,
            Aggregate(
                "e_all",
                AggregateFunction.COUNT,
                "s1",
                sibling_filter,
                False,
                fact.origin,
            ),
            Arithmetic(
                "e_ratio", ExpressionBinaryOperator.DIVIDE, ("e2", "e_all"), fact.origin
            ),
        ),
        outputs=(RequestedOutput("output_1", "e_ratio", fact.origin),),
    )
    index = analyze_requested_fact(fact, inputs=inputs, input_denotations=denotations)
    request = replace(request, index=index)
    outcome = verify_source_strategy(captured["plan"], request=request)
    assert isinstance(outcome, SourceStrategyVerificationFailure)
    aggregate_requirement = next(
        item
        for item in index.boolean_requirements
        if item.owner_expression_ref.endswith(":e2")
    )
    assert (
        request.invocation_options_for_owner(
            aggregate_requirement.requirement_ref,
            branch_id=request.strategy.branches[0].branch_id,
        )
        == ()
    )
