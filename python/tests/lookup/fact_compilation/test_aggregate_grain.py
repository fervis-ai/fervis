"""Aggregates retain the identity of the entities they observe after joins."""

from dataclasses import replace
from decimal import Decimal

import pytest

from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    RequestedOutput,
    AllResults,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
)
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessStatus,
    RelationRows,
)
from fervis.lookup.source_binding.verification import (
    VerifiedSourceStrategy,
    verify_source_strategy,
)
from tests.lookup.fact_compilation import test_compiler as fixtures


@pytest.mark.parametrize("include_empty_category", [False, True])
def test_count_of_subject_set_uses_its_identity_after_a_one_to_many_join(monkeypatch, include_empty_category):
    captured = {}
    original_verify = fixtures.verify_source_strategy

    def capture(plan, *, request):
        captured.update(plan=plan, request=request)
        return original_verify(plan, request=request)

    monkeypatch.setattr(fixtures, "verify_source_strategy", capture)
    fixtures.test_declared_association_compiles_to_existing_join_and_grouped_aggregate()
    request = captured["request"]
    fact = request.index.requested_fact
    fact = replace(
        fact,
        expressions=(
            Aggregate(
                "e_categories",
                AggregateFunction.COUNT,
                "s_category",
                None,
                False,
                fact.origin,
            ),
        ),
        subject=replace(fact.subject, set_ref="s_category"),
        grouping_refs=(),
        ordering=(),
        selection=AllResults(),
        outputs=(RequestedOutput("o_categories", "e_categories", fact.origin),),
    )
    request = replace(
        request, index=analyze_requested_fact(fact, inputs={}, input_denotations={})
    )
    plan = replace(
        captured["plan"],
        subject_binding=replace(
            captured["plan"].subject_binding, subject_ref="fact_1:set:s_category"
        ),
    )
    request = request.for_bindings(
        plan.set_bindings, plan.fact_bindings, plan.association_bindings
    )
    plan = replace(plan, strategy=request.strategy)
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    program = compile_verified_source_strategy(verified).answer_program
    data = {
        "source_events": (
            {"event_id": "e1", "category_id": "c1", "amount": Decimal(10)},
            {"event_id": "e2", "category_id": "c1", "amount": Decimal(20)},
        ),
        "source_categories": ({"category_id": "c1"},),
    }
    if include_empty_category:
        data["source_categories"] += ({"category_id": "c2"},)
    execution = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    relation.id,
                    data[relation.source.row_source_id],
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for relation in program.relations
            ),
            operations=tuple(
                ExecutableOperation(op.id, op.spec, op.output_relation)
                for op in program.operations
            ),
        )
    )
    assert execution.issue is None
    row = execution.relation(program.operations[-1].output_relation).rows[0]
    assert [
        row[output.field_id] for output in program.result_projection.relation_outputs
    ] == [2 if include_empty_category else 1]


def test_composite_observation_grain_preserves_distinct_entities_with_equal_amounts():
    from fervis.lookup.answer_program.operations import (
        AggregateSpec,
        AggregationSpec,
        AggregationFunction,
    )
    from fervis.lookup.answer_program.expressions import FieldRef

    rows = (
        {"tenant": "a", "id": "1", "amount": Decimal(10), "included": False},
        {"tenant": "a", "id": "1", "amount": Decimal(10), "included": True},
        {"tenant": "b", "id": "1", "amount": Decimal(10), "included": True},
        {"tenant": "a", "id": None, "amount": Decimal(99), "included": True},
    )
    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    "rows",
                    rows,
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                ),
            ),
            operations=(
                ExecutableOperation(
                    "aggregate",
                    AggregateSpec(
                        "rows",
                        (),
                        (
                            AggregationSpec(
                                AggregationFunction.COUNT,
                                "count",
                                grain_fields=("tenant", "id"),
                                filter=FieldRef("included"),
                            ),
                            AggregationSpec(
                                AggregationFunction.SUM,
                                "total",
                                "amount",
                                grain_fields=("tenant", "id"),
                            ),
                        ),
                    ),
                    "result",
                ),
            ),
        )
    )
    assert result.issue is None
    assert result.relation("result").rows == ({"count": 2, "total": Decimal(20)},)


def test_conflicting_values_for_one_observation_identity_are_rejected():
    import pytest
    from fervis.lookup.answer_program.operations import (
        AggregateSpec,
        AggregationSpec,
        AggregationFunction,
    )
    from fervis.lookup.plan_execution.errors import RelationEngineError

    with pytest.raises(RelationEngineError, match="conflicting values"):
        execute_operations(
            RelationEngineInput(
                relations=(
                    RelationRows(
                        "rows",
                        (
                            {"id": "1", "amount": Decimal(10)},
                            {"id": "1", "amount": Decimal(20)},
                        ),
                        completeness=CompletenessProof(
                            status=CompletenessStatus.COMPLETE
                        ),
                    ),
                ),
                operations=(
                    ExecutableOperation(
                        "aggregate",
                        AggregateSpec(
                            "rows",
                            (),
                            (
                                AggregationSpec(
                                    AggregationFunction.SUM,
                                    "total",
                                    "amount",
                                    grain_fields=("id",),
                                ),
                            ),
                        ),
                        "result",
                    ),
                ),
            )
        )
