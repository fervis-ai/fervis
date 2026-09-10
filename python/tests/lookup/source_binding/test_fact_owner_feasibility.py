"""Fields cannot come from reads that cannot represent their owning rows."""

import pytest

from fervis.lookup.source_binding.model import (
    CandidateSourceStrategy,
    SourceStrategyBranch,
)
from fervis.lookup.source_binding.model import SemanticSourceBindingRequest
from tests.lookup.source_binding._candidate_fixture import daily_observation_request


def test_observed_date_cannot_come_from_a_calendar_that_cannot_own_revenue():
    upstream = daily_observation_request()
    strategy = CandidateSourceStrategy(
        "fact_1",
        (
            SourceStrategyBranch(
                "b",
                tuple(source.id for source in upstream.source_catalog.sources),
                (),
                (),
            ),
        ),
    )
    request = SemanticSourceBindingRequest(
        upstream.index, strategy, upstream.source_catalog, ()
    )
    assert request.row_references_for_set("fact_1:set:s1") == ("raw_observations",)
    assert request.returned_field_refs_for_fact("fact_1:fact:f1") == (
        "source_field:raw_observations:runtime_date",
    )
    assert request.returned_field_refs_for_fact("fact_1:fact:f2") == (
        "source_field:raw_observations:amount",
    )


@pytest.mark.parametrize("can_filter_date", (False, True))
def test_predicate_date_must_be_available_on_its_owning_rows(can_filter_date):
    from dataclasses import replace
    from fervis.lookup.question_contract.model import (
        Aggregate,
        AggregateFunction,
        NullCheck,
        RequestedOutput,
    )
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.expression_operators import ExpressionUnaryOperator

    upstream = daily_observation_request()
    fact = upstream.index.requested_fact
    fact = replace(
        fact,
        grouping_refs=(),
        expressions=(
            NullCheck("present", ExpressionUnaryOperator.NOT_NULL, "f1", fact.origin),
            Aggregate("count", AggregateFunction.COUNT, "s1", None, False, fact.origin),
        ),
        qualification_ref="present",
        outputs=(RequestedOutput("count", "count", fact.origin),),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    raw = upstream.source_catalog.sources[-1]
    missing_date = replace(
        raw,
        id="missing_date",
        params=raw.params if can_filter_date else (),
        fields=tuple(field for field in raw.fields if field.id == "amount"),
    )
    request = replace(
        upstream,
        index=index,
        source_catalog=replace(
            upstream.source_catalog,
            sources=(*upstream.source_catalog.sources, missing_date),
        ),
    )
    assert (
        "missing_date" in request.row_references_for_set("fact_1:set:s1")
    ) is can_filter_date


def test_associated_roles_reject_a_disconnected_type_compatible_source():
    from dataclasses import replace
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query

    request = employee_query().request
    source = request.source_catalog.sources[0]
    disconnected = replace(source, id='unrelated_rows', candidate_keys=(), entity_references=())
    request = replace(request, source_catalog=replace(
        request.source_catalog, sources=(source, disconnected)
    ))
    assert 'unrelated_rows' not in request.row_references_for_set('fact_1:set:manager')
    assert 'unrelated_rows' not in request.row_references_for_set('fact_1:set:employee')
