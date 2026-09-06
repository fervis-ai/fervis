"""A daily observation request with complete and incomplete candidate reads."""

from dataclasses import replace

from fervis.lookup.available_sources import (
    AvailableSourceCatalog,
    SourceContractSnapshot,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    AllResults,
    FactTerm,
    InstanceInterpretation,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
    TemporalBucket,
    TemporalGrain,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import (
    RowSourceField,
    RowSourceValueType,
    build_row_source_catalog,
)
from fervis.lookup.semantic_types import (
    NumericType,
    SourceOrigin,
    SourceOriginKind,
    TemporalPointType,
)


from fervis.lookup.source_binding.model import SemanticSourceBindingRequest
from fervis.lookup.source_binding.candidates import candidate_source_strategy


def daily_observation_request():
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "daily observation totals")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(
            FactTerm("f1", "s1", TemporalPointType(), origin),
            FactTerm("f2", "s1", NumericType(), origin),
        ),
        expressions=(
            TemporalBucket("e1", "f1", TemporalGrain.DAY, origin),
            Aggregate("e2", AggregateFunction.SUM, "f2", None, False, origin),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=("e1",),
        outputs=(
            RequestedOutput("o1", "e1", origin),
            RequestedOutput("o2", "e2", origin),
        ),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    [calendar] = build_row_source_catalog(RelationCatalog()).sources
    raw = replace(
        calendar,
        id="raw_observations",
        fields=(
            *calendar.fields,
            RowSourceField(
                "amount", "amount", "amount", RowSourceValueType.DECIMAL, ()
            ),
        ),
    )
    catalog = AvailableSourceCatalog(
        SourceContractSnapshot.from_content("{}"), (calendar, raw), ()
    )
    return SemanticSourceBindingRequest(
        index, candidate_source_strategy(index, catalog, ()), catalog, ()
    )
