import pytest

from fervis.lookup.query_enrichment.semantic import RecallBucketMatch
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
)
from fervis.lookup.relation_catalog import (
    CatalogField,
    CatalogParam,
    EndpointRead,
    ParamSource,
    RelationCatalog,
)
from fervis.lookup.relation_catalog.selection.selector import (
    SemanticCatalogSelectionRequest,
    select_semantic_relation_catalog,
)
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
from fervis.lookup.semantic_types import TextType


def test_semantic_catalog_selection_uses_requirement_resource_matches() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "service events")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("event_set", origin),),
        associations=(),
        facts=(),
        expressions=(
            Aggregate(
                id="event_count",
                function=AggregateFunction.COUNT,
                argument_ref="event_set",
                filter_ref=None,
                distinct_argument=False,
                origin=origin,
            ),
        ),
        subject=Subject("event_set", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "event_count", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="list_service_events",
                endpoint_name="list_service_events",
                resource_names=("service event",),
            ),
            EndpointRead(
                id="list_inventory_items",
                endpoint_name="list_inventory_items",
                resource_names=("inventory item",),
            ),
        )
    )

    result = select_semantic_relation_catalog(
        SemanticCatalogSelectionRequest(
            relation_catalog=catalog,
            indexes=(index,),
            resource_matches=(
                _match("fact_1:recall:population", "service event"),
                _match("fact_1:recall:output_1", "service event"),
            ),
            max_reads_per_fact=10,
        )
    )

    assert result.selected_read_ids == ("list_service_events",)
    assert tuple(read.id for read in result.relation_catalog.reads) == (
        "list_service_events",
    )


def test_semantic_catalog_selection_preserves_a_directly_callable_exact_read() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "service events")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("event_set", origin),),
        associations=(),
        facts=(),
        expressions=(
            Aggregate(
                id="event_count",
                function=AggregateFunction.COUNT,
                argument_ref="event_set",
                filter_ref=None,
                distinct_argument=False,
                origin=origin,
            ),
        ),
        subject=Subject("event_set", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "event_count", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    detail_reads = tuple(
        EndpointRead(
            id=f"get_service_event_{index}",
            endpoint_name=f"get_service_event_{index}",
            resource_names=("service event",),
            params=(
                CatalogParam(
                    ref=f"get_service_event_{index}.path.event_id",
                    name="event_id",
                    source=ParamSource.PATH,
                    type="string",
                    required=True,
                ),
            ),
            fields=tuple(
                CatalogField(
                    ref=f"field.service_event_{index}_{field_index}",
                    path=f"service_event_{index}_{field_index}",
                    type="string",
                )
                for field_index in range(3)
            ),
        )
        for index in range(4)
    )
    catalog = RelationCatalog(
        reads=(
            *detail_reads,
            EndpointRead(
                id="list_service_events",
                endpoint_name="list_service_events",
                resource_names=("service event",),
            ),
        )
    )

    result = select_semantic_relation_catalog(
        SemanticCatalogSelectionRequest(
            relation_catalog=catalog,
            indexes=(index,),
            resource_matches=(
                _match("fact_1:recall:population", "service event"),
                _match("fact_1:recall:output_1", "service event"),
            ),
            max_reads_per_fact=1,
        )
    )

    assert "list_service_events" in result.selected_read_ids


def test_semantic_catalog_selection_round_robins_requirements_not_resource_names() -> (
    None
):
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "measure by group")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("measurements", origin),),
        associations=(),
        facts=(FactTerm("group_name", "measurements", TextType(), origin),),
        expressions=(
            Aggregate(
                id="measurement_count",
                function=AggregateFunction.COUNT,
                argument_ref="measurements",
                filter_ref=None,
                distinct_argument=False,
                origin=origin,
            ),
        ),
        subject=Subject(
            "measurements", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE
        ),
        qualification_ref=None,
        grouping_refs=("group_name",),
        outputs=(
            RequestedOutput("group", "group_name", origin),
            RequestedOutput("count", "measurement_count", origin),
        ),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    catalog = RelationCatalog(
        reads=(
            *(
                EndpointRead(
                    id=f"measure_read_{suffix}",
                    endpoint_name=f"list_measure_{suffix}",
                    resource_names=(f"measure {suffix}",),
                )
                for suffix in ("a", "b", "c")
            ),
            EndpointRead(
                id="group_read",
                endpoint_name="list_groups",
                resource_names=("group entity",),
            ),
        )
    )

    result = select_semantic_relation_catalog(
        SemanticCatalogSelectionRequest(
            relation_catalog=catalog,
            indexes=(index,),
            resource_matches=(
                _match(
                    "fact_1:recall:population",
                    "measure a",
                    "measure b",
                    "measure c",
                ),
                _match("fact_1:recall:group", "group entity"),
                _match(
                    "fact_1:recall:count",
                    "measure a",
                    "measure b",
                    "measure c",
                ),
            ),
            max_reads_per_fact=1,
        )
    )

    assert len(result.selected_read_ids) == 3
    assert "group_read" in result.selected_read_ids
    assert (
        len(
            set(result.selected_read_ids)
            & {"measure_read_a", "measure_read_b", "measure_read_c"}
        )
        == 2
    )


def test_semantic_catalog_selection_requires_exact_requirement_coverage() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "service events")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("event_set", origin),),
        associations=(),
        facts=(),
        expressions=(
            Aggregate(
                id="event_count",
                function=AggregateFunction.COUNT,
                argument_ref="event_set",
                filter_ref=None,
                distinct_argument=False,
                origin=origin,
            ),
        ),
        subject=Subject("event_set", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "event_count", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})

    with pytest.raises(
        ValueError,
        match="semantic resource matches must cover every recall bucket once",
    ):
        select_semantic_relation_catalog(
            SemanticCatalogSelectionRequest(
                relation_catalog=RelationCatalog(),
                indexes=(index,),
                resource_matches=(),
                max_reads_per_fact=10,
            )
        )


def _match(bucket_ref: str, *resource_names: str) -> RecallBucketMatch:
    return RecallBucketMatch(
        bucket_ref=bucket_ref,
        exhaustive_resource_names=resource_names,
        matching_resource_names=resource_names,
    )
