from ._helpers import *  # noqa: F403

from dataclasses import dataclass, field
from typing import Any

from tests.lookup.plan_execution.invocation_helpers import compile_and_invoke
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.fact_plan.fact_plan import FactPlan


@dataclass
class _RowsDataAccess:
    rows: tuple[dict[str, object], ...]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        rows = tuple(
            row
            for row in self.rows
            if all(row.get(str(key).split(".")[-1]) == value for key, value in args.items())
        )
        return {
            "endpointName": endpoint_name,
            "responseStatus": 200,
            "responseBody": {"data": list(rows)},
            "truncated": False,
            "pageCount": 1,
        }


def _count_metric_selection() -> dict[str, object]:
    return {
        "selection_basis": "The requested answer is a count of matching rows.",
        "id": "metric_1",
        "kind": "count_records",
    }


def _count_function_selection() -> dict[str, object]:
    return {
        "selection_basis": "Counting rows requires the count aggregate function.",
        "id": "function_count",
        "value": "count",
    }

def test_pattern_prompt_count_metric_uses_source_identity_not_predicate_fulfillment():
    request = FactPlanRequest(
        question="How many active records are there?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="active records count",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            description="count of active records",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ,
                    read_id="list_records",
                ),
                cardinality="many",
                available_field_ids=("record_key", "is_active"),
                available_fields=(
                    SourceField(
                        field_id="record_key",
                        type="uuid",
                        roles=("identity",),
                        identity=IdentityMetadata(
                            entity_ref="record",
                            identity_field="record_key",
                            primary_key=True,
                            stable=True,
                        ),
                    ),
                    SourceField(field_id="is_active", type="boolean"),
                ),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="source_1.record_key",
                        field_id="record_key",
                        row_cardinality="many",
                    ),
                    SourceEvidenceItem(
                        evidence_id="source_1.is_active",
                        field_id="is_active",
                        row_cardinality="many",
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation=(
                            "The active record count is determined by rows where "
                            "is_active is true."
                        ),
                        row_count_basis_evidence_ids=("source_1.record_key",),
                        group_key_evidence_ids=("source_1.is_active",),
                        scope_evidence_ids=("source_1.is_active",),
                    ),
                ),
            ),
        ),
    )

    prompt = _pattern_fact_plan_prompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="fact_1",
                    plan_selection_id="fact_1.aggregate_scalar.sb_1",
                    source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                    plan_shape="aggregate_scalar",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )

    assert _text_prompt_section(
        prompt,
        label="Scalar aggregate operation choices",
        next_label="Decision Scope",
    ) == "\n".join(
        (
            '<fact id="fact_1">',
            '  <source_binding id="sb_1" read="list_records">',
            '    <operation family="aggregate_scalar">',
            "      <metric_candidates>",
            '        <metric id="metric_1" kind="count_records" allowed_functions="count" />',
            "      </metric_candidates>",
            "      <function_candidates>",
            '        <function id="function_count" value="count" meaning="number of matching rows" />',
            "      </function_candidates>",
            "    </operation>",
            "  </source_binding>",
            "</fact>",
        )
    )


def test_count_metric_options_allow_concrete_row_population_count_without_identity_field():
    row_source_id = api_row_source_id("list_summary", "data")
    source = BoundSource(
        id="sb_1",
        requested_fact_id="fact_1",
        answer_population=_answer_population(),
        source=DraftRelationSource(
            kind=SourceKind.API_READ,
            read_id="list_summary",
            row_source_id=row_source_id,
        ),
        cardinality="many",
        available_field_ids=("amount", "count"),
        available_fields=(
            SourceField(field_id="amount", type="decimal"),
            SourceField(field_id="count", type="integer"),
        ),
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="row_population.data",
                field_id="data",
                type="row_population",
                row_cardinality="many",
                row_source_id=row_source_id,
            ),
        ),
        fulfillments=(
            SourceFulfillment(
                requested_fact_id="fact_1",
                answer_output_id="answer_1",
                match_basis_explanation="Rows are countable.",
                row_count_basis_evidence_ids=("row_population.data",),
            ),
        ),
    )

    assert scalar_aggregate_choices_for_source(
        source,
        requested_fact_id="fact_1",
        plan_shape="aggregate_scalar",
    ) == {
        "requested_fact_id": "fact_1",
        "source_binding_id": "sb_1",
        "read_id": "list_summary",
        "plan_shape": "aggregate_scalar",
        "metric_candidates": (
            {
                "id": "metric_1",
                "answer_output_id": "answer_1",
                "kind": "count_records",
                "count_basis": {
                    "kind": "row_population",
                    "row_source_id": row_source_id,
                    "row_path_id": "data",
                    "row_cardinality": "many",
                },
                "allowed_functions": ("count",),
            },
        ),
        "function_candidates": (
            {
                "id": "function_count",
                "value": "count",
                "meaning": "number of matching rows",
            },
        ),
    }


def test_count_metric_options_allow_nested_many_row_population_under_one_row_source():
    row_source_id = api_row_source_id("list_summary", "staffs")
    source = BoundSource(
        id="sb_1",
        requested_fact_id="fact_1",
        answer_population=_answer_population(),
        source=DraftRelationSource(
            kind=SourceKind.API_READ,
            read_id="list_summary",
            row_source_id=row_source_id,
        ),
        cardinality="one",
        available_field_ids=("total",),
        available_fields=(SourceField(field_id="total", type="integer"),),
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="row_population.staffs",
                field_id="staffs",
                type="row_population",
                row_cardinality="many",
                row_source_id=row_source_id,
            ),
        ),
        fulfillments=(
            SourceFulfillment(
                requested_fact_id="fact_1",
                answer_output_id="answer_1",
                match_basis_explanation="Nested staff rows are countable.",
                row_count_basis_evidence_ids=("row_population.staffs",),
            ),
        ),
    )

    assert scalar_aggregate_choices_for_source(
        source,
        requested_fact_id="fact_1",
        plan_shape="aggregate_scalar",
    ) == {
        "requested_fact_id": "fact_1",
        "source_binding_id": "sb_1",
        "read_id": "list_summary",
        "plan_shape": "aggregate_scalar",
        "metric_candidates": (
            {
                "id": "metric_1",
                "answer_output_id": "answer_1",
                "kind": "count_records",
                "count_basis": {
                    "kind": "row_population",
                    "row_source_id": row_source_id,
                    "row_path_id": "staffs",
                    "row_cardinality": "many",
                },
                "allowed_functions": ("count",),
            },
        ),
        "function_candidates": (
            {
                "id": "function_count",
                "value": "count",
                "meaning": "number of matching rows",
            },
        ),
    }


def test_structural_row_count_metric_satisfies_answer_output_without_raw_field():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "pattern": "aggregate_scalar",
                    "requested_fact_id": "fact_1",
                    "source_binding_id": "sb_1",
                    "answer_output_ids": ["answer_1"],
                    "metric": _count_metric_selection(),
                    "function": _count_function_selection(),
                }
            ]
        },
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ,
                    read_id="list_records",
                    row_source_id=api_row_source_id("list_records", "data"),
                ),
                cardinality="many",
                available_field_ids=("rank",),
                available_fields=(SourceField(field_id="rank", type="integer"),),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="row_population.data",
                        field_id="data",
                        type="row_population",
                        row_cardinality="many",
                        row_source_id=api_row_source_id("list_records", "data"),
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="Rows are countable.",
                        row_count_basis_evidence_ids=("row_population.data",),
                    ),
                ),
            ),
        ),
    )

    aggregation = plan.operations[0].spec.aggregations[0]
    assert aggregation.function.value == "count"
    assert aggregation.input_field == ""


def test_structural_row_count_metric_binds_relation_to_selected_row_population():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "pattern": "aggregate_scalar",
                    "requested_fact_id": "fact_1",
                    "source_binding_id": "sb_1",
                    "answer_output_ids": ["answer_1"],
                    "metric": _count_metric_selection(),
                    "function": _count_function_selection(),
                }
            ]
        },
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ,
                    read_id="list_records",
                    row_source_id=api_row_source_id("list_records", "data"),
                ),
                cardinality="many",
                available_field_ids=("rank",),
                available_fields=(SourceField(field_id="rank", type="integer"),),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="row_population.data",
                        field_id="data",
                        type="row_population",
                        row_cardinality="many",
                        row_source_id=api_row_source_id("list_records", "data"),
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="Data rows are countable.",
                        row_count_basis_evidence_ids=("row_population.data",),
                    ),
                ),
            ),
        ),
    )

    assert plan.relations[0].source.row_source_id == api_row_source_id(
        "list_records",
        "data",
    )


def test_structural_row_count_metric_executes_count_over_selected_row_population():
    row_source_id = api_row_source_id("list_records", "data")
    source = BoundSource(
        id="sb_1",
        requested_fact_id="fact_1",
        answer_population=_answer_population(),
        source=DraftRelationSource(
            kind=SourceKind.API_READ,
            read_id="list_records",
            row_source_id=row_source_id,
        ),
        source_invocations=(
            DraftRelationSource(
                kind=SourceKind.API_READ,
                read_id="list_records",
                row_source_id=row_source_id,
                param_bindings=(
                    DraftEndpointParamBinding(param_id="status", value="OPEN"),
                ),
            ),
            DraftRelationSource(
                kind=SourceKind.API_READ,
                read_id="list_records",
                row_source_id=row_source_id,
                param_bindings=(
                    DraftEndpointParamBinding(param_id="status", value="CLOSED"),
                ),
            ),
        ),
        cardinality="many",
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="row_population.data",
                field_id="data",
                type="row_population",
                row_cardinality="many",
                row_source_id=row_source_id,
            ),
        ),
        fulfillments=(
            SourceFulfillment(
                requested_fact_id="fact_1",
                answer_output_id="answer_1",
                match_basis_explanation="Data rows are countable.",
                row_count_basis_evidence_ids=("row_population.data",),
            ),
        ),
    )
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "pattern": "aggregate_scalar",
                    "requested_fact_id": "fact_1",
                    "source_binding_id": "sb_1",
                    "answer_output_ids": ["answer_1"],
                    "metric": _count_metric_selection(),
                    "function": _count_function_selection(),
                }
            ]
        },
        bound_sources=(source,),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="list_records",
                endpoint_name="list_records",
                params=(
                    CatalogParam(
                        ref="list_records.query.status",
                        name="status",
                        source=ParamSource.QUERY,
                        type="string",
                    ),
                ),
                row_paths=(
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
            ),
        )
    )

    result = compile_and_invoke(
        plan=FactPlan(outcome=plan),
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="record count",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            description="count",
                        ),
                    ),
                ),
            )
        ),
        catalog=catalog,
        data_access_port=_RowsDataAccess(
            (
                {"rank": 1, "status": "OPEN"},
                {"rank": 2, "status": "CLOSED"},
                {"rank": 3, "status": "ARCHIVED"},
            )
        ),
        memory=LookupMemory(),
    )

    assert result.issue is None
    assert result.fact_result is not None
    assert result.fact_result.outcome.relations[-1].rows == ({"count": 2},)


def test_structural_row_count_metric_preserves_selected_executable_row_source():
    selected_row_source_id = api_row_source_id(
        "list_records",
        "data__include_items_true",
    )
    source = BoundSource(
        id="sb_1",
        requested_fact_id="fact_1",
        answer_population=_answer_population(),
        source=DraftRelationSource(
            kind=SourceKind.API_READ,
            read_id="list_records",
            row_source_id=selected_row_source_id,
            param_bindings=(
                DraftEndpointParamBinding(
                    param_id="include_items",
                    value=True,
                ),
            ),
        ),
        cardinality="many",
        available_field_ids=("rank",),
        available_fields=(SourceField(field_id="rank", type="integer"),),
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="row_population.data",
                field_id="data",
                type="row_population",
                row_cardinality="many",
                row_source_id=selected_row_source_id,
            ),
        ),
        fulfillments=(
            SourceFulfillment(
                requested_fact_id="fact_1",
                answer_output_id="answer_1",
                match_basis_explanation="Data rows are countable.",
                row_count_basis_evidence_ids=("row_population.data",),
            ),
        ),
    )

    assert scalar_aggregate_choices_for_source(
        source,
        requested_fact_id="fact_1",
        plan_shape="aggregate_scalar",
    ) == {
        "requested_fact_id": "fact_1",
        "source_binding_id": "sb_1",
        "read_id": "list_records",
        "plan_shape": "aggregate_scalar",
        "metric_candidates": (
            {
                "id": "metric_1",
                "answer_output_id": "answer_1",
                "kind": "count_records",
                "count_basis": {
                    "kind": "row_population",
                    "row_source_id": selected_row_source_id,
                    "row_path_id": "data",
                    "row_cardinality": "many",
                },
                "allowed_functions": ("count",),
            },
        ),
        "function_candidates": (
            {
                "id": "function_count",
                "value": "count",
                "meaning": "number of matching rows",
            },
        ),
    }

    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "pattern": "aggregate_scalar",
                    "requested_fact_id": "fact_1",
                    "source_binding_id": "sb_1",
                    "answer_output_ids": ["answer_1"],
                    "metric": _count_metric_selection(),
                    "function": _count_function_selection(),
                }
            ]
        },
        bound_sources=(source,),
    )

    assert plan.relations[0].source.row_source_id == selected_row_source_id


def test_pattern_prompt_does_not_require_raw_record_fields_for_count_metric_answer():
    request = FactPlanRequest(
        question="How many active records are there?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="active records count",
                    answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(kind=SourceKind.API_READ, read_id="list_records"),
                cardinality="many",
                available_field_ids=("record_key", "name", "is_active"),
                available_fields=(
                    SourceField(
                        field_id="record_key",
                        type="uuid",
                        roles=("identity",),
                    ),
                    SourceField(field_id="name", type="string"),
                    SourceField(field_id="is_active", type="boolean"),
                ),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="source_1.record_key",
                        field_id="record_key",
                        row_cardinality="many",
                    ),
                    SourceEvidenceItem(
                        evidence_id="source_1.name",
                        field_id="name",
                        row_cardinality="many",
                    ),
                    SourceEvidenceItem(
                        evidence_id="source_1.is_active",
                        field_id="is_active",
                        row_cardinality="many",
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation=(
                            "The records are countable by their record identity."
                        ),
                        group_key_evidence_ids=(
                            "source_1.record_key",
                            "source_1.name",
                            "source_1.is_active",
                        ),
                    ),
                ),
            ),
        ),
    )

    prompt = _pattern_fact_plan_prompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="fact_1",
                    plan_selection_id="fact_1.aggregate_scalar.sb_1",
                    source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                    plan_shape="aggregate_scalar",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )

    assert "Required fulfillment evidence:" not in prompt

def test_list_rows_preserves_source_identity_field_as_relation_grain():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "pattern": "list_rows",
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "source_binding_id": "sb_sales",
                    "output_fields": [{"field_id": "sale_id"}],
                }
            ]
        },
        bound_sources=(
            BoundSource(
                id="sb_sales",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(kind=SourceKind.API_READ, read_id="sales"),
                cardinality="many",
                available_field_ids=("sale_id",),
                available_fields=(
                    SourceField(
                        field_id="sale_id",
                        type="uuid",
                        roles=(
                            FieldBindingRole.IDENTITY.value,
                            FieldBindingRole.OUTPUT.value,
                        ),
                        identity=IdentityMetadata(
                            entity_ref="sale",
                            identity_field="sale_id",
                            primary_key=True,
                            stable=True,
                        ),
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="sale_id answers answer_1.",
                        group_key_evidence_ids=("sale_id",),
                    ),
                ),
            ),
        ),
    )

    assert plan.relations[0].grain_keys == ("sale_id",)

def test_grouped_rows_deduplicates_output_fields_that_repeat_group_fields():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "pattern": "grouped_rows",
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1", "answer_2"],
                    "source_binding_id": "sb_sales",
                    "group_fields": [{"field_id": "sale_id"}],
                    "output_fields": [
                        {"field_id": "item_count"},
                        {"field_id": "sale_id"},
                    ],
                }
            ]
        },
        bound_sources=(
            BoundSource(
                id="sb_sales",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(kind=SourceKind.API_READ, read_id="sales"),
                cardinality="many",
                available_field_ids=("sale_id", "item_count"),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="sale_id answers answer_1.",
                        group_key_evidence_ids=("sale_id",),
                    ),
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_2",
                        match_basis_explanation="item_count answers answer_2.",
                        group_key_evidence_ids=("item_count",),
                    ),
                ),
            ),
        ),
    )

    assert [field.field_id for field in plan.relations[0].fields] == [
        "sale_id",
        "item_count",
    ]
    project = plan.operations[0].spec
    assert [field.output or field.source for field in project.fields] == [
        "sale_id",
        "item_count",
    ]

def test_grouped_rows_fulfillment_tracks_answer_value_field_after_output_dedupe():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "pattern": "grouped_rows",
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "source_binding_id": "sb_sales",
                    "group_fields": [
                        {"field_id": "sale_id"},
                        {"field_id": "snapshot_merch_name"},
                    ],
                    "output_fields": [
                        {"field_id": "sale_id"},
                        {"field_id": "snapshot_merch_name"},
                    ],
                }
            ]
        },
        bound_sources=(
            BoundSource(
                id="sb_sales",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(kind=SourceKind.API_READ, read_id="sales"),
                cardinality="many",
                available_field_ids=("sale_id", "snapshot_merch_name"),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation=(
                            "snapshot_merch_name answers answer_1."
                        ),
                        group_key_evidence_ids=("snapshot_merch_name",),
                    ),
                ),
            ),
        ),
    )

    render_field_by_id = {
        output.id: output.field_id for output in plan.render_spec.relation_outputs
    }

    assert render_field_by_id[plan.fulfillment[0].render_output_id] == (
        "snapshot_merch_name"
    )
