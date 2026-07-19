import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from uuid import UUID

import pytest

from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogEndpointMetadata,
    CatalogFact,
    CatalogField,
    CatalogParam,
    EndpointRead,
    EntityReference,
    EntityReferenceComponent,
    FieldRequirement,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)
from fervis.lookup.lineage.source_reads import SourceReadLineageScope
from fervis.lookup.plan_execution.authorized_sources import (
    AuthorizedExecutionSources,
)
from fervis.lookup.answer_program.instantiation import (
    _materialize_execution,
)
from fervis.lookup.plan_execution.errors import (
    RelationEngineError,
    VerificationError,
)
from tests.lookup.plan_execution.invocation_helpers import compile_and_invoke
from fervis.lookup.plan_execution.verification import (
    verify_fact_plan as verify_fact_plan_impl,
)
from fervis.lineage.enums import ProofNodeKind, SourceReadStatus
from fervis.lineage.recorder import (
    CatalogEndpointWrite,
    RunArtifactWrite,
    SourceReadWrite,
)
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment
from fervis.lookup.fact_plan.fact_plan import FactPlan
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AggregationSpec,
    AntiJoinSpec,
    ComputeSpec,
    CrossJoinSpec,
    FilterSpec,
    JoinKey,
    JoinSpec,
    Operation,
    Predicate,
    PredicateOperator,
    ProjectField,
    ProjectSpec,
    ProjectToKeySpec,
    OrderSpec,
    RelationRole,
    RelationRoleRef,
    RoleExpandSpec,
    RoleMapping,
    SortDirection,
    SortKey,
    Take,
)
from fervis.lookup.answer_program.expressions import (
    BinaryExpression,
    ExpressionBinaryOperator,
    FieldRef,
    ParameterRef,
)
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    EndpointParamBinding,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_plan.row_sources import (
    api_row_source_id,
    build_row_source_catalog,
)
from fervis.lookup.answer_program.values import (
    ConstantRef,
    FactValue,
    NodeOutputRef,
    TimeComponent,
    ValueComponent,
)
from fervis.lookup.answer_program import BindingSet, compiler_input_context
from fervis.lookup.answer_program import AnswerProgramContractError
from fervis.lookup.answer_program.values import LiteralType
from fervis.lookup.canonical_data import entity_key_value
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
)
from fervis.lookup.answer_program.result_projection import (
    EntityKeyProjection,
    EntityKeyProjectionComponent,
    RelationResultOutput,
    ScalarResultOutput,
    ResultProjection,
)


def _answer_plan(**kwargs) -> AnswerProgram:
    result_projection = kwargs.get("result_projection")
    operations = tuple(kwargs.get("operations", ()))
    fulfillment = _default_fulfillment(result_projection, operations)
    kwargs.pop("requested_facts", None)
    if "fulfillment" not in kwargs:
        kwargs["fulfillment"] = fulfillment
    return AnswerProgram(
        **kwargs,
    )


def _question_contract(
    description: str = "answer",
    *,
    binding_target_ids: tuple[str, ...] = ("answer",),
    known_inputs: tuple[RequestedFactKnownInput, ...] = (),
) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description=description,
                answer_outputs=tuple(
                    RequestedFactAnswerOutput(id=binding_target_id, role="ANSWER_VALUE")
                    for binding_target_id in binding_target_ids
                ),
                known_inputs=known_inputs,
            ),
        )
    )


def _known_reference(
    input_id: str,
    text: str,
    *,
    value_meaning_hint: str = "",
) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=text,
        value_meaning_hint=value_meaning_hint,
        role=LiteralInputRole.REFERENCE_VALUE,
    )


def _known_time(input_id: str, text: str) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=text,
        role=LiteralInputRole.TIME_VALUE,
    )


def _known_result_limit(
    input_id: str, text: str, value: int | str
) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=str(value),
        role=LiteralInputRole.RESULT_LIMIT,
    )


def _rank_limit_constant(value: int) -> ConstantRef:
    return ConstantRef(
        constant_id=f"rank-limit.{value}",
        version_ref="rank@1",
        value=FactValue.literal(
            id=f"rank-limit.{value}",
            literal_type=LiteralType.NUMBER,
            value=str(value),
        ),
    )


def _constant_expression(value: FactValue, *, ref_id: str) -> ConstantRef:
    return ConstantRef(
        constant_id=f"test.{ref_id}",
        version_ref="test@1",
        value=value,
    )


def verify_fact_plan(plan: FactPlan, **kwargs):
    original_plan = plan
    catalog = kwargs.pop("catalog", _catalog())
    question_contract = kwargs.pop(
        "question_contract",
        _question_contract(
            _default_description(plan),
            binding_target_ids=_result_output_ids(plan),
        ),
    )
    explicit_values = tuple(kwargs.pop("available_values", ()))
    kwargs.pop("available_value_uses", ())
    input_context = compiler_input_context(
        values=(
            *_available_values_for_contract(question_contract),
            *explicit_values,
        ),
        question_contract=question_contract,
    )
    if isinstance(plan.outcome, AnswerProgram):
        plan = replace(
            plan,
            outcome=replace(
                plan.outcome,
                parameters=input_context.program_inputs.parameters,
            ),
            bindings=input_context.program_inputs.bindings,
        )
        from fervis.lookup.answer_program.compilation import compile_answer_program
        from fervis.lookup.answer_program.instantiation import (
            ExecutionEnvironment,
            instantiate_answer_program,
        )

        memory_relations = tuple(kwargs.pop("memory_relations", ()))
        authorized_sources = kwargs.pop("authorized_sources", None)
        catalog_selection = kwargs.pop("catalog_selection", None)
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
            memory_relations=memory_relations,
        )
        instantiate_answer_program(
            program,
            bindings,
            ExecutionEnvironment(
                catalog=execution_catalog,
                authorized_sources=authorized_sources,
                catalog_selection=catalog_selection,
                memory_relations=memory_relations,
            ),
        )
    else:
        verify_fact_plan_impl(
            plan,
            question_contract=question_contract,
            catalog=catalog,
            **kwargs,
        )
    return original_plan


def _grounded_input_use(
    *,
    value_id: str,
    requested_fact_id: str,
    row_source_id: str | None = None,
    param_id: str,
    value_component=ValueComponent.VALUE,
) -> GroundedInputUse:
    row_source_id = row_source_id or api_row_source_id("records", "data")
    return GroundedInputUse(
        id=f"grounded::{row_source_id}::{param_id}::{value_component.value}",
        value_id=value_id,
        row_source_id=row_source_id,
        param_id=param_id,
        requested_fact_id=requested_fact_id,
        value_component=value_component,
    )


def _grounded_time_value(
    value_id: str,
    *,
    start: str,
    end: str,
) -> FactValue:
    return FactValue.time(
        id=value_id,
        expression=value_id,
        resolved_start=start,
        resolved_end=end,
        granularity="day",
    )


def _available_values_for_contract(
    question_contract: QuestionContract,
) -> tuple[FactValue, ...]:
    values: list[FactValue] = []
    if question_contract.question_inputs:
        known_inputs = question_contract.question_inputs
    else:
        known_inputs = tuple(
            known
            for fact in question_contract.requested_facts
            for known in fact.known_inputs
        )
    for known in known_inputs:
        if known.is_result_limit:
            values.append(
                FactValue.literal(
                    id=known.id,
                    known_input_id=known.id,
                    literal_type=LiteralType.NUMBER,
                    value=known.resolved_value_text,
                    proof_refs=(f"known_input:{known.id}",),
                )
            )
            continue
        if known.is_time_value:
            values.append(
                FactValue.time(
                    id=known.id,
                    known_input_id=known.id,
                    expression=known.text,
                    resolved_start="2026-04-08",
                    resolved_end="2026-04-08",
                    granularity="day",
                    proof_refs=(f"known_input:{known.id}",),
                )
            )
            continue
        values.append(
            FactValue.named(
                id=known.id,
                known_input_id=known.id,
                text=known.text,
                proof_refs=(f"known_input:{known.id}",),
            )
        )
    return tuple(values)


def _default_description(plan: FactPlan) -> str:
    outcome = plan.outcome
    result_projection = outcome.result_projection
    if result_projection is not None and result_projection.scalar_outputs:
        return result_projection.scalar_outputs[0].scalar_id
    if result_projection.relation_outputs:
        slot = result_projection.relation_outputs[0]
        if slot.entity_key is not None:
            return slot.entity_key.entity_kind
        return _source_description(outcome, slot.relation_id, slot.field_id)
    return "field.name"


def _result_output_ids(plan: FactPlan) -> tuple[str, ...]:
    outcome = plan.outcome
    binding_target_ids = tuple(
        slot.id
        for slot in (
            *outcome.result_projection.relation_outputs,
            *outcome.result_projection.scalar_outputs,
        )
    )
    return binding_target_ids or ("answer",)


def _source_description(answer: AnswerProgram, relation_id: str, field_id: str) -> str:
    seen: set[tuple[str, str]] = set()
    return _source_description_inner(answer, relation_id, field_id, seen=seen)


def _source_description_inner(
    answer: AnswerProgram,
    relation_id: str,
    field_id: str,
    *,
    seen: set[tuple[str, str]],
) -> str:
    key = (relation_id, field_id)
    if key in seen:
        return field_id
    seen.add(key)
    bindings = {
        relation.id: {field.field_id: field.field_id for field in relation.fields}
        for relation in answer.relations
    }
    operation = next(
        (item for item in answer.operations if item.output_relation == relation_id),
        None,
    )
    if operation is None:
        return bindings.get(relation_id, {}).get(field_id, field_id)
    spec = operation.spec
    if isinstance(spec, ProjectSpec):
        for field in spec.fields:
            if (field.output or field.source) == field_id:
                return _source_description_inner(
                    answer, spec.input_relation, field.source, seen=seen
                )
    if isinstance(spec, ProjectToKeySpec):
        if field_id in spec.key_fields:
            return _source_description_inner(
                answer, spec.input_relation, field_id, seen=seen
            )
    if isinstance(spec, AggregateSpec):
        if field_id in spec.group_by:
            return _source_description_inner(
                answer, spec.input_relation, field_id, seen=seen
            )
        for aggregation in spec.aggregations:
            if aggregation.output_field == field_id and aggregation.input_field:
                return _source_description_inner(
                    answer, spec.input_relation, aggregation.input_field, seen=seen
                )
    return field_id


def _default_fulfillment(
    result_projection: ResultProjection | None,
    operations: tuple[Operation, ...],
) -> tuple[FactFulfillment, ...]:
    if result_projection is not None:
        outputs = (
            *result_projection.relation_outputs,
            *result_projection.scalar_outputs,
        )
        if outputs:
            return tuple(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id=output.id,
                    result_output_id=output.id,
                )
                for output in outputs
            )
    return (
        FactFulfillment(
            requested_fact_id="rf_answer",
            answer_output_id="answer",
            result_output_id="answer",
        ),
    )


def _source() -> RelationSource:
    return RelationSource(
        kind=SourceKind.API_READ,
        read_id="records",
    )


def _entity_reference(
    reference_id: str,
    component_id: str,
    field_ref: str,
    *,
    target_entity_kind: str | None = None,
) -> EntityReference:
    entity_kind = target_entity_kind or reference_id
    return EntityReference(
        id=f"{reference_id}_reference",
        target_entity_kind=entity_kind,
        target_key_id="primary_key",
        components=(
            EntityReferenceComponent(
                target_component_id=component_id,
                local_field_ref=field_ref,
            ),
        ),
    )


def _authority_read(entity_kind: str, component_id: str) -> EndpointRead:
    field_ref = f"authority.{entity_kind}.{component_id}"
    return EndpointRead(
        id=f"{entity_kind}_authority",
        endpoint_name=f"list_{entity_kind}_authority",
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref=field_ref,
                path=f"data.{component_id}",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                id="primary_key",
                entity_kind=entity_kind,
                components=(
                    CandidateKeyComponent(id=component_id, field_ref=field_ref),
                ),
                primary=True,
            ),
        ),
    )


def _catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                catalog_endpoint=_catalog_endpoint_metadata(),
                params=(
                    CatalogParam(
                        ref="list_records.query.start_date",
                        name="start_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(ref="field.name", path="data.name", type="string"),
                    CatalogField(
                        ref="field.entity_id",
                        path="data.entity_id",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.observed_id",
                        path="data.observed_id",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.row_id",
                        path="data.row_id",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.event_id",
                        path="data.event_id",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.display", path="data.display", type="string"
                    ),
                    CatalogField(
                        ref="field.entity_label",
                        path="data.entity_label",
                        type="string",
                    ),
                    CatalogField(ref="field.amount", path="data.amount", type="number"),
                    CatalogField(
                        ref="field.other_name",
                        path="data.other_name",
                        type="string",
                    ),
                ),
                entity_references=(
                    _entity_reference("entity", "entity_id", "field.entity_id"),
                    _entity_reference(
                        "observed_entity",
                        "entity_id",
                        "field.observed_id",
                        target_entity_kind="entity",
                    ),
                    _entity_reference("row", "row_id", "field.row_id"),
                    _entity_reference("event", "event_id", "field.event_id"),
                ),
            ),
            _authority_read("entity", "entity_id"),
            _authority_read("row", "row_id"),
            _authority_read("event", "event_id"),
        )
    )


def _catalog_with_root_and_data_row_paths() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                catalog_endpoint=_catalog_endpoint_metadata(),
                row_paths=(
                    RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.total",
                        path="total",
                        type="number",
                        row_path_id="root",
                    ),
                    CatalogField(
                        ref="field.name",
                        path="data.name",
                        type="string",
                        row_path_id="data",
                    ),
                ),
            ),
        )
    )


def _catalog_endpoint_metadata() -> CatalogEndpointMetadata:
    return CatalogEndpointMetadata(
        catalog_endpoint_key="django_tests_list_records:test",
        endpoint_name="list_records",
        framework_kind="django_drf",
        source_namespace_kind="django_app",
        source_namespace_path=("tests",),
        route_method="GET",
        route_path_template="/records/",
        handler_ref="tests.RecordView",
        domain_resource_names=("record",),
    )


class _ReplayDataAccess:
    def __init__(self, *, responses: dict[str, dict[str, object]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def read(self, *, endpoint_name: str, args: dict[str, object]) -> dict[str, object]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        return self.responses[endpoint_name]


class _SourceReadRecorder:
    def __init__(self) -> None:
        self.catalog_endpoints: list[CatalogEndpointWrite] = []
        self.source_reads: list[SourceReadWrite] = []
        self.artifacts: list[RunArtifactWrite] = []

    def record_catalog_endpoint(
        self,
        catalog_endpoint: CatalogEndpointWrite,
    ) -> CatalogEndpointWrite:
        self.catalog_endpoints.append(catalog_endpoint)
        return catalog_endpoint

    def record_source_read(self, source_read: SourceReadWrite) -> SourceReadWrite:
        self.source_reads.append(source_read)
        return source_read

    def record_artifact(self, artifact: RunArtifactWrite) -> RunArtifactWrite:
        self.artifacts.append(artifact)
        return artifact


def _canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _project_operation(
    *,
    input_relation: str = "rows",
    output_relation: str = "result",
) -> Operation:
    return Operation(
        id="project",
        spec=ProjectSpec(
            input_relation=input_relation,
            fields=(ProjectField(source="name", output="name"),),
        ),
        output_relation=output_relation,
    )


def _rows_relation() -> Relation:
    return Relation(
        id="rows",
        source=_source(),
        fields=(
            RelationField(
                field_id="name",
                roles=(FieldBindingRole.OUTPUT,),
            ),
        ),
    )


def _row_filter_operation(
    expression,
    *,
    field_id: str = "name",
) -> Operation:
    return Operation(
        id="filter_rows",
        spec=FilterSpec(
            input_relation="rows",
            predicate=Predicate(
                left=FieldRef(field_id),
                operator=PredicateOperator.EQUALS,
                right=expression,
            ),
        ),
        output_relation="filtered_rows",
    )


def _known_input_filter(known_input_id: str) -> Operation:
    return _row_filter_operation(
        _constant_expression(
            FactValue.named(
                id=f"filter_{known_input_id}",
                text="yesterday",
                proof_refs=(f"known_input:{known_input_id}",),
            ),
            ref_id=f"filter_{known_input_id}",
        ),
        field_id="name",
    )


def test_filter_references_existing_value():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                _row_filter_operation(ParameterRef("missing")),
                _project_operation(input_relation="filtered_rows"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(AnswerProgramContractError, match="unknown parameter"):
        verify_fact_plan(plan)


def test_filter_can_reference_known_question_input():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                _row_filter_operation(ParameterRef("question.person_name")),
                _project_operation(input_relation="filtered_rows"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(
        plan,
        question_contract=_question_contract(
            known_inputs=(_known_reference("person_name", "Alice"),)
        ),
    )


def test_known_inputs_are_inventory_not_automatic_obligations():
    prior_context = _known_reference("prior_sales_context", "KES 80k yesterday")
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(_project_operation(input_relation="rows"),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(
        plan,
        question_contract=QuestionContract(
            question_inputs=(prior_context,),
            requested_facts=(
                RequestedFact(
                    id="rf_answer",
                    description="answer",
                    answer_outputs=(
                        RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                    ),
                ),
            ),
        ),
    )


def test_unclassified_literal_cannot_supply_filter_value():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                _row_filter_operation({"literal": "Alice"}),
                _project_operation(input_relation="filtered_rows"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(
        AnswerProgramContractError,
        match="declared value origin",
    ):
        verify_fact_plan(plan)


def test_unused_known_time_input_does_not_require_runtime_anchors():
    prior_context = _known_time("prior_day_context", "yesterday")
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(_project_operation(input_relation="rows"),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(
        plan,
        question_contract=QuestionContract(
            question_inputs=(prior_context,),
            requested_facts=(
                RequestedFact(
                    id="rf_answer",
                    description="answer",
                    answer_outputs=(
                        RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                    ),
                ),
            ),
        ),
    )


def test_endpoint_time_param_can_reference_grounded_time_input():
    runtime_date = _grounded_time_value(
        "runtime_date",
        start="2026-04-08",
        end="2026-04-08",
    )
    relation = _rows_relation()
    relation = replace(
        relation,
        source=replace(
            relation.source,
            param_bindings=(
                EndpointParamBinding(
                    param_id="start_date",
                    value_expr=ConstantRef(
                        constant_id="context.runtime_date",
                        version_ref="context-value@1",
                        value=runtime_date,
                        component=TimeComponent.START.value,
                    ),
                ),
            ),
        ),
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(relation,),
            operations=(
                _known_input_filter("period"),
                _project_operation(input_relation="filtered_rows"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(
        plan,
    )


def test_endpoint_time_param_requires_grounded_time_value():
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                params=(
                    CatalogParam(
                        ref="list_records.query.start_date",
                        name="start_date",
                        source=ParamSource.QUERY,
                        type="date",
                        required=True,
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(ref="field.name", path="data.name", type="string"),
                ),
            ),
        )
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(_project_operation(input_relation="rows"),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="requires source param start_date"):
        verify_fact_plan(plan, catalog=catalog)


def test_endpoint_requirement_uses_selected_time_component():
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                params=(
                    CatalogParam(
                        ref="list_records.query.end_date",
                        name="end_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.name",
                        path="data.name",
                        type="string",
                        requirements=(
                            FieldRequirement(
                                param_ref="list_records.query.end_date",
                                value="2026-04-30",
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    month = _grounded_time_value(
        "month",
        start="2026-04-01",
        end="2026-04-30",
    )
    relation = Relation(
        id="rows",
        source=RelationSource(
            kind=SourceKind.API_READ,
            read_id="records",
            param_bindings=(
                EndpointParamBinding(
                    param_id="end_date",
                    value_expr=ConstantRef(
                        constant_id="context.month",
                        version_ref="context-value@1",
                        value=month,
                        component=TimeComponent.END.value,
                    ),
                ),
            ),
        ),
        fields=(
            RelationField(
                field_id="name",
                roles=(FieldBindingRole.OUTPUT,),
            ),
        ),
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(relation,),
            operations=(_project_operation(input_relation="rows"),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(
        plan,
        catalog=catalog,
    )


def test_empty_relation_id_is_rejected():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(_project_operation(input_relation=""),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="relation requires id"):
        verify_fact_plan(plan)


def test_empty_operation_id_is_rejected():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                Operation(
                    id="",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="name"),),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="operation requires id"):
        verify_fact_plan(plan)


def test_fact_local_known_inputs_are_canonicalized_to_shared_question_inputs():
    person = _known_reference("person_name", "Alice")
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description="answer",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                ),
                known_inputs=(person,),
            ),
            RequestedFact(
                id="rf_context",
                description="context",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="context", role="ANSWER_VALUE"),
                ),
                known_inputs=(person,),
            ),
        )
    )

    assert question_contract.question_inputs == (person,)
    assert [fact.input_refs for fact in question_contract.requested_facts] == [
        ("person_name",),
        ("person_name",),
    ]


def test_shared_question_input_refs_are_valid_across_answer_requests():
    period = _known_time("period", "yesterday")
    question_contract = QuestionContract(
        question_inputs=(period,),
        requested_facts=(
            RequestedFact(
                id="rf_amount",
                description="sales amount yesterday",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="amount", role="ANSWER_VALUE"),
                ),
                known_inputs=(period,),
                input_refs=("period",),
            ),
            RequestedFact(
                id="rf_store",
                description="sales store yesterday",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="store", role="ANSWER_VALUE"),
                ),
                known_inputs=(period,),
                input_refs=("period",),
            ),
        ),
    )
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_amount",
                    answer_output_id="amount",
                    result_output_id="amount",
                ),
                FactFulfillment(
                    requested_fact_id="rf_store",
                    answer_output_id="store",
                    result_output_id="store",
                ),
            ),
            relations=(_rows_relation(),),
            operations=(
                _known_input_filter("period"),
                _project_operation(input_relation="filtered_rows"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="amount",
                        relation_id="result",
                        field_id="name",
                    ),
                    RelationResultOutput(
                        id="store",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    assert (
        verify_fact_plan(plan, question_contract=question_contract, catalog=_catalog())
        is plan
    )


def test_known_limit_input_must_match_order_take_limit():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                Operation(
                    id="top_rows",
                    spec=OrderSpec(
                        input_relation="rows",
                        order_by=(SortKey(field="name", direction=SortDirection.DESC),),
                        tie_breakers=(
                            SortKey(field="name", direction=SortDirection.ASC),
                        ),
                        selection=Take(
                            limit=ParameterRef("question.result_limit")
                        ),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(
        plan,
        question_contract=_question_contract(
            known_inputs=(_known_result_limit("result_limit", "top 5", 5),)
        ),
    )


def test_order_take_allows_literal_limit_without_bound_known_input():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                Operation(
                    id="top_rows",
                    spec=OrderSpec(
                        input_relation="rows",
                        order_by=(SortKey(field="name", direction=SortDirection.DESC),),
                        tie_breakers=(
                            SortKey(field="name", direction=SortDirection.ASC),
                        ),
                        selection=Take(limit=_rank_limit_constant(5)),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(plan)


def test_order_take_expression_requires_positive_integer_value():
    for value in ("0", "-1", "5.5"):
        with pytest.raises(VerificationError, match="positive integer"):
            plan = FactPlan(
                outcome=_answer_plan(
                    relations=(_rows_relation(),),
                    operations=(
                        Operation(
                            id="top_rows",
                            spec=OrderSpec(
                                input_relation="rows",
                                order_by=(
                                    SortKey(
                                        field="name",
                                        direction=SortDirection.DESC,
                                    ),
                                ),
                                tie_breakers=(
                                    SortKey(
                                        field="name",
                                        direction=SortDirection.ASC,
                                    ),
                                ),
                                selection=Take(
                                    limit=_constant_expression(
                                        FactValue.literal(
                                            id=f"rank_limit_{value}",
                                            literal_type=LiteralType.NUMBER,
                                            value=value,
                                        ),
                                        ref_id=f"rank_limit_{value}",
                                    )
                                ),
                            ),
                            output_relation="result",
                        ),
                    ),
                    result_projection=ResultProjection(relation_outputs=()),
                )
            )
            verify_fact_plan(plan)


def test_order_take_rejects_non_numeric_expression():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                Operation(
                    id="top_rows",
                    spec=OrderSpec(
                        input_relation="rows",
                        order_by=(SortKey(field="name", direction=SortDirection.DESC),),
                        tie_breakers=(
                            SortKey(field="name", direction=SortDirection.ASC),
                        ),
                        selection=Take(
                            limit=_constant_expression(
                                FactValue.literal(
                                    id="rank_limit_four",
                                    literal_type=LiteralType.STRING,
                                    value="four",
                                ),
                                ref_id="rank_limit_four",
                            )
                        ),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="positive integer"):
        verify_fact_plan(plan)


def test_fulfillment_uses_visible_requested_fact_id_and_rendered_output():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(_project_operation(),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="name", relation_id="result", field_id="name"
                    ),
                )
            ),
        )
    )
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description="person name",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="name", role="ANSWER_VALUE"),
                ),
            ),
        )
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(ref="field.name", path="data.name", type="string"),
                ),
                facts=(
                    CatalogFact(
                        ref="person.name",
                        field_ref="field.name",
                        read_id="records",
                    ),
                ),
            ),
        )
    )

    assert (
        verify_fact_plan(plan, question_contract=question_contract, catalog=catalog)
        is plan
    )


def test_field_binding_id_must_exist_on_row_source():
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="name",
                    result_output_id="name",
                ),
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            relations=(
                Relation(
                    id="rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="restricted.full_value",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(
                            ProjectField(
                                source="restricted.full_value",
                                output="answer",
                            ),
                        ),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="answer",
                    ),
                )
            ),
        )
    )
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description="restricted full value",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                ),
            ),
        )
    )

    with pytest.raises(VerificationError, match="unknown source field"):
        verify_fact_plan(plan, question_contract=question_contract, catalog=_catalog())


def test_proof_backed_scalar_output_can_satisfy_requested_derived_fact():
    current_sales = FactValue.literal(
        id="current_sales",
        literal_type=LiteralType.NUMBER,
        value="35",
        proof_refs=("prior.sales_total",),
    )
    target_value = FactValue.literal(
        id="target_value",
        literal_type=LiteralType.NUMBER,
        value="100",
        proof_refs=("question.target",),
    )
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            relations=(_rows_relation(),),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="name", output="name"),),
                    ),
                    output_relation="result",
                ),
                Operation(
                    id="compute",
                    spec=ComputeSpec(
                        expression=BinaryExpression(
                            operator=ExpressionBinaryOperator.SUBTRACT,
                            left=_constant_expression(
                                target_value,
                                ref_id="target_value",
                            ),
                            right=_constant_expression(
                                current_sales,
                                ref_id="current_sales",
                            ),
                        ),
                        output_scalar="remaining",
                    ),
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="name", relation_id="result", field_id="name"
                    ),
                ),
                scalar_outputs=(
                    ScalarResultOutput(
                        id="answer",
                        scalar_id="remaining",
                    ),
                ),
            ),
        )
    )
    question_contract = _question_contract("remaining")
    assert (
        verify_fact_plan(
            plan,
            question_contract=question_contract,
            catalog=_catalog(),
        )
        is plan
    )
    program = plan.outcome
    bindings = BindingSet()
    compiled = _materialize_execution(
        answer=program,
        bindings=bindings,
        catalog=_catalog(),
        row_sources=build_row_source_catalog(_catalog()),
    )

    assert any(
        node.kind is ProofNodeKind.SCALAR and node.id == "scalar:remaining"
        for node in compiled.proof_graph.nodes
    )


def test_chained_compute_scalar_output_preserves_evidence_proof():
    current_sales = FactValue.literal(
        id="current_sales",
        literal_type=LiteralType.NUMBER,
        value="35",
        proof_refs=("prior.sales_total",),
    )
    target_value = FactValue.literal(
        id="target_value",
        literal_type=LiteralType.NUMBER,
        value="100",
        proof_refs=("question.target",),
    )
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            relations=(_rows_relation(),),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="name", output="name"),),
                    ),
                    output_relation="result",
                ),
                Operation(
                    id="subtotal",
                    spec=ComputeSpec(
                        expression=BinaryExpression(
                            operator=ExpressionBinaryOperator.SUBTRACT,
                            left=_constant_expression(
                                target_value,
                                ref_id="target_value",
                            ),
                            right=_constant_expression(
                                current_sales,
                                ref_id="current_sales",
                            ),
                        ),
                        output_scalar="subtotal",
                    ),
                ),
                Operation(
                    id="final",
                    spec=ComputeSpec(
                        expression=NodeOutputRef(
                            node_id="subtotal",
                            output_id="subtotal",
                        ),
                        output_scalar="final_total",
                    ),
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="name",
                        relation_id="result",
                        field_id="name",
                    ),
                ),
                scalar_outputs=(
                    ScalarResultOutput(
                        id="answer",
                        scalar_id="final_total",
                    ),
                ),
            ),
        )
    )
    assert (
        verify_fact_plan(
            plan,
            question_contract=_question_contract("remaining"),
            catalog=_catalog(),
        )
        is plan
    )


def test_fulfillment_must_reference_projected_relation_field():
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="internal_total",
                ),
            ),
            relations=(_rows_relation(),),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(
                            ProjectField(source="name", output="display_total"),
                            ProjectField(source="name", output="internal_total"),
                        ),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="display_total",
                    ),
                )
            ),
        )
    )

    with pytest.raises(
        VerificationError, match="fulfillment result output is not projected"
    ):
        verify_fact_plan(plan)


def test_fulfillment_allows_fact_scoped_selected_read_without_global_selection():
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_sales",
                    answer_output_id="sales_total",
                    result_output_id="sales_total",
                ),
            ),
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                    ),
                    fields=(
                        RelationField(
                            field_id="staff_name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                        RelationField(
                            field_id="sales_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="staff_name"),),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="sales_total",
                        relation_id="result",
                        field_id="staff_name",
                    ),
                )
            ),
        )
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.staff_name",
                        path="data.staff_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.sales_total",
                        path="data.sales_total",
                        row_path_id="data",
                        type="number",
                    ),
                ),
                facts=(
                    CatalogFact(ref="staff.name", field_ref="field.staff_name"),
                    CatalogFact(ref="sales.total", field_ref="field.sales_total"),
                ),
            ),
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=RelationCatalog(),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_sales",
                query_terms=("sales", "total"),
                rankings=(
                    CatalogSelectionRanking(
                        read_id="records",
                        score=2,
                        matched_terms=("sales", "total"),
                        matched_fact_refs=("sales.total",),
                        matched_field_refs=("field.sales_total",),
                    ),
                ),
                selected_read_ids=("records",),
            ),
        ),
        selected_read_ids=(),
    )

    verify_fact_plan(
        plan,
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="rf_sales",
                    description="sales total",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="sales_total", role="ANSWER_VALUE"
                        ),
                    ),
                ),
            )
        ),
        catalog=catalog,
        catalog_selection=selection,
    )


def test_fulfillment_allows_source_binding_authorized_api_replay():
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_sales",
                    answer_output_id="sales_total",
                    result_output_id="sales_total",
                ),
            ),
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                    ),
                    fields=(
                        RelationField(
                            field_id="sales_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="sales_total"),),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="sales_total",
                        relation_id="result",
                        field_id="sales_total",
                    ),
                )
            ),
        )
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.sales_total",
                        path="data.sales_total",
                        row_path_id="data",
                        type="number",
                    ),
                ),
            ),
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_sales",
                query_terms=("shade",),
                rankings=(),
                selected_read_ids=(),
            ),
        ),
        selected_read_ids=(),
    )

    execution_sources = AuthorizedExecutionSources.from_pipeline_sources(
        full_catalog=catalog,
        catalog_selection=selection,
        relation_sources=(RelationSource(kind=SourceKind.API_READ, read_id="records"),),
    )

    assert execution_sources.api_read_ids == ("records",)
    assert [read.id for read in execution_sources.relation_catalog.reads] == ["records"]

    verify_fact_plan(
        plan,
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="rf_sales",
                    description="sales total",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="sales_total", role="ANSWER_VALUE"
                        ),
                    ),
                ),
            )
        ),
        catalog=selection.relation_catalog,
        catalog_selection=selection,
        authorized_sources=execution_sources,
    )


def test_execution_uses_same_authorized_catalog_as_verification():
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_sales",
                    answer_output_id="sales_total",
                    result_output_id="sales_total",
                ),
            ),
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(kind=SourceKind.API_READ, read_id="records"),
                    fields=(
                        RelationField(
                            field_id="sales_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="sales_total"),),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="sales_total",
                        relation_id="result",
                        field_id="sales_total",
                    ),
                )
            ),
        )
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.sales_total",
                        path="data.sales_total",
                        row_path_id="data",
                        type="number",
                    ),
                ),
            ),
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=RelationCatalog(),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_sales",
                query_terms=("shade",),
                rankings=(),
                selected_read_ids=(),
            ),
        ),
        selected_read_ids=(),
    )
    execution_sources = AuthorizedExecutionSources.from_pipeline_sources(
        full_catalog=catalog,
        catalog_selection=selection,
        relation_sources=(RelationSource(kind=SourceKind.API_READ, read_id="records"),),
    )
    data_access = _ReplayDataAccess(
        responses={
            "list_records": {
                "responseStatus": 200,
                "responseBody": {"data": [{"sales_total": "12"}]},
            }
        }
    )

    result = compile_and_invoke(
        plan=plan,
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="rf_sales",
                    description="sales total",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="sales_total", role="ANSWER_VALUE"
                        ),
                    ),
                ),
            )
        ),
        catalog=selection.relation_catalog,
        catalog_selection=selection,
        authorized_sources=execution_sources,
        data_access_port=data_access,
        memory=LookupMemory(),
    )

    assert data_access.requests == [{"endpointName": "list_records", "args": {}}]
    assert result.relations[0].rows == ({"sales_total": Decimal("12")},)


def test_api_execution_records_source_read_lineage():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(kind=SourceKind.API_READ, read_id="records"),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(_project_operation(),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )
    data_access = _ReplayDataAccess(
        responses={
            "list_records": {
                "responseStatus": 200,
                "responseBody": {"data": [{"name": "Downtown"}]},
                "pageCount": 1,
                "truncated": False,
            }
        }
    )
    recorder = _SourceReadRecorder()

    result = compile_and_invoke(
        plan=plan,
        question_contract=_question_contract(),
        catalog=_catalog(),
        data_access_port=data_access,
        memory=LookupMemory(),
        source_read_lineage=SourceReadLineageScope(
            run_id="run_1",
            step_id="step_execute",
            recorder=recorder,
        ),
    )

    assert result.relations[0].rows == ({"name": "Downtown"},)
    assert len(recorder.source_reads) == 1
    source_read = recorder.source_reads[0]
    assert source_read.run_id == "run_1"
    assert source_read.step_id == "step_execute"
    UUID(source_read.catalog_endpoint_id)
    assert (
        source_read.catalog_endpoint_id
        == recorder.catalog_endpoints[0].catalog_endpoint_id
    )
    assert recorder.catalog_endpoints[0].endpoint_name == "list_records"
    assert (
        recorder.catalog_endpoints[0].catalog_endpoint_key
        == "django_tests_list_records:test"
    )
    assert source_read.args_json == {}
    assert source_read.row_count == 1
    assert source_read.completeness_json == {
        "pageCount": 1,
        "truncated": False,
    }
    assert source_read.response_hash == _canonical_json_hash(
        {"data": [{"name": "Downtown"}]}
    )
    source_read_ref = f"source_read:{source_read.source_read_id}"
    assert source_read_ref in result.proof_refs
    relation_nodes = [
        node
        for node in result.proof_graph.nodes
        if node.kind is ProofNodeKind.RELATION and node.id == "relation:rows"
    ]
    assert relation_nodes[0].proof_refs == ("read:list_records", source_read_ref)


def test_api_execution_records_one_source_read_for_one_backend_request():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="summary",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                        row_source_id=api_row_source_id("records", "root"),
                    ),
                    fields=(
                        RelationField(
                            field_id="total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                        row_source_id=api_row_source_id("records", "data"),
                    ),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_summary",
                    spec=ProjectSpec(
                        input_relation="summary",
                        fields=(ProjectField(source="total", output="total"),),
                    ),
                    output_relation="summary_result",
                ),
                _project_operation(input_relation="rows", output_relation="row_result"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="summary_answer",
                        relation_id="summary_result",
                        field_id="total",
                    ),
                    RelationResultOutput(
                        id="row_answer",
                        relation_id="row_result",
                        field_id="name",
                    ),
                )
            ),
        )
    )
    data_access = _ReplayDataAccess(
        responses={
            "list_records": {
                "responseStatus": 200,
                "responseBody": {
                    "total": 1,
                    "data": [{"name": "Downtown"}],
                },
            }
        }
    )
    recorder = _SourceReadRecorder()

    result = compile_and_invoke(
        plan=plan,
        question_contract=_question_contract(
            binding_target_ids=("summary_answer", "row_answer")
        ),
        catalog=_catalog_with_root_and_data_row_paths(),
        data_access_port=data_access,
        memory=LookupMemory(),
        source_read_lineage=SourceReadLineageScope(
            run_id="run_1",
            step_id="step_execute",
            recorder=recorder,
        ),
    )

    assert data_access.requests == [{"endpointName": "list_records", "args": {}}]
    assert [item.completeness_json for item in recorder.source_reads] == [
        {"pageCount": 1, "truncated": False},
    ]
    assert [item.row_count for item in recorder.source_reads] == [1]
    assert len({item.source_read_id for item in recorder.source_reads}) == 1
    assert all(
        f"source_read:{item.source_read_id}" in result.proof_refs
        for item in recorder.source_reads
    )


def test_api_execution_records_failed_source_read_when_response_shape_is_invalid():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(kind=SourceKind.API_READ, read_id="records"),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(_project_operation(),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )
    data_access = _ReplayDataAccess(
        responses={
            "list_records": {
                "responseStatus": 200,
                "responseBody": {"unexpected": []},
            }
        }
    )
    recorder = _SourceReadRecorder()

    with pytest.raises(RelationEngineError) as exc_info:
        compile_and_invoke(
            plan=plan,
            question_contract=_question_contract(),
            catalog=_catalog(),
            data_access_port=data_access,
            memory=LookupMemory(),
            source_read_lineage=SourceReadLineageScope(
                run_id="run_1",
                step_id="step_execute",
                recorder=recorder,
            ),
        )

    assert len(recorder.source_reads) == 1
    source_read = recorder.source_reads[0]
    UUID(source_read.catalog_endpoint_id)
    assert (
        source_read.catalog_endpoint_id
        == recorder.catalog_endpoints[0].catalog_endpoint_id
    )
    assert recorder.catalog_endpoints[0].endpoint_name == "list_records"
    assert source_read.status == SourceReadStatus.SUCCEEDED
    assert source_read.error_json == {}
    assert "response row path data is unavailable" in str(exc_info.value)


def test_api_execution_records_failed_source_read_when_response_status_is_invalid():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(kind=SourceKind.API_READ, read_id="records"),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(_project_operation(),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )
    data_access = _ReplayDataAccess(
        responses={
            "list_records": {
                "responseStatus": "ok",
                "responseBody": {"data": [{"name": "Downtown"}]},
            }
        }
    )
    recorder = _SourceReadRecorder()

    with pytest.raises(RelationEngineError):
        compile_and_invoke(
            plan=plan,
            question_contract=_question_contract(),
            catalog=_catalog(),
            data_access_port=data_access,
            memory=LookupMemory(),
            source_read_lineage=SourceReadLineageScope(
                run_id="run_1",
                step_id="step_execute",
                recorder=recorder,
            ),
        )

    assert len(recorder.source_reads) == 1
    source_read = recorder.source_reads[0]
    assert source_read.status == SourceReadStatus.FAILED
    assert source_read.response_hash == _canonical_json_hash(
        {"data": [{"name": "Downtown"}]}
    )
    assert source_read.error_json["responseStatus"] == "ok"
    assert "integer HTTP status" in source_read.error_json["error"]


def test_api_execution_rows_contain_only_declared_relation_fields():
    result = _execute_location_id_plan_with_observed_label()

    assert result.relations[0].rows == ({"location_id": "loc_1"},)


def test_api_execution_result_keeps_observed_source_rows_as_row_context():
    result = _execute_location_id_plan_with_observed_label()

    assert result.row_context.rows_for_relation("rows") == (
        {"location_id": "loc_1", "location_name": "Midtown"},
    )


def _execute_location_id_plan_with_observed_label():
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_location",
                    answer_output_id="location",
                    result_output_id="location",
                ),
            ),
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(kind=SourceKind.API_READ, read_id="records"),
                    fields=(
                        RelationField(
                            field_id="location_id",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="location_id"),),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="location",
                        relation_id="result",
                        field_id="location_id",
                    ),
                )
            ),
        )
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.location_id",
                        path="data.location_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="field.location_name",
                        path="data.location_name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="location",
                        components=(
                            CandidateKeyComponent(
                                id="location_id",
                                field_ref="field.location_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
            ),
        )
    )
    data_access = _ReplayDataAccess(
        responses={
            "list_records": {
                "responseStatus": 200,
                "responseBody": {
                    "data": [
                        {
                            "location_id": "loc_1",
                            "location_name": "Midtown",
                        },
                    ]
                },
            }
        }
    )

    return compile_and_invoke(
        plan=plan,
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="rf_location",
                    description="location",
                    answer_outputs=(
                        RequestedFactAnswerOutput(id="location", role="ANSWER_VALUE"),
                    ),
                ),
            )
        ),
        catalog=catalog,
        data_access_port=data_access,
        memory=LookupMemory(),
    )


def test_filter_operation_constrains_rows_before_aggregate():
    area_expression = _constant_expression(
        FactValue.identity(
            id="nairobi_area",
            key=entity_key_value("area", "primary_key", {"area_id": "area_nairobi"}),
            display_value="Nairobi",
            proof_refs=("known_input:input_1",),
        ),
        ref_id="nairobi_area",
    )
    area_expression = replace(area_expression, component="key_component:area_id")
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="source_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="locations",
                    ),
                    fields=(
                        RelationField(
                            field_id="location_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="area_id",
                            roles=(FieldBindingRole.PREDICATE,),
                        ),
                        RelationField(
                            field_id="metric_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="filter_area",
                    spec=FilterSpec(
                        input_relation="source_rows",
                        predicate=Predicate(
                            left=FieldRef("area_id"),
                            operator=PredicateOperator.EQUALS,
                            right=area_expression,
                        ),
                    ),
                    output_relation="rows",
                ),
                Operation(
                    id="sum_metric",
                    spec=AggregateSpec(
                        input_relation="rows",
                        group_by=(),
                        aggregations=(
                            AggregationSpec(
                                function=AggregationFunction.SUM,
                                input_field="metric_total",
                                output_field="total",
                            ),
                        ),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="total",
                    ),
                )
            ),
        )
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="locations",
                endpoint_name="list_locations",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.location_id",
                        path="data.location_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="field.area_id",
                        path="data.area_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="field.metric_total",
                        path="data.metric_total",
                        row_path_id="data",
                        type="number",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="location",
                        components=(
                            CandidateKeyComponent(
                                id="location_id",
                                field_ref="field.location_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
            ),
        )
    )
    data_access = _ReplayDataAccess(
        responses={
            "list_locations": {
                "responseStatus": 200,
                "responseBody": {
                    "data": [
                        {
                            "location_id": "loc_1",
                            "area_id": "area_nairobi",
                            "metric_total": "10",
                        },
                        {
                            "location_id": "loc_2",
                            "area_id": "area_nairobi",
                            "metric_total": "20",
                        },
                        {
                            "location_id": "loc_3",
                            "area_id": "area_mombasa",
                            "metric_total": "999",
                        },
                    ]
                },
            }
        }
    )

    result = compile_and_invoke(
        plan=plan,
        question_contract=_question_contract("metric total"),
        catalog=catalog,
        data_access_port=data_access,
        memory=LookupMemory(),
    )

    filtered_rows = next(item for item in result.relations if item.id == "rows")
    assert filtered_rows.rows == (
        {
            "location_id": "loc_1",
            "area_id": "area_nairobi",
            "metric_total": Decimal("10"),
        },
        {
            "location_id": "loc_2",
            "area_id": "area_nairobi",
            "metric_total": Decimal("20"),
        },
    )


def test_fulfillment_rejects_rendered_scalar_without_evidence_proof():
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            relations=(_rows_relation(),),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="name"),),
                    ),
                    output_relation="result",
                ),
                Operation(
                    id="compute",
                    spec=ComputeSpec(
                        expression=_constant_expression(
                            FactValue.literal(
                                id="current_value",
                                literal_type=LiteralType.NUMBER,
                                value="100",
                            ),
                            ref_id="current_value",
                        ),
                        output_scalar="current_total",
                    ),
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="name",
                        relation_id="result",
                        field_id="name",
                    ),
                ),
                scalar_outputs=(
                    ScalarResultOutput(
                        id="answer",
                        scalar_id="current_total",
                    ),
                ),
            ),
        )
    )

    with pytest.raises(VerificationError, match="requires evidence proof"):
        verify_fact_plan(
            plan,
            question_contract=_question_contract(
                "answer", binding_target_ids=("name", "answer")
            ),
        )


def test_filter_targets_are_verified_against_catalog_and_relations():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                _row_filter_operation(
                    ParameterRef("question.known"),
                    field_id="field.missing",
                ),
                _project_operation(input_relation="filtered_rows"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="unknown field"):
        verify_fact_plan(
            plan,
            catalog=_catalog(),
            question_contract=_question_contract(
                known_inputs=(_known_reference("known", "Known"),)
            ),
        )


def test_fulfillment_rejects_known_input_proof_from_unrelated_join_branch():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="answer_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                    ),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
                Relation(
                    id="scoped_source_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                    ),
                    fields=(
                        RelationField(
                            field_id="entity_id",
                            roles=(FieldBindingRole.PREDICATE,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="filter_scoped_rows",
                    spec=FilterSpec(
                        input_relation="scoped_source_rows",
                        predicate=Predicate(
                            left=FieldRef("entity_id"),
                            operator=PredicateOperator.EQUALS,
                            right=_constant_expression(
                                FactValue.named(
                                    id="filter_entity_1",
                                    text="entity_1",
                                    proof_refs=("known_input:input_1",),
                                ),
                                ref_id="filter_entity_1",
                            ),
                        ),
                    ),
                    output_relation="scoped_rows",
                ),
                Operation(
                    id="join_unrelated_scope",
                    spec=CrossJoinSpec(
                        left="answer_rows",
                        right="scoped_rows",
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(
        VerificationError, match="fulfillment result output missing input proof"
    ):
        verify_fact_plan(
            plan,
            question_contract=QuestionContract(
                requested_facts=(
                    RequestedFact(
                        id="rf_answer",
                        description="records in area",
                        input_refs=("input_1",),
                        known_inputs=(_known_reference("input_1", "London"),),
                        answer_outputs=(
                            RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                        ),
                    ),
                )
            ),
        )


def test_count_fulfillment_does_not_inherit_unrelated_field_proof():
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(ref="field.name", path="data.name", type="string"),
                    CatalogField(
                        ref="field.row_id",
                        path="data.row_id",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="row",
                        components=(
                            CandidateKeyComponent(
                                id="row_id",
                                field_ref="field.row_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
                facts=(
                    CatalogFact(
                        ref="known_input:input_1",
                        field_ref="field.name",
                        read_id="records",
                    ),
                ),
            ),
        )
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.PREDICATE,),
                        ),
                        RelationField(
                            field_id="row_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="count_rows",
                    spec=AggregateSpec(
                        input_relation="rows",
                        group_by=(),
                        aggregations=(
                            AggregationSpec(
                                function=AggregationFunction.COUNT,
                                output_field="total",
                            ),
                        ),
                    ),
                    output_relation="totals",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="totals",
                        field_id="total",
                    ),
                )
            ),
        )
    )

    with pytest.raises(
        VerificationError, match="fulfillment result output missing input proof"
    ):
        verify_fact_plan(
            plan,
            catalog=catalog,
            question_contract=QuestionContract(
                requested_facts=(
                    RequestedFact(
                        id="rf_answer",
                        description="count rows for named input",
                        input_refs=("input_1",),
                        known_inputs=(_known_reference("input_1", "Name"),),
                        answer_outputs=(
                            RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                        ),
                    ),
                )
            ),
        )


def test_role_expand_generated_role_field_carries_evidence_proof():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="amount",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="expand_roles",
                    spec=RoleExpandSpec(
                        input_relation="rows",
                        mappings=(
                            RoleMapping(
                                role="current",
                                source_field="amount",
                                output_field="value",
                            ),
                        ),
                        output_fields=("role",),
                    ),
                    output_relation="expanded",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="expanded",
                        field_id="role",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(plan, catalog=_catalog())


def test_api_identity_binding_can_use_catalog_display_field_as_relation_grain():
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="staff_records",
                endpoint_name="list_staff",
                row_paths=(
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.staff_id",
                        path="data.staff_id",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.staff_name",
                        path="data.staff_name",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="staff",
                        components=(
                            CandidateKeyComponent(
                                id="staff_id",
                                field_ref="field.staff_id",
                            ),
                        ),
                        primary=True,
                        context_field_refs=("field.staff_name",),
                    ),
                ),
            ),
        )
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="staff_candidates",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="staff_records",
                    ),
                    fields=(
                        RelationField(
                            field_id="staff_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="staff_name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="staff_candidates",
                        fields=(ProjectField(source="staff_name"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="staff",
                        relation_id="answer_rows",
                        field_id="staff_name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(plan, catalog=catalog)


def test_api_identity_binding_declares_relation_grain_without_entity_metadata():
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                row_paths=(
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.record_id",
                        path="data.record_id",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.status",
                        path="data.status",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.name",
                        path="data.name",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="record",
                        components=(
                            CandidateKeyComponent(
                                id="record_id",
                                field_ref="field.record_id",
                            ),
                        ),
                        primary=True,
                        context_field_refs=("field.name",),
                    ),
                ),
            ),
        )
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="records",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                    ),
                    fields=(
                        RelationField(
                            field_id="record_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="records",
                        fields=(ProjectField(source="name"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="record",
                        relation_id="answer_rows",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(plan, catalog=catalog)


def test_api_identity_binding_requires_primary_stable_row_key():
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="records",
                endpoint_name="list_records",
                row_paths=(
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.reference_code",
                        path="data.reference_code",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.name",
                        path="data.name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
            ),
        )
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="records",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                    ),
                    fields=(
                        RelationField(
                            field_id="reference_code",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="records",
                        fields=(ProjectField(source="name"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="record",
                        relation_id="answer_rows",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="field role is not allowed"):
        verify_fact_plan(plan, catalog=catalog)


def test_relation_field_binding_ids_must_be_unique():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(_project_operation(input_relation="rows"),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="duplicate field"):
        verify_fact_plan(plan)


def test_api_relation_child_rows_can_include_parent_identity_fields():
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="orders",
                endpoint_name="list_orders",
                row_paths=(
                    RowPath(
                        id="orders",
                        path="orders",
                        cardinality=RowCardinality.MANY,
                    ),
                    RowPath(
                        id="items",
                        path="orders.items",
                        cardinality=RowCardinality.MANY,
                        parent_path="orders",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.orders.order_id",
                        path="orders.order_id",
                        row_path_id="orders",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.orders.items.sku",
                        path="orders.items.sku",
                        row_path_id="items",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="order",
                        components=(
                            CandidateKeyComponent(
                                id="order_id",
                                field_ref="field.orders.order_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
            ),
        )
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="item_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="orders",
                    ),
                    fields=(
                        RelationField(
                            field_id="order_id",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                        RelationField(
                            field_id="sku",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="item_rows",
                        fields=(ProjectField(source="sku"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="sku",
                        relation_id="answer_rows",
                        field_id="sku",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(plan, catalog=catalog)


def test_api_relation_field_requirements_are_satisfied_by_row_source_default():
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sales",
                params=(
                    CatalogParam(
                        ref="list_sales.query.include_items",
                        name="include_items",
                        source=ParamSource.QUERY,
                        type="boolean",
                    ),
                ),
                row_paths=(
                    RowPath(
                        id="items",
                        path="data.items",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.items.name",
                        path="data.items.name",
                        row_path_id="items",
                        type="string",
                        requirements=(
                            FieldRequirement(
                                param_ref="list_sales.query.include_items",
                                value=True,
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    relation = Relation(
        id="item_rows",
        source=RelationSource(
            kind=SourceKind.API_READ,
            read_id="sales",
        ),
        fields=(
            RelationField(
                field_id="name",
                roles=(FieldBindingRole.OUTPUT,),
            ),
        ),
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(relation,),
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="item_rows",
                        fields=(ProjectField(source="name", output="item_name"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="item_name",
                        relation_id="answer_rows",
                        field_id="item_name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(
        plan,
        catalog=catalog,
        question_contract=_question_contract(
            binding_target_ids=("item_name",),
        ),
    )


def test_compute_node_output_requires_matching_prior_operation():
    plan = FactPlan(
        outcome=_answer_plan(
            operations=(
                Operation(
                    id="compute",
                    spec=ComputeSpec(
                        expression=NodeOutputRef(
                            node_id="missing",
                            output_id="target",
                        ),
                        output_scalar="result",
                    ),
                ),
            ),
            result_projection=ResultProjection(relation_outputs=()),
        )
    )

    with pytest.raises(VerificationError, match="unbound scalar input"):
        verify_fact_plan(plan, catalog=_catalog())


def test_predicate_scalar_rhs_requires_prior_scalar_output():
    relation = Relation(
        id="rows",
        source=_source(),
        fields=(
            RelationField(field_id="name", roles=(FieldBindingRole.OUTPUT,)),
            RelationField(field_id="amount", roles=(FieldBindingRole.PREDICATE,)),
        ),
    )
    operation = Operation(
        id="filter",
        spec=FilterSpec(
            input_relation="rows",
            predicate=Predicate(
                left=FieldRef("amount"),
                operator=PredicateOperator.LTE,
                right=NodeOutputRef(node_id="missing", output_id="max_amount"),
            ),
        ),
        output_relation="filtered",
    )
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(relation,),
            operations=(operation,),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="name",
                        relation_id="filtered",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="unbound scalar input"):
        verify_fact_plan(plan)


def test_operation_input_references_existing_relation_or_prior_operation():
    plan = FactPlan(
        outcome=_answer_plan(
            operations=(
                Operation(
                    id="project",
                    spec=ProjectSpec(
                        input_relation="missing_rows",
                        fields=(ProjectField(source="name", output="name"),),
                    ),
                    output_relation="result",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="unknown input"):
        verify_fact_plan(plan)


def test_operation_ids_are_unique():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                _project_operation(input_relation="rows"),
                _project_operation(input_relation="rows"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="duplicate operation"):
        verify_fact_plan(plan)


def test_relation_and_output_relation_ids_are_unique():
    duplicate_relations = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(id="rows", source=_source()),
                Relation(id="rows", source=_source()),
            ),
            operations=(_project_operation(input_relation="rows"),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="duplicate relation"):
        verify_fact_plan(duplicate_relations)

    duplicate_output = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                _project_operation(input_relation="rows", output_relation="rows"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="rows", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="duplicate relation"):
        verify_fact_plan(duplicate_output)


def test_operation_field_references_must_exist_on_input_relation_contracts():
    cases = (
        Operation(
            id="filter",
            spec=FilterSpec(
                input_relation="rows",
                predicate=Predicate(
                    left=FieldRef("missing"),
                    operator=PredicateOperator.EQUALS,
                    right=FieldRef("name"),
                ),
            ),
            output_relation="filtered",
        ),
        Operation(
            id="join",
            spec=JoinSpec(
                left="rows",
                right="other_rows",
                join_keys=(JoinKey(left="name", right="missing"),),
            ),
            output_relation="joined",
        ),
        Operation(
            id="rank",
            spec=OrderSpec(
                input_relation="rows",
                order_by=(SortKey(field="missing", direction=SortDirection.ASC),),
                tie_breakers=(SortKey(field="name", direction=SortDirection.ASC),),
                selection=Take(limit=_rank_limit_constant(5)),
            ),
            output_relation="ranked",
        ),
        Operation(
            id="aggregate",
            spec=AggregateSpec(
                input_relation="rows",
                group_by=("name",),
                aggregations=(
                    AggregationSpec(
                        function=AggregationFunction.SUM,
                        input_field="missing",
                        output_field="total",
                    ),
                ),
            ),
            output_relation="totals",
        ),
    )

    for operation in cases:
        final_relation = operation.output_relation
        final_field = "total" if operation.id == "aggregate" else "name"
        question_contract = (
            _question_contract(
                known_inputs=(_known_result_limit("result_limit", "top 5", 5),)
            )
            if operation.id == "rank"
            else _question_contract()
        )
        plan = FactPlan(
            outcome=_answer_plan(
                relations=(
                    _rows_relation(),
                    Relation(
                        id="other_rows",
                        source=_source(),
                        fields=(
                            RelationField(
                                field_id="other_name",
                                roles=(FieldBindingRole.OUTPUT,),
                            ),
                        ),
                    ),
                ),
                operations=(operation,),
                result_projection=ResultProjection(
                    relation_outputs=(
                        RelationResultOutput(
                            id="answer",
                            relation_id=final_relation,
                            field_id=final_field,
                        ),
                    )
                ),
            )
        )

        with pytest.raises(VerificationError, match="references unknown field"):
            verify_fact_plan(plan, question_contract=question_contract)


def test_compute_scalar_outputs_are_unique():
    expression = _constant_expression(
        FactValue.literal(
            id="target",
            literal_type=LiteralType.NUMBER,
            value="5",
        ),
        ref_id="target",
    )
    duplicate_scalars = FactPlan(
        outcome=_answer_plan(
            operations=(
                Operation(
                    id="compute_a",
                    spec=ComputeSpec(
                        expression=expression,
                        output_scalar="result",
                    ),
                ),
                Operation(
                    id="compute_b",
                    spec=ComputeSpec(
                        expression=expression,
                        output_scalar="result",
                    ),
                ),
            ),
            result_projection=ResultProjection(relation_outputs=()),
        )
    )

    with pytest.raises(VerificationError, match="duplicate scalar"):
        verify_fact_plan(duplicate_scalars)


def test_compute_scalar_outputs_do_not_shadow_aggregate_fields():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="aggregate",
                    spec=AggregateSpec(
                        input_relation="rows",
                        group_by=(),
                        aggregations=(
                            AggregationSpec(
                                function=AggregationFunction.COUNT,
                                output_field="total",
                            ),
                        ),
                    ),
                    output_relation="totals",
                ),
                Operation(
                    id="compute",
                    spec=ComputeSpec(
                        expression=_constant_expression(
                            FactValue.literal(
                                id="target",
                                literal_type=LiteralType.NUMBER,
                                value="5",
                            ),
                            ref_id="target",
                        ),
                        output_scalar="total",
                    ),
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="totals", field_id="total"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="duplicate scalar"):
        verify_fact_plan(plan)


def test_result_output_references_existing_operation_output():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(_project_operation(output_relation="result"),),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="missing", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="unknown operation output"):
        verify_fact_plan(plan)


def test_result_output_cannot_bypass_operation_output():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                _project_operation(input_relation="rows", output_relation="result"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="rows", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="unknown operation output"):
        verify_fact_plan(plan)


def test_result_output_requires_known_output_field():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                _project_operation(input_relation="rows", output_relation="result"),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="result",
                        field_id="missing",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="unknown output field"):
        verify_fact_plan(plan)


def test_result_outputs_may_use_multiple_final_relations():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(_rows_relation(),),
            operations=(
                _project_operation(input_relation="rows", output_relation="result_a"),
                Operation(
                    id="project_b",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="name", output="name"),),
                    ),
                    output_relation="result_b",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer_a",
                        relation_id="result_a",
                        field_id="name",
                    ),
                    RelationResultOutput(
                        id="answer_b",
                        relation_id="result_b",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    verify_fact_plan(plan)


def test_render_relation_must_be_terminal_operation_output():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                _project_operation(
                    input_relation="rows", output_relation="intermediate"
                ),
                Operation(
                    id="final_project",
                    spec=ProjectSpec(
                        input_relation="intermediate",
                        fields=(ProjectField(source="name"),),
                    ),
                    output_relation="final_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="intermediate",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="terminal final relation"):
        verify_fact_plan(plan)


def test_entity_result_output_requires_key_metadata():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="candidate_rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="entity_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="display",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
                Relation(
                    id="observed_rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="observed_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="missing_entities",
                    spec=AntiJoinSpec(
                        candidate=RelationRoleRef(
                            relation_id="candidate_rows",
                            role=RelationRole.ANTI_JOIN_CANDIDATE,
                            required_identity_fields=("entity_id",),
                        ),
                        observed=RelationRoleRef(
                            relation_id="observed_rows",
                            role=RelationRole.ANTI_JOIN_OBSERVED,
                            required_identity_fields=("observed_id",),
                        ),
                        join_keys=(JoinKey(left="entity_id", right="observed_id"),),
                        output_fields=(ProjectField(source="display", output="name"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="answer_rows",
                        field_id="entity_id",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="entity key metadata"):
        verify_fact_plan(plan)


def test_derived_candidate_output_fields_keep_roles():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="source_rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="row_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="entity_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                    ),
                ),
                Relation(
                    id="observed_rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="observed_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="entity_rows",
                    spec=ProjectToKeySpec(
                        input_relation="source_rows",
                        key_fields=("entity_id",),
                    ),
                    output_relation="entity_rows",
                ),
                Operation(
                    id="missing_entities",
                    spec=AntiJoinSpec(
                        candidate=RelationRoleRef(
                            relation_id="entity_rows",
                            role=RelationRole.ANTI_JOIN_CANDIDATE,
                            required_identity_fields=("entity_id",),
                        ),
                        observed=RelationRoleRef(
                            relation_id="observed_rows",
                            role=RelationRole.ANTI_JOIN_OBSERVED,
                            required_identity_fields=("observed_id",),
                        ),
                        join_keys=(JoinKey(left="entity_id", right="observed_id"),),
                        output_fields=(
                            ProjectField(source="entity_id", output="name"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="answer_rows",
                        field_id="name",
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="field has wrong binding role"):
        verify_fact_plan(plan)


def test_project_cannot_partially_change_relation_grain_for_coverage_role():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="event_rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="entity_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="event_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="display",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
                Relation(
                    id="observed_rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="observed_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_entity",
                    spec=ProjectSpec(
                        input_relation="event_rows",
                        fields=(
                            ProjectField(source="entity_id"),
                            ProjectField(source="display"),
                        ),
                    ),
                    output_relation="entity_rows",
                ),
                Operation(
                    id="missing_entities",
                    spec=AntiJoinSpec(
                        candidate=RelationRoleRef(
                            relation_id="entity_rows",
                            role=RelationRole.ANTI_JOIN_CANDIDATE,
                            required_identity_fields=("entity_id",),
                        ),
                        observed=RelationRoleRef(
                            relation_id="observed_rows",
                            role=RelationRole.ANTI_JOIN_OBSERVED,
                            required_identity_fields=("observed_id",),
                        ),
                        join_keys=(JoinKey(left="entity_id", right="observed_id"),),
                        output_fields=(ProjectField(source="display"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(relation_outputs=()),
        )
    )

    with pytest.raises(VerificationError, match="exact relation grain"):
        verify_fact_plan(plan)


@pytest.mark.parametrize(
    ("entity_kind", "key_id", "component_id"),
    (
        ("other_entity", "primary_key", "entity_id"),
        ("entity", "undeclared_key", "entity_id"),
        ("entity", "primary_key", "other_component"),
    ),
)
def test_project_to_key_rejects_undeclared_entity_key(
    entity_kind: str,
    key_id: str,
    component_id: str,
) -> None:
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="source_rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="entity_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="entity_label",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="entity_rows",
                    spec=ProjectToKeySpec(
                        input_relation="source_rows",
                        key_fields=("entity_id",),
                    ),
                    output_relation="entity_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="entity_rows",
                        entity_key=EntityKeyProjection(
                            entity_kind=entity_kind,
                            key_id=key_id,
                            components=(
                                EntityKeyProjectionComponent(
                                    component_id=component_id,
                                    field_id="entity_id",
                                ),
                            ),
                        ),
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="declared entity key"):
        verify_fact_plan(plan)


def test_aggregate_outputs_only_group_key_and_measure():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="source_rows",
                    source=_source(),
                    fields=(
                        RelationField(
                            field_id="entity_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="entity_label",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                        RelationField(
                            field_id="amount",
                            roles=(FieldBindingRole.PREDICATE,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="totals",
                    spec=AggregateSpec(
                        input_relation="source_rows",
                        group_by=("entity_id",),
                        aggregations=(
                            AggregationSpec(
                                function=AggregationFunction.SUM,
                                input_field="amount",
                                output_field="total",
                            ),
                        ),
                    ),
                    output_relation="totals",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer",
                        relation_id="totals",
                        entity_key=EntityKeyProjection(
                            entity_kind="entity",
                            key_id="primary_key",
                            components=(
                                EntityKeyProjectionComponent(
                                    component_id="entity_id",
                                    field_id="entity_id",
                                ),
                            ),
                        ),
                    ),
                )
            ),
        )
    )

    verify_fact_plan(plan)
