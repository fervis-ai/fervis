from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import (
    CatalogField,
    CatalogParam,
    CandidateKey,
    CandidateKeyComponent,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
    EndpointRead,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.answer_program import (
    AnswerProgram,
)
from fervis.lookup.answer_program.compilation import compile_answer_program
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.answer_program.model import FactFulfillment
from fervis.lookup.fact_plan.fact_plan import FactPlan
from fervis.lookup.answer_program.operations import (
    Operation,
    ProjectField,
    ProjectSpec,
)
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ResultProjection,
)
from fervis.lookup.answer_program.relations import EndpointParamBinding
from fervis.lookup.answer_program import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRef,
    ParameterRole,
    ParameterValueType,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.canonical_data import entity_key_value

from tests.testkit.assertions import subset_mismatches
from tests.testkit.question_contract import question_contract_from_payload
from tests.testkit.serialization import portable_value


def run_identity_set_binding_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    data_access = _DataAccess(responses=tuple(input_payload["responses"]))
    question_contract = question_contract_from_payload(
        {
            "requested_facts": [
                {
                    "id": "fact_1",
                    "description": "sales for prior stores",
                    "answer_outputs": [{"id": "answer_1", "role": "ANSWER_VALUE"}],
                }
            ]
        }
    )
    catalog = _catalog(param_type=str(input_payload.get("param_type") or "uuid"))
    draft = _plan(param_value=tuple(input_payload["identity_values"]))
    if not isinstance(draft.outcome, AnswerProgram):
        raise ValueError("identity-set fixture requires answer program")
    program, bindings = compile_answer_program(
        draft.outcome,
        question_contract=question_contract,
        catalog=catalog,
        bindings=draft.bindings,
    )
    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(
            catalog=catalog,
        ),
        ports=RuntimePorts(
            data_access_port=data_access,
            memory=LookupMemory(),
        ),
    )
    actual = {
        "request_param_values": [
            portable_value(request["args"]["sales.query.store_id"])
            for request in data_access.requests
        ],
        "sale_ids": [row["sale_id"] for row in result.relations[0].rows],
        "rows": list(result.relations[0].rows),
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _catalog(*, param_type: str) -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                params=(
                    CatalogParam(
                        ref="sales.query.store_id",
                        name="store_id",
                        source=ParamSource.QUERY,
                        type=param_type,
                        entity_target=EntityKeyComponentTarget(
                            entity_kind="store",
                            key_id="primary_key",
                            component_id="store_id",
                        ),
                    ),
                ),
                row_paths=(
                    RowPath(
                        id="results",
                        path="results",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sales.field.sale_id",
                        path="results.sale_id",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="sales.field.store_id",
                        path="results.store_id",
                        type="uuid",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="sale",
                        components=(
                            CandidateKeyComponent(
                                id="sale_id",
                                field_ref="sales.field.sale_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
                entity_references=(
                    EntityReference(
                        id="store_reference",
                        target_entity_kind="store",
                        target_key_id="primary_key",
                        components=(
                            EntityReferenceComponent(
                                target_component_id="store_id",
                                local_field_ref="sales.field.store_id",
                            ),
                        ),
                    ),
                ),
            ),
            EndpointRead(
                id="stores",
                endpoint_name="list_store_list",
                fields=(
                    CatalogField(
                        ref="stores.field.store_id",
                        path="results.store_id",
                        type="uuid",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="store",
                        components=(
                            CandidateKeyComponent(
                                id="store_id",
                                field_ref="stores.field.store_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
            ),
        )
    )


def _plan(*, param_value: object) -> FactPlan:
    identity_values = tuple(str(value) for value in param_value)
    parameter_id = "question.store_ids"
    binding_value = FactValue.identity_set(
        id="store_ids",
        keys=tuple(
            entity_key_value("store", "primary_key", {"store_id": value})
            for value in identity_values
        ),
    )
    return FactPlan(
        bindings=BindingSet.from_bindings(
            (
                ParameterBinding(
                    parameter_id=parameter_id,
                    value=binding_value,
                    provenance=BindingProvenance(
                        kind=BindingProvenanceKind.QUESTION_INPUT,
                    ),
                ),
            )
        ),
        outcome=AnswerProgram(
            parameters=(
                ParameterDeclaration(
                    id=parameter_id,
                    role=ParameterRole.QUESTION_INPUT,
                    value_type=ParameterValueType.IDENTITY_SET,
                ),
            ),
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    result_output_id="answer_1",
                ),
            ),
            relations=(
                Relation(
                    id="sales_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="sales",
                        param_bindings=(
                            EndpointParamBinding(
                                param_id="store_id",
                                value_expr=ParameterRef(parameter_id=parameter_id),
                            ),
                        ),
                    ),
                    fields=(
                        RelationField(
                            field_id="sale_id",
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="store_id",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_sales",
                    spec=ProjectSpec(
                        input_relation="sales_rows",
                        fields=(
                            ProjectField(source="sale_id", output="answer_sale_id"),
                            ProjectField(source="store_id"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer_1",
                        relation_id="answer_rows",
                        field_id="store_id",
                        role="answer_value",
                    ),
                ),
            ),
        ),
    )


class _DataAccess:
    def __init__(self, *, responses: tuple[dict[str, Any], ...]) -> None:
        self.responses = {
            _response_key(response["param_value"]): response["body"]
            for response in responses
        }
        self.requests: list[dict[str, Any]] = []

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        value = args["sales.query.store_id"]
        return {
            "endpointName": endpoint_name,
            "responseStatus": 200,
            "responseBody": self.responses[_response_key(value)],
        }


def _response_key(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)
