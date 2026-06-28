from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import (
    CatalogField,
    CatalogParam,
    EndpointRead,
    IdentityMetadata,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.plan_execution.runner import execute_fact_plan
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.fact_plan.fact_plan import (
    AnswerPlan,
    FactFulfillment,
    FactPlan,
)
from fervis.lookup.fact_plan.operations import (
    Operation,
    ProjectField,
    ProjectSpec,
)
from fervis.lookup.fact_plan.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_plan.render_spec import (
    RenderRelationOutput,
    RenderSpec,
)
from fervis.lookup.fact_plan.relations import EndpointParamBinding

from tests.testkit.assertions import subset_mismatches
from tests.testkit.question_contract import question_contract_from_payload
from tests.testkit.serialization import portable_value


def run_identity_set_binding_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    data_access = _DataAccess(responses=tuple(input_payload["responses"]))
    result = execute_fact_plan(
        plan=_plan(param_value=tuple(input_payload["identity_values"])),
        question_contract=question_contract_from_payload(
            {
                "requested_facts": [
                    {
                        "id": "fact_1",
                        "description": "sales for prior stores",
                        "answer_outputs": [{"id": "answer_1"}],
                    }
                ]
            }
        ),
        catalog=_catalog(param_type=str(input_payload.get("param_type") or "uuid")),
        data_access_port=data_access,
        memory=LookupMemory(),
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
                        identity=IdentityMetadata(
                            entity_ref="store",
                            identity_field="store_id",
                            primary_key=True,
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
                        identity=IdentityMetadata(
                            entity_ref="sale",
                            identity_field="sale_id",
                            primary_key=True,
                        ),
                    ),
                    CatalogField(
                        ref="sales.field.store_id",
                        path="results.store_id",
                        type="uuid",
                        identity=IdentityMetadata(
                            entity_ref="store",
                            identity_field="store_id",
                            primary_key=True,
                        ),
                    ),
                ),
            ),
        )
    )


def _plan(*, param_value: object) -> FactPlan:
    return FactPlan(
        outcome=AnswerPlan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    render_output_id="answer_1",
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
                                value=param_value,
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
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="answer_1",
                        relation_id="answer_rows",
                        field_id="store_id",
                        role="answer_value",
                    ),
                ),
            ),
        )
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
