from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.compilation import compile_answer_program
from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment
from fervis.lookup.answer_program.operations import Operation
from fervis.lookup.answer_program.relations import (
    EndpointParamBinding,
    FieldBindingRole,
    PopulationCoverageClaim,
    PopulationCoverageRole,
    Relation,
    RelationField,
    RelationSource,
    RelationSourceAppliedFilter,
    SourceKind,
)
from fervis.lookup.question_contract import MembershipTestRef
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ResultProjection,
)
from fervis.lookup.answer_program.values import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    FactValue,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRef,
    ParameterRole,
    ParameterValueType,
)
from fervis.lookup.canonical_data import entity_key_value
from fervis.lookup.answer_program import AnswerProgramContractError
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.instantiation import (
    ExecutionEnvironment,
    instantiate_answer_program,
)

from tests.testkit.algorithms.relation_engine import operation_spec_from_payload
from tests.testkit.assertions import status_mismatches
from tests.testkit.catalog import catalog_from_payload
from tests.testkit.question_contract import question_contract_from_payload


def run_population_constraint_coverage_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    try:
        catalog = catalog_from_payload(_catalog_payload(input_payload))
        question_contract = question_contract_from_payload(
            input_payload["question_contract"]
        )
        program, bindings = compile_answer_program(
            _program(input_payload),
            question_contract=question_contract,
            catalog=catalog,
            bindings=_bindings(input_payload),
        )
        execution = instantiate_answer_program(
            program,
            bindings,
            ExecutionEnvironment(catalog=catalog),
        )
        del execution
    except (AnswerProgramContractError, VerificationError):
        actual_status = "rejected"
    else:
        actual_status = "accepted"
    return status_mismatches(
        actual_status=actual_status,
        expected=payload["expect"],
    )


def _program(payload: dict[str, Any]) -> AnswerProgram:
    input_ids = tuple(str(item["id"]) for item in payload.get("inputs") or ())
    result = payload["result"]
    return AnswerProgram(
        fulfillment=(
            FactFulfillment(
                requested_fact_id=str(result.get("requested_fact_id") or "fact_1"),
                answer_output_id=str(result.get("answer_output_id") or "answer_1"),
                result_output_id=str(result.get("answer_output_id") or "answer_1"),
            ),
        ),
        parameters=tuple(
            ParameterDeclaration(
                id=_parameter_id(input_id),
                role=ParameterRole.QUESTION_INPUT,
                value_type=ParameterValueType.IDENTITY,
            )
            for input_id in input_ids
        ),
        relations=tuple(_relation(item) for item in payload["relations"]),
        operations=tuple(
            Operation(
                id=str(item["id"]),
                spec=operation_spec_from_payload(item["spec"]),
                output_relation=str(item["output_relation"]),
            )
            for item in payload.get("operations") or ()
        ),
        result_projection=ResultProjection(
            relation_outputs=(
                RelationResultOutput(
                    id=str(result.get("answer_output_id") or "answer_1"),
                    relation_id=str(result["relation_id"]),
                    field_id=str(result["field_id"]),
                    role="answer_value",
                ),
            )
        ),
    )


def _relation(payload: dict[str, Any]) -> Relation:
    identity_fields = {str(item) for item in payload.get("identity_fields") or ()}
    predicate_fields = {
        str(item) for item in payload.get("predicate_fields") or ()
    }
    return Relation(
        id=str(payload["id"]),
        source=RelationSource(
            kind=SourceKind.API_READ,
            read_id=str(payload["id"]),
            param_bindings=tuple(
                EndpointParamBinding(
                    param_id=str(item["param_id"]),
                    value_expr=ParameterRef(
                        parameter_id=_parameter_id(str(item["input_id"]))
                    ),
                    proof_refs=(f"known_input:{item['input_id']}",),
                )
                for item in payload.get("param_inputs") or ()
            ),
            applied_filters=tuple(
                RelationSourceAppliedFilter(
                    predicate_field_ids=(str(item["field_id"]),),
                    value_expr=ParameterRef(
                        parameter_id=_parameter_id(str(item["input_id"]))
                    ),
                    proof_refs=(
                        f"returned_filter:{payload['id']}.{item['field_id']}",
                    ),
                )
                for item in payload.get("applied_inputs") or ()
            ),
            population_coverage_claims=tuple(
                PopulationCoverageClaim(
                    test_ref=MembershipTestRef(
                        requested_fact_id=str(
                            item.get("requested_fact_id") or "fact_1"
                        ),
                        membership_test_id=str(item["membership_test_id"]),
                    ),
                    role=PopulationCoverageRole(str(item["role"])),
                    proof_refs=(str(item["proof_ref"]),),
                )
                for item in payload.get("coverage_claims") or ()
            ),
            proof_refs=tuple(str(ref) for ref in payload.get("source_proof_refs") or ()),
        ),
        fields=tuple(
            RelationField(
                field_id=str(field_id),
                roles=(
                    (FieldBindingRole.IDENTITY, FieldBindingRole.OUTPUT)
                    if str(field_id) in identity_fields
                    else (
                        (FieldBindingRole.PREDICATE, FieldBindingRole.OUTPUT)
                        if str(field_id) in predicate_fields
                        else (FieldBindingRole.OUTPUT,)
                    )
                ),
            )
            for field_id in payload["fields"]
        ),
    )


def _bindings(payload: dict[str, Any]) -> BindingSet:
    return BindingSet.from_bindings(
        tuple(
            ParameterBinding(
                parameter_id=_parameter_id(str(item["id"])),
                value=FactValue.identity(
                    id=f"value_{item['id']}",
                    key=entity_key_value(
                        str(item["entity_kind"]),
                        str(item.get("key_id") or "primary_key"),
                        {
                            str(item["key_component_id"]): str(
                                item["canonical_value"]
                            )
                        },
                    ),
                    display_value=str(item["value"]),
                    known_input_id=str(item["id"]),
                    proof_refs=(f"known_input:{item['id']}",),
                    applies_to_requested_fact_ids=(
                        str(item.get("requested_fact_id") or "fact_1"),
                    ),
                ),
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.QUESTION_INPUT,
                    refs=(f"known_input:{item['id']}",),
                ),
            )
            for item in payload.get("inputs") or ()
        )
    )


def _catalog_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "reads": [
            {
                "id": str(relation["id"]),
                "endpoint_name": str(relation["id"]),
                "resource_names": [str(relation["id"])],
                "row_paths": [
                    {"id": "data", "path": "data", "cardinality": "many"}
                ],
                "fields": [
                    {
                        "ref": f"field.{relation['id']}.{field_id}",
                        "path": f"data.{field_id}",
                        "row_path_id": "data",
                        "type": "string",
                    }
                    for field_id in relation["fields"]
                ],
                "params": [
                    {
                        "ref": f"{relation['id']}.query.{item['param_id']}",
                        "name": str(item["param_id"]),
                        "source": "query",
                        "type": "string",
                    }
                    for item in relation.get("param_inputs") or ()
                ],
                "candidate_keys": (
                    [
                        {
                            "id": "primary_key",
                            "entity_kind": str(relation["id"]),
                            "primary": True,
                            "components": [
                                {
                                    "id": str(field_id),
                                    "field_ref": (
                                        f"field.{relation['id']}.{field_id}"
                                    ),
                                }
                                for field_id in relation.get("identity_fields") or ()
                            ],
                        }
                    ]
                    if relation.get("identity_fields")
                    else []
                ),
                "entity_references": [
                    {
                        "id": str(reference["id"]),
                        "target_entity_kind": str(reference["target_entity_kind"]),
                        "target_key_id": str(
                            reference.get("target_key_id") or "primary_key"
                        ),
                        "components": [
                            {
                                "target_component_id": str(
                                    component["target_component_id"]
                                ),
                                "local_field_ref": (
                                    f"field.{relation['id']}."
                                    f"{component['local_field_id']}"
                                ),
                            }
                            for component in reference["components"]
                        ],
                    }
                    for reference in relation.get("entity_references") or ()
                ],
            }
            for relation in payload["relations"]
        ]
    }


def _parameter_id(input_id: str) -> str:
    return f"question.{input_id}"
