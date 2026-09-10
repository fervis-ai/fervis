"""Edges cannot silently supply the meanings of their endpoint populations."""

import pytest
from jsonschema import validate
from fervis.lookup.source_binding.parser import compile_source_realization
from fervis.lookup.source_binding.schema import build_semantic_source_realization_schema
from tests.lookup.relational_engine.test_scoped_compilation import employee_query


def example():
    verified = employee_query()
    request, plan = verified.request, verified.binding_plan
    payload = {
        "set_bindings": {
            ref: [
                {
                    "branch_id": item.branch_id,
                    "mapping_basis": f"Independent role: {ref}",
                    "rows_ref": item.identity_ref or item.source_ref,
                }
                for item in values
            ]
            for ref, values in plan.set_bindings.items()
        },
        "fact_bindings": {
            ref: [
                {
                    "branch_id": "branch",
                    "mapping_basis": "Observed role salary.",
                    "field_ref": "source_field:employees:salary",
                }
            ]
            for ref in request.model_authored_fact_refs
        },
        "association_bindings": {
            ref: [
                {
                    "branch_id": item.branch_id,
                    "mapping_basis": "Declared reporting link.",
                    "realization_ref": item.relation_evidence_ref
                    or item.source_refs[0],
                    "reference_from_set_ref": item.reference_from_set_ref,
                }
                for item in values
            ]
            for ref, values in plan.association_bindings.items()
        },
    }
    return request, payload


def test_all_nodes_have_independent_realizations_before_edges():
    request, payload = example()
    validate(payload, build_semantic_source_realization_schema(request))
    result = compile_source_realization(payload, request=request)
    for ref, values in result.set_bindings.items():
        assert values[0].mapping_basis == f"Independent role: {ref}"


def test_an_edge_cannot_supply_a_missing_node_assignment():
    request, payload = example()
    payload["set_bindings"].pop(next(iter(payload["set_bindings"])))
    with pytest.raises(ValueError, match="set binding"):
        compile_source_realization(payload, request=request)


def test_relationship_evidence_cannot_reassign_an_independent_endpoint():
    request, payload = example()
    edge = next(iter(payload["association_bindings"].values()))[0]
    edge["realization_ref"] = request.source_catalog.sources[0].id
    edge["reference_from_set_ref"] = None
    with pytest.raises(ValueError, match="incompatible assigned endpoint rows"):
        compile_source_realization(payload, request=request)


def test_association_payload_cannot_override_an_endpoint_assignment():
    request, payload = example()
    edge = next(iter(payload["association_bindings"].values()))[0]
    edge["to_rows_ref"] = "some_other_population"
    with pytest.raises(ValueError):
        compile_source_realization(payload, request=request)
