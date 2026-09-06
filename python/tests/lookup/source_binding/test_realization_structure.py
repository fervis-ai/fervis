import pytest
from tests.lookup.relational_engine.test_scoped_compilation import employee_query
from fervis.lookup.source_binding.parser import compile_source_realization


def test_realization_rejects_co_resident_independent_rows_before_membership():
    verified = employee_query()
    request = verified.request
    source = request.source_catalog.sources[0]
    payload = {
        "set_bindings": {},
        "fact_bindings": {
            ref: [
                {
                    "branch_id": "branch",
                    "mapping_basis": "Salary.",
                    "field_ref": f"source_field:{source.id}:salary",
                }
            ]
            for ref in request.model_authored_fact_refs
        },
        "association_bindings": {
            ref: [
                {
                    "branch_id": "branch",
                    "mapping_basis": "Shared producer.",
                    "from_rows_ref": source.identity_evidence[0].identity_ref,
                    "to_rows_ref": source.identity_evidence[0].identity_ref,
                    "realization_ref": source.id,
                }
            ]
            for ref in verified.binding_plan.association_bindings
        },
    }
    with pytest.raises(
        ValueError, match="association realization has incompatible endpoint rows"
    ):
        compile_source_realization(payload, request=request)


def test_co_residence_evidence_must_name_the_selected_producer():
    from dataclasses import replace

    from fervis.lookup.relation_catalog.row_sources.model import RowSourceIdentityKind
    from fervis.lookup.source_binding.model import AssociationRealizationKind
    from fervis.lookup.source_binding.occurrences import occurrence_scope

    verified = employee_query()
    plan = verified.binding_plan
    reference = next(
        identity
        for identity in verified.request.source_catalog.identity_evidence
        if identity.kind is RowSourceIdentityKind.ENTITY_REFERENCE
    )
    sets = dict(plan.set_bindings)
    manager = "fact_1:set:manager"
    sets[manager] = (
        replace(
            sets[manager][0],
            identity_ref=reference.identity_ref,
            identity_field_refs=reference.field_refs,
        ),
    )
    associations = {
        ref: (
            replace(
                values[0],
                kind=AssociationRealizationKind.CO_RESIDENT,
                source_refs=("unrelated_producer",),
                relation_evidence_ref=None,
            ),
        )
        for ref, values in plan.association_bindings.items()
    }
    invalid = replace(plan, set_bindings=sets, association_bindings=associations)
    with pytest.raises(ValueError, match="co-resident evidence must name its producer"):
        occurrence_scope(verified.request, invalid, "branch")


def _self_relationship_payload():
    verified = employee_query()
    request, plan = verified.request, verified.binding_plan
    identity = request.source_catalog.identity_evidence[0].identity_ref
    edge = request.source_catalog.relation_evidence[0].evidence_ref
    payload = {
        "set_bindings": {},
        "fact_bindings": {
            ref: [
                {
                    "branch_id": "branch",
                    "mapping_basis": "Salary on the role.",
                    "field_ref": "source_field:employees:salary",
                }
            ]
            for ref in request.model_authored_fact_refs
        },
        "association_bindings": {
            "fact_1:association:management": [
                {
                    "branch_id": "branch",
                    "mapping_basis": "Employee manager reference.",
                    "from_rows_ref": identity,
                    "to_rows_ref": identity,
                    "realization_ref": edge,
                    "reference_from_set_ref": "fact_1:set:employee",
                }
            ],
        },
    }
    return request, payload


def test_declared_self_relationship_derives_both_independent_roles():
    from jsonschema import validate, ValidationError
    from fervis.lookup.source_binding.schema import (
        build_semantic_source_realization_schema,
    )

    request, payload = _self_relationship_payload()
    identity = request.source_catalog.identity_evidence[0].identity_ref
    schema = build_semantic_source_realization_schema(request)
    validate(payload, schema)
    result = compile_source_realization(payload, request=request)
    assert {
        ref: values[0].identity_ref for ref, values in result.set_bindings.items()
    } == {
        "fact_1:set:employee": identity,
        "fact_1:set:manager": identity,
    }
    payload["association_bindings"]["fact_1:association:management"][0][
        "realization_ref"
    ] = "employees"
    with pytest.raises(ValidationError):
        validate(payload, schema)
    with pytest.raises(ValueError, match="incompatible endpoint rows"):
        compile_source_realization(payload, request=request)


def test_shared_set_cannot_change_rows_between_associations():
    from dataclasses import replace
    from copy import deepcopy
    from fervis.lookup.question_contract import FactLocalRef
    from fervis.lookup.relation_catalog.row_sources.model import RowSourceIdentityKind

    request, payload = _self_relationship_payload()
    old_ref = FactLocalRef.from_token("fact_1:association:management")
    new_ref = FactLocalRef.from_token("fact_1:association:other_management")
    index = replace(
        request.index,
        association_requirement_refs=(
            *request.index.association_requirement_refs,
            new_ref,
        ),
        term_by_ref={
            **request.index.term_by_ref,
            new_ref: replace(request.index.term_by_ref[old_ref], id="other_management"),
        },
    )
    request = replace(request, index=index)
    conflicting = deepcopy(payload["association_bindings"][old_ref.token])
    reference = next(
        item.identity_ref
        for item in request.source_catalog.identity_evidence
        if item.kind is RowSourceIdentityKind.ENTITY_REFERENCE
    )
    conflicting[0]["from_rows_ref"] = reference
    payload["association_bindings"][new_ref.token] = conflicting
    with pytest.raises(ValueError, match="disagree on their shared set rows"):
        compile_source_realization(payload, request=request)
