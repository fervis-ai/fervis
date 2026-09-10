"""Exercise role membership through production parsing, verification, and execution."""

from dataclasses import replace
from decimal import Decimal
from fervis.host_api.contracts.population import ParameterPopulation, ParameterRowValues

import pytest
from jsonschema import Draft202012Validator, validate

from tests.lookup.relational_engine.test_scoped_compilation import employee_query
from fervis.lookup.question_contract.model import (
    FactTerm,
    BooleanComposition,
    BooleanCompositionOperator,
    Quantify,
    Quantifier,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.semantic_types import BooleanType
from fervis.lookup.qualification import BooleanRequirementUseSite
from fervis.lookup.relation_catalog.row_sources.model import (
    RowSourceField,
    RowSourceParam,
    RowSourceValueType,
)
from fervis.lookup.answer_program.relations import FieldBindingRole
from fervis.lookup.source_binding.choice_requirements import requirement_choice_surfaces

from fervis.lookup.source_binding.parser import (
    compile_source_realization,
    compile_source_binding_plan,
)
from fervis.lookup.source_binding.schema import (
    build_semantic_source_realization_schema,
    build_semantic_source_binding_schema,
)
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    CompletenessProof,
    CompletenessStatus,
)
from fervis.lookup.answer_program.inputs import (
    program_value_expressions,
    resolve_value_expression,
    resolved_value_expression_type,
)
from fervis.lookup.answer_program.expressions import (
    expression_references,
    expression_input_id,
)
from fervis.lookup.answer_program.values import ConstantRef, ParameterRef


def _run_role_membership(
    encoding,
    parameter_default,
    include_inactive_employees,
    *,
    employee_predicate=False,
    manager_parameter_predicate=True,
):
    employee_predicate = not include_inactive_employees
    original = employee_query()
    fact = original.request.index.requested_fact
    fact = replace(
        fact,
        facts=(
            *fact.facts,
            FactTerm("manager_active", "manager", BooleanType(), fact.origin),
        ),
        expressions=(
            BooleanComposition(
                "inactive",
                BooleanCompositionOperator.NOT,
                ("manager_active",),
                fact.origin,
            ),
            Quantify(
                "has_inactive_manager",
                Quantifier.EXISTS,
                "manager",
                ("management",),
                "inactive",
                fact.origin,
            ),
        ),
        qualification_ref="has_inactive_manager",
    )
    if employee_predicate:
        fact = replace(
            fact,
            facts=(
                *fact.facts,
                FactTerm("employee_active", "employee", BooleanType(), fact.origin),
            ),
            expressions=(
                *fact.expressions,
                BooleanComposition(
                    "both",
                    BooleanCompositionOperator.AND,
                    ("employee_active", "has_inactive_manager"),
                    fact.origin,
                ),
            ),
            qualification_ref="both",
        )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    source = original.request.source_catalog.sources[0]
    source = replace(
        source,
        fields=(
            *source.fields,
            RowSourceField(
                "active",
                "field.active",
                "Active employee",
                RowSourceValueType.BOOLEAN
                if encoding == "boolean"
                else RowSourceValueType.CHOICE,
                (FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE),
                choices=("false", "true")
                if encoding == "boolean"
                else ("inactive", "active"),
            ),
        ),
    )
    if parameter_default != "absent":
        source = replace(
            source,
            params=(
                RowSourceParam(
                    "active",
                    "param.active",
                    "Activity filter",
                    RowSourceValueType.BOOLEAN,
                    choices=("false", "true"),
                    default=parameter_default,
                    population=ParameterPopulation(
                        field_path="field.active",
                        value_mapping=(
                            ParameterRowValues(
                                "false",
                                ("false" if encoding == "boolean" else "inactive",),
                            ),
                            ParameterRowValues(
                                "true", ("true" if encoding == "boolean" else "active",)
                            ),
                        ),
                    ),
                ),
            ),
        )
    request = replace(
        original.request,
        index=index,
        realized_fact_fields=(),
        source_catalog=replace(original.request.source_catalog, sources=(source,)),
    )
    scalar_ref = index.fact_local_ref_by_local_id["manager_active"].token
    row_payload = {
        "set_bindings": {
            ref: [
                {
                    "branch_id": v.branch_id,
                    "mapping_basis": v.mapping_basis,
                    "rows_ref": v.identity_ref or v.source_ref,
                }
                for v in values
            ]
            for ref, values in original.binding_plan.set_bindings.items()
        },
        "fact_bindings": {
            scalar_ref: [
                {
                    "branch_id": "branch",
                    "mapping_basis": "Returned Boolean activity.",
                    "field_ref": "source_field:employees:active",
                }
            ]
            if encoding == "boolean"
            else []
        },
        "association_bindings": {
            ref: [
                {
                    "branch_id": value.branch_id,
                    "mapping_basis": value.mapping_basis,
                    "realization_ref": value.relation_evidence_ref,
                    "reference_from_set_ref": value.reference_from_set_ref,
                }
                for value in values
            ]
            for ref, values in original.binding_plan.association_bindings.items()
        },
    }
    if employee_predicate:
        row_payload["fact_bindings"][
            index.fact_local_ref_by_local_id["employee_active"].token
        ] = (
            [
                {
                    "branch_id": "branch",
                    "mapping_basis": "Employee activity.",
                    "field_ref": "source_field:employees:active",
                }
            ]
            if encoding == "boolean"
            else []
        )
    schema = build_semantic_source_realization_schema(request)
    Draft202012Validator.check_schema(schema)
    validate(row_payload, schema)
    if encoding == "enum":
        assert (
            schema["properties"]["fact_bindings"]["properties"][scalar_ref]["minItems"]
            == 0
        )
    realization = compile_source_realization(row_payload, request=request)
    membership = realization
    inner = next(
        r
        for r in index.boolean_requirements
        if r.use_site is BooleanRequirementUseSite.QUANTIFIER_CONDITION
    )
    payload = {
        "resolved_input_applications": {"branch": []},
        "finite_choice_applications": {"branch": {}},
        "choice_requirement_applications": {
            "branch": {
                surface.surface_ref: {
                    choice.value: {
                        "mapping_basis": "The exception belongs to the inactive-manager condition.",
                        "selected_by_requirements": [inner.requirement_ref]
                        if choice.value in ("false", "inactive")
                        and (
                            manager_parameter_predicate
                            or surface.kind.value == "RETURNED_FIELD"
                        )
                        else [],
                    }
                    for choice in surface.values
                }
                for surface in requirement_choice_surfaces(membership.request, "branch")
            }
        },
    }
    for owner in membership.request.invocation_application_owner_refs:
        options = membership.request.finite_choice_options_for_owner(
            owner, branch_id="branch"
        )
        if not options:
            continue
        surface, choices = options[0]
        if owner.startswith("subject_scope:"):
            selected = [choice.value for choice in choices]
        elif owner in {
            r.requirement_ref
            for r in index.boolean_requirements
            if r.use_site is BooleanRequirementUseSite.POPULATION
        }:
            selected = ["true"]
        elif owner == inner.requirement_ref and manager_parameter_predicate:
            selected = ["false"]
        else:
            payload["finite_choice_applications"]["branch"][owner] = None
            continue
        payload["finite_choice_applications"]["branch"][owner] = {
            "surface_ref": surface.surface_ref,
            "selected_choice_values": selected,
            "application_basis": "Complete population scope or explicit employee activity predicate.",
        }
    if employee_predicate:
        owner = next(
            r.requirement_ref
            for r in index.boolean_requirements
            if r.use_site is BooleanRequirementUseSite.POPULATION
            and r.atom_ref.value_ref.endswith(":employee_active")
        )
        for choices in payload["choice_requirement_applications"]["branch"].values():
            for value, application in choices.items():
                if value in ("true", "active"):
                    application["selected_by_requirements"].append(owner)
    validate(payload, build_semantic_source_binding_schema(membership.request))
    plan = compile_source_binding_plan(payload, realization=membership)
    verified = verify_source_strategy(plan, request=membership.request)
    assert isinstance(verified, VerifiedSourceStrategy)
    compilation = compile_verified_source_strategy(verified)
    program = compilation.answer_program
    scalars = {}
    for expression in program_value_expressions(program):
        for leaf in expression_references(expression.expression).leaves:
            if isinstance(leaf, (ConstantRef, ParameterRef)):
                value = resolve_value_expression(
                    leaf, bindings=compilation.initial_bindings
                )
                scalars[expression_input_id(leaf)] = ScalarInput(
                    expression_input_id(leaf),
                    value.value,
                    resolved_value_expression_type(leaf, value),
                )
    rows = (
        {
            "id": "inactive_employee",
            "manager_id": "inactive_manager",
            "salary": Decimal(100),
            "active": False,
        },
        {
            "id": "active_employee",
            "manager_id": "inactive_manager",
            "salary": Decimal(50),
            "active": True,
        },
        {
            "id": "inactive_manager",
            "manager_id": None,
            "salary": Decimal(200),
            "active": False,
        },
    )
    if encoding == "enum":
        rows = tuple(
            {**row, "active": "active" if row["active"] else "inactive"} for row in rows
        )

    def read_rows(relation):
        if relation.source.param_bindings:
            admitted = resolve_value_expression(
                relation.source.param_bindings[0].value_expr,
                bindings=compilation.initial_bindings,
            ).value
        elif parameter_default in ("true", "false"):
            admitted = parameter_default == "true"
        else:
            return rows
        return tuple(
            row
            for row in rows
            if (row["active"] if encoding == "boolean" else row["active"] == "active")
            is admitted
        )

    result = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    r.id,
                    read_rows(r),
                    ("id",),
                    {
                        "id": "string",
                        "manager_id": "string",
                        "salary": "decimal",
                        "active": "boolean" if encoding == "boolean" else "string",
                    },
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for r in program.relations
            ),
            operations=tuple(
                ExecutableOperation(o.id, o.spec, o.output_relation)
                for o in program.operations
            ),
            scalar_inputs=tuple(scalars.values()),
        )
    )
    assert result.issue is None
    output = program.result_projection.relation_outputs[0]
    assert [
        row[output.entity_key.components[0].field_id]
        for row in result.relation(output.relation_id).rows
    ] == (
        ["inactive_employee", "active_employee"]
        if include_inactive_employees
        else ["active_employee"]
    )


@pytest.mark.parametrize("include_inactive_employees", [False, True])
@pytest.mark.parametrize("parameter_default", ["absent", None, "true", "false"])
@pytest.mark.parametrize("encoding", ["boolean", "enum"])
def test_explicit_employee_and_manager_predicates_preserve_scope(
    encoding, parameter_default, include_inactive_employees
):
    _run_role_membership(encoding, parameter_default, include_inactive_employees)
