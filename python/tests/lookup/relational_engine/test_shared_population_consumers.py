"""A filtered aggregate and existential consumer share an unfiltered API read."""

from dataclasses import replace
from fervis.lookup.available_sources import SourceFieldBinding
from fervis.lookup.relation_catalog.row_sources import (
    RowSourceField,
    RowSourceValueType,
)
from fervis.lookup.source_binding.parser import (
    compile_source_realization,
    compile_source_binding_plan,
)
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.source_binding.choice_requirements import requirement_choice_surfaces
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.qualification import BooleanPolarity
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
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    RelationEngineInput,
    ExecutableOperation,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    CompletenessProof,
    CompletenessStatus,
)
from tests.lookup.source_binding.test_scoped_invocation_predicates import (
    aggregate_quantifier_request,
)


def test_filtered_aggregate_and_opposite_exists_keep_both_row_states():
    request, _ = aggregate_quantifier_request()
    source = request.source_catalog.sources[0]
    field = RowSourceField("flag", "field.flag", "Flag", RowSourceValueType.BOOLEAN, ())
    source = replace(
        source,
        fields=(field,),
        params=tuple(replace(p, required=False) for p in source.params),
    )
    request = replace(
        request, source_catalog=replace(request.source_catalog, sources=(source,))
    )
    branch = request.strategy.branches[0].branch_id
    realization = compile_source_realization(
        {
            "set_bindings": {
                request.index.subject_obligation.subject_set_ref.token: [
                    {
                        "branch_id": branch,
                        "mapping_basis": "Source rows",
                        "rows_ref": source.id,
                    }
                ]
            },
            "fact_bindings": {
                "fact_1:fact:flag": [
                    {
                        "branch_id": branch,
                        "mapping_basis": "Returned flag",
                        "field_ref": SourceFieldBinding(source.id, field).ref,
                    }
                ]
            },
            "association_bindings": {},
        },
        request=request,
    )
    request = realization.request
    choices = {
        surface.surface_ref: {
            choice.value: {
                "mapping_basis": "The returned Boolean realizes the signed condition",
                "selected_by_requirements": [
                    r.requirement_ref
                    for r in request.index.boolean_requirements
                    if (r.atom_ref.polarity is BooleanPolarity.POSITIVE)
                    == (choice.value == "true")
                ],
            }
            for choice in surface.values
        }
        for surface in requirement_choice_surfaces(request, branch)
    }
    plan = compile_source_binding_plan(
        {
            "resolved_input_applications": {branch: []},
            "finite_choice_applications": {branch: {}},
            "choice_requirement_applications": {branch: choices},
        },
        realization=realization,
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    compiled = compile_verified_source_strategy(verified)
    program = compiled.answer_program
    assert all(not relation.source.param_bindings for relation in program.relations)
    scalars = {}
    for expression in program_value_expressions(program):
        for leaf in expression_references(expression.expression).leaves:
            if isinstance(leaf, (ConstantRef, ParameterRef)):
                value = resolve_value_expression(
                    leaf, bindings=compiled.initial_bindings
                )
                scalars[expression_input_id(leaf)] = ScalarInput(
                    expression_input_id(leaf),
                    value.value,
                    resolved_value_expression_type(leaf, value),
                )
    result = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    relation.id,
                    ({"flag": True}, {"flag": False}),
                    field_types={"flag": "boolean"},
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for relation in program.relations
            ),
            operations=tuple(
                ExecutableOperation(op.id, op.spec, op.output_relation)
                for op in program.operations
            ),
            scalar_inputs=tuple(scalars.values()),
        )
    )
    assert result.issue is None
    assert result.scalars[program.result_projection.scalar_outputs[0].scalar_id] is True
