"""Explicit state ownership reaches a declared returned-choice correspondence."""

from copy import deepcopy
from dataclasses import replace

import pytest

from fervis.lookup.available_sources import AvailableSourceCatalog
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.relation_catalog.row_sources import (
    RowSourceField,
    RowSourceValueType,
)
from fervis.lookup.source_binding.verification import (
    VerifiedSourceStrategy,
    verify_source_strategy,
)
from fervis.lookup.answer_program.operations import FilterSpec
from fervis.lookup.answer_program.values import ConstantRef, ParameterRef
from fervis.lookup.answer_program.inputs import (
    resolve_value_expression,
    resolved_value_expression_type,
)
from fervis.lookup.answer_program.expressions import (
    expression_input_id,
    expression_references,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessStatus,
    RelationRows,
)
from tests.lookup.source_binding import test_choice_requirements as fixtures
from tests.lookup.source_binding._fixtures import (
    compile_binding_fixture,
    validate_binding_fixture,
)


@pytest.mark.parametrize("with_parameter", [True, False])
def test_explicit_cancellation_survives_returned_boolean_state_filter(
    monkeypatch, with_parameter
):
    saved = {}

    class Captured(Exception):
        pass

    def capture(payload, *, request):
        saved.update(payload=deepcopy(payload), request=request)
        raise Captured

    monkeypatch.setattr(fixtures, "validate_binding_fixture", capture)
    with pytest.raises(Captured):
        fixtures.test_direct_boolean_requirement_owns_matching_truth_choice_application()
    request = saved["request"]
    payload = saved["payload"]
    source = replace(
        request.source_catalog.sources[0],
        read_id="list_sales",
        fields=(
            RowSourceField(
                "state_bit",
                "field.state_bit",
                "Cancellation state",
                RowSourceValueType.BOOLEAN,
                (),
                choices=("false", "true"),
            ),
        ),
    )
    if not with_parameter:
        source = replace(source, params=())
    request = replace(
        request,
        source_catalog=AvailableSourceCatalog(
            request.source_catalog.contract_snapshot, (source,), ()
        ),
    )
    branch = request.strategy.branches[0]
    requirement = request.index.boolean_requirements[0].requirement_ref
    reviews = payload["subject_binding"]["branch_realizations"][0][
        "finite_choice_reviews"
    ]
    returned = deepcopy(reviews["source_surface:source_sales:parameter:is_canceled"])
    returned["surface_mapping_basis"] = (
        "This returned Boolean expresses the explicitly requested cancellation condition."
    )
    returned["choice_reviews"]["true"]["selected_by_requirements"] = [requirement]
    if not with_parameter:
        reviews.clear()
        payload["finite_choice_applications"] = {branch.branch_id: {}}
        payload["fact_bindings"]["fact_1:fact:f1"] = [
            {
                "branch_id": branch.branch_id,
                "mapping_basis": "The returned Boolean states cancellation.",
                "field_ref": "source_field:source_sales:state_bit",
            }
        ]
    reviews["source_surface:source_sales:field:state_bit"] = returned
    validate_binding_fixture(payload, request=request)
    plan = compile_binding_fixture(payload, request=request)
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    field_review = next(
        review
        for review in plan.subject_binding.branch_realizations[0].surface_reviews
        if review.surface_ref.endswith(":field:state_bit")
    )
    selected = next(
        choice
        for choice in field_review.choice_reviews
        if choice.explicit_user_override_applies
    )
    assert selected.selection_requirement_refs == (requirement,)
    # Ordinary coverage admits the explicit exception; the Boolean predicate
    # then selects exactly the requested state, preserving OR composition.
    assert selected.choice_ref in field_review.included_choice_refs
    result = compile_verified_source_strategy(verified)
    program = result.answer_program
    scalars = {}
    for operation in program.operations:
        if isinstance(operation.spec, FilterSpec):
            for leaf in expression_references(operation.spec.condition).leaves:
                if not isinstance(leaf, (ConstantRef, ParameterRef)):
                    continue
                resolved = resolve_value_expression(
                    leaf, bindings=result.initial_bindings
                )
                scalars[expression_input_id(leaf)] = ScalarInput(
                    expression_input_id(leaf),
                    resolved.value,
                    resolved_value_expression_type(leaf, resolved),
                )
    execution = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    program.relations[0].id,
                    ({"state_bit": False}, {"state_bit": True}),
                    field_types={"state_bit": "boolean"},
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                ),
            ),
            operations=tuple(
                ExecutableOperation(op.id, op.spec, op.output_relation)
                for op in program.operations
            ),
            scalar_inputs=tuple(scalars.values()),
        )
    )
    assert execution.issue is None
    assert execution.relation(program.operations[-1].output_relation).rows == (
        {"aggregate_1": 1},
    )
    returned["choice_reviews"]["true"]["selected_by_requirements"] = [
        "source_required:invented"
    ]
    with pytest.raises(ValueError, match="incompatible explicit requirement"):
        compile_binding_fixture(payload, request=request)


def test_optional_state_disjunct_preserves_ordinary_rows_and_is_not_pushed_into_request(
    monkeypatch,
):
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.question_contract.model import (
        BooleanComposition,
        BooleanCompositionOperator,
        Comparison,
        FactTerm,
        InputTerm,
    )
    from fervis.lookup.expression_operators import ExpressionBinaryOperator
    from fervis.lookup.semantic_types import NumericType
    from fervis.lookup.grounding import CanonicalInputValue
    from fervis.lookup.answer_program.values import FactValue, LiteralType
    from tests.lookup.fact_compilation.test_compiler import _denotation

    saved = {}

    class Captured(Exception):
        pass

    def capture(payload, *, request):
        saved.update(payload=deepcopy(payload), request=request)
        raise Captured

    monkeypatch.setattr(fixtures, "validate_binding_fixture", capture)
    with pytest.raises(Captured):
        fixtures.test_direct_boolean_requirement_owns_matching_truth_choice_application()
    request, payload = saved["request"], saved["payload"]
    fact = request.index.requested_fact
    threshold = InputTerm("i1", fact.origin, "100", NumericType())
    fact = replace(
        fact,
        facts=(*fact.facts, FactTerm("f_amount", "s1", NumericType(), fact.origin)),
        expressions=(
            *fact.expressions,
            Comparison(
                "e_large", ExpressionBinaryOperator.GT, "f_amount", "i1", fact.origin
            ),
            BooleanComposition(
                "e_either",
                BooleanCompositionOperator.OR,
                ("f1", "e_large"),
                fact.origin,
            ),
        ),
        qualification_ref="e_either",
    )
    index = analyze_requested_fact(
        fact, inputs={threshold.id: threshold}, input_denotations=_denotation(threshold)
    )
    branch = replace(
        request.strategy.branches[0],
        qualification_clause_refs=tuple(
            clause.clause_ref for clause in index.qualification.clauses
        ),
    )
    source = replace(
        request.source_catalog.sources[0],
        read_id="list_sales",
        fields=(
            RowSourceField(
                "state_bit",
                "field.state_bit",
                "Cancellation",
                RowSourceValueType.BOOLEAN,
                (),
                choices=("false", "true"),
            ),
            RowSourceField(
                "amount", "field.amount", "Amount", RowSourceValueType.DECIMAL, ()
            ),
        ),
    )
    value = FactValue.literal(
        id="threshold",
        known_input_id="i1",
        literal_type=LiteralType.NUMBER,
        value="100",
        proof_refs=("question_input:i1",),
    )
    request = replace(
        request,
        index=index,
        strategy=replace(request.strategy, branches=(branch,)),
        source_catalog=AvailableSourceCatalog(
            request.source_catalog.contract_snapshot, (source,), ()
        ),
        canonical_values=(
            CanonicalInputValue(
                "threshold",
                "i1",
                tuple(use.use_ref for use in index.input_use_sites),
                value,
                ("question_input:i1",),
            ),
        ),
    )
    state_requirement = next(
        item.requirement_ref
        for item in index.boolean_requirements
        if item.atom_ref.value_ref.endswith(":f1")
    )
    assert not request.invocation_preserves_population(state_requirement)
    payload["finite_choice_applications"] = {branch.branch_id: {}}
    payload["fact_bindings"]["fact_1:fact:f_amount"] = [
        {
            "branch_id": branch.branch_id,
            "mapping_basis": "The returned amount supplies the threshold comparison.",
            "field_ref": "source_field:source_sales:amount",
        }
    ]
    reviews = payload["subject_binding"]["branch_realizations"][0][
        "finite_choice_reviews"
    ]
    parameter_review = reviews["source_surface:source_sales:parameter:is_canceled"]
    parameter_review["choice_reviews"]["true"]["selected_by_requirements"] = [
        state_requirement
    ]
    reviews["source_surface:source_sales:field:state_bit"] = deepcopy(parameter_review)
    validate_binding_fixture(payload, request=request)
    plan = compile_binding_fixture(payload, request=request)
    assert plan.invocation_applications == ()
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    result = compile_verified_source_strategy(verified)
    program = result.answer_program
    scalars = {}
    for operation in program.operations:
        if isinstance(operation.spec, FilterSpec):
            for leaf in expression_references(operation.spec.condition).leaves:
                if isinstance(leaf, (ConstantRef, ParameterRef)):
                    resolved = resolve_value_expression(
                        leaf, bindings=result.initial_bindings
                    )
                    scalars[expression_input_id(leaf)] = ScalarInput(
                        expression_input_id(leaf),
                        resolved.value,
                        resolved_value_expression_type(leaf, resolved),
                    )
    execution = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    program.relations[0].id,
                    (
                        {"state_bit": True, "amount": 20},
                        {"state_bit": False, "amount": 150},
                        {"state_bit": False, "amount": 20},
                    ),
                    field_types={"state_bit": "boolean", "amount": "decimal"},
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                ),
            ),
            operations=tuple(
                ExecutableOperation(op.id, op.spec, op.output_relation)
                for op in program.operations
            ),
            scalar_inputs=tuple(scalars.values()),
        )
    )
    assert execution.issue is None
    assert execution.relation(program.operations[-1].output_relation).rows == (
        {"aggregate_1": 2},
    )


def test_positive_logical_property_can_map_to_false_on_an_inverse_boolean_parameter(
    monkeypatch,
):
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind

    saved = {}

    class Captured(Exception):
        pass

    def capture(payload, *, request):
        saved.update(payload=deepcopy(payload), request=request)
        raise Captured

    monkeypatch.setattr(fixtures, "validate_binding_fixture", capture)
    with pytest.raises(Captured):
        fixtures.test_direct_boolean_requirement_owns_matching_truth_choice_application()
    request, payload = saved["request"], saved["payload"]
    fact = request.index.requested_fact
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "inactive sales")
    fact = replace(fact, origin=origin, facts=(replace(fact.facts[0], origin=origin),))
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    source = request.source_catalog.sources[0]
    source = replace(
        source,
        params=(
            replace(
                source.params[0],
                id="is_active",
                param_ref="source_sales.is_active",
                name="is_active",
            ),
        ),
    )
    request = replace(
        request,
        index=index,
        source_catalog=AvailableSourceCatalog(
            request.source_catalog.contract_snapshot, (source,), ()
        ),
    )
    branch = request.strategy.branches[0]
    requirement = index.boolean_requirements[0].requirement_ref
    application = payload["finite_choice_applications"][branch.branch_id][requirement]
    application.update(
        surface_ref="source_surface:source_sales:parameter:is_active",
        selected_choice_values=["false"],
        application_basis="An inactive sale has is_active=false.",
    )
    reviews = payload["subject_binding"]["branch_realizations"][0][
        "finite_choice_reviews"
    ]
    review = reviews.pop("source_surface:source_sales:parameter:is_canceled")
    review["surface_mapping_basis"] = (
        "The parameter describes whether a sale is active."
    )
    review["choice_reviews"]["false"].update(
        baseline_decision="EXCLUDE",
        choice_domain_meaning="Inactive sales.",
        decision_basis="Inactive instances are outside the ordinary current population.",
    )
    review["choice_reviews"]["true"].update(
        baseline_decision="INCLUDE",
        choice_domain_meaning="Active sales.",
        decision_basis="Active instances belong to the ordinary population.",
    )
    reviews["source_surface:source_sales:parameter:is_active"] = review
    validate_binding_fixture(payload, request=request)
    plan = compile_binding_fixture(payload, request=request)
    assert (
        request.source_catalog.choice_value(
            plan.invocation_applications[0].value_ref
        ).value
        == "false"
    )
    assert (
        plan.subject_binding.branch_realizations[0]
        .surface_reviews[0]
        .choice_reviews[0]
        .explicit_user_override_applies
    )
