"""Retrieval predicates preserve every consumer of their logical population."""

from dataclasses import replace
import pytest

from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    BooleanComposition,
    BooleanCompositionOperator,
    FactTerm,
    Quantifier,
    RequestedOutput,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.qualification import BooleanRequirementUseSite
from fervis.lookup.semantic_types import BooleanType
from tests.lookup.relational_engine.test_scoped_compilation import employee_query


def _request(quantifier, *, shared=False, disjunction=False):
    original = employee_query(quantifier)
    fact = original.request.index.requested_fact
    origin = fact.origin
    quantified = fact.expressions[-1]
    condition = "manager_active"
    extras = ()
    if disjunction:
        extras = (
            BooleanComposition(
                "either",
                BooleanCompositionOperator.OR,
                ("manager_active", "employee_active"),
                origin,
            ),
        )
        condition = "either"
    fact = replace(
        fact,
        facts=(
            *fact.facts,
            FactTerm("manager_active", "manager", BooleanType(), origin),
            FactTerm("employee_active", "employee", BooleanType(), origin),
        ),
        expressions=(*extras, replace(quantified, condition_ref=condition)),
    )
    if shared:
        fact = replace(
            fact,
            expressions=(
                *fact.expressions,
                Aggregate(
                    "total", AggregateFunction.COUNT, "manager", None, False, origin
                ),
            ),
            outputs=(*fact.outputs, RequestedOutput("manager_count", "total", origin)),
        )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    request = replace(original.request, index=index)
    requirement = next(
        r
        for r in index.boolean_requirements
        if r.use_site is BooleanRequirementUseSite.QUANTIFIER_CONDITION
        and r.atom_ref.value_ref.endswith(":manager_active")
    )
    return request, requirement


@pytest.mark.parametrize(
    "quantifier,allowed",
    [
        (Quantifier.EXISTS, True),
        (Quantifier.NOT_EXISTS, True),
        (Quantifier.FORALL, False),
    ],
)
def test_quantification_controls_whether_nonmatching_rows_may_be_discarded(
    quantifier, allowed
):
    request, requirement = _request(quantifier)
    assert (
        request.invocation_preserves_population(requirement.requirement_ref) is allowed
    )


@pytest.mark.parametrize("quantifier", [Quantifier.EXISTS, Quantifier.NOT_EXISTS])
def test_shared_population_cannot_be_narrowed_for_only_one_consumer(quantifier):
    request, requirement = _request(quantifier, shared=True)
    assert not request.invocation_preserves_population(requirement.requirement_ref)


def test_one_disjunct_cannot_be_pushed_as_a_required_invocation_filter():
    request, requirement = _request(Quantifier.EXISTS, disjunction=True)
    assert not request.invocation_preserves_population(requirement.requirement_ref)


def test_co_resident_read_cannot_filter_the_unqualified_outer_population():
    request, requirement = _request(Quantifier.EXISTS)
    refs = tuple(
        request.index.fact_local_ref_by_local_id[name].token
        for name in ("employee", "manager")
    )
    assert not request.invocation_preserves_population(
        requirement.requirement_ref, affected_set_refs=refs
    )


def test_another_relational_path_still_consumes_its_intermediate_population():
    from fervis.lookup.question_contract.model import (
        SetTerm,
        AssociationTerm,
        RelatedRow,
    )

    request, requirement = _request(Quantifier.NOT_EXISTS)
    fact = request.index.requested_fact
    origin = fact.origin
    fact = replace(
        fact,
        sets=(*fact.sets, SetTerm("department", origin)),
        associations=(
            *fact.associations,
            AssociationTerm("department_for_manager", "manager", "department", origin),
        ),
        expressions=(
            *fact.expressions,
            RelatedRow(
                "department_row",
                "department",
                ("management", "department_for_manager"),
                None,
                origin,
            ),
        ),
        outputs=(
            *fact.outputs,
            RequestedOutput("department", "department_row", origin),
        ),
    )
    request = replace(
        request, index=analyze_requested_fact(fact, inputs={}, input_denotations={})
    )
    assert not request.invocation_preserves_population(requirement.requirement_ref)


def test_aggregate_in_same_condition_still_requires_nonmatching_rows():
    from fervis.lookup.question_contract.model import (
        Comparison,
        InputTerm,
        InputDenotation,
        InputDenotationKind,
    )
    from fervis.lookup.expression_operators import ExpressionBinaryOperator
    from fervis.lookup.semantic_types import DecimalType, UnitlessMeasure

    request, _ = _request(Quantifier.EXISTS)
    fact = request.index.requested_fact
    origin = fact.origin
    threshold = InputTerm("threshold", origin, "1", DecimalType(UnitlessMeasure()))
    quantified = fact.expressions[-1]
    fact = replace(
        fact,
        expressions=(
            Aggregate("total", AggregateFunction.COUNT, "manager", None, False, origin),
            Comparison(
                "multiple", ExpressionBinaryOperator.GT, "total", "threshold", origin
            ),
            BooleanComposition(
                "both",
                BooleanCompositionOperator.AND,
                ("manager_active", "multiple"),
                origin,
            ),
            replace(quantified, condition_ref="both"),
        ),
    )
    index = analyze_requested_fact(
        fact,
        inputs={"threshold": threshold},
        input_denotations={
            "threshold": InputDenotation(
                "threshold_meaning",
                "threshold",
                "Minimum count",
                "Literal number",
                None,
                InputDenotationKind.NON_IDENTITY_SCALAR,
            ),
        },
    )
    request = replace(request, index=index)
    requirement = next(
        r
        for r in index.boolean_requirements
        if r.use_site is BooleanRequirementUseSite.QUANTIFIER_CONDITION
        and r.atom_ref.value_ref.endswith(":manager_active")
    )
    assert not request.invocation_preserves_population(requirement.requirement_ref)


def aggregate_quantifier_request():
    from fervis.lookup.question_contract.model import Comparison, FactTerm, Quantify
    from fervis.lookup.expression_operators import ExpressionBinaryOperator
    from tests.lookup.source_binding.test_choice_requirements import (
        _source_required_choice_request,
    )

    request, _, _ = _source_required_choice_request(choices=("false", "true"))
    fact = request.index.requested_fact
    origin = fact.origin
    fact = replace(
        fact,
        facts=(FactTerm("flag", "s1", BooleanType(), origin),),
        expressions=(
            Aggregate("total", AggregateFunction.COUNT, "s1", "flag", False, origin),
            Comparison(
                "equal", ExpressionBinaryOperator.EQUALS, "total", "total", origin
            ),
            BooleanComposition(
                "not_flag", BooleanCompositionOperator.NOT, ("flag",), origin
            ),
            Quantify(
                "opposite_exists", Quantifier.EXISTS, "s1", (), "not_flag", origin
            ),
            BooleanComposition(
                "answer",
                BooleanCompositionOperator.AND,
                ("equal", "opposite_exists"),
                origin,
            ),
        ),
        outputs=(RequestedOutput("output_1", "answer", origin),),
    )
    request = replace(
        request, index=analyze_requested_fact(fact, inputs={}, input_denotations={})
    )
    requirement = next(
        r
        for r in request.index.boolean_requirements
        if r.use_site is BooleanRequirementUseSite.AGGREGATE_FILTER
    )
    return request, requirement


def test_aggregate_filter_cannot_change_another_quantifier_consumer():
    request, requirement = aggregate_quantifier_request()
    assert not request.invocation_preserves_population(requirement.requirement_ref)
