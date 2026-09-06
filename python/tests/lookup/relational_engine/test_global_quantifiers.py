from dataclasses import replace
import pytest

from tests.lookup.fact_compilation.test_compiler import _compile_memory_count
from fervis.lookup.question_contract.model import (
    FactTerm,
    NullCheck,
    Quantify,
    Quantifier,
    RequestedOutput,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact, Singleton
from fervis.lookup.semantic_types import TextType
from fervis.lookup.expression_operators import ExpressionUnaryOperator
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.source_binding.model import (
    FactRealization,
    FactRealizationKind,
    SourceRealization,
)
from fervis.lookup.source_binding.membership import parse_source_membership
from fervis.lookup.source_binding.parser import compile_source_binding_plan
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory


@pytest.mark.parametrize(
    ("quantifier", "empty", "expected"),
    [
        (Quantifier.EXISTS, False, True),
        (Quantifier.EXISTS, True, False),
        (Quantifier.NOT_EXISTS, False, False),
        (Quantifier.NOT_EXISTS, True, True),
        (Quantifier.FORALL, False, True),
        (Quantifier.FORALL, True, True),
    ],
)
@pytest.mark.parametrize("qualification_kind", ("none", "present", "absent"))
@pytest.mark.parametrize("branch_count", (1, 2))
def test_unassociated_quantification_returns_one_boolean(quantifier, empty, expected, qualification_kind, branch_count):
    if qualification_kind == "absent":
        expected = quantifier is not Quantifier.EXISTS
    _, _, memory, _, _, original = _compile_memory_count(
        () if empty else ({"event_id": "one"}, {"event_id": "two"})
    )
    memory = replace(memory, field_types={"event_id": "string"})
    fact = original.request.index.requested_fact
    fact = replace(
        fact,
        facts=(FactTerm("identifier", "s1", TextType(), fact.origin),),
        expressions=(
            NullCheck(
                "has_value", ExpressionUnaryOperator.NOT_NULL, "identifier", fact.origin
            ),
            *(
                (NullCheck("missing_value", ExpressionUnaryOperator.IS_NULL, "identifier", fact.origin),)
                if qualification_kind == "absent" else ()
            ),
            Quantify("quantified", quantifier, "s1", (), "has_value", fact.origin),
        ),
        outputs=(RequestedOutput("value", "quantified", fact.origin),),
        qualification_ref={"none": None, "present": "has_value", "absent": "missing_value"}[qualification_kind],
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    assert isinstance(index.result_grain, Singleton)
    source = next(
        s
        for s in build_row_source_catalog(
            RelationCatalog(), memory_relations=(memory,)
        ).sources
        if s.memory_ref == memory.id
    )
    base_branch = original.request.strategy.branches[0]
    branches = tuple(replace(base_branch, branch_id=f"branch_{i}", qualification_clause_refs=tuple(
        clause.clause_ref for clause in index.qualification.clauses)) for i in range(branch_count))
    request = replace(
        original.request,
        index=index,
        strategy=replace(original.request.strategy, branches=branches),
        source_catalog=replace(original.request.source_catalog, sources=(source,)),
    )
    branch_ids = tuple(branch.branch_id for branch in branches)
    realization = SourceRealization(
        request,
        {ref: tuple(replace(values[0], branch_id=branch_id) for branch_id in branch_ids)
         for ref, values in original.binding_plan.set_bindings.items()},
        {
            index.fact_local_ref_by_local_id["identifier"].token: (
                FactRealization(
                    branch_id,
                    "Observed identifier.",
                    source.id,
                    FactRealizationKind.RETURNED_FIELD,
                    None,
                    ("event_id",),
                    (source.id, "event_id"),
                )
                for branch_id in branch_ids
            )
        },
        {},
    )
    membership = parse_source_membership({branch: {} for branch in branch_ids}, realization=realization)
    parsed = compile_source_binding_plan(
        {
            "resolved_input_applications": {branch: [] for branch in branch_ids},
            "finite_choice_applications": {branch: {} for branch in branch_ids},
            "choice_requirement_applications": {branch: {} for branch in branch_ids},
        },
        membership=membership,
    )
    verified = verify_source_strategy(parsed, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    compiled = compile_verified_source_strategy(verified)
    result = invoke_answer_program(
        program=compiled.answer_program,
        bindings=compiled.initial_bindings,
        environment=ExecutionEnvironment(
            catalog=RelationCatalog(), memory_relations=(memory,)
        ),
        ports=RuntimePorts(
            data_access_port=None, memory=LookupMemory(relations=(memory,))
        ),
    )
    assert result.issue is None
    assert list(result.fact_result.outcome.scalars.values()) == [expected]
