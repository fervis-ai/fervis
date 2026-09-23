"""Global Boolean domains remain independent across alternative source branches."""

from dataclasses import replace
import pytest
from tests.lookup.fact_compilation.test_compiler import _compile_memory_count
from fervis.lookup.question_contract.model import (
    SetTerm,
    FactTerm,
    NullCheck,
    Quantify,
    Quantifier,
    RequestedOutput,
    BooleanComposition,
    BooleanCompositionOperator,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.semantic_types import TextType
from fervis.lookup.expression_operators import ExpressionUnaryOperator
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.source_binding.model import (
    SourceRealization,
    SetRealization,
    FactRealization,
    FactRealizationKind,
)
from fervis.lookup.source_binding.parser import compile_source_binding_plan
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory


@pytest.mark.parametrize("qualified", (False, True))
@pytest.mark.parametrize("quantifier", tuple(Quantifier))
@pytest.mark.parametrize("subject_empty", (False, True))
@pytest.mark.parametrize("compose", (False, True))
@pytest.mark.parametrize("populations", ((0,), (1,), (0, 1), (1, 0)))
def test_independent_global_domain_uses_all_branches(
    quantifier, subject_empty, compose, populations, qualified
):
    _, _, subject, _, _, original = _compile_memory_count(
        () if subject_empty else ({"event_id": "subject"},)
    )
    subject = replace(subject, field_types={"event_id": "string"})
    memories = (
        subject,
        *(
            replace(
                subject,
                id=f"other_{i}",
                rows=() if not size else ({"event_id": "other"},),
            )
            for i, size in enumerate(populations)
        ),
    )
    sources = build_row_source_catalog(
        RelationCatalog(), memory_relations=memories
    ).sources
    origin = original.request.index.requested_fact.origin
    fact = replace(
        original.request.index.requested_fact,
        qualification_ref="condition1" if qualified else None,
        sets=(SetTerm("s1", origin), SetTerm("s2", origin)),
        facts=(
            FactTerm("id1", "s1", TextType(), origin),
            FactTerm("id2", "s2", TextType(), origin),
        ),
        expressions=(
            NullCheck("condition1", ExpressionUnaryOperator.NOT_NULL, "id1", origin),
            NullCheck(
                "condition2",
                ExpressionUnaryOperator.IS_NULL
                if quantifier is Quantifier.FORALL
                else ExpressionUnaryOperator.NOT_NULL,
                "id2",
                origin,
            ),
            Quantify("exists1", Quantifier.EXISTS, "s1", (), "condition1", origin),
            Quantify("quantified2", quantifier, "s2", (), "condition2", origin),
            BooleanComposition(
                "both",
                BooleanCompositionOperator.AND,
                ("exists1", "quantified2"),
                origin,
            ),
        ),
        outputs=(
            RequestedOutput("answer", "both" if compose else "quantified2", origin),
        ),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    if compose:
        assert index.output_qualification("answer") == index.qualification
    else:
        from fervis.lookup.qualification import QualificationDNF

        assert index.output_qualification("answer") == QualificationDNF.true(
            index.requested_fact_id
        )
    by_memory = {source.memory_ref: source for source in sources}
    branches = tuple(
        replace(
            original.request.strategy.branches[0],
            branch_id=f"branch_{i}",
            source_refs=(by_memory[subject.id].id, by_memory[other.id].id),
            qualification_clause_refs=tuple(
                clause.clause_ref for clause in index.qualification.clauses
            ),
        )
        for i, other in enumerate(memories[1:])
    )
    request = replace(
        original.request,
        index=index,
        source_catalog=replace(original.request.source_catalog, sources=tuple(sources)),
        strategy=replace(original.request.strategy, branches=branches),
        realized_fact_fields=(),
        realized_set_sources=(),
    )
    sets, fields = {}, {}
    for number, set_id in enumerate(("s1", "s2"), 1):
        set_ref = index.fact_local_ref_by_local_id[set_id].token
        field_ref = index.fact_local_ref_by_local_id[f"id{number}"].token
        sets[set_ref] = []
        fields[field_ref] = []
        for branch, other in zip(branches, memories[1:]):
            source = by_memory[subject.id if set_id == "s1" else other.id]
            sets[set_ref].append(
                SetRealization(
                    branch.branch_id,
                    "The declared row domain.",
                    source.id,
                    None,
                    (),
                    (source.id,),
                )
            )
            fields[field_ref].append(
                FactRealization(
                    branch.branch_id,
                    "Observed identifier.",
                    source.id,
                    FactRealizationKind.RETURNED_FIELD,
                    None,
                    ("event_id",),
                    (source.id,),
                )
            )
    realization = SourceRealization(
        request,
        {k: tuple(v) for k, v in sets.items()},
        {k: tuple(v) for k, v in fields.items()},
        {},
    )
    membership = realization
    plan = compile_source_binding_plan(
        {
            "resolved_input_applications": {b.branch_id: [] for b in branches},
            "finite_choice_applications": {b.branch_id: {} for b in branches},
            "choice_requirement_applications": {b.branch_id: {} for b in branches},
        },
        realization=membership,
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    compiled = compile_verified_source_strategy(verified)
    execution = invoke_answer_program(
        program=compiled.answer_program,
        bindings=compiled.initial_bindings,
        environment=ExecutionEnvironment(
            catalog=RelationCatalog(), memory_relations=memories
        ),
        ports=RuntimePorts(
            data_access_port=None, memory=LookupMemory(relations=memories)
        ),
    )
    assert execution.issue is None
    expected = (
        bool(any(populations))
        if quantifier is Quantifier.EXISTS
        else not any(populations)
    )
    if compose:
        expected = expected and not subject_empty
    assert list(execution.fact_result.outcome.scalars.values()) == [expected]
