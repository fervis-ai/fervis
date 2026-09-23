from dataclasses import replace
import pytest

from tests.lookup.fact_compilation.test_compiler import _compile_memory_count
from fervis.lookup.question_contract.model import (
    FactTerm,
    RequestedOutput,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.semantic_types import TextType
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.source_binding.model import (
    FactRealization,
    FactRealizationKind,
    SourceRealization,
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


@pytest.mark.parametrize(("position", "expected"), [(1, ["c"]), (2, ["b"]), (4, [])])
def test_position_selection_compiles_and_invokes_with_a_typed_parameter(
    position, expected
):
    from fervis.lookup.question_contract.model import (
        PositionWithTies,
        Ordering,
        OrderingDirection,
        InputTerm,
        InputDenotation,
        InputDenotationKind,
    )
    from fervis.lookup.semantic_types import IntegerType
    from fervis.lookup.grounding.semantic import (
        deterministic_scalar_values,
        grounding_partitions,
    )

    _, _, memory, _, _, original = _compile_memory_count(
        ({"event_id": "a"}, {"event_id": "c"}, {"event_id": "b"})
    )
    memory = replace(
        memory, field_types={"event_id": "string"}, grain_keys=("event_id",)
    )
    fact = original.request.index.requested_fact
    fact = replace(
        fact,
        facts=(FactTerm("identifier", "s1", TextType(), fact.origin),),
        expressions=(),
        grouping_refs=("identifier",),
        outputs=(RequestedOutput("value", "identifier", fact.origin),),
        ordering=(Ordering("identifier", OrderingDirection.DESCENDING, fact.origin),),
        selection=PositionWithTies("position"),
    )
    term = InputTerm("position", fact.origin, str(position), IntegerType())
    inputs = {term.id: term}
    denotation = InputDenotation(
        "position_meaning",
        term.id,
        "requested position",
        "Explicit one-based ordinal.",
        None,
        InputDenotationKind.NON_IDENTITY_SCALAR,
    )
    index = analyze_requested_fact(
        fact, inputs=inputs, input_denotations={term.id: denotation}
    )
    canonical = deterministic_scalar_values(
        grounding_partitions(index.input_use_sites), inputs=inputs
    )
    source = next(
        s
        for s in build_row_source_catalog(
            RelationCatalog(), memory_relations=(memory,)
        ).sources
        if s.memory_ref == memory.id
    )
    request = replace(
        original.request,
        index=index,
        canonical_values=canonical,
        source_catalog=replace(original.request.source_catalog, sources=(source,)),
    )
    branch = request.strategy.branches[0].branch_id
    realization = SourceRealization(
        request,
        original.binding_plan.set_bindings,
        {
            index.fact_local_ref_by_local_id["identifier"].token: (
                FactRealization(
                    branch,
                    "Observed identifier.",
                    source.id,
                    FactRealizationKind.RETURNED_FIELD,
                    None,
                    ("event_id",),
                    (source.id, "event_id"),
                ),
            )
        },
        {},
    )
    membership = realization
    parsed = compile_source_binding_plan(
        {
            "resolved_input_applications": {branch: []},
            "finite_choice_applications": {branch: {}},
            "choice_requirement_applications": {branch: {}},
        },
        realization=membership,
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
    if not expected:
        from fervis.lookup.outcomes.model import NoData

        assert isinstance(result.fact_result.outcome, NoData)
    else:
        assert [
            next(iter(row.values.values()))
            for row in result.fact_result.outcome.projected_rows
        ] == expected
