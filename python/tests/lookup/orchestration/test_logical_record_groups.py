"""Related observed rows can own group identities without nominal keys."""

from dataclasses import replace

import pytest

from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.question_contract.model import (
    RequestedFact,
    SetTerm,
    AssociationTerm,
    FactTerm,
    Aggregate,
    AggregateFunction,
    RequestedOutput,
    Subject,
    InstanceInterpretation,
    Ordering,
    OrderingDirection,
    FirstRankWithTies,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, IdentifierType
from fervis.lookup.relation_catalog import RelationCatalog, CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.available_sources import snapshot_source_catalog, SourceFieldBinding
from fervis.lookup.orchestration.logical_planning import (
    realize_and_compile_logical_plan,
)
from fervis.lookup.fact_compilation.model import FactCompilationResult
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.contract_codec import (
    canonical_answer_program_json,
    decode_answer_program,
)
from fervis.lookup.memory.projection import LookupMemory
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize("duplicate_parent", [False, True])
def test_rank_groups_by_related_observed_record_occurrences(duplicate_parent):
    origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT, "Parent with the most items"
    )
    fact = RequestedFact(
        "fact_1",
        origin,
        (SetTerm("item", origin), SetTerm("parent", origin)),
        (AssociationTerm("ownership", "item", "parent", origin),),
        (FactTerm("parent_ref", "ownership", IdentifierType("parent"), origin),),
        (Aggregate("n", AggregateFunction.COUNT, "item", None, False, origin),),
        Subject("item", InstanceInterpretation.RESOURCE_POPULATION),
        None,
        ("parent_ref",),
        (RequestedOutput("parent", "parent_ref", origin),),
        (Ordering("n", OrderingDirection.DESCENDING, origin),),
        FirstRankWithTies(),
        (),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    logical = ParsedSemanticQuestionContract(
        "Parent with most items.", QuestionContract((), (fact,)), (index,)
    )
    items = replace(
        _read("items", value_type="string"),
        candidate_keys=(),
        fields=(
            *_read("items", value_type="string").fields,
            CatalogField(
                "items.parent_id", "string", path="parent_id", row_path_id="root"
            ),
        ),
    )
    parents = replace(_read("parents", value_type="string"), candidate_keys=())
    catalog = RelationCatalog(reads=(items, parents))
    sources = {
        source.read_id: source
        for source in build_api_row_source_catalog(catalog).sources
    }
    fields = {
        name: {
            field.path: SourceFieldBinding(source.id, field).ref
            for field in source.fields
        }
        for name, source in sources.items()
    }
    turns = []

    def turn(purpose, prompt, parse):
        turns.append(prompt)
        branch = prompt.request.strategy.branches[0].branch_id
        if len(turns) == 1:
            return parse(
                {
                    "set_bindings": {
                        f"fact_1:set:{role}": [
                            {
                                "branch_id": branch,
                                "mapping_basis": "Observed record occurrences.",
                                "rows_ref": sources[name].id,
                                "record_fields": (
                                    [{"name": "id", "field_ref": fields[name]["id"]}]
                                    if role == "parent"
                                    else []
                                ),
                            }
                        ]
                        for role, name in (("item", "items"), ("parent", "parents"))
                    },
                    "fact_bindings": {},
                    "association_bindings": {
                        "fact_1:association:ownership": [
                            {
                                "branch_id": branch,
                                "mapping_basis": "Match the observed parent property to the associated parent row.",
                                "realization_ref": None,
                                "reference_from_set_ref": None,
                                "field_pairs": [
                                    {
                                        "from_field_ref": fields["items"]["parent_id"],
                                        "to_field_ref": fields["parents"]["id"],
                                    }
                                ],
                            }
                        ]
                    },
                }
            )
        return parse(
            {
                "populations": {
                    f"fact_1:set:{role}": [
                        {
                            "branch_id": branch,
                            "logical_set_meaning": origin.meaning,
                            "mapping_basis": "All declared records.",
                            "population": {"kind": "exact_population"},
                        }
                    ]
                    for role in ("item", "parent")
                }
            }
        )

    result = realize_and_compile_logical_plan(
        logical,
        sources_by_fact={"fact_1": snapshot_source_catalog(tuple(sources.values()))},
        canonical_values=(),
        turn=turn,
    )
    assert isinstance(result, FactCompilationResult)
    program = decode_answer_program(
        canonical_answer_program_json(result.answer_program)
    )

    class Port:
        def read(self, *, endpoint_name, args):
            rows = (
                [{"id": "p"}, {"id": "q"}] + ([{"id": "p"}] if duplicate_parent else [])
                if endpoint_name == "parents"
                else [
                    {"id": str(i), "parent_id": "p" if i < 3 else "q"} for i in range(5)
                ]
            )
            return {"responseStatus": 200, "responseBody": rows}

    executed = invoke_answer_program(
        program=program,
        bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(Port(), LookupMemory()),
    )
    assert executed.issue is None
    assert [
        next(iter(row.values.values()))
        for row in executed.fact_result.outcome.projected_rows
    ] == [{"id": "p"}] * (2 if duplicate_parent else 1)
