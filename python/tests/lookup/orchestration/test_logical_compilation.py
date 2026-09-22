"""Typed orchestration preserves its specification through native REST replay."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fervis.lookup.orchestration import semantic_compilation as shared
from fervis.lookup.orchestration.logical_compilation import compile_logical_question
from fervis.lookup.question_contract import QuestionContractRequest
from fervis.lookup.turn_prompts import HostPromptContext
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.query_enrichment.semantic import (
    SemanticQueryEnrichmentResult,
    RecallBucketMatch,
)
from fervis.lookup.read_eligibility import (
    SemanticReadEligibilityResult,
    ReadRequirementAssessment,
    SemanticReadDecision,
)
from fervis.lookup.answer_program.operations import SqlQuerySpec
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.contract_codec import (
    canonical_answer_program_json,
    decode_answer_program,
)
from tests.lookup.question_contract.test_question_frame import _frame_payload
from tests.lookup.relational_engine.test_dependent_reads import _read


def count_payload():
    path = (
        Path(__file__).resolve().parents[2]
        / "conformance/cases/algorithms/semantic_kernel/distinct_set_refs_require_declared_association.yaml"
    )
    payload = deepcopy(yaml.safe_load(path.read_text())["input"]["payload"])
    fact = payload["outcome"]["answer_requests"][0]
    fact["origin"]["meaning"] = "store count"
    fact["candidate_set"]["instance_kind"] = "store"
    fact["outputs"]["requested_value_outputs"][0]["expression"]["argument"][
        "set_ref"
    ] = "s1"
    return payload


@pytest.mark.parametrize("outcome", ["normal", "empty", "dropped", "unavailable"])
@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_typed_runtime_authors_independent_contract_then_executes_native_count(
    monkeypatch, annotated, batched, outcome
):
    seen = []
    logical = []

    def turn(purpose, *, prompt, parse, **kwargs):
        name = type(prompt).__name__
        seen.append(name)
        if name == "SemanticQuestionFrameTurnPrompt":
            result = parse(_frame_payload())
        elif name == "SemanticQuestionContractTurnPrompt":
            result = parse(count_payload())
            logical.append(result.contract)
        elif name == "SemanticQueryEnrichmentTurnPrompt":
            result = SemanticQueryEnrichmentResult(
                tuple(
                    RecallBucketMatch(bucket.bucket_ref, ("stores",), ("stores",))
                    for bucket in prompt.request.recall_buckets
                ),
                (),
            )
        elif name == "PaginationDiscoveryPrompt":
            result = parse({"reads": {
                read.id: {
                    "mapping_basis": "The declared response is a complete collection.",
                    "mode": "single_response",
                    "row_path_ref": None,
                    "position_parameter_ref": None,
                    "size_parameter_ref": None,
                    "total_field_ref": None,
                    "continuation_field_ref": None,
                }
                for read in prompt.request.targets
            }})
        elif name == "SemanticSourceRealizationTurnPrompt":
            if outcome == "unavailable":
                return SimpleNamespace(
                    result=parse(
                        {
                            "kind": "unavailable_source_realization",
                            "unmet_requirement_refs": [
                                ref.token
                                for ref in prompt.request.index.source_requirement_refs
                            ],
                            "explanation": "The available rows do not establish the requested store population.",
                        }
                    )
                )
            branch = prompt.request.strategy.branches[0].branch_id
            ref = "fact_1:set:s1"
            rows_ref = prompt.request.row_references_for_set(ref)[0]
            result = parse(
                {
                    "set_bindings": {
                        ref: [
                            {
                                "branch_id": branch,
                                "mapping_basis": "The API rows establish store records.",
                                "rows_ref": rows_ref,
                                "record_fields": [],
                            }
                        ]
                    },
                    "fact_bindings": {},
                    "association_bindings": {},
                }
            )
        elif name == "SetPopulationTurnPrompt":
            branch = prompt.request.strategy.branches[0].branch_id
            result = parse(
                {
                    "populations": {
                        "fact_1:set:s1": [
                            {
                                "branch_id": branch,
                                "logical_set_meaning": "store",
                                "mapping_basis": "All rows are store records.",
                                "population": {"kind": "exact_population"},
                            }
                        ]
                    }
                }
            )
        else:
            raise AssertionError(name)
        return SimpleNamespace(result=result)

    assessed = []

    def eligibility(eligibility_request, **kwargs):
        candidates = tuple(
            source
            for source in eligibility_request.source_catalog.sources
            if source.read_id
        )
        assessed.extend(source.read_id for source in candidates)
        return SemanticReadEligibilityResult(
            tuple(
                ReadRequirementAssessment(
                    "fact_1",
                    source.read_id,
                    (source.id,),
                    source.read_id,
                    tuple(field.field_ref for field in source.fields),
                    "The rows supply store records.",
                    SemanticReadDecision.DROP
                    if outcome == "dropped"
                    else SemanticReadDecision.RETAIN,
                )
                for source in candidates
            ),
            (),
        )

    monkeypatch.setattr(shared, "_turn", turn)
    monkeypatch.setattr(shared, "_read_eligibility_turn", eligibility)
    reads = []

    class Port:
        def read(self, *, endpoint_name, args):
            reads.append((endpoint_name, args))
            return {
                "responseStatus": 200,
                "responseBody": [{"id": 1}, {"id": 2}, {"id": 3}],
            }

    catalog = RelationCatalog(
        reads=tuple(
            replace(
                _read(name),
                resource_names=("stores",),
                **({} if annotated else {"candidate_keys": ()}),
            )
            for name in (("stores", "more_stores") if batched else ("stores",))
        )
    )
    if outcome == "empty":
        catalog = RelationCatalog()
    request = shared.SemanticCompilationRequest(
        "typed-runtime",
        "How many stores?",
        QuestionContractRequest("How many stores?", {}),
        catalog,
        (),
        Port(),
        None,
        "openai",
        1,
        1 if batched else 10,
        None,
        {},
        HostPromptContext(),
    )
    result = compile_logical_question(request)
    if outcome != "normal":
        assert isinstance(result, shared.SemanticCompilationImpossible)
        assert result.question_contract is logical[0]
        assert result.blocked_fact_ids == ("fact_1",)
        assert reads == []
        assert seen == [
            "SemanticQuestionFrameTurnPrompt",
            "SemanticQuestionContractTurnPrompt",
            "SemanticQueryEnrichmentTurnPrompt",
        ] + (
            ["SemanticSourceRealizationTurnPrompt"] if outcome == "unavailable" else []
        )
        return
    assert isinstance(result, shared.SemanticCompilationSuccess)
    assert result.question_contract is logical[0]
    assert set(assessed) == {read.id for read in catalog.reads}
    assert seen == [
        "SemanticQuestionFrameTurnPrompt",
        "SemanticQuestionContractTurnPrompt",
        "SemanticQueryEnrichmentTurnPrompt",
        "SemanticSourceRealizationTurnPrompt",
        "SetPopulationTurnPrompt",
    ]
    program = decode_answer_program(
        canonical_answer_program_json(result.compilation.answer_program)
    )
    assert program.fact_template == logical[0].requested_facts
    assert not any(
        isinstance(operation.spec, SqlQuerySpec) for operation in program.operations
    )
    assert reads == []
    executed = invoke_answer_program(
        program=program,
        bindings=result.compilation.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(Port(), LookupMemory()),
    )
    assert executed.issue is None
    assert (
        next(iter(executed.fact_result.outcome.projected_rows[0].values.values())) == 3
    )
    assert len(reads) == 1


def test_two_requested_facts_keep_distinct_outputs_through_one_rest_read(monkeypatch):
    frame = _frame_payload()
    frame["outcome"]["answer_requests"].append(
        deepcopy(frame["outcome"]["answer_requests"][0])
    )
    contract = count_payload()
    second = deepcopy(contract["outcome"]["answer_requests"][0])
    second["requested_fact_ref"] = "fact_2"
    contract["outcome"]["answer_requests"].append(second)
    authored = []

    def turn(purpose, *, prompt, parse, **kwargs):
        name = type(prompt).__name__
        if name == "SemanticQuestionFrameTurnPrompt":
            result = parse(frame)
        elif name == "SemanticQuestionContractTurnPrompt":
            result = parse(contract)
            authored.append(result.contract)
        elif name == "SemanticQueryEnrichmentTurnPrompt":
            result = SemanticQueryEnrichmentResult(
                tuple(
                    RecallBucketMatch(bucket.bucket_ref, ("stores",), ("stores",))
                    for bucket in prompt.request.recall_buckets
                ),
                (),
            )
        elif name == "PaginationDiscoveryPrompt":
            result = parse({"reads": {
                read.id: {
                    "mapping_basis": "The declared response is a complete collection.",
                    "mode": "single_response",
                    "row_path_ref": None,
                    "position_parameter_ref": None,
                    "size_parameter_ref": None,
                    "total_field_ref": None,
                    "continuation_field_ref": None,
                }
                for read in prompt.request.targets
            }})
        elif name == "SemanticSourceRealizationTurnPrompt":
            fact_id = prompt.request.index.requested_fact_id
            branch = prompt.request.strategy.branches[0].branch_id
            set_ref = f"{fact_id}:set:s1"
            result = parse({
                "set_bindings": {set_ref: [{
                    "branch_id": branch,
                    "mapping_basis": "The REST rows are the store population.",
                    "rows_ref": prompt.request.row_references_for_set(set_ref)[0],
                    "record_fields": [],
                }]},
                "fact_bindings": {},
                "association_bindings": {},
            })
        elif name == "SetPopulationTurnPrompt":
            fact_id = prompt.request.index.requested_fact_id
            branch = prompt.request.strategy.branches[0].branch_id
            result = parse({"populations": {f"{fact_id}:set:s1": [{
                "branch_id": branch,
                "logical_set_meaning": "store",
                "mapping_basis": "The source lists all stores.",
                "population": {"kind": "exact_population"},
            }]}})
        else:
            raise AssertionError(name)
        return SimpleNamespace(result=result)

    def eligibility(eligibility_request, **kwargs):
        return SemanticReadEligibilityResult(
            tuple(
                ReadRequirementAssessment(
                    index.requested_fact_id,
                    source.read_id,
                    (source.id,),
                    source.read_id,
                    tuple(field.field_ref for field in source.fields),
                    "Store rows provide the population for this fact.",
                    SemanticReadDecision.RETAIN,
                )
                for index in eligibility_request.indexes
                for source in eligibility_request.source_catalog.sources
                if source.read_id
            ),
            (),
        )

    monkeypatch.setattr(shared, "_turn", turn)
    monkeypatch.setattr(shared, "_read_eligibility_turn", eligibility)
    reads = []

    class Port:
        def read(self, *, endpoint_name, args):
            reads.append((endpoint_name, args))
            return {"responseStatus": 200, "responseBody": [{"id": 1}, {"id": 2}]}

    catalog = RelationCatalog(reads=(replace(_read("stores"), resource_names=("stores",)),))
    question = "How many stores?"
    request = shared.SemanticCompilationRequest(
        "typed-two-facts", question, QuestionContractRequest(question, {}),
        catalog, (), Port(), None, "openai", 1, 10, None, {}, HostPromptContext(),
    )
    result = compile_logical_question(request)
    assert isinstance(result, shared.SemanticCompilationSuccess)
    assert result.question_contract is authored[0]
    program = decode_answer_program(canonical_answer_program_json(result.compilation.answer_program))
    assert [fact.id for fact in program.fact_template] == ["fact_1", "fact_2"]
    assert not any(isinstance(operation.spec, SqlQuerySpec) for operation in program.operations)
    executed = invoke_answer_program(
        program=program,
        bindings=result.compilation.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(Port(), LookupMemory()),
    )
    assert executed.issue is None
    assert [
        next(iter(row.values.values()))
        for row in executed.fact_result.outcome.projected_rows
    ] == [2, 2]
    assert reads
