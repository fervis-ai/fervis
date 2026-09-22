"""An addressed collection can bind a supplied reference without a resolver API."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.available_sources import snapshot_source_catalog
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.orchestration.logical_planning import prepare_logical_realizations
from fervis.lookup.orchestration.logical_planning import realize_and_compile_logical_plan
from fervis.lookup.question_contract import (
    QuestionContract, parse_semantic_question_contract, parse_semantic_question_frame,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.relation_catalog import CatalogParam, ParamSource, RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize(
    ("identifier", "address_type"),
    [
        ("42", "integer"),
        ("00000000-0000-0000-0000-000000000001", "uuid"),
        ("12.5", "number"),
    ],
)
def test_addressed_rows_keep_the_identity_input_without_entity_key_authority(
    identifier, address_type
):
    path = Path(__file__).resolve().parents[2] / (
        "conformance/cases/algorithms/semantic_kernel/"
        "identity_input_owns_target_set_and_direct_association.yaml"
    )
    fixture = yaml.safe_load(path.read_text())["input"]
    texts = tuple(fixture["question_context_texts"])
    meaning = parse_semantic_question_frame(fixture["frame_payload"], question_context_texts=texts)
    parsed = parse_semantic_question_contract(
        fixture["payload"], meaning=meaning, question_context_texts=texts
    )
    input_term = replace(parsed.contract.inputs[0], operand=identifier)
    denotation = parsed.contract.input_denotations[0]
    contract = QuestionContract(
        (input_term,), parsed.contract.requested_facts, (denotation,)
    )
    index = analyze_requested_fact(
        contract.requested_facts[0],
        inputs={input_term.id: input_term},
        input_denotations={denotation.input_ref: denotation},
    )
    logical = replace(parsed, contract=contract, semantic_indexes=(index,))
    value = FactValue.named(
        id="address:i1", known_input_id="i1", text=identifier,
        proof_refs=("question_input:i1",),
    )
    canonical = CanonicalInputValue(
        value.id, "i1", tuple(use.use_ref for use in index.input_use_sites),
        value, value.proof_refs,
    )
    read = replace(
        _read("facilities"),
        candidate_keys=(),
        resource_names=("facilities",),
        params=(CatalogParam(
            "district_id", "district_id", ParamSource.PATH, address_type,
            required=True,
        ),),
    )
    sources = build_api_row_source_catalog(RelationCatalog(reads=(read,))).sources
    request, = prepare_logical_realizations(
        logical,
        sources_by_fact={"fact_1": snapshot_source_catalog(sources)},
        canonical_values=(canonical,),
    )
    options = request.address_reference_options()
    assert request.strategy.branches
    assert request.address_scope_options_for_set("fact_1:set:s2")
    assert any(
        option.parameter_ref == "district_id"
        for option in options
    )
    assert request.source_catalog.identity_evidence == ()
    assert request.canonical_values[0].input_ref == "i1"

    source_ref = sources[0].id
    address_use = index.input_use_sites[0].use_ref

    def turn(purpose, prompt, parse):
        name = type(prompt).__name__
        branch = prompt.request.strategy.branches[0].branch_id
        if name == "SemanticSourceRealizationTurnPrompt":
            return parse({
                "set_bindings": {
                    "fact_1:set:s1": [{
                        "branch_id": branch,
                        "mapping_basis": "Facility rows are the addressed population.",
                        "rows_ref": source_ref,
                        "record_fields": [],
                    }],
                    "fact_1:set:s2": [{
                        "branch_id": branch,
                        "mapping_basis": "The required path value scopes these rows to the supplied district.",
                        "rows_ref": source_ref,
                        "record_fields": [],
                        "address_parameter_ref": "district_id",
                        "address_input_use_ref": address_use,
                    }],
                },
                "fact_bindings": {},
                "association_bindings": {"fact_1:association:a1": [{
                    "branch_id": branch,
                    "mapping_basis": "The same addressed read proves facility membership.",
                    "realization_ref": source_ref,
                    "reference_from_set_ref": None,
                    "field_pairs": [],
                }]},
            })
        if name == "SetPopulationTurnPrompt":
            return parse({"populations": {"fact_1:set:s1": [{
                "branch_id": branch,
                "logical_set_meaning": "facilities",
                "mapping_basis": "The addressed rows are the facility candidates for this request.",
                "population": {"kind": "exact_population"},
            }]}})
        if name == "SemanticSourceBindingTurnPrompt":
            applications = []
            for owner in prompt.request.invocation_application_owner_refs:
                options = prompt.request.invocation_options_for_owner(
                    owner, branch_id=branch
                )
                for option in options[:1]:
                    applications.append({
                        "kind": "request_application",
                        "mapping_basis": "The required address parameter receives the supplied district literal.",
                        "owner_ref": owner,
                        "value_ref": option.value_ref,
                        "value_component": option.projection.value,
                        "target_ref": option.target_ref,
                    })
            return parse({
                "resolved_input_applications": {branch: applications},
                "finite_choice_applications": {branch: {}},
                "choice_requirement_applications": {branch: {}},
            })
        raise AssertionError(name)

    from fervis.lookup.fact_compilation.model import FactCompilationResult
    outcome = realize_and_compile_logical_plan(
        logical,
        sources_by_fact={"fact_1": snapshot_source_catalog(sources)},
        canonical_values=(canonical,),
        turn=turn,
    )
    assert isinstance(outcome, FactCompilationResult)
    from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
    from fervis.lookup.answer_program.operations import FilterSpec, SqlQuerySpec

    program = decode_answer_program(canonical_answer_program_json(outcome.answer_program))
    assert not any(isinstance(operation.spec, SqlQuerySpec) for operation in program.operations)
    assert not any(isinstance(operation.spec, FilterSpec) for operation in program.operations), [
        operation.spec for operation in program.operations
    ]

    class Port:
        def read(self, *, endpoint_name, args):
            assert endpoint_name == "facilities"
            assert str(args["district_id"]) == identifier
            return {"responseStatus": 200, "responseBody": [{"id": 1}, {"id": 2}]}

    result = invoke_answer_program(
        program=program,
        bindings=outcome.initial_bindings,
        environment=ExecutionEnvironment(catalog=RelationCatalog(reads=(read,))),
        ports=RuntimePorts(Port(), LookupMemory()),
    )
    assert result.issue is None
    assert next(iter(result.fact_result.outcome.projected_rows[0].values.values())) == 2
    assert any(parameter.fixed_value_fingerprint for parameter in program.parameters)
    from fervis.lookup.answer_program.values import BindingSet

    wrong = BindingSet.from_bindings(tuple(
        replace(
            binding,
            value=FactValue.named(
                id=binding.value.id,
                known_input_id="i1",
                text="43",
                proof_refs=binding.value.proof_refs,
            ),
        )
        if binding.parameter_id == program.parameters[0].id else binding
        for binding in outcome.initial_bindings.bindings
    ))
    with pytest.raises(ValueError, match="fixed|fingerprint|binding"):
        invoke_answer_program(
            program=program,
            bindings=wrong,
            environment=ExecutionEnvironment(catalog=RelationCatalog(reads=(read,))),
            ports=RuntimePorts(Port(), LookupMemory()),
        )
