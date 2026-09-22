"""A one-row source can witness a referent value without becoming entity rows."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
from fervis.lookup.answer_program.operations import ReferenceGuardSpec, SqlQuerySpec
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.available_sources import SourceFieldBinding, snapshot_source_catalog
from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.orchestration.logical_planning import realize_and_compile_logical_plan
from fervis.lookup.question_contract import (
    QuestionContract, parse_semantic_question_contract, parse_semantic_question_frame,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.relation_catalog import CatalogField, RelationCatalog, RowCardinality
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from tests.lookup.relational_engine.test_dependent_reads import _read


def _logical():
    path = Path(__file__).resolve().parents[2] / (
        "conformance/cases/algorithms/semantic_kernel/"
        "identity_input_owns_target_set_and_direct_association.yaml"
    )
    fixture = yaml.safe_load(path.read_text())["input"]
    texts = tuple(fixture["question_context_texts"])
    meaning = parse_semantic_question_frame(fixture["frame_payload"], question_context_texts=texts)
    original = parse_semantic_question_contract(
        fixture["payload"], meaning=meaning, question_context_texts=texts
    )
    input_term = replace(original.contract.inputs[0], operand="the reporting district")
    denotation = replace(
        original.contract.input_denotations[0],
        reference_descriptions=("the reporting district",),
    )
    contract = QuestionContract(
        (input_term,), original.contract.requested_facts, (denotation,)
    )
    index = analyze_requested_fact(
        contract.requested_facts[0],
        inputs={input_term.id: input_term},
        input_denotations={denotation.input_ref: denotation},
    )
    return replace(original, contract=contract, semantic_indexes=(index,))


@pytest.mark.parametrize("wrong_join", [False, True])
def test_reference_proxy_uses_one_current_scalar_without_claiming_entity_rows(wrong_join):
    logical = _logical()
    (index,) = logical.semantic_indexes
    value = FactValue.named(
        id="reference:i1", known_input_id="i1", text="the reporting district",
        proof_refs=("question_input:i1",),
    )
    canonical = CanonicalInputValue(
        value.id, "i1", tuple(use.use_ref for use in index.input_use_sites),
        value, value.proof_refs,
    )
    facilities = replace(
        _read("facilities", value_type="string"),
        candidate_keys=(),
        resource_names=("facilities",),
        fields=(*_read("facilities", value_type="string").fields,
                CatalogField("facility.district", "string", path="district", row_path_id="root")),
    )
    settings_read = _read("settings", value_type="string")
    settings_read = replace(
        settings_read,
        candidate_keys=(),
        resource_names=("settings",),
        row_paths=tuple(replace(path, cardinality=RowCardinality.ONE)
                        for path in settings_read.row_paths),
        fields=(*settings_read.fields,
                CatalogField("settings.reporting_district_id", "string",
                             path="reporting_district_id", row_path_id="root")),
    )
    catalog = RelationCatalog(reads=(facilities, settings_read))
    sources = {source.read_id: source for source in build_api_row_source_catalog(catalog).sources}
    fields = {
        name: {field.path: SourceFieldBinding(source.id, field).ref for field in source.fields}
        for name, source in sources.items()
    }
    called = []

    def turn(purpose, prompt, parse):
        name = type(prompt).__name__
        called.append(name)
        branch = prompt.request.strategy.branches[0].branch_id
        if name == "SemanticSourceRealizationTurnPrompt":
            assert sources["settings"].id in prompt.request.row_references_for_set("fact_1:set:s2")
            return parse({
                "set_bindings": {
                    "fact_1:set:s1": [{
                        "branch_id": branch, "mapping_basis": "Facility rows are the counted unit.",
                        "rows_ref": sources["facilities"].id, "record_fields": [],
                    }],
                    "fact_1:set:s2": [{
                        "branch_id": branch,
                        "mapping_basis": "One settings row carries the reporting district value, not district rows.",
                        "rows_ref": sources["settings"].id,
                        "record_fields": [],
                        "reference_proxy_field_ref": fields["settings"]["reporting_district_id"],
                    }],
                },
                "fact_bindings": {},
                "association_bindings": {"fact_1:association:a1": [{
                    "branch_id": branch,
                    "mapping_basis": "Observed district values connect facilities to the selected setting.",
                    "realization_ref": None,
                    "reference_from_set_ref": None,
                    "field_pairs": [{
                        "from_field_ref": fields["facilities"]["district"],
                        "to_field_ref": fields["settings"][
                            "id" if wrong_join else "reporting_district_id"
                        ],
                    }],
                }]},
            })
        if name == "SetPopulationTurnPrompt":
            assert set(prompt.realization.set_bindings) == {"fact_1:set:s1", "fact_1:set:s2"}
            return parse({"populations": {"fact_1:set:s1": [{
                "branch_id": branch,
                "logical_set_meaning": "facilities",
                "mapping_basis": "Every facility row is a candidate.",
                "population": {"kind": "exact_population"},
            }]}})
        if name == "DescriptorReferenceTurnPrompt":
            from fervis.lookup.source_binding.reference_prompt import descriptor_tasks
            tasks = descriptor_tasks(prompt.realization)
            assert all(fields["settings"]["reporting_district_id"] in options
                       for _, _, _, options in tasks.values())
            return parse({"references": {
                ref: {
                    "mapping_basis": "The complete one-row setting projects the reporting district value.",
                    "field_ref": fields["settings"]["reporting_district_id"],
                    "choice_value": None,
                }
                for ref in tasks
            }})
        if name == "SemanticSourceBindingTurnPrompt":
            return parse({
                "resolved_input_applications": {branch: []},
                "finite_choice_applications": {branch: {}},
                "choice_requirement_applications": {branch: {}},
            })
        raise AssertionError(name)

    def compile_case():
        return realize_and_compile_logical_plan(
            logical,
            sources_by_fact={"fact_1": snapshot_source_catalog(tuple(sources.values()))},
            canonical_values=(canonical,),
            turn=turn,
        )

    if wrong_join:
        with pytest.raises(ValueError, match="Reference proxy association"):
            compile_case()
        return
    outcome = compile_case()
    from fervis.lookup.fact_compilation.model import FactCompilationResult
    assert isinstance(outcome, FactCompilationResult)
    program = decode_answer_program(canonical_answer_program_json(outcome.answer_program))
    assert any(isinstance(operation.spec, ReferenceGuardSpec) for operation in program.operations)
    assert not any(isinstance(operation.spec, SqlQuerySpec) for operation in program.operations)
    assert called == [
        "SemanticSourceRealizationTurnPrompt", "SetPopulationTurnPrompt",
        "DescriptorReferenceTurnPrompt", "SemanticSourceBindingTurnPrompt",
    ]

    class Port:
        def __init__(self, selected):
            self.selected = selected
            self.calls = []

        def read(self, *, endpoint_name, args):
            self.calls.append((endpoint_name, args))
            assert args == {}
            if endpoint_name == "settings":
                return {"responseStatus": 200, "responseBody": {
                    "id": "setting", "reporting_district_id": self.selected,
                }}
            return {"responseStatus": 200, "responseBody": [
                {"id": str(i), "district": "d1" if i < 2 else "d2"}
                for i in range(5)
            ]}

    for selected, expected in (("d1", 2), ("d2", 3)):
        port = Port(selected)
        result = invoke_answer_program(
            program=program,
            bindings=outcome.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(port, LookupMemory()),
        )
        assert result.issue is None
        assert next(iter(result.fact_result.outcome.projected_rows[0].values.values())) == expected
        assert {name for name, _ in port.calls} == {"settings", "facilities"}
