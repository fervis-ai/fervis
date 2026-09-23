"""An original supplied address can reveal a schema-free REST row set."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml
import pytest

from fervis.lookup.orchestration import semantic_compilation as shared
from fervis.lookup.orchestration.logical_compilation import compile_logical_question
from fervis.lookup.question_contract import QuestionContractRequest
from fervis.lookup.query_enrichment.semantic import (
    SemanticQueryEnrichmentResult, RecallBucketMatch, InputResourceSearchTerms,
)
from fervis.lookup.read_eligibility import (
    ReadRequirementAssessment, SemanticReadDecision, SemanticReadEligibilityResult,
)
from fervis.lookup.relation_catalog import CatalogParam, ParamSource, RelationCatalog
from fervis.lookup.turn_prompts import HostPromptContext
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize("identifier,address_type,parsed", [
    ("00000000-0000-0000-0000-000000000001", "uuid", "00000000-0000-0000-0000-000000000001"),
    ("42", "integer", 42),
    ("12.5", "number", 12.5),
    ("12.5", "decimal", "12.5"),
])
def test_schema_free_address_compiles_typed_count_from_original_input(
    monkeypatch, identifier, address_type, parsed,
):
    question = f"How many facilities are in {identifier}?"
    fixture_path = Path(__file__).resolve().parents[2] / (
        "conformance/cases/algorithms/semantic_kernel/"
        "identity_input_owns_target_set_and_direct_association.yaml"
    )
    fixture = yaml.safe_load(fixture_path.read_text())["input"]

    def replace_name(value):
        if isinstance(value, str):
            return value.replace("River District", identifier)
        if isinstance(value, list):
            return [replace_name(item) for item in value]
        if isinstance(value, dict):
            return {key: replace_name(item) for key, item in value.items()}
        return value

    frame = replace_name(fixture["frame_payload"])
    contract_payload = replace_name(fixture["payload"])
    seen = []

    def turn(purpose, *, prompt, parse, **kwargs):
        name = type(prompt).__name__
        seen.append(name)
        if name == "SemanticQuestionFrameTurnPrompt":
            result = parse(frame)
        elif name == "SemanticQuestionContractTurnPrompt":
            result = parse(contract_payload)
        elif name == "SemanticQueryEnrichmentTurnPrompt":
            result = SemanticQueryEnrichmentResult(tuple(
                RecallBucketMatch(bucket.bucket_ref, ("facilities",), ("facilities",))
                for bucket in prompt.request.recall_buckets
            ), tuple(InputResourceSearchTerms(task.input_use_ref, ("districts",))
                      for task in prompt.request.reference_tasks))
        elif name == "InspectionInputTurnPrompt":
            result = parse({"reads": {"facilities": {
                "kind": "bound_arguments",
                "mapping_basis": "The question supplies the district path UUID.",
                "parameter_inputs": {"district_id": "i1"},
            }}})
        elif name == "PaginationDiscoveryPrompt":
            result = parse({"reads": {read.id: {
                "mapping_basis": "The endpoint returns the complete addressed collection.",
                "mode": "single_response", "row_path_ref": None,
                "position_parameter_ref": None, "size_parameter_ref": None,
                "total_field_ref": None, "continuation_field_ref": None,
            } for read in prompt.request.targets}})
        elif name == "SemanticSourceRealizationTurnPrompt":
            branch = prompt.request.strategy.branches[0].branch_id
            source_ref = prompt.request.row_references_for_set("fact_1:set:s1")[0]
            address_use = prompt.request.index.input_use_sites[0].use_ref
            result = parse({
                "set_bindings": {
                    "fact_1:set:s1": [{"branch_id": branch,
                        "mapping_basis": "Addressed API rows are facilities.",
                        "rows_ref": source_ref, "record_fields": []}],
                    "fact_1:set:s2": [{"branch_id": branch,
                        "mapping_basis": "The supplied district scopes these rows.",
                        "rows_ref": source_ref, "record_fields": [],
                        "address_parameter_ref": "district_id",
                        "address_input_use_ref": address_use}],
                },
                "fact_bindings": {},
                "association_bindings": {"fact_1:association:a1": [{
                    "branch_id": branch,
                    "mapping_basis": "The addressed read establishes membership.",
                    "realization_ref": source_ref,
                    "reference_from_set_ref": None, "field_pairs": [],
                }]},
            })
        elif name == "SetPopulationTurnPrompt":
            branch = prompt.request.strategy.branches[0].branch_id
            result = parse({"populations": {"fact_1:set:s1": [{
                "branch_id": branch, "logical_set_meaning": "facilities",
                "mapping_basis": "The addressed rows are exactly the requested population.",
                "population": {"kind": "exact_population"},
            }]}})
        elif name == "SemanticSourceBindingTurnPrompt":
            branch = prompt.request.strategy.branches[0].branch_id
            applications = []
            for owner in prompt.request.invocation_application_owner_refs:
                option = prompt.request.invocation_options_for_owner(owner, branch_id=branch)[0]
                applications.append({
                    "kind": "request_application",
                    "mapping_basis": "The supplied district UUID owns the path argument.",
                    "owner_ref": owner, "value_ref": option.value_ref,
                    "value_component": option.projection.value,
                    "target_ref": option.target_ref,
                })
            result = parse({
                "resolved_input_applications": {branch: applications},
                "finite_choice_applications": {branch: {}},
                "choice_requirement_applications": {branch: {}},
            })
        else:
            raise AssertionError(name)
        return SimpleNamespace(result=result)

    def eligibility(eligibility_request, **kwargs):
        return SemanticReadEligibilityResult(tuple(
            ReadRequirementAssessment(
                "fact_1", source.read_id, (source.id,), source.read_id,
                tuple(field.field_ref for field in source.fields),
                "The read covers addressed facilities.", SemanticReadDecision.RETAIN,
            )
            for source in eligibility_request.source_catalog.sources if source.read_id
        ), ())

    monkeypatch.setattr(shared, "_turn", turn)
    monkeypatch.setattr(shared, "_read_eligibility_turn", eligibility)
    from fervis.lookup.source_reads.access_model import ReadAccessCatalog
    monkeypatch.setattr(shared, "_discover_read_access", lambda *args, **kwargs: ReadAccessCatalog())

    read = replace(
        _read("facilities"), path="/districts/{district_id}/facilities",
        resource_names=("facilities",), fields=(), row_paths=(), candidate_keys=(),
        params=(CatalogParam("district_id", "district_id", ParamSource.PATH,
                             address_type, required=True),),
        source_metadata={"representation_authority": "unobserved"},
    )
    reads = []

    class Port:
        def read(self, *, endpoint_name, args):
            reads.append((endpoint_name, args))
            assert args == {"district_id": parsed}
            return {"responseStatus": 200, "responseFormat": "json",
                    "responseBody": [{"id": 1}, {"id": 2}]}

    request = shared.SemanticCompilationRequest(
        "schema-free-address", question, QuestionContractRequest(question, {}),
        RelationCatalog(reads=(read,)), (), Port(), None, "openai", 1, 10,
        None, {}, HostPromptContext(),
    )
    result = compile_logical_question(request)
    assert isinstance(result, shared.SemanticCompilationSuccess)
    assert "InspectionInputTurnPrompt" in seen
    assert reads == [("facilities", {"district_id": parsed})]
    from fervis.lookup.answer_program.operations import SqlQuerySpec
    assert not any(isinstance(operation.spec, SqlQuerySpec)
                   for operation in result.compilation.answer_program.operations)
    from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program

    program = decode_answer_program(canonical_answer_program_json(
        result.compilation.answer_program
    ))
    executed = invoke_answer_program(
        program=program, bindings=result.compilation.initial_bindings,
        environment=ExecutionEnvironment(catalog=result.catalog_selection.relation_catalog),
        ports=RuntimePorts(Port(), LookupMemory()),
    )
    assert executed.issue is None
    assert next(iter(executed.fact_result.outcome.projected_rows[0].values.values())) == 2
    assert reads == [
        ("facilities", {"district_id": parsed}),
        ("facilities", {"district_id": parsed}),
    ]
    from fervis.lookup.orchestration.execution_sources import prepare_execution_catalog

    fresh_catalog = prepare_execution_catalog(
        run_id="schema-free-address-replay", catalog=RelationCatalog(reads=(read,)),
        program=program, bindings=result.compilation.initial_bindings,
        data_access_port=Port(), lineage_step_sink=None,
    )
    replay = invoke_answer_program(
        program=program, bindings=result.compilation.initial_bindings,
        environment=ExecutionEnvironment(catalog=fresh_catalog),
        ports=RuntimePorts(Port(), LookupMemory()),
    )
    assert replay.issue is None
    assert next(iter(replay.fact_result.outcome.projected_rows[0].values.values())) == 2
    assert len(reads) == 4
    from fervis.lookup.orchestration.logical_compilation import (
        _verify_compiled_inspection_addresses,
    )
    from fervis.lookup.source_reads.representation import inspection_request_fingerprint
    observed = result.catalog_selection.relation_catalog.read("facilities")
    wrong_observation = replace(observed, source_metadata={
        **(observed.source_metadata or {}),
        "observed_request_fingerprint": inspection_request_fingerprint({
            "district_id": "00000000-0000-0000-0000-000000000002"
        }),
    })
    with pytest.raises(ValueError, match="differs from inspected"):
        _verify_compiled_inspection_addresses(SimpleNamespace(
            full_catalog=RelationCatalog(reads=(wrong_observation,)),
        ), result.compilation)

    from fervis.lookup.answer_program.relations import EndpointParamBinding
    from fervis.lookup.answer_program.values import ConstantRef, FactValue, LiteralType

    optional = CatalogParam("representation", "representation", ParamSource.QUERY,
                            "string", required=False)
    observed = replace(observed, params=(*observed.params, optional))
    representation = FactValue.literal(
        id="compact", literal_type=LiteralType.STRING, value="compact"
    )
    relation = next(item for item in program.relations
                    if item.source.read_id == "facilities")
    changed_relation = replace(relation, source=replace(
        relation.source, param_bindings=(*relation.source.param_bindings,
            EndpointParamBinding("representation", ConstantRef(
                "compact", "optional-representation@1", representation
            ))),
    ))
    changed_program = replace(program, relations=tuple(
        changed_relation if item.id == relation.id else item
        for item in program.relations
    ))
    with pytest.raises(ValueError, match="differs from inspected"):
        _verify_compiled_inspection_addresses(SimpleNamespace(
            full_catalog=RelationCatalog(reads=(observed,)),
        ), SimpleNamespace(
            answer_program=changed_program,
            initial_bindings=result.compilation.initial_bindings,
        ))
