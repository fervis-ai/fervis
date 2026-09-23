"""Each named member of an anonymous collection resolves under its own guard."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
from fervis.lookup.answer_program.operations import ReferenceGuardSpec, SqlQuerySpec
from fervis.lookup.answer_program.values import BindingSet, FactValue
from fervis.lookup.available_sources import SourceFieldBinding, snapshot_source_catalog
from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
from fervis.lookup.expression_operators import ExpressionBinaryOperator
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.orchestration.logical_planning import realize_and_compile_logical_plan
from fervis.lookup.question_contract import (
    QuestionContract, parse_semantic_question_contract, parse_semantic_question_frame,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import Comparison
from fervis.lookup.relation_catalog import CatalogField, RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.semantic_types import CollectionType, TextType
from fervis.lookup.canonical_data import EntityKeyValue, EntityKeyComponentValue
from fervis.lookup.source_binding.reference_bindings import SelectedReferenceChoice
from fervis.lookup.clarification.model import ClarificationOption, GroundingIdentityResponse
from fervis.lookup.orchestration.logical_compilation import _selected_reference_choices
from tests.lookup.relational_engine.test_dependent_reads import _read


def test_member_clarification_enters_typed_reference_choice_with_user_proof():
    key = EntityKeyValue("districts", "primary", (EntityKeyComponentValue("id", "d3"),))
    response = GroundingIdentityResponse(
        "choose_lake_d3", "clarify_i1", "fact_1", "i1",
        ClarificationOption("d3", key=key), "Lake District",
    )
    assert _selected_reference_choices((response,)) == (
        SelectedReferenceChoice(
            "fact_1", "i1", "Lake District", key,
            "clarification_response:choose_lake_d3",
        ),
    )
    with pytest.raises(ValueError, match="key or observed properties"):
        _selected_reference_choices((replace(response, option=ClarificationOption("none")),))


@pytest.mark.parametrize("data_case", ["distinct", "same_entity", "missing", "selected_member", "selected_observed_member"])
def test_named_collection_rechecks_each_member_and_counts_once(data_case):
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
    input_term = replace(
        parsed.contract.inputs[0], operand=("River District", "Lake District"),
        value_type=CollectionType(TextType()),
    )
    denotation = parsed.contract.input_denotations[0]
    original = parsed.contract.requested_facts[0]
    fact = replace(
        original,
        expressions=tuple(
            replace(node, operator=ExpressionBinaryOperator.IN)
            if isinstance(node, Comparison) and node.right_ref == "i1"
            else node
            for node in original.expressions
        ),
    )
    contract = QuestionContract((input_term,), (fact,), (denotation,))
    index = analyze_requested_fact(
        fact,
        inputs={input_term.id: input_term},
        input_denotations={denotation.input_ref: denotation},
    )
    logical = replace(parsed, contract=contract, semantic_indexes=(index,))
    value = FactValue.string_set(
        id="reference_names:i1", known_input_id="i1", values=input_term.operand,
        proof_refs=("question_input:i1",),
    )
    canonical = CanonicalInputValue(
        value.id, "i1", tuple(use.use_ref for use in index.input_use_sites),
        value, value.proof_refs,
    )
    facilities = replace(
        _read("facilities", value_type="string"), candidate_keys=(),
        fields=(*_read("facilities", value_type="string").fields,
                CatalogField("district", "string", path="district", row_path_id="root")),
    )
    districts = replace(
        _read("districts", value_type="string"),
        candidate_keys=() if data_case != "selected_member" else _read("districts", value_type="string").candidate_keys,
        fields=(*_read("districts", value_type="string").fields,
                CatalogField("name", "string", path="name", row_path_id="root"),
                CatalogField("alias", "string", path="alias", row_path_id="root", nullable=True),
                CatalogField("region", "string", path="region", row_path_id="root")),
    )
    catalog = RelationCatalog(reads=(facilities, districts))
    sources = {source.read_id: source for source in build_api_row_source_catalog(catalog).sources}
    available = snapshot_source_catalog(tuple(sources.values()))
    district_identity = next(
        (item.identity_ref for item in available.identity_evidence if item.source_ref == sources["districts"].id),
        None,
    )
    fields = {
        name: {field.path: SourceFieldBinding(source.id, field).ref for field in source.fields}
        for name, source in sources.items()
    }

    def turn(purpose, prompt, parse):
        name = type(prompt).__name__
        branch = prompt.request.strategy.branches[0].branch_id
        if name == "SemanticSourceRealizationTurnPrompt":
            return parse({
                "set_bindings": {
                    f"fact_1:set:{set_id}": [{
                        "branch_id": branch, "mapping_basis": "Complete source records.",
                        "rows_ref": district_identity if set_id == "s2" and district_identity else sources[source_name].id,
                        "record_fields": [],
                    }]
                    for set_id, source_name in (("s1", "facilities"), ("s2", "districts"))
                },
                "fact_bindings": {},
                "association_bindings": {"fact_1:association:a1": [{
                    "branch_id": branch,
                    "mapping_basis": "Observed district values connect facility and district records.",
                    "realization_ref": None,
                    "reference_from_set_ref": None,
                    "field_pairs": [{
                        "from_field_ref": fields["facilities"]["district"],
                        "to_field_ref": fields["districts"]["id"],
                    }],
                }]},
            })
        if name == "SetPopulationTurnPrompt":
            return parse({"populations": {
                f"fact_1:set:{set_id}": [{
                    "branch_id": branch,
                    "logical_set_meaning": index.term_by_ref[next(
                        ref for ref in index.term_by_ref if ref.token == f"fact_1:set:{set_id}"
                    )].origin.meaning,
                    "mapping_basis": "All returned records are in this source set.",
                    "population": {"kind": "exact_population"},
                }]
                for set_id in ("s1", "s2")
            }})
        if name == "LiteralReferenceTurnPrompt":
            from fervis.lookup.source_binding.reference_prompt import reference_tasks

            tasks = reference_tasks(prompt.realization)
            assert len(tasks) == 2
            return parse({"references": {
                ref: {
                    "mapping_basis": "Name and alias are observed identifying properties.",
                    "field_refs": [fields["districts"]["name"], fields["districts"]["alias"]],
                }
                for ref in tasks
            }})
        if name == "SemanticSourceBindingTurnPrompt":
            request = prompt.request
            return parse({
                "resolved_input_applications": {branch: [
                    {
                        "kind": "no_request_application",
                        "mapping_basis": "Resolve each name against current district records.",
                        "owner_ref": owner,
                        "value_ref": value_ref,
                    }
                    for owner in request.invocation_application_owner_refs
                    for value_ref in request.unapplied_input_value_refs_for_owner(
                        owner, branch_id=branch
                    )
                ]},
                "finite_choice_applications": {branch: {}},
                "choice_requirement_applications": {branch: {}},
            })
        raise AssertionError(name)

    from fervis.lookup.fact_compilation.model import FactCompilationResult
    if data_case == "distinct":
        altered = replace(canonical, typed_value=FactValue.string_set(
            id=value.id, known_input_id="i1",
            values=("River District", "Other District"),
            proof_refs=value.proof_refs,
        ))
        with pytest.raises(ValueError, match="reference collection differs"):
            realize_and_compile_logical_plan(
                logical,
                sources_by_fact={"fact_1": snapshot_source_catalog(tuple(sources.values()))},
                canonical_values=(altered,),
                turn=turn,
            )
    selected = (
        (SelectedReferenceChoice(
            "fact_1", "i1", "Lake District",
            EntityKeyValue("districts", "primary", (EntityKeyComponentValue("id", "d3"),)),
            "clarification_response:choose_lake_d3",
        ),)
        if data_case == "selected_member" else ()
    )
    if data_case == "selected_observed_member":
        from fervis.lookup.identity_types import ObservedReferenceValue
        selected = (SelectedReferenceChoice(
            "fact_1", "i1", "Lake District", None,
            "clarification_response:choose_lake_west",
            observed_source_ref=sources["districts"].id,
            observed_properties=(ObservedReferenceValue(
                next(field.field_ref for field in sources["districts"].fields
                     if field.path == "region"),
                "string", "region", "West",
            ),),
        ),)
    if data_case == "selected_member":
        from fervis.lookup.source_binding.verification import SourceStrategyVerificationFailure

        bad = realize_and_compile_logical_plan(
            logical, sources_by_fact={"fact_1": available},
            canonical_values=(canonical,),
            selected_reference_choices=(replace(
                selected[0], key=replace(selected[0].key, entity_kind="other")
            ),), turn=turn,
        )
        assert isinstance(bad, SourceStrategyVerificationFailure)
    compiled = realize_and_compile_logical_plan(
        logical,
        sources_by_fact={"fact_1": available},
        canonical_values=(canonical,),
        selected_reference_choices=selected,
        turn=turn,
    )
    assert isinstance(compiled, FactCompilationResult)
    program = decode_answer_program(canonical_answer_program_json(compiled.answer_program))
    assert not any(isinstance(op.spec, SqlQuerySpec) for op in program.operations)
    assert sum(isinstance(op.spec, ReferenceGuardSpec) for op in program.operations) == 2
    calls = []
    drop_chosen = False
    change_chosen_property = False
    change_chosen_name = False

    class Port:
        def read(self, *, endpoint_name, args):
            calls.append(endpoint_name)
            assert args == {}
            if endpoint_name == "districts":
                return {"responseStatus": 200, "responseBody": [
                    {"id": "d1", "name": "River District",
                     "alias": "Lake District" if data_case == "same_entity" else None,
                     "region": "North"},
                    {"id": "d2", "name": "Other" if data_case in {"same_entity", "missing"} else "Lake District",
                     "alias": None, "region": "East"},
                    *([{"id": "d3",
                        "name": "Other District" if change_chosen_name else "Lake District",
                        "alias": None,
                        "region": "South" if change_chosen_property else "West"}]
                      if data_case in {"selected_member", "selected_observed_member"}
                      and not drop_chosen else []),
                ]}
            return {"responseStatus": 200, "responseBody": [
                {"id": str(i), "district": "d1" if i < 2 else "d2" if i < 5 else "d3"}
                for i in range(9 if data_case in {"selected_member", "selected_observed_member"} else 5)
            ]}

    if data_case == "selected_member":
        from fervis.lookup.identity_types import IdentityExecutionFailureReason

        unselected = realize_and_compile_logical_plan(
            logical, sources_by_fact={"fact_1": available},
            canonical_values=(canonical,), turn=turn,
        )
        assert isinstance(unselected, FactCompilationResult)
        ambiguous = invoke_answer_program(
            program=unselected.answer_program,
            bindings=unselected.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(), LookupMemory()),
        )
        assert ambiguous.issue.reference.reason is IdentityExecutionFailureReason.AMBIGUOUS_RESULT
        assert {key.component_value("id") for key in ambiguous.issue.reference.candidates} == {"d2", "d3"}
    if data_case == "selected_observed_member":
        unselected = realize_and_compile_logical_plan(
            logical, sources_by_fact={"fact_1": available},
            canonical_values=(canonical,), turn=turn,
        )
        assert isinstance(unselected, FactCompilationResult)
        ambiguous = invoke_answer_program(
            program=unselected.answer_program,
            bindings=unselected.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(), LookupMemory()),
        )
        assert len(ambiguous.issue.reference.observed_candidates) == 2
        assert ambiguous.issue.reference.candidates == ()

    result = invoke_answer_program(
        program=program,
        bindings=compiled.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(Port(), LookupMemory()),
    )
    if data_case == "missing":
        assert result.issue.reference.operand == "Lake District"
    else:
        assert result.issue is None
        expected = 6 if data_case in {"selected_member", "selected_observed_member"} else 5 if data_case == "distinct" else 2
        assert next(iter(result.fact_result.outcome.projected_rows[0].values.values())) == expected
    if data_case in {"selected_member", "selected_observed_member"}:
        drop_chosen = True
        missing_choice = invoke_answer_program(
            program=program, bindings=compiled.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(), LookupMemory()),
        )
        assert missing_choice.issue.reference.operand == "Lake District"
        drop_chosen = False
    if data_case == "selected_observed_member":
        for change in ("property", "name"):
            change_chosen_property = change == "property"
            change_chosen_name = change == "name"
            changed_choice = invoke_answer_program(
                program=program, bindings=compiled.initial_bindings,
                environment=ExecutionEnvironment(catalog=catalog),
                ports=RuntimePorts(Port(), LookupMemory()),
            )
            assert changed_choice.issue.reference.operand == "Lake District"
        change_chosen_property = change_chosen_name = False
    assert calls
    replacement = FactValue.string_set(
        id=value.id, known_input_id="i1",
        values=("River District", "Other District"),
        proof_refs=value.proof_refs,
    )
    changed = BindingSet.from_bindings(tuple(
        replace(item, value=replacement)
        if item.parameter_id == program.parameters[0].id else item
        for item in compiled.initial_bindings.bindings
    ))
    before = len(calls)
    with pytest.raises(ValueError, match="fixed|fingerprint|binding"):
        invoke_answer_program(
            program=program, bindings=changed,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(), LookupMemory()),
        )
    assert len(calls) == before
