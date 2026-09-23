"""Resolve an anonymous reference on its bound carrier before counting related rows."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fervis.lookup.question_contract import (
    parse_semantic_question_frame,
    parse_semantic_question_contract,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.relation_catalog import (
    RelationCatalog,
    CatalogField,
    CatalogParam,
    ParamSource,
)
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.available_sources import snapshot_source_catalog, SourceFieldBinding
from fervis.lookup.orchestration.logical_planning import (
    realize_and_compile_logical_plan,
)
from fervis.lookup.fact_compilation.model import FactCompilationResult
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.contract_codec import (
    canonical_answer_program_json,
    decode_answer_program,
)
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize(
    ("use_runtime", "descriptor"),
    [(False, False), (True, False), (False, True)],
)
@pytest.mark.parametrize("duplicate_name", [False, True])
def test_literal_reference_is_guarded_before_related_count_and_recomputed_on_replay(
    duplicate_name, use_runtime, descriptor, monkeypatch
):
    path = (
        Path(__file__).resolve().parents[2]
        / "conformance/cases/algorithms/semantic_kernel/identity_input_owns_target_set_and_direct_association.yaml"
    )
    fixture = yaml.safe_load(path.read_text())["input"]
    meaning = parse_semantic_question_frame(
        fixture["frame_payload"],
        question_context_texts=tuple(fixture["question_context_texts"]),
    )
    logical = parse_semantic_question_contract(
        fixture["payload"],
        meaning=meaning,
        question_context_texts=tuple(fixture["question_context_texts"]),
    )
    if descriptor:
        from fervis.lookup.question_contract import QuestionContract
        from fervis.lookup.question_contract.analysis import analyze_requested_fact

        input_term = replace(logical.contract.inputs[0], operand="the default district")
        denotation = replace(
            logical.contract.input_denotations[0],
            reference_descriptions=("the default district",),
        )
        contract = QuestionContract(
            (input_term,), logical.contract.requested_facts, (denotation,)
        )
        logical = replace(
            logical,
            contract=contract,
            semantic_indexes=(analyze_requested_fact(
                contract.requested_facts[0],
                inputs={input_term.id: input_term},
                input_denotations={denotation.input_ref: denotation},
            ),),
        )
    (index,) = logical.semantic_indexes
    value = FactValue.named(
        id="reference_text:i1",
        known_input_id="i1",
        text=logical.contract.inputs[0].operand,
        proof_refs=("question_input:i1",),
    )
    canonical = CanonicalInputValue(
        value.id,
        "i1",
        tuple(use.use_ref for use in index.input_use_sites),
        value,
        value.proof_refs,
    )
    facilities = replace(
        _read("facilities", value_type="string"),
        candidate_keys=(),
        resource_names=("facilities",),
        params=(
            CatalogParam(
                "search",
                "search",
                ParamSource.QUERY,
                "string",
                description="Search facility labels.",
            ),
        ),
        fields=(
            *_read("facilities", value_type="string").fields,
            CatalogField(
                "facilities.district", "string", path="district", row_path_id="root"
            ),
        ),
    )
    districts = replace(
        _read("districts", value_type="string"),
        candidate_keys=(),
        resource_names=("districts",),
        fields=(
            *_read("districts", value_type="string").fields,
            CatalogField("districts.name", "string", path="name", row_path_id="root"),
            CatalogField("districts.is_default", "boolean", path="is_default", row_path_id="root"),
            CatalogField("districts.nickname", "string", path="nickname",
                         row_path_id="root", nullable=True),
        ),
    )
    catalog = RelationCatalog(reads=(facilities, districts))
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
    calls = []

    def turn(purpose, prompt, parse):
        name = type(prompt).__name__
        calls.append(name)
        if name == "PaginationDiscoveryPrompt":
            return parse({"reads": {
                read.id: {
                    "mapping_basis": "The declared collection response is complete.",
                    "mode": "single_response",
                    "row_path_ref": None,
                    "position_parameter_ref": None,
                    "size_parameter_ref": None,
                    "total_field_ref": None,
                    "continuation_field_ref": None,
                }
                for read in prompt.request.targets
            }})
        branch = prompt.request.strategy.branches[0].branch_id
        if name == "SemanticSourceRealizationTurnPrompt":
            return parse(
                {
                    "set_bindings": {
                        f"fact_1:set:{ref}": [
                            {
                                "branch_id": branch,
                                "mapping_basis": "Declared record population.",
                                "rows_ref": sources[name].id,
                                "record_fields": [],
                            }
                        ]
                        for ref, name in (("s1", "facilities"), ("s2", "districts"))
                    },
                    "fact_bindings": {},
                    "association_bindings": {
                        "fact_1:association:a1": [
                            {
                                "branch_id": branch,
                                "mapping_basis": "Match the observed district value to the district record property.",
                                "realization_ref": None,
                                "reference_from_set_ref": None,
                                "field_pairs": [
                                    {
                                        "from_field_ref": fields["facilities"][
                                            "district"
                                        ],
                                        "to_field_ref": fields["districts"]["id"],
                                    }
                                ],
                            }
                        ]
                    },
                }
            )
        if name == "SetPopulationTurnPrompt":
            return parse(
                {
                    "populations": {
                        ref: [
                            {
                                "branch_id": branch,
                                "logical_set_meaning": index.term_by_ref[
                                    next(
                                        item
                                        for item in index.term_by_ref
                                        if item.token == ref
                                    )
                                ].origin.meaning,
                                "mapping_basis": "All returned records belong to this set.",
                                "population": {"kind": "exact_population"},
                            }
                        ]
                        for ref in prompt.realization.set_bindings
                    }
                }
            )
        if name == "LiteralReferenceTurnPrompt":
            from fervis.lookup.source_binding.reference_prompt import reference_tasks

            with pytest.raises(ValueError, match="selected carrier"):
                parse({"references": {
                    ref: {"mapping_basis": "Attempt to borrow an unrelated property's value.",
                          "field_refs": [fields["facilities"]["id"]]}
                    for ref in reference_tasks(prompt.realization)
                }})

            return parse(
                {
                    "references": {
                        ref: {
                            "mapping_basis": "The supplied district name identifies a matching district record.",
                            "field_refs": [fields["districts"]["name"]],
                        }
                        for ref in reference_tasks(prompt.realization)
                    }
                }
            )
        if name == "DescriptorReferenceTurnPrompt":
            from fervis.lookup.source_binding.reference_prompt import descriptor_tasks

            return parse({"references": {
                ref: {
                    "mapping_basis": "The declared default Boolean identifies the selected district.",
                    "field_ref": fields["districts"]["is_default"],
                    "choice_value": "true",
                }
                for ref in descriptor_tasks(prompt.realization)
            }})
        if name == "SemanticSourceBindingTurnPrompt":
            request = prompt.request
            plan = parse(
                {
                    "resolved_input_applications": {
                        branch: [
                            {
                                "kind": "no_request_application",
                                "mapping_basis": "Resolve against observed rows.",
                                "owner_ref": owner,
                                "value_ref": ref,
                            }
                            for owner in request.invocation_application_owner_refs
                            for ref in request.unapplied_input_value_refs_for_owner(
                                owner, branch_id=branch
                            )
                        ]
                    },
                    "finite_choice_applications": {branch: {}},
                    "choice_requirement_applications": {branch: {}},
                }
            )
            from fervis.lookup.source_binding.verification import verify_source_strategy, SourceStrategyVerificationFailure
            assert isinstance(verify_source_strategy(replace(plan, reference_bindings=()), request=request),
                              SourceStrategyVerificationFailure)
            return plan
        raise AssertionError(name)

    if not use_runtime and not duplicate_name and not descriptor:
        altered = replace(canonical, typed_value=FactValue.named(
            id=value.id, known_input_id="i1", text="Other District",
            proof_refs=value.proof_refs,
        ))
        with pytest.raises(ValueError, match="named reference differs"):
            realize_and_compile_logical_plan(
                logical,
                sources_by_fact={"fact_1": snapshot_source_catalog(tuple(sources.values()))},
                canonical_values=(altered,),
                turn=turn,
            )

    if use_runtime:
        from types import SimpleNamespace
        from fervis.lookup.orchestration import semantic_compilation as shared
        from fervis.lookup.orchestration.logical_compilation import (
            compile_logical_question,
        )
        from fervis.lookup.question_contract import QuestionContractRequest
        from fervis.lookup.turn_prompts import HostPromptContext
        from fervis.lookup.query_enrichment.semantic import (
            SemanticQueryEnrichmentResult,
            RecallBucketMatch,
            InputResourceSearchTerms,
        )
        from fervis.lookup.read_eligibility import (
            SemanticReadEligibilityResult,
            ReadRequirementAssessment,
            SemanticReadDecision,
        )

        def scripted(purpose, *, prompt, parse, **kwargs):
            name = type(prompt).__name__
            if name == "SemanticQuestionFrameTurnPrompt":
                return SimpleNamespace(result=parse(fixture["frame_payload"]))
            if name == "SemanticQuestionContractTurnPrompt":
                return SimpleNamespace(result=parse(fixture["payload"]))
            if name == "SemanticQueryEnrichmentTurnPrompt":
                return SimpleNamespace(
                    result=SemanticQueryEnrichmentResult(
                        tuple(
                            RecallBucketMatch(
                                bucket.bucket_ref,
                                ("facilities", "districts"),
                                ("facilities", "districts"),
                            )
                            for bucket in prompt.request.recall_buckets
                        ),
                        tuple(
                            InputResourceSearchTerms(task.input_use_ref, ("districts",))
                            for task in prompt.request.reference_tasks
                        ),
                    )
                )
            return SimpleNamespace(result=turn(purpose, prompt, parse))

        def eligibility(eligibility_request, **kwargs):
            return SemanticReadEligibilityResult(
                tuple(
                    ReadRequirementAssessment(
                        "fact_1",
                        source.read_id,
                        (source.id,),
                        source.read_id,
                        tuple(field.field_ref for field in source.fields),
                        "The API supplies required records.",
                        SemanticReadDecision.RETAIN,
                    )
                    for source in eligibility_request.source_catalog.sources
                    if source.read_id
                ),
                (),
            )

        monkeypatch.setattr(shared, "_turn", scripted)
        monkeypatch.setattr(shared, "_read_eligibility_turn", eligibility)

        class NoPlanningReads:
            def read(self, **kwargs):
                raise AssertionError(
                    "Reference values must be resolved at execution, not frozen while planning"
                )

        question = fixture["question_context_texts"][0]
        request = shared.SemanticCompilationRequest(
            "reference-test",
            question,
            QuestionContractRequest(question, {}),
            catalog,
            (),
            NoPlanningReads(),
            None,
            "openai",
            1,
            10,
            None,
            {},
            HostPromptContext(),
        )
        outcome = compile_logical_question(request)
        assert isinstance(outcome, shared.SemanticCompilationSuccess)
        result = FactCompilationResult(
            outcome.compilation.answer_program, outcome.compilation.initial_bindings
        )
    else:
        result = realize_and_compile_logical_plan(
            logical,
            sources_by_fact={
                "fact_1": snapshot_source_catalog(tuple(sources.values()))
            },
            canonical_values=(canonical,),
            turn=turn,
        )
    assert isinstance(result, FactCompilationResult)
    program = decode_answer_program(
        canonical_answer_program_json(result.answer_program)
    )
    reads = []

    class Port:
        def __init__(self, selected, *, changed_default=False):
            self.selected = selected
            self.changed_default = changed_default

        def read(self, *, endpoint_name, args):
            reads.append(endpoint_name)
            assert args == {}, (
                "Unresolved district text must not become a facility search argument"
            )
            rows = (
                [
                    {
                        "id": "d1",
                        "name": "River District" if self.selected == "d1" else "Other",
                        "is_default": self.selected == "d1" and not self.changed_default,
                    },
                    {
                        "id": "d2",
                        "name": "River District"
                        if self.selected == "d2" or duplicate_name
                        else "Other",
                        "is_default": self.selected == "d2" or duplicate_name,
                    },
                ]
                if endpoint_name == "districts"
                else [
                    {"id": str(i), "district": "d1" if i < 2 else "d2"}
                    for i in range(5)
                ]
            )
            return {"responseStatus": 200, "responseBody": rows}

    for selected, expected in [("d1", 2), ("d2", 3)]:
        executed = invoke_answer_program(
            program=program,
            bindings=result.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(selected), LookupMemory()),
        )
        if duplicate_name and selected == "d1":
            assert executed.issue.reference.reason.value == "AMBIGUOUS_RESULT"
        else:
            assert executed.issue is None
            assert (
                next(
                    iter(executed.fact_result.outcome.projected_rows[0].values.values())
                )
                == expected
            )
    if duplicate_name and not descriptor:
        from fervis.lookup.orchestration.terminal_results import reference_clarification_fact_result
        from fervis.lookup.clarification.response import parse_clarification_response
        from fervis.lookup.orchestration.logical_compilation import _selected_reference_choices

        ambiguous = invoke_answer_program(
            program=program, bindings=result.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port("d1"), LookupMemory()),
        )
        clarification = reference_clarification_fact_result(
            ambiguous.issue, contract=logical.contract
        ).outcome.clarifications[0]
        assert len(clarification.subjects[0].options) == 2
        option = next(item for item in clarification.subjects[0].options
                      if item.observed_properties[0].value == "d1")
        response = parse_clarification_response(
            clarification, response_id="choose_d1", response_text=option.label,
            selected_option_id=option.id,
        )
        choice = _selected_reference_choices((response,), contract=logical.contract)
        calls.clear()
        if use_runtime:
            selected_outcome = compile_logical_question(
                replace(request, clarification_responses=(response,))
            )
            assert isinstance(selected_outcome, shared.SemanticCompilationSuccess)
            selected_program = FactCompilationResult(
                selected_outcome.compilation.answer_program,
                selected_outcome.compilation.initial_bindings,
            )
        else:
            selected_program = realize_and_compile_logical_plan(
                logical,
                sources_by_fact={
                    "fact_1": snapshot_source_catalog(tuple(sources.values()))
                },
                canonical_values=(canonical,), selected_reference_choices=choice,
                turn=turn,
            )
        assert isinstance(selected_program, FactCompilationResult)
        from fervis.lookup.answer_program.operations import ReferenceGuardSpec
        from fervis.lookup.plan_execution.errors import VerificationError

        guard = next(op for op in selected_program.answer_program.operations
                     if isinstance(op.spec, ReferenceGuardSpec)
                     and op.spec.observed_properties)
        forged = replace(guard, spec=replace(
            guard.spec,
            observed_properties=(replace(
                guard.spec.observed_properties[0],
                source_field_ref="forged.source.property",
            ), *guard.spec.observed_properties[1:]),
        ))
        forged_program = replace(
            selected_program.answer_program,
            operations=tuple(forged if op.id == guard.id else op
                             for op in selected_program.answer_program.operations),
        )
        read_count = len(reads)
        with pytest.raises(VerificationError, match="source field authority"):
            invoke_answer_program(
                program=forged_program,
                bindings=selected_program.initial_bindings,
                environment=ExecutionEnvironment(catalog=catalog),
                ports=RuntimePorts(Port("d1"), LookupMemory()),
            )
        assert len(reads) == read_count
        chosen = invoke_answer_program(
            program=selected_program.answer_program,
            bindings=selected_program.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port("d1"), LookupMemory()),
        )
        assert chosen.issue is None
        assert next(iter(chosen.fact_result.outcome.projected_rows[0].values.values())) == 2
        changed_other_property = invoke_answer_program(
            program=selected_program.answer_program,
            bindings=selected_program.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port("d1", changed_default=True), LookupMemory()),
        )
        assert changed_other_property.issue.reference.reason.value == "NOT_FOUND"
        changed_name = invoke_answer_program(
            program=selected_program.answer_program,
            bindings=selected_program.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port("d2"), LookupMemory()),
        )
        assert changed_name.issue.reference.reason.value == "NOT_FOUND"
    assert calls == [
        "SemanticSourceRealizationTurnPrompt",
        "SetPopulationTurnPrompt",
        "DescriptorReferenceTurnPrompt" if descriptor else "LiteralReferenceTurnPrompt",
        "SemanticSourceBindingTurnPrompt",
    ]
