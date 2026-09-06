"""A category predicate can be realized by a declared Boolean choice."""

from dataclasses import replace

import pytest
from jsonschema import validate

from tests.lookup.fact_compilation.test_compiler import (
    _compile_memory_count,
    _denotation,
)
from fervis.lookup.question_contract.model import (
    InputTerm,
    InputDenotationKind,
    FactTerm,
    Comparison,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.semantic_types import TextType
from fervis.lookup.expression_operators import ExpressionBinaryOperator
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.answer_program.values import FactValue, LiteralType
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.source_binding.parser import (
    compile_source_realization,
    compile_source_binding_plan,
)
from fervis.lookup.source_binding.schema import (
    build_semantic_source_realization_schema,
    build_semantic_source_binding_schema,
)
from fervis.lookup.source_binding.membership import parse_source_membership
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory


def test_text_category_uses_boolean_predicate_without_casting_its_raw_value():
    rows = (
        {"event_id": "a", "is_active": True},
        {"event_id": "b", "is_active": True},
        {"event_id": "c", "is_active": False},
    )
    _, _, memory, _, _, original = _compile_memory_count(rows)
    memory = replace(memory, field_types={"event_id": "string", "is_active": "boolean"})
    fact = original.request.index.requested_fact
    operand = InputTerm("active_input", fact.origin, "active", TextType())
    fact = replace(
        fact,
        facts=(FactTerm("state", "s1", TextType(), fact.origin),),
        expressions=(
            Comparison(
                "active_test",
                ExpressionBinaryOperator.EQUALS,
                "state",
                operand.id,
                fact.origin,
            ),
            *fact.expressions,
        ),
        qualification_ref="active_test",
    )
    index = analyze_requested_fact(
        fact,
        inputs={operand.id: operand},
        input_denotations=_denotation(
            operand, kind=InputDenotationKind.NON_IDENTITY_SCALAR
        ),
    )
    source = next(
        s
        for s in build_row_source_catalog(
            RelationCatalog(), memory_relations=(memory,)
        ).sources
        if s.memory_ref == memory.id
    )
    canonical = CanonicalInputValue(
        "active_value",
        operand.id,
        tuple(u.use_ref for u in index.input_use_sites),
        FactValue.literal(
            id="active_value",
            literal_type=LiteralType.STRING,
            value="active",
            known_input_id=operand.id,
            proof_refs=("question_input:active_input",),
        ),
        ("question_input:active_input",),
    )
    request = replace(
        original.request,
        index=index,
        canonical_values=(canonical,),
        source_catalog=replace(original.request.source_catalog, sources=(source,)),
    )
    branch = request.strategy.branches[0].branch_id
    state_ref = index.fact_local_ref_by_local_id["state"].token
    payload = {
        "set_bindings": {
            index.subject_obligation.subject_set_ref.token: [
                {
                    "branch_id": branch,
                    "mapping_basis": "Record population.",
                    "rows_ref": source.id,
                }
            ]
        },
        "fact_bindings": {
            state_ref: [
                {
                    "branch_id": branch,
                    "mapping_basis": "Activity is encoded by the Boolean field.",
                    "field_ref": f"source_field:{source.id}:is_active",
                }
            ]
        },
        "association_bindings": {},
    }
    from fervis.lookup.question_contract.model import RequestedOutput

    observed_fact = replace(
        fact, outputs=(RequestedOutput("status", "state", fact.origin),)
    )
    observed_index = analyze_requested_fact(
        observed_fact,
        inputs={operand.id: operand},
        input_denotations=_denotation(
            operand, kind=InputDenotationKind.NON_IDENTITY_SCALAR
        ),
    )
    observed_request = replace(request, index=observed_index)
    assert (
        f"source_field:{source.id}:is_active"
        not in observed_request.returned_field_refs_for_fact(state_ref)
    )
    validate(payload, build_semantic_source_realization_schema(request))
    realization = compile_source_realization(payload, request=request)
    from fervis.lookup.relation_catalog.row_sources.model import RowSourceField, RowSourceValueType
    unrelated = RowSourceField('other_flag', 'other_flag', 'Unrelated flag', RowSourceValueType.BOOLEAN, ())
    expanded_source = replace(source, fields=(*source.fields, unrelated))
    expanded = replace(realization.request, source_catalog=replace(realization.request.source_catalog, sources=(expanded_source,)))
    other_surface = next(item for item in expanded.source_catalog.choice_surfaces if item.target_ref == 'other_flag')
    assert expanded.choice_requirement_refs(other_surface, branch_id=branch) == ()
    surface = next(
        s for s in request.source_catalog.choice_surfaces if s.target_ref == "is_active"
    )
    assert {c.value for c in surface.values} == {"false", "true"}
    membership = parse_source_membership(
        {
            branch: {"fact_1:set:s1": {
                surface.surface_ref: {
                    "surface_mapping_basis": "Both states are records.",
                    "choice_reviews": {
                        c.value: {
                            "choice_domain_meaning": "Activity flag.",
                            "decision_basis": "Both states are ordinary records.",
                            "baseline_decision": "INCLUDE",
                        }
                        for c in surface.values
                    },
                }
            }}
        },
        realization=realization,
    )
    requirement = index.boolean_requirements[0].requirement_ref
    binding_payload = {
        "resolved_input_applications": {branch: []},
        "finite_choice_applications": {branch: {}},
        "choice_requirement_applications": {
            branch: {
                surface.surface_ref: {
                    c.value: {
                        "mapping_basis": "True means active.",
                        "selected_by_requirements": [requirement]
                        if c.value == "true"
                        else [],
                    }
                    for c in surface.values
                }
            }
        },
    }
    validate(
        binding_payload,
        build_semantic_source_binding_schema(membership.realization.request),
    )
    plan = compile_source_binding_plan(binding_payload, membership=membership)
    assert state_ref not in plan.fact_bindings
    verified = verify_source_strategy(plan, request=membership.realization.request)
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
    assert result.fact_result.outcome.projected_rows[0].values == {"fact_1.output_1": 2}

    # The category was compiled into a fixed truth mapping. It cannot be
    # rebound while retaining the old predicate.
    from fervis.lookup.answer_program.values import BindingSet
    from fervis.lookup.answer_program.errors import AnswerProgramContractError

    changed = BindingSet.from_bindings(
        tuple(
            replace(
                binding,
                value=replace(
                    binding.value,
                    payload=replace(binding.value.payload, value="inactive"),
                ),
            )
            if binding.value.id == "active_value"
            else binding
            for binding in compiled.initial_bindings.bindings
        )
    )
    with pytest.raises(AnswerProgramContractError, match="fixed"):
        invoke_answer_program(
            program=compiled.answer_program,
            bindings=changed,
            environment=ExecutionEnvironment(
                catalog=RelationCatalog(), memory_relations=(memory,)
            ),
            ports=RuntimePorts(
                data_access_port=None, memory=LookupMemory(relations=(memory,))
            ),
        )
