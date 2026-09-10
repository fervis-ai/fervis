"""Logical set membership must execute before any consumer counts its rows."""

from dataclasses import replace
import pytest
from jsonschema import Draft7Validator

from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
from fervis.lookup.available_sources import SourceFieldBinding
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import (
    build_row_source_catalog,
    RowSourceKind,
)
from fervis.lookup.source_binding.parser import (
    compile_source_realization,
    compile_source_binding_plan,
)
from fervis.lookup.source_binding.schema import build_semantic_source_realization_schema
from fervis.lookup.source_binding.set_population import apply_set_populations
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from tests.lookup.fact_compilation.test_compiler import _compile_memory_count


def _request(rows, field_types):
    _, _, memory, _, _, initial = _compile_memory_count(
        rows, meaning="requested subset count"
    )
    memory = replace(memory, field_types=field_types)
    source = next(
        item
        for item in build_row_source_catalog(
            RelationCatalog(), memory_relations=(memory,)
        ).sources
        if item.kind is RowSourceKind.MEMORY_READ
    )
    request = replace(
        initial.request,
        source_catalog=replace(initial.request.source_catalog, sources=(source,)),
    )
    return request, memory, source


def _compile(request, source, population, fact_bindings=None):
    branch = request.strategy.branches[0].branch_id
    set_ref = request.index.subject_obligation.subject_set_ref.token
    payload = {
        "set_bindings": {
            set_ref: [
                {
                    "branch_id": branch,
                    "mapping_basis": "Materialize the requested set exactly",
                    "rows_ref": source.id,
                }
            ]
        },
        "fact_bindings": fact_bindings or {},
        "association_bindings": {},
    }
    Draft7Validator(build_semantic_source_realization_schema(request)).validate(payload)
    realized = compile_source_realization(payload, request=request)
    realized = apply_set_populations(
        {
            "populations": {
                set_ref: [
                    {
                        "branch_id": branch,
                        "logical_set_meaning": request.index.requested_fact.sets[
                            0
                        ].origin.meaning,
                        "mapping_basis": "Selected logical set",
                        "population": population,
                    }
                ]
            }
        },
        realization=realized,
    )
    plan = compile_source_binding_plan(
        {
            "resolved_input_applications": {branch: []},
            "finite_choice_applications": {branch: {}},
            "choice_requirement_applications": {branch: {}},
        },
        realization=realized,
    )
    verified = verify_source_strategy(plan, request=realized.request)
    assert isinstance(verified, VerifiedSourceStrategy)
    return compile_verified_source_strategy(verified)


def _count(compiled, memory):
    execution = invoke_answer_program(
        program=compiled.answer_program,
        bindings=compiled.initial_bindings,
        environment=ExecutionEnvironment(
            catalog=RelationCatalog(),
            memory_relations=(memory,),
            expression_values={"ANCHOR_TIMEZONE": "UTC"},
            expression_types={"ANCHOR_TIMEZONE": "string"},
        ),
        ports=RuntimePorts(
            data_access_port=None, memory=LookupMemory(relations=(memory,))
        ),
    )
    assert execution.issue is None
    return execution.fact_result.outcome.projected_rows[0].values["fact_1.output_1"]


@pytest.mark.parametrize(
    "members,expected_value,expected",
    [
        ((True, False, False), True, 1),
        ((False, False, False), True, 0),
        ((True, False, False), False, 2),
        ((False, False, False), False, 3),
    ],
)
def test_set_membership_filters_before_counting(members, expected_value, expected):
    rows = tuple(
        {"event_id": str(i), "eligible": member} for i, member in enumerate(members)
    )
    request, memory, source = _request(
        rows, {"event_id": "string", "eligible": "boolean"}
    )
    population = {
        "kind": "restricted_population",
        "condition": {
            "kind": "boolean_field",
            "field_ref": SourceFieldBinding(source.id, source.field("eligible")).ref,
            "expected_value": expected_value,
        },
    }
    from fervis.lookup.source_binding.membership import (
        membership_schema,
        membership_definitions,
    )

    Draft7Validator(
        {**membership_schema(request), "$defs": membership_definitions(request)}
    ).validate(population)
    assert _count(_compile(request, source, population), memory) == expected
    assert _count(_compile(request, source, {"kind": "exact_population"}), memory) == 3


def test_membership_can_compare_declared_numeric_fields_without_inventing_thresholds():
    rows = (
        {"event_id": "a", "available": 2, "required": 4},
        {"event_id": "b", "available": 8, "required": 4},
        {"event_id": "c", "available": 4, "required": 4},
    )
    request, memory, source = _request(
        rows, {"event_id": "string", "available": "integer", "required": "integer"}
    )
    population = {
        "kind": "restricted_population",
        "condition": {
            "kind": "binary",
            "operator": "lt",
            "left": {
                "kind": "field",
                "field_ref": SourceFieldBinding(
                    source.id, source.field("available")
                ).ref,
            },
            "right": {
                "kind": "field",
                "field_ref": SourceFieldBinding(
                    source.id, source.field("required")
                ).ref,
            },
        },
    }
    assert _count(_compile(request, source, population), memory) == 1


@pytest.mark.parametrize("field_type", ["any", "integer"])
def test_membership_requires_a_declared_boolean_expression(field_type):
    request, _, source = _request(
        ({"event_id": "a", "eligible": 1},),
        {"event_id": "string", "eligible": field_type},
    )
    with pytest.raises(ValueError, match="concrete scalar|must be Boolean"):
        _compile(
            request,
            source,
            {
                "kind": "restricted_population",
                "condition": {
                    "kind": "field",
                    "field_ref": SourceFieldBinding(
                        source.id, source.field("eligible")
                    ).ref,
                },
            },
        )


def test_restricting_a_related_set_preserves_parent_rows_for_absence(monkeypatch):
    from fervis.lookup.answer_program.expressions import FieldRef, UnaryExpression
    from fervis.lookup.expression_operators import ExpressionUnaryOperator
    from fervis.lookup.question_contract.model import Quantify, Quantifier
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from tests.lookup.fact_compilation import test_compiler as fixture

    saved = {}

    def compile_restricted(verified):
        request, plan = verified.request, verified.binding_plan
        fact = request.index.requested_fact
        fact = replace(
            fact,
            expressions=tuple(
                replace(node, quantifier=Quantifier.NOT_EXISTS)
                if isinstance(node, Quantify)
                else node
                for node in fact.expressions
            ),
        )
        request = replace(
            request,
            index=analyze_requested_fact(
                fact,
                inputs=request.index.input_by_ref,
                input_denotations=request.index.input_denotation_by_ref,
            ),
        )
        sets = dict(plan.set_bindings)
        [related] = sets["fact_1:set:s_area"]
        sets["fact_1:set:s_area"] = (
            replace(
                related,
                membership=UnaryExpression(
                    ExpressionUnaryOperator.IS_NULL, FieldRef("area_id")
                ),
            ),
        )
        verified = verify_source_strategy(
            replace(plan, set_bindings=sets), request=request
        )
        assert isinstance(verified, VerifiedSourceStrategy)
        return compile_verified_source_strategy(verified)

    def capture_execution(**kwargs):
        result = invoke_answer_program(**kwargs)
        saved["result"] = result
        raise Captured

    class Captured(Exception):
        pass

    monkeypatch.setattr(fixture, "compile_verified_source_strategy", compile_restricted)
    monkeypatch.setattr(fixture, "invoke_answer_program", capture_execution)
    with pytest.raises(Captured):
        fixture.test_co_resident_exists_filters_subject_rows_before_counting()
    result = saved["result"]
    assert result.issue is None
    assert result.fact_result.outcome.projected_rows[0].values == {"fact_1.output_1": 3}


def test_membership_uses_declared_choice_values_as_pinned_program_inputs():
    request, memory, source = _request(
        ({"event_id": "a", "eligible": True}, {"event_id": "b", "eligible": False}),
        {"event_id": "string", "eligible": "boolean"},
    )
    choice = next(
        item for item in request.source_catalog.choice_values if item.value == "true"
    )
    population = {
        "kind": "restricted_population",
        "condition": {
            "kind": "binary",
            "operator": "equals",
            "left": {
                "kind": "field",
                "field_ref": SourceFieldBinding(
                    source.id, source.field("eligible")
                ).ref,
            },
            "right": {"kind": "value", "value_ref": choice.value_ref},
        },
    }
    compiled = _compile(request, source, population)
    assert _count(compiled, memory) == 1
    assert compiled.initial_bindings.bindings


def test_source_population_decision_cannot_be_an_unparsed_promise():
    from fervis.lookup.provider_contract import ProviderObject
    from fervis.lookup.source_binding.membership import parse_membership

    request, _, source = _request(
        ({"event_id": "a", "eligible": True},),
        {"event_id": "string", "eligible": "boolean"},
    )
    with pytest.raises(ValueError, match="unexpected field"):
        parse_membership(
            ProviderObject({"kind": "exact_population", "filter_later": "eligible"}),
            source=source,
            request=request,
        )
    with pytest.raises(ValueError, match="missing required field"):
        parse_membership(
            ProviderObject({"kind": "restricted_population"}),
            source=source,
            request=request,
        )
    with pytest.raises(ValueError, match="current-run authority"):
        parse_membership(
            ProviderObject(
                {
                    "kind": "restricted_population",
                    "condition": {
                        "kind": "binary",
                        "operator": "equals",
                        "left": {
                            "kind": "field",
                            "field_ref": SourceFieldBinding(
                                source.id, source.field("eligible")
                            ).ref,
                        },
                        "right": {"kind": "value", "value_ref": "invented_true"},
                    },
                }
            ),
            source=source,
            request=request,
        )


@pytest.mark.parametrize("composite", [False, True])
def test_membership_checks_grounded_operand_projection(monkeypatch, composite):
    from fervis.lookup.answer_program.values import FactValue
    from fervis.lookup.canonical_data import EntityKeyValue, EntityKeyComponentValue
    from fervis.lookup.provider_contract import ProviderObject
    from fervis.lookup.source_binding.membership import (
        parse_membership,
        membership_definitions,
    )
    from tests.lookup.fact_compilation import test_compiler as fixture

    saved = {}

    class Captured(Exception):
        pass

    def capture(verified):
        saved["request"] = verified.request
        raise Captured

    monkeypatch.setattr(fixture, "compile_verified_source_strategy", capture)
    with pytest.raises(Captured):
        fixture.test_co_resident_exists_filters_subject_rows_before_counting()
    request = saved["request"]
    [canonical] = request.canonical_values
    if composite:
        key = canonical.typed_value.payload.key
        value = FactValue.identity(
            id=canonical.canonical_value_id,
            known_input_id=canonical.input_ref,
            key=EntityKeyValue(
                key.entity_kind,
                key.key_id,
                (*key.components, EntityKeyComponentValue("tenant_id", "tenant-a")),
            ),
        )
        request = replace(
            request, canonical_values=(replace(canonical, typed_value=value),)
        )
    source = request.source_catalog.sources[0]
    payload = ProviderObject(
        {
            "kind": "restricted_population",
            "condition": {
                "kind": "binary",
                "operator": "equals",
                "left": {
                    "kind": "field",
                    "field_ref": SourceFieldBinding(
                        source.id, source.field("area_id")
                    ).ref,
                },
                "right": {"kind": "value", "value_ref": canonical.canonical_value_id},
            },
        }
    )
    choices = (
        membership_definitions(request)
        .get("membership_value", {})
        .get("properties", {})
        .get("value_ref", {})
        .get("enum", [])
    )
    if composite:
        assert canonical.canonical_value_id not in choices
        with pytest.raises(ValueError, match="projection"):
            parse_membership(payload, source=source, request=request)
    else:
        assert canonical.canonical_value_id in choices
        assert parse_membership(payload, source=source, request=request) is not None


def test_population_interpretation_cannot_borrow_a_consumers_temporal_filter():
    from fervis.lookup.answer_program.values import FactValue
    from fervis.lookup.question_contract.model import InputTerm, FactTerm, Comparison
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.expression_operators import ExpressionBinaryOperator
    from fervis.lookup.semantic_types import TemporalScopeType, DateType
    from tests.lookup.fact_compilation.test_compiler import _denotation
    from fervis.lookup.grounding.semantic import CanonicalInputValue

    request, memory, source = _request(
        (
            {"event_id": "a", "day": "2026-03-15"},
            {"event_id": "b", "day": "2026-04-01"},
        ),
        {"event_id": "string", "day": "date"},
    )
    term = InputTerm(
        "period", request.index.requested_fact.origin, "March 2026", TemporalScopeType()
    )
    value = FactValue.time(
        id="march",
        known_input_id="period",
        expression="March 2026",
        resolved_start="2026-03-01",
        resolved_end="2026-03-31",
        granularity="month",
        proof_refs=("question:period",),
    )
    fact = request.index.requested_fact
    fact = replace(
        fact,
        facts=(FactTerm("when", "s1", DateType(), fact.origin),),
        expressions=(
            *fact.expressions,
            Comparison(
                "in_period",
                ExpressionBinaryOperator.WITHIN,
                "when",
                "period",
                fact.origin,
            ),
        ),
        qualification_ref="in_period",
    )
    index = analyze_requested_fact(
        fact, inputs={"period": term}, input_denotations=_denotation(term)
    )
    branch = replace(
        request.strategy.branches[0],
        qualification_clause_refs=tuple(
            c.clause_ref for c in index.qualification.clauses
        ),
    )
    request = replace(
        request,
        index=index,
        strategy=replace(request.strategy, branches=(branch,)),
        canonical_values=(
            CanonicalInputValue(
                "march",
                "period",
                tuple(use.use_ref for use in index.input_use_sites),
                value,
                ("question:period",),
            ),
        ),
    )
    population = {
        "kind": "restricted_population",
        "condition": {
            "kind": "binary",
            "operator": "within",
            "left": {
                "kind": "field",
                "field_ref": SourceFieldBinding(source.id, source.field("day")).ref,
            },
            "right": {"kind": "value", "value_ref": "march"},
        },
    }
    binding = {
        "fact_1:fact:when": [
            {
                "branch_id": branch.branch_id,
                "mapping_basis": "Declared date",
                "field_ref": SourceFieldBinding(source.id, source.field("day")).ref,
            }
        ]
    }
    with pytest.raises(ValueError, match="current-run authority"):
        _compile(request, source, population, binding)
    assert (
        _count(_compile(request, source, {"kind": "exact_population"}, binding), memory)
        == 1
    )


def test_population_prompt_excludes_question_and_consumer_predicates():
    from fervis.lookup.source_binding.set_population import SetPopulationTurnPrompt
    from fervis.lookup.turn_prompts import TurnPromptContext

    request, _, source = _request(
        ({"event_id": "a", "eligible": True},),
        {"event_id": "string", "eligible": "boolean"},
    )
    ref = request.index.subject_obligation.subject_set_ref.token
    branch = request.strategy.branches[0].branch_id
    realized = compile_source_realization(
        {
            "set_bindings": {
                ref: [
                    {
                        "branch_id": branch,
                        "mapping_basis": "UNTRUSTED_PREVIOUS_POPULATION_PROMISE",
                        "rows_ref": source.id,
                    }
                ]
            },
            "fact_bindings": {},
            "association_bindings": {},
        },
        request=request,
    )
    payload = SetPopulationTurnPrompt(realized).to_model_payload(
        TurnPromptContext(current_question="CONSUMER_QUESTION_SECRET")
    )
    assert "CONSUMER_QUESTION_SECRET" not in payload.prompt_text
    assert "UNTRUSTED_PREVIOUS_POPULATION_PROMISE" not in payload.prompt_text
    assert "boolean_requirements" not in payload.prompt_text
    assert "requested subset count" in payload.prompt_text


def test_source_selection_receives_declared_field_meaning_before_choosing_rows():
    from fervis.lookup.source_binding.prompt import SemanticSourceRealizationTurnPrompt
    from fervis.lookup.turn_prompts import TurnPromptContext

    request, _, source = _request(
        ({"event_id": "a", "eligible": True},),
        {"event_id": "string", "eligible": "boolean"},
    )
    source = replace(
        source,
        fields=tuple(
            replace(field, description="False exactly denotes a failed observation.")
            if field.id == "eligible"
            else field
            for field in source.fields
        ),
    )
    request = replace(
        request, source_catalog=replace(request.source_catalog, sources=(source,))
    )
    payload = SemanticSourceRealizationTurnPrompt(request).to_model_payload(
        TurnPromptContext(current_question="Count failures")
    )
    assert "False exactly denotes a failed observation." in payload.prompt_text


def test_membership_schema_requires_dependence_on_source_rows():
    from fervis.lookup.source_binding.membership import (
        membership_schema,
        membership_definitions,
    )

    request, _, _ = _request(
        ({"event_id": "a", "eligible": True},),
        {"event_id": "string", "eligible": "boolean"},
    )
    schema = {**membership_schema(request), "$defs": membership_definitions(request)}
    false_value = next(
        item.value_ref
        for item in request.source_catalog.choice_values
        if item.value == "false"
    )
    constant = {
        "kind": "restricted_population",
        "condition": {"kind": "value", "value_ref": false_value},
    }
    assert not Draft7Validator(schema).is_valid(constant)
