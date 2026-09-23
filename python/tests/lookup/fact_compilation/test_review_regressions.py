from dataclasses import replace
from decimal import Decimal
import pytest

from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import Arithmetic, Aggregate, AllResults
from fervis.lookup.answer_program.expressions import ExpressionBinaryOperator
from fervis.lookup.source_binding.verification import VerifiedSourceStrategy, verify_source_strategy
from tests.lookup.fact_compilation import test_compiler as fixtures


def capture_fixture(monkeypatch, fixture):
    captured = {}
    original = fixtures.verify_source_strategy
    def capture(plan, *, request):
        captured.update(plan=plan, request=request)
        return original(plan, request=request)
    monkeypatch.setattr(fixtures, 'verify_source_strategy', capture)
    fixture()
    return captured['plan'], captured['request']


@pytest.mark.parametrize("self_reference", [False, True])
def test_identity_projection_uses_selected_reference(monkeypatch, self_reference):
    plan, request = capture_fixture(monkeypatch, fixtures.test_candidate_grain_is_preserved_by_related_set_aggregate)
    source, = request.source_catalog.sources
    staff_reference, = source.entity_references
    staff_field = source.field(staff_reference.components[0].local_field_id)
    manager_reference = replace(staff_reference, id='manager_reference', components=(replace(staff_reference.components[0], local_field_id='manager_id'),))
    source = replace(source, fields=(*source.fields, replace(staff_field, id='manager_id', path='data.manager_id')), entity_references=(manager_reference, staff_reference))
    if self_reference:
        source = replace(source, candidate_keys=tuple(replace(k, entity_kind="staff") for k in source.candidate_keys))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    program = compile_verified_source_strategy(verified).answer_program
    output, = (o for o in program.result_projection.relation_outputs if o.entity_key is not None)
    assert tuple(c.field_id for c in output.entity_key.components) == (staff_field.id,)


def test_aggregate_accepts_computed_scalar_argument(monkeypatch):
    plan, request = capture_fixture(monkeypatch, fixtures.test_declared_association_compiles_to_existing_join_and_grouped_aggregate)
    fact = request.index.requested_fact
    squared = Arithmetic('squared', ExpressionBinaryOperator.MULTIPLY, ('f_amount', 'f_amount'), fact.origin)
    fact = replace(fact, selection=AllResults(), expressions=(squared, *(replace(e, argument_ref='squared') if isinstance(e, Aggregate) else e for e in fact.expressions)))
    request = replace(request, index=analyze_requested_fact(fact, inputs={}, input_denotations={}))
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    compiled = compile_verified_source_strategy(verified)
    program = compiled.answer_program
    from fervis.lookup.answer_program.persistence import parse_stored_program_invocation, program_invocation_id
    from fervis.lookup.contract_codec import answer_program_id, canonical_answer_program_json, canonical_binding_set_json
    from fervis.lineage.enums import ProgramInvocationKind
    program_id = answer_program_id(program)
    stored = parse_stored_program_invocation(
        invocation_id=program_invocation_id(run_id="test-run", program_id=program_id, bindings=compiled.initial_bindings),
        run_id="test-run", program_id=program_id, canonical_json=canonical_answer_program_json(program),
        bindings_json=canonical_binding_set_json(compiled.initial_bindings), kind=ProgramInvocationKind.COMPILED_QUESTION.value,
    )
    assert stored.program == program
    from fervis.lookup.answer_program.operations import AggregateSpec, ProjectSpec
    aggregate, = (o.spec for o in program.operations if isinstance(o.spec, AggregateSpec))
    projected = {o.output_field for op in program.operations if isinstance(op.spec, ProjectSpec) for o in op.spec.outputs}
    assert aggregate.aggregations[0].input_field in projected
    assert aggregate.aggregations[0].grain_fields

    from fervis.lookup.plan_execution.operation_engine import execute_operations
    from fervis.lookup.plan_execution.operation_runtime import ExecutableOperation, RelationEngineInput
    from fervis.lookup.plan_execution.relations import CompletenessProof, CompletenessStatus, RelationRows
    data = {
        "source_events": ({"event_id": "e1", "category_id": "c1", "amount": Decimal(10)}, {"event_id": "e2", "category_id": "c1", "amount": Decimal(20)}),
        "source_categories": ({"category_id": "c1"},),
    }
    result = execute_operations(RelationEngineInput(
        relations=tuple(RelationRows(r.id, data[r.source.row_source_id], completeness=CompletenessProof(status=CompletenessStatus.COMPLETE)) for r in program.relations),
        operations=tuple(ExecutableOperation(o.id, o.spec, o.output_relation) for o in program.operations),
    ))
    assert result.issue is None
    row, = result.relation(program.operations[-1].output_relation).rows
    assert Decimal(500) in row.values()
