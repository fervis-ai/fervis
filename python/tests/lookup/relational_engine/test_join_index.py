"""Equality joins retain relational semantics without quadratic comparisons."""
from decimal import Decimal

from fervis.lookup.answer_program.operations import JoinKey, JoinMode, JoinSpec
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_engine import shared
from fervis.lookup.plan_execution.operation_runtime import ExecutableOperation, RelationEngineInput
from fervis.lookup.plan_execution.relations import RelationRows
from tests.lookup.relational_engine.test_schema_and_nulls import COMPLETE


def _join(left, right, left_type='integer', right_type='integer'):
    result = execute_operations(RelationEngineInput(
        relations=(
            RelationRows('left', tuple(left), field_types={'left_id': left_type}, completeness=COMPLETE),
            RelationRows('right', tuple(right), field_types={'right_id': right_type, 'label': 'string'}, completeness=COMPLETE),
        ),
        operations=(ExecutableOperation('join', JoinSpec('left', 'right', (JoinKey('left_id', 'right_id'),), JoinMode.LEFT), 'joined'),),
    ))
    assert result.issue is None
    return result.relation('joined').rows


def test_join_preserves_duplicate_order_nulls_and_numeric_coercion():
    assert _join(
        ({'left_id': 1}, {'left_id': None}, {'left_id': 2}),
        ({'right_id': Decimal('1.00'), 'label': 'first'},
         {'right_id': None, 'label': 'null'},
         {'right_id': Decimal('1.0'), 'label': 'second'}),
        left_type='integer', right_type='decimal',
    ) == (
        {'left_id': 1, 'right_id': Decimal('1.00'), 'label': 'first'},
        {'left_id': 1, 'right_id': Decimal('1.0'), 'label': 'second'},
        {'left_id': None, 'right_id': None, 'label': None},
        {'left_id': 2, 'right_id': None, 'label': None},
    )


def test_join_does_not_compare_each_pair_of_distinct_keys(monkeypatch):
    calls = 0
    original = shared.declared_equal
    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    monkeypatch.setattr(shared, 'declared_equal', counted)
    count = 100
    result = _join(({'left_id': i} for i in range(count)),
                   ({'right_id': i, 'label': str(i)} for i in reversed(range(count))))
    assert [row['right_id'] for row in result] == list(range(count))
    assert calls <= 3 * count


def test_anti_join_matches_different_declared_numeric_types():
    from fervis.lookup.answer_program.operations import AntiJoinSpec, RelationRoleRef, RelationRole, NamedExpression
    from fervis.lookup.answer_program.expressions import FieldRef
    spec = AntiJoinSpec(
        RelationRoleRef('c', RelationRole.ANTI_JOIN_CANDIDATE, ('id',)),
        RelationRoleRef('o', RelationRole.ANTI_JOIN_OBSERVED, ('observed_id',)),
        (JoinKey('id', 'observed_id'),), (NamedExpression('id', FieldRef('id')),),
    )
    result = execute_operations(RelationEngineInput(relations=(
        RelationRows('c', ({'id': 1}, {'id': 2}), ('id',), {'id':'integer'}, completeness=COMPLETE),
        RelationRows('o', ({'observed_id': Decimal('1.00')},), ('observed_id',), {'observed_id':'decimal'}, completeness=COMPLETE),
    ), operations=(ExecutableOperation('anti', spec, 'out'),)))
    assert result.relation('out').rows == ({'id': 2},)


def test_universal_join_matches_different_declared_numeric_types():
    from fervis.lookup.answer_program.operations import UniversalConditionSpec, RelationRoleRef, RelationRole, NamedExpression
    from fervis.lookup.answer_program.expressions import FieldRef
    spec = UniversalConditionSpec(
        RelationRoleRef('c', RelationRole.UNIVERSAL_CANDIDATE_SUBJECT, ('id',)),
        RelationRoleRef('d', RelationRole.UNIVERSAL_REQUIRED_DIMENSION, ('dimension',)),
        RelationRoleRef('o', RelationRole.UNIVERSAL_OBSERVATION, ('observed_id', 'observed_dimension')),
        (JoinKey('id', 'observed_id'),), (JoinKey('dimension', 'observed_dimension'),),
        FieldRef('passes'), (NamedExpression('id', FieldRef('id')),),
    )
    result = execute_operations(RelationEngineInput(relations=(
        RelationRows('c', ({'id': 1},), ('id',), {'id':'integer'}, completeness=COMPLETE),
        RelationRows('d', ({'dimension': 7},), ('dimension',), {'dimension':'integer'}, completeness=COMPLETE),
        RelationRows('o', ({'observed_id': Decimal('1.00'), 'observed_dimension': Decimal('7.0'), 'passes':True},),
                     ('observed_id', 'observed_dimension'), {'observed_id':'decimal', 'observed_dimension':'decimal', 'passes':'boolean'}, completeness=COMPLETE),
    ), operations=(ExecutableOperation('universal', spec, 'out'),)))
    assert result.relation('out').rows == ({'id': 1},)
