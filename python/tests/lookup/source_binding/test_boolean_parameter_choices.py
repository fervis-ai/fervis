"""Boolean request controls have a finite domain even without ChoiceField metadata."""
from dataclasses import replace
from fervis.lookup.relation_catalog.row_sources import RowSourceParam, RowSourceValueType
from fervis.lookup.source_binding.model import source_required_inputs_are_satisfiable, source_binding_clarification
from tests.lookup.source_binding._candidate_fixture import daily_observation_request


def request_with_boolean_control():
    request = daily_observation_request()
    source = request.source_catalog.sources[-1]
    control = RowSourceParam('include_details','records.query.include_details','include_details',
        RowSourceValueType.BOOLEAN,source='query',required=True,
        description='Include the detailed observation fields.')
    source = replace(source,params=(control,))
    return replace(request,source_catalog=replace(request.source_catalog,sources=(source,)),
        strategy=replace(request.strategy,branches=tuple(replace(branch,source_refs=(source.id,)) for branch in request.strategy.branches)))


def test_boolean_parameter_exposes_true_and_false_choices():
    request = request_with_boolean_control()
    surface = next(item for item in request.source_catalog.choice_surfaces if item.target_ref=='records.query.include_details')
    assert [(item.value,item.declared_type) for item in surface.values] == [
        ('false',RowSourceValueType.BOOLEAN),('true',RowSourceValueType.BOOLEAN)]


def test_required_boolean_control_can_be_selected_without_inventing_user_input():
    request = request_with_boolean_control()
    assert source_required_inputs_are_satisfiable(request.source_catalog.sources[0],values=())
    assert source_binding_clarification(request) is None


def test_explicit_boolean_choice_restrictions_are_preserved():
    request = request_with_boolean_control()
    source = request.source_catalog.sources[0]
    source = replace(source,params=(replace(source.params[0],choices=('true',)),))
    catalog = replace(request.source_catalog,sources=(source,))
    surface = next(item for item in catalog.choice_surfaces if item.target_ref=='records.query.include_details')
    assert [item.value for item in surface.values] == ['true']
