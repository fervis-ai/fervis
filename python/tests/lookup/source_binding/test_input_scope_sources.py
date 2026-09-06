"""Invocation targets must belong to the logical rows owning the predicate."""
from dataclasses import replace
from tests.lookup.relational_engine.test_scoped_compilation import employee_query
from fervis.lookup.relation_catalog.row_sources.model import row_source_relation_evidence


def test_manager_threshold_cannot_filter_an_independent_employee_source():
    verified = employee_query(manager_minimum=True)
    source = verified.request.source_catalog.sources[0]
    manager_source = replace(source, id='managers', params=tuple(
        replace(param, param_ref='managers.min_salary') for param in source.params
    ))
    edges = row_source_relation_evidence((source, manager_source))
    edge = next(e for e in edges if e.left_source_ref == source.id and e.right_source_ref == manager_source.id)
    request = replace(verified.request,
        source_catalog=replace(verified.request.source_catalog, sources=(source, manager_source), relation_evidence=edges),
        strategy=replace(verified.request.strategy, branches=tuple(
            replace(branch, source_refs=(source.id, manager_source.id), relation_evidence_refs=tuple(e.evidence_ref for e in edges))
            for branch in verified.request.strategy.branches
        )),
    )
    plan = verified.binding_plan
    sets = dict(plan.set_bindings)
    sets['fact_1:set:manager'] = tuple(replace(value, source_ref=manager_source.id,
        identity_ref=manager_source.identity_evidence[0].identity_ref) for value in sets['fact_1:set:manager'])
    facts = {ref: tuple(replace(value, source_ref=manager_source.id) if 'manager_salary' in ref else value for value in values)
             for ref, values in plan.fact_bindings.items()}
    associations = {ref: tuple(replace(value, source_refs=(source.id, manager_source.id), relation_evidence_ref=edge.evidence_ref,
                                        reference_from_set_ref=None) for value in values)
                    for ref, values in plan.association_bindings.items()}
    request = request.for_bindings(sets, facts, associations)
    options = [option for owner in request.invocation_application_owner_refs
               for option in request.direct_value_options_for_owner(owner, branch_id='branch')]
    assert options
    assert {option.source_ref for option in options} == {'managers'}
