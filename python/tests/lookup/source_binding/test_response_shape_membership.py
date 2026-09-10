from fervis.host_api.contracts import ParameterSemantics
from dataclasses import replace

from tests.lookup.relational_engine.test_scoped_compilation import employee_query
from fervis.lookup.relation_catalog.row_sources.model import (
    RowSourceParam,
    RowSourceValueType,
)
from fervis.lookup.source_binding.model import SourceRealization
from fervis.lookup.source_binding.choice_requirements import requirement_choice_surfaces


def test_response_shape_choices_are_not_ordinary_row_membership_states():
    verified = employee_query()
    source = verified.request.source_catalog.sources[0]
    source = replace(
        source,
        params=(
            RowSourceParam(
                "ordering",
                "records.ordering",
                "ordering",
                RowSourceValueType.CHOICE,
                choices=("id", "salary"),
                default="id",
                semantics=ParameterSemantics.RESPONSE_SHAPE,
            ),
            RowSourceParam(
                "status",
                "records.status",
                "status",
                RowSourceValueType.CHOICE,
                choices=("active", "inactive"),
            ),
        ),
    )
    request = replace(
        verified.request,
        source_catalog=replace(verified.request.source_catalog, sources=(source,)),
    )
    plan = verified.binding_plan
    realization = SourceRealization(
        request, plan.set_bindings, plan.fact_bindings, plan.association_bindings
    )
    assert requirement_choice_surfaces(realization.request, "branch") == ()
    assert any(
        surface.target_ref == "records.ordering"
        for surface in request.source_catalog.choice_surfaces
    )


def test_response_shape_parameter_cannot_realize_a_boolean_row_predicate():
    from fervis.lookup.question_contract.model import FactTerm
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.semantic_types import BooleanType

    verified = employee_query()
    fact = verified.request.index.requested_fact
    fact = replace(
        fact,
        facts=(*fact.facts, FactTerm("flag", "employee", BooleanType(), fact.origin)),
        expressions=(),
        qualification_ref="flag",
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    source = verified.request.source_catalog.sources[0]
    source = replace(
        source,
        params=(
            RowSourceParam(
                "include_labels",
                "records.include_labels",
                "include labels",
                RowSourceValueType.BOOLEAN,
                choices=("false", "true"),
                semantics=ParameterSemantics.RESPONSE_SHAPE,
            ),
        ),
    )
    request = replace(
        verified.request,
        index=index,
        realized_fact_fields=(),
        source_catalog=replace(verified.request.source_catalog, sources=(source,)),
    )
    [surface] = request.source_catalog.choice_surfaces
    assert request.choice_requirement_refs(surface, branch_id="branch") == ()
