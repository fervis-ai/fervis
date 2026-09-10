"""Scoped plans must pass production binding parsing, not only verification."""

from dataclasses import replace
import pytest
from jsonschema import validate

from tests.lookup.relational_engine.test_scoped_compilation import employee_query
from tests.lookup.relational_engine.test_coverage_compilation import coverage_query
from fervis.lookup.question_contract.model import Quantifier
from fervis.lookup.source_binding.model import SourceRealization
from fervis.lookup.source_binding.parser import compile_source_binding_plan
from fervis.lookup.source_binding.schema import build_semantic_source_binding_schema
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.fact_compilation import compile_verified_source_strategy


@pytest.mark.parametrize(
    "shape", ["exists", "not_exists", "forall", "coverage", "related_row"]
)
def test_production_binding_parses_relational_boolean_mechanics(shape):
    original = (
        coverage_query()
        if shape == "coverage"
        else employee_query(
            Quantifier.EXISTS if shape == "related_row" else Quantifier(shape)
        )
    )
    if shape == "related_row":
        from fervis.lookup.question_contract.model import AssociationTerm, RelatedRow
        from fervis.lookup.question_contract.analysis import analyze_requested_fact
        from fervis.lookup.relation_catalog.row_sources.model import (
            RowSourceField,
            RowSourceValueType,
            RowSourceEntityReference,
            RowSourceEntityReferenceComponent,
            row_source_relation_evidence,
        )
        from fervis.lookup.source_binding.model import (
            AssociationRealization,
            AssociationRealizationKind,
        )

        fact = original.request.index.requested_fact
        fact = replace(
            fact,
            associations=(
                *fact.associations,
                AssociationTerm("approval", "employee", "manager", fact.origin),
            ),
            expressions=(
                RelatedRow(
                    "related", "manager", ("management", "approval"), None, fact.origin
                ),
            ),
            qualification_ref="related",
        )
        index = analyze_requested_fact(fact, inputs={}, input_denotations={})
        source = original.request.source_catalog.sources[0]
        source = replace(
            source,
            fields=(
                *source.fields,
                RowSourceField(
                    "approver_id",
                    "field.approver_id",
                    "Approver",
                    RowSourceValueType.STRING,
                    source.fields[0].allowed_roles,
                ),
            ),
            entity_references=(
                *source.entity_references,
                RowSourceEntityReference(
                    "approver",
                    "employee",
                    "pk",
                    (RowSourceEntityReferenceComponent("id", "approver_id"),),
                ),
            ),
        )
        links = row_source_relation_evidence((source,))
        edge = next(
            item
            for item in links
            if item.evidence_ref
            not in {
                item.evidence_ref
                for item in original.request.source_catalog.relation_evidence
            }
        )
        branch = replace(
            original.request.strategy.branches[0],
            relation_evidence_refs=tuple(item.evidence_ref for item in links),
        )
        strategy = replace(original.request.strategy, branches=(branch,))
        request = replace(
            original.request,
            index=index,
            strategy=strategy,
            source_catalog=replace(
                original.request.source_catalog,
                sources=(source,),
                relation_evidence=links,
            ),
        )
        binding = replace(
            original.binding_plan,
            strategy=strategy,
            association_bindings={
                **original.binding_plan.association_bindings,
                index.fact_local_ref_by_local_id["approval"].token: (
                    AssociationRealization(
                        "branch",
                        "Declared approver relation.",
                        AssociationRealizationKind.DECLARED_RELATION,
                        (source.id, source.id),
                        edge.evidence_ref,
                        (source.id, edge.evidence_ref),
                        index.fact_local_ref_by_local_id["employee"].token,
                    ),
                ),
            },
        )
        original = replace(original, request=request, binding_plan=binding)
    plan = original.binding_plan
    realization = SourceRealization(
        original.request,
        plan.set_bindings,
        plan.fact_bindings,
        plan.association_bindings,
    )
    membership = realization
    payload = {
        "resolved_input_applications": {"branch": []},
        "finite_choice_applications": {"branch": {}},
        "choice_requirement_applications": {"branch": {}},
    }
    validate(payload, build_semantic_source_binding_schema(membership.request))
    parsed = compile_source_binding_plan(payload, realization=membership)
    verified = verify_source_strategy(parsed, request=membership.request)
    assert isinstance(verified, VerifiedSourceStrategy)
    assert compile_verified_source_strategy(verified).answer_program.operations
