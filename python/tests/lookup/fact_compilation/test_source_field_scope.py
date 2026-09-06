"""Physical field names remain local when realized reads are joined."""

from dataclasses import replace
from decimal import Decimal

from fervis.lookup.available_sources import AvailableSourceCatalog
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    CompletenessProof,
    CompletenessStatus,
)
from fervis.lookup.source_binding.verification import (
    VerifiedSourceStrategy,
    verify_source_strategy,
)
from tests.lookup.fact_compilation import test_compiler as fixtures


def test_join_executes_with_different_primary_keys_both_named_id(monkeypatch):
    captured = {}
    original = fixtures.verify_source_strategy

    def capture(plan, *, request):
        captured.update(plan=plan, request=request)
        return original(plan, request=request)

    monkeypatch.setattr(fixtures, "verify_source_strategy", capture)
    fixtures.test_declared_association_compiles_to_existing_join_and_grouped_aggregate()
    request = captured["request"]
    sources = []
    for source in request.source_catalog.sources:
        old_key = "event_id" if source.id == "source_events" else "category_id"
        sources.append(
            replace(
                source,
                fields=tuple(
                    replace(field, id="id") if field.id == old_key else field
                    for field in source.fields
                ),
                candidate_keys=tuple(
                    replace(
                        key,
                        components=tuple(
                            replace(component, field_id="id")
                            for component in key.components
                        ),
                    )
                    for key in source.candidate_keys
                ),
            )
        )
    evidence = replace(
        request.source_catalog.relation_evidence[0], right_field_refs=("id",)
    )
    request = replace(
        request,
        source_catalog=AvailableSourceCatalog(
            request.source_catalog.contract_snapshot, tuple(sources), (evidence,)
        ),
    )
    verified = verify_source_strategy(captured["plan"], request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    program = compile_verified_source_strategy(verified).answer_program
    data = {
        "source_events": (
            {"id": "event-1", "category_id": "category-1", "amount": Decimal(10)},
        ),
        "source_categories": ({"id": "category-1"},),
    }
    execution = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    relation.id,
                    data[relation.source.row_source_id],
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for relation in program.relations
            ),
            operations=tuple(
                ExecutableOperation(
                    operation.id, operation.spec, operation.output_relation
                )
                for operation in program.operations[:-1]
            ),
        )
    )
    assert execution.issue is None
    rows = execution.relation(program.operations[-2].output_relation).rows
    assert len(rows) == 1
    assert Decimal(10) in rows[0].values()
    assert rows[0]["source_field:source_categories:id"] == "category-1"


def test_alternative_source_branches_union_shared_identity_without_join_scoping():
    from fervis.lookup.relation_catalog.row_sources import (
        RowSourceCandidateKey,
        RowSourceKeyComponent,
    )
    from fervis.lookup.source_binding.model import SubjectObligationBinding

    *_, original = fixtures._compile_memory_count(({"event_id": "event-1"},))
    source = replace(
        original.request.source_catalog.sources[0],
        candidate_keys=(
            RowSourceCandidateKey(
                "primary_key",
                "event",
                (RowSourceKeyComponent("event_id", "event_id"),),
                primary=True,
            ),
        ),
    )
    from fervis.lookup.relation_catalog.row_sources.builder import memory_row_source_id

    second = replace(
        source, id=memory_row_source_id("second_events"), memory_ref="second_events"
    )
    first_branch = original.request.strategy.branches[0]
    second_branch = replace(
        first_branch,
        branch_id=first_branch.branch_id + "_second",
        source_refs=(second.id,),
    )
    strategy = replace(
        original.request.strategy, branches=(first_branch, second_branch)
    )
    catalog = AvailableSourceCatalog(
        original.request.source_catalog.contract_snapshot, (source, second), ()
    )
    request = replace(original.request, strategy=strategy, source_catalog=catalog)
    subject_ref = request.index.subject_obligation.subject_set_ref.token
    original_set = original.binding_plan.set_bindings[subject_ref][0]
    realizations = tuple(
        replace(
            original_set,
            branch_id=branch.branch_id,
            source_ref=read.id,
            identity_ref=read.identity_evidence[0].identity_ref,
            identity_field_refs=read.identity_evidence[0].field_refs,
        )
        for branch, read in ((first_branch, source), (second_branch, second))
    )
    original_subject = original.binding_plan.subject_binding
    plan = replace(
        original.binding_plan,
        strategy=strategy,
        set_bindings={subject_ref: realizations},
        subject_binding=SubjectObligationBinding(
            subject_ref,
            tuple(
                replace(
                    original_subject.branch_realizations[0], branch_id=branch.branch_id
                )
                for branch in strategy.branches
            ),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    program = compile_verified_source_strategy(verified).answer_program
    data = (
        ({"event_id": "event-1"}, {"event_id": "event-2"}),
        ({"event_id": "event-2"}, {"event_id": "event-3"}),
    )
    execution = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    relation.id,
                    rows,
                    grain_keys=("event_id",),
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for relation, rows in zip(program.relations, data, strict=True)
            ),
            operations=tuple(
                ExecutableOperation(
                    operation.id, operation.spec, operation.output_relation
                )
                for operation in program.operations
            ),
        )
    )
    assert execution.issue is None
    assert tuple(
        execution.relation(program.operations[-1].output_relation).rows[0].values()
    ) == (3,)


def test_joined_alternative_branches_normalize_identity_and_values_before_union(
    monkeypatch,
):
    captured = {}
    original_verify = fixtures.verify_source_strategy

    def capture(plan, *, request):
        captured.update(plan=plan, request=request)
        return original_verify(plan, request=request)

    monkeypatch.setattr(fixtures, "verify_source_strategy", capture)
    fixtures.test_declared_association_compiles_to_existing_join_and_grouped_aggregate()
    original = captured["request"]
    source_map = {
        source.id: source.id + "_alternative"
        for source in original.source_catalog.sources
    }
    clones = tuple(
        replace(source, id=source_map[source.id])
        for source in original.source_catalog.sources
    )
    edge = original.source_catalog.relation_evidence[0]
    clone_edge = replace(
        edge,
        evidence_ref=edge.evidence_ref + "_alternative",
        left_source_ref=source_map[edge.left_source_ref],
        right_source_ref=source_map[edge.right_source_ref],
    )
    catalog = AvailableSourceCatalog(
        original.source_catalog.contract_snapshot,
        (*original.source_catalog.sources, *clones),
        (edge, clone_edge),
    )
    first_branch = original.strategy.branches[0]
    next_branch = replace(
        first_branch,
        branch_id=first_branch.branch_id + "_alternative",
        source_refs=tuple(source_map[ref] for ref in first_branch.source_refs),
        relation_evidence_refs=(clone_edge.evidence_ref,),
    )
    strategy = replace(original.strategy, branches=(first_branch, next_branch))

    def clone_realization(value):
        identity = (
            original.source_catalog.identity(value.identity_ref)
            if value.identity_ref
            else None
        )
        clone_identity = (
            next(
                (
                    item
                    for item in catalog.identity_evidence
                    if item.source_ref == source_map[value.source_ref]
                    and item.entity_kind == identity.entity_kind
                    and item.key_id == identity.key_id
                ),
                None,
            )
            if identity
            else None
        )
        return replace(
            value,
            branch_id=next_branch.branch_id,
            source_ref=source_map[value.source_ref],
            identity_ref=clone_identity.identity_ref if clone_identity else None,
        )

    plan = captured["plan"]
    plan = replace(
        plan,
        strategy=strategy,
        set_bindings={
            ref: (*values, *(clone_realization(value) for value in values))
            for ref, values in plan.set_bindings.items()
        },
        fact_bindings={
            ref: (*values, *(clone_realization(value) for value in values))
            for ref, values in plan.fact_bindings.items()
        },
        association_bindings={
            ref: (
                *values,
                *(
                    replace(
                        value,
                        branch_id=next_branch.branch_id,
                        source_refs=tuple(
                            source_map[source] for source in value.source_refs
                        ),
                        relation_evidence_ref=clone_edge.evidence_ref,
                    )
                    for value in values
                ),
            )
            for ref, values in plan.association_bindings.items()
        },
        subject_binding=replace(
            plan.subject_binding,
            branch_realizations=(
                *plan.subject_binding.branch_realizations,
                replace(
                    plan.subject_binding.branch_realizations[0],
                    branch_id=next_branch.branch_id,
                ),
            ),
        ),
    )
    request = replace(original, strategy=strategy, source_catalog=catalog)
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    program = compile_verified_source_strategy(verified).answer_program
    event1 = {"event_id": "event-1", "category_id": "category-1", "amount": Decimal(10)}
    event2 = {"event_id": "event-2", "category_id": "category-1", "amount": Decimal(20)}
    data = {
        "source_events": (event1,),
        "source_categories": ({"category_id": "category-1"},),
        "source_events_alternative": (event1, event2),
        "source_categories_alternative": ({"category_id": "category-1"},),
    }
    execution = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    relation.id,
                    data[relation.source.row_source_id],
                    grain_keys=("event_id",)
                    if "events" in relation.source.row_source_id
                    else ("category_id",),
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for relation in program.relations
            ),
            operations=tuple(
                ExecutableOperation(
                    operation.id, operation.spec, operation.output_relation
                )
                for operation in program.operations[:-1]
            ),
        )
    )
    assert execution.issue is None
    rows = execution.relation(program.operations[-2].output_relation).rows
    assert len(rows) == 1
    assert Decimal(30) in rows[0].values()


def test_missing_related_identity_cannot_form_an_entity_group(monkeypatch):
    captured = {}
    original = fixtures.verify_source_strategy

    def capture(plan, *, request):
        captured.update(plan=plan, request=request)
        return original(plan, request=request)

    monkeypatch.setattr(fixtures, "verify_source_strategy", capture)
    fixtures.test_declared_association_compiles_to_existing_join_and_grouped_aggregate()
    program = compile_verified_source_strategy(
        original(captured["plan"], request=captured["request"])
    ).answer_program
    rows = {
        "source_events": (
            {"event_id": "one", "category_id": "present", "amount": Decimal(10)},
            {"event_id": "two", "category_id": "missing", "amount": Decimal(100)},
        ),
        "source_categories": ({"category_id": "present"},),
    }
    result = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    relation.id,
                    rows[relation.source.row_source_id],
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for relation in program.relations
            ),
            operations=tuple(
                ExecutableOperation(op.id, op.spec, op.output_relation)
                for op in program.operations[:-1]
            ),
        )
    )
    assert result.issue is None
    groups = result.relation(program.operations[-2].output_relation).rows
    assert len(groups) == 1
    assert groups[0]["source_field:source_categories:category_id"] == "present"
    assert Decimal(10) in groups[0].values()
