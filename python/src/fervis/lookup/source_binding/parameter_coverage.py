"""Check known API predicates against explicitly admitted logical rows."""

from fervis.lookup.available_sources import SourceChoiceSurfaceKind
from fervis.lookup.question_contract import FactLocalRef, FactTerm
from fervis.lookup.relation_catalog.row_sources.model import RowSourceValueType
from fervis.lookup.semantic_types import BooleanType


def uncovered_parameter_scopes(plan, *, request) -> tuple[str, ...]:
    requirements = {
        item.requirement_ref: item for item in request.index.boolean_requirements
    }

    def physical_predicate(owner, branch_id):
        requirement = requirements.get(owner)
        if requirement is None:
            return ()
        ref = FactLocalRef.from_token(requirement.atom_ref.value_ref)
        if not isinstance(
            request.index.term_by_ref.get(ref), FactTerm
        ) or not isinstance(request.index.value_type(ref), BooleanType):
            return ()
        result = []
        for fact_ref, branch, source_ref, fields in request.realized_fact_fields:
            if fact_ref != ref.token or branch != branch_id or len(fields) != 1:
                continue
            source = request.source_catalog.source(source_ref)
            field = next(
                field for field in source.fields if field.field_ref == fields[0]
            )
            if field.type is RowSourceValueType.BOOLEAN:
                result.append(
                    (
                        source_ref,
                        fields[0],
                        requirement.atom_ref.polarity.value == "positive",
                    )
                )
        return tuple(result)

    # An explicit invocation predicate supplies a guarantee about a physical
    # field. That guarantee is invariant across occurrences of its producer.
    guarantees: dict[tuple[str, str, str, str, str], set[bool]] = {}
    for application in plan.invocation_applications:
        for source, field, truth in physical_predicate(
            application.owner_ref, application.branch_id
        ):
            if source != application.source_ref:
                continue
            for target in application.target_applications:
                guarantees.setdefault(
                    (
                        application.branch_id,
                        source,
                        target.target_ref,
                        application.value_ref,
                        field,
                    ),
                    set(),
                ).add(truth)
    failed = []
    for branch in plan.subject_binding.branch_realizations:
        for parameter in branch.surface_reviews:
            surface = request.source_catalog.choice_surface(parameter.surface_ref)
            if (
                parameter.owner_set_ref is None
                or surface.kind is not SourceChoiceSurfaceKind.REQUEST_PARAMETER
            ):
                continue
            for returned in branch.surface_reviews:
                other = request.source_catalog.choice_surface(returned.surface_ref)
                if (
                    returned.owner_set_ref != parameter.owner_set_ref
                    or other.source_ref != surface.source_ref
                    or other.kind is not SourceChoiceSurfaceKind.RETURNED_FIELD
                ):
                    continue
                requested = {
                    truth
                    for choice in returned.choice_reviews
                    for owner in choice.selection_requirement_refs
                    for source, field, truth in physical_predicate(
                        owner, branch.branch_id
                    )
                    if source == surface.source_ref and field == other.target_ref
                }
                admitted = [
                    guarantees.get(
                        (
                            branch.branch_id,
                            surface.source_ref,
                            surface.target_ref,
                            choice,
                            other.target_ref,
                        ),
                        set(),
                    )
                    for choice in parameter.included_choice_refs
                ]
                if (
                    requested
                    and admitted
                    and all(admitted)
                    and not requested <= set.union(*admitted)
                ):
                    failed.append(parameter.surface_ref)
    return tuple(dict.fromkeys(failed))
