"""Logical read occurrences, distinct from their catalog producers."""

from dataclasses import dataclass

from fervis.lookup.question_contract import AssociationTerm, FactLocalRef
from fervis.lookup.relation_catalog.row_sources.model import RowSourceIdentityKind
from fervis.lookup.source_binding.model import AssociationRealizationKind


@dataclass(frozen=True)
class ReadOccurrence:
    id: str
    source_ref: str
    set_refs: tuple[str, ...]


@dataclass(frozen=True)
class OccurrenceLink:
    association_ref: str
    left_occurrence: str
    right_occurrence: str
    left_fields: tuple[str, ...]
    right_fields: tuple[str, ...]


@dataclass(frozen=True)
class OccurrenceScope:
    occurrences: tuple[ReadOccurrence, ...]
    links: tuple[OccurrenceLink, ...]

    def for_set(self, set_ref: str) -> ReadOccurrence:
        matches = tuple(item for item in self.occurrences if set_ref in item.set_refs)
        if len(matches) != 1:
            raise ValueError("logical set requires exactly one read occurrence")
        return matches[0]

    def for_source(self, source_ref: str) -> tuple[ReadOccurrence, ...]:
        return tuple(item for item in self.occurrences if item.source_ref == source_ref)

    def applications_for(
        self, request, plan, *, branch_id: str, occurrence: ReadOccurrence
    ):
        """The invocation applications actually allocated to one logical read."""
        return tuple(
            application
            for application in plan.invocation_applications
            if application.branch_id == branch_id
            and application.source_ref == occurrence.source_ref
            and self.owner_applies(
                request,
                application.owner_ref,
                source_ref=occurrence.source_ref,
                occurrence_ref=occurrence.id,
            )
        )

    def owner_applies(
        self, request, owner_ref: str | None, *, source_ref: str, occurrence_ref: str
    ) -> bool:
        from fervis.lookup.question_contract.domains import value_set_dependencies

        from fervis.lookup.source_binding.population_values import population_owner

        if owner_ref is not None and owner_ref.startswith("read_population:"):
            return any(
                owner_ref == population_owner(occurrence_ref, param.param_ref)
                for param in request.source_catalog.source(source_ref).params
            )

        requirement = next(
            (
                item
                for item in request.index.boolean_requirements
                if item.requirement_ref == owner_ref
            ),
            None,
        )
        if requirement is None:
            return True
        dependency = FactLocalRef.from_token(requirement.atom_ref.value_ref)
        owners = {
            self.for_set(ref).id
            for ref in value_set_dependencies(request.index, dependency)
            if self.for_set(ref).source_ref == source_ref
        }
        if len(owners) > 1:
            raise ValueError(
                "one invocation predicate cannot constrain multiple logical read roles"
            )
        return occurrence_ref in owners


def occurrence_scope(request, plan, branch_id: str) -> OccurrenceScope:
    bound = {
        ref: next(value for value in values if value.branch_id == branch_id)
        for ref, values in plan.set_bindings.items()
    }
    parent = {ref: ref for ref in bound}

    def root(ref):
        while parent[ref] != ref:
            ref = parent[ref]
        return ref

    def owns_row(ref):
        identity = bound[ref].identity_ref
        return (
            identity is None
            or request.source_catalog.identity(identity).kind
            is RowSourceIdentityKind.ENTITY_ROW
        )

    associations = []
    for ref, values in plan.association_bindings.items():
        term = request.index.term_by_ref[FactLocalRef.from_token(ref)]
        assert isinstance(term, AssociationTerm)
        left = request.index.fact_local_ref_by_local_id[term.from_set_ref].token
        right = request.index.fact_local_ref_by_local_id[term.to_set_ref].token
        value = next(item for item in values if item.branch_id == branch_id)
        associations.append((ref, left, right, value))
        if value.kind is AssociationRealizationKind.CO_RESIDENT:
            if bound[left].source_ref != bound[right].source_ref:
                raise ValueError("co-resident roles require one producer")
            if value.source_refs != (bound[left].source_ref,):
                raise ValueError("co-resident evidence must name its producer")
            if bound[left].membership is not None or bound[right].membership is not None:
                continue
            lroot, rroot = root(left), root(right)
            if lroot != rroot:
                members = tuple(ref for ref in bound if root(ref) in {lroot, rroot})
                if sum(owns_row(ref) for ref in members) > 1:
                    raise ValueError(
                        "co-residence cannot identify two independent entity-row roles"
                    )
                parent[rroot] = lroot
    groups: dict[str, list[str]] = {}
    for ref in bound:
        groups.setdefault(root(ref), []).append(ref)
    counts: dict[str, int] = {}
    for refs in groups.values():
        source_ref = bound[refs[0]].source_ref
        counts[source_ref] = counts.get(source_ref, 0) + 1
    occurrences = tuple(
        ReadOccurrence(
            id=bound[refs[0]].source_ref
            if counts[bound[refs[0]].source_ref] == 1
            else f"{bound[refs[0]].source_ref}@{refs[0]}",
            source_ref=bound[refs[0]].source_ref,
            set_refs=tuple(refs),
        )
        for refs in groups.values()
    )
    subject = request.index.subject_obligation.subject_set_ref.token
    occurrences = tuple(
        sorted(occurrences, key=lambda item: (subject not in item.set_refs, item.id))
    )
    temporary = OccurrenceScope(occurrences, ())
    links = []
    for ref, left_set, right_set, value in associations:
        if value.kind is AssociationRealizationKind.CO_RESIDENT:
            left, right = temporary.for_set(left_set), temporary.for_set(right_set)
            if left.id != right.id:
                source = request.source_catalog.source(left.source_ref)
                grain = source.stable_grain_field_refs
                if not grain:
                    raise ValueError("separate co-resident memberships require a stable source row key")
                fields = tuple(next(field.id for field in source.fields if field.field_ref == field_ref) for field_ref in grain)
                links.append(OccurrenceLink(ref, left.id, right.id, fields, fields))
            continue
        evidence = next(
            e
            for e in request.source_catalog.relation_evidence
            if e.evidence_ref == value.relation_evidence_ref
        )
        left, right = temporary.for_set(left_set), temporary.for_set(right_set)
        if evidence.left_source_ref == evidence.right_source_ref:
            if value.reference_from_set_ref not in {left_set, right_set}:
                raise ValueError("self-reference requires an explicit source-end role")
            if value.reference_from_set_ref == right_set:
                left, right = right, left
        elif left.source_ref != evidence.left_source_ref:
            left, right = right, left
        if (left.source_ref, right.source_ref) != (
            evidence.left_source_ref,
            evidence.right_source_ref,
        ):
            raise ValueError(
                "association endpoints disagree with their read occurrences"
            )
        links.append(
            OccurrenceLink(
                ref,
                left.id,
                right.id,
                evidence.left_field_refs,
                evidence.right_field_refs,
            )
        )
    return OccurrenceScope(occurrences, tuple(links))
