"""Canonical Boolean normalization and qualification-proof algebra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fervis.types.enums import StrEnum


MAX_QUALIFICATION_DNF_CLAUSES = 128


class BooleanPolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True, order=True)
class BooleanAtomRef:
    value_ref: str
    polarity: BooleanPolarity = BooleanPolarity.POSITIVE

    def __post_init__(self) -> None:
        if not self.value_ref:
            raise ValueError("Boolean atom requires value reference")

    def negated(self) -> BooleanAtomRef:
        return BooleanAtomRef(
            value_ref=self.value_ref,
            polarity=(
                BooleanPolarity.NEGATIVE
                if self.polarity is BooleanPolarity.POSITIVE
                else BooleanPolarity.POSITIVE
            ),
        )


@dataclass(frozen=True)
class QualificationClause:
    clause_ref: str
    atom_refs: tuple[BooleanAtomRef, ...]


@dataclass(frozen=True)
class QualificationDNF:
    clauses: tuple[QualificationClause, ...]

    @classmethod
    def true(cls, requested_fact_id: str) -> QualificationDNF:
        return qualification_dnf(requested_fact_id, (frozenset(),))

    @classmethod
    def false(cls) -> QualificationDNF:
        return cls(clauses=())


@dataclass(frozen=True)
class BooleanFormula:
    operator: str
    argument_refs: tuple[str, ...]


class BooleanRequirementUseSite(StrEnum):
    POPULATION = "population"
    AGGREGATE_FILTER = "aggregate_filter"
    QUANTIFIER_CONDITION = "quantifier_condition"
    COVERAGE_REQUIRED_MEMBER = "coverage_required_member"
    COVERAGE_CONDITION = "coverage_condition"


@dataclass(frozen=True)
class BooleanRequirement:
    requirement_ref: str
    atom_ref: BooleanAtomRef
    use_site: BooleanRequirementUseSite
    owner_expression_ref: str | None = None


@dataclass(frozen=True)
class QualificationAtomProof:
    atom_ref: BooleanAtomRef
    proof_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.proof_refs or any(not ref for ref in self.proof_refs):
            raise ValueError("qualification atomic proof requires evidence refs")


@dataclass(frozen=True)
class QualificationGuarantee:
    requested_fact_id: str
    formula: QualificationDNF
    atom_proofs: tuple[QualificationAtomProof, ...]

    def __post_init__(self) -> None:
        if not self.requested_fact_id:
            raise ValueError("qualification guarantee requires requested fact")
        formula_atoms = {
            atom for clause in self.formula.clauses for atom in clause.atom_refs
        }
        proved_atoms = tuple(proof.atom_ref for proof in self.atom_proofs)
        if len(set(proved_atoms)) != len(proved_atoms):
            raise ValueError("qualification guarantee repeats atomic proof")
        if set(proved_atoms) != formula_atoms:
            raise ValueError("qualification guarantee lacks atomic proof")


@dataclass(frozen=True)
class SubjectGuarantee:
    requested_fact_id: str
    subject_set_ref: str
    interpretation: str
    proof_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.requested_fact_id or not self.subject_set_ref:
            raise ValueError("subject guarantee requires fact and subject")
        if not self.interpretation or not self.proof_refs:
            raise ValueError("subject guarantee requires interpretation proof")


def qualification_dnf(
    requested_fact_id: str,
    clauses: tuple[frozenset[BooleanAtomRef], ...],
) -> QualificationDNF:
    if not requested_fact_id:
        raise ValueError("qualification requires requested fact")
    canonical: set[frozenset[BooleanAtomRef]] = set()
    for clause in clauses:
        polarities: dict[str, set[BooleanPolarity]] = {}
        for atom in clause:
            polarities.setdefault(atom.value_ref, set()).add(atom.polarity)
        if any(len(items) > 1 for items in polarities.values()):
            raise ValueError("qualification clause contains contradictory atom")
        canonical.add(frozenset(clause))
    ordered = sorted(
        canonical,
        key=lambda clause: tuple(sorted(clause)),
    )
    if len(ordered) > MAX_QUALIFICATION_DNF_CLAUSES:
        raise ValueError("qualification_dnf_clause_limit")
    return QualificationDNF(
        clauses=tuple(
            QualificationClause(
                clause_ref=f"{requested_fact_id}:qualification_clause:{index}",
                atom_refs=tuple(sorted(clause)),
            )
            for index, clause in enumerate(ordered, start=1)
        )
    )


def atom_dnf(requested_fact_id: str, atom: BooleanAtomRef) -> QualificationDNF:
    return qualification_dnf(requested_fact_id, (frozenset({atom}),))


def normalize_qualification(
    *,
    requested_fact_id: str,
    root_ref: str | None,
    formulas: Mapping[str, BooleanFormula | None],
) -> QualificationDNF:
    """Normalize one referenced Boolean DAG to canonical DNF."""
    if root_ref is None:
        return QualificationDNF.true(requested_fact_id)
    visiting: set[str] = set()
    completed: dict[tuple[str, bool], QualificationDNF] = {}

    def visit(value_ref: str, *, negated: bool) -> QualificationDNF:
        cache_key = (value_ref, negated)
        if cache_key in completed:
            return completed[cache_key]
        if value_ref in visiting:
            raise ValueError("Boolean formula contains a cycle")
        formula = formulas.get(value_ref)
        if formula is None:
            result = atom_dnf(
                requested_fact_id,
                BooleanAtomRef(
                    value_ref,
                    BooleanPolarity.NEGATIVE if negated else BooleanPolarity.POSITIVE,
                ),
            )
            completed[cache_key] = result
            return result
        visiting.add(value_ref)
        try:
            if formula.operator == "not":
                if len(formula.argument_refs) != 1:
                    raise ValueError("not requires one Boolean argument")
                result = visit(formula.argument_refs[0], negated=not negated)
            else:
                operator = formula.operator
                if operator not in {"and", "or"} or len(formula.argument_refs) < 2:
                    raise ValueError("invalid Boolean composition")
                if negated:
                    operator = "or" if operator == "and" else "and"
                parts = tuple(
                    visit(item, negated=negated) for item in formula.argument_refs
                )
                result = parts[0]
                for part in parts[1:]:
                    result = (
                        and_dnf(requested_fact_id, result, part)
                        if operator == "and"
                        else or_dnf(requested_fact_id, result, part)
                    )
        finally:
            visiting.remove(value_ref)
        completed[cache_key] = result
        return result

    return visit(root_ref, negated=False)


def and_dnf(
    requested_fact_id: str,
    left: QualificationDNF,
    right: QualificationDNF,
) -> QualificationDNF:
    clauses = tuple(
        frozenset(left_clause.atom_refs) | frozenset(right_clause.atom_refs)
        for left_clause in left.clauses
        for right_clause in right.clauses
    )
    return qualification_dnf(requested_fact_id, clauses)


def or_dnf(
    requested_fact_id: str,
    left: QualificationDNF,
    right: QualificationDNF,
) -> QualificationDNF:
    return qualification_dnf(
        requested_fact_id,
        tuple(
            frozenset(clause.atom_refs) for clause in (*left.clauses, *right.clauses)
        ),
    )


def not_dnf(
    requested_fact_id: str,
    value: QualificationDNF,
) -> QualificationDNF:
    result = QualificationDNF.true(requested_fact_id)
    for clause in value.clauses:
        negated_clause = QualificationDNF.false()
        for atom in clause.atom_refs:
            negated_clause = or_dnf(
                requested_fact_id,
                negated_clause,
                atom_dnf(requested_fact_id, atom.negated()),
            )
        result = and_dnf(requested_fact_id, result, negated_clause)
    return result


def qualification_entails(
    actual: QualificationDNF,
    requested: QualificationDNF,
) -> bool:
    return all(
        any(
            set(requested_clause.atom_refs).issubset(actual_clause.atom_refs)
            for requested_clause in requested.clauses
        )
        for actual_clause in actual.clauses
    )


def relative_dnf(
    requested_fact_id: str,
    *,
    given: QualificationDNF,
    condition: QualificationDNF,
) -> QualificationDNF:
    """Canonicalize a condition under facts guaranteed by an outer scope."""
    if not given.clauses:
        return QualificationDNF.true(requested_fact_id)
    guaranteed = set.intersection(*(set(clause.atom_refs) for clause in given.clauses))
    residual: set[frozenset[BooleanAtomRef]] = set()
    for clause in condition.clauses:
        if any(atom.negated() in guaranteed for atom in clause.atom_refs):
            continue
        residual.add(frozenset(set(clause.atom_refs) - guaranteed))
    minimal = tuple(
        clause for clause in residual if not any(other < clause for other in residual)
    )
    return qualification_dnf(requested_fact_id, minimal)


def boolean_requirements(
    *,
    requested_fact_id: str,
    formula: QualificationDNF,
    use_site: BooleanRequirementUseSite,
    owner_expression_ref: str | None = None,
) -> tuple[BooleanRequirement, ...]:
    atoms = sorted({atom for clause in formula.clauses for atom in clause.atom_refs})
    owner = owner_expression_ref or "qualification"
    return tuple(
        BooleanRequirement(
            requirement_ref=(
                f"{requested_fact_id}:{use_site.value}:{owner}:"
                f"{atom.value_ref}:{atom.polarity.value}"
            ),
            atom_ref=atom,
            use_site=use_site,
            owner_expression_ref=owner_expression_ref,
        )
        for atom in atoms
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
