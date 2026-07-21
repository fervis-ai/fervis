"""Relation contract primitives for fact-plan verification."""

from dataclasses import field

from ._shared import (
    FieldBindingRole,
    FieldRef,
    NamedExpression,
    VerificationError,
    dataclass,
)
from fervis.lookup.question_contract import MembershipTestRef


@dataclass(frozen=True)
class PopulationCoverage:
    row_tests: frozenset[MembershipTestRef] = frozenset()
    condition_tests: frozenset[MembershipTestRef] = frozenset()

    @property
    def all_tests(self) -> frozenset[MembershipTestRef]:
        return frozenset({*self.row_tests, *self.condition_tests})

    def additive(self, *others: "PopulationCoverage") -> "PopulationCoverage":
        row_tests = set(self.row_tests)
        condition_tests = set(self.condition_tests)
        for other in others:
            row_tests.update(other.row_tests)
            condition_tests.update(other.condition_tests)
        return PopulationCoverage(
            row_tests=frozenset(row_tests),
            condition_tests=frozenset(condition_tests),
        )

    @classmethod
    def guaranteed_by_every(
        cls, coverages: tuple["PopulationCoverage", ...]
    ) -> "PopulationCoverage":
        if not coverages:
            return cls()
        row_tests = set(coverages[0].row_tests)
        condition_tests = set(coverages[0].condition_tests)
        for coverage in coverages[1:]:
            row_tests.intersection_update(coverage.row_tests)
            condition_tests.intersection_update(coverage.condition_tests)
        return cls(
            row_tests=frozenset(row_tests),
            condition_tests=frozenset(condition_tests),
        )

    @classmethod
    def common_population_contributors(
        cls, coverages: tuple["PopulationCoverage", ...]
    ) -> "PopulationCoverage":
        population_coverages = tuple(
            coverage for coverage in coverages if coverage.all_tests
        )
        return cls.guaranteed_by_every(population_coverages)


@dataclass(frozen=True)
class ProofLineage:
    value_refs: frozenset[str] = frozenset()
    population_coverage: PopulationCoverage = PopulationCoverage()

    @classmethod
    def value(cls, refs: frozenset[str]) -> "ProofLineage":
        return cls(value_refs=refs)

    def with_population_coverage(self, coverage: PopulationCoverage) -> "ProofLineage":
        return ProofLineage(
            value_refs=self.value_refs,
            population_coverage=coverage,
        )

    def merge(self, *others: "ProofLineage") -> "ProofLineage":
        value_refs = set(self.value_refs)
        coverages = [self.population_coverage]
        for other in others:
            value_refs.update(other.value_refs)
            coverages.append(other.population_coverage)
        return ProofLineage(
            value_refs=frozenset(value_refs),
            population_coverage=PopulationCoverage.common_population_contributors(
                tuple(coverages)
            ),
        )

    def fulfillment_refs(self) -> frozenset[str]:
        return self.value_refs


@dataclass(frozen=True)
class RelationEntityKeyComponent:
    component_id: str
    field_id: str


@dataclass(frozen=True)
class RelationEntityKey:
    entity_kind: str
    key_id: str
    components: tuple[RelationEntityKeyComponent, ...]


@dataclass(frozen=True)
class RelationContract:
    fields: dict[str, frozenset[FieldBindingRole]]
    grain_keys: tuple[str, ...]
    field_proofs: dict[str, ProofLineage]
    field_types: dict[str, str] = field(default_factory=dict)
    entity_keys: tuple[RelationEntityKey, ...] = ()
    population_proof: ProofLineage = ProofLineage()


@dataclass(frozen=True)
class ScalarContract:
    proof: ProofLineage = ProofLineage()
    population_derived: bool = False

    def combine(self, *others: "ScalarContract") -> "ScalarContract":
        operands = (self, *others)
        population_operands = tuple(
            operand for operand in operands if operand.population_derived
        )
        return ScalarContract(
            proof=ProofLineage(
                value_refs=frozenset(
                    ref for operand in operands for ref in operand.proof.value_refs
                ),
                population_coverage=PopulationCoverage.guaranteed_by_every(
                    tuple(
                        operand.proof.population_coverage
                        for operand in population_operands
                    )
                ),
            ),
            population_derived=bool(population_operands),
        )


def _copy_contract(
    contracts: dict[str, RelationContract],
    relation_id: str,
) -> RelationContract:
    contract = _contract(contracts, relation_id)
    return RelationContract(
        fields=dict(contract.fields),
        grain_keys=contract.grain_keys,
        field_proofs=dict(contract.field_proofs),
        field_types=dict(contract.field_types),
        entity_keys=contract.entity_keys,
        population_proof=contract.population_proof,
    )


def _project_entity_keys(
    source: RelationContract,
    projections: dict[str, str],
) -> tuple[RelationEntityKey, ...]:
    projected: list[RelationEntityKey] = []
    for key in source.entity_keys:
        if any(component.field_id not in projections for component in key.components):
            continue
        components = tuple(
            RelationEntityKeyComponent(
                component_id=component.component_id,
                field_id=projections[component.field_id],
            )
            for component in key.components
        )
        projected.append(
            RelationEntityKey(
                entity_kind=key.entity_kind,
                key_id=key.key_id,
                components=components,
            )
        )
    return tuple(dict.fromkeys(projected))


def _combined_entity_keys(
    *contracts: RelationContract,
) -> tuple[RelationEntityKey, ...]:
    return tuple(
        dict.fromkeys(key for contract in contracts for key in contract.entity_keys)
    )


def _common_entity_keys(
    contracts: tuple[RelationContract, ...],
) -> tuple[RelationEntityKey, ...]:
    if not contracts:
        return ()
    common = set(contracts[0].entity_keys)
    for contract in contracts[1:]:
        common.intersection_update(contract.entity_keys)
    return tuple(key for key in contracts[0].entity_keys if key in common)


def _project_contract_grain(
    source: RelationContract,
    fields: tuple[NamedExpression, ...],
) -> tuple[str, ...]:
    if not source.grain_keys:
        return ()
    projections = {
        output.expression.field_id: output.output_field
        for output in fields
        if isinstance(output.expression, FieldRef)
    }
    if not all(field in projections for field in source.grain_keys):
        return ()
    return tuple(projections[field] for field in source.grain_keys)


def _contract(
    contracts: dict[str, RelationContract],
    relation_id: str,
) -> RelationContract:
    if relation_id not in contracts:
        raise VerificationError(f"operation references unknown input {relation_id}")
    return contracts[relation_id]


def _field_roles(
    contract: RelationContract,
    field: str,
    label: str,
) -> frozenset[FieldBindingRole]:
    if field not in contract.fields:
        raise VerificationError(f"{label} references unknown field")
    return contract.fields[field]


def _field_proof(
    contract: RelationContract,
    field: str,
    label: str,
) -> ProofLineage:
    if field not in contract.fields:
        raise VerificationError(f"{label} references unknown field")
    return contract.field_proofs.get(field, ProofLineage())


def _union_field_roles(
    contracts: dict[str, RelationContract],
    relation_ids: tuple[str, ...],
    field: str,
) -> frozenset[FieldBindingRole]:
    roles: set[FieldBindingRole] = set()
    for relation_id in relation_ids:
        roles.update(_field_roles(_contract(contracts, relation_id), field, "union"))
    return frozenset(roles)


def _union_field_proof(
    contracts: dict[str, RelationContract],
    relation_ids: tuple[str, ...],
    field: str,
) -> ProofLineage:
    proofs = tuple(
        _field_proof(_contract(contracts, relation_id), field, "union")
        for relation_id in relation_ids
    )
    return ProofLineage(
        value_refs=frozenset(ref for proof in proofs for ref in proof.value_refs),
        population_coverage=PopulationCoverage.guaranteed_by_every(
            tuple(proof.population_coverage for proof in proofs)
        ),
    )


def _join_contract_grain(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    return (*left, *(field for field in right if field not in left))
