"""Relation contract primitives for fact-plan verification."""

from ._shared import FieldBindingRole, ProjectField, VerificationError, dataclass


@dataclass(frozen=True)
class ProofLineage:
    value_refs: frozenset[str] = frozenset()
    population_scope_refs: frozenset[str] = frozenset()

    @classmethod
    def value(cls, refs: frozenset[str]) -> "ProofLineage":
        return cls(value_refs=refs)

    def with_population_scope(self, refs: frozenset[str]) -> "ProofLineage":
        if not refs:
            return self
        return ProofLineage(
            value_refs=self.value_refs,
            population_scope_refs=frozenset({*self.population_scope_refs, *refs}),
        )

    def merge(self, *others: "ProofLineage") -> "ProofLineage":
        value_refs = set(self.value_refs)
        population_scope_refs = set(self.population_scope_refs)
        for other in others:
            value_refs.update(other.value_refs)
            population_scope_refs.update(other.population_scope_refs)
        return ProofLineage(
            value_refs=frozenset(value_refs),
            population_scope_refs=frozenset(population_scope_refs),
        )

    def fulfillment_refs(self) -> frozenset[str]:
        return frozenset({*self.value_refs, *self.population_scope_refs})


@dataclass(frozen=True)
class RelationContract:
    fields: dict[str, frozenset[FieldBindingRole]]
    grain_keys: tuple[str, ...]
    field_proofs: dict[str, ProofLineage]
    population_proof: ProofLineage = ProofLineage()


def _copy_contract(
    contracts: dict[str, RelationContract],
    relation_id: str,
) -> RelationContract:
    contract = _contract(contracts, relation_id)
    return RelationContract(
        fields=dict(contract.fields),
        grain_keys=contract.grain_keys,
        field_proofs=dict(contract.field_proofs),
        population_proof=contract.population_proof,
    )


def _project_contract_grain(
    source: RelationContract,
    fields: tuple[ProjectField, ...],
) -> tuple[str, ...]:
    if not source.grain_keys:
        return ()
    projections = {
        field.source: field.output or field.source
        for field in fields
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


def _relation_proof(contract: RelationContract) -> ProofLineage:
    proof = contract.population_proof
    for field_proof in contract.field_proofs.values():
        proof = proof.merge(field_proof)
    return proof


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
    proof = ProofLineage()
    for relation_id in relation_ids:
        proof = proof.merge(
            _field_proof(_contract(contracts, relation_id), field, "union")
        )
    return proof


def _join_contract_grain(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    return (*left, *(field for field in right if field not in left))
