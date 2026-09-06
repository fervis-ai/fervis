"""Relation contract primitives for answer-program verification."""

from dataclasses import field

from ._shared import (
    FieldBindingRole,
    NamedExpression,
    VerificationError,
    dataclass,
)
from fervis.lookup.qualification import (
    BooleanAtomRef,
    QualificationAtomProof,
    QualificationGuarantee,
    SubjectGuarantee,
    and_dnf,
    or_dnf,
)


@dataclass(frozen=True)
class ProofLineage:
    value_refs: frozenset[str] = frozenset()

    @classmethod
    def value(cls, refs: frozenset[str]) -> "ProofLineage":
        return cls(value_refs=refs)

    def merge(self, *others: "ProofLineage") -> "ProofLineage":
        value_refs = set(self.value_refs)
        for other in others:
            value_refs.update(other.value_refs)
        return ProofLineage(value_refs=frozenset(value_refs))

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
class RelationSemanticGuarantee:
    qualification: QualificationGuarantee
    subject: SubjectGuarantee

    def __post_init__(self) -> None:
        if self.qualification.requested_fact_id != self.subject.requested_fact_id:
            raise ValueError("relation semantic guarantee must belong to one fact")


@dataclass(frozen=True)
class RelationContract:
    fields: dict[str, frozenset[FieldBindingRole]]
    grain_keys: tuple[str, ...]
    field_proofs: dict[str, ProofLineage]
    field_types: dict[str, str] = field(default_factory=dict)
    entity_keys: tuple[RelationEntityKey, ...] = ()
    row_proof: ProofLineage = ProofLineage()
    semantic_guarantees: dict[str, RelationSemanticGuarantee] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ScalarContract:
    proof: ProofLineage = ProofLineage()
    semantic_guarantees: dict[str, RelationSemanticGuarantee] = field(
        default_factory=dict
    )

    def combine(self, *others: "ScalarContract") -> "ScalarContract":
        operands = (self, *others)
        return ScalarContract(
            proof=ProofLineage(
                value_refs=frozenset(
                    ref for operand in operands for ref in operand.proof.value_refs
                )
            ),
            semantic_guarantees=_combine_semantic_guarantee_maps(
                tuple(operand.semantic_guarantees for operand in operands),
                disjoin=False,
            ),
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
        row_proof=contract.row_proof,
        semantic_guarantees=dict(contract.semantic_guarantees),
    )


def _combine_semantic_guarantees(
    contracts: tuple[RelationContract, ...],
    *,
    disjoin: bool,
) -> dict[str, RelationSemanticGuarantee]:
    return _combine_semantic_guarantee_maps(
        tuple(contract.semantic_guarantees for contract in contracts),
        disjoin=disjoin,
    )


def _combine_semantic_guarantee_maps(
    guarantee_maps: tuple[dict[str, RelationSemanticGuarantee], ...],
    *,
    disjoin: bool,
) -> dict[str, RelationSemanticGuarantee]:
    fact_ids = (
        set.intersection(*(set(item) for item in guarantee_maps))
        if disjoin and guarantee_maps
        else {fact_id for item in guarantee_maps for fact_id in item}
    )
    by_fact: dict[str, list[RelationSemanticGuarantee]] = {}
    for guarantee_map in guarantee_maps:
        for fact_id, guarantee in guarantee_map.items():
            if fact_id in fact_ids:
                by_fact.setdefault(fact_id, []).append(guarantee)
    output: dict[str, RelationSemanticGuarantee] = {}
    for fact_id, fact_guarantees in by_fact.items():
        subject = fact_guarantees[0].subject
        if any(item.subject != subject for item in fact_guarantees[1:]):
            raise VerificationError("combined relations disagree on subject guarantee")
        formula = fact_guarantees[0].qualification.formula
        operation = or_dnf if disjoin else and_dnf
        for item in fact_guarantees[1:]:
            formula = operation(
                fact_id,
                formula,
                item.qualification.formula,
            )
        proof_refs: dict[BooleanAtomRef, list[str]] = {}
        for guarantee in fact_guarantees:
            for proof in guarantee.qualification.atom_proofs:
                proof_refs.setdefault(proof.atom_ref, []).extend(proof.proof_refs)
        output[fact_id] = RelationSemanticGuarantee(
            qualification=QualificationGuarantee(
                requested_fact_id=fact_id,
                formula=formula,
                atom_proofs=tuple(
                    QualificationAtomProof(
                        atom_ref=atom,
                        proof_refs=tuple(dict.fromkeys(refs)),
                    )
                    for atom, refs in sorted(proof_refs.items())
                ),
            ),
            subject=subject,
        )
    return output


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


def _project_contract_grain(contract: RelationContract, outputs: tuple[NamedExpression, ...]) -> tuple[str, ...]:
    from fervis.lookup.plan_execution.expression_schema import projected_grain
    return projected_grain(contract.grain_keys, outputs)


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
        value_refs=frozenset(ref for proof in proofs for ref in proof.value_refs)
    )


def _join_contract_grain(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    return (*left, *(field for field in right if field not in left))
