"""Closed contracts for declared answer-program structural capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from fervis.types.enums import StrEnum

from fervis.lookup.answer_program.values import (
    ParameterBinding,
    ParameterDeclaration,
    ParameterValueType,
)
from fervis.lookup.answer_program.operations import PredicateOperator


class CapabilityKind(StrEnum):
    NARROW_COUNT = "narrow_population"


@dataclass(frozen=True)
class NarrowPopulationCapability:
    id: str
    parameter: ParameterDeclaration
    relation_id: str
    field_id: str
    operator: PredicateOperator
    requested_fact_ids: tuple[str, ...]
    proof_refs: tuple[str, ...]
    function_semantics_version: str = "1"
    kind: CapabilityKind = field(
        default=CapabilityKind.NARROW_COUNT,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.requested_fact_ids, tuple):
            raise TypeError("capability requested_fact_ids must be a tuple")
        if not isinstance(self.proof_refs, tuple):
            raise TypeError("capability proof_refs must be a tuple")
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("capability requires id")
        if not isinstance(self.parameter, ParameterDeclaration):
            raise TypeError("capability parameter must be ParameterDeclaration")
        if not isinstance(self.operator, PredicateOperator):
            raise TypeError("capability operator must be PredicateOperator")
        if (
            not isinstance(self.relation_id, str)
            or not self.relation_id
            or not isinstance(self.field_id, str)
            or not self.field_id
        ):
            raise ValueError("population capability requires relation and field")
        if (
            self.operator is not PredicateOperator.IN
            or self.parameter.value_type is not ParameterValueType.STRING_SET
        ):
            raise ValueError("population capability requires string-set membership")
        if not self.parameter.required:
            raise ValueError("structural capability parameter must be required")
        if not self.parameter.semantic_control_ref:
            raise ValueError(
                "structural capability parameter requires semantic-control identity"
            )
        if (
            not self.requested_fact_ids
            or any(
                not isinstance(fact_id, str) or not fact_id
                for fact_id in self.requested_fact_ids
            )
            or len(set(self.requested_fact_ids)) != len(self.requested_fact_ids)
        ):
            raise ValueError("population capability requires unique requested facts")
        if not self.proof_refs or any(
            not isinstance(ref, str) or not ref for ref in self.proof_refs
        ):
            raise ValueError("population capability requires proof obligations")
        if (
            not isinstance(self.function_semantics_version, str)
            or not self.function_semantics_version
        ):
            raise ValueError(
                "population capability requires function semantics version"
            )


@dataclass(frozen=True)
class CapabilityApplication:
    capability_id: str
    binding: ParameterBinding

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, str) or not self.capability_id:
            raise ValueError("capability application requires capability id")
        if not isinstance(self.binding, ParameterBinding):
            raise TypeError("capability application binding must be ParameterBinding")
