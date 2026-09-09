"""Bounded current-run projection of retained source authority."""

from __future__ import annotations

from fervis.lookup.answer_program.values import FactValue, LiteralType
from dataclasses import asdict, dataclass
from hashlib import sha256

from fervis.lookup.canonical_data import canonical_runtime_json
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceField,
    RowSourceIdentityEvidence,
    RowSourceParam,
    RowSourceValueType,
    RowSourceRelationEvidence,
    row_source_relation_evidence,
)
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogScalarParameterValue,
)
from fervis.lookup.read_eligibility import (
    SemanticReadDecision,
    SemanticReadEligibilityResult,
)
from fervis.types.enums import StrEnum
from fervis.lookup.source_reads.access_model import ReadAccessCatalog


class SourceChoiceSurfaceKind(StrEnum):
    REQUEST_PARAMETER = "REQUEST_PARAMETER"
    RETURNED_FIELD = "RETURNED_FIELD"


SourceRelationEvidence = RowSourceRelationEvidence


@dataclass(frozen=True)
class SourceChoiceValue:
    value_ref: str
    source_ref: str
    surface_ref: str
    surface_kind: SourceChoiceSurfaceKind
    value: CatalogScalarParameterValue
    label: str
    declared_type: RowSourceValueType

    @property
    def boolean_value(self) -> bool | None:
        if self.declared_type is not RowSourceValueType.BOOLEAN:
            return None
        from fervis.lookup.plan_execution.declared_values import parse_declared_value

        value = parse_declared_value(self.value, "boolean")
        assert isinstance(value, bool)
        return value


@dataclass(frozen=True)
class SourceChoiceSurface:
    surface_ref: str
    source_ref: str
    target_ref: str
    kind: SourceChoiceSurfaceKind
    label: str
    description: str
    values: tuple[SourceChoiceValue, ...]
    declared_entity_kind: str = ""


@dataclass(frozen=True)
class SourceFieldBinding:
    source_ref: str
    field: RowSourceField

    @property
    def ref(self) -> str:
        return f"source_field:{self.source_ref}:{self.field.id}"


@dataclass(frozen=True)
class SourceContractSnapshot:
    ref: str
    content: str

    def __post_init__(self) -> None:
        fingerprint = sha256(self.content.encode("utf-8")).hexdigest()
        if not self.ref.startswith("source_contract:") or not self.ref.endswith(
            f":{fingerprint}"
        ):
            raise ValueError("source contract snapshot ref does not match its content")

    @classmethod
    def from_content(
        cls,
        content: str,
        *,
        namespace: str = "test",
    ) -> SourceContractSnapshot:
        if not namespace.strip() or ":" in namespace:
            raise ValueError("source contract snapshot namespace is invalid")
        fingerprint = sha256(content.encode("utf-8")).hexdigest()
        return cls(
            ref=f"source_contract:{namespace}:{fingerprint}",
            content=content,
        )


@dataclass(frozen=True)
class AvailableSourceCatalog:
    contract_snapshot: SourceContractSnapshot
    sources: tuple[RowSource, ...]
    relation_evidence: tuple[SourceRelationEvidence, ...]
    read_access: ReadAccessCatalog = ReadAccessCatalog()

    @property
    def execution_sources(self) -> tuple[RowSource, ...]:
        return tuple({**{source.id:source for source in self.read_access.sources}, **{source.id:source for source in self.sources}}.values())

    @property
    def field_bindings(self) -> tuple[SourceFieldBinding, ...]:
        return tuple(
            SourceFieldBinding(source.id, field)
            for source in self.sources
            for field in source.fields
        )

    def field_binding(self, ref: str) -> SourceFieldBinding:
        for binding in self.field_bindings:
            if binding.ref == ref:
                return binding
        raise ValueError("field binding references an unknown source field")

    def source(self, source_ref: str) -> RowSource:
        for source in self.execution_sources:
            if source.id == source_ref:
                return source
        raise KeyError(source_ref)

    def select(
        self,
        *,
        source_refs: frozenset[str],
        relation_evidence_refs: frozenset[str],
    ) -> AvailableSourceCatalog:
        known_sources = {source.id for source in self.sources}
        known_relations = {item.evidence_ref for item in self.relation_evidence}
        if not source_refs <= known_sources:
            raise ValueError("selected strategy references an unavailable source")
        if not relation_evidence_refs <= known_relations:
            raise ValueError(
                "selected strategy references unavailable relation evidence"
            )
        relations = tuple(
            item
            for item in self.relation_evidence
            if item.evidence_ref in relation_evidence_refs
        )
        if any(
            {item.left_source_ref, item.right_source_ref} - source_refs
            for item in relations
        ):
            raise ValueError("selected relation evidence leaves the source strategy")
        return AvailableSourceCatalog(
            contract_snapshot=self.contract_snapshot,
            sources=tuple(
                source for source in self.sources if source.id in source_refs
            ),
            relation_evidence=relations,
            read_access=self.read_access,
        )

    @property
    def choice_surfaces(self) -> tuple[SourceChoiceSurface, ...]:
        return tuple(
            _parameter_choice_surface(source, param)
            for source in self.sources
            for param in source.params
            if param.finite_choices
        ) + tuple(
            _field_choice_surface(source, field)
            for source in self.sources
            for field in source.fields
            if field.finite_choices
        )

    @property
    def choice_values(self) -> tuple[SourceChoiceValue, ...]:
        return tuple(
            value for surface in self.choice_surfaces for value in surface.values
        )

    def choice_surface(self, surface_ref: str) -> SourceChoiceSurface:
        for surface in self.choice_surfaces:
            if surface.surface_ref == surface_ref:
                return surface
        raise KeyError(surface_ref)

    def choice_surface_at(
        self,
        source_ref: str,
        target_ref: str,
        kind: SourceChoiceSurfaceKind,
    ) -> SourceChoiceSurface | None:
        return next(
            (
                surface
                for surface in self.choice_surfaces
                if surface.source_ref == source_ref
                and surface.target_ref == target_ref
                and surface.kind is kind
            ),
            None,
        )

    def choice_value(self, value_ref: str) -> SourceChoiceValue:
        for value in self.choice_values:
            if value.value_ref == value_ref:
                return value
        raise KeyError(value_ref)

    @property
    def identity_evidence(self) -> tuple[RowSourceIdentityEvidence, ...]:
        return tuple(
            evidence for source in self.sources for evidence in source.identity_evidence
        )

    def identity(self, identity_ref: str) -> RowSourceIdentityEvidence:
        for value in self.identity_evidence:
            if value.identity_ref == identity_ref:
                return value
        raise KeyError(identity_ref)


def _parameter_choice_surface(
    source: RowSource,
    param: RowSourceParam,
) -> SourceChoiceSurface:
    surface_ref = f"source_surface:{source.id}:parameter:{param.id}"
    values = tuple(
        SourceChoiceValue(
            value_ref=(f"source_choice:{source.id}:parameter:{param.id}:{position}"),
            source_ref=source.id,
            surface_ref=surface_ref,
            surface_kind=SourceChoiceSurfaceKind.REQUEST_PARAMETER,
            value=value,
            label=(param.choice_labels or {}).get(str(value), str(value)),
            declared_type=param.type,
        )
        for position, value in enumerate(param.finite_choices)
    )
    return SourceChoiceSurface(
        surface_ref=surface_ref,
        source_ref=source.id,
        target_ref=param.param_ref,
        kind=SourceChoiceSurfaceKind.REQUEST_PARAMETER,
        label=param.name,
        description=param.description,
        values=values,
    )


def _field_choice_surface(
    source: RowSource,
    field: RowSourceField,
) -> SourceChoiceSurface:
    surface_ref = f"source_surface:{source.id}:field:{field.id}"
    values = tuple(
        SourceChoiceValue(
            value_ref=f"source_choice:{source.id}:field:{field.id}:{position}",
            source_ref=source.id,
            surface_ref=surface_ref,
            surface_kind=SourceChoiceSurfaceKind.RETURNED_FIELD,
            value=value,
            label=str(value),
            declared_type=field.type,
        )
        for position, value in enumerate(field.finite_choices)
    )
    return SourceChoiceSurface(
        surface_ref=surface_ref,
        source_ref=source.id,
        target_ref=field.field_ref,
        kind=SourceChoiceSurfaceKind.RETURNED_FIELD,
        label=field.label,
        description=field.description,
        values=values,
        declared_entity_kind=field.declared_entity_kind,
    )


def build_available_source_catalog(
    row_sources: RowSourceCatalog,
    *,
    read_eligibility: SemanticReadEligibilityResult,
    snapshot_namespace: str = "test",
    read_access: ReadAccessCatalog = ReadAccessCatalog(),
    requested_fact_id: str | None = None,
) -> AvailableSourceCatalog:
    retained_by_assessment = tuple(
        (assessment, source_refs)
        for assessment in read_eligibility.read_assessments
        if requested_fact_id is None or assessment.requested_fact_id == requested_fact_id
        if assessment.decision is SemanticReadDecision.RETAIN
        for source_refs in (
            _retained_row_source_refs(
                assessment.source_refs,
                retained_field_refs=frozenset(assessment.relevant_field_refs),
                row_sources=row_sources,
            ),
        )
    )
    retained_refs = {
        source_ref
        for _assessment, source_refs in retained_by_assessment
        for source_ref in source_refs
    }
    known_refs = {source.id for source in row_sources.sources}
    if not retained_refs <= known_refs:
        raise ValueError("retained read references an unknown source")
    sources = tuple(
        source for source in row_sources.sources if source.id in retained_refs
    )
    return snapshot_source_catalog(sources, snapshot_namespace=snapshot_namespace, read_access=read_access)


def snapshot_source_catalog(
    sources: tuple[RowSource, ...], *, snapshot_namespace: str = "test",
    read_access: ReadAccessCatalog = ReadAccessCatalog(),
) -> AvailableSourceCatalog:
    """Capture source authority independently of a factual retention decision."""
    relations = row_source_relation_evidence(sources)
    snapshot_payload = {
        "sources": [asdict(source) for source in sources],
        "relation_evidence": [asdict(item) for item in relations],
    }
    if read_access.dependencies:
        snapshot_payload['read_access']=[{'source_ref':item.source_ref,'parent_source_ref':item.parent_source_ref,
                                         'arguments':[asdict(arg) for arg in item.arguments]} for item in read_access.dependencies]
    snapshot_content = canonical_runtime_json(snapshot_payload)
    return AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content(
            snapshot_content,
            namespace=snapshot_namespace,
        ),
        sources=sources,
        relation_evidence=relations,
        read_access=read_access,
    )


def _retained_row_source_refs(
    source_refs: tuple[str, ...],
    *,
    retained_field_refs: frozenset[str],
    row_sources: RowSourceCatalog,
) -> tuple[str, ...]:
    """Project each retained field onto its shallowest containing row set."""

    if not retained_field_refs or len(source_refs) < 2:
        return source_refs
    sources = tuple(row_sources.source(source_ref) for source_ref in source_refs)
    selected: set[str] = set()
    for field_ref in retained_field_refs:
        containing = tuple(
            source
            for source in sources
            if field_ref in {field.field_ref for field in source.fields}
        )
        if not containing:
            continue
        shallowest_depth = min(_row_path_depth(source) for source in containing)
        selected.update(
            source.id
            for source in containing
            if _row_path_depth(source) == shallowest_depth
        )
    return (
        tuple(source.id for source in sources if source.id in selected) or source_refs
    )


def _row_path_depth(source: RowSource) -> int:
    return len(tuple(part for part in source.row_path.split(".") if part))


__all__ = [
    "AvailableSourceCatalog",
    "SourceChoiceSurface",
    "SourceChoiceSurfaceKind",
    "SourceChoiceValue",
    "SourceRelationEvidence",
    "build_available_source_catalog",
]


def source_choice_literal(choice, *, snapshot_ref: str) -> FactValue:
    return source_value_literal(
        value_ref=choice.value_ref, value=choice.value, declared_type=choice.declared_type,
        label=choice.label, source_ref=choice.source_ref,
        proof_refs=(snapshot_ref, choice.source_ref, choice.surface_ref, choice.value_ref),
    )


def source_value_literal(*, value_ref, value, declared_type, label, source_ref, proof_refs) -> FactValue:
    """One typed scalar representation for source choices and read controls."""
    from fervis.lookup.plan_execution.declared_values import parse_declared_value
    if declared_type is RowSourceValueType.BOOLEAN:
        literal_type = LiteralType.BOOLEAN
        value = "true" if parse_declared_value(value, "boolean") else "false"
    elif declared_type in {RowSourceValueType.INTEGER, RowSourceValueType.DECIMAL,
                           RowSourceValueType.NUMBER, RowSourceValueType.FLOAT, RowSourceValueType.DOUBLE}:
        literal_type, value = LiteralType.NUMBER, str(value)
    else:
        literal_type, value = LiteralType.STRING, str(value)
    return FactValue.literal(
        id=value_ref, literal_type=literal_type, value=value, label=label,
        proof_refs=proof_refs, source_refs=(source_ref,),
    )
