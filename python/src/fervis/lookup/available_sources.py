"""Bounded current-run projection of retained source authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256

from fervis.lookup.canonical_data import canonical_runtime_json
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceField,
    RowSourceIdentityEvidence,
    RowSourceParam,
)
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogScalarParameterValue,
)
from fervis.lookup.read_eligibility import (
    SemanticReadDecision,
    SemanticReadEligibilityResult,
)
from fervis.types.enums import StrEnum


class SourceChoiceSurfaceKind(StrEnum):
    REQUEST_PARAMETER = "REQUEST_PARAMETER"
    RETURNED_FIELD = "RETURNED_FIELD"


@dataclass(frozen=True)
class SourceRelationEvidence:
    evidence_ref: str
    left_source_ref: str
    right_source_ref: str
    left_field_refs: tuple[str, ...]
    right_field_refs: tuple[str, ...]


@dataclass(frozen=True)
class SourceChoiceValue:
    value_ref: str
    source_ref: str
    surface_ref: str
    surface_kind: SourceChoiceSurfaceKind
    value: CatalogScalarParameterValue
    label: str


@dataclass(frozen=True)
class SourceChoiceSurface:
    surface_ref: str
    source_ref: str
    kind: SourceChoiceSurfaceKind
    label: str
    description: str
    values: tuple[SourceChoiceValue, ...]


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

    def source(self, source_ref: str) -> RowSource:
        for source in self.sources:
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
        )

    @property
    def choice_surfaces(self) -> tuple[SourceChoiceSurface, ...]:
        return tuple(
            _parameter_choice_surface(source, param)
            for source in self.sources
            for param in source.params
            if param.choices
        ) + tuple(
            _field_choice_surface(source, field)
            for source in self.sources
            for field in source.fields
            if field.choices
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
    values = tuple(
        SourceChoiceValue(
            value_ref=(f"source_choice:{source.id}:parameter:{param.id}:{position}"),
            source_ref=source.id,
            surface_ref=param.param_ref,
            surface_kind=SourceChoiceSurfaceKind.REQUEST_PARAMETER,
            value=value,
            label=(param.choice_labels or {}).get(str(value), str(value)),
        )
        for position, value in enumerate(param.choices)
    )
    return SourceChoiceSurface(
        surface_ref=param.param_ref,
        source_ref=source.id,
        kind=SourceChoiceSurfaceKind.REQUEST_PARAMETER,
        label=param.name,
        description=param.description,
        values=values,
    )


def _field_choice_surface(
    source: RowSource,
    field: RowSourceField,
) -> SourceChoiceSurface:
    values = tuple(
        SourceChoiceValue(
            value_ref=f"source_choice:{source.id}:field:{field.id}:{position}",
            source_ref=source.id,
            surface_ref=field.field_ref,
            surface_kind=SourceChoiceSurfaceKind.RETURNED_FIELD,
            value=value,
            label=str(value),
        )
        for position, value in enumerate(field.choices)
    )
    return SourceChoiceSurface(
        surface_ref=field.field_ref,
        source_ref=source.id,
        kind=SourceChoiceSurfaceKind.RETURNED_FIELD,
        label=field.label,
        description=field.description,
        values=values,
    )


def build_available_source_catalog(
    row_sources: RowSourceCatalog,
    *,
    read_eligibility: SemanticReadEligibilityResult,
    snapshot_namespace: str = "test",
) -> AvailableSourceCatalog:
    retained_by_assessment = tuple(
        (assessment, source_refs)
        for assessment in read_eligibility.read_assessments
        if assessment.decision is SemanticReadDecision.RETAIN
        for source_refs in (
            _retained_row_source_refs(
                assessment.source_refs,
                retained_field_refs=frozenset(
                    assessment.relevant_field_refs
                ),
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
    relations = _relation_evidence(sources)
    snapshot_payload = {
        "sources": [asdict(source) for source in sources],
        "relation_evidence": [asdict(item) for item in relations],
    }
    snapshot_content = canonical_runtime_json(snapshot_payload)
    return AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content(
            snapshot_content,
            namespace=snapshot_namespace,
        ),
        sources=sources,
        relation_evidence=relations,
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
    return tuple(source.id for source in sources if source.id in selected) or source_refs


def _row_path_depth(source: RowSource) -> int:
    return len(tuple(part for part in source.row_path.split(".") if part))


def _relation_evidence(
    sources: tuple[RowSource, ...],
) -> tuple[SourceRelationEvidence, ...]:
    evidence: list[SourceRelationEvidence] = []
    for left in sources:
        for reference in left.entity_references:
            local_by_component = {
                component.target_component_id: component.local_field_id
                for component in reference.components
            }
            for right in sources:
                if right.id == left.id:
                    continue
                for key in right.candidate_keys:
                    if (
                        key.entity_kind != reference.target_entity_kind
                        or key.id != reference.target_key_id
                    ):
                        continue
                    right_by_component = {
                        component.id: component.field_id for component in key.components
                    }
                    if set(local_by_component) != set(right_by_component):
                        continue
                    component_ids = tuple(sorted(local_by_component))
                    evidence_ref = (
                        f"source_relation:{left.id}:{reference.id}:{right.id}:{key.id}"
                    )
                    evidence.append(
                        SourceRelationEvidence(
                            evidence_ref=evidence_ref,
                            left_source_ref=left.id,
                            right_source_ref=right.id,
                            left_field_refs=tuple(
                                local_by_component[item] for item in component_ids
                            ),
                            right_field_refs=tuple(
                                right_by_component[item] for item in component_ids
                            ),
                        )
                    )
    return tuple(sorted(evidence, key=lambda item: item.evidence_ref))


__all__ = [
    "AvailableSourceCatalog",
    "SourceChoiceSurface",
    "SourceChoiceSurfaceKind",
    "SourceChoiceValue",
    "SourceRelationEvidence",
    "build_available_source_catalog",
]
