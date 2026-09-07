"""Row-source contracts projected from the relation catalog."""

from __future__ import annotations

from fervis.host_api.contracts import ParameterSemantics

from fervis.host_api.contracts.population import ParameterPopulation

from dataclasses import dataclass
from typing import TYPE_CHECKING
from fervis.types.enums import StrEnum

from fervis.lookup.relation_catalog.model import (
    CatalogFactAvailability,
    EntityKeyComponentTarget,
    ParamSource,
    RowCardinality,
)
from fervis.lookup.relation_catalog.parameter_values import CatalogParameterValue
from fervis.lookup.answer_program.relations import FieldBindingRole

if TYPE_CHECKING:
    from fervis.lookup.semantic_types import ValueType


class RowSourceKind(StrEnum):
    API_READ = "api_read"
    MEMORY_READ = "memory_read"
    GENERATED_CALENDAR = "generated_calendar"


class RowSourceValueType(StrEnum):
    ANY = "any"
    ARRAY = "array"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    DATE = "date"
    DATETIME = "datetime"
    DECIMAL = "decimal"
    DOUBLE = "double"
    DURATION = "duration"
    FLOAT = "float"
    INTEGER = "integer"
    JSON = "json"
    LIST = "list"
    NUMBER = "number"
    OBJECT = "object"
    PATH = "path"
    PK = "pk"
    STRING = "string"
    TIME = "time"
    UUID = "uuid"
    UNKNOWN = "unknown"


CALENDAR_ROW_SOURCE_ID = "rs_calendar_days"
CALENDAR_DATE_FIELD_ID = "runtime_date"
CALENDAR_START_PARAM_ID = "interval_start"
CALENDAR_END_PARAM_ID = "interval_end"
CALENDAR_START_PARAM_REF = "__calendar__.interval_start"
CALENDAR_END_PARAM_REF = "__calendar__.interval_end"
CALENDAR_MAX_ROWS = 366
_MISSING = object()


def _finite_choices(
    value_type: RowSourceValueType, choices: tuple[str, ...]
) -> tuple[str, ...]:
    return choices or (
        ("false", "true") if value_type is RowSourceValueType.BOOLEAN else ()
    )


@dataclass(frozen=True)
class RowSourceField:
    id: str
    field_ref: str
    label: str
    type: RowSourceValueType
    allowed_roles: tuple[FieldBindingRole, ...]
    choices: tuple[str, ...] = ()
    fact_refs: tuple[str, ...] = ()
    answer_output_ids: tuple[str, ...] = ()
    path: str = ""
    response_path: str = ""
    description: str = ""
    declared_entity_kind: str = ""
    nullable: bool = False
    declared_value_domain: bool = True
    request_parameter_ref: str = ""

    @property
    def finite_choices(self) -> tuple[str, ...]:
        return _finite_choices(self.type, self.choices)

    @property
    def can_carry_lookup_text(self) -> bool:
        return not self.declared_entity_kind and self.type in {
            RowSourceValueType.STRING,
            RowSourceValueType.ARRAY,
            RowSourceValueType.LIST,
            RowSourceValueType.ANY,
        }


@dataclass(frozen=True)
class RowSourceParam:
    id: str
    param_ref: str
    name: str
    type: RowSourceValueType
    source: ParamSource | str = ""
    required: bool = False
    choices: tuple[str, ...] = ()
    choice_labels: dict[str, str] | None = None
    description: str = ""
    default: CatalogParameterValue = None
    default_is_known: bool = True
    default_source: str = ""
    entity_target: EntityKeyComponentTarget | None = None
    semantics: ParameterSemantics = ParameterSemantics.OPAQUE_QUERY_PARAM
    population: ParameterPopulation | None = None

    @property
    def finite_choices(self) -> tuple[str, ...]:
        return _finite_choices(self.type, self.choices)

    @property
    def accepts_lookup_text(self) -> bool:
        return self.type in {RowSourceValueType.STRING, RowSourceValueType.ANY}


@dataclass(frozen=True)
class RowSourceBlockedFact:
    fact_ref: str
    availability: CatalogFactAvailability
    field_id: str = ""
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RowSourceKeyComponent:
    id: str
    field_id: str


@dataclass(frozen=True)
class RowSourceCandidateKey:
    id: str
    entity_kind: str
    components: tuple[RowSourceKeyComponent, ...]
    primary: bool = False
    stable: bool = True
    context_field_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RowSourceEntityReferenceComponent:
    target_component_id: str
    local_field_id: str


@dataclass(frozen=True)
class RowSourceEntityReference:
    id: str
    target_entity_kind: str
    target_key_id: str
    components: tuple[RowSourceEntityReferenceComponent, ...]
    context_field_ids: tuple[str, ...] = ()


class RowSourceIdentityKind(StrEnum):
    ENTITY_ROW = "entity_row"
    ENTITY_REFERENCE = "entity_reference"


@dataclass(frozen=True)
class RowSourceIdentityEvidence:
    identity_ref: str
    kind: RowSourceIdentityKind
    source_ref: str
    entity_kind: str
    key_id: str
    field_refs: tuple[str, ...]


@dataclass(frozen=True)
class RowSourceRelationEvidence:
    evidence_ref: str
    left_source_ref: str
    right_source_ref: str
    left_field_refs: tuple[str, ...]
    right_field_refs: tuple[str, ...]


def row_source_relation_evidence(
    sources: tuple[RowSource, ...],
) -> tuple[RowSourceRelationEvidence, ...]:
    """Project declared entity references onto compatible candidate keys."""

    evidence: list[RowSourceRelationEvidence] = []
    for left in sources:
        for reference in left.entity_references:
            local_by_component = {
                component.target_component_id: component.local_field_id
                for component in reference.components
            }
            for right in sources:
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
                    evidence.append(
                        RowSourceRelationEvidence(
                            evidence_ref=(
                                f"source_relation:{left.id}:{reference.id}:"
                                f"{right.id}:{key.id}"
                            ),
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


def row_source_value_type(raw_value: str) -> RowSourceValueType:
    try:
        return RowSourceValueType(raw_value.strip().casefold())
    except ValueError:
        return RowSourceValueType.UNKNOWN


def row_source_value_type_is_scalar(value_type: RowSourceValueType) -> bool:
    return value_type not in {
        RowSourceValueType.ANY,
        RowSourceValueType.ARRAY,
        RowSourceValueType.JSON,
        RowSourceValueType.LIST,
        RowSourceValueType.OBJECT,
        RowSourceValueType.UNKNOWN,
    }


@dataclass(frozen=True)
class RowSource:
    id: str
    kind: RowSourceKind
    label: str
    read_id: str = ""
    endpoint_name: str = ""
    resource_names: tuple[str, ...] = ()
    memory_ref: str = ""
    description: str = ""
    row_path_id: str = ""
    row_path: str = ""
    parent_row_path: str = ""
    parent_row_cardinality: RowCardinality | None = None
    row_cardinality: RowCardinality = RowCardinality.MANY
    fields: tuple[RowSourceField, ...] = ()
    candidate_keys: tuple[RowSourceCandidateKey, ...] = ()
    entity_references: tuple[RowSourceEntityReference, ...] = ()
    params: tuple[RowSourceParam, ...] = ()
    blocked_facts: tuple[RowSourceBlockedFact, ...] = ()

    @property
    def stable_grain_field_refs(self) -> tuple[str, ...]:
        keys = tuple(key for key in self.candidate_keys if key.stable)
        key = next((key for key in keys if key.primary), keys[0] if keys else None)
        return (
            tuple(
                self.field(component.field_id).field_ref for component in key.components
            )
            if key is not None
            else ()
        )

    def fields_supporting_type(
        self, value_type: ValueType
    ) -> tuple[RowSourceField, ...]:
        """Return declared fields admissible under the shared value-type rules."""
        from .semantic_types import row_source_type_supports_semantic_type

        return tuple(
            field
            for field in self.fields
            if row_source_type_supports_semantic_type(field.type, value_type)
        )

    @property
    def request_argument_fields(self) -> tuple[RowSourceField, ...]:
        return tuple(RowSourceField(
            id=f"request_argument:{self.id}:{param.param_ref}",
            field_ref=f"request_argument:{self.id}:{param.param_ref}",
            label=f"request path argument {param.name}", type=param.type,
            allowed_roles=(FieldBindingRole.REQUEST_ARGUMENT,), choices=param.choices,
            description="The path argument used for this row's request, not a returned row property.",
            nullable=not param.required, request_parameter_ref=param.param_ref,
        ) for param in self.params if param.source == "path")

    def field(self, field_id: str) -> RowSourceField:
        for item in (*self.fields, *self.request_argument_fields):
            if item.id == field_id:
                return item
        raise KeyError(field_id)

    @property
    def identity_evidence(self) -> tuple[RowSourceIdentityEvidence, ...]:
        return tuple(
            RowSourceIdentityEvidence(
                identity_ref=f"source_identity:{self.id}:candidate_key:{key.id}",
                kind=RowSourceIdentityKind.ENTITY_ROW,
                source_ref=self.id,
                entity_kind=key.entity_kind,
                key_id=key.id,
                field_refs=tuple(
                    self.field(component.field_id).field_ref
                    for component in key.components
                ),
            )
            for key in self.candidate_keys
        ) + tuple(
            RowSourceIdentityEvidence(
                identity_ref=(
                    f"source_identity:{self.id}:entity_reference:{reference.id}"
                ),
                source_ref=self.id,
                kind=RowSourceIdentityKind.ENTITY_REFERENCE,
                entity_kind=reference.target_entity_kind,
                key_id=reference.target_key_id,
                field_refs=tuple(
                    self.field(component.local_field_id).field_ref
                    for component in reference.components
                ),
            )
            for reference in self.entity_references
        )

    @property
    def related_resource_field_ids(self) -> frozenset[str]:
        component_field_ids = (
            component.local_field_id
            for reference in self.entity_references
            for component in reference.components
        )
        context_field_ids = (
            field_id
            for reference in self.entity_references
            for field_id in reference.context_field_ids
        )
        return frozenset(component_field_ids) | frozenset(context_field_ids)

    def param(self, param_id: str) -> RowSourceParam:
        for item in self.params:
            if item.id == param_id:
                return item
        raise KeyError(param_id)


@dataclass(frozen=True)
class RowSourceCatalog:
    sources: tuple[RowSource, ...] = ()

    def find(self, source_id: str) -> RowSource | None:
        return next((item for item in self.sources if item.id == source_id), None)

    def source(self, source_id: str) -> RowSource:
        source = self.find(source_id)
        if source is not None:
            return source
        raise KeyError(source_id)
