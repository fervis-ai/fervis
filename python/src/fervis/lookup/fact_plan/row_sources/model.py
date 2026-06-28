"""Row source public data model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fervis.lookup.relation_catalog.model import (
    CatalogFactAvailability,
    IdentityMetadata,
    ParamSource,
    RowCardinality,
)
from fervis.lookup.fact_plan.relations import FieldBindingRole


class RowSourceKind(StrEnum):
    API_READ = "api_read"
    MEMORY_READ = "memory_read"
    GENERATED_CALENDAR = "generated_calendar"


class RowSourceParamSemantics(StrEnum):
    OPAQUE_QUERY_PARAM = "opaque_query_param"
    POPULATION_FILTER = "population_filter"
    RESPONSE_SHAPE = "response_shape"


CALENDAR_ROW_SOURCE_ID = "rs_calendar_days"
CALENDAR_DATE_FIELD_ID = "runtime_date"
CALENDAR_START_PARAM_ID = "interval_start"
CALENDAR_END_PARAM_ID = "interval_end"
CALENDAR_START_PARAM_REF = "__calendar__.interval_start"
CALENDAR_END_PARAM_REF = "__calendar__.interval_end"
CALENDAR_MAX_ROWS = 366
_MISSING = object()


@dataclass(frozen=True)
class RowSourceField:
    id: str
    field_ref: str
    label: str
    type: str
    allowed_roles: tuple[FieldBindingRole, ...]
    choices: tuple[str, ...] = ()
    identity: IdentityMetadata | None = None
    fact_refs: tuple[str, ...] = ()
    answer_output_ids: tuple[str, ...] = ()
    path: str = ""
    response_path: str = ""
    description: str = ""


@dataclass(frozen=True)
class RowSourceParam:
    id: str
    param_ref: str
    name: str
    type: str
    source: ParamSource | str = ""
    required: bool = False
    choices: tuple[str, ...] = ()
    choice_labels: dict[str, str] | None = None
    default: object = None
    default_source: str = ""
    identity: IdentityMetadata | None = None
    semantics: RowSourceParamSemantics = RowSourceParamSemantics.OPAQUE_QUERY_PARAM


@dataclass(frozen=True)
class RowSourceBlockedFact:
    fact_ref: str
    availability: CatalogFactAvailability
    field_id: str = ""
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RowSource:
    id: str
    kind: RowSourceKind
    label: str
    read_id: str = ""
    resource_names: tuple[str, ...] = ()
    memory_ref: str = ""
    description: str = ""
    row_path_id: str = ""
    row_path: str = ""
    parent_row_path: str = ""
    row_cardinality: RowCardinality = RowCardinality.MANY
    fields: tuple[RowSourceField, ...] = ()
    params: tuple[RowSourceParam, ...] = ()
    blocked_facts: tuple[RowSourceBlockedFact, ...] = ()

    def field(self, field_id: str) -> RowSourceField:
        for item in self.fields:
            if item.id == field_id:
                return item
        raise KeyError(field_id)

    def param(self, param_id: str) -> RowSourceParam:
        for item in self.params:
            if item.id == param_id:
                return item
        raise KeyError(param_id)


@dataclass(frozen=True)
class RowSourceCatalog:
    sources: tuple[RowSource, ...] = ()

    def source(self, source_id: str) -> RowSource:
        for item in self.sources:
            if item.id == source_id:
                return item
        raise KeyError(source_id)
