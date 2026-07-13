"""Lookup relation catalog model and support logic."""

# ruff: noqa: F401

from fervis.lookup.relation_catalog.model import (
    CatalogEndpointMetadata,
    CatalogField,
    CatalogFact,
    CatalogFactAvailability,
    CatalogParam,
    CandidateKey,
    CandidateKeyAuthority,
    CandidateKeyAuthorityComponent,
    CandidateKeyComponent,
    CompletenessPolicy,
    EndpointRead,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
    FieldRequirement,
    PaginationMetadata,
    PaginationMode,
    ParamSource,
    RelationCatalog,
    ResponseEnvelopeMetadata,
    RowCardinality,
    RowPath,
)
from fervis.lookup.relation_catalog.ports import (
    RelationCatalogProvider,
    RelationDataAccessPort,
)
from fervis.lookup.relation_catalog.row_paths import infer_field_row_path_id
from fervis.lookup.relation_catalog.validation import (
    CatalogValidationError,
    parse_relation_catalog,
)
from fervis.lookup.relation_catalog.facts import (
    blocked_catalog_fact,
    catalog_fact_by_ref,
    catalog_facts,
)
from fervis.lookup.relation_catalog.interface_tokens import (
    CatalogInterfaceKind,
    CatalogInterfaceSide,
    CatalogInterfaceToken,
    catalog_input_param_token,
    catalog_output_field_refs_by_token,
    catalog_output_field_token,
)
from fervis.lookup.relation_catalog.identity import (
    entity_identity_field_ids,
    primary_stable_key_entity_kinds,
    read_has_primary_stable_key,
    source_field_is_entity_identity,
)
