"""Framework-neutral Fervis endpoint catalog contracts."""

from .endpoint import (
    CatalogEndpointContract,
    CandidateKeyContract,
    CandidateKeyComponentContract,
    CandidateKeyAuthorityComponentContract,
    CandidateKeyAuthorityContract,
    EndpointContract,
    EntityKeyComponentTargetContract,
    EntityReferenceComponentContract,
    EntityReferenceContract,
    FrameworkKind,
    ParameterContract,
    ResponseFieldContract,
    SourceNamespaceKind,
    make_catalog_endpoint_key,
)
from .authority import (
    ReadAuthority,
    ReadContextRef,
    read_context_ref_matches,
)
from .execution import CompiledReadRequest, ReadTransportOverlay
from .read import ReadInvocation
from .pagination import PaginationContract, PaginationKind
from .response_conformance import (
    DeclaredResponseShape,
    ObservedResponseShape,
    ResponseConformanceResult,
    check_response_conformance,
    declared_response_shape,
    observed_response_shape,
)
from fervis.host_api.contracts.credentials import DelegatedReadCredential

__all__ = [
    "CatalogEndpointContract",
    "CandidateKeyContract",
    "CandidateKeyComponentContract",
    "CandidateKeyAuthorityComponentContract",
    "CandidateKeyAuthorityContract",
    "CompiledReadRequest",
    "DeclaredResponseShape",
    "DelegatedReadCredential",
    "EndpointContract",
    "EntityKeyComponentTargetContract",
    "EntityReferenceComponentContract",
    "EntityReferenceContract",
    "FrameworkKind",
    "ObservedResponseShape",
    "ParameterContract",
    "PaginationContract",
    "PaginationKind",
    "ResponseFieldContract",
    "ResponseConformanceResult",
    "ReadAuthority",
    "ReadInvocation",
    "ReadTransportOverlay",
    "SourceNamespaceKind",
    "ReadContextRef",
    "check_response_conformance",
    "declared_response_shape",
    "observed_response_shape",
    "read_context_ref_matches",
    "make_catalog_endpoint_key",
]
