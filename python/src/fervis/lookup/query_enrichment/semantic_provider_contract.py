"""Provider-authored semantic Query Enrichment DTOs."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderOutput


@dataclass(frozen=True)
class RecallBucketMatchOutput(ProviderOutput):
    bucket_ref: str
    exhaustive_resource_names: tuple[str, ...]
    matching_resource_names: tuple[str, ...]


@dataclass(frozen=True)
class InputResourceSearchTermsOutput(ProviderOutput):
    input_use_ref: str
    catalog_search_terms: tuple[str, ...]


@dataclass(frozen=True)
class SemanticQueryEnrichmentOutput(ProviderOutput):
    recall_bucket_matches: tuple[RecallBucketMatchOutput, ...]
    input_resource_search_terms: tuple[InputResourceSearchTermsOutput, ...]


__all__ = tuple(name for name in globals() if not name.startswith("_"))
