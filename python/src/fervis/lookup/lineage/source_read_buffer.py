"""Buffer source-read lineage until its owning execution step is recorded."""

from __future__ import annotations

from dataclasses import dataclass, field

from fervis.lineage.recorder import CatalogEndpointWrite, SourceReadWrite
from fervis.lookup.lineage.source_reads import SourceReadLineageScope


@dataclass
class SourceReadLineageBuffer:
    catalog_endpoints: list[CatalogEndpointWrite] = field(default_factory=list)
    source_reads: list[SourceReadWrite] = field(default_factory=list)

    def record_catalog_endpoint(
        self,
        catalog_endpoint: CatalogEndpointWrite,
    ) -> CatalogEndpointWrite:
        self.catalog_endpoints.append(catalog_endpoint)
        return catalog_endpoint

    def record_source_read(self, source_read: SourceReadWrite) -> SourceReadWrite:
        self.source_reads.append(source_read)
        return source_read


@dataclass(frozen=True)
class BufferedSourceReadLineage:
    scope: SourceReadLineageScope | None = None
    buffer: SourceReadLineageBuffer | None = None

    @property
    def source_reads(self) -> tuple[SourceReadWrite, ...]:
        return tuple(self.buffer.source_reads) if self.buffer is not None else ()

    @property
    def catalog_endpoints(self) -> tuple[CatalogEndpointWrite, ...]:
        if self.buffer is None:
            return ()
        by_key: dict[tuple[str, str], CatalogEndpointWrite] = {}
        for endpoint in self.buffer.catalog_endpoints:
            key = (endpoint.run_id, endpoint.catalog_endpoint_key)
            existing = by_key.get(key)
            if existing is not None and existing != endpoint:
                raise ValueError(
                    "conflicting catalog endpoint lineage for "
                    f"{endpoint.catalog_endpoint_key!r} in run {endpoint.run_id!r}"
                )
            by_key[key] = endpoint
        return tuple(by_key.values())


def buffered_source_read_lineage(
    *,
    run_id: str,
    step_id: str | None,
) -> BufferedSourceReadLineage:
    if step_id is None:
        return BufferedSourceReadLineage()
    buffer = SourceReadLineageBuffer()
    return BufferedSourceReadLineage(
        scope=SourceReadLineageScope(
            run_id=run_id,
            step_id=step_id,
            recorder=buffer,
        ),
        buffer=buffer,
    )
