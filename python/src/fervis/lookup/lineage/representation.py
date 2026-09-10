"""Retain source inspection responses at their own deterministic lineage step."""

from fervis.lineage.enums import SourceInspectionPhase

from fervis.lookup.lineage.source_read_buffer import SourceInspectionAudit
from fervis.lookup.lineage.source_reads import record_source_read_observation
from fervis.lookup.source_reads.response import (
    observe_source_read_response,
    source_read_completeness,
)


class RepresentationInspectionAudit(SourceInspectionAudit):
    def __init__(self, *, run_id, sink, phase=SourceInspectionPhase.SEMANTIC):
        super().__init__(run_id=run_id, sink=sink, phase=SourceInspectionPhase(phase))

    def observe(self, read, result):
        record_source_read_observation(
            self.buffered.scope,
            source_read_key=read.id,
            endpoint_name=read.endpoint_name,
            catalog_endpoint=read.catalog_endpoint,
            args={},
            observation=observe_source_read_response(
                result, endpoint_name=read.endpoint_name
            ),
            response_body=result.get("responseBody"),
            completeness_json={
                **source_read_completeness(result),
                "responseFormat": result.get("responseFormat"),
            },
        )
