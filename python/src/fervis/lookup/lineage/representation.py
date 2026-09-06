"""Retain source inspection responses at their own deterministic lineage step."""

from fervis.lineage.enums import SourceInspectionPhase

from fervis.lookup.lineage.source_read_buffer import buffered_source_read_lineage
from fervis.lookup.lineage.source_reads import record_source_read_observation
from fervis.lookup.source_reads.response import (
    observe_source_read_response,
    source_read_completeness,
)


class RepresentationInspectionAudit:
    def __init__(self, *, run_id, sink, phase=SourceInspectionPhase.SEMANTIC):
        self.sink = sink
        self.phase = SourceInspectionPhase(phase)
        self.buffered = buffered_source_read_lineage(
            run_id=run_id,
            step_id=sink.source_inspection_step_id(self.phase)
            if sink is not None
            else None,
        )

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

    def flush(self):
        if self.sink is not None and self.buffered.source_reads:
            self.sink.record_source_inspection(
                phase=self.phase,
                catalog_endpoints=self.buffered.catalog_endpoints,
                source_reads=self.buffered.source_reads,
                artifacts=self.buffered.artifacts,
            )
