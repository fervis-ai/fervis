"""Generated relation sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from fervis.lookup.plan_execution.errors import RelationEngineError
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessSourceKind,
    CompletenessStatus,
    PaginationCompleteness,
    RelationRows,
    RelationSetKind,
)


@dataclass(frozen=True)
class GeneratedCalendarRelationSource:
    id: str
    start: date
    end: date
    output_date_field: str
    max_rows: int


def generate_calendar_relation(
    calendar: GeneratedCalendarRelationSource,
) -> RelationRows:
    if not calendar.output_date_field:
        raise RelationEngineError("generated_calendar requires output date field")
    if calendar.max_rows < 1:
        raise RelationEngineError("generated_calendar requires positive max rows")
    if calendar.start > calendar.end:
        raise RelationEngineError("generated_calendar requires start <= end")

    days = (calendar.end - calendar.start).days + 1
    if days > calendar.max_rows:
        raise RelationEngineError("generated_calendar exceeds max rows")

    rows = tuple(
        {
            calendar.output_date_field: (
                calendar.start + timedelta(days=offset)
            ).isoformat()
        }
        for offset in range(days)
    )
    return RelationRows(
        id=calendar.id,
        rows=rows,
        grain_keys=(calendar.output_date_field,),
        field_types={calendar.output_date_field: "date"},
        field_answer_output_ids={},
        completeness=CompletenessProof(
            status=CompletenessStatus.COMPLETE,
            source_kind=CompletenessSourceKind.GENERATED_CALENDAR,
            set_kind=RelationSetKind.UNIVERSE,
            scope_fingerprint=(
                f"{calendar.start.isoformat()}..{calendar.end.isoformat()}"
            ),
            proof_refs=("generated_calendar",),
            pagination=PaginationCompleteness.NOT_PAGINATED,
        ),
    )
