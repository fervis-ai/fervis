"""Canonical API read source-candidate grouping."""

from __future__ import annotations

from .model import RowSource, RowSourceKind


def endpoint_parameter_source(sources: tuple[RowSource, ...]) -> RowSource:
    """Return the canonical row-source owner for one endpoint's parameters."""

    if not sources:
        raise ValueError("endpoint parameter source requires a row source")
    read_ids = {source.read_id for source in sources}
    if len(read_ids) != 1:
        raise ValueError("endpoint parameter sources must belong to one read")
    return min(
        sources,
        key=lambda source: (
            len(tuple(part for part in source.row_path.split(".") if part)),
            source.id,
        ),
    )


def api_read_source_groups(
    sources: tuple[RowSource, ...],
) -> tuple[tuple[RowSource, ...], ...]:
    groups: dict[
        tuple[str, tuple[tuple[str, str], ...]],
        list[RowSource],
    ] = {}
    for source in sources:
        if source.kind != RowSourceKind.API_READ or not source.read_id:
            continue
        if not source.fields:
            continue
        groups.setdefault(_api_read_source_group_key(source), []).append(source)
    return tuple(tuple(group_sources) for group_sources in groups.values())


def read_row_source_counts(
    groups: tuple[tuple[RowSource, ...], ...],
) -> dict[str, int]:
    output: dict[str, int] = {}
    for group in groups:
        if not group:
            continue
        read_id = group[0].read_id
        output[read_id] = output.get(read_id, 0) + 1
    return output


def _api_read_source_group_key(
    source: RowSource,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        source.read_id,
        tuple(
            sorted(
                (
                    param.param_ref,
                    str(param.default),
                )
                for param in source.params
                if param.default_source == "source_variant" and param.required
            )
        ),
    )
