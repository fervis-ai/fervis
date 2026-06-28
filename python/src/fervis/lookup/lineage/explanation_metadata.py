"""Non-model-facing metadata for lineage explanation projection."""

from __future__ import annotations

from typing import Any

from fervis.lineage.step_summary import StepSummaryDetail, StepSummaryItem


LINEAGE_EXPLANATION_PATHS = "lineage_explanation_paths"

LineageExplanationPath = tuple[str, ...]


def lineage_explanation_metadata(
    *paths: LineageExplanationPath,
) -> dict[str, list[list[str]]]:
    return {LINEAGE_EXPLANATION_PATHS: [list(path) for path in paths]}


def lineage_explanation_paths_from_payload(
    payload: dict[str, Any],
) -> tuple[LineageExplanationPath, ...]:
    raw_paths = payload.get(LINEAGE_EXPLANATION_PATHS)
    if not isinstance(raw_paths, list):
        return ()
    paths: list[LineageExplanationPath] = []
    for raw_path in raw_paths:
        if isinstance(raw_path, list) and all(
            isinstance(item, str) for item in raw_path
        ):
            paths.append(tuple(raw_path))
    return tuple(paths)


def lineage_explanation_items(
    source: dict[str, Any],
    *,
    metadata: tuple[LineageExplanationPath, ...],
) -> tuple[StepSummaryItem, ...]:
    return tuple(
        item
        for path in metadata
        for expanded_path in _expand_path(source, pattern=path)
        if (item := _lineage_explanation_item(source, path=expanded_path))
    )


def _lineage_explanation_item(
    source: dict[str, Any], *, path: LineageExplanationPath
) -> StepSummaryItem | None:
    value = _path_value(source, path=path)
    text = _explanation_text(value)
    if not text:
        return None
    return StepSummaryItem(
        text=f"{path[-1]} ({'.'.join(path)}): {text}",
        detail=StepSummaryDetail.VERBOSE,
        is_explanation=True,
        path=path,
    )


def _explanation_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float | bool):
        return str(value)
    return ""


def _path_value(source: object, *, path: LineageExplanationPath) -> object:
    current = source
    for part in path:
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
            continue
        return None
    return current


def _expand_path(
    source: object, *, pattern: LineageExplanationPath
) -> tuple[LineageExplanationPath, ...]:
    return tuple(_expand_path_parts(source, pattern=pattern, prefix=()))


def _expand_path_parts(
    current: object,
    *,
    pattern: LineageExplanationPath,
    prefix: LineageExplanationPath,
) -> list[LineageExplanationPath]:
    if not pattern:
        return [prefix]
    part, *rest = pattern
    rest_pattern = tuple(rest)
    if part == "*":
        if isinstance(current, dict):
            return [
                expanded
                for key, value in current.items()
                for expanded in _expand_path_parts(
                    value,
                    pattern=rest_pattern,
                    prefix=(*prefix, str(key)),
                )
            ]
        if isinstance(current, list):
            return [
                expanded
                for index, value in enumerate(current)
                for expanded in _expand_path_parts(
                    value,
                    pattern=rest_pattern,
                    prefix=(*prefix, str(index)),
                )
            ]
    return _expand_path_parts(
        _path_value(current, path=(part,)),
        pattern=rest_pattern,
        prefix=(*prefix, part),
    )
