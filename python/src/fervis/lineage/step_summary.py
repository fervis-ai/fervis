"""Typed run-step summary contract."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from typing import Any


STEP_SUMMARY_ITEMS = "step_summary_items"
STEP_SEMANTIC_ITEMS = "step_semantic_items"


class StepSummaryDetail(StrEnum):
    COMPACT = "compact"
    VERBOSE = "verbose"
    DEBUG = "debug"


@dataclass(frozen=True)
class StepSummaryItem:
    text: str
    detail: StepSummaryDetail = StepSummaryDetail.COMPACT
    is_explanation: bool = False
    path: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "detail": self.detail.value,
            "is_explanation": self.is_explanation,
            "path": list(self.path),
        }


@dataclass(frozen=True)
class StepSemanticItem:
    kind: str
    payload: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "payload": dict(self.payload),
        }


def step_summary_json(*items: StepSummaryItem) -> dict[str, object]:
    if not items:
        return {}
    return {STEP_SUMMARY_ITEMS: [item.to_json() for item in items]}


def step_semantic_json(*items: StepSemanticItem) -> dict[str, object]:
    if not items:
        return {}
    return {STEP_SEMANTIC_ITEMS: [item.to_json() for item in items]}


def merge_step_summary_json(
    summary: dict[str, object],
    *items: StepSummaryItem,
) -> dict[str, object]:
    output = dict(summary)
    if items:
        output[STEP_SUMMARY_ITEMS] = [
            item.to_json() for item in (*step_summary_items_from_json(summary), *items)
        ]
    return output


def merge_step_semantic_json(
    summary: dict[str, object],
    *items: StepSemanticItem,
) -> dict[str, object]:
    output = dict(summary)
    if items:
        output[STEP_SEMANTIC_ITEMS] = [
            item.to_json() for item in (*step_semantic_items_from_json(summary), *items)
        ]
    return output


def step_summary_items_from_json(
    payload: dict[str, object],
) -> tuple[StepSummaryItem, ...]:
    raw_items = payload.get(STEP_SUMMARY_ITEMS)
    if not isinstance(raw_items, list):
        return ()
    return tuple(
        item
        for raw_item in raw_items
        if isinstance(raw_item, dict)
        if (item := _step_summary_item(raw_item)) is not None
    )


def step_semantic_items_from_json(
    payload: dict[str, object],
) -> tuple[StepSemanticItem, ...]:
    raw_items = payload.get(STEP_SEMANTIC_ITEMS)
    if not isinstance(raw_items, list):
        return ()
    return tuple(
        item
        for raw_item in raw_items
        if isinstance(raw_item, dict)
        if (item := _step_semantic_item(raw_item)) is not None
    )


def _step_summary_item(raw: dict[Any, Any]) -> StepSummaryItem | None:
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    return StepSummaryItem(
        text=text,
        detail=_detail(raw.get("detail")),
        is_explanation=bool(raw.get("is_explanation")),
        path=tuple(str(item) for item in raw.get("path") or () if item),
    )


def _step_semantic_item(raw: dict[Any, Any]) -> StepSemanticItem | None:
    kind = str(raw.get("kind") or "").strip()
    payload = raw.get("payload")
    if not kind or not isinstance(payload, dict):
        return None
    return StepSemanticItem(
        kind=kind,
        payload={str(key): value for key, value in payload.items()},
    )


def _detail(value: object) -> StepSummaryDetail:
    try:
        return StepSummaryDetail(str(value or StepSummaryDetail.COMPACT.value))
    except ValueError:
        return StepSummaryDetail.COMPACT
