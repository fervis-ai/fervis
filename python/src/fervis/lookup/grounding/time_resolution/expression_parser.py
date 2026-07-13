"""Deterministic time expression parsing."""

from __future__ import annotations

import re

from .contract import IntentField, IntentKind, Policy, Unit


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

RELATIVE_DAY_OFFSETS = {
    "yesterday": -1,
    "today": 0,
    "tomorrow": 1,
}
LOCAL_RELATIVE_DAY_OFFSETS = {
    "the day before": -1,
    "the previous day": -1,
    "previous day": -1,
    "the day after": 1,
    "the following day": 1,
    "following day": 1,
}
RELATIVE_PERIOD_OFFSETS = {
    "last": -1,
    "this": 0,
    "next": 1,
}

MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?"
)


def intent_from_expression(expression: str) -> dict[str, object] | None:
    text = _normalize(expression)
    if text in RELATIVE_DAY_OFFSETS:
        return {
            IntentField.KIND: IntentKind.POINT,
            IntentField.PRECISION: Unit.DAY,
            IntentField.RELATIVE: {
                IntentField.UNIT: Unit.DAY,
                IntentField.OFFSET: RELATIVE_DAY_OFFSETS[text],
            },
        }
    if text in LOCAL_RELATIVE_DAY_OFFSETS:
        return {
            IntentField.KIND: IntentKind.POINT,
            IntentField.PRECISION: Unit.DAY,
            IntentField.RELATIVE: {
                IntentField.UNIT: Unit.DAY,
                IntentField.OFFSET: LOCAL_RELATIVE_DAY_OFFSETS[text],
            },
        }
    composed_relative_day = _composed_relative_day_intent(text)
    if composed_relative_day is not None:
        return composed_relative_day
    if text in {"mtd", "month to date"}:
        return _relative_period(Unit.MONTH, offset=0, mode=Policy.TO_DATE)

    parsed = (
        _window_intent(text)
        or _open_range_intent(text)
        or _relative_period_intent(text)
        or _quarter_period_intent(text)
        or _month_day_intent(text)
        or _month_period_intent(text)
        or _iso_date_intent(text)
    )
    return parsed


def local_relative_day_offset(expression: str) -> int | None:
    return LOCAL_RELATIVE_DAY_OFFSETS.get(_normalize(expression))


def _composed_relative_day_intent(text: str) -> dict[str, object] | None:
    for local_text, local_offset in LOCAL_RELATIVE_DAY_OFFSETS.items():
        for anchor_text, anchor_offset in RELATIVE_DAY_OFFSETS.items():
            if text == f"{local_text} {anchor_text}":
                return {
                    IntentField.KIND: IntentKind.POINT,
                    IntentField.PRECISION: Unit.DAY,
                    IntentField.RELATIVE: {
                        IntentField.UNIT: Unit.DAY,
                        IntentField.OFFSET: local_offset + anchor_offset,
                    },
                }
    return None


def _window_intent(text: str) -> dict[str, object] | None:
    match = re.fullmatch(
        r"(last|past|previous|next|coming) (\d+) (day|days|week|weeks|month|months)",
        text,
    )
    if match is None:
        return None
    direction = Policy.FUTURE if match.group(1) in {"next", "coming"} else Policy.PAST
    return {
        IntentField.KIND: IntentKind.WINDOW,
        IntentField.UNIT: _singular_unit(match.group(3)),
        IntentField.COUNT: int(match.group(2)),
        IntentField.DIRECTION: direction,
    }


def _open_range_intent(text: str) -> dict[str, object] | None:
    match = re.fullmatch(r"since (.+)", text)
    if match is None:
        return None
    start = _date_value(match.group(1))
    if start is None:
        return None
    return {
        IntentField.KIND: IntentKind.OPEN_RANGE,
        IntentField.START: start,
    }


def _relative_period_intent(text: str) -> dict[str, object] | None:
    match = re.fullmatch(
        r"(last|this|next) (day|week|month|quarter|year)(?: (so far|to date))?",
        text,
    )
    if match is None:
        return None
    unit = match.group(2)
    offset = RELATIVE_PERIOD_OFFSETS[match.group(1)]
    mode = Policy.TO_DATE if match.group(3) or offset == 0 else Policy.FULL
    if unit == Unit.DAY:
        return {
            IntentField.KIND: IntentKind.POINT,
            IntentField.PRECISION: Unit.DAY,
            IntentField.RELATIVE: {
                IntentField.UNIT: Unit.DAY,
                IntentField.OFFSET: offset,
            },
        }
    return _relative_period(unit, offset=offset, mode=mode)


def _quarter_period_intent(text: str) -> dict[str, object] | None:
    match = re.fullmatch(r"(?:q|quarter )([1-4])(?: (\d{4}))?", text)
    if match is None:
        return None
    named: dict[str, object] = {IntentField.VALUE: int(match.group(1))}
    if match.group(2):
        named[IntentField.YEAR] = int(match.group(2))
    return {
        IntentField.KIND: IntentKind.PERIOD,
        IntentField.UNIT: Unit.QUARTER,
        IntentField.MODE: Policy.FULL,
        IntentField.NAMED: named,
    }


def _month_day_intent(text: str) -> dict[str, object] | None:
    value = _date_value(text)
    if value is None:
        return None
    return {
        IntentField.KIND: IntentKind.POINT,
        IntentField.PRECISION: Unit.DAY,
        IntentField.VALUE: value,
    }


def _month_period_intent(text: str) -> dict[str, object] | None:
    match = re.fullmatch(f"({MONTH_PATTERN})(?: (\\d{{4}}))?", text)
    if match is None:
        return None
    named: dict[str, object] = {IntentField.VALUE: MONTHS[match.group(1)]}
    if match.group(2):
        named[IntentField.YEAR] = int(match.group(2))
    return {
        IntentField.KIND: IntentKind.PERIOD,
        IntentField.UNIT: Unit.MONTH,
        IntentField.MODE: Policy.FULL,
        IntentField.NAMED: named,
    }


def _iso_date_intent(text: str) -> dict[str, object] | None:
    value = _iso_date_value(text)
    if value is None:
        return None
    return {
        IntentField.KIND: IntentKind.POINT,
        IntentField.PRECISION: Unit.DAY,
        IntentField.VALUE: value,
    }


def _relative_period(unit: str, *, offset: int, mode: str) -> dict[str, object]:
    return {
        IntentField.KIND: IntentKind.PERIOD,
        IntentField.UNIT: unit,
        IntentField.MODE: mode,
        IntentField.RELATIVE: {IntentField.OFFSET: offset},
    }


def _date_value(text: str) -> dict[str, int | str] | None:
    iso = _iso_date_value(text)
    if iso is not None:
        return iso
    match = re.fullmatch(f"({MONTH_PATTERN}) ([0-3]?\\d)(?: (\\d{{4}}))?", text)
    if match is None:
        return None
    value: dict[str, int | str] = {
        IntentField.MONTH: MONTHS[match.group(1)],
        IntentField.DAY: int(match.group(2)),
    }
    if match.group(3):
        value[IntentField.YEAR] = int(match.group(3))
    else:
        value[IntentField.YEAR_POLICY] = Policy.MOST_RECENT
    return value


def _iso_date_value(text: str) -> dict[str, int | str] | None:
    match = re.fullmatch(r"(\d{4}) ([0-1]?\d) ([0-3]?\d)", text)
    if match is None:
        return None
    return {
        IntentField.YEAR: int(match.group(1)),
        IntentField.MONTH: int(match.group(2)),
        IntentField.DAY: int(match.group(3)),
    }


def _normalize(expression: str) -> str:
    normalized = str(expression or "").casefold().strip()
    normalized = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", normalized)
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    normalized = normalized.replace("-", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _singular_unit(unit: str) -> str:
    return unit[:-1] if unit.endswith("s") else unit
