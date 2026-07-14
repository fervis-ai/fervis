from __future__ import annotations

from fervis.types.enums import StrEnum


class Status:
    RESOLVED = "resolved"
    NEEDS_CLARIFICATION = "needs_clarification"


class Field:
    STATUS = "status"
    EXPRESSION = "expression"
    TIMEZONE = "timezone"
    ANCHOR_PERIOD = "anchor_period"
    ANCHOR_SOURCE = "anchor_source"
    START = "start"
    END = "end"
    START_DATE = "start_date"
    END_DATE = "end_date"
    INTENT = "intent"
    CLARIFICATION = "clarification_question"


class IntentKind:
    POINT = "point"
    RANGE = "range"
    PERIOD = "period"
    OPEN_RANGE = "open_range"
    WINDOW = "window"


class IntentField:
    KIND = "kind"
    PRECISION = "precision"
    VALUE = "value"
    START = "start"
    END = "end"
    RELATIVE = "relative"
    NAMED = "named"
    ANCHOR_PERIOD = "anchor_period"
    UNIT = "unit"
    OFFSET = "offset"
    MODE = "mode"
    WEEKDAY = "weekday"
    COUNT = "count"
    DIRECTION = "direction"
    START_DATE = "start_date"
    END_DATE = "end_date"
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    YEAR_POLICY = "year_policy"


class Unit:
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class Policy:
    FULL = "full"
    TO_DATE = "to_date"
    MOST_RECENT = "most_recent"
    MONDAY = "MONDAY"
    PAST = "past"
    FUTURE = "future"
    ANCHOR_DATE = "anchor_date"


class AnchorSource(StrEnum):
    RUNTIME_DEFAULT = "runtime_default"
    INTENT = "intent"
    EXPLICIT_VALUE = "explicit_value"
