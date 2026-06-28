"""Resolve structured time intents into inclusive date ranges."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .contract import AnchorSource, Field, IntentField, IntentKind, Policy, Status, Unit
from .date_values import normalize_date_value, resolve_date_value
from .expression_parser import intent_from_expression, local_relative_day_offset
from .intent_validation import validate_time_intent

DEFAULT_TIMEZONE = "UTC"

CALENDAR_UNITS = {Unit.DAY, Unit.WEEK, Unit.MONTH, Unit.QUARTER, Unit.YEAR}
ANCHOR_PERIOD_UNITS = {
    Unit.HOUR,
    Unit.DAY,
    Unit.WEEK,
    Unit.MONTH,
    Unit.QUARTER,
    Unit.YEAR,
}
ROLLING_WINDOW_UNITS = {Unit.DAY, Unit.WEEK, Unit.MONTH}
POINT_PRECISIONS = {Unit.DAY}


@dataclass(frozen=True)
class AnchorPeriod:
    unit: str
    start: date
    end: date

    def to_dict(self) -> dict[str, str]:
        return {
            IntentField.UNIT: self.unit,
            Field.START_DATE: self.start.isoformat(),
            Field.END_DATE: self.end.isoformat(),
        }


@dataclass(frozen=True)
class ResolverContext:
    expression: str
    anchor: date
    timezone: str
    anchor_period: AnchorPeriod | None = None
    anchor_source: str = AnchorSource.RUNTIME_DEFAULT


def resolve_time(
    expression: str = "",
    *,
    intent: dict[str, Any] | None = None,
    anchor_date: date | str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """Resolve a trusted time intent into inclusive YYYY-MM-DD dates."""

    ctx = ResolverContext(
        expression=str(expression or "").strip(),
        anchor=_coerce_date(anchor_date) or _today_in_timezone(timezone),
        timezone=timezone,
    )
    intent = _resolved_intent_from_inputs(ctx.expression, intent=intent)
    if not intent:
        return _needs_clarification(ctx, "Provide a structured time intent.")
    if local_relative_day_offset(
        ctx.expression
    ) is not None and not _meaningful_mapping(intent.get(IntentField.ANCHOR_PERIOD)):
        return _needs_clarification(ctx, "Provide an explicit temporal anchor.")
    validate_time_intent(intent)
    return _resolve_intent(ctx, intent)


def _resolved_intent_from_inputs(
    expression: str,
    *,
    intent: dict[str, Any] | None,
) -> dict[str, Any] | None:
    parsed = intent_from_expression(expression)
    if not intent:
        return parsed
    if not isinstance(intent, dict):
        return parsed
    if intent.get(IntentField.KIND):
        return intent
    if parsed is None:
        return intent
    return {
        **parsed,
        **(
            {IntentField.ANCHOR_PERIOD: intent[IntentField.ANCHOR_PERIOD]}
            if IntentField.ANCHOR_PERIOD in intent
            else {}
        ),
    }


def _resolve_intent(ctx: ResolverContext, intent: dict[str, Any]) -> dict[str, Any]:
    kind = _required_string(intent, IntentField.KIND)
    ctx = _context_for_intent(ctx, intent)
    if kind == IntentKind.POINT:
        precision = _required_string(intent, IntentField.PRECISION)
        if precision not in POINT_PRECISIONS:
            raise ValueError(f"Unsupported point precision: {precision}")
        if not intent.get(IntentField.VALUE):
            ctx = _ensure_anchor_period(ctx, Unit.DAY)
        value = _point_value(ctx, intent)
        if intent.get(IntentField.VALUE):
            resolved_ctx = _with_anchor_period(
                ctx,
                AnchorPeriod(unit=Unit.DAY, start=value, end=value),
                AnchorSource.EXPLICIT_VALUE,
            )
        else:
            resolved_ctx = _with_anchor(ctx, value)
        return _resolved(
            resolved_ctx,
            value,
            value,
            _normalized_point_intent(intent=intent, value=value),
        )
    if kind == IntentKind.RANGE:
        start = _date_value(ctx.anchor, intent.get(IntentField.START))
        end = _date_value(ctx.anchor, intent.get(IntentField.END))
        _validate_order(start, end)
        resolved_ctx = _with_anchor_period(
            ctx,
            AnchorPeriod(unit=Unit.DAY, start=start, end=end),
            AnchorSource.EXPLICIT_VALUE,
        )
        return _resolved(
            resolved_ctx,
            start,
            end,
            {
                IntentField.KIND: IntentKind.RANGE,
                IntentField.START: _normalized_date_value(
                    anchor=ctx.anchor,
                    raw=intent.get(IntentField.START),
                    value=start,
                ),
                IntentField.END: _normalized_date_value(
                    anchor=ctx.anchor,
                    raw=intent.get(IntentField.END),
                    value=end,
                ),
            },
        )
    if kind == IntentKind.PERIOD:
        unit = _required_string(intent, IntentField.UNIT)
        ctx = _ensure_anchor_period(ctx, unit)
        anchor = _anchor_for_period(ctx)
        mode = str(intent.get(IntentField.MODE) or Policy.FULL).strip() or Policy.FULL
        if mode not in {Policy.FULL, Policy.TO_DATE}:
            raise ValueError(f"Unsupported period mode: {mode}")
        named = _meaningful_mapping(intent.get(IntentField.NAMED))
        relative = _meaningful_mapping(intent.get(IntentField.RELATIVE))
        if named:
            start, end, normalized_named = _named_period_bounds(
                anchor=anchor,
                unit=unit,
                named=named,
            )
            output_intent = {
                IntentField.KIND: IntentKind.PERIOD,
                IntentField.UNIT: unit,
                IntentField.MODE: mode,
                IntentField.NAMED: normalized_named,
            }
        else:
            offset = 0
            if relative:
                offset = _optional_int(relative, IntentField.OFFSET, default=0)
            start, end = _calendar_unit_bounds(anchor, unit, offset)
            output_intent = {
                IntentField.KIND: IntentKind.PERIOD,
                IntentField.UNIT: unit,
                IntentField.MODE: mode,
                IntentField.RELATIVE: {IntentField.OFFSET: offset},
            }
        if mode == Policy.TO_DATE:
            end = min(anchor, end)
        if unit == Unit.WEEK:
            output_intent["week_starts_on"] = Policy.MONDAY
        return _resolved(_with_anchor(ctx, anchor), start, end, output_intent)
    if kind == IntentKind.WINDOW:
        ctx = _ensure_anchor_period(ctx, Unit.DAY)
        anchor = _anchor_for_period(ctx)
        unit = _required_string(intent, IntentField.UNIT).rstrip("s")
        count = _required_int(intent, IntentField.COUNT)
        direction = str(intent.get(IntentField.DIRECTION) or Policy.PAST).strip()
        if count < 1:
            raise ValueError("window intent requires count >= 1.")
        if direction == Policy.PAST:
            start = _rolling_window_start(anchor, unit, count)
            end = anchor
        elif direction == Policy.FUTURE:
            start = anchor
            end = _rolling_window_end(anchor, unit, count)
        else:
            raise ValueError(f"Unsupported window direction: {direction}")
        return _resolved(
            _with_anchor(ctx, anchor),
            start,
            end,
            {
                IntentField.KIND: IntentKind.WINDOW,
                IntentField.UNIT: unit,
                IntentField.COUNT: count,
                IntentField.DIRECTION: direction,
            },
        )
    if kind == IntentKind.OPEN_RANGE:
        ctx = _ensure_anchor_period(ctx, Unit.DAY)
        anchor = _anchor_for_period(ctx)
        start = _date_value(ctx.anchor, intent.get(IntentField.START))
        _validate_order(start, anchor)
        return _resolved(
            _with_anchor(ctx, anchor),
            start,
            anchor,
            {
                IntentField.KIND: IntentKind.OPEN_RANGE,
                IntentField.START: _normalized_date_value(
                    anchor=ctx.anchor,
                    raw=intent.get(IntentField.START),
                    value=start,
                ),
            },
        )
    raise ValueError(f"Unsupported structured time intent: {kind}")


def _point_value(ctx: ResolverContext, intent: dict[str, Any]) -> date:
    if intent.get(IntentField.VALUE):
        return _date_value(ctx.anchor, intent.get(IntentField.VALUE))
    relative = _meaningful_mapping(intent.get(IntentField.RELATIVE))
    if relative:
        unit = _required_string(relative, IntentField.UNIT)
        offset = _optional_int(relative, IntentField.OFFSET, default=0)
        if unit != Unit.DAY:
            raise ValueError("point.relative currently supports only unit=day.")
        return ctx.anchor + timedelta(days=offset)
    named = _meaningful_mapping(intent.get(IntentField.NAMED))
    if named and IntentField.WEEKDAY in named:
        weekday = _required_int(named, IntentField.WEEKDAY)
        offset = _optional_int(named, IntentField.OFFSET, default=0)
        return _weekday_date(ctx.anchor, weekday, offset)
    raise ValueError("point intent requires value, relative, or named weekday.")


def _normalized_point_intent(*, intent: dict[str, Any], value: date) -> dict[str, Any]:
    payload = {
        IntentField.KIND: IntentKind.POINT,
        IntentField.PRECISION: _required_string(intent, IntentField.PRECISION),
    }
    relative = _meaningful_mapping(intent.get(IntentField.RELATIVE))
    named = _meaningful_mapping(intent.get(IntentField.NAMED))
    if relative:
        payload[IntentField.RELATIVE] = {
            IntentField.UNIT: _required_string(relative, IntentField.UNIT),
            IntentField.OFFSET: _optional_int(
                relative,
                IntentField.OFFSET,
                default=0,
            ),
        }
        return payload
    if named and IntentField.WEEKDAY in named:
        payload[IntentField.NAMED] = {
            IntentField.WEEKDAY: _required_int(named, IntentField.WEEKDAY),
            IntentField.OFFSET: _optional_int(
                named,
                IntentField.OFFSET,
                default=0,
            ),
        }
        return payload
    payload[IntentField.VALUE] = _normalized_date_value(
        anchor=value,
        raw=intent.get(IntentField.VALUE),
        value=value,
    )
    return payload


def _date_value(anchor: date, raw: Any) -> date:
    return resolve_date_value(anchor, raw)


def _normalized_date_value(*, anchor: date, raw: Any, value: date) -> Any:
    del anchor
    return normalize_date_value(raw, value)


def _named_period_bounds(
    *,
    anchor: date,
    unit: str,
    named: dict[str, Any],
) -> tuple[date, date, dict[str, Any]]:
    value = _required_int(named, IntentField.VALUE)
    year = _optional_int(named, "year", default=None)
    if unit == Unit.MONTH:
        resolved_year = year if year is not None else _most_recent_year(anchor, value)
        start, end = _month_bounds(resolved_year, value)
        normalized_named = {IntentField.VALUE: value, "year": resolved_year}
        if year is None:
            normalized_named["year_policy"] = Policy.MOST_RECENT
        return start, end, normalized_named
    if unit == Unit.QUARTER:
        if value < 1 or value > 4:
            raise ValueError(
                "period.named.value must be a quarter number between 1 and 4."
            )
        resolved_year = (
            year if year is not None else _most_recent_quarter_year(anchor, value)
        )
        start_month = (value - 1) * 3 + 1
        start = date(resolved_year, start_month, 1)
        end = _month_bounds(resolved_year, start_month + 2)[1]
        normalized_named = {IntentField.VALUE: value, "year": resolved_year}
        if year is None:
            normalized_named["year_policy"] = Policy.MOST_RECENT
        return start, end, normalized_named
    if unit == Unit.YEAR:
        resolved_year = year if year is not None else value
        return (
            date(resolved_year, 1, 1),
            date(resolved_year, 12, 31),
            {"year": resolved_year},
        )
    raise ValueError(f"period.named is unsupported for unit={unit}.")


def _context_for_intent(
    ctx: ResolverContext, intent: dict[str, Any]
) -> ResolverContext:
    period = _anchor_period_from_intent(intent)
    if period is None:
        return ctx
    return ResolverContext(
        expression=ctx.expression,
        anchor=period.end,
        timezone=ctx.timezone,
        anchor_period=period,
        anchor_source=AnchorSource.INTENT,
    )


def _anchor_period_from_intent(intent: dict[str, Any]) -> AnchorPeriod | None:
    value = intent.get(IntentField.ANCHOR_PERIOD)
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("anchor_period must be an object.")
        unit = _required_string(value, IntentField.UNIT)
        if unit not in ANCHOR_PERIOD_UNITS:
            raise ValueError(f"Unsupported anchor_period unit: {unit}")
        start = _parse_iso_date(_required_string(value, IntentField.START_DATE))
        end = _parse_iso_date(_required_string(value, IntentField.END_DATE))
        _validate_order(start, end)
        return AnchorPeriod(unit=unit, start=start, end=end)
    return None


def _ensure_anchor_period(ctx: ResolverContext, unit: str) -> ResolverContext:
    if ctx.anchor_period is not None:
        return ctx
    start, end = _calendar_unit_bounds(ctx.anchor, _calendar_anchor_unit(unit), 0)
    return ResolverContext(
        expression=ctx.expression,
        anchor=ctx.anchor,
        timezone=ctx.timezone,
        anchor_period=AnchorPeriod(unit=unit, start=start, end=end),
        anchor_source=AnchorSource.RUNTIME_DEFAULT,
    )


def _calendar_anchor_unit(unit: str) -> str:
    return Unit.DAY if unit == Unit.HOUR else unit


def _anchor_for_period(ctx: ResolverContext) -> date:
    if ctx.anchor_period is None:
        return ctx.anchor
    if ctx.anchor_source == AnchorSource.RUNTIME_DEFAULT:
        return ctx.anchor
    return ctx.anchor_period.end


def _with_anchor(ctx: ResolverContext, anchor: date) -> ResolverContext:
    return ResolverContext(
        expression=ctx.expression,
        anchor=anchor,
        timezone=ctx.timezone,
        anchor_period=ctx.anchor_period,
        anchor_source=ctx.anchor_source,
    )


def _with_anchor_period(
    ctx: ResolverContext,
    period: AnchorPeriod,
    source: str,
) -> ResolverContext:
    return ResolverContext(
        expression=ctx.expression,
        anchor=period.end,
        timezone=ctx.timezone,
        anchor_period=period,
        anchor_source=source,
    )


def _calendar_unit_bounds(anchor: date, unit: str, offset: int) -> tuple[date, date]:
    if unit not in CALENDAR_UNITS:
        raise ValueError(f"Unsupported calendar unit: {unit}")
    if unit == Unit.DAY:
        value = anchor + timedelta(days=offset)
        return value, value
    if unit == Unit.WEEK:
        start = anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=offset)
        return start, start + timedelta(days=6)
    if unit == Unit.MONTH:
        start = _shift_month(anchor.replace(day=1), offset)
        return start, _month_bounds(start.year, start.month)[1]
    if unit == Unit.QUARTER:
        current_quarter = ((anchor.month - 1) // 3) + 1
        quarter_index = (anchor.year * 4) + current_quarter - 1 + offset
        year, zero_based_quarter = divmod(quarter_index, 4)
        start_month = (zero_based_quarter * 3) + 1
        start = date(year, start_month, 1)
        return start, _month_bounds(year, start_month + 2)[1]
    start = date(anchor.year + offset, 1, 1)
    return start, date(anchor.year + offset, 12, 31)


def _weekday_date(anchor: date, weekday: int, offset: int) -> date:
    if weekday < 0 or weekday > 6:
        raise ValueError("weekday must be between 0 and 6, where Monday is 0.")
    days_since = (anchor.weekday() - weekday) % 7
    return anchor - timedelta(days=days_since) + timedelta(weeks=offset)


def _rolling_window_start(anchor: date, unit: str, count: int) -> date:
    if unit not in ROLLING_WINDOW_UNITS:
        raise ValueError(f"Unsupported rolling window unit: {unit}")
    if unit == Unit.DAY:
        return anchor - timedelta(days=count - 1)
    if unit == Unit.WEEK:
        return anchor - timedelta(weeks=count) + timedelta(days=1)
    return _shift_month(anchor.replace(day=1), -count + 1)


def _rolling_window_end(anchor: date, unit: str, count: int) -> date:
    if unit not in ROLLING_WINDOW_UNITS:
        raise ValueError(f"Unsupported rolling window unit: {unit}")
    if unit == Unit.DAY:
        return anchor + timedelta(days=count - 1)
    if unit == Unit.WEEK:
        return anchor + timedelta(weeks=count) - timedelta(days=1)
    start = anchor.replace(day=1)
    target = _shift_month(start, count - 1)
    return _month_bounds(target.year, target.month)[1]


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12.")
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _most_recent_year(anchor: date, month: int) -> int:
    return anchor.year if month <= anchor.month else anchor.year - 1


def _most_recent_quarter_year(anchor: date, quarter: int) -> int:
    current_quarter = ((anchor.month - 1) // 3) + 1
    return anchor.year if quarter <= current_quarter else anchor.year - 1


def _shift_month(value: date, offset: int) -> date:
    month_index = value.year * 12 + value.month - 1 + offset
    year, zero_month = divmod(month_index, 12)
    return date(year, zero_month + 1, 1)


def _coerce_date(value: date | str | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return _parse_iso_date(value)


def _today_in_timezone(timezone_name: str) -> date:
    name = str(timezone_name or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        tz = ZoneInfo(name)
    except Exception as exc:
        raise ValueError(f"Unsupported timezone: {name}") from exc
    return datetime.now(UTC).astimezone(tz).date()


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _required_string(intent: dict[str, Any], field: str) -> str:
    value = str(intent.get(field) or "").strip()
    if not value:
        raise ValueError(f"{field} is required.")
    return value


def _required_int(intent: dict[str, Any], field: str) -> int:
    if field not in intent:
        raise ValueError(f"{field} is required.")
    return _strict_int(intent[field], field)


def _optional_int(
    intent: dict[str, Any],
    field: str,
    *,
    default: int | None,
) -> int | None:
    value = intent.get(field)
    if value is None:
        return default
    return _strict_int(value, field)


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} requires integer value.")
    return value


def _validate_order(start: date, end: date) -> None:
    if start > end:
        raise ValueError("start_date must be on or before end_date.")


def _resolved(
    ctx: ResolverContext,
    start: date,
    end: date,
    intent: dict[str, Any],
) -> dict[str, Any]:
    intent = dict(intent)
    anchor_period = _required_anchor_period(ctx).to_dict()
    payload = {
        Field.STATUS: Status.RESOLVED,
        Field.EXPRESSION: ctx.expression,
        Field.TIMEZONE: ctx.timezone,
        Field.ANCHOR_SOURCE: ctx.anchor_source,
        Field.ANCHOR_PERIOD: anchor_period,
        Field.START: start.isoformat(),
        Field.END: end.isoformat(),
        Field.INTENT: intent,
    }
    if ctx.anchor_source == AnchorSource.INTENT:
        intent.setdefault(IntentField.ANCHOR_PERIOD, anchor_period)
    return payload


def _needs_clarification(ctx: ResolverContext, question: str) -> dict[str, Any]:
    ctx = _ensure_anchor_period(ctx, Unit.DAY)
    return {
        Field.STATUS: Status.NEEDS_CLARIFICATION,
        Field.EXPRESSION: ctx.expression,
        Field.TIMEZONE: ctx.timezone,
        Field.ANCHOR_SOURCE: ctx.anchor_source,
        Field.ANCHOR_PERIOD: _required_anchor_period(ctx).to_dict(),
        Field.CLARIFICATION: question,
        Field.INTENT: None,
    }


def _required_anchor_period(ctx: ResolverContext) -> AnchorPeriod:
    if ctx.anchor_period is None:
        raise ValueError("anchor_period was not initialized.")
    return ctx.anchor_period


def _meaningful_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if item not in (None, "")}
