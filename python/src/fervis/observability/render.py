"""Human-readable observability renderers."""

from __future__ import annotations

from enum import StrEnum

from fervis.observability.usage import RuntimeUsageReport


class UsageRenderDetail(StrEnum):
    COMPACT = "compact"
    VERBOSE = "verbose"
    DEBUG = "debug"

    def includes_verbose(self) -> bool:
        return self in {self.VERBOSE, self.DEBUG}

    def includes_debug(self) -> bool:
        return self is self.DEBUG


def render_usage_report(
    report: RuntimeUsageReport,
    *,
    detail: UsageRenderDetail = UsageRenderDetail.COMPACT,
) -> str:
    lines = [f"Usage {report.scope.value} {report.scope_id}"]
    if report.cost_micros_by_currency:
        totals = ", ".join(
            f"{currency} {_format_micros(amount)}"
            for currency, amount in sorted(report.cost_micros_by_currency.items())
        )
        lines.append(f"  cost total: {totals}")
    else:
        lines.append("  cost total: unavailable")
    if report.usage_totals:
        lines.append("  usage:")
        for (usage_kind, unit), quantity in sorted(
            report.usage_totals.items(),
            key=lambda item: (item[0][0].value, item[0][1].value),
        ):
            lines.append(f"    {usage_kind.value}: {quantity} {unit.value}")
    if report.duration_ms_total:
        lines.append(f"  inference time: {report.duration_ms_total} ms")
    if report.missing_cost_count:
        lines.append(f"  missing cost rows: {report.missing_cost_count}")
    if report.unpriced_usage_count:
        lines.append(f"  unpriced usage rows: {report.unpriced_usage_count}")
    if report.pricing_versions:
        lines.append(f"  pricing versions: {', '.join(report.pricing_versions)}")
    if detail.includes_verbose():
        _append_usage_calls(lines, report, detail=detail)
    return "\n".join(lines)


def _append_usage_calls(
    lines: list[str], report: RuntimeUsageReport, *, detail: UsageRenderDetail
) -> None:
    if report.calls:
        lines.append("  calls:")
        for call in report.calls:
            lines.append(
                f"    {call.step_key.value}#{call.call_index}: "
                f"{call.provider}/{call.model_key} {call.status.value}"
            )
            size_parts = [
                f"prompt={call.prompt_chars}",
                f"schema={call.schema_chars}",
                f"tool_spec={call.tool_spec_chars}",
            ]
            if call.raw_output_chars is not None:
                size_parts.append(f"raw_output={call.raw_output_chars}")
            lines.append(f"      chars: {', '.join(size_parts)}")
            if call.reasoning_effort:
                lines.append(f"      reasoning effort: {call.reasoning_effort}")
            if call.duration_ms is not None:
                lines.append(f"      duration: {call.duration_ms} ms")
            if detail.includes_debug() and call.usage_rows:
                lines.append("      usage rows:")
                for usage in call.usage_rows:
                    lines.append(
                        f"        {usage.usage_kind.value}: "
                        f"{usage.quantity} {usage.unit.value}"
                    )


def _format_micros(amount: int) -> str:
    whole = amount // 1_000_000
    fractional = amount % 1_000_000
    return f"{whole}.{fractional:06d}"
