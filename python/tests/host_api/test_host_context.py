from datetime import datetime, timezone

from fervis.host_api.context import HostContext


def test_default_host_context_resolves_utc_calendar_date() -> None:
    instant = datetime(2026, 7, 12, 23, 30, tzinfo=timezone.utc)

    assert HostContext().today(now=instant).isoformat() == "2026-07-12"
