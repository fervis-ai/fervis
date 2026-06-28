from datetime import date

from fervis.lookup.grounding.time_resolution import resolve_time
from fervis.lookup.grounding.time_resolution import resolver as time_resolver


def test_default_anchor_uses_requested_timezone(monkeypatch):
    monkeypatch.setattr(
        time_resolver,
        "_today_in_timezone",
        lambda timezone_name: (
            date(2026, 4, 30)
            if timezone_name == "Pacific/Honolulu"
            else date(2026, 5, 1)
        ),
        raising=False,
    )

    result = resolve_time(
        "today",
        intent={
            "kind": "point",
            "precision": "day",
            "relative": {"unit": "day", "offset": 0},
        },
        timezone="Pacific/Honolulu",
    )

    assert {
        "timezone": result["timezone"],
        "start": result["start"],
        "end": result["end"],
    } == {
        "timezone": "Pacific/Honolulu",
        "start": "2026-04-30",
        "end": "2026-04-30",
    }
