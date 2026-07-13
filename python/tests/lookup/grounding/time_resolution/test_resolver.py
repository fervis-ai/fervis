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


def test_explicit_calendar_date_uses_the_year_stated_in_the_expression():
    result = resolve_time(
        "January 1, 2030",
        intent={
            "kind": "point",
            "precision": "day",
            "value": {
                "month": 1,
                "day": 1,
                "year_policy": "most_recent",
            },
        },
        anchor_date="2026-07-13",
    )

    assert (result["start"], result["end"]) == ("2030-01-01", "2030-01-01")
