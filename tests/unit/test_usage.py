"""What a server answered, bucketed by hour and model, filtered by a time window."""

from datetime import UTC, datetime

from ruling.usage import UsageMeter

T0 = datetime(2026, 10, 16, 14, 20, tzinfo=UTC).timestamp()


def test_counts_land_in_the_hour_they_were_answered_in_per_model():
    clock = {"now": T0}
    meter = UsageMeter(now=lambda: clock["now"])
    meter.record("m", 1, 10)
    meter.record("tuned", 3, 30)
    clock["now"] += 1800                        # still 14:xx
    meter.record("m", 1, 5)
    clock["now"] += 1800                        # 15:20
    meter.record("m", 2, 7)
    assert meter.buckets() == [
        {"hour": "2026-10-16T14:00:00+00:00", "model": "m", "decisions": 2, "input_tokens": 15},
        {"hour": "2026-10-16T14:00:00+00:00", "model": "tuned", "decisions": 3, "input_tokens": 30},
        {"hour": "2026-10-16T15:00:00+00:00", "model": "m", "decisions": 2, "input_tokens": 7},
    ]


def test_the_window_keeps_hours_starting_within_it_and_reads_naive_times_as_utc():
    clock = {"now": T0}
    meter = UsageMeter(now=lambda: clock["now"])
    for _ in range(3):
        meter.record("m", 1, 1)
        clock["now"] += 3600
    hours = lambda **window: [b["hour"] for b in meter.buckets(**window)]
    assert hours(starting_on=datetime(2026, 10, 16, 15)) == ["2026-10-16T15:00:00+00:00", "2026-10-16T16:00:00+00:00"]
    assert hours(ending_before=datetime(2026, 10, 16, 15, tzinfo=UTC)) == ["2026-10-16T14:00:00+00:00"]
    # A start inside an hour leaves that hour out: its bucket began before the window.
    assert hours(starting_on=datetime(2026, 10, 16, 14, 30), ending_before=datetime(2026, 10, 16, 16)) == \
        ["2026-10-16T15:00:00+00:00"]
    assert meter.buckets(starting_on=datetime(2027, 1, 1)) == []
