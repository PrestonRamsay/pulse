from datetime import datetime, timedelta

from delay import is_late

SCHEDULED = datetime(2026, 10, 15, 12, 0)


def test_exactly_3_minutes_late_is_not_late():
    assert is_late(SCHEDULED, SCHEDULED + timedelta(minutes=3)) is False


def test_3_minutes_1_second_late_is_late():
    assert is_late(SCHEDULED, SCHEDULED + timedelta(minutes=3, seconds=1)) is True


def test_5_minutes_early_is_not_late():
    assert is_late(SCHEDULED, SCHEDULED - timedelta(minutes=5)) is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
