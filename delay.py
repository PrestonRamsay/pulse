from datetime import timedelta

LATE_THRESHOLD = timedelta(minutes=3)


def is_late(scheduled, actual):
    """Return True if actual is more than 3 minutes after scheduled."""
    return actual - scheduled > LATE_THRESHOLD
