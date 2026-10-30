"""Represent the policy command schedule at 30 Hz."""


def command_time_ns(index):
    """Return the timestamp of a nonnegative command index in nanoseconds."""
    if not isinstance(index, int) or index < 0:
        raise ValueError('The command index must be a nonnegative integer.')
    return round(index * 1_000_000_000 / 30)
