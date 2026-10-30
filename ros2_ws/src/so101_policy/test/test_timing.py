"""Check the 30 Hz schedule used by both policy execution modes."""

import pytest

from so101_policy.timing import command_time_ns


@pytest.mark.parametrize('index, expected', [
    (0, 0), (1, 33_333_333), (30, 1_000_000_000), (50, 1_666_666_667),
])
def test_command_timestamps(index, expected):
    assert command_time_ns(index) == expected


def test_full_chunk_has_strictly_increasing_timestamps():
    timestamps = [command_time_ns(index) for index in range(51)]
    intervals = [end - start for start, end in zip(timestamps, timestamps[1:])]
    assert set(intervals) == {33_333_333, 33_333_334}
    assert timestamps[-1] == 1_666_666_667


@pytest.mark.parametrize('index', [-1, 0.5])
def test_invalid_command_index_is_rejected(index):
    with pytest.raises(ValueError):
        command_time_ns(index)
