import pytest
from pytorch_ie.utils.span import has_overlap

from pie_utils.span.slice import (
    distance_center,
    distance_inner,
    distance_outer,
    is_contained_in,
)


def test_has_overlap():
    # if start_end[0] <= other_start_end[0] < start_end[1]
    start_end = (0, 4)
    other_start_end = (0, 6)
    assert has_overlap(start_end, other_start_end)

    start_end = (0, 4)
    other_start_end = (3, 6)
    assert has_overlap(start_end, other_start_end)

    # if start_end[0] < other_start_end[1] <= start_end[1]
    start_end = (0, 4)
    other_start_end = (3, 4)
    assert has_overlap(start_end, other_start_end)

    start_end = (0, 4)
    other_start_end = (2, 3)
    assert has_overlap(start_end, other_start_end)

    # reverse order of above cases
    # if other_start_end[0] <= start_end[0] < other_start_end[1]
    start_end = (0, 6)
    other_start_end = (0, 4)
    assert has_overlap(start_end, other_start_end)

    start_end = (3, 6)
    other_start_end = (0, 4)
    assert has_overlap(start_end, other_start_end)

    # other_start_end[0] < start_end[1] <= other_start_end[1]
    start_end = (2, 3)
    other_start_end = (0, 4)
    assert has_overlap(start_end, other_start_end)

    start_end = (3, 4)
    other_start_end = (0, 4)
    assert has_overlap(start_end, other_start_end)

    # Non overlapped
    start_end = (5, 6)
    other_start_end = (0, 4)
    assert not has_overlap(start_end, other_start_end)

    start_end = (0, 4)
    other_start_end = (4, 6)
    assert not has_overlap(start_end, other_start_end)


def test_is_contained_in():
    # if other_start_end[0] <= start_end[0] and start_end[1] <= other_start_end[1]
    start_end = (5, 6)
    other_start_end = (4, 7)
    assert is_contained_in(start_end, other_start_end)

    start_end = (4, 5)
    other_start_end = (4, 7)
    assert is_contained_in(start_end, other_start_end)

    start_end = (5, 7)
    other_start_end = (4, 7)
    assert is_contained_in(start_end, other_start_end)

    # not contained
    start_end = (5, 7)
    other_start_end = (7, 9)
    assert not is_contained_in(start_end, other_start_end)

    start_end = (0, 4)
    other_start_end = (4, 7)
    assert not is_contained_in(start_end, other_start_end)


def test_distance_center():
    start_end = (0, 4)
    other_start_end = (6, 10)
    distance = distance_center(start_end, other_start_end)
    assert distance == 6.0


def test_distance_outer():
    start_end = (0, 4)
    other_start_end = (6, 10)
    distance = distance_outer(start_end, other_start_end)
    assert distance == 10.0


def test_distance_inner():
    start_end = (0, 4)
    other_start_end = (2, 6)
    with pytest.raises(AssertionError) as e:
        distance_inner(start_end, other_start_end)
        assert e.value == "can not calculate inner span distance for overlapping spans"

    start_end = (0, 4)
    other_start_end = (4, 6)
    distance = distance_inner(start_end, other_start_end)
    assert distance == 0

    start_end = (4, 6)
    other_start_end = (0, 3)
    distance = distance_inner(start_end, other_start_end)
    assert distance == 1
