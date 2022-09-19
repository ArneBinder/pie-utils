import pytest
from pytorch_ie.utils.span import has_overlap

from pie_utils.span.slice import (
    distance,
    distance_center,
    distance_inner,
    distance_outer,
    get_overlap_len,
    is_contained_in,
)


def test_get_overlap_length():
    # default case : when indices_2 follows the indices_1 and both have overlap
    indices_1 = (0, 7)
    indices_2 = (3, 9)
    len = get_overlap_len(indices_1, indices_2)
    assert len == 4

    # when indices_1 follows the indices_2 and both have overlap
    indices_1 = (3, 5)
    indices_2 = (0, 4)
    len = get_overlap_len(indices_1, indices_2)
    assert len == 1

    # when indices have no overlap
    indices_1 = (0, 4)
    indices_2 = (4, 9)
    len = get_overlap_len(indices_1, indices_2)
    assert len == 0


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


@pytest.mark.parametrize("distance_type", ["inner", "outer", "center", "up"])
def test_distance(distance_type):
    start_end = (0, 4)
    other_start_end = (6, 10)
    if distance_type == "up":
        with pytest.raises(ValueError) as e:
            distance(start_end, other_start_end, distance_type)
            assert e.value == "unknown distance_type=up. use one of: center, inner, outer"
    else:
        distance_ = distance(start_end, other_start_end, distance_type)
        if distance_type == "inner":
            assert distance_ == 2.0
        elif distance_type == "outer":
            assert distance_ == 10.0
        else:
            assert distance_ == 6.0
