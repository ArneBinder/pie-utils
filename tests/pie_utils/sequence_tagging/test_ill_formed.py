import pytest

from pie_utils.sequence_tagging.ill_formed import (
    InvalidTagSequence,
    fix_bio,
    fix_bioul,
    fix_boul,
    remove_bio,
    remove_bioul,
    remove_boul,
)

BIOUL_ENCODING_NAME = "BIOUL"
BOUL_ENCODING_NAME = "BOUL"


def test_fix_tag_sequence_with_ill_formed_boul_tag():
    # Incorrect Cases:
    # 1. If there is no L tag
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "O",
    ]
    new_boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "L-background_claim",
    ]
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence
    # 2. If there is U tag within a span
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "U-background_claim",
        "L-background_claim",
        "O",
    ]
    new_boul_sequence = ["O", "B-background_claim", "O", "O", "L-background_claim", "O"]

    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-background_claim",
        "O",
        "U-data",
        "L-background_claim",
        "O",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "U-data",
        "U-background_claim",
        "O",
    ]

    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "U-data",
        "B-background_claim",
        "L-background_claim",
    ]

    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "O",
        "L-background_claim",
    ]
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-data",
        "O",
    ]
    new_boul_sequence = [
        "U-data",
        "O",
    ]

    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-data",
        "L-data",
        "U-background_claim",
    ]

    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence
    # 3. If span starts with L

    boul_sequence = [
        "O",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "U-background_claim",
    ]
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-background_claim",
        "O",
        "O",
    ]
    new_boul_sequence = [
        "U-background_claim",
        "O",
        "O",
    ]
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    # L O L to O O L
    boul_sequence = [
        "L-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "O",
        "L-background_claim",
    ]
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "O",
        "O",
        "L-background_claim",
    ]
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "U-data",
        "B-background_claim",
        "L-background_claim",
    ]
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "L-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-data",
        "L-data",
        "B-background_claim",
        "L-background_claim",
    ]
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence


def test_fix_tag_sequence_with_ill_formed_bioul_tag():
    # Incorrect Cases:
    # 1. If there is no L tag
    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "O",  # "L-background_claim",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "L-background_claim",
        "O",  # "L-background_claim",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    # 2. If there is U tag within a span
    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "U-background_claim",
        "L-background_claim",
        "O",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
        "O",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "U-data",
        "L-background_claim",
        "O",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "U-data",
        "U-background_claim",
        "O",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    # 3. If there is O tag within a span
    bioul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "I-background_claim",
        "L-background_claim",
    ]

    new_bioul_sequence = [
        "O",
        "U-background_claim",
        "O",
        "B-background_claim",
        "L-background_claim",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    # 4. If span starts with I
    bioul_sequence = [
        "O",
        "I-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "U-background_claim",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "B-data",
        "I-data",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "B-data",
        "L-data",
        "U-background_claim",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "U-data",
        "O",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == bioul_sequence
    bioul_sequence = [
        "U-data",
        "O",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "U-data",
        "O",
        "U-background_claim",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "U-background_claim",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "L-background_claim",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence


def test_fix_tag_sequence_with_ill_formed_bio_tag():

    bio_sequence = [
        "B-background_claim",
        "B-background_claim",
        "I-background_claim",
    ]
    new_bio_sequence = [
        "B-background_claim",
        "B-background_claim",
        "I-background_claim",
    ]
    new_tag_sequence = fix_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "B-background_claim",
        "B-data",
        "I-background_claim",
    ]
    new_bio_sequence = [
        "B-background_claim",
        "B-data",
        "B-background_claim",
    ]
    new_tag_sequence = fix_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "B-background_claim",
        "B-data",
        "I-data",
    ]
    new_bio_sequence = [
        "B-background_claim",
        "B-data",
        "I-data",
    ]
    new_tag_sequence = fix_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = ["B-background_claim", "I-background_claim", "I-data", "O"]
    new_bio_sequence = [
        "B-background_claim",
        "I-background_claim",
        "B-data",
        "O",
    ]
    new_tag_sequence = fix_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "I-background_claim",
        "I-background_claim",
        "O",
    ]
    new_bio_sequence = [
        "B-background_claim",
        "I-background_claim",
        "O",
    ]
    new_tag_sequence = fix_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "B-background_claim",
        "I-background_claim",
        "B-data",
    ]
    new_bio_sequence = [
        "B-background_claim",
        "I-background_claim",
        "B-data",
    ]
    new_tag_sequence = fix_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence


def test_remove_ill_formed__bioul_tag_sequence():
    bioul_sequence = [
        "B-data",
        "I-data",
        "L-data",
        "O",
        "I-data",
        "B-data",
        "L-data",
    ]
    new_bioul_sequence = [
        "B-data",
        "I-data",
        "L-data",
        "O",
        "O",
        "B-data",
        "L-data",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-data",
        "I-data",
        "L-data",
        "O",
        "I-data",
        "U-data",
        "L-data",
    ]
    new_bioul_sequence = [
        "B-data",
        "I-data",
        "L-data",
        "O",
        "O",
        "U-data",
        "O",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "O",  # "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",  # "L-background_claim",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    # 2. If there is U tag within a span
    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "U-background_claim",
        "L-background_claim",
        "O",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "U-data",
        "L-background_claim",
        "O",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    # 3. If there is O tag within a span
    bioul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "I-background_claim",
        "L-background_claim",
    ]

    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    # 4. If span starts with I
    bioul_sequence = [
        "O",
        "I-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "B-data",
        "I-data",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "U-data",
        "O",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "U-data",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence


def test_remove_ill_formed_boul_tag_sequence():

    boul_sequence = [
        "B-data",
        "O",
        "L-data",
        "O",
        "O",
        "U-data",
    ]
    new_boul_sequence = [
        "B-data",
        "O",
        "L-data",
        "O",
        "O",
        "U-data",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "O",
        "L-data",
        "O",
        "B-data",
        "U-data",
        "L-data",
    ]
    new_boul_sequence = [
        "B-data",
        "O",
        "L-data",
        "O",
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "O",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence
    # 2. If there is U tag within a span
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "U-background_claim",
        "L-background_claim",
        "O",
    ]
    new_boul_sequence = ["O", "O", "O", "O", "O", "O"]

    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-background_claim",
        "O",
        "U-data",
        "L-background_claim",
        "O",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]

    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "U-data",
        "O",
        "O",
    ]

    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "U-background_claim",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-data",
        "O",
    ]
    new_boul_sequence = [
        "U-data",
        "O",
    ]

    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]

    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "O",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-background_claim",
        "O",
        "O",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "L-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-data",
        "L-data",
        "O",
        "O",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence


def test_remove_ill_formed__bio_tag_sequence():
    bio_sequence = [
        "B-data",
        "B-data",
        "I-data",
        "O",
    ]
    new_bio_sequence = [
        "B-data",
        "B-data",
        "I-data",
        "O",
    ]

    new_tag_sequence = remove_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "B-data",
        "I-data",
        "I-background_claim",
        "O",
    ]
    new_bio_sequence = [
        "B-data",
        "I-data",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "I-data",
        "B-data",
        "I-data",
        "O",
    ]
    new_bio_sequence = [
        "O",
        "B-data",
        "I-data",
        "O",
    ]
    new_tag_sequence = remove_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "B-background_claim",
        "B-background_claim",
        "I-background_claim",
    ]
    new_bio_sequence = [
        "B-background_claim",
        "B-background_claim",
        "I-background_claim",
    ]
    new_tag_sequence = remove_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "B-background_claim",
        "B-data",
        "I-background_claim",
    ]
    new_bio_sequence = [
        "B-background_claim",
        "B-data",
        "O",
    ]
    new_tag_sequence = remove_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "B-background_claim",
        "B-data",
        "I-data",
    ]
    new_bio_sequence = [
        "B-background_claim",
        "B-data",
        "I-data",
    ]
    new_tag_sequence = remove_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = ["B-background_claim", "I-background_claim", "I-data", "O"]
    new_bio_sequence = [
        "B-background_claim",
        "I-background_claim",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence

    bio_sequence = [
        "I-background_claim",
        "I-background_claim",
        "O",
    ]
    new_bio_sequence = [
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_bio(bio_sequence)
    assert new_tag_sequence == new_bio_sequence


def test_invalid_tag_sequence():
    bio_sequence = [
        "B-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    with pytest.raises(InvalidTagSequence):
        fix_bio(bio_sequence)

    bioul_sequence = [
        "U-background_claim",
        "M-background_claim",
    ]
    with pytest.raises(InvalidTagSequence, match=f"{bioul_sequence}"):
        fix_bioul(bioul_sequence)

    boul_sequence = [
        "U-background_claim",
        "M-background_claim",
    ]
    with pytest.raises(InvalidTagSequence):
        fix_boul(boul_sequence)
