import pytest

from pie_utils.sequence_tagging.encoding import (
    bioul_to_boul,
    boul_to_bioul,
    tag_sequence_to_token_spans,
    token_spans_to_tag_sequence,
)

BIOUL_ENCODING_NAME = "BIOUL"
BIO_ENCODING_NAME = "IOB2"
BOUL_ENCODING_NAME = "BOUL"


@pytest.fixture
def char_to_token_mappings():
    return [
        {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            5: 2,
            6: 2,
            7: 2,
            8: 2,
            9: 2,
            11: 3,
            12: 3,
            14: 4,
            15: 4,
            16: 4,
            17: 4,
            18: 4,
            19: 4,
            20: 5,
            22: 6,
            23: 6,
            24: 6,
            25: 6,
            27: 7,
            28: 7,
            30: 8,
            31: 8,
            33: 9,
            34: 9,
            35: 9,
            36: 9,
            37: 9,
            38: 9,
            39: 9,
            40: 9,
            42: 10,
            43: 10,
            44: 10,
            45: 10,
            46: 10,
            48: 11,
            49: 11,
            50: 11,
            51: 11,
        },
        {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            8: 2,
            9: 2,
            11: 3,
            13: 4,
            14: 4,
            15: 4,
            16: 4,
            17: 4,
            19: 5,
            20: 5,
            21: 5,
            22: 5,
            23: 6,
            25: 7,
            26: 7,
            27: 7,
            28: 7,
            29: 7,
            31: 8,
            32: 8,
            33: 9,
            34: 9,
            35: 9,
            36: 10,
            38: 11,
            39: 11,
            41: 12,
            42: 12,
            43: 12,
            45: 13,
            46: 13,
            47: 13,
            48: 13,
            49: 14,
            50: 15,
            52: 16,
            53: 16,
            54: 16,
            55: 16,
            56: 16,
            57: 17,
        },
    ]


@pytest.fixture
def special_tokens_masks():
    return [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]


@pytest.fixture
def true_tag_sequences():
    return {
        BIOUL_ENCODING_NAME: [
            [
                "O",
                "U-person",
                "O",
                "O",
                "U-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "U-person",
                "O",
            ],
            [
                "O",
                "U-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "I-person",
                "I-person",
                "L-person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
            ],
        ],
        BOUL_ENCODING_NAME: [
            [
                "O",
                "U-person",
                "O",
                "O",
                "U-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "U-person",
                "O",
            ],
            [
                "O",
                "U-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "O",
                "O",
                "L-person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
            ],
        ],
        BIO_ENCODING_NAME: [
            [
                "O",
                "B-person",
                "O",
                "O",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "O",
            ],
            [
                "O",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "I-person",
                "I-person",
                "I-person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
            ],
        ],
    }


@pytest.mark.parametrize(
    "encoding",
    [BIO_ENCODING_NAME, BIOUL_ENCODING_NAME, BOUL_ENCODING_NAME, None],
)
def test_spans_to_tag_sequence(encoding, special_tokens_masks, true_tag_sequences):
    labeled_spans = [
        [("person", (1, 2)), ("city", (4, 5)), ("person", (11, 12))],
        [("city", (1, 2)), ("person", (7, 11))],
    ]
    base_sequence_lengths = [len(special_tokens_masks[0]), len(special_tokens_masks[1])]
    if encoding is None:
        with pytest.raises(ValueError) as e:
            labeled_span = labeled_spans[0]
            token_spans_to_tag_sequence(
                labeled_spans=labeled_span,
                base_sequence_length=base_sequence_lengths[0],
                coding_scheme="None",
            )
    else:
        labeled_span = labeled_spans[0]
        tag_sequence = token_spans_to_tag_sequence(
            labeled_spans=labeled_span,
            base_sequence_length=base_sequence_lengths[0],
            coding_scheme=encoding,
        )
        assert tag_sequence == true_tag_sequences[encoding][0]
        labeled_span = labeled_spans[1]
        tag_sequence = token_spans_to_tag_sequence(
            labeled_spans=labeled_span,
            base_sequence_length=base_sequence_lengths[1],
            coding_scheme=encoding,
        )
        assert tag_sequence == true_tag_sequences[encoding][1]


@pytest.mark.parametrize(
    "encoding",
    [BIOUL_ENCODING_NAME, BIO_ENCODING_NAME, BOUL_ENCODING_NAME, None],
)
def test_tag_sequence_to_span(encoding, true_tag_sequences):
    sequence_to_span = {
        BIOUL_ENCODING_NAME: [
            (
                ["O", "B-city", "O", "B-city", "U-city"],
                [
                    ("city", (1, 1)),
                    ("city", (3, 4)),
                ],
            ),
            (["B-city", "I-city", "I-person", "L-person"], [("city", (0, 1)), ("person", (2, 3))]),
            (["B-city", "I-city", "L-person"], [("city", (0, 1)), ("person", (2, 2))]),
            (["B-city", "I-city", "B-person"], [("city", (0, 1)), ("person", (2, 2))]),
            (["B-city", "I-city", "B-city"], [("city", (0, 2))]),
            (
                ["L-city", "I-city", "L-person"],
                [("city", (0, 0)), ("city", (1, 1)), ("person", (2, 2))],
            ),
            (["B-city", "I-city", "I-city"], [("city", (0, 2))]),
            (["B-city", "O", "U-city"], [("city", (0, 0)), ("city", (2, 2))]),
        ],
        BOUL_ENCODING_NAME: [
            (
                ["O", "B-city", "O", "B-city", "U-city"],  # what is this O is actual O not I
                [("city", (1, 4))],
            ),
            (["B-city", "O", "O", "L-person"], [("city", (0, 2)), ("person", (3, 3))]),  # ERROR
            (
                ["L-city", "O", "L-person"],
                [("city", (0, 0)), ("person", (1, 2))],
            ),
            (["B-city", "O", "O"], [("city", (0, 2))]),  # ERROR
            (
                ["B-city", "O", "U-city"],
                [
                    ("city", (0, 2)),
                ],
            ),
        ],
        BIO_ENCODING_NAME: [
            (["O", "B-city", "O", "B-city", "I-city"], [("city", (1, 1)), ("city", (3, 4))]),
            (["B-city", "I-city", "I-person", "I-person"], [("city", (0, 1)), ("person", (2, 3))]),
        ],
    }
    if encoding is None:
        with pytest.raises(ValueError) as e:
            tag_sequence = sequence_to_span[BIOUL_ENCODING_NAME][0][0]
            tag_sequence_to_token_spans(tag_sequence, coding_scheme=encoding)
    else:
        for tag_sequence, labeled_span in sequence_to_span[encoding]:
            computed_labeled_span = tag_sequence_to_token_spans(
                tag_sequence, coding_scheme=encoding
            )
            computed_labeled_span = sorted(computed_labeled_span, key=lambda x: x[1][0])
            assert computed_labeled_span == labeled_span


@pytest.mark.parametrize(
    "encoding",
    [BIOUL_ENCODING_NAME, BIO_ENCODING_NAME, BOUL_ENCODING_NAME, None],
)
def test_tag_sequence_to_span_without_include_ill_formed(encoding, true_tag_sequences):
    sequence_to_span = {
        BIOUL_ENCODING_NAME: [
            (
                ["O", "B-city", "O", "B-city", "L-city"],
                [("city", (3, 4))],
            ),
            (["B-city", "L-city", "I-person", "L-person"], [("city", (0, 1))]),
            (
                ["B-city", "I-city", "L-city", "B-person", "U-person", "L-person"],
                [("city", (0, 2))],
            ),
            (
                ["B-city", "I-city", "L-city", "I-person", "U-person", "L-person"],
                [("city", (0, 2)), ("person", (4, 4))],
            ),
            (["B-city", "I-city", "B-city"], []),
        ],
        BOUL_ENCODING_NAME: [
            (
                ["O", "B-city", "O", "L-city", "U-city"],  # what is this O is actual O not I
                [("city", (1, 3)), ("city", (4, 4))],
            ),
            (["B-city", "O", "L-city", "L-person"], [("city", (0, 2))]),
            (
                ["B-city", "O", "L-city", "B-person", "U-city", "L-person"],
                [("city", (0, 2))],
            ),
            (
                ["B-city", "O", "L-city", "O", "U-city", "L-person"],
                [("city", (0, 2)), ("city", (4, 4))],
            ),
        ],
        BIO_ENCODING_NAME: [
            (["O", "B-city", "O", "B-city", "I-city"], [("city", (1, 1)), ("city", (3, 4))]),
            (["B-city", "I-city", "I-person", "I-person"], [("city", (0, 1))]),
        ],
    }
    if encoding is None:
        with pytest.raises(ValueError) as e:
            tag_sequence = sequence_to_span[BIOUL_ENCODING_NAME][0][0]
            tag_sequence_to_token_spans(
                tag_sequence, coding_scheme=encoding, include_ill_formed=False
            )
    else:
        for tag_sequence, labeled_span in sequence_to_span[encoding]:
            computed_labeled_span = tag_sequence_to_token_spans(
                tag_sequence, coding_scheme=encoding, include_ill_formed=False
            )
            computed_labeled_span = sorted(computed_labeled_span, key=lambda x: x[1][0])
            assert computed_labeled_span == labeled_span


def test_bioul_to_boul():
    bioul_sequence = [
        "O",
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
        "O",
        "U-data",
        "O",
        "O",
    ]
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "L-background_claim",
        "O",
        "U-data",
        "O",
        "O",
    ]
    new_tag_sequence = bioul_to_boul(bioul_sequence)
    assert new_tag_sequence == boul_sequence


def test_boul_to_bioul():
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "L-background_claim",
        "O",
        "U-data",
        "O",
        "O",
    ]
    bioul_sequence = [
        "O",
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
        "O",
        "U-data",
        "O",
        "O",
    ]
    new_tag_sequence = boul_to_bioul(boul_sequence)
    assert new_tag_sequence == bioul_sequence
