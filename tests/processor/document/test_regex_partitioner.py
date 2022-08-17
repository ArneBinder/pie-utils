import re

import pytest
from pytorch_ie.annotations import LabeledSpan

from pie_utils.processor.document import DocumentWithPartitions
from pie_utils.processor.document.regex_partitioner import (
    RegexPartitioner,
    _get_partitions_with_matcher,
)
from pie_utils.span.slice import have_overlap


def test_regex_partitioner():
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. RegexPartitioner
    # partitions the text based on the given pattern. Since skip_initial_partition is True, so initial part of the text
    # will be ignored. label_whitelist do not contain <middle>, so it will be ignored as well. After partitioning, there
    # will be two partitions one with label <start> and another with <end>.
    document = DocumentWithPartitions(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partitions
    labels = [partition.label for partition in partitions]
    assert len(partitions) == 4
    assert labels == ["partition"] * len(partitions)
    assert str(partitions[0]) == "This is initial text."
    assert str(partitions[1]) == "<start>Jane lives in Berlin. this is no sentence about Karl."
    assert (
        str(partitions[2]) == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
    )
    assert str(partitions[3]) == "<end>Karl enjoys sunny days in Berlin."


def test_regex_partitioner_with_statistics():
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    TEXT2 = "This is initial text.<start>Lily is mother of Harry.<end>Beth greets Emma."

    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=["<start>", "<middle>", "<end>"],
        skip_initial_partition=True,
        collect_statistics=True,
    )

    # The document contains a text separated by some markers like <start>, <middle> and <end>. After partitioning, there
    # are three partitions excluding initial part. Therefore, document length is not be equal to sum of partitions.
    document = DocumentWithPartitions(text=TEXT1)
    new_document = regex_partitioner(document)
    partitions = new_document.partitions
    assert len(partitions) == 3
    partition_lengths = regex_partitioner._statistics["partition_lengths"]
    assert partition_lengths == [60, 66, 38]
    num_partitions = regex_partitioner._statistics["num_partitions"]
    assert num_partitions == [3]
    document_lengths = regex_partitioner._statistics["document_lengths"]
    assert document_lengths == [len(TEXT1)]
    # Sum of partition length should be less than document length since initial partition is excluded.
    assert sum(partition_lengths) < sum(document_lengths)

    # The document contains a text separated by some markers like <start> and <end>. RegexPartitioner appends statistics
    # from each document, therefore statistics contains information from previous document as well. After partitioning,
    # there are two partitions excluding initial part. Therefore, the sum of document lengths is not be equal to sum of
    # partitions.
    document = DocumentWithPartitions(text=TEXT2)
    new_document = regex_partitioner(document)
    partitions = new_document.partitions
    assert len(partitions) == 2
    partition_lengths = regex_partitioner._statistics["partition_lengths"]
    assert partition_lengths == [60, 66, 38, 31, 22]
    num_partitions = regex_partitioner._statistics["num_partitions"]
    assert num_partitions == [3, 2]
    document_lengths = regex_partitioner._statistics["document_lengths"]
    assert document_lengths == [len(TEXT1), len(TEXT2)]
    # Sum of partition length should be less than document length since initial partition is excluded.
    assert sum(partition_lengths) < sum(document_lengths)

    with pytest.raises(
        TypeError,
        match=r"type of given key \[<class 'str'>\] or value \[<class 'float'>\] is incorrect.",
    ):
        regex_partitioner.update_statistics("num_partitions", 1.0)

    regex_partitioner.show_statistics()


@pytest.mark.parametrize("label_whitelist", [["<start>", "<middle>", "<end>"], [], None])
@pytest.mark.parametrize("skip_initial_partition", [True, False])
def test_regex_partitioner_without_label_group_id(label_whitelist, skip_initial_partition):
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_whitelist=label_whitelist,
        skip_initial_partition=skip_initial_partition,
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. Since label_group_id is
    # None, therefore there will be no partitions.
    document = DocumentWithPartitions(text=TEXT1)
    new_document = regex_partitioner(document)
    partitions = new_document.partitions
    assert [partition.label for partition in partitions] == ["partition"] * len(partitions)
    if skip_initial_partition:
        if label_whitelist == ["<start>", "<middle>", "<end>"] or label_whitelist == []:
            assert len(partitions) == 0
        else:  # label_whitelist is None
            assert len(partitions) == 3
            assert (
                str(partitions[0])
                == "<start>Jane lives in Berlin. this is no sentence about Karl."
            )
            assert (
                str(partitions[1])
                == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[2]) == "<end>Karl enjoys sunny days in Berlin."
    else:  # skip_initial_partition is False
        if label_whitelist == ["<start>", "<middle>", "<end>"] or label_whitelist == []:
            assert len(partitions) == 0
        else:  # label_whitelist is None
            assert len(partitions) == 4
            assert str(partitions[0]) == "This is initial text."
            assert (
                str(partitions[1])
                == "<start>Jane lives in Berlin. this is no sentence about Karl."
            )
            assert (
                str(partitions[2])
                == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[3]) == "<end>Karl enjoys sunny days in Berlin."


@pytest.mark.parametrize(
    "label_whitelist", [["partition", "<start>", "<end>"], ["<start>", "<end>"], [], None]
)
@pytest.mark.parametrize("skip_initial_partition", [True, False])
def test_regex_partitioner_with_label_group_id(label_whitelist, skip_initial_partition):
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=label_whitelist,
        skip_initial_partition=skip_initial_partition,
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. Since label_group_id is
    # None, therefore there will be no partitions.
    document = DocumentWithPartitions(text=TEXT1)
    new_document = regex_partitioner(document)
    partitions = new_document.partitions
    labels = [partition.label for partition in partitions]
    if skip_initial_partition:
        if label_whitelist == ["<start>", "<end>"] or label_whitelist == [
            "partition",
            "<start>",
            "<end>",
        ]:
            assert len(partitions) == 2
            assert labels == ["<start>", "<end>"]
            assert (
                str(partitions[0])
                == "<start>Jane lives in Berlin. this is no sentence about Karl.<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[1]) == "<end>Karl enjoys sunny days in Berlin."
        elif label_whitelist == []:
            assert len(partitions) == 0
        else:  # label_whitelist is None
            assert len(partitions) == 3
            assert labels == ["<start>", "<middle>", "<end>"]
            assert (
                str(partitions[0])
                == "<start>Jane lives in Berlin. this is no sentence about Karl."
            )
            assert (
                str(partitions[1])
                == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[2]) == "<end>Karl enjoys sunny days in Berlin."
    else:  # skip_initial_partition is False
        if label_whitelist == ["<start>", "<end>"]:
            assert len(partitions) == 2
            assert labels == ["<start>", "<end>"]
            assert (
                str(partitions[0])
                == "<start>Jane lives in Berlin. this is no sentence about Karl.<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[1]) == "<end>Karl enjoys sunny days in Berlin."
        elif label_whitelist == ["partition", "<start>", "<end>"]:
            assert len(partitions) == 3
            assert labels == ["partition", "<start>", "<end>"]
            assert str(partitions[0]) == "This is initial text."
            assert (
                str(partitions[1])
                == "<start>Jane lives in Berlin. this is no sentence about Karl.<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[2]) == "<end>Karl enjoys sunny days in Berlin."
        elif label_whitelist == []:
            assert len(partitions) == 0
        else:  # label_whitelist is None
            assert len(partitions) == 4
            assert labels == ["partition", "<start>", "<middle>", "<end>"]
            assert str(partitions[0]) == "This is initial text."
            assert (
                str(partitions[1])
                == "<start>Jane lives in Berlin. this is no sentence about Karl."
            )
            assert (
                str(partitions[2])
                == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[3]) == "<end>Karl enjoys sunny days in Berlin."


@pytest.mark.parametrize("label_whitelist", [["partition"], [], None])
@pytest.mark.parametrize("skip_initial_partition", [True, False])
def test_regex_partitioner_with_no_match_found(skip_initial_partition, label_whitelist):
    TEXT2 = "This is initial text.<start>Lily is mother of Harry.<end>Beth greets Emma."
    regex_partitioner = RegexPartitioner(
        pattern="(<middle>)",
        label_group_id=0,
        label_whitelist=label_whitelist,
        skip_initial_partition=skip_initial_partition,
    )
    # The document contains a text separated by some markers like <start> and <end>. Since the given pattern is not
    # found in the document text therefore document is not partitioned.
    document = DocumentWithPartitions(text=TEXT2)
    new_document = regex_partitioner(document)

    partitions = new_document.partitions
    if skip_initial_partition:
        if label_whitelist == ["partition"]:
            assert len(partitions) == 0
        elif label_whitelist == []:
            assert len(partitions) == 0
        else:  # label_whitelist is None
            assert len(partitions) == 0
    else:
        if label_whitelist == ["partition"]:
            assert len(partitions) == 1
            assert str(partitions[0]) == TEXT2
            assert partitions[0].label == "partition"
        elif label_whitelist == []:
            assert len(partitions) == 0
        else:  # label_whitelist is None
            assert len(partitions) == 1
            assert str(partitions[0]) == TEXT2
            assert partitions[0].label == "partition"


def test_get_partitions_with_matcher():
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. finditer method is used
    # which returns non overlapping match from the text. Therefore, none of the partition created should have overlapped
    # span and all of them should be instances of LabeledSpan.
    document = DocumentWithPartitions(text=TEXT1)
    matcher = re.compile("(<start>|<middle>|<end>)").finditer
    partitions = []
    for partition in _get_partitions_with_matcher(
        document=document,
        matcher=matcher,
        label_group_id=0,
        label_whitelist=["<start>", "<middle>", "<end>"],
    ):
        assert isinstance(partition, LabeledSpan)
        for p in partitions:
            assert not have_overlap((p.start, p.end), (partition.start, partition.end))
        partitions.append(partition)
