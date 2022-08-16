import re

import pytest
from pytorch_ie.annotations import LabeledSpan

from pie_utils.processor.document import DocumentWithPartition
from pie_utils.processor.document.split_into_partition import (
    SplitDocumentToPartitions,
    get_partitions_with_matcher,
)
from pie_utils.span.slice import have_overlap

TEXT1 = """This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl.
<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor.
<end>Karl enjoys sunny days in Berlin.
"""

TEXT2 = """This is initial text.<start>Lily is mother of Harry.
<end>Beth greets Emma.
"""


def test_split_document_to_partitions():
    split_document_to_partitions = SplitDocumentToPartitions(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=["<start>", "<end>"],
        skip_initial_partition=True,
    )
    document = DocumentWithPartition(text=TEXT1)
    new_document = split_document_to_partitions(document)

    partitions = new_document.partition
    assert len(partitions) == 2
    partition = partitions[0]
    assert (
        new_document.text[partition.start : partition.end]
        == "<start>Jane lives in Berlin. this is no sentence about Karl.\n<middle>Seattle is a rainy city. "
        "Jenny Durkan is the city's mayor.\n"
    )
    assert partition.label == "<start>"

    partition = partitions[1]
    assert (
        new_document.text[partition.start : partition.end]
        == "<end>Karl enjoys sunny days in Berlin.\n"
    )
    assert partition.label == "<end>"


def test_test_split_document_to_partitions_with_statistics():
    split_document_to_partitions_with_statistics = SplitDocumentToPartitions(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=["<start>", "<middle>", "<end>"],
        skip_initial_partition=True,
        collect_statistics=True,
    )
    document = DocumentWithPartition(text=TEXT1)
    new_document = split_document_to_partitions_with_statistics(document)
    partitions = new_document.partition
    assert len(partitions) == 3
    partition_lengths = split_document_to_partitions_with_statistics._statistics[
        "partition_lengths"
    ]
    assert partition_lengths == [61, 67, 39]
    num_partitions = split_document_to_partitions_with_statistics._statistics["num_partitions"]
    assert num_partitions == [3]
    document_lengths = split_document_to_partitions_with_statistics._statistics["document_lengths"]
    assert document_lengths == [len(TEXT1)]
    assert sum(document_lengths) != sum(partition_lengths)

    document = DocumentWithPartition(text=TEXT2)
    new_document = split_document_to_partitions_with_statistics(document)
    partitions = new_document.partition
    assert len(partitions) == 2
    partition_lengths = split_document_to_partitions_with_statistics._statistics[
        "partition_lengths"
    ]
    assert partition_lengths == [61, 67, 39, 32, 23]
    num_partitions = split_document_to_partitions_with_statistics._statistics["num_partitions"]
    assert num_partitions == [3, 2]
    document_lengths = split_document_to_partitions_with_statistics._statistics["document_lengths"]
    assert document_lengths == [len(TEXT1), len(TEXT2)]
    assert sum(document_lengths) != sum(partition_lengths)

    with pytest.raises(TypeError) as e:
        split_document_to_partitions_with_statistics.update_statistics("num_partitions", 1.0)
        assert e.value == "type of given key str or value float is incorrect."

    split_document_to_partitions_with_statistics.show_statistics()


def test_split_document_to_partitions_with_initial_partition():

    split_document_to_partitions = SplitDocumentToPartitions(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        collect_statistics=True,
        initial_partition_label="<initial_part>",
    )

    document = DocumentWithPartition(text=TEXT2)
    new_document = split_document_to_partitions(document)

    partitions = new_document.partition
    assert len(partitions) == 3
    partition = partitions[0]
    assert new_document.text[partition.start : partition.end] == "This is initial text."
    assert partition.label == "<initial_part>"
    document_lengths = split_document_to_partitions._statistics["document_lengths"]
    partition_lengths = split_document_to_partitions._statistics["partition_lengths"]
    assert document_lengths == [len(TEXT2)]
    assert sum(document_lengths) == sum(partition_lengths)


def test_split_document_to_partitions_without_label_group_id():
    split_document_to_partitions = SplitDocumentToPartitions(
        pattern="(<start>|<middle>|<end>)",
        label_whitelist=["<start>", "<middle>", "<end>"],
        skip_initial_partition=True,
    )
    document = DocumentWithPartition(text=TEXT1)
    new_document = split_document_to_partitions(document)

    partitions = new_document.partition
    assert len(partitions) == 0


def test_get_partitions_with_matcher():
    document = DocumentWithPartition(text=TEXT1)
    matcher = re.compile("(<start>|<middle>|<end>)").finditer
    partitions = []
    for partition in get_partitions_with_matcher(
        document=document,
        matcher=matcher,
        label_group_id=0,
        label_whitelist=["<start>", "<middle>", "<end>"],
    ):
        assert isinstance(partition, LabeledSpan)
        for p in partitions:
            assert not have_overlap((p.start, p.end), (partition.start, partition.end))
        partitions.append(partition)
