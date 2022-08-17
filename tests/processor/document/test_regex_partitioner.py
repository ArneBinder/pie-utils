import re

import pytest
from pytorch_ie.annotations import LabeledSpan

from pie_utils.processor.document import DocumentWithPartition
from pie_utils.processor.document.regex_partitioner import (
    RegexPartitioner,
    get_partitions_with_matcher,
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
        label_group_id=0,
        label_whitelist=["<start>", "<end>"],
        skip_initial_partition=True,
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. RegexPartitioner
    # partitions the text based on the given pattern. Since skip_initial_partition is True, so initial part of the text
    # will be ignored. label_whitelist do not contain <middle>, so it will be ignored as well. After partitioning, there
    # will be two partitions one with label <start> and another with <end>.
    document = DocumentWithPartition(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partition
    assert len(partitions) == 2
    partition = partitions[0]
    assert (
        new_document.text[partition.start : partition.end]
        == "<start>Jane lives in Berlin. this is no sentence about Karl.<middle>Seattle is a rainy city. "
        "Jenny Durkan is the city's mayor."
    )
    assert partition.label == "<start>"

    partition = partitions[1]
    assert (
        new_document.text[partition.start : partition.end]
        == "<end>Karl enjoys sunny days in Berlin."
    )
    assert partition.label == "<end>"


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
    document = DocumentWithPartition(text=TEXT1)
    new_document = regex_partitioner(document)
    partitions = new_document.partition
    assert len(partitions) == 3
    partition_lengths = regex_partitioner._statistics["partition_lengths"]
    assert partition_lengths == [60, 66, 38]
    num_partitions = regex_partitioner._statistics["num_partitions"]
    assert num_partitions == [3]
    document_lengths = regex_partitioner._statistics["document_lengths"]
    assert document_lengths == [len(TEXT1)]
    assert sum(document_lengths) != sum(partition_lengths)

    # The document contains a text separated by some markers like <start> and <end>. RegexPartitioner appends statistics
    # from each document, therefore statistics contains information from previous document as well. After partitioning,
    # there are two partitions excluding initial part. Therefore, the sum of document lengths is not be equal to sum of
    # partitions.
    document = DocumentWithPartition(text=TEXT2)
    new_document = regex_partitioner(document)
    partitions = new_document.partition
    assert len(partitions) == 2
    partition_lengths = regex_partitioner._statistics["partition_lengths"]
    assert partition_lengths == [60, 66, 38, 31, 22]
    num_partitions = regex_partitioner._statistics["num_partitions"]
    assert num_partitions == [3, 2]
    document_lengths = regex_partitioner._statistics["document_lengths"]
    assert document_lengths == [len(TEXT1), len(TEXT2)]
    assert sum(document_lengths) != sum(partition_lengths)

    with pytest.raises(TypeError) as e:
        regex_partitioner.update_statistics("num_partitions", 1.0)
        assert e.value == "type of given key str or value float is incorrect."

    regex_partitioner.show_statistics()


def test_regex_partitioner_with_initial_partition():
    TEXT2 = "This is initial text.<start>Lily is mother of Harry.<end>Beth greets Emma."
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        collect_statistics=True,
        initial_partition_label="<initial_part>",
    )
    document = DocumentWithPartition(text=TEXT2)
    new_document = regex_partitioner(document)

    # The document contains a text separated by some markers like <start> and <end>. Since skip_initial_partition is
    # False, therefore including initial partition there will be total three partitions. Label for first partition
    # should be value of initial_partition_label and partition lengths should sum up to document length.
    partitions = new_document.partition
    assert len(partitions) == 3
    partition = partitions[0]
    assert new_document.text[partition.start : partition.end] == "This is initial text."
    assert partition.label == "<initial_part>"
    document_lengths = regex_partitioner._statistics["document_lengths"]
    partition_lengths = regex_partitioner._statistics["partition_lengths"]
    assert document_lengths == [len(TEXT2)]
    assert sum(document_lengths) == sum(partition_lengths)


def test_regex_partitioner_without_label_group_id():
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_whitelist=["<start>", "<middle>", "<end>"],
        skip_initial_partition=True,
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. Since label_group_id is
    # None, therefore there will be no partitions.
    document = DocumentWithPartition(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partition
    assert len(partitions) == 0

    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        skip_initial_partition=True,
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. Since label_group_id is
    # None, therefore there will be no partitions.
    document = DocumentWithPartition(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partition
    assert len(partitions) == 0

    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_whitelist=["<start>", "<middle>", "<end>"],
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. Though label_group_id is
    # None but skip_initial_partition is False, therefore whole document will be part of initial partition.
    document = DocumentWithPartition(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partition
    assert len(partitions) == 1
    assert new_document.text[partitions[0].start : partitions[0].end] == TEXT1

    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. In this case, a partition
    # will be created with label "initial part" but it's content will be up to first matched pattern i.e. just the
    # initial part of the document.
    document = DocumentWithPartition(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partition
    assert len(partitions) == 1
    assert new_document.text[partitions[0].start : partitions[0].end] == "This is initial text."


def test_get_partitions_with_matcher():
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. finditer method is used
    # which returns non overlapping match from the text. Therefore, none of the partition created should have overlapped
    # span and all of them should be instances of LabeledSpan.
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


def test_regex_partitioner_with_no_parts_added():
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=[],
        skip_initial_partition=True,
    )

    # The document contains a text separated by some markers like <start>, <middle> and <end>. Since there is no label
    # in label_whitelist, therefore no partitions are created.
    document = DocumentWithPartition(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partition
    assert len(partitions) == 0

    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=[],
    )

    # The document contains a text separated by some markers like <start>, <middle> and <end>. Although there is no label
    # in label_whitelist but skip_initial_partition is False, therefore whole document will be created as a single
    # partition with initial_partition_label.
    document = DocumentWithPartition(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partition
    assert len(partitions) == 1
    partition = partitions[0]
    assert partition.label == "initial part"
    assert new_document.text[partition.start : partition.end] == TEXT1


def test_regex_partitioner_with_no_match_found():
    TEXT2 = "This is initial text.<start>Lily is mother of Harry.<end>Beth greets Emma."
    regex_partitioner = RegexPartitioner(
        pattern="(<middle>)",
        label_group_id=0,
        label_whitelist=[],
        skip_initial_partition=True,
    )
    # The document contains a text separated by some markers like <start> and <end>. Since the given pattern is not
    # found in the document text therefore document is not partitioned.
    document = DocumentWithPartition(text=TEXT2)
    new_document = regex_partitioner(document)

    partitions = new_document.partition
    assert len(partitions) == 0
