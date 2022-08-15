from pie_utils.processor.document import DocumentWithPartition
from pie_utils.processor.document.split_into_partitions import SplitDocumentToPartitions

TEXT1 = """<start>Jane lives in Berlin. this is no sentence about Karl.
<start>Seattle is a rainy city. Jenny Durkan is the city's mayor.
<start>Karl enjoys sunny days in Berlin.
"""

TEXT2 = """<start>Lily is mother of Harry.
<start>Beth greets Emma.
"""


def test_split_document_to_partitions():
    split_document_to_partitions = SplitDocumentToPartitions(
        pattern="<start>",
        label_group_id=0,
        label_whitelist=["<start>"],
        skip_initial_partition=True,
    )
    document = DocumentWithPartition(text=TEXT1)
    new_document = split_document_to_partitions(document)

    partitions = new_document.partition
    assert len(partitions) == 3
    partition = partitions[0]
    assert (
        new_document.text[partition.start : partition.end]
        == "<start>Jane lives in Berlin. this is no sentence about Karl.\n"
    )
    partition = partitions[1]
    assert (
        new_document.text[partition.start : partition.end]
        == "<start>Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
    )
    partition = partitions[2]
    assert (
        new_document.text[partition.start : partition.end]
        == "<start>Karl enjoys sunny days in Berlin.\n"
    )


def test_test_split_document_to_partitions_with_statistics():
    split_document_to_partitions_with_statistics = SplitDocumentToPartitions(
        pattern="<start>",
        label_group_id=0,
        label_whitelist=["<start>"],
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
    assert partition_lengths == [61, 66, 41]
    num_partitions = split_document_to_partitions_with_statistics._statistics["num_partitions"]
    assert num_partitions == [3]
    partition_texts = split_document_to_partitions_with_statistics._statistics["partition_texts"]
    assert "".join(partition_texts) == TEXT1

    document = DocumentWithPartition(text=TEXT2)
    new_document = split_document_to_partitions_with_statistics(document)
    partitions = new_document.partition
    assert len(partitions) == 2
    partition_lengths = split_document_to_partitions_with_statistics._statistics[
        "partition_lengths"
    ]
    assert partition_lengths == [61, 66, 41, 32, 25]
    num_partitions = split_document_to_partitions_with_statistics._statistics["num_partitions"]
    assert num_partitions == [3, 2]
    partition_texts = split_document_to_partitions_with_statistics._statistics["partition_texts"]
    assert "".join(partition_texts) == TEXT1 + TEXT2
