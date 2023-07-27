from pytorch_ie.annotations import Span

from pie_utils.document.processors import TextLengthsCollector
from tests.document.processors.common import DocumentWithPartitions


def test_text_lengths_collector():
    text_lengths_collector = TextLengthsCollector(
        tokenizer_name_or_path="bert-base-uncased",
    )
    document = DocumentWithPartitions(
        text="Jane lives in Berlin. This is a sentence about Karl.\n"
    )
    processed_document = text_lengths_collector(document)
    assert document == processed_document
    assert text_lengths_collector.text_lengths == [14]
    assert text_lengths_collector.num_docs == 1


def test_text_lengths_collector_with_partition():
    text_lengths_collector = TextLengthsCollector(
        tokenizer_name_or_path="bert-base-uncased",
        partition_layer="partitions",
    )
    document = DocumentWithPartitions(
        text="Jane lives in Berlin. This is a sentence about Karl.\n"
    )
    sentence1_text = "Jane lives in Berlin."
    sentence1 = Span(start=0, end=len(sentence1_text))
    document.partitions.append(sentence1)
    assert str(sentence1) == sentence1_text
    sentence2_text = "This is a sentence about Karl."
    sentence2 = Span(start=sentence1.end + 1, end=sentence1.end + 1 + len(sentence2_text))
    document.partitions.append(sentence2)
    assert str(sentence2) == sentence2_text

    text_lengths_collector.enter_dataset(None)
    processed_document = text_lengths_collector(document)
    assert document == processed_document
    assert text_lengths_collector.text_lengths == [7, 9]
    assert text_lengths_collector.num_docs == 1
    assert text_lengths_collector.num_parts == 2
    text_lengths_collector.exit_dataset(None)
