from pytorch_ie.annotations import Span

from pie_utils.processor.document import DocumentWithPartition
from pie_utils.processor.document.text_lengths_collector import TextLengthsCollector


def test_text_lengths_collector():
    text_lengths_collector = TextLengthsCollector(
        tokenizer_name_or_path="bert-base-uncased",
    )
    document = DocumentWithPartition(text="Jane lives in Berlin. This is a sentence about Karl.\n")
    processed_document = text_lengths_collector(document)
    assert document == processed_document
    assert text_lengths_collector.text_lengths == [14]
    assert text_lengths_collector.num_docs == 1


def test_text_lengths_collector_with_partition():
    text_lengths_collector = TextLengthsCollector(
        tokenizer_name_or_path="bert-base-uncased",
        use_partition=True,
    )
    document = DocumentWithPartition(text="Jane lives in Berlin. This is a sentence about Karl.\n")
    sentence1_text = "Jane lives in Berlin."
    sentence1 = Span(start=0, end=len(sentence1_text))
    document.partition.append(sentence1)
    assert str(sentence1) == sentence1_text
    sentence2_text = "This is a sentence about Karl."
    sentence2 = Span(start=sentence1.end + 1, end=sentence1.end + 1 + len(sentence2_text))
    document.partition.append(sentence2)
    assert str(sentence2) == sentence2_text

    processed_document = text_lengths_collector(document)
    assert document == processed_document
    assert text_lengths_collector.text_lengths == [7, 9]
    assert text_lengths_collector.num_docs == 1
    assert text_lengths_collector.num_parts == 2

    text_lengths_collector.show_statistics()
