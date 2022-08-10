import re

import pytest

from pie_utils.processor.document import DocumentWithEntitiesRelationsAndPartition
from pie_utils.processor.document.split_document_into_partitions import (
    SplitDocumentToPartitions,
)

TEXT = """<start>Jane lives in Berlin. this is no sentence about Karl.
<start>Seattle is a rainy city. Jenny Durkan is the city's mayor.
<start>Karl enjoys sunny days in Berlin.
"""


@pytest.fixture
def split_document_to_partitions():
    return SplitDocumentToPartitions(
        matcher=re.compile("<start>").finditer,
        label_group_id=0,
        label_whitelist=["<start>"],
        skip_initial_partition=True,
    )


def test_split_document_to_partitions(split_document_to_partitions):
    document = DocumentWithEntitiesRelationsAndPartition(text=TEXT)
    new_document = split_document_to_partitions(document)
    assert len(new_document.partition) == 3
