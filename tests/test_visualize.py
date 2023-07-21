from dataclasses import dataclass
from typing import Optional

import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextDocument

from pie_utils.document.visualization import print_document_annotation_graph


@dataclass
class MyDocument(TextDocument):
    text2: Optional[str] = None
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")
    entity_belongs_to_partition: AnnotationList[BinaryRelation] = annotation_field(
        targets=["entities", "partitions"]
    )


@pytest.mark.parametrize(
    "swap_edges",
    [True, False],
)
def test_print_document_annotation_graph(swap_edges):
    document = MyDocument(text="Hello World")

    print_document_annotation_graph(
        annotation_graph=document._annotation_graph,
        remove_node="_artificial_root",
        add_root_node="root",
        swap_edges=swap_edges,
    )
