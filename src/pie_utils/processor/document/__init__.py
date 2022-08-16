from dataclasses import dataclass

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument


@dataclass
class DocumentWithEntitiesAndRelations(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclass
class DocumentWithPartition(TextDocument):
    partition: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclass
class DocumentWithEntitiesRelationsAndPartition(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    partition: AnnotationList[LabeledSpan] = annotation_field(target="text")
