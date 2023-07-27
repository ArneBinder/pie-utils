from dataclasses import dataclass

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument


@dataclass
class DocumentWithEntitiesAndRelations(TextBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclass
class DocumentWithPartitions(TextBasedDocument):
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclass
class DocumentWithEntitiesRelationsAndPartitions(TextBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")
