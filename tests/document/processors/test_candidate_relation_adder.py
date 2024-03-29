import copy
import json
import logging
from dataclasses import dataclass

import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument

from pie_utils.document.processors import CandidateRelationAdder
from tests.document.processors.common import DocumentWithEntitiesRelationsAndPartitions


@pytest.fixture
def document1():
    TEXT1 = "Jane lives in Berlin. this is no sentence about Karl\n"
    ENTITY_JANE_TEXT1 = LabeledSpan(start=0, end=4, label="person")
    ENTITY_BERLIN_TEXT1 = LabeledSpan(start=14, end=20, label="city")
    ENTITY_KARL_TEXT1 = LabeledSpan(start=48, end=52, label="person")
    SENTENCE1_TEXT1 = LabeledSpan(start=0, end=21, label="sentence")
    REL_JANE_LIVES_IN_BERLIN = BinaryRelation(
        head=ENTITY_JANE_TEXT1, tail=ENTITY_BERLIN_TEXT1, label="lives_in"
    )

    document = DocumentWithEntitiesRelationsAndPartitions(text=TEXT1)
    document.entities.extend([ENTITY_JANE_TEXT1, ENTITY_BERLIN_TEXT1, ENTITY_KARL_TEXT1])
    document.relations.append(REL_JANE_LIVES_IN_BERLIN)
    document.partitions.append(SENTENCE1_TEXT1)
    return document


@pytest.fixture
def document2():
    TEXT2 = "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
    ENTITY_SEATTLE_TEXT2 = LabeledSpan(start=0, end=7, label="city")
    ENTITY_JENNY_TEXT2 = LabeledSpan(start=25, end=37, label="person")
    SENTENCE1_TEXT2 = LabeledSpan(start=0, end=24, label="sentence")
    SENTENCE2_TEXT2 = LabeledSpan(start=25, end=58, label="sentence")
    REL_JENNY_MAYOR_OF_SEATTLE = BinaryRelation(
        head=ENTITY_JENNY_TEXT2, tail=ENTITY_SEATTLE_TEXT2, label="mayor_of"
    )

    document = DocumentWithEntitiesRelationsAndPartitions(text=TEXT2)
    document.entities.extend([ENTITY_SEATTLE_TEXT2, ENTITY_JENNY_TEXT2])
    document.relations.append(REL_JENNY_MAYOR_OF_SEATTLE)
    document.partitions.extend([SENTENCE1_TEXT2, SENTENCE2_TEXT2])
    return document


@pytest.fixture
def document3():
    TEXT3 = "Karl enjoys sunny days in Berlin."
    ENTITY_KARL_TEXT3 = LabeledSpan(start=0, end=4, label="person")
    ENTITY_BERLIN_TEXT3 = LabeledSpan(start=26, end=32, label="city")
    SENTENCE1_TEXT3 = LabeledSpan(start=0, end=33, label="sentence")

    document = DocumentWithEntitiesRelationsAndPartitions(text=TEXT3)
    document.entities.extend([ENTITY_KARL_TEXT3, ENTITY_BERLIN_TEXT3])
    document.partitions.append(SENTENCE1_TEXT3)
    return document


def test_candidate_relation_adder(document1):
    candidate_relation_adder = CandidateRelationAdder(
        label="no_relation",
        collect_statistics=True,
    )
    document = document1

    original_document = copy.deepcopy(document)
    document = candidate_relation_adder(document)
    assert len(document) == 3

    # Document contains three entities, therefore total number of relations would be 6. One relation was already
    # available in the document, so 5 new relation candidates added.
    original_document = original_document
    entities = document.entities
    assert len(entities) == 3
    relations = document.relations
    assert len(relations) == 6
    original_relations = original_document.relations
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert str(relation.head) == "Berlin"
    assert str(relation.tail) == "Jane"
    assert relation.label == "no_relation"
    relation = relations[2]
    assert str(relation.head) == "Berlin"
    assert str(relation.tail) == "Karl"
    assert relation.label == "no_relation"
    relation = relations[3]
    assert str(relation.head) == "Karl"
    assert str(relation.tail) == "Berlin"
    assert relation.label == "no_relation"
    relation = relations[4]
    assert str(relation.head) == "Jane"
    assert str(relation.tail) == "Karl"
    assert relation.label == "no_relation"
    relation = relations[5]
    assert str(relation.head) == "Karl"
    assert str(relation.tail) == "Jane"
    assert relation.label == "no_relation"


def test_candidate_relation_adder_use_predictions():
    candidate_relation_adder = CandidateRelationAdder(
        label="no_relation",
        use_predictions=True,
    )
    TEXT1 = "Jane lives in Berlin. this is no sentence about Karl\n"
    ENTITY_JANE_TEXT1 = LabeledSpan(start=0, end=4, label="person")
    ENTITY_BERLIN_TEXT1 = LabeledSpan(start=14, end=20, label="city")
    ENTITY_KARL_TEXT1 = LabeledSpan(start=48, end=52, label="person")
    SENTENCE1_TEXT1 = LabeledSpan(start=0, end=21, label="sentence")
    REL_JANE_LIVES_IN_BERLIN = BinaryRelation(
        head=ENTITY_JANE_TEXT1, tail=ENTITY_BERLIN_TEXT1, label="lives_in"
    )

    document = DocumentWithEntitiesRelationsAndPartitions(text=TEXT1)
    document.entities.predictions.extend(
        [ENTITY_JANE_TEXT1, ENTITY_BERLIN_TEXT1, ENTITY_KARL_TEXT1]
    )
    document.relations.predictions.append(REL_JANE_LIVES_IN_BERLIN)
    document.partitions.append(SENTENCE1_TEXT1)

    original_document = copy.deepcopy(document)
    document = candidate_relation_adder(document)
    assert len(document) == 3

    # Document contains three entities, therefore total number of relations would be 6. One relation was already
    # available in the document, so 5 new relation candidates added.
    original_document = original_document
    entities = document.entities.predictions
    assert len(entities) == 3
    relations = document.relations.predictions
    assert len(relations) == 6
    original_relations = original_document.relations.predictions
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert str(relation.head) == "Berlin"
    assert str(relation.tail) == "Jane"
    assert relation.label == "no_relation"
    relation = relations[2]
    assert str(relation.head) == "Berlin"
    assert str(relation.tail) == "Karl"
    assert relation.label == "no_relation"
    relation = relations[3]
    assert str(relation.head) == "Karl"
    assert str(relation.tail) == "Berlin"
    assert relation.label == "no_relation"
    relation = relations[4]
    assert str(relation.head) == "Jane"
    assert str(relation.tail) == "Karl"
    assert relation.label == "no_relation"
    relation = relations[5]
    assert str(relation.head) == "Karl"
    assert str(relation.tail) == "Jane"
    assert relation.label == "no_relation"


def test_candidate_relation_adder_with_statistics(document1, caplog):
    candidate_relation_adder_with_statistics = CandidateRelationAdder(
        label="no_relation",
        collect_statistics=True,
        max_distance=8,
    )

    caplog.set_level(logging.INFO)
    caplog.clear()

    document = document1
    candidate_relation_adder_with_statistics.enter_dataset(None)
    candidate_relation_adder_with_statistics(document)
    candidate_relation_adder_with_statistics.exit_dataset(None)
    assert len(caplog.records) == 1
    log_description, log_json = caplog.records[0].message.split("\n", maxsplit=1)
    assert log_description.strip() == "Statistics:"
    assert json.loads(log_json) == {
        "num_total_relation_candidates": 6,
        "num_available_relations": 1,
        "available_rels_within_allowed_distance": 0,
        "num_added_relation_not_taken": 5,
        "num_rels_within_allowed_distance": 0,
        "num_candidates_not_taken": {"lives_in": 1, "no_relation": 5},
        "distances_taken": {},
    }

    with pytest.raises(TypeError) as e:
        candidate_relation_adder_with_statistics.update_statistics("no_relation", 1.0)
        assert e.value == "type of given key str or value float is incorrect."

    with pytest.raises(TypeError) as e:
        candidate_relation_adder_with_statistics.update_statistics("no_relation", {"new": 1.0})
        assert e.value == "type of given key str or value float is incorrect."


def test_candidate_relation_adder_without_sort_by_distance(document1):
    candidate_relation_adder_without_sort_by_distance = CandidateRelationAdder(
        label="no_relation",
        sort_by_distance=False,
    )

    document = document1

    original_document = copy.deepcopy(document)
    document = candidate_relation_adder_without_sort_by_distance(document)
    assert len(document) == 3

    # Document contains three entities, therefore total number of relations would be 6. One relation was already
    # available in the document, so 5 new relation candidates added.
    original_document = original_document
    entities = document.entities
    assert len(entities) == 3
    relations = document.relations
    assert len(relations) == 6
    original_relations = original_document.relations
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert relation.label == "no_relation"
    relation = relations[2]
    assert relation.label == "no_relation"
    relation = relations[3]
    assert relation.label == "no_relation"
    relation = relations[4]
    assert relation.label == "no_relation"
    relation = relations[5]
    assert relation.label == "no_relation"


def test_candidate_relation_adder_with_n_max_factor(document1):
    candidate_relation_adder_with_no_relation_upper_bound = CandidateRelationAdder(
        label="no_relation",
        n_max_factor=3,
    )

    document = document1

    original_document = copy.deepcopy(document)
    document = candidate_relation_adder_with_no_relation_upper_bound(document)
    assert len(document) == 3

    # Document contains three entities, therefore total number of relations would be 6. One relation was already
    # available in the document but since we limit total number of no relation added by a threshold 3, therefore
    # total available relations after processing will be 4.
    original_document = original_document
    entities = document.entities
    assert len(entities) == 3
    relations = document.relations
    assert len(relations) == 4  # one original and three no_relation
    original_relations = original_document.relations
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert str(relation.head) == "Berlin"
    assert str(relation.tail) == "Jane"
    assert relation.label == "no_relation"
    relation = relations[2]
    assert str(relation.head) == "Berlin"
    assert str(relation.tail) == "Karl"
    assert relation.label == "no_relation"


def test_candidate_relation_adder_with_n_max(document1):
    candidate_relation_adder = CandidateRelationAdder(n_max=3)

    document = document1

    original_document = copy.deepcopy(document)
    document = candidate_relation_adder(document)
    assert len(document) == 3

    # Document contains three entities, therefore total number of relations would be 6. One relation was already
    # available in the document, so 5 new relation candidates added.
    original_document = original_document
    entities = document.entities
    assert len(entities) == 3
    relations = document.relations
    assert len(relations) == 4
    original_relations = original_document.relations
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert relation.label == "no_relation"
    relation = relations[2]
    assert relation.label == "no_relation"
    relation = relations[3]
    assert relation.label == "no_relation"


def test_candidate_relation_adder_with_partitions(document2):
    candidate_relation_adder_with_partition = CandidateRelationAdder(
        label="no_relation",
        partition_layer="partitions",
    )

    document = document2

    original_document = copy.deepcopy(document)
    candidate_relation_adder_with_partition(document)
    assert len(document) == 3

    # Document contains two partitions, both containing one entity. Since there is no entity pair in any partition,
    # therefore, no new relation candidate is possible.

    original_document = original_document
    entities = document.entities
    assert len(entities) == 2
    relations = document.relations
    assert len(relations) == 1
    original_relations = original_document.relations
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    partition = document.partitions
    assert len(partition) == 2


def test_candidate_relation_adder_with_partitions_and_max_distance(document3):
    # Along with the partition, now we check if a new candidate relation meets max_distance condition or not. That means
    # the inner distance between arguments of a relation should be less than max_distance.
    candidate_relation_adder_with_partition_and_max_distance = CandidateRelationAdder(
        label="no_relation",
        partition_layer="partitions",
        max_distance=8,
    )

    document = document3

    original_document = copy.deepcopy(document)
    candidate_relation_adder_with_partition_and_max_distance(document)
    assert len(document) == 3

    # Document contains two candidate entity pairs with span distance 22 which is greater than maximum allowed span
    # distance, there none of these entity pairs are added as relation candidates.
    original_document = original_document
    entities = document.entities
    assert len(entities) == 2
    relations = document.relations
    assert len(relations) == 0
    original_relations = original_document.relations
    assert len(original_relations) == 0
    partition = document.partitions
    assert len(partition) == 1


def test_rel_layer_with_multiple_target_layers():
    @dataclass
    class MyDocument(TextDocument):
        entities1: AnnotationList[LabeledSpan] = annotation_field(target="text")
        entities2: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(
            targets=["entities1", "entities2"]
        )

    candidate_relation_adder = CandidateRelationAdder(relation_layer="relations")

    document = MyDocument(text="Hello world!")
    with pytest.raises(ValueError) as e:
        candidate_relation_adder(document)
        assert (
            e.value
            == "Relation layer must have exactly one target layer but found the following target layers: "
            "['entities1', 'entities2']"
        )
