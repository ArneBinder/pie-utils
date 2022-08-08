import copy

import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan

from pie_utils.processor.document import DocumentWithEntitiesRelationsAndPartition
from pie_utils.processor.document.candidate_relation_adder import CandidateRelationAdder

TEXT1 = "Jane lives in Berlin. this is no sentence about Karl\n"
TEXT2 = "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
TEXT3 = "Karl enjoys sunny days in Berlin."

ENTITY_JANE_TEXT1 = LabeledSpan(start=0, end=4, label="person")
ENTITY_BERLIN_TEXT1 = LabeledSpan(start=14, end=20, label="city")
ENTITY_KARL_TEXT1 = LabeledSpan(start=48, end=52, label="person")
SENTENCE1_TEXT1 = LabeledSpan(start=0, end=21, label="sentence")
REL_JANE_LIVES_IN_BERLIN = BinaryRelation(
    head=ENTITY_JANE_TEXT1, tail=ENTITY_BERLIN_TEXT1, label="lives_in"
)

ENTITY_SEATTLE_TEXT2 = LabeledSpan(start=0, end=7, label="city")
ENTITY_JENNY_TEXT2 = LabeledSpan(start=25, end=37, label="person")
SENTENCE1_TEXT2 = LabeledSpan(start=0, end=24, label="sentence")
SENTENCE2_TEXT2 = LabeledSpan(start=25, end=58, label="sentence")
REL_JENNY_MAYOR_OF_SEATTLE = BinaryRelation(
    head=ENTITY_JENNY_TEXT2, tail=ENTITY_SEATTLE_TEXT2, label="mayor_of"
)

ENTITY_KARL_TEXT3 = LabeledSpan(start=0, end=4, label="person")
ENTITY_BERLIN_TEXT3 = LabeledSpan(start=26, end=32, label="city")
SENTENCE1_TEXT3 = LabeledSpan(start=0, end=33, label="sentence")


def test_candidate_relation_adder():
    candidate_relation_adder = CandidateRelationAdder(
        label="no_relation",
        collect_statistics=True,
    )
    document = DocumentWithEntitiesRelationsAndPartition(text=TEXT1)
    document.entities.extend([ENTITY_JANE_TEXT1, ENTITY_BERLIN_TEXT1, ENTITY_KARL_TEXT1])
    document.relations.append(REL_JANE_LIVES_IN_BERLIN)
    document.partition.append(SENTENCE1_TEXT1)

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
    assert str(relation.head) == str(ENTITY_BERLIN_TEXT1)
    assert str(relation.tail) == str(ENTITY_JANE_TEXT1)
    assert relation.label == "no_relation"
    relation = relations[2]
    assert str(relation.head) == str(ENTITY_BERLIN_TEXT1)
    assert str(relation.tail) == str(ENTITY_KARL_TEXT1)
    assert relation.label == "no_relation"
    relation = relations[3]
    assert str(relation.head) == str(ENTITY_KARL_TEXT1)
    assert str(relation.tail) == str(ENTITY_BERLIN_TEXT1)
    assert relation.label == "no_relation"
    relation = relations[4]
    assert str(relation.head) == str(ENTITY_JANE_TEXT1)
    assert str(relation.tail) == str(ENTITY_KARL_TEXT1)
    assert relation.label == "no_relation"
    relation = relations[5]
    assert str(relation.head) == str(ENTITY_KARL_TEXT1)
    assert str(relation.tail) == str(ENTITY_JANE_TEXT1)
    assert relation.label == "no_relation"


def test_candidate_relation_adder_with_statistics():
    candidate_relation_adder_with_statistics = CandidateRelationAdder(
        label="no_relation",
        collect_statistics=True,
        max_distance=8,
    )

    document = DocumentWithEntitiesRelationsAndPartition(text=TEXT1)
    document.entities.extend([ENTITY_JANE_TEXT1, ENTITY_BERLIN_TEXT1, ENTITY_KARL_TEXT1])
    document.relations.append(REL_JANE_LIVES_IN_BERLIN)
    document.partition.append(SENTENCE1_TEXT1)

    document = candidate_relation_adder_with_statistics(document)
    assert len(document) == 3
    original_stats = {
        "num_total_relation_candidates": 6,
        "num_available_relations": 1,
        "available_rels_within_allowed_distance": 0,
        "num_added_relation_not_taken": 5,
        "num_rels_within_allowed_distance": 0,
        "num_candidates_not_taken": {"lives_in": 1, "no_relation": 5},
        "distances_taken": {},
    }
    statistics = candidate_relation_adder_with_statistics._statistics
    candidate_relation_adder_with_statistics.show_statistics()
    assert statistics == original_stats

    with pytest.raises(TypeError) as e:
        candidate_relation_adder_with_statistics.update_statistics("no_relation", 1.0)
        assert e.value == "type of given key str or value float is incorrect."

    with pytest.raises(TypeError) as e:
        candidate_relation_adder_with_statistics.update_statistics("no_relation", {"new": 1.0})
        assert e.value == "type of given key str or value float is incorrect."


def test_candidate_relation_adder_without_sort_by_distance():
    candidate_relation_adder_without_sort_by_distance = CandidateRelationAdder(
        label="no_relation",
        sort_by_distance=False,
    )

    document = DocumentWithEntitiesRelationsAndPartition(text=TEXT1)
    document.entities.extend([ENTITY_JANE_TEXT1, ENTITY_BERLIN_TEXT1, ENTITY_KARL_TEXT1])
    document.relations.append(REL_JANE_LIVES_IN_BERLIN)
    document.partition.append(SENTENCE1_TEXT1)

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


def test_candidate_relation_adder_with_no_relation_upper_bound():
    candidate_relation_adder_with_no_relation_upper_bound = CandidateRelationAdder(
        label="no_relation",
        added_relations_upper_bound_factor=3,
    )

    document = DocumentWithEntitiesRelationsAndPartition(text=TEXT1)
    document.entities.extend([ENTITY_JANE_TEXT1, ENTITY_BERLIN_TEXT1, ENTITY_KARL_TEXT1])
    document.relations.append(REL_JANE_LIVES_IN_BERLIN)
    document.partition.append(SENTENCE1_TEXT1)

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
    assert str(relation.head) == str(ENTITY_BERLIN_TEXT1)
    assert str(relation.tail) == str(ENTITY_JANE_TEXT1)
    assert relation.label == "no_relation"
    relation = relations[2]
    assert str(relation.head) == str(ENTITY_BERLIN_TEXT1)
    assert str(relation.tail) == str(ENTITY_KARL_TEXT1)
    assert relation.label == "no_relation"


def test_candidate_relation_adder_with_partitions():
    candidate_relation_adder_with_partition = CandidateRelationAdder(
        label="no_relation",
        use_partition=True,
    )

    document = DocumentWithEntitiesRelationsAndPartition(text=TEXT2)
    document.entities.extend([ENTITY_SEATTLE_TEXT2, ENTITY_JENNY_TEXT2])
    document.relations.append(REL_JENNY_MAYOR_OF_SEATTLE)
    document.partition.extend([SENTENCE1_TEXT2, SENTENCE2_TEXT2])

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
    partition = document.partition
    assert len(partition) == 2


def test_candidate_relation_adder_with_partitions_and_max_distance():
    # Along with the partition, now we check if a new candidate relation meets max_distance condition or not. That means
    # the inner distance between arguments of a relation should be less than max_distance.
    candidate_relation_adder_with_partition_and_max_distance = CandidateRelationAdder(
        label="no_relation",
        use_partition=True,
        max_distance=8,
    )

    document = DocumentWithEntitiesRelationsAndPartition(text=TEXT3)
    document.entities.extend([ENTITY_KARL_TEXT3, ENTITY_BERLIN_TEXT3])
    document.partition.append(SENTENCE1_TEXT3)

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
    partition = document.partition
    assert len(partition) == 1
