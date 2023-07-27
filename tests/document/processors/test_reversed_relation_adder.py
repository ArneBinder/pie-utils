import copy
import json
import logging

import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan

from pie_utils.document.processors import ReversedRelationAdder
from tests.document.processors.common import DocumentWithEntitiesAndRelations

TEXT1 = "Lily is mother of Harry."
TEXT2 = "Beth greets Emma."
TEXT3 = "Jamie meets John."

ENTITY_LILY_TEXT1 = LabeledSpan(start=0, end=4, label="person")
ENTITY_HARRY_TEXT1 = LabeledSpan(start=18, end=23, label="person")

REL_LILY_MOTHER_OF_HARRY = BinaryRelation(
    head=ENTITY_LILY_TEXT1, tail=ENTITY_HARRY_TEXT1, label="mother_of"
)
REL_HARRY_SON_OF_LILY = BinaryRelation(
    head=ENTITY_HARRY_TEXT1, tail=ENTITY_LILY_TEXT1, label="son_of"
)

ENTITY_BETH_TEXT2 = LabeledSpan(start=0, end=4, label="person")
ENTITY_EMMA_TEXT2 = LabeledSpan(start=12, end=16, label="person")

ENTITY_JAMIE_TEXT3 = LabeledSpan(start=0, end=5, label="person")
ENTITY_JOHN_TEXT3 = LabeledSpan(start=12, end=16, label="person")

REL_JAMIE_MEETS_JOHN = BinaryRelation(
    head=ENTITY_JAMIE_TEXT3, tail=ENTITY_JOHN_TEXT3, label="meets"
)
REL_JOHN_MEETS_R_JAMIE = BinaryRelation(
    head=ENTITY_JOHN_TEXT3, tail=ENTITY_JAMIE_TEXT3, label="meets_reversed"
)
REL_JOHN_MEETS_S_JAMIE = BinaryRelation(
    head=ENTITY_JOHN_TEXT3, tail=ENTITY_JAMIE_TEXT3, label="meets"
)


def test_reversed_relation(caplog):
    reverse_relation_adder = ReversedRelationAdder(
        symmetric_relation_labels=[],
        collect_statistics=True,
    )

    document = DocumentWithEntitiesAndRelations(text=TEXT3)
    jamie = ENTITY_JAMIE_TEXT3.copy()
    john = ENTITY_JOHN_TEXT3.copy()
    document.entities.extend([jamie, john])
    document.relations.append(REL_JAMIE_MEETS_JOHN.copy(head=jamie, tail=john))

    original_document = copy.deepcopy(document)
    caplog.set_level(logging.INFO)
    caplog.clear()
    reverse_relation_adder.enter_dataset(None)
    reverse_relation_adder(document)
    reverse_relation_adder.exit_dataset(None)

    # Document contains two entities and one relation. One reverse relation will be created resulting in total two
    # relations.
    relations = document.relations
    assert len(relations) == 2
    original_relations = original_document.relations
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert str(relation.head) == str(john)
    assert str(relation.tail) == str(jamie)
    assert relation.label == "meets_reversed"

    assert len(caplog.records) == 1
    log_description, log_json = caplog.records[0].message.split("\n", maxsplit=1)
    assert log_description.strip() == "Statistics:"
    assert json.loads(log_json) == {
        "added_relations": {"meets_reversed": 1},
        "already_reversed_relations": {},
        "num_available_relations": 1,
        "num_added_relations": 1,
    }


def test_reversed_relation_prediction():
    reverse_relation_adder = ReversedRelationAdder(
        symmetric_relation_labels=[],
        use_predictions=True,
    )

    document = DocumentWithEntitiesAndRelations(text=TEXT3)
    jamie = ENTITY_JAMIE_TEXT3.copy()
    john = ENTITY_JOHN_TEXT3.copy()
    document.entities.extend([jamie, john])
    document.relations.predictions.append(REL_JAMIE_MEETS_JOHN.copy(head=jamie, tail=john))

    original_document = copy.deepcopy(document)
    reverse_relation_adder(document)

    # Document contains two entities and one relation. One reverse relation will be created resulting in total two
    # relations.
    relations = document.relations.predictions
    assert len(relations) == 2
    original_relations = original_document.relations.predictions
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert str(relation.head) == str(john)
    assert str(relation.tail) == str(jamie)
    assert relation.label == "meets_reversed"


def test_with_already_reversed_relations():
    # since allow_already_reversed_relations is False by default and document contains reversed relations, it will
    # generate LookupError exception.
    reverse_relation_adder = ReversedRelationAdder(
        symmetric_relation_labels=[],
    )
    document = DocumentWithEntitiesAndRelations(text=TEXT1)
    lily = ENTITY_LILY_TEXT1.copy()
    harry = ENTITY_HARRY_TEXT1.copy()
    document.entities.extend([lily, harry])
    document.relations.extend(
        [
            REL_LILY_MOTHER_OF_HARRY.copy(head=lily, tail=harry),
            REL_HARRY_SON_OF_LILY.copy(head=harry, tail=lily),
        ]
    )

    with pytest.raises(LookupError) as e:
        reverse_relation_adder(document)
    assert (
        str(e.value) == f"Entity pair of new relation "
        f"({BinaryRelation(label='mother_of_reversed', head=harry, tail=lily)}) "
        f"already belongs to a relation: {REL_HARRY_SON_OF_LILY}"
    )


def test_with_already_reversed_relations_allow(caplog):
    reverse_relation_adder_with_allow_already_reversed_relations = ReversedRelationAdder(
        symmetric_relation_labels=[],
        allow_already_reversed_relations=True,
        collect_statistics=True,
    )

    document = DocumentWithEntitiesAndRelations(text=TEXT1)
    lily = ENTITY_LILY_TEXT1.copy()
    harry = ENTITY_HARRY_TEXT1.copy()
    document.entities.extend([lily, harry])
    document.relations.extend(
        [
            REL_LILY_MOTHER_OF_HARRY.copy(head=lily, tail=harry),
            REL_HARRY_SON_OF_LILY.copy(head=harry, tail=lily),
        ]
    )

    original_document = copy.deepcopy(document)
    caplog.set_level(logging.INFO)
    caplog.clear()
    reverse_relation_adder_with_allow_already_reversed_relations.enter_dataset(None)
    reverse_relation_adder_with_allow_already_reversed_relations(document)
    reverse_relation_adder_with_allow_already_reversed_relations.exit_dataset(None)

    # Document contains two entities and two relations which are reverse of each other. After setting
    # allow_already_reversed_relations as True, we do not expect any new relation to be created.

    relations = document.relations
    assert len(relations) == 2
    original_relations = original_document.relations
    assert len(original_relations) == 2
    assert str(relations[0]) == str(original_relations[0])
    assert str(relations[1]) == str(original_relations[1])

    assert len(caplog.records) == 1
    log_description, log_json = caplog.records[0].message.split("\n", maxsplit=1)
    assert log_description.strip() == "Statistics:"
    assert json.loads(log_json) == {
        "added_relations": {},
        "already_reversed_relations": {"mother_of_reversed": 1, "son_of_reversed": 1},
        "num_available_relations": 2,
        "num_added_relations": 0,
    }


def test_symmetric_relation():
    reverse_relation_adder_with_sym_rel = ReversedRelationAdder(
        symmetric_relation_labels=["meets"],
    )

    document = DocumentWithEntitiesAndRelations(text=TEXT3)
    jamie = ENTITY_JAMIE_TEXT3.copy()
    john = ENTITY_JOHN_TEXT3.copy()
    document.entities.extend([jamie, john])
    document.relations.append(REL_JAMIE_MEETS_JOHN.copy(head=jamie, tail=john))

    original_document = copy.deepcopy(document)
    reverse_relation_adder_with_sym_rel(document)

    # Document contains a relation labeled 'meets' which is also given in a list as symmetric_relation_labels.
    # Therefore, a reverse relation will be created without suffix '_reversed'. Finally, there will be two relations
    # both containing relation label as 'meets'
    relations = document.relations
    assert len(relations) == 2
    original_relations = original_document.relations
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert str(relation.head) == str(john)
    assert str(relation.tail) == str(jamie)
    assert relation.label == "meets"
