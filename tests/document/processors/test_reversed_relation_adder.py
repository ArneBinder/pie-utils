import copy

import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan

from pie_utils.document import DocumentWithEntitiesAndRelations, ReversedRelationAdder

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


def test_reversed_relation():

    reverse_relation_adder = ReversedRelationAdder(
        symmetric_relation_labels=[],
    )

    document = DocumentWithEntitiesAndRelations(text=TEXT3)
    document.entities.extend([ENTITY_JAMIE_TEXT3, ENTITY_JOHN_TEXT3])
    document.relations.append(REL_JAMIE_MEETS_JOHN)

    original_document = copy.deepcopy(document)
    reverse_relation_adder(document)

    # Document contains two entities and one relation. One reverse relation will be created resulting in total two
    # relations.
    relations = document.relations
    assert len(relations) == 2
    original_relations = original_document.relations
    assert len(original_relations) == 1
    assert str(relations[0]) == str(original_relations[0])
    relation = relations[1]
    assert str(relation.head) == str(ENTITY_JOHN_TEXT3)
    assert str(relation.tail) == str(ENTITY_JAMIE_TEXT3)
    assert relation.label == "meets_reversed"
    statistics = {
        "added_relations": {"meets_reversed": 1},
        "already_reversed_relations": {},
        "num_available_relations": 1,
        "num_added_relations": 1,
    }
    assert reverse_relation_adder._statistics == statistics
    reverse_relation_adder.show_statistics()


def test_with_already_reversed_relations():
    # since allow_already_reversed_relations is False by default and document contains reversed relations, it will
    # generate LookupError exception.
    reverse_relation_adder = ReversedRelationAdder(
        symmetric_relation_labels=[],
    )
    document = DocumentWithEntitiesAndRelations(text=TEXT1)
    document.entities.extend([ENTITY_LILY_TEXT1, ENTITY_HARRY_TEXT1])
    document.relations.extend([REL_LILY_MOTHER_OF_HARRY, REL_HARRY_SON_OF_LILY])

    with pytest.raises(LookupError) as e:
        reverse_relation_adder(document)
    assert (
        str(e.value) == f"Entity pair of new relation "
        f"({BinaryRelation(label='mother_of_reversed', head=ENTITY_HARRY_TEXT1, tail=ENTITY_LILY_TEXT1)}) "
        f"already belongs to a relation: {REL_HARRY_SON_OF_LILY}"
    )


def test_with_already_reversed_relations_allow():
    reverse_relation_adder_with_allow_already_reversed_relations = ReversedRelationAdder(
        symmetric_relation_labels=[],
        allow_already_reversed_relations=True,
    )

    document = DocumentWithEntitiesAndRelations(text=TEXT1)
    document.entities.extend([ENTITY_LILY_TEXT1, ENTITY_HARRY_TEXT1])
    document.relations.extend([REL_LILY_MOTHER_OF_HARRY, REL_HARRY_SON_OF_LILY])

    original_document = copy.deepcopy(document)
    reverse_relation_adder_with_allow_already_reversed_relations(document)

    # Document contains two entities and two relations which are reverse of each other. After setting
    # allow_already_reversed_relations as True, we do not expect any new relation to be created.

    relations = document.relations
    assert len(relations) == 2
    original_relations = original_document.relations
    assert len(original_relations) == 2
    assert str(relations[0]) == str(original_relations[0])
    assert str(relations[1]) == str(original_relations[1])

    statistics = {
        "added_relations": {},
        "already_reversed_relations": {"mother_of_reversed": 1, "son_of_reversed": 1},
        "num_available_relations": 2,
        "num_added_relations": 0,
    }
    assert reverse_relation_adder_with_allow_already_reversed_relations._statistics == statistics


def test_symmetric_relation():
    reverse_relation_adder_with_sym_rel = ReversedRelationAdder(
        symmetric_relation_labels=["meets"],
    )

    document = DocumentWithEntitiesAndRelations(text=TEXT3)
    document.entities.extend([ENTITY_JAMIE_TEXT3, ENTITY_JOHN_TEXT3])
    document.relations.append(REL_JAMIE_MEETS_JOHN)

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
    assert str(relation.head) == str(ENTITY_JOHN_TEXT3)
    assert str(relation.tail) == str(ENTITY_JAMIE_TEXT3)
    assert relation.label == "meets"
