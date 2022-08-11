from __future__ import annotations

import json
import logging
from collections import defaultdict

from pytorch_ie.annotations import BinaryRelation

from pie_utils.statistics import WithStatistics

from ..document import DocumentWithEntitiesAndRelations

logger = logging.getLogger(__name__)


class ReversedRelationAdder(WithStatistics):
    """TODO.

    :param symmetric_relation_labels TODO
    :param label_suffix TODO
    :param allow_already_reversed_relations TODO
    """

    def __int__(
        self,
        symmetric_relation_labels: list[str] | None = None,
        label_suffix: str = "_reversed",
        allow_already_reversed_relations: bool = False,
    ):
        self.symmetric_relation_labels = symmetric_relation_labels or []
        self.label_suffix = label_suffix
        self.allow_already_reversed_relations = allow_already_reversed_relations

    def reset_statistics(self):
        self.added_relations = defaultdict(int)

    def show_statistics(self, description: str | None = None):
        description = description or "Statistics"
        logger.info(
            f"{description}: added reversed relations\n{json.dumps(dict(self.added_relations))}"
        )

    def __call__(
        self, document: DocumentWithEntitiesAndRelations
    ) -> DocumentWithEntitiesAndRelations:
        # get all relations before adding any reversed
        rels = list(document.relations)

        available_relations = {(rel.head, rel.tail): rel for rel in rels}
        for rel in rels:
            new_label = (
                rel.label
                if rel.label in self.symmetric_relation_labels
                else f"{rel.label}{self.label_suffix}"
            )
            new_relation = BinaryRelation(label=new_label, head=rel.tail, tail=rel.head)
            if (new_relation.head, new_relation.tail) in available_relations:
                # If an entity pair of reveCandidatersed relation is present in the available relations then we check if we want
                # to allow already existing reversed relations or not. If we allow then we do not add the reversed
                # relation to the document but move on to the next relation otherwise we generate a LookupError
                # exception.
                if self.allow_already_reversed_relations:
                    continue
                else:
                    raise LookupError(
                        f"Entity pair of new relation ({new_relation}) already belongs to a relation: "
                        f"{available_relations[(new_relation.head, new_relation.tail)]}"
                    )
            else:
                document.relations.append(new_relation)
                self.added_relations[new_relation.label] += 1

        return document
