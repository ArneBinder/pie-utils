from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import TypeVar

from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.annotations import BinaryRelation
from pytorch_ie.core import Document
from pytorch_ie.data.common import EnterDatasetMixin, ExitDatasetMixin

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


class ReversedRelationAdder(EnterDatasetMixin, ExitDatasetMixin):
    """ReversedRelationAdder adds binary relations to a document by reversing already existing
    relations in the document. Reversing of a relation is done by swapping head and tail span in a
    relation.

    :param label_suffix : A string to be appended as suffix with relation label (the default value
        is _reversed)
    :param symmetric_relation_labels : List of symmetric relation labels which when reversed will
        use original relation label instead of suffixed with label_suffix (the default value is
        None)
    :param allow_already_reversed_relations : A boolean value that allows to have existing reversed
        relation in the document. This means that document originally contains a pair of relations
        which are reverse of each other. If this parameter is disabled and such relation is found
        then an exception is raised. (the default value is False)
    """

    def __init__(
        self,
        relation_layer: str = "relations",
        label_suffix: str = "_reversed",
        symmetric_relation_labels: list[str] | None = None,
        allow_already_reversed_relations: bool = False,
        use_predictions: bool = False,
        collect_statistics: bool = False,
    ):
        self.symmetric_relation_labels = symmetric_relation_labels or []
        self.label_suffix = label_suffix
        self.relation_layer = relation_layer
        self.use_predictions = use_predictions
        self.allow_already_reversed_relations = allow_already_reversed_relations
        self.collect_statistics = collect_statistics
        self.reset_statistics()

    def reset_statistics(self):
        self._statistics = {
            "added_relations": defaultdict(int),
            "already_reversed_relations": defaultdict(int),
            "num_available_relations": 0,
            "num_added_relations": 0,
        }

    def show_statistics(self, description: str | None = None):
        description = description or "Statistics"
        logger.info(f"{description}:\n{json.dumps(dict(self._statistics))}")

    def __call__(self, document: D) -> D:
        # get all relations before adding any reversed
        rel_layer = document[self.relation_layer]
        if self.use_predictions:
            rel_layer = rel_layer.predictions
        rels = list(rel_layer)
        if self.collect_statistics:
            self._statistics["num_available_relations"] += len(rels)
        available_relations = {(rel.head, rel.tail): rel for rel in rels}
        for rel in rels:
            new_label = (
                rel.label
                if rel.label in self.symmetric_relation_labels
                else f"{rel.label}{self.label_suffix}"
            )
            new_relation = BinaryRelation(label=new_label, head=rel.tail, tail=rel.head)
            if (new_relation.head, new_relation.tail) in available_relations:
                # If an entity pair of reversed relation is present in the available relations then we check if we want
                # to allow already existing reversed relations or not. If we allow then we do not add the reversed
                # relation to the document but move on to the next relation otherwise we generate a LookupError
                # exception.
                if self.allow_already_reversed_relations:
                    if self.collect_statistics:
                        self._statistics["already_reversed_relations"][new_relation.label] += 1
                    continue
                else:
                    raise LookupError(
                        f"Entity pair of new relation ({new_relation}) already belongs to a relation: "
                        f"{available_relations[(new_relation.head, new_relation.tail)]}"
                    )
            else:
                rel_layer.append(new_relation)
                if self.collect_statistics:
                    self._statistics["added_relations"][new_relation.label] += 1
                    self._statistics["num_added_relations"] += 1

        return document

    def enter_dataset(self, dataset: Dataset | IterableDataset, name: str | None = None) -> None:
        if self.collect_statistics:
            self.reset_statistics()

    def exit_dataset(self, dataset: Dataset | IterableDataset, name: str | None = None) -> None:
        if self.collect_statistics:
            self.show_statistics(description=name)
