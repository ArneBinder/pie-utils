from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, TypeVar

from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.annotations import BinaryRelation
from pytorch_ie.core import Document
from pytorch_ie.data.common import EnterDatasetMixin, ExitDatasetMixin
from pytorch_ie.utils.span import is_contained_in

from pie_utils.span.slice import distance

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=Document)


class CandidateRelationAdder(EnterDatasetMixin, ExitDatasetMixin):
    """CandidateRelationAdder adds binary relations to a document based on various parameters. It
    goes through combinations of available entity pairs as possible candidates for new relations.
    Entity pairs which are already part of document as a relation are not added again.

    params:
        label: label for the new relations to be added
        use_partition: A boolean parameter to enable partition wise relation creation. If this parameter is enabled then
                    it uses the partition of the document which is expected to be a list of non-overlapping spans. In
                    this case, entity pairs that are not completely in the same partition entry are discarded.
        max_distance: An optional parameter which restricts the maximum distance between entities of the new relations.
                    If the distance between entities of the candidate pair is more than the value of this parameter then
                    the candidate pair is discarded.
        distance_type: The type of distance to be calculated between two entities when using max_distance. It can be
                        inner (inner), outer or center.
        added_rels_upper_bound_factor: [DEPRECATED] It is an optional float parameter to restrict the number of added
                                        relations by an upper bound. The upper bound n is calculated as,
                                                n = added_rels_upper_bound_factor * num_available_relations
                                    see collect_statistics for the definition of num_available_relations.
                                    Note: This is deprecated and should not be used because during testing, the
                                    num_available_relations should not be available. Use max_distance instead which does
                                     not depend on that.
        sort_by_distance: This parameter decides if candidate entity pairs are ordered in sorted manner or not. This is
                        only relevant when using added_rels_upper_bound_factor to restrict the number of added relations
                         because only the first n candidate pairs after sorting are taken. Therefore, it allows to
                         discard the highest distance pairs. Sorting is done based on the distance between the entity
                         pairs.
        collect_statistics: This parameter defines whether advanced statistics are collected or not. Few terms to know
                            in order to understand the statistics properly:
                            1. available relations: relations which are part of the document originally.
                            2. candidate relations: possible entity pairs in the document. This includes available
                                                    relations as well. Note: an entity cannot form a pair with itself.
                            3. added relations: entity pairs which are not in available relations but in candidate
                                                relations. This means available relations and added relations are
                                                mutually exclusive.
                        If this parameter is enabled then following statistics are collected:
                        1. num_total_relation_candidates: Number of possible entity pairs in the document.
                        2. num_available_relations: Number of available relations in the document.
                        3. available_rels_within_allowed_distance: Number of available relations where the distance between
                                                                entity pairs does not exceed max_distance
                        4. num_added_relation_not_taken: Number of added relations which are not taken due to exceeding
                                                        max_distance or upper bound on allowed new relations.
                        5. num_rels_within_allowed_distance: Number of candidate entity pairs whose distance is less
                                                            than max_distance.
                        6. num_candidates_not_taken: Candidate relations not taken due to exceeding max_distance or
                                                    upper bound on allowed new relations. It is a dictionary with
                                                    relation label as key and number of such relations not taken as value
                        7. distances_taken: Dictionary containing relation label as key and list of distances between
                                            the entity pairs of corresponding relations.
    """

    def __init__(
        self,
        label: str = "no_relation",
        relation_layer: str = "relations",
        use_predictions: bool = False,
        partition_layer: str | None = None,
        max_distance: int | None = None,
        distance_type: str = "inner",
        sort_by_distance: bool = True,
        n_max: int | None = None,
        # this should not be used during prediction, because it will leak gold relation information!
        n_max_factor: float | None = None,
        collect_statistics: bool = False,
    ):
        self.label = label
        self.relation_layer = relation_layer
        self.use_predictions = use_predictions
        self.partition_layer = partition_layer
        self.max_distance = max_distance
        self.distance_type = distance_type
        self.collect_statistics = collect_statistics
        self.sort_by_distance = sort_by_distance
        self.n_max = n_max
        self.n_max_factor = n_max_factor
        self.reset_statistics()

    def reset_statistics(self):
        self._statistics: dict[str, Any] = {
            "num_total_relation_candidates": 0,
            "num_available_relations": 0,
            "available_rels_within_allowed_distance": 0,
            "num_added_relation_not_taken": 0,
            "num_rels_within_allowed_distance": 0,
            "num_candidates_not_taken": defaultdict(int),
            "distances_taken": defaultdict(list),
        }

    def show_statistics(self, description: str | None = None):
        description = description or "Statistics"
        logger.info(f"{description}: \n{json.dumps(self._statistics, indent=2)}")

    def update_statistics(self, key: str, value: int | dict[str, list] | dict[str, int]):
        if self.collect_statistics:
            if isinstance(value, int):
                self._statistics[key] += value
            elif isinstance(value, Dict):
                for k, v in value.items():
                    if isinstance(v, List):
                        self._statistics[key][k] += v
                    elif isinstance(v, int):
                        self._statistics[key][k] += v
                    else:
                        raise TypeError(
                            f"type of given key [{type(key)}] or value [{type(value)}] is incorrect."
                        )
            else:
                raise TypeError(
                    f"type of given key [{type(key)}] or value [{type(value)}] is incorrect."
                )

    def __call__(self, document: D) -> D:
        rel_layer = document[self.relation_layer]
        if self.use_predictions:
            rel_layer = rel_layer.predictions

        available_relation_mapping = {(rel.head, rel.tail): rel for rel in rel_layer}
        if self.partition_layer is not None:
            available_partitions = document[self.partition_layer]
        else:
            available_partitions = [None]

        entity_layer = rel_layer.target_layer
        if self.use_predictions:
            entity_layer = entity_layer.predictions

        candidates_with_distance = {}
        distances_taken = defaultdict(list)
        num_relations_in_partition = 0
        available_rels_within_allowed_distance = set()
        for partition in available_partitions:
            if partition is not None:
                available_entities = [
                    entity
                    for entity in entity_layer
                    if is_contained_in(
                        (entity.start, entity.end), (partition.start, partition.end)
                    )
                ]
            else:
                available_entities = list(entity_layer)
            for head in available_entities:
                for tail in available_entities:
                    if head == tail:
                        continue
                    d = distance(
                        (head.start, head.end),
                        (tail.start, tail.end),
                        self.distance_type,
                    )
                    if self.max_distance is not None and d > self.max_distance:
                        continue
                    if (head, tail) in available_relation_mapping:
                        num_relations_in_partition += 1
                        distances_taken[available_relation_mapping[(head, tail)].label].append(d)
                        available_rels_within_allowed_distance.add(
                            available_relation_mapping[(head, tail)]
                        )
                        continue
                    candidates_with_distance[(head, tail)] = d
        if self.sort_by_distance:
            candidates_with_distance_list = sorted(
                candidates_with_distance.items(), key=lambda item: item[1]
            )
        else:
            candidates_with_distance_list = list(candidates_with_distance.items())
            random.shuffle(candidates_with_distance_list)
        n_added = 0
        if self.n_max is not None:
            candidates_with_distance_list = candidates_with_distance_list[: self.n_max]
        if self.n_max_factor is not None:
            n_max_by_factor = int(len(rel_layer) * self.n_max_factor)
            candidates_with_distance_list = candidates_with_distance_list[:n_max_by_factor]
        num_total_candidates = len(entity_layer) * len(entity_layer) - len(entity_layer)
        self.update_statistics("num_total_relation_candidates", num_total_candidates)
        num_available_relations = len(rel_layer)
        self.update_statistics("num_available_relations", num_available_relations)
        for (head, tail), d in candidates_with_distance_list:
            new_relation = BinaryRelation(label=self.label, head=head, tail=tail)
            rel_layer.append(new_relation)
            distances_taken[self.label].append(d)
            n_added += 1

        if self.collect_statistics:
            self.update_statistics("distances_taken", distances_taken)
            self.update_statistics(
                "available_rels_within_allowed_distance",
                len(available_rels_within_allowed_distance),
            )
            available_rels_exceeding_allowed_distance = (
                set(available_relation_mapping.values()) - available_rels_within_allowed_distance
            )

            num_rels_within_allowed_distance = sum(len(v) for v in distances_taken.values())
            self.update_statistics(
                "num_rels_within_allowed_distance", num_rels_within_allowed_distance
            )
            num_rels_taken = (
                len(available_rels_exceeding_allowed_distance) + num_rels_within_allowed_distance
            )
            num_added_relation_not_taken = num_total_candidates - num_rels_taken
            self.update_statistics("num_added_relation_not_taken", num_added_relation_not_taken)
            num_candidates_not_taken: dict[str, int] = defaultdict(lambda: 0)
            for rel in available_rels_exceeding_allowed_distance:
                num_candidates_not_taken[rel.label] += 1
            num_candidates_not_taken[self.label] = num_added_relation_not_taken
            self.update_statistics("num_candidates_not_taken", num_candidates_not_taken)

        return document

    def enter_dataset(self, dataset: Dataset | IterableDataset, name: str | None = None) -> None:
        if self.collect_statistics:
            self.reset_statistics()

    def exit_dataset(self, dataset: Dataset | IterableDataset, name: str | None = None) -> None:
        if self.collect_statistics:
            self.show_statistics(description=name)
