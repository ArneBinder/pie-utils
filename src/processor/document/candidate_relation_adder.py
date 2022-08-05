from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, overload

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.utils.span import is_contained_in

from pie_utils.span.slice import distance
from pie_utils.statistics import WithStatistics

logger = logging.getLogger(__name__)


@dataclass
class DocumentWithEntitiesRelationsAndPartition(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    partition: AnnotationList[Span] = annotation_field(target="text")


class CandidateRelationAdder(WithStatistics):
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

    @overload
    def update_statistics(self, key: str, value: int):
        pass

    @overload
    def update_statistics(self, key: str, value: dict[str, list]):
        pass

    @overload
    def update_statistics(self, key: str, value: dict[str, int]):
        pass

    def update_statistics(self, key: str, value: int | dict[str, list] | dict[str, int]):
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

    def __init__(
        self,
        label: str = "no_relation",
        use_partition: bool | None = False,
        max_distance: int | None = None,
        distance_type: str = "inner",
        added_relations_upper_bound_factor: int | None = None,
        collect_statistics: bool = False,
        sort_by_distance: bool = True,
    ):
        self.label = label
        self.use_partition = use_partition
        self.max_distance = max_distance
        self.distance_type = distance_type
        self.added_rels_upper_bound_factor = added_relations_upper_bound_factor
        self.collect_statistics = collect_statistics
        self.sort_by_distance = sort_by_distance
        self.reset_statistics()

    def __call__(
        self,
        document: DocumentWithEntitiesRelationsAndPartition,
    ) -> DocumentWithEntitiesRelationsAndPartition:
        available_relations = document.relations
        available_relation_mapping = {(rel.head, rel.tail): rel for rel in available_relations}
        if self.use_partition:
            available_partitions = document.partition
        else:
            available_partitions = [Span(start=0, end=len(document.text))]
        entities = document.entities
        candidates_with_distance = {}
        distances_taken = defaultdict(list)
        num_relations_in_partition = 0
        available_rels_within_allowed_distance = set()
        for partition in available_partitions:
            available_entities = [
                entity
                for entity in entities
                if is_contained_in((entity.start, entity.end), (partition.start, partition.end))
            ]
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
        num_total_candidates = len(entities) * len(entities) - len(entities)
        self.update_statistics("num_total_relation_candidates", num_total_candidates)
        num_available_relations = len(available_relations)
        self.update_statistics("num_available_relations", num_available_relations)
        for (head, tail), d in candidates_with_distance_list:
            if self.added_rels_upper_bound_factor is not None and num_relations_in_partition > 0:
                num_added_relation = n_added + 1
                if num_added_relation > min(
                    num_available_relations * self.added_rels_upper_bound_factor,
                    num_total_candidates,
                ):
                    break
            new_relation = BinaryRelation(label=self.label, head=head, tail=tail)
            document.relations.append(new_relation)
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
