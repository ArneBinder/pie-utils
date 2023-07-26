from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Iterable, TypeVar

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document

from pie_utils.span.slice import distance, is_contained_in

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def _remove_overlapping_entities(
    entities: Iterable[dict[str, Any]], relations: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sorted_entities = sorted(entities, key=lambda span: span["start"])
    entities_wo_overlap = []
    skipped_entities = []
    last_end = 0
    for entity_dict in sorted_entities:
        if entity_dict["start"] < last_end:
            skipped_entities.append(entity_dict)
        else:
            entities_wo_overlap.append(entity_dict)
            last_end = entity_dict["end"]
    if len(skipped_entities) > 0:
        logger.warning(f"skipped overlapping entities: {skipped_entities}")
    valid_entity_ids = {entity_dict["_id"] for entity_dict in entities_wo_overlap}
    valid_relations = [
        relation_dict
        for relation_dict in relations
        if relation_dict["head"] in valid_entity_ids and relation_dict["tail"] in valid_entity_ids
    ]
    return entities_wo_overlap, valid_relations


def remove_overlapping_entities(
    doc: D,
    entity_layer_name: str = "entities",
    relation_layer_name: str = "relations",
) -> D:
    document_dict = doc.asdict()
    entities_wo_overlap, valid_relations = _remove_overlapping_entities(
        entities=document_dict[entity_layer_name]["annotations"],
        relations=document_dict[relation_layer_name]["annotations"],
    )

    document_dict[entity_layer_name] = {
        "annotations": entities_wo_overlap,
        "predictions": [],
    }
    document_dict[relation_layer_name] = {
        "annotations": valid_relations,
        "predictions": [],
    }
    new_doc = type(doc).fromdict(document_dict)

    return new_doc


def trim_spans(
    document: D,
    span_layer: str,
    text_field: str = "text",
    relation_layer: str | None = None,
    skip_empty: bool = True,
    inplace: bool = False,
) -> D:
    """Remove the whitespace at the beginning and end of span annotations. If a relation layer is
    given, the relations will be updated to point to the new spans.

    Args:
        document: The document to trim its span annotations.
        span_layer: The name of the span layer to trim.
        text_field: The name of the text field in the document.
        relation_layer: The name of the relation layer to update. If None, the relations will not be updated.
        skip_empty: If True, empty spans will be skipped. Otherwise, an error will be raised.
        inplace: If False, the document will be copied before trimming.

    Returns:
        The document with trimmed spans.
    """
    if not inplace:
        document = type(document).fromdict(document.asdict())

    spans: AnnotationList[LabeledSpan] = document[span_layer]
    old2new_spans = {}
    text = getattr(document, text_field)
    for span in spans:
        span_text = text[span.start : span.end]
        new_start = span.start + len(span_text) - len(span_text.lstrip())
        new_end = span.end - len(span_text) + len(span_text.rstrip())
        # if the new span is empty and skip_empty is True, skip it
        if new_start < new_end or not skip_empty:
            # if skip_empty is False and the new span is empty, log a warning and create a new zero-width span
            # by using the old start position
            if new_start >= new_end:
                logger.warning(
                    f"Span {span} is empty after trimming. {'Skipping it.' if skip_empty else ''}"
                )
                new_start = span.start
                new_end = span.start
            new_span = LabeledSpan(
                start=new_start,
                end=new_end,
                label=span.label,
                score=span.score,
            )
            old2new_spans[span] = new_span

    spans.clear()
    spans.extend(old2new_spans.values())

    if relation_layer is not None:
        relations: AnnotationList[BinaryRelation] = document[relation_layer]
        new_relations = []
        for relation in relations:
            if relation.head not in old2new_spans or relation.tail not in old2new_spans:
                logger.warning(
                    f"Relation {relation} is removed because one of its spans was removed."
                )
                continue
            head = old2new_spans[relation.head]
            tail = old2new_spans[relation.tail]
            new_relations.append(
                BinaryRelation(
                    head=head,
                    tail=tail,
                    label=relation.label,
                    score=relation.score,
                )
            )
        relations.clear()
        relations.extend(new_relations)

    return document


def add_reversed_relations(
    document: D,
    relation_layer: str = "relations",
    symmetric_relation_labels: list[str] | None = None,
    label_suffix: str = "_reversed",
    allow_already_reversed_relations: bool = False,
    use_predictions: bool = False,
    inplace: bool = False,
) -> D:
    if not inplace:
        document = type(document).fromdict(document.asdict())
    symmetric_relation_labels = symmetric_relation_labels or []

    rel_layer = document[relation_layer]
    if use_predictions:
        rel_layer = rel_layer.predictions

    # get all relations before adding any reversed
    available_relations = {(rel.head, rel.tail): rel for rel in rel_layer}
    for rel in list(rel_layer):
        new_label = (
            rel.label if rel.label in symmetric_relation_labels else f"{rel.label}{label_suffix}"
        )
        new_relation = BinaryRelation(label=new_label, head=rel.tail, tail=rel.head)
        if (new_relation.head, new_relation.tail) in available_relations:
            # If an entity pair of reversed relation is present in the available relations then we check if we want
            # to allow already existing reversed relations or not. If we allow then we do not add the reversed
            # relation to the document but move on to the next relation otherwise we generate a LookupError
            # exception.
            if allow_already_reversed_relations:
                continue
            else:
                raise LookupError(
                    f"Entity pair of new relation ({new_relation}) already belongs to a relation: "
                    f"{available_relations[(new_relation.head, new_relation.tail)]}"
                )
        else:
            rel_layer.append(new_relation)

    return document


def get_single_target_layer(document: Document, layer: AnnotationList):
    if len(layer._targets) != 1:
        raise Exception(
            f"the layer is expected to have exactly one target layer, but it has "
            f"the following targets: {layer._targets}"
        )
    target_layer_name = layer._targets[0]
    return document[target_layer_name]


def add_candidate_relations(
    document: D,
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
    inplace: bool = True,
) -> D:
    if not inplace:
        document = type(document).fromdict(document.asdict())
    rel_layer = document[relation_layer]
    if use_predictions:
        rel_layer = rel_layer.predictions
    available_relation_mapping = {(rel.head, rel.tail): rel for rel in rel_layer}
    if partition_layer is not None:
        available_partitions = document[partition_layer]
    else:
        available_partitions = [None]
    entity_layer = get_single_target_layer(document=document, layer=rel_layer)
    if use_predictions:
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
                if is_contained_in((entity.start, entity.end), (partition.start, partition.end))
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
                    distance_type,
                )
                if max_distance is not None and d > max_distance:
                    continue
                if (head, tail) in available_relation_mapping:
                    num_relations_in_partition += 1
                    distances_taken[available_relation_mapping[(head, tail)].label].append(d)
                    available_rels_within_allowed_distance.add(
                        available_relation_mapping[(head, tail)]
                    )
                    continue
                candidates_with_distance[(head, tail)] = d
    if sort_by_distance:
        candidates_with_distance_list = sorted(
            candidates_with_distance.items(), key=lambda item: item[1]
        )
    else:
        candidates_with_distance_list = list(candidates_with_distance.items())
        random.shuffle(candidates_with_distance_list)
    n_added = 0
    if n_max is not None:
        candidates_with_distance_list = candidates_with_distance_list[:n_max]
    if n_max_factor is not None:
        n_max_by_factor = int(len(rel_layer) * n_max_factor)
        candidates_with_distance_list = candidates_with_distance_list[:n_max_by_factor]
    # num_total_candidates = len(entities) * len(entities) - len(entities)
    # update_statistics("num_total_relation_candidates", num_total_candidates)
    # num_available_relations = len(rel_layer)
    # update_statistics("num_available_relations", num_available_relations)
    for (head, tail), d in candidates_with_distance_list:
        new_relation = BinaryRelation(label=label, head=head, tail=tail)
        rel_layer.append(new_relation)
        distances_taken[label].append(d)
        n_added += 1

    return document
