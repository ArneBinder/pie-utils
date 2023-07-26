from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Iterable, TypeVar

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


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

