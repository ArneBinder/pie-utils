from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Iterable, Iterator, Match

from pytorch_ie.annotations import LabeledSpan, Span

from pie_utils.span.slice import have_overlap
from pie_utils.statistics import WithStatistics

from ..document import DocumentWithPartition

logger = logging.getLogger(__name__)


def get_partitions_with_matcher(
    document: DocumentWithPartition,
    matcher: Callable[[str], Iterable[Match]],
    label_group_id: int | None = None,  # = 1,
    label_whitelist: list[str] | None = None,
    skip_initial_partition: bool = False,  # = True
) -> Iterator[Span]:
    """Spans are created starting with the beginning of matching entries end ending with the start
    of the next matching one or the end of the document.

    Entries with xml node names that are not in label_whitelist, if it is set, will not be
    considered as match, i.e. their content will be added to the previous matching entry. If
    label_group_id is set, the content of the respective match group will be taken as label.
    Otherwise, it is set to None. If the flag skip_initial_partition is enabled, the content before
    the first match is not added as a partition. Note that the initial partition will get None as
    label since no matched element is available.
    """
    previous_start = previous_label = None
    if not skip_initial_partition:
        previous_start = 0
        previous_label = "<initial_part>"  # This is added here because if we want to keep initial partition then without
        # setting a label here, None is added as label which result in exception. We can have a parameter for
        # initial_label.
    for match in matcher(document.text):
        if label_group_id is not None:
            start = match.start(label_group_id)
            end = match.end(label_group_id)
            label = document.text[start:end]
        else:
            label = None
        if label_whitelist is None or label in label_whitelist:
            if previous_start is not None:
                end = match.start()
                span = LabeledSpan(start=previous_start, end=end, label=previous_label)
                yield span

            previous_start = match.start()
            previous_label = label

    if previous_start is not None:
        end = len(document.text)
        span = LabeledSpan(start=previous_start, end=end, label=previous_label)
        yield span


class SplitDocumentToPartitions(WithStatistics):
    def __init__(
        self,
        pattern: str,
        label_group_id: int | None = None,  # = 1,
        label_whitelist: list[str] | None = None,
        skip_initial_partition: bool = False,  # = True
        collect_statistics: bool = False,
    ):
        self.label_group_id = label_group_id
        self.label_whitelist = label_whitelist
        self.skip_initial_partition = skip_initial_partition
        self.matcher = re.compile(pattern).finditer
        self.collect_statistics = collect_statistics
        self.reset_statistics()

    def reset_statistics(self):
        self._statistics: dict[str, Any] = {
            "partition_lengths": [],
            "num_partitions": [],
            "document_lengths": [],
        }

    def show_statistics(self, description: str | None = None):
        description = description or "Statistics"
        logger.info(f"{description}: \n{json.dumps(self._statistics, indent=2)}")

    def update_statistics(self, key: str, value: int | str | list):
        if self.collect_statistics:
            if isinstance(value, list):
                self._statistics[key] += value
            elif isinstance(value, str) or isinstance(value, int):
                self._statistics[key].append(value)
            else:
                raise TypeError(
                    f"type of given key [{type(key)}] or value [{type(value)}] is incorrect."
                )

    def __call__(self, document: DocumentWithPartition):
        partitions_for_doc: list[Span] = []
        partition_lengths = []
        for partition in get_partitions_with_matcher(
            document,
            matcher=self.matcher,
            skip_initial_partition=self.skip_initial_partition,
            label_whitelist=self.label_whitelist,
            label_group_id=self.label_group_id,
        ):

            document.partition.append(partition)
            partitions_for_doc.append(partition)
            partition_lengths.append(partition.end - partition.start)

        if self.collect_statistics:
            self.update_statistics("num_partitions", len(document.partition))
            self.update_statistics("partition_lengths", partition_lengths)
            self.update_statistics("document_lengths", len(document.text))

        return document
