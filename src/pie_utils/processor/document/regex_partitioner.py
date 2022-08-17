from __future__ import annotations

import json
import logging
import re
import statistics
from typing import Any, Callable, Iterable, Iterator, Match

from pytorch_ie.annotations import LabeledSpan

from pie_utils.statistics import WithStatistics

from ..document import DocumentWithPartition

logger = logging.getLogger(__name__)


def get_partitions_with_matcher(
    document: DocumentWithPartition,
    matcher: Callable[[str], Iterable[Match]],
    label_group_id: int | None = None,  # = 1,
    label_whitelist: list[str] | None = None,
    skip_initial_partition: bool = False,  # = True
    initial_partition_label: str = "initial part",
) -> Iterator[LabeledSpan]:
    """This method yields LabeledSpans as partitions of the given document. matcher is used to
    search for a pattern in the document. If pattern is found then it returns Match objects that
    contains matched groups. The required match group is selected using label_group_id to create a
    partition label. The span of the partition ranges between two matched pattern. It should be
    noted that none of the partitions overlap.

    Besides the regular partitioning of the document, there are some other cases listed below:
    1. Document can be converted into single partition with initial_partition_label if skip_initial_partition is False
       and label_whitelist is empty list or label_group_id is None.
    2. If only initial part of the document is required as a partition then set label_group_id and label_whitelist as
       None and skip_initial_partition as False.
    3. There will be no partitions if
        a. matcher could not find pattern in document text
        b. label_whitelist is empty list and skip_initial_partition is True
        c. label_group_id and label_whitelist is None and skip_initial_partition is True
        d. label_group_id is None and skip_initial_partition is False

    :param document: A Document that is to be partitioned
    :param matcher: A method that is used to match a pattern in the document text and return an iterator yielding the
                    Match objects.
    :param label_whitelist: A list of labels which are allowed to form a partition. (the default value is None)
    :param label_group_id: An integer value to select suitable match group from Match object. (the default value is None)
    :param skip_initial_partition: A boolean value that prevents initial partition to be saved in the document. (the
                                default value is False)
    :param initial_partition_label: A string value used as a partition label for initial partition. This is only used when
                                skip_initial_partition is False.
    """
    previous_start = previous_label = None
    if not skip_initial_partition:
        previous_start = 0
        previous_label = initial_partition_label
    for match in matcher(document.text):
        if label_group_id is not None:
            start = match.start(label_group_id)
            end = match.end(label_group_id)
            label = document.text[start:end]
        else:
            label = None
        if label_whitelist is None or label in label_whitelist:
            if previous_start is not None and previous_label is not None:
                end = match.start()
                span = LabeledSpan(start=previous_start, end=end, label=previous_label)
                yield span

            previous_start = match.start()
            previous_label = label

    if previous_start is not None and previous_label is not None:
        end = len(document.text)
        span = LabeledSpan(start=previous_start, end=end, label=previous_label)
        yield span


class RegexPartitioner(WithStatistics):
    """RegexPartitioner partitions a document into multiple parts using a regular expression.

    :param pattern: A regular expression to search for in the text.
    :param label_group_id: An integer value to select required match group from Match object (the default value is None)
    :param label_whitelist: A list of labels which are allowed to form a partition. (the default value is None)
    :param skip_initial_partition: A boolean value that prevents initial partition to be saved in the document. (the
                                default value is False)
    :param initial_partition_label: A string value to be used as partition label for the initial partition. This label doesn't
                                have to be included in label_whitelist. This is only used when skip_initial_partition is
                                False. (the default value is "initial part")
    :param collect_statistics: A boolean value that allows to collect relevant statistics of the document after
                                partitioning. When this parameter is enabled, following stats are collected:
                                1. partition_lengths: list of lengths of all partitions
                                2. num_partitions: list of number of partitions in each document
                                3. document_lengths: list of document lengths
                                show_statistics can be used to get statistical insight over these lists.
    """

    def __init__(
        self,
        pattern: str,
        label_group_id: int | None = None,  # = 1,
        label_whitelist: list[str] | None = None,
        skip_initial_partition: bool = False,  # = True
        initial_partition_label: str = "initial part",
        collect_statistics: bool = False,
    ):
        self.label_group_id = label_group_id
        self.label_whitelist = label_whitelist
        self.skip_initial_partition = skip_initial_partition
        self.matcher = re.compile(pattern).finditer
        self.initial_partition_label = initial_partition_label
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
        statistics_show = {
            key: {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "stddev": statistics.pstdev(values),
            }
            for key, values in self._statistics.items()
        }

        logger.info(f"{description}: \n{json.dumps(statistics_show, indent=2)}")

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
        partition_lengths = []
        for partition in get_partitions_with_matcher(
            document,
            matcher=self.matcher,
            skip_initial_partition=self.skip_initial_partition,
            label_whitelist=self.label_whitelist,
            label_group_id=self.label_group_id,
            initial_partition_label=self.initial_partition_label,
        ):
            document.partition.append(partition)
            partition_lengths.append(partition.end - partition.start)

        if self.collect_statistics:
            self.update_statistics("num_partitions", len(document.partition))
            self.update_statistics("partition_lengths", partition_lengths)
            self.update_statistics("document_lengths", len(document.text))

        return document
