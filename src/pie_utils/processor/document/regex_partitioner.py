from __future__ import annotations

import json
import logging
import re
import statistics
from typing import Any, Callable, Iterable, Iterator, Match

from pytorch_ie.annotations import LabeledSpan

from pie_utils.statistics import WithStatistics

from ..document import DocumentWithPartitions

logger = logging.getLogger(__name__)


def _get_partitions_with_matcher(
    document: DocumentWithPartitions,
    matcher: Callable[[str], Iterable[Match]],
    label_group_id: int | None = None,  # = 1,
    label_whitelist: list[str] | None = None,
    skip_initial_partition: bool = False,  # = True
    default_partition_label: str = "partition",
    initial_partition_label: str | None = None,
) -> Iterator[LabeledSpan]:
    """This method yields LabeledSpans as partitions of the given document. matcher is used to
    search for a pattern in the document. If the pattern is found, it returns a Match object that
    contains matched groups. A partition is then created using a span in the matched groups. The
    span of a partition starts from the first match (inclusive) and ends at the next match
    (exclusive) or at the end of the document. A partition is labeled either using the default_partition_label or using the list
    of labels available in label_whitelist. It should be noted that none of the partitions overlap.

    :param document: A Document that is to be partitioned
    :param matcher: A method that is used to find a pattern in the document text and return an iterator yielding the
                    Match objects, e.g. re.compile(PATTERN).finditer
    :param label_group_id: An integer value (default:None) to select the desired match group from the Match object. This match
                            group is then used to create a label for the partition.
    :param label_whitelist: An optional list of labels (default:None) which are allowed to form a partition if label_group_id
                            is not None. label_whitelist is the whitelist for the labels created using label_group_id.
                            If label_whitelist is None, then all the labels created using label_group_id will form a
                            partition.
    :param skip_initial_partition: A boolean value (default:False) that prevents the initial partition to be saved in the
                                document.
    :param default_partition_label: A string value (default:partition) to be used as the default label for the parts if no
                                    label_group_id for the match object is provided.
    :param initial_partition_label: A string value (default:None) to be used as a label for the initial
                                partition. This is only used when skip_initial_partition is False. If it is None then
                                default_partition_label is used as initial_partition_label.
    """
    if initial_partition_label is None:
        initial_partition_label = default_partition_label
    previous_start = previous_label = None
    if not skip_initial_partition:
        if label_whitelist is None or initial_partition_label in label_whitelist:
            previous_start = 0
            previous_label = initial_partition_label
    for match in matcher(document.text):
        if label_group_id is not None:
            start = match.start(label_group_id)
            end = match.end(label_group_id)
            label = document.text[start:end]
        else:
            label = default_partition_label
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
    """RegexPartitioner partitions a document into multiple partitions using a regular expression.
    For more information, refer to get_partitions_with_matcher() method.

    :param pattern: A regular expression to search for in the text. It is also included at the beginning of each partition.
    :param collect_statistics: A boolean value (default:False) that allows to collect relevant statistics of the
                                document after partitioning. When this parameter is enabled, following stats are
                                collected:
                                1. partition_lengths: list of lengths of all partitions
                                2. num_partitions: list of number of partitions in each document
                                3. document_lengths: list of document lengths
                                show_statistics can be used to get statistical insight over these lists.
    :param partitioner_kwargs: keyword arguments for get_partitions_with_matcher() method
    """

    def __init__(self, pattern: str, collect_statistics: bool = False, **partitioner_kwargs):
        self.matcher = re.compile(pattern).finditer
        self.collect_statistics = collect_statistics
        self.reset_statistics()
        self.partitioner_kwargs = partitioner_kwargs

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

    def __call__(self, document: DocumentWithPartitions):
        partition_lengths = []
        for partition in _get_partitions_with_matcher(
            document, matcher=self.matcher, **self.partitioner_kwargs
        ):
            document.partitions.append(partition)
            partition_lengths.append(partition.end - partition.start)

        if self.collect_statistics:
            self.update_statistics("num_partitions", len(document.partitions))
            self.update_statistics("partition_lengths", partition_lengths)
            self.update_statistics("document_lengths", len(document.text))

        return document
