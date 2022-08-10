import logging
from typing import Callable, Iterable, Iterator, List, Match, Optional

from pytorch_ie.annotations import Span

from pie_utils.processor.document import DocumentWithEntitiesRelationsAndPartition
from pie_utils.span.slice import have_overlap
from pie_utils.statistics import WithStatistics

logger = logging.getLogger(__name__)


def get_partitions_with_matcher(
    document: DocumentWithEntitiesRelationsAndPartition,
    matcher: Callable[[str], Iterable[Match]],
    label_group_id: Optional[int] = None,  # = 1,
    label_whitelist: Optional[List[str]] = None,
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
    previous_start = None
    if not skip_initial_partition:
        previous_start = 0
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
                span = Span(start=previous_start, end=end)
                yield span

            previous_start = match.start()

    if previous_start is not None:
        end = len(document.text)
        span = Span(start=previous_start, end=end)
        yield span


class SplitDocumentToPartitions(WithStatistics):
    def __init__(
        self,
        **kwargs,
    ):
        self.kwargs = kwargs

    def reset_statistics(self):
        pass

    def show_statistics(self, description: Optional[str] = None):
        pass

    def __call__(self, document: DocumentWithEntitiesRelationsAndPartition):
        partitions_for_doc: List[Span] = []
        for partition in get_partitions_with_matcher(document, **self.kwargs):
            # just a sanity check
            for s in partitions_for_doc:
                if have_overlap((s.start, s.end), (partition.start, partition.end)):
                    logger.error(f"overlap: {partition} with {s}")

            document.partition.append(partition)
            partitions_for_doc.append(partition)
        # todo:
        # 1. collect distribution of lengths of partition
        # 2. collect total number of partitions in a document
        # 3. full text of each partition

        return document
