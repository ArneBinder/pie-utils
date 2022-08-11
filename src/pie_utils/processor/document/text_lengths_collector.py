from __future__ import annotations

import json
import logging
import statistics

from pytorch_ie.annotations import Span
from transformers import AutoTokenizer

from pie_utils.statistics import WithStatistics

from ..document import DocumentWithPartition

logger = logging.getLogger(__name__)


class TextLengthsCollector(WithStatistics):
    """This document processors collects the text lengths in means of token numbers and allows to
    show them as json dict and, if plotext is installed, as histogram. Its nature is purely
    statistical, it does not modify the documents.

    Presented values:
     * min, max, mean, and stddev of the collected text lengths,
     * num_docs (number of processed documents), and
     * if use_partition is enabled, num_parts (number of precessed parts)

    :param tokenizer_name_or_path the identifier of the Huggingface tokenizer that will be used
    :param use_partition a boolean flag to enable considering a partition, i.e. tokenize and
        collect the lengths for the partition entries (e.g. sentences or sections) individually.
    :param tokenizer_kwargs a dictionary containing further keyword arguments passed when calling
        the tokenizer
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        use_partition: bool | None = False,
        tokenizer_kwargs: dict | None = None,
    ):
        self.use_partition = use_partition
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.reset_statistics()

    def reset_statistics(self):
        self.text_lengths = []
        self.num_docs = 0
        self.num_parts = 0

    def show_statistics(self, description: str | None = None):
        description = description or "Statistics for text lengths"
        caption = f"{description} (tokenizer_name_or_path={self.tokenizer_name_or_path})"
        try:
            import plotext as plt

            plt.clf()
            plt.hist(data=self.text_lengths)
            plt.title(caption)
            plt.show()

        # exclude from test coverage since this would require to uninstall plotext and
        # just a simple logging is performed here
        except ModuleNotFoundError:  # pragma: no cover
            logger.info("install plotext to display the data as histogram at the console")

        stats = {
            "min": min(self.text_lengths),
            "max": max(self.text_lengths),
            "mean": statistics.mean(self.text_lengths),
            "stddev": statistics.pstdev(self.text_lengths),
            "num_docs": self.num_docs,
        }
        if self.use_partition:
            stats["num_parts"] = self.num_parts

        logger.info(f"{caption}):\n{json.dumps(stats, indent=2)}")

    def __call__(self, document: DocumentWithPartition) -> DocumentWithPartition:
        partition = (
            document.partition if self.use_partition else [Span(start=0, end=len(document.text))]
        )
        tokenized = self.tokenizer(
            [document.text[part.start : part.end] for part in partition], **self.tokenizer_kwargs
        )
        new_lengths = [len(encoding) for encoding in tokenized.encodings]
        self.text_lengths.extend(new_lengths)
        self.num_parts += len(partition)
        self.num_docs += 1
        return document
