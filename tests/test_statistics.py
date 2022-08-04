import logging
from typing import Optional

from src.pie_utils.statistics import WithStatistics

logger = logging.getLogger(__name__)


class DummyCollector(WithStatistics):
    def __init__(self):
        self.number = 0

    def reset_statistics(self):
        self.number = 0

    def show_statistics(self, description: Optional[str] = None):
        description = description or "Statistics"
        logger.info(f"{description}: {self.number}")


def test_statistics():
    dummy = DummyCollector()
    assert dummy.number == 0

    dummy.number += 1
    assert dummy.number == 1

    dummy.show_statistics()
