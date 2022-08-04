from abc import abstractmethod
from typing import Optional


class WithStatistics:
    @abstractmethod
    def reset_statistics(self):
        """resets value of property statistics."""
        raise NotImplementedError

    @abstractmethod
    def show_statistics(self, description: Optional[str] = None):
        """logs value of statistics to the console."""
        raise NotImplementedError
