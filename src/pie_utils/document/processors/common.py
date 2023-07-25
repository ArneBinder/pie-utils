from abc import abstractmethod
from typing import Optional, Union

from pytorch_ie import Dataset, IterableDataset


class EnterDatasetMixin:
    """Mixin for processors that enter a dataset context."""

    @abstractmethod
    def enter_dataset(
        self, dataset: Union[Dataset, IterableDataset], name: Optional[str] = None
    ) -> None:
        """Enter dataset context."""
        ...


class ExitDatasetMixin:
    """Mixin for processors that exit a dataset context."""

    @abstractmethod
    def exit_dataset(
        self, dataset: Union[Dataset, IterableDataset], name: Optional[str] = None
    ) -> None:
        """Exit dataset context."""
        ...


class EnterDatasetDictMixin:
    """Mixin for processors that enter a dataset dict context."""

    @abstractmethod
    def enter_dataset_dict(self, dataset_dict) -> None:
        """Enter dataset dict context."""
        ...


class ExitDatasetDictMixin:
    """Mixin for processors that exit a dataset dict context."""

    @abstractmethod
    def exit_dataset_dict(self, dataset_dict) -> None:
        """Exit dataset dict context."""
        ...
