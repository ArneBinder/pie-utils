import copy
import logging
from typing import Callable, Dict, Optional

from pytorch_ie import Dataset
from pytorch_ie.core import Document

from pie_utils.statistics import WithStatistics

logger = logging.getLogger(__name__)

DatasetDict = Dict[str, Dataset]


def create_train_test_split(
    dataset: DatasetDict, split_name: str, **train_test_split_kwargs
) -> DatasetDict:
    """This method divides a dataset using a given split_name into train and test split of given
    length. It then returns new dataset with train and test split of required length.

    params:
    dataset : instance of Dataset to split
    split_name : split of given dataset used to create new dataset
    **train_test_split_kwargs : information related to new train and test splits such as split names and respective
                                lengths
    """
    new_dataset = copy.copy(dataset)
    logger.info(f'create new splits from split "{split_name}"')
    split = new_dataset.pop(split_name)
    # see https://github.com/ChristophAlt/pytorch-ie/issues/189
    new_splits = {
        split_name: Dataset.from_hf_dataset(new_split, document_type=split.document_type)
        for split_name, new_split in split.train_test_split(**train_test_split_kwargs).items()
    }
    overlapping_split_names = set(new_dataset.keys()) & set(new_splits.keys())
    if len(overlapping_split_names) > 0:
        raise Exception(f"dataset already has these splits: {overlapping_split_names}")
    new_dataset.update(new_splits)
    return new_dataset


def process_documents(
    dataset: DatasetDict,
    document_processors: Dict[str, Callable[[Document], Document]],
    single_map: bool = True,
) -> DatasetDict:
    """This method uses given document processors to update each document of the dataset and
    returns updated dataset.

    IMPORTANT: Works only for document processors that do not change the document type.

    params:
    dataset: instance of Dataset to be processed
    document_processors : dictionary containing document processors which will be used to update documents in given
                        dataset
    single_map: if True (default), create a single mapping function from all document_processors and apply that
    """

    # test coverage is not captured for functions passed to Dataset.map
    def _process_document(doc):  # pragma: no cover
        for p_name, p in document_processors.items():
            doc = p(doc)
        return doc

    new_dataset = copy.copy(dataset)
    if single_map:
        logger.info(f"call document preprocessors: {list(document_processors)}")
        for s, d in new_dataset.items():
            for p_name, p in document_processors.items():
                if isinstance(p, WithStatistics):
                    p.reset_statistics()
            new_dataset[s] = d.map(_process_document)
            for p_name, p in document_processors.items():
                if isinstance(p, WithStatistics):
                    p.show_statistics(f"Statistics for {p_name} at split {s}")
    else:
        for p_name, p in document_processors.items():
            logger.info(f"call document preprocessor: {p_name}")
            for s, d in new_dataset.items():
                if isinstance(p, WithStatistics):
                    p.reset_statistics()
                new_dataset[s] = d.map(p)
                if isinstance(p, WithStatistics):
                    p.show_statistics(f"Statistics for {p_name} at split {s}")
    return new_dataset


def process_datasets(
    dataset: DatasetDict,
    dataset_processors: Dict[str, Callable[[Dataset], Optional[Dataset]]],
) -> DatasetDict:
    """This method can update different splits of dataset using given dataset processors and
    consequently returns updated or same dataset.

    params:
    dataset: instance of Dataset
    dataset_processors: dictionary containing dataset processors which will be used to update each dataset split in given
                        dataset
    """
    new_dataset = copy.copy(dataset)
    for p_name, p in dataset_processors.items():
        logger.info(f"call dataset processor: {p_name}")
        for split_name, split in new_dataset.items():
            processed_or_none = p(split)
            if processed_or_none is not None:
                new_dataset[split_name] = processed_or_none
    return new_dataset


def rename_splits(dataset: DatasetDict, names: Dict[str, str]) -> DatasetDict:
    """This method renames the split name of the given data using a dictionary mapping and returns
    updated dataset.

    params:
    dataset: instance of Dataset
    name: dictionary containing old split names mapping to new split names
    """
    new_dataset = copy.copy(dataset)
    for old_name, new_name in names.items():
        new_dataset[new_name] = new_dataset.pop(old_name)
    return new_dataset
