import copy
import logging
from typing import Callable, Dict, Optional

from pytorch_ie import Dataset
from pytorch_ie.core import Document

logger = logging.getLogger(__name__)


def create_train_test_split(
    dataset: Dict[str, Dataset], split_name: str, **train_test_split_kwargs
):
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
    dataset: Dict[str, Dataset],
    document_processors: Dict[str, Callable[[Document], Document]],
    single_map: bool = True,
):
    """This method uses given document processors to update each document of the dataset and
    returns updated dataset.

    params:
    dataset: instance of Dataset to be processed
    document_processors : dictionary containing document processors which will be used to update documents in given
                        dataset
    single_map: if True (default), create a single mapping function from all document_processors and apply that
    """

    def _process_document(doc: Document) -> Document:
        for p_name, p in document_processors.items():
            doc = p(doc)
        return doc

    new_dataset = copy.copy(dataset)
    if single_map:
        logger.info(f"call document preprocessors: {list(document_processors)}")
        for s, d in new_dataset.items():
            new_dataset[s] = d.map(_process_document)
    else:
        for p_name, p in document_processors.items():
            logger.info(f"call document preprocessor: {p_name}")
            for s, d in new_dataset.items():
                new_dataset[s] = d.map(p)
    return new_dataset


def process_datasets(
    dataset: Dict[str, Dataset],
    dataset_processors: Dict[str, Callable[[Dataset], Optional[Dataset]]],
):
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


def rename_splits(dataset: Dict[str, Dataset], names: Dict[str, str]):
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