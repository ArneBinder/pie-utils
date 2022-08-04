import logging
from dataclasses import dataclass
from typing import Optional

import datasets
import pytest
from pytorch_ie import Dataset
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextDocument

from pie_utils.dataset import (
    create_train_test_split,
    process_datasets,
    process_documents,
    rename_splits,
)
from pie_utils.statistics import WithStatistics

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def dataset():
    return datasets.load_dataset("pie/conll2003")


def test_create_train_test_split(dataset):
    """Given dataset is pie/conll2003 dataset from HuggingFace with train, validation and test
    split.

    New dataset should be created from the train split of given dataset which should contain train
    and test splits. Size of train split should be 100 and test should be 10.
    """
    dataset_to_split = {"train": dataset["train"]}
    new_dataset = create_train_test_split(
        dataset=dataset_to_split,
        split_name="train",
        train_size=100,
        test_size=10,
    )
    assert set(new_dataset.keys()) == {"train", "test"}
    for k, v in new_dataset.items():
        assert isinstance(v, Dataset)
    assert len(new_dataset["train"]) == 100
    assert len(new_dataset["test"]) == 10


def test_create_train_test_split_already_exists(dataset):
    with pytest.raises(Exception, match="dataset already has these splits: {'test'}"):
        create_train_test_split(
            dataset={"train": dataset["train"], "test": dataset["test"]},
            split_name="train",
        )


class DummyDocumentProcessor:
    def __call__(self, document: Document) -> Document:
        return document


@dataclass
class CoNLL2003Document(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


class CopyAnnotationToPredictionsDocumentPreprocessor(WithStatistics):
    def reset_statistics(self):
        self.num_copied = 0

    def show_statistics(self, description: Optional[str] = None):
        logger.info(f"number of copied entities: {self.num_copied}")

    def __call__(self, document: CoNLL2003Document) -> CoNLL2003Document:
        entities = list(document.entities)
        document.entities.predictions.extend(entities)
        self.num_copied += len(entities)
        return document


@pytest.mark.parametrize("single_map", [True, False])
def test_process_documents(dataset, single_map):
    """Given dataset is pie/conll2003 dataset from HuggingFace with train, validation and test
    split.

    Two document processor is used to update documents in the given dataset. dummy document
    processor does nothing to the document, however copy document processor copies entities to
    predictions in each document. After applying these document processors, updated dataset should
    have same entities and predictions in each document.
    """
    new_dataset = process_documents(
        dataset=dataset,
        document_processors={
            "dummy": DummyDocumentProcessor(),
            "copy": CopyAnnotationToPredictionsDocumentPreprocessor(),
        },
        single_map=single_map,
    )
    for k, v in new_dataset.items():
        assert isinstance(v, Dataset)
        for d in v:
            assert set(d.entities) == set(d.entities.predictions)


def test_rename_splits(dataset):
    """Given dataset is pie/conll2003 dataset from HuggingFace with train, validation and test
    split.

    validation split should be renamed to dev and test split should be renamed to validation.
    """
    new_dataset = rename_splits(dataset, names={"validation": "dev", "test": "validation"})
    assert set(new_dataset.keys()) == {"train", "dev", "validation"}
    assert dataset["validation"] == new_dataset["dev"]
    assert dataset["test"] == new_dataset["validation"]


def collect_mean_text_lengths(dataset: Dataset):
    res = {"mean text length": sum(len(doc.text) for doc in dataset) / len(dataset)}
    assert res["mean text length"] > 0.0
    return dataset


def test_process_datasets(dataset):
    """Given dataset is pie/conll2003 dataset from HuggingFace with train, validation and test
    split.

    text_length dataset processor calculates mean text length in the dataset. Each split in the
    given dataset should have a positive mean text length.
    """
    new_dataset = process_datasets(
        dataset, dataset_processors={"text length": collect_mean_text_lengths}
    )
    for k, v in new_dataset.items():
        assert v == dataset[k]
