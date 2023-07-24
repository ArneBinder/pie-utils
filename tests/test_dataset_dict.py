import logging
from dataclasses import dataclass
from pathlib import Path

import datasets
import pytest
from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument

from pie_utils import DatasetDict
from pie_utils.dataset_dict import get_pie_dataset_type
from tests import FIXTURES_ROOT

logger = logging.getLogger(__name__)

DATA_PATH = FIXTURES_ROOT / "dataset_dict" / "conll2003_extract"


@pytest.fixture(scope="module")
def dataset():
    return datasets.load_dataset("pie/conll2003")


@pytest.mark.skip(reason="don't create fixture data again")
def test_create_fixture_data():
    conll2003 = DatasetDict.load_dataset("pie/conll2003")
    for split in list(conll2003):
        # restrict all splits to 3 examples
        conll2003 = conll2003.select(split=split, stop=3)
    conll2003.to_json(FIXTURES_ROOT / "dataset_dict" / "conll2003_extract_new")


@dataclass
class DocumentWithEntitiesAndRelations(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


@pytest.fixture(scope="module")
def dataset_dict():
    return DatasetDict.from_json(
        data_dir=DATA_PATH, document_type=DocumentWithEntitiesAndRelations
    )


def test_from_json(dataset_dict):
    assert set(dataset_dict) == {"train", "test", "validation"}


def test_to_json_and_back(dataset_dict, tmp_path):
    path = Path(tmp_path) / "dataset_dict"
    dataset_dict.to_json(path)
    dataset_dict_from_json = DatasetDict.from_json(
        data_dir=path,
        document_type=dataset_dict.document_type,
    )
    assert set(dataset_dict_from_json) == set(dataset_dict)
    for split in dataset_dict:
        assert len(dataset_dict_from_json[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_from_json[split], dataset_dict[split]):
            assert doc1 == doc2


def test_document_type_empty_no_splits():
    # the document type is not defined if the dataset does not contain any splits
    assert DatasetDict().document_type is None


def test_document_type_different_types(dataset_dict):
    # load the example dataset as a different document type
    dataset_dict_different_type = DatasetDict.from_json(
        data_dir=DATA_PATH,
        document_type=TextDocument,
    )
    assert dataset_dict_different_type.document_type is TextDocument
    # create a dataset dict with different document types for train and test splits
    dataset_dict_different_types = DatasetDict(
        {
            "train": dataset_dict["train"],
            "test": dataset_dict_different_type["test"],
        }
    )
    # accessing the document type should raise an error with the message that starts with
    # "dataset contains splits with different document types:"
    with pytest.raises(ValueError) as excinfo:
        dataset_dict_different_types.document_type
        assert str(excinfo.value).startswith(
            "dataset contains splits with different document types:"
        )


@pytest.fixture(scope="module")
def iterable_dataset_dict():
    return DatasetDict.from_json(
        data_dir=DATA_PATH, document_type=DocumentWithEntitiesAndRelations, streaming=True,
    )


def test_iterable_dataset_dict(iterable_dataset_dict):
    assert set(iterable_dataset_dict) == {"train", "test", "validation"}
