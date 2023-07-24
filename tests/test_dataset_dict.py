import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import datasets
import pytest
from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field
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
        data_dir=DATA_PATH,
        document_type=DocumentWithEntitiesAndRelations,
        streaming=True,
    )


def test_iterable_dataset_dict(iterable_dataset_dict):
    assert set(iterable_dataset_dict) == {"train", "test", "validation"}


def test_get_pie_dataset_type():
    hf_ds = datasets.load_dataset("json", data_dir=DATA_PATH, split="train")
    assert get_pie_dataset_type(hf_ds) == Dataset
    hf_ds_iterable = datasets.load_dataset(
        "json", data_dir=DATA_PATH, split="train", streaming=True
    )
    assert get_pie_dataset_type(hf_ds_iterable) == IterableDataset
    with pytest.raises(ValueError):
        get_pie_dataset_type("not a dataset")


def map_fn(doc):
    doc.text = doc.text.upper()
    return doc


@pytest.mark.parametrize(
    "function",
    [map_fn, "tests.test_dataset_dict.map_fn"],
)
def test_map(dataset_dict, function):
    dataset_dict_mapped = dataset_dict.map(function)
    for split in dataset_dict:
        assert len(dataset_dict_mapped[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_mapped[split], dataset_dict[split]):
            assert doc1.text == doc2.text.upper()


def test_map_noop(dataset_dict):
    dataset_dict_mapped = dataset_dict.map()
    for split in dataset_dict:
        assert len(dataset_dict_mapped[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_mapped[split], dataset_dict[split]):
            assert doc1 == doc2


def test_map_with_result_document_type(dataset_dict):
    dataset_dict_mapped = dataset_dict.map(result_document_type=TextDocument)
    for split in dataset_dict:
        assert len(dataset_dict_mapped[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_mapped[split], dataset_dict[split]):
            assert isinstance(doc1, TextDocument)
            assert isinstance(doc2, DocumentWithEntitiesAndRelations)
            assert doc1.text == doc2.text


def test_select(dataset_dict):
    # select documents by index
    dataset_dict_selected = dataset_dict.select(
        split="train",
        indices=[0, 2],
    )
    assert len(dataset_dict_selected["train"]) == 2
    assert dataset_dict_selected["train"][0] == dataset_dict["train"][0]
    assert dataset_dict_selected["train"][1] == dataset_dict["train"][2]

    # select documents by range
    dataset_dict_selected = dataset_dict.select(
        split="train",
        stop=2,
        start=1,
        step=1,
    )
    assert len(dataset_dict_selected["train"]) == 1
    assert dataset_dict_selected["train"][0] == dataset_dict["train"][1]

    # calling with no arguments that do result in the creation of indices should return the same dataset,
    # but will log a warning if other arguments (here "any_arg") are passed
    dataset_dict_selected = dataset_dict.select(split="train", any_arg="ignored")
    assert len(dataset_dict_selected["train"]) == len(dataset_dict["train"])
    assert dataset_dict_selected["train"][0] == dataset_dict["train"][0]
    assert dataset_dict_selected["train"][1] == dataset_dict["train"][1]
    assert dataset_dict_selected["train"][2] == dataset_dict["train"][2]


def test_rename_splits(dataset_dict):
    mapping = {
        "train": "train_renamed",
        "test": "test_renamed",
        "validation": "validation_renamed",
    }
    dataset_dict_renamed = dataset_dict.rename_splits(mapping)
    assert set(dataset_dict_renamed) == set(mapping.values())
    for split in dataset_dict:
        split_renamed = mapping[split]
        assert len(dataset_dict_renamed[split_renamed]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_renamed[split_renamed], dataset_dict[split]):
            assert doc1 == doc2


def test_rename_split_noop(dataset_dict):
    dataset_dict_renamed = dataset_dict.rename_splits()
    assert set(dataset_dict_renamed) == set(dataset_dict)
    for split in dataset_dict:
        assert len(dataset_dict_renamed[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_renamed[split], dataset_dict[split]):
            assert doc1 == doc2


def assert_doc_lists_equal(docs: Iterable[Document], other_docs: Iterable[Document]):
    assert all(doc1 == doc2 for doc1, doc2 in zip(docs, other_docs))


def test_add_test_split(dataset_dict):
    dataset_dict_with_test = dataset_dict.add_test_split(
        source_split="test", target_split="new_test", test_size=1, shuffle=False
    )
    assert "new_test" in dataset_dict_with_test
    assert len(dataset_dict_with_test["new_test"]) + len(dataset_dict_with_test["test"]) == len(
        dataset_dict["test"]
    )
    assert len(dataset_dict_with_test["new_test"]) == 1
    assert len(dataset_dict_with_test["test"]) == 2
    assert_doc_lists_equal(dataset_dict_with_test["new_test"], dataset_dict["test"][2:])
    assert_doc_lists_equal(dataset_dict_with_test["test"], dataset_dict["test"][:2])
    test_ids = [doc.id for doc in dataset_dict_with_test["test"]]
    new_test_ids = [doc.id for doc in dataset_dict_with_test["new_test"]]
    assert set(test_ids).intersection(set(new_test_ids)) == set()

    # remaining splits should be unchanged
    assert len(dataset_dict_with_test["train"]) == len(dataset_dict["train"])
    assert len(dataset_dict_with_test["validation"]) == len(dataset_dict["validation"])
    assert_doc_lists_equal(dataset_dict_with_test["train"], dataset_dict["train"])
    assert_doc_lists_equal(dataset_dict_with_test["validation"], dataset_dict["validation"])


def test_drop_splits(dataset_dict):
    dataset_dict_dropped = dataset_dict.drop_splits(["train", "validation"])
    assert set(dataset_dict_dropped) == {"test"}
    assert len(dataset_dict_dropped["test"]) == len(dataset_dict["test"])
    assert_doc_lists_equal(dataset_dict_dropped["test"], dataset_dict["test"])
