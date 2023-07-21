import logging
from dataclasses import dataclass

import datasets
import pytest
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument

from pie_utils import DatasetDict
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


@pytest.fixture(scope="module")
def dataset_dict():
    @dataclass
    class DocumentWithEntitiesAndRelations(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    # get all folders in DATA_PATH that contain a documents.jsonl file
    data_files = {
        folder.name: str(folder / "documents.jsonl")
        for folder in DATA_PATH.iterdir()
        if (folder / "documents.jsonl").exists()
    }
    return DatasetDict.from_json(
        data_files=data_files, document_type=DocumentWithEntitiesAndRelations
    )


def test_dataset_dict_from_json(dataset_dict):
    assert set(dataset_dict) == {"train", "test", "validation"}
