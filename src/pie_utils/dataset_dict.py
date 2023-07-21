import json
import logging
import os
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    SupportsIndex,
    Type,
    TypeVar,
    Union,
)

import datasets
from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.core import Document

from pie_utils.hydra import resolve_target

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def get_split_type(
    dataset_split: Union[datasets.Dataset, datasets.IterableDataset]
) -> Union[Type[Dataset], Type[IterableDataset]]:
    if isinstance(dataset_split, datasets.Dataset):
        return Dataset
    elif isinstance(dataset_split, datasets.IterableDataset):
        return IterableDataset
    else:
        raise ValueError(
            f"dataset_split must be of type Dataset or IterableDataset, but is {type(dataset_split)}"
        )


class DatasetDict(datasets.DatasetDict):
    def __getitem__(self, k) -> Dataset:
        return super().__getitem__(k)

    @classmethod
    def load_dataset(cls, *args, **kwargs):
        return cls(datasets.load_dataset(*args, **kwargs))

    @classmethod
    def from_hf_dataset(
        cls, hf_dataset: datasets.DatasetDict, document_type: Union[str, Type[Document]]
    ) -> datasets.DatasetDict:
        doc_type = resolve_target(document_type)
        res = type(hf_dataset)(
            {
                k: get_split_type(v).from_hf_dataset(v, document_type=doc_type)
                for k, v in hf_dataset.items()
            }
        )
        return res

    def map(
        self,
        function: Optional[Union[Callable, str]] = None,
        result_type: Optional[Union[str, Type[D]]] = None,
        **kwargs,
    ) -> "DatasetDict":
        if function is not None:
            func = resolve_target(function)
        else:

            def identity(x):
                return x

            func = identity
        map_kwargs = dict(function=func, fn_kwargs=kwargs)
        if result_type is not None:
            map_kwargs["result_document_type"] = resolve_target(result_type)
        result = type(self)({k: v.map(**map_kwargs) for k, v in self.items()})

        return result

    def select(
        self,
        split: str,
        start: Optional[SupportsIndex] = None,
        stop: Optional[SupportsIndex] = None,
        step: Optional[SupportsIndex] = None,
        **kwargs,
    ) -> "DatasetDict":
        if stop is not None:
            range_args = [stop]
            if start is not None:
                range_args = [start] + range_args
            if step is not None:
                range_args = range_args + [step]
            kwargs["indices"] = range(*range_args)
        pie_split = self[split]
        if "indices" in kwargs:
            self[split] = Dataset.from_hf_dataset(
                dataset=pie_split.select(**kwargs), document_type=self[split].document_type
            )
        else:
            if len(kwargs) > 0:
                logger.warning(
                    f"arguments for dataset.select() available, but they do not contain 'indices' which is required, "
                    f"so we do not call select. provided arguments: \n{json.dumps(kwargs, indent=2)}"
                )
        return self

    def rename_splits(
        self,
        mapping: Optional[Dict[str, str]] = None,
        keep_other_splits: bool = True,
    ) -> "DatasetDict":
        if mapping is None:
            mapping = {}
        result = type(self)(
            {
                mapping.get(name, name): data
                for name, data in self.items()
                if name in mapping or keep_other_splits
            }
        )
        return result

    def add_test_split(
        self,
        source_split: str = "train",
        target_split: str = "test",
        **kwargs,
    ) -> "DatasetDict":
        split_result_hf = self[source_split].train_test_split(**kwargs)
        split_result = type(self)(
            {
                name: Dataset.from_hf_dataset(ds, document_type=self[source_split].document_type)
                for name, ds in split_result_hf.items()
            }
        )
        res = type(self)(self)
        res[source_split] = split_result["train"]
        res[target_split] = split_result["test"]
        split_sizes = {k: len(v) for k, v in res.items()}
        logger.info(f"dataset size after adding the split: {split_sizes}")
        return res

    def drop_splits(self, split_names: List[str]) -> "DatasetDict":
        result = type(self)({name: ds for name, ds in self.items() if name not in split_names})
        return result

    def concat_splits(self, splits: List[str], target: str) -> "DatasetDict":
        result = type(self)({name: ds for name, ds in self.items() if name not in splits})
        splits_to_concat = [self[name] for name in splits]
        # ensure that the document types are the same
        document_type = None
        dataset_type = None
        for split in splits_to_concat:
            if document_type is not None and split.document_type != document_type:
                raise ValueError(
                    f"document types of splits to concatenate differ: {document_type} != {split.document_type}"
                )
            document_type = split.document_type
            if dataset_type is not None and type(split) != dataset_type:
                raise ValueError(
                    f"dataset types of splits to concatenate differ: {dataset_type} != {type(split)}"
                )
            dataset_type = type(split)
        if document_type is None or dataset_type is None:
            raise ValueError("please provide at least one split to concatenate")
        concatenated = datasets.concatenate_datasets(splits_to_concat)
        result[target] = dataset_type.from_hf_dataset(concatenated, document_type=document_type)
        split_sizes = {k: len(v) for k, v in result.items()}
        logger.info(f"dataset size after concatenating splits: {split_sizes}")
        return result

    def filter(
        self,
        split: str,
        function: Optional[Union[Callable, str]] = None,
        result_split_name: Optional[str] = None,
        **kwargs,
    ) -> "DatasetDict":
        if function is not None:
            # create a shallow copy to not modify the input
            result = type(self)(self)
            if isinstance(function, str):
                function = resolve_target(function)

            pie_split = result[split]
            if isinstance(pie_split, Dataset):
                hf_split = datasets.Dataset(**Dataset.get_base_kwargs(pie_split))
            elif isinstance(pie_split, IterableDataset):
                hf_split = datasets.IterableDataset(**IterableDataset.get_base_kwargs(pie_split))
            else:
                raise Exception(f"dataset split has unknown type: {type(pie_split)}")
            hf_split_filtered = hf_split.filter(function=function, **kwargs)
            target_split_name = result_split_name or split
            result[target_split_name] = type(pie_split).from_hf_dataset(
                dataset=hf_split_filtered, document_type=pie_split.document_type
            )
            logger.info(
                f"filtered split [{target_split_name}] has {len(result[target_split_name])} entries"
            )
            return result
        else:
            return self

    def move_to_new_split(
        self,
        ids: List[str],
        source_split: str = "train",
        target_split: str = "test",
    ) -> "DatasetDict":
        ids_set = set(ids)
        dataset_without_ids = self.filter(
            dataset=self,
            split=source_split,
            function=lambda ex: ex["id"] not in ids_set,
        )
        dataset_with_only_ids = self.filter(
            dataset=self,
            split=source_split,
            function=lambda ex: ex["id"] in ids_set,
        )
        dataset_without_ids[target_split] = dataset_with_only_ids[source_split]

        split_sizes = {k: len(v) for k, v in dataset_without_ids.items()}
        logger.info(f"dataset size after moving to new split: {split_sizes}")
        return dataset_without_ids

    def cast_document_type(
        self, new_document_type: Union[Type[Document], str], **kwargs
    ) -> "DatasetDict":
        new_type = resolve_target(new_document_type)

        result = type(self)(
            {
                name: ds.cast_document_type(new_document_type=new_type, **kwargs)
                for name, ds in self.items()
            }
        )
        return result

    def to_json(self, path: str, **kwargs):
        for split, dataset in self.items():
            split_path = os.path.join(path, split)
            logger.info(f'serialize documents to "{split_path}" ...')
            os.makedirs(split_path, exist_ok=True)
            file_name = os.path.join(split_path, "documents.jsonl")
            with open(file_name, "w") as f:
                for doc in dataset:
                    f.write(json.dumps(doc.asdict(), **kwargs) + "\n")

    @classmethod
    def from_json(
        cls,
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]],
        document_type: Union[Type[Document], str],
        **kwargs,
    ) -> "DatasetDict":
        return cls.from_hf_dataset(
            datasets.load_dataset("json", data_files=data_files, **kwargs),
            document_type=document_type,
        )
