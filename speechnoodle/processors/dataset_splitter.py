# dataset_splitter.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from datasets import Dataset, DatasetDict, Features
from datasets import Sequence as DatasetSequence
from datasets import Value

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitRatios:
    train: float = field(default=0.8, metadata={"min": 0.0, "max": 1.0})
    validation: float = field(default=0.1, metadata={"min": 0.0, "max": 1.0})
    test: float = field(default=0.1, metadata={"min": 0.0, "max": 1.0})

    def __post_init__(self) -> None:
        for field_name, field_value in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            min_value = field_value.metadata.get("min", float("-inf"))
            max_value = field_value.metadata.get("max", float("inf"))
            if not min_value <= value <= max_value:
                raise ValueError(f"{field_name} must be between {min_value} and {max_value}")

        total = self.train + self.validation + self.test
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError("Split ratios must sum to 1")


class DatasetSplitterError(Exception):
    """Custom exception for DatasetSplitter errors."""


class DatasetSplitter:
    def __init__(self, split_ratios: SplitRatios = SplitRatios()) -> None:
        self.split_ratios = split_ratios

    def process(self, dataset: Dataset) -> DatasetDict:
        if not self.validate_dataset(dataset):
            raise DatasetSplitterError("Invalid dataset")

        try:
            logger.info("Splitting dataset into train, validation, and test sets")
            dataset_dict = self._split_dataset(dataset)
            self._log_split_info(dataset_dict)
            return dataset_dict
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise DatasetSplitterError(f"Failed to split dataset: {str(e)}") from e

    def _split_dataset(self, dataset: Dataset) -> DatasetDict:
        total_size = len(dataset)
        train_size, val_size, test_size = self._calculate_split_sizes(total_size)

        # Single split operation
        splits = dataset.train_test_split(
            train_size=train_size, test_size=val_size + test_size, shuffle=True, seed=42
        )
        train_dataset = splits["train"]
        temp_test = splits["test"]

        # Split the remaining data into validation and test
        val_test_split = temp_test.train_test_split(
            train_size=val_size, test_size=test_size, shuffle=True, seed=42
        )

        return DatasetDict(
            {
                "train": train_dataset,
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            }
        )

    def _calculate_split_sizes(self, total_size: int) -> tuple[int, int, int]:
        train_size = int(total_size * self.split_ratios.train)
        val_size = int(total_size * self.split_ratios.validation)
        test_size = total_size - train_size - val_size
        return train_size, val_size, test_size

    def _log_split_info(self, dataset_dict: DatasetDict) -> None:
        logger.info("Dataset split information:")
        for split, dataset in dataset_dict.items():
            logger.info(f"{split.capitalize()} set size: {len(dataset)}")
            if dataset:
                logger.info(f"Sample from {split} set:")
                sample = dataset[0]
                for key, value in sample.items():
                    if key != "audio":  # Skip logging audio data
                        logger.info(f"  {key}: {value}")
                logger.info("-" * 50)

    @staticmethod
    def validate_dataset(dataset: Dataset) -> bool:
        if not isinstance(dataset, Dataset):
            logger.error("Input is not a valid Dataset object")
            return False

        if not dataset:
            logger.error("Dataset is empty")
            return False

        return True

    @staticmethod
    def get_column_types(dataset: Dataset) -> dict[str, str]:
        return {
            name: DatasetSplitter._get_feature_type(feature)
            for name, feature in dataset.features.items()
        }

    @staticmethod
    def _get_feature_type(feature: Features) -> str:
        if isinstance(feature, Value):
            return cast(str, feature.dtype)
        if isinstance(feature, DatasetSequence):
            return f"Sequence[{DatasetSplitter._get_feature_type(feature.feature)}]"
        return str(type(feature).__name__)

    def get_split_statistics(
        self, dataset_dict: DatasetDict
    ) -> dict[str, dict[str, int | dict[str, int]]]:
        stats = {}
        for split, dataset in dataset_dict.items():
            split_stats: dict[str, int | dict[str, int]] = {
                "size": len(dataset),
                "column_counts": {},
            }
            for column in dataset.column_names:
                if column != "audio":
                    column_counts = cast(dict[str, int], split_stats["column_counts"])
                    column_counts[column] = dataset.num_rows - dataset[column].count()
            stats[split] = split_stats
        return stats

    def get_column_statistics(self, dataset: Dataset, column: str) -> dict[str, float] | None:
        if column not in dataset.features:
            logger.warning(f"Column '{column}' not found in dataset")
            return None

        try:
            values = dataset[column]
            return {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            }
        except Exception as e:
            logger.error(f"Error calculating statistics for column '{column}': {str(e)}")
            return None
