# dataset_filter.py

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict
from datasets import Sequence as DatasetSequence
from datasets import Value
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatasetExample(TypedDict, total=False):
    speech_duration: float
    snr: float
    c50: float
    stoi: float
    pesq: float
    sdr: float
    audio: np.ndarray | Sequence[float]  # More specific type for audio data


@dataclass(frozen=True)
class FilterConditions:
    speech_duration: float = 0.0
    snr: float = 15.0
    c50: float = 15.0
    stoi: float = 0.75
    pesq: float = 2.0
    sdr: float = 0.0


class DatasetFilterError(Exception):
    """Custom exception for DatasetFilter errors."""


class DatasetFilter:
    def __init__(self, filter_conditions: FilterConditions = FilterConditions()) -> None:
        self.filter_conditions = filter_conditions

    def process(self, dataset_dict: DatasetDict) -> DatasetDict:
        try:
            logger.info("Filtering dataset")
            filtered_dataset_dict = DatasetDict()
            for split, dataset in dataset_dict.items():
                if not self.validate_dataset(dataset):
                    raise DatasetFilterError(f"Invalid dataset for split: {split}")
                logger.info(f"Filtering {split} split")
                filtered_dataset = self._filter_dataset(dataset)
                filtered_dataset = self.apply_advanced_filtering(filtered_dataset)
                filtered_dataset_dict[split] = filtered_dataset
                logger.info(f"Filtered {split} split: {len(filtered_dataset)} examples")
                self.visualize_filtering_results(dataset, filtered_dataset)

            self._log_filtered_dataset_info(filtered_dataset_dict)
            return filtered_dataset_dict
        except Exception as e:
            logger.error(f"Error filtering dataset: {str(e)}")
            raise DatasetFilterError(f"Failed to filter dataset: {str(e)}") from e

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        filter_conditions = self._create_filter_function()
        filtered_dataset = dataset.filter(filter_conditions)
        self._log_dataset_statistics(dataset, filtered_dataset)
        return filtered_dataset

    def _create_filter_function(self) -> Callable[[DatasetExample], bool]:
        def filter_function(example: DatasetExample) -> bool:
            return all(
                self._safe_compare(example.get(key), value)
                for key, value in vars(self.filter_conditions).items()
            )

        return filter_function

    @staticmethod
    def _safe_compare(value: float | None, threshold: float) -> bool:
        return value is not None and value > threshold

    def _log_filtered_dataset_info(self, dataset_dict: DatasetDict) -> None:
        logger.info("Filtered dataset information:")
        for split, dataset in dataset_dict.items():
            logger.info(f"{split.capitalize()} set size: {len(dataset)}")
            if len(dataset) > 0:
                logger.info(f"Sample from filtered {split} set:")
                sample = dataset[0]
                for key, value in sample.items():
                    if key != "audio":  # Skip logging audio data
                        logger.info(f"  {key}: {value}")
                logger.info("-" * 50)

    def _log_dataset_statistics(self, original_dataset: Dataset, filtered_dataset: Dataset) -> None:
        logger.info(f"Original dataset size: {len(original_dataset)}")
        logger.info(f"Filtered dataset size: {len(filtered_dataset)}")
        logger.info(f"Columns in the dataset: {filtered_dataset.column_names}")

        logger.info("Samples meeting each condition:")
        for column, threshold in vars(self.filter_conditions).items():
            original_count = self._safe_count_condition(
                original_dataset, lambda x: self._safe_compare(x.get(column), threshold)
            )
            filtered_count = self._safe_count_condition(
                filtered_dataset, lambda x: self._safe_compare(x.get(column), threshold)
            )
            logger.info(f"{column} > {threshold}: {filtered_count}/{original_count}")

    def _safe_count_condition(
        self,
        dataset: Dataset,
        condition: Callable[[dict[str, float | np.ndarray | Sequence[float]]], bool],
    ) -> int:
        return sum(1 for x in dataset if condition(x))

    @staticmethod
    def get_missing_columns(dataset: Dataset, required_columns: list[str]) -> list[str]:
        return [col for col in required_columns if col not in dataset.column_names]

    def validate_dataset(self, dataset: Dataset) -> bool:
        required_columns = [field.name for field in fields(FilterConditions)]
        missing_columns = self.get_missing_columns(dataset, required_columns)

        if missing_columns:
            logger.warning(f"Missing columns in dataset: {', '.join(missing_columns)}")
            return False

        invalid_columns = self._check_column_types(dataset, required_columns)
        if invalid_columns:
            logger.warning(f"Invalid column types: {', '.join(invalid_columns)}")
            return False

        return True

    def _check_column_types(self, dataset: Dataset, columns: list[str]) -> list[str]:
        invalid_columns = []
        for column in columns:
            if column in dataset.features:
                if not isinstance(dataset.features[column], (Value, DatasetSequence)):
                    invalid_columns.append(column)
            else:
                invalid_columns.append(column)
        return invalid_columns

    @staticmethod
    def get_column_statistics(dataset: Dataset, column: str) -> dict[str, float | int]:
        if column not in dataset.features:
            return {}

        values = [example[column] for example in dataset if column in example]
        if not values:
            return {}

        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
            "most_common": Counter(values).most_common(1)[0][0],
        }

    def apply_advanced_filtering(self, dataset: Dataset) -> Dataset:
        """Apply advanced filtering techniques like outlier removal."""
        filtered_dataset = dataset
        for column in vars(self.filter_conditions).keys():
            if column in dataset.features:
                values = [ex[column] for ex in dataset if column in ex]
                z_scores = stats.zscore(values)
                filtered_dataset = filtered_dataset.filter(
                    lambda example, idx: abs(z_scores[idx]) <= 3, with_indices=True
                )
        return filtered_dataset

    def visualize_filtering_results(
        self, original_dataset: Dataset, filtered_dataset: Dataset
    ) -> None:
        """Create histograms to visualize the effect of filtering on each condition."""
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle("Filtering Results")

        for idx, (column, threshold) in enumerate(vars(self.filter_conditions).items()):
            ax = axs[idx // 2, idx % 2]
            original_values = [ex[column] for ex in original_dataset if column in ex]
            filtered_values = [ex[column] for ex in filtered_dataset if column in ex]

            ax.hist(original_values, bins=50, alpha=0.5, label="Original")
            ax.hist(filtered_values, bins=50, alpha=0.5, label="Filtered")
            ax.axvline(threshold, color="r", linestyle="dashed", linewidth=2)
            ax.set_title(f"{column} (threshold: {threshold})")
            ax.legend()

        plt.tight_layout()
        plt.savefig("filtering_results.png")
        logger.info("Filtering results visualization saved as 'filtering_results.png'")
