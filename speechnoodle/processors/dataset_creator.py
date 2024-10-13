# dataset_creator.py

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, Features, IterableDataset, Value
from numpy.typing import NDArray

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AudioMetadata(TypedDict):
    text: str
    segment_id: int
    start: float
    end: float
    source_file: str


@dataclass(frozen=True)
class ProcessedAudio:
    segmented_audio: list[NDArray[np.float32]] = field(default_factory=list)
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if not isinstance(self.segmented_audio, list):
            raise ValueError("segmented_audio must be a list")
        if not isinstance(self.metadata, pd.DataFrame):
            raise ValueError("metadata must be a pandas DataFrame")
        if len(self.segmented_audio) != len(self.metadata):
            raise ValueError("Length of segmented_audio must match number of rows in metadata")
        if not all(isinstance(segment, np.ndarray) for segment in self.segmented_audio):
            raise ValueError("All elements in segmented_audio must be numpy arrays")


class DatasetCreatorError(Exception):
    """Custom exception for DatasetCreator errors."""


class DatasetCreator:
    def __init__(self, sampling_rate: int = 16000) -> None:
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        self.sampling_rate = sampling_rate
        self.features = Features(
            {
                "audio": Audio(sampling_rate=self.sampling_rate),
                "text": Value("string"),
                "segment_id": Value("int64"),
                "start": Value("float32"),
                "end": Value("float32"),
                "source_file": Value("string"),
            }
        )

    def process(self, processed_audio_list: list[ProcessedAudio]) -> Dataset:
        if not processed_audio_list:
            raise DatasetCreatorError("Empty processed audio list")

        logger.info("Creating dataset from processed audio")
        combined_metadata = self._combine_metadata(processed_audio_list)
        dataset = self._create_dataset(combined_metadata)
        dataset = self._add_audio_column(dataset, processed_audio_list)
        self._log_dataset_info(dataset)
        return dataset

    def _combine_metadata(self, processed_audio_list: list[ProcessedAudio]) -> pd.DataFrame:
        metadata_list: list[pd.DataFrame] = []
        for pa in processed_audio_list:
            if not isinstance(pa, ProcessedAudio):
                raise DatasetCreatorError(f"Invalid object in list: {type(pa)}")
            metadata_list.append(pa.metadata)
        return pd.concat(metadata_list, ignore_index=True)

    def _create_dataset(self, metadata: pd.DataFrame) -> Dataset:
        try:
            dataset = Dataset.from_pandas(metadata, features=self.features)
            return dataset
        except Exception as e:
            raise DatasetCreatorError(f"Failed to create dataset from metadata: {str(e)}") from e

    def _add_audio_column(
        self, dataset: Dataset, processed_audio_list: list[ProcessedAudio]
    ) -> Dataset:
        audio_data: list[NDArray[np.float32]] = [
            segment for audio in processed_audio_list for segment in audio.segmented_audio
        ]
        if len(audio_data) != len(dataset):
            raise DatasetCreatorError("Mismatch between audio segments and dataset length")

        try:
            dataset = dataset.add_column("audio", audio_data)
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
            return dataset
        except Exception as e:
            raise DatasetCreatorError(f"Failed to add audio column: {str(e)}") from e

    def _log_dataset_info(self, dataset: Dataset) -> None:
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Columns in the dataset: {dataset.column_names}")
        if dataset:
            logger.info("Sample from the dataset:")
            sample = dataset[0]
            for key, value in sample.items():
                if key != "audio":  # Skip logging audio data
                    logger.info(f"  {key}: {value}")
            logger.info("-" * 50)

    def validate_metadata(self, metadata: pd.DataFrame) -> bool:
        required_columns: dict[str, Any] = {
            "text": str,
            "segment_id": int,
            "start": float,
            "end": float,
            "source_file": str,
        }
        for col, dtype in required_columns.items():
            if col not in metadata.columns:
                logger.error(f"Missing required column: {col}")
                return False
            if not pd.api.types.is_dtype_equal(metadata[col].dtype, dtype):
                logger.error(
                    f"Column {col} has incorrect dtype. Expected {dtype}, got {metadata[col].dtype}"
                )
                return False
        return True

    def get_dataset_statistics(self, dataset: Dataset) -> dict[str, Any]:
        stats: dict[str, Any] = {"total_samples": len(dataset), "column_stats": {}}
        for column in dataset.column_names:
            if column != "audio":
                column_data = dataset[column]
                column_stats = self._get_column_statistics(column_data)
                stats["column_stats"][column] = column_stats
        return stats

    @staticmethod
    def _get_column_statistics(column_data: Iterable[Any]) -> dict[str, Any]:
        data = list(column_data)
        if all(isinstance(x, (int, float)) for x in data):
            return {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
            }
        if all(isinstance(x, str) for x in data):
            return {"unique_values": len(set(data)), "most_common": max(set(data), key=data.count)}
        return {"type": str(type(data[0]))}

    def check_audio_consistency(self, dataset: Dataset) -> bool:
        try:
            audio_lengths = set()
            for audio in dataset["audio"]:
                audio_lengths.add(len(audio["array"]))
                if len(audio_lengths) > 1:
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking audio consistency: {str(e)}")
            return False

    def save_dataset(self, dataset: Dataset, path: str) -> None:
        try:
            dataset.save_to_disk(path)
            logger.info(f"Dataset saved to {path}")
        except Exception as e:
            raise DatasetCreatorError(f"Failed to save dataset: {str(e)}") from e

    def create_iterable_dataset(
        self, processed_audio_list: list[ProcessedAudio]
    ) -> IterableDataset:
        def gen_func() -> Iterable[AudioMetadata]:
            for pa in processed_audio_list:
                for audio, meta in zip(pa.segmented_audio, pa.metadata.itertuples()):
                    yield {
                        "audio": audio,
                        "text": meta.text,
                        "segment_id": meta.segment_id,
                        "start": meta.start,
                        "end": meta.end,
                        "source_file": meta.source_file,
                    }

        return IterableDataset.from_generator(gen_func, features=self.features)
