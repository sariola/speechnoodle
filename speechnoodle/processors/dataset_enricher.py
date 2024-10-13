from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from datasets import Audio, Dataset, DatasetDict
from phonemizer.backend import EspeakBackend
from torchaudio.pipelines import SQUIM_OBJECTIVE

from speechnoodle.processors.enrichment_functions import (
    add_phonemes,
    add_speech_duration,
    add_squim_quality_estimation,
)
from speechnoodle.processors.pipeline_step import PipelineStep
from speechnoodle.utils.exceptions import DatasetEnricherError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnrichmentFunction(Protocol):
    """Protocol for enrichment functions."""

    def __call__(
        self, batch: dict[str, list[np.ndarray] | list[int] | list[str]]
    ) -> dict[str, list[float | str]]:
        """Apply enrichment to a batch of data.

        Args:
            batch (Dict[str, Union[List[np.ndarray], List[int], List[str]]]): A batch of examples from the dataset.

        Returns:
            Dict[str, List[Union[float, str]]]: A dictionary containing the enriched data.

        Raises:
            Exception: If there's an error during the enrichment process.
        """
        ...


@dataclass(frozen=True)
class DatasetEnricherConfig:
    """Configuration for DatasetEnricher.

    Attributes:
        enrichment_functions (List[EnrichmentFunction]): List of enrichment functions to apply.
        cpu_num_workers (int): Number of CPU workers for non-GPU operations.
        batch_size (int): Batch size for GPU operations.
        audio_column_name (str): Name of the audio column in the dataset.
        text_column_name (str): Name of the text column in the dataset.
        rename_columns (bool): Whether to rename audio and text columns to 'audio' and 'text'.
        device (str): Device to use for computations ('cuda' or 'cpu').
    """

    enrichment_functions: list[EnrichmentFunction] = field(
        default_factory=lambda: [add_speech_duration, add_squim_quality_estimation, add_phonemes]
    )
    cpu_num_workers: int = 1
    batch_size: int = 32
    audio_column_name: str = "audio"
    text_column_name: str = "text"
    rename_columns: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DatasetEnricherResult:
    """Result of DatasetEnricher processing.

    Attributes:
        enriched_dataset (DatasetDict): The enriched dataset.
        stats (Dict[str, Dict[str, float]]): Statistics for each split and feature.
    """

    enriched_dataset: DatasetDict
    stats: dict[str, dict[str, float]] = field(default_factory=dict)


class DatasetEnricher(PipelineStep):
    """Enriches a dataset with additional features and metadata.

    This class processes a dataset by adding various audio-related features based on
    the provided enrichment functions.

    Attributes:
        config (DatasetEnricherConfig): Configuration for the enrichment process.
        squim_model (SQUIM_OBJECTIVE): SQUIM model for quality estimation.
        phonemizer (EspeakBackend): Phonemizer for text-to-phoneme conversion.
    """

    def __init__(self, config: DatasetEnricherConfig = DatasetEnricherConfig()) -> None:
        """Initialize the DatasetEnricher.

        Args:
            config (DatasetEnricherConfig, optional): Configuration for the enrichment process.
                Defaults to DatasetEnricherConfig().

        Raises:
            DatasetEnricherError: If there's an error initializing the models.
        """
        self.config: DatasetEnricherConfig = config
        try:
            self.squim_model: SQUIM_OBJECTIVE = SQUIM_OBJECTIVE.from_pretrained().to(
                self.config.device
            )
            self.phonemizer: EspeakBackend = EspeakBackend(language="fi", preserve_punctuation=True)
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise DatasetEnricherError(f"Failed to initialize models: {str(e)}") from e

    def process(self, dataset: Dataset | DatasetDict) -> DatasetEnricherResult:
        """Process the input dataset and enrich it with additional features.

        Args:
            dataset (Union[Dataset, DatasetDict]): The input dataset to enrich.

        Returns:
            DatasetEnricherResult: The enriched dataset and processing statistics.

        Raises:
            DatasetEnricherError: If there's an error during the enrichment process.
        """
        try:
            logger.info("Starting dataset enrichment process")

            dataset_dict = self._ensure_dataset_dict(dataset)
            self._validate_dataset(dataset_dict)

            if self.config.rename_columns:
                dataset_dict = self._rename_columns(dataset_dict)

            enriched_dataset = self._enrich_dataset(dataset_dict)
            enriched_dataset = self._filter_none_values(enriched_dataset)
            stats = self._calculate_stats(enriched_dataset)

            logger.info("Dataset enrichment process completed successfully")
            return DatasetEnricherResult(enriched_dataset=enriched_dataset, stats=stats)

        except Exception as e:
            logger.error(f"Error in dataset enrichment process: {str(e)}")
            raise DatasetEnricherError(f"Failed to enrich dataset: {str(e)}") from e

    def _ensure_dataset_dict(self, dataset: Dataset | DatasetDict) -> DatasetDict:
        """Ensure the input is a DatasetDict.

        Args:
            dataset (Union[Dataset, DatasetDict]): The input dataset.

        Returns:
            DatasetDict: The dataset as a DatasetDict.

        Raises:
            DatasetEnricherError: If the input is neither Dataset nor DatasetDict.
        """
        if isinstance(dataset, Dataset):
            return DatasetDict({"train": dataset})
        elif isinstance(dataset, DatasetDict):
            return dataset
        else:
            raise DatasetEnricherError(f"Unexpected dataset type: {type(dataset)}")

    def _validate_dataset(self, dataset: DatasetDict) -> None:
        """Validate the input dataset.

        Args:
            dataset (DatasetDict): The input dataset to validate.

        Raises:
            DatasetEnricherError: If the dataset is invalid or missing required columns.
        """
        required_columns = [self.config.audio_column_name, self.config.text_column_name]
        for split, ds in dataset.items():
            if not isinstance(ds, Dataset):
                raise DatasetEnricherError(f"Invalid dataset type for split '{split}': {type(ds)}")
            missing_columns = [col for col in required_columns if col not in ds.column_names]
            if missing_columns:
                raise DatasetEnricherError(
                    f"Missing required columns in split '{split}': {', '.join(missing_columns)}"
                )
            if not isinstance(ds.features[self.config.audio_column_name], Audio):
                raise DatasetEnricherError(
                    f"Invalid 'audio' feature type in split '{split}': {type(ds.features[self.config.audio_column_name])}"
                )

    def _rename_columns(self, dataset: DatasetDict) -> DatasetDict:
        """Rename audio and text columns to 'audio' and 'text'.

        Args:
            dataset (DatasetDict): The input dataset.

        Returns:
            DatasetDict: The dataset with renamed columns.

        Raises:
            DatasetEnricherError: If there's an error renaming the columns.
        """
        try:
            return DatasetDict(
                {
                    split: ds.rename_columns(
                        {
                            self.config.audio_column_name: "audio",
                            self.config.text_column_name: "text",
                        }
                    )
                    for split, ds in dataset.items()
                }
            )
        except Exception as e:
            logger.error(f"Error renaming columns: {str(e)}")
            raise DatasetEnricherError(f"Failed to rename columns: {str(e)}") from e

    def _enrich_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Apply enrichment processes to the dataset.

        Args:
            dataset (DatasetDict): The input dataset to enrich.

        Returns:
            DatasetDict: The enriched dataset.

        Raises:
            DatasetEnricherError: If there's an error during the enrichment process.
        """
        try:
            for func in self.config.enrichment_functions:
                dataset = dataset.map(
                    lambda x: self._safe_process(func, x)[0],
                    batched=True,
                    batch_size=self.config.batch_size,
                    num_proc=self.config.cpu_num_workers,
                    desc=f"Applying {func.__name__}",
                )

            return dataset
        except Exception as e:
            logger.error(f"Error enriching dataset: {str(e)}")
            raise DatasetEnricherError(f"Failed to enrich dataset: {str(e)}") from e

    def _safe_process(
        self,
        func: EnrichmentFunction,
        example: dict[str, list[np.ndarray] | list[int] | list[str]],
    ) -> tuple[dict[str, list[float | str]] | None, bool]:
        """Safely process a single example, catching and logging any exceptions.

        Args:
            func (EnrichmentFunction): The enrichment function to apply to the example.
            example (Dict[str, Union[List[np.ndarray], List[int], List[str]]]): The example to process.

        Returns:
            Tuple[Optional[Dict[str, List[Union[float, str]]]], bool]: The processed example (or None if an error occurred) and a success flag.
        """
        try:
            result = func(example)
            return result, True
        except Exception as e:
            logger.warning(f"Error processing sample with {func.__name__}: {str(e)}")
            return None, False

    def _filter_none_values(self, dataset: DatasetDict) -> DatasetDict:
        """Filter out samples with None values from the dataset.

        Args:
            dataset (DatasetDict): The input dataset to filter.

        Returns:
            DatasetDict: The filtered dataset.

        Raises:
            DatasetEnricherError: If there's an error during the filtering process.
        """
        try:
            filtered_dataset = DatasetDict()
            for split, ds in dataset.items():
                original_size = len(ds)
                filtered_ds = ds.filter(
                    lambda x: all(x[col] is not None for col in ds.column_names)
                )
                filtered_size = len(filtered_ds)
                logger.info(
                    f"Filtered {original_size - filtered_size} samples with None values from split {split}"
                )
                filtered_dataset[split] = filtered_ds
            return filtered_dataset
        except Exception as e:
            logger.error(f"Error filtering None values: {str(e)}")
            raise DatasetEnricherError(f"Failed to filter None values: {str(e)}") from e

    def _calculate_stats(self, dataset: DatasetDict) -> dict[str, dict[str, float]]:
        """Calculate statistics for the enriched dataset.

        Args:
            dataset (DatasetDict): The enriched dataset.

        Returns:
            Dict[str, Dict[str, float]]: Statistics for each split and feature.

        Raises:
            DatasetEnricherError: If there's an error calculating dataset statistics.
        """
        try:
            stats: dict[str, dict[str, float]] = {}
            for split, ds in dataset.items():
                split_stats: dict[str, float] = {}
                for column in ds.column_names:
                    if isinstance(ds.features[column], Audio) and ds.features[column].dtype in [
                        "float32",
                        "float64",
                    ]:
                        values: np.ndarray = np.array(ds[column])
                        split_stats[f"{column}_mean"] = float(np.mean(values))
                        split_stats[f"{column}_std"] = float(np.std(values))
                        split_stats[f"{column}_min"] = float(np.min(values))
                        split_stats[f"{column}_max"] = float(np.max(values))
                stats[split] = split_stats
            return stats
        except Exception as e:
            logger.error(f"Error calculating dataset statistics: {str(e)}")
            raise DatasetEnricherError(f"Failed to calculate dataset statistics: {str(e)}") from e

    def _log_enrichment_summary(
        self, dataset: DatasetDict, stats: dict[str, dict[str, float]]
    ) -> None:
        """Log a summary of the enrichment process.

        Args:
            dataset (DatasetDict): The enriched dataset.
            stats (Dict[str, Dict[str, float]]): Statistics for each split and feature.

        Raises:
            DatasetEnricherError: If there's an error logging the enrichment summary.
        """
        try:
            for split, ds in dataset.items():
                logger.info(f"Enriched split '{split}':")
                logger.info(f"  - Number of samples: {len(ds)}")
                logger.info(f"  - Columns: {', '.join(ds.column_names)}")
                if split in stats:
                    for stat_name, stat_value in stats[split].items():
                        logger.info(f"  - {stat_name}: {stat_value:.4f}")
        except Exception as e:
            logger.error(f"Error logging enrichment summary: {str(e)}")
            raise DatasetEnricherError(f"Failed to log enrichment summary: {str(e)}") from e

    @staticmethod
    def _validate_audio_sampling_rate(
        audio: dict[str, np.ndarray | int], expected_rate: int = 16000
    ) -> None:
        """Validate the sampling rate of an audio sample.

        Args:
            audio (Dict[str, Union[np.ndarray, int]]): The audio sample to validate.
            expected_rate (int, optional): The expected sampling rate. Defaults to 16000.

        Raises:
            ValueError: If the sampling rate does not match the expected rate.
        """
        if audio["sampling_rate"] != expected_rate:
            raise ValueError(
                f"Expected sampling rate of {expected_rate}, but got {audio['sampling_rate']}"
            )

    @staticmethod
    def _convert_audio_to_tensor(audio: np.ndarray, device: str) -> torch.Tensor:
        """Convert an audio numpy array to a PyTorch tensor.

        Args:
            audio (np.ndarray): The audio array to convert.
            device (str): The device to move the tensor to.

        Returns:
            torch.Tensor: The audio as a PyTorch tensor.
        """
        return torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
