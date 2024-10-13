"""Audio Dataset Creator module for processing and creating audio datasets.

This module provides the main AudioDatasetCreator class and related components
for creating, processing, and filtering audio datasets.
"""

from __future__ import annotations

import gc
import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import torch
from datasets import DatasetDict
from tqdm import tqdm

from .processors.audio_processor import (
    AudioFile,
    AudioProcessor,
    AudioProcessorConfig,
    ProcessedAudio,
)
from .processors.dataset_creator import DatasetCreator, DatasetCreatorConfig
from .processors.dataset_enricher import DatasetEnricher, DatasetEnricherConfig
from .processors.dataset_filter import DatasetFilter, FilterConditions
from .processors.dataset_splitter import DatasetSplitter, SplitRatios

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg")


class AudioDatasetCreatorError(Exception):
    """Custom exception for AudioDatasetCreator errors."""


class ProcessorResult(TypedDict):
    """Represents the result of a processor operation.

    :param success: Indicates if the operation was successful.
    :param data: The processed data or None if unsuccessful.
    :param error: Error message if unsuccessful, None otherwise.
    """

    success: bool
    data: list[AudioFile] | list[ProcessedAudio] | DatasetDict | None
    error: str | None


@dataclass(frozen=True)
class AudioDatasetCreatorConfig:
    """Configuration for AudioDatasetCreator.

    :param split_ratios: Ratios for splitting the dataset.
    :param filter_conditions: Conditions for filtering the dataset.
    :param enricher_config: Configuration for the DatasetEnricher.
    :param audio_processor_config: Configuration for the AudioProcessor.
    :param dataset_creator_config: Configuration for the DatasetCreator.
    """

    split_ratios: SplitRatios = field(
        default_factory=lambda: SplitRatios(train=0.8, validation=0.1, test=0.1)
    )
    filter_conditions: FilterConditions = field(
        default_factory=lambda: FilterConditions(
            speech_duration=0, snr=15, c50=15, stoi=0.75, pesq=2.0, sdr=0
        )
    )
    enricher_config: DatasetEnricherConfig = field(default_factory=DatasetEnricherConfig)
    audio_processor_config: AudioProcessorConfig = field(default_factory=AudioProcessorConfig)
    dataset_creator_config: DatasetCreatorConfig = field(default_factory=DatasetCreatorConfig)


class PipelineStep(ABC):
    """Abstract base class for pipeline steps.

    All pipeline steps must inherit from this class and implement the process method.
    """

    @abstractmethod
    def process(
        self, input_data: list[AudioFile] | list[ProcessedAudio] | DatasetDict
    ) -> list[AudioFile] | list[ProcessedAudio] | DatasetDict:
        """Process the input data.

        :param input_data: The input data to process.
        :return: The processed data.
        """
        pass


class AudioDatasetCreator:
    """Main class for creating and processing audio datasets.

    This class orchestrates the pipeline of processing steps to create
    an audio dataset from raw audio files.
    """

    def __init__(
        self,
        config: AudioDatasetCreatorConfig | None = None,
        pipeline: list[PipelineStep] | None = None,
    ) -> None:
        """Initialize the AudioDatasetCreator.

        :param config: Configuration for the dataset creator. If None, uses default config.
        :param pipeline: Custom pipeline steps. If None, uses default pipeline.
        """
        self.config = config if config is not None else AudioDatasetCreatorConfig()

        if pipeline is None:
            self.pipeline: list[PipelineStep] = [
                AudioProcessor(config=self.config.audio_processor_config),
                DatasetCreator(config=self.config.dataset_creator_config),
                DatasetEnricher(self.config.enricher_config),
                DatasetSplitter(self.config.split_ratios),
                DatasetFilter(self.config.filter_conditions),
            ]
        else:
            self.pipeline = pipeline

    async def create_dataset(
        self, source_dir: str, dest_dir: str, num_dirs: int | None = None
    ) -> None:
        """Create a dataset from audio files in the source directory.

        :param source_dir: Path to the source directory containing audio files.
        :param dest_dir: Path to the destination directory for the processed dataset.
        :param num_dirs: Number of subdirectories to process. If None, process all.
        :raises AudioDatasetCreatorError: If the source directory doesn't exist.
        """
        source_dir_path = Path(source_dir)
        dest_dir_path = Path(dest_dir)

        if not source_dir_path.is_dir():
            raise AudioDatasetCreatorError(f"Source directory does not exist: {source_dir_path}")

        if not dest_dir_path.exists():
            dest_dir_path.mkdir(parents=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        all_dirs = [d for d in source_dir_path.iterdir() if d.is_dir()]
        total_dirs = len(all_dirs)
        processed_dirs = 0
        skipped_dirs = 0

        for dir_path in tqdm(
            all_dirs[:num_dirs] if num_dirs else all_dirs, desc="Processing directories"
        ):
            output_dir = dest_dir_path / f"final_output_{dir_path.name}"
            if await self._process_directory(dir_path, output_dir):
                processed_dirs += 1
            else:
                skipped_dirs += 1

            remaining = total_dirs - processed_dirs - skipped_dirs
            logger.info(
                f"Progress: {processed_dirs} processed, {skipped_dirs} skipped, {remaining} remaining"
            )

            if remaining > 0:
                time.sleep(5)  # Wait for 5 seconds before next directory

        logger.info(
            f"Finished processing. Total: {processed_dirs} processed, {skipped_dirs} skipped, {remaining} remaining"
        )

    async def _process_directory(self, input_dir: Path, output_dir: Path) -> bool:
        """Process a single directory of audio files.

        :param input_dir: Path to the input directory.
        :param output_dir: Path to the output directory.
        :return: True if processing was successful, False otherwise.
        """
        try:
            logger.info(f"Starting to process directory: {input_dir}")

            if output_dir.exists():
                logger.info(
                    f"Result for {input_dir.name} already exists in destination. Skipping processing."
                )
                return False

            temp_output = Path(f"temp_output_{input_dir.name}")
            if temp_output.exists():
                shutil.rmtree(temp_output)
                logger.info(f"Removed existing temporary output directory: {temp_output}")

            temp_output.mkdir(parents=True, exist_ok=True)

            audio_files = self._get_audio_files(input_dir)
            if not audio_files:
                logger.error("No valid audio files found in the input directory")
                return False

            result = await self._run_pipeline(audio_files)

            if result["success"] and isinstance(result["data"], DatasetDict):
                result["data"].save_to_disk(str(temp_output))
                logger.info(f"Dataset saved to temporary directory: {temp_output}")

                if not self._move_files(temp_output, output_dir):
                    logger.error("Failed to move files to final destination")
                    return False

                if not self._verify_transfer(temp_output, output_dir):
                    logger.error("Transfer verification failed")
                    return False

                shutil.rmtree(temp_output)
                logger.info(f"Removed temporary output directory: {temp_output}")

                logger.info(f"Successfully processed {input_dir.name}")
                return True
            else:
                logger.error(f"Pipeline execution failed: {result.get('error', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Unexpected error processing {input_dir.name}: {e}")
            return False

    def _get_audio_files(self, raw_audio_dir: Path) -> list[AudioFile]:
        """Get a list of AudioFile objects from the given directory.

        :param raw_audio_dir: Path to the directory containing audio files.
        :return: List of AudioFile objects.
        """
        audio_files: list[AudioFile] = []
        for file_path in tqdm(list(raw_audio_dir.iterdir()), desc="Reading audio files"):
            if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                try:
                    with open(file_path, "rb") as f:
                        audio_data = f.read()
                    audio_files.append(AudioFile(path=str(file_path), data=audio_data))
                except OSError as e:
                    logger.error(f"Error reading audio file {file_path}: {str(e)}")
        return audio_files

    async def _run_pipeline(self, initial_input: list[AudioFile]) -> ProcessorResult:
        """Run the processing pipeline on the input data.

        :param initial_input: List of AudioFile objects to process.
        :return: ProcessorResult containing the result of the pipeline execution.
        """
        result: list[AudioFile] | list[ProcessedAudio] | DatasetDict = initial_input
        for i, step in enumerate(self.pipeline):
            logger.info(
                f"Running pipeline step {i+1}/{len(self.pipeline)}: {step.__class__.__name__}"
            )
            try:
                result = step.process(result)
                if result is None:
                    return ProcessorResult(
                        success=False, data=None, error="Pipeline step returned None"
                    )
                logger.info(f"Pipeline step {step.__class__.__name__} completed successfully")
            except Exception as e:
                logger.error(f"Error in pipeline at step {step.__class__.__name__}: {str(e)}")
                return ProcessorResult(success=False, data=None, error=str(e))
        return ProcessorResult(success=True, data=result, error=None)

    @staticmethod
    def _verify_transfer(source: Path, destination: Path) -> bool:
        """Verify that all files were correctly transferred from source to destination.

        :param source: Path to the source directory.
        :param destination: Path to the destination directory.
        :return: True if verification succeeds, False otherwise.
        """
        source_files = list(source.rglob("*"))
        dest_files = list(destination.rglob("*"))

        if len(source_files) != len(dest_files):
            logger.error(
                f"File count mismatch: {len(source_files)} in source, {len(dest_files)} in destination"
            )
            return False

        for src_file in source_files:
            if src_file.is_file():
                dest_file = destination / src_file.relative_to(source)
                if not dest_file.exists():
                    logger.error(f"File missing in destination: {dest_file}")
                    return False
                if src_file.stat().st_size != dest_file.stat().st_size:
                    logger.error(f"File size mismatch: {src_file}")
                    return False

        logger.info("Transfer verification successful")
        return True

    @staticmethod
    def _move_files(source: Path, destination: Path) -> bool:
        """Move files from source directory to destination directory.

        :param source: Path to the source directory.
        :param destination: Path to the destination directory.
        :return: True if all files were moved successfully, False otherwise.
        """
        try:
            if not destination.exists():
                destination.mkdir(parents=True)
            for item in source.glob("*"):
                if item.is_file():
                    shutil.move(str(item), str(destination / item.name))
                elif item.is_dir():
                    shutil.move(str(item), str(destination / item.name))
            logger.info(f"Moved files from {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error moving files: {e}")
            return False
