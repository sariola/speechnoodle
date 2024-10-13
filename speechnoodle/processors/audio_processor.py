from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TypedDict, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydub import AudioSegment
from transformers import Pipeline, pipeline
from transformers.utils import is_flash_attn_2_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TranscriptionChunk(TypedDict):
    start: float
    end: float
    text: str


@dataclass
class AudioFile:
    """Represents an audio file with its path and binary data.

    :param path: The file path of the audio file.
    :param data: The binary content of the audio file.
    """

    path: str
    data: bytes


@dataclass
class ProcessedAudio:
    """Represents processed audio data with segmented audio and metadata.

    :param segmented_audio: List of numpy arrays containing audio segments.
    :param metadata: DataFrame containing metadata for each audio segment.
    """

    segmented_audio: list[NDArray[np.float32]] = field(default_factory=list)
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        """Validates the ProcessedAudio object after initialization.

        :raises ValueError: If the object's attributes don't meet the expected criteria.
        """
        if not isinstance(self.segmented_audio, list):
            raise ValueError("segmented_audio must be a list")
        if not isinstance(self.metadata, pd.DataFrame):
            raise ValueError("metadata must be a pandas DataFrame")
        if len(self.segmented_audio) != len(self.metadata):
            raise ValueError("Length of segmented_audio must match number of rows in metadata")
        if not all(isinstance(segment, np.ndarray) for segment in self.segmented_audio):
            raise ValueError("All elements in segmented_audio must be numpy arrays")


class AudioProcessorError(Exception):
    """Base exception for AudioProcessor errors."""


class AudioFileError(AudioProcessorError):
    """Exception for errors related to audio files."""


class TranscriptionError(AudioProcessorError):
    """Exception for errors during transcription."""


class ProcessingError(AudioProcessorError):
    """Exception for errors during audio processing."""


@dataclass
class AudioProcessorConfig:
    """Configuration for the AudioProcessor.

    :param model_name: Name of the Whisper model to use.
    :param language: Language code for transcription.
    :param chunk_length_s: Length of audio chunks in seconds.
    :param batch_size: Batch size for processing.
    :param device: Device to use for processing (cuda or cpu).
    :param max_file_size_mb: Maximum allowed file size in MB.
    :param supported_formats: List of supported audio file formats.
    """

    model_name: str = "openai/whisper-large-v3"
    language: str = "finnish"
    chunk_length_s: int = 30
    batch_size: int = 24
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_file_size_mb: int = 100
    supported_formats: list[str] = field(default_factory=lambda: ["mp3", "wav", "flac", "ogg"])


class AudioProcessor:
    """Processes audio files for transcription and segmentation.

    This class handles the entire pipeline of audio processing, including:
    - Setting up the Whisper model for transcription
    - Validating and processing audio files
    - Transcribing audio
    - Segmenting audio based on transcription
    - Creating metadata for processed audio

    :param config: Configuration for the AudioProcessor.
    """

    def __init__(self, config: AudioProcessorConfig = AudioProcessorConfig()):
        """Initialize the AudioProcessor with the given configuration.

        :param config: Configuration for the AudioProcessor.
        """
        self.config = config
        self.whisper_pipe: Pipeline | None = self._setup_whisper_pipeline()

    def _setup_whisper_pipeline(self) -> Pipeline | None:
        """Set up the Whisper pipeline for audio transcription.

        :return: Configured Whisper pipeline.
        :raises AudioProcessorError: If setup fails.
        """
        try:
            logger.info("Setting up Whisper pipeline")
            whisper_pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.config.model_name,
                torch_dtype=torch.float16,
                device=self.config.device,
                model_kwargs={
                    "attn_implementation": "flash_attention_2"
                    if is_flash_attn_2_available()
                    else "sdpa",
                },
                generate_kwargs={"language": self.config.language, "task": "transcribe"},
            )
            logger.info("Whisper pipeline set up successfully")
            return whisper_pipe
        except Exception as e:
            logger.error(f"Error setting up Whisper pipeline: {str(e)}")
            raise AudioProcessorError(f"Failed to set up Whisper pipeline: {str(e)}")

    def process(self, audio_files: list[AudioFile]) -> list[ProcessedAudio]:
        """Process a list of audio files.

        :param audio_files: List of AudioFile objects to process.
        :return: List of ProcessedAudio objects.
        """
        processed_audio_list: list[ProcessedAudio] = []
        for audio_file in audio_files:
            try:
                self._validate_audio_file(audio_file)
                processed_audio = self._process_single_audio(audio_file)
                if processed_audio:
                    processed_audio_list.append(processed_audio)
            except AudioFileError as e:
                logger.error(f"Error processing audio file {audio_file.path}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error processing audio file {audio_file.path}: {str(e)}")
        return processed_audio_list

    def _validate_audio_file(self, audio_file: AudioFile) -> None:
        """Validate an audio file.

        :param audio_file: AudioFile object to validate.
        :raises AudioFileError: If the audio file is invalid.
        """
        if not isinstance(audio_file.path, str) or not audio_file.path:
            raise AudioFileError("Invalid audio file path")
        if not isinstance(audio_file.data, bytes) or not audio_file.data:
            raise AudioFileError("Invalid audio file data")
        if len(audio_file.data) > self.config.max_file_size_mb * 1024 * 1024:
            raise AudioFileError(
                f"Audio file exceeds maximum size of {self.config.max_file_size_mb}MB"
            )
        file_extension = os.path.splitext(audio_file.path)[1][1:].lower()
        if file_extension not in self.config.supported_formats:
            raise AudioFileError(f"Unsupported audio format: {file_extension}")

    def _process_single_audio(self, audio_file: AudioFile) -> ProcessedAudio | None:
        """Process a single audio file.

        :param audio_file: AudioFile object to process.
        :return: ProcessedAudio object if successful, None otherwise.
        """
        try:
            logger.info(f"Processing audio file: {audio_file.path}")
            start_time = time.time()

            transcription = self._transcribe_audio(audio_file.data)
            if transcription is None:
                return None

            processed_chunks = self._process_transcription(transcription)
            if processed_chunks is None:
                return None

            segmented_audio = self._segment_audio(audio_file.data, processed_chunks)
            if not segmented_audio:
                return None

            metadata_df = self._create_metadata(processed_chunks, audio_file.path)
            if metadata_df.empty:
                return None

            processing_time = time.time() - start_time
            logger.info(f"Processed audio file {audio_file.path} in {processing_time:.2f} seconds")

            return ProcessedAudio(segmented_audio=segmented_audio, metadata=metadata_df)
        except Exception as e:
            logger.error(f"Error processing audio file {audio_file.path}: {str(e)}")
            return None

    def _transcribe_audio(
        self, audio_data: bytes
    ) -> dict[str, list[dict[str, list[float] | str]]] | None:
        """Transcribe audio data using the Whisper pipeline.

        :param audio_data: Binary audio data to transcribe.
        :return: Transcription result or None if transcription fails.
        :raises TranscriptionError: If transcription fails.
        """
        if self.whisper_pipe is None:
            raise TranscriptionError("Whisper pipeline is not initialized")
        try:
            logger.info("Transcribing audio")
            result = self.whisper_pipe(
                audio_data,
                chunk_length_s=self.config.chunk_length_s,
                batch_size=self.config.batch_size,
                return_timestamps=True,
                generate_kwargs={"language": self.config.language},
            )
            logger.info("Audio transcription completed")
            return cast(dict[str, list[dict[str, list[float] | str]]], result)
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")

    def _process_transcription(
        self, transcription: dict[str, list[dict[str, list[float] | str]]]
    ) -> list[TranscriptionChunk] | None:
        """Process the raw transcription into a list of TranscriptionChunks.

        :param transcription: Raw transcription from the Whisper pipeline.
        :return: List of TranscriptionChunk objects or None if processing fails.
        :raises ProcessingError: If transcription processing fails.
        """
        try:
            logger.info("Processing transcription")
            chunks = transcription.get("chunks", [])
            processed_chunks: list[TranscriptionChunk] = []
            for chunk in chunks:
                timestamp = chunk.get("timestamp")
                if timestamp and isinstance(timestamp, list) and len(timestamp) == 2:
                    processed_chunk: TranscriptionChunk = {
                        "start": timestamp[0],
                        "end": timestamp[1],
                        "text": chunk.get("text", ""),
                    }
                    processed_chunks.append(processed_chunk)
            logger.info(f"Processed {len(processed_chunks)} transcription chunks")
            return processed_chunks
        except Exception as e:
            logger.error(f"Error processing transcription: {str(e)}")
            raise ProcessingError(f"Failed to process transcription: {str(e)}")

    def _segment_audio(
        self, audio_data: bytes, chunks: list[TranscriptionChunk]
    ) -> list[NDArray[np.float32]]:
        """Segment audio data based on transcription chunks.

        :param audio_data: Binary audio data to segment.
        :param chunks: List of TranscriptionChunk objects.
        :return: List of segmented audio as numpy arrays.
        :raises ProcessingError: If audio segmentation fails.
        """
        try:
            logger.info("Segmenting audio")
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            segmented_audio: list[NDArray[np.float32]] = []
            for chunk in chunks:
                start_ms = int(chunk["start"] * 1000)
                end_ms = int(chunk["end"] * 1000)
                segment = audio[start_ms:end_ms]
                samples = np.array(segment.get_array_of_samples())
                segmented_audio.append(samples.astype(np.float32))
            logger.info(f"Segmented audio into {len(segmented_audio)} chunks")
            return segmented_audio
        except Exception as e:
            logger.error(f"Error segmenting audio: {str(e)}")
            raise ProcessingError(f"Failed to segment audio: {str(e)}")

    def _create_metadata(self, chunks: list[TranscriptionChunk], file_path: str) -> pd.DataFrame:
        """Create metadata DataFrame from transcription chunks.

        :param chunks: List of TranscriptionChunk objects.
        :param file_path: Path of the source audio file.
        :return: DataFrame containing metadata for each chunk.
        :raises ProcessingError: If metadata creation fails.
        """
        try:
            logger.info("Creating metadata")
            metadata = [
                {
                    "segment_id": i,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                    "source_file": file_path,
                }
                for i, chunk in enumerate(chunks)
            ]
            df = pd.DataFrame(metadata)
            logger.info(f"Created metadata with {len(df)} entries")
            return df
        except Exception as e:
            logger.error(f"Error creating metadata: {str(e)}")
            raise ProcessingError(f"Failed to create metadata: {str(e)}")

    def clear_pipeline(self) -> None:
        """Clear the Whisper pipeline from memory.

        This method should be called when the AudioProcessor is no longer needed
        to free up GPU memory.
        """
        if self.whisper_pipe is not None:
            del self.whisper_pipe
            self.whisper_pipe = None
            torch.cuda.empty_cache()
            logger.info("Cleared Whisper pipeline from memory")

    def get_audio_statistics(self, audio_data: bytes) -> dict[str, float]:
        """Get statistics for an audio file.

        :param audio_data: Binary audio data.
        :return: Dictionary containing audio statistics.
        :raises AudioFileError: If statistics retrieval fails.
        """
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            return {
                "duration_seconds": len(audio) / 1000,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "bits_per_sample": audio.sample_width * 8,
            }
        except Exception as e:
            logger.error(f"Error getting audio statistics: {str(e)}")
            raise AudioFileError(f"Failed to get audio statistics: {str(e)}")
