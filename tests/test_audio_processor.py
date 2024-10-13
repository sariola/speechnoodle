import io
from typing import Literal, TypedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from pydub import AudioSegment
from typing_extensions import TypedDict

from speechnoodle.processors.audio_processor import (
    AudioFile,
    AudioProcessor,
    AudioProcessorConfig,
    AudioProcessorError,
    ProcessedAudio,
    TranscriptionChunk,
)


class RawTranscriptionChunk(TypedDict):
    timestamp: list[float]
    text: str


class TranscriptionChunk(TypedDict):
    start: float
    end: float
    text: str


@pytest.fixture
def audio_processor() -> AudioProcessor:
    """Fixture to create an AudioProcessor instance for testing.

    Returns:
        AudioProcessor: An instance of AudioProcessor with default configuration.
    """
    return AudioProcessor()


@pytest.fixture
def sample_audio_file() -> AudioFile:
    """Fixture to create a sample AudioFile for testing.

    Returns:
        AudioFile: A sample AudioFile instance containing 1 second of silence.
    """
    audio = AudioSegment.silent(duration=1000)  # 1 second of silence
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return AudioFile(path="test.wav", data=buffer.getvalue())


def test_process_single_audio_success(
    audio_processor: AudioProcessor, sample_audio_file: AudioFile
) -> None:
    """Test successful processing of a single audio file.

    This test mocks the internal methods of AudioProcessor to simulate
    successful processing of an audio file and verifies the output.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        sample_audio_file (AudioFile): A sample audio file for testing.
    """
    with (
        patch.object(AudioProcessor, "_transcribe_audio") as mock_transcribe,
        patch.object(AudioProcessor, "_process_transcription") as mock_process,
        patch.object(AudioProcessor, "_segment_audio") as mock_segment,
        patch.object(AudioProcessor, "_create_metadata") as mock_metadata,
    ):
        mock_transcribe.return_value = {"chunks": [{"timestamp": [0, 1], "text": "Test"}]}
        mock_process.return_value = [{"start": 0, "end": 1, "text": "Test"}]
        mock_segment.return_value = [np.array([0, 1, 2], dtype=np.float32)]
        mock_metadata.return_value = pd.DataFrame({"segment_id": [0], "text": ["Test"]})

        result = audio_processor._process_single_audio(sample_audio_file)

        assert isinstance(result, ProcessedAudio)
        assert len(result.segmented_audio) == 1
        assert isinstance(result.segmented_audio[0], np.ndarray)
        assert isinstance(result.metadata, pd.DataFrame)
        assert not result.metadata.empty


def test_process_single_audio_failure(
    audio_processor: AudioProcessor, sample_audio_file: AudioFile
) -> None:
    """Test failure handling in processing a single audio file.

    This test simulates a failure in the transcription step and verifies
    that the method returns None as expected.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        sample_audio_file (AudioFile): A sample audio file for testing.
    """
    with patch.object(AudioProcessor, "_transcribe_audio", return_value=None):
        result = audio_processor._process_single_audio(sample_audio_file)
        assert result is None


def test_process_multiple_audio_files(
    audio_processor: AudioProcessor, sample_audio_file: AudioFile
) -> None:
    """Test processing of multiple audio files.

    This test verifies that the process method correctly handles multiple
    audio files and returns the expected number of ProcessedAudio objects.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        sample_audio_file (AudioFile): A sample audio file for testing.
    """
    with patch.object(AudioProcessor, "_process_single_audio") as mock_process:
        mock_process.return_value = ProcessedAudio(
            segmented_audio=[np.array([0, 1, 2], dtype=np.float32)],
            metadata=pd.DataFrame({"segment_id": [0], "text": ["Test"]}),
        )

        results = audio_processor.process([sample_audio_file, sample_audio_file])

        assert len(results) == 2
        assert all(isinstance(result, ProcessedAudio) for result in results)


def test_process_with_invalid_audio_file(audio_processor: AudioProcessor) -> None:
    """Test processing with an invalid audio file.

    This test checks that the process method correctly handles invalid
    audio files by returning an empty list.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    invalid_file = AudioFile(path="", data=b"")
    results = audio_processor.process([invalid_file])
    assert len(results) == 0


def test_validate_audio_file(audio_processor: AudioProcessor, sample_audio_file: AudioFile) -> None:
    """Test validation of audio files.

    This test checks that the validate_audio_file method correctly
    identifies valid and invalid audio files.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        sample_audio_file (AudioFile): A sample audio file for testing.
    """
    assert audio_processor._validate_audio_file(sample_audio_file) is None

    invalid_file = AudioFile(path="", data=b"")
    with pytest.raises(AudioProcessorError):
        audio_processor._validate_audio_file(invalid_file)


def test_get_audio_statistics(
    audio_processor: AudioProcessor, sample_audio_file: AudioFile
) -> None:
    """Test retrieval of audio statistics.

    This test verifies that the get_audio_statistics method returns
    the expected statistics for a given audio file.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        sample_audio_file (AudioFile): A sample audio file for testing.
    """
    stats = audio_processor.get_audio_statistics(sample_audio_file.data)
    assert isinstance(stats, dict)
    assert "duration_seconds" in stats
    assert "sample_rate" in stats
    assert "channels" in stats
    assert "bits_per_sample" in stats


@pytest.mark.parametrize(
    "audio_length,expected_chunks",
    [
        (5000, 1),  # 5 seconds, should be 1 chunk
        (35000, 2),  # 35 seconds, should be 2 chunks
        (65000, 3),  # 65 seconds, should be 3 chunks
    ],
)
def test_transcribe_audio_chunking(
    audio_processor: AudioProcessor, audio_length: int, expected_chunks: int
) -> None:
    """Test audio chunking during transcription.

    This test verifies that the _transcribe_audio method correctly chunks
    audio data of various lengths and calls the Whisper pipeline the
    expected number of times.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        audio_length (int): Length of the test audio in milliseconds.
        expected_chunks (int): Expected number of chunks.
    """
    with patch.object(AudioProcessor, "_setup_whisper_pipeline") as mock_setup:
        mock_pipe = MagicMock()
        mock_setup.return_value = mock_pipe

        audio = AudioSegment.silent(duration=audio_length)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        audio_data = buffer.getvalue()

        audio_processor._transcribe_audio(audio_data)

        _, kwargs = mock_pipe.call_args
        assert kwargs["chunk_length_s"] == 30
        assert mock_pipe.call_count == expected_chunks


def test_process_transcription_format(audio_processor: AudioProcessor) -> None:
    """Test the format of processed transcriptions.

    This test checks that the _process_transcription method correctly
    formats the raw transcription data into the expected structure.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    raw_transcription: dict[Literal["chunks"], list[RawTranscriptionChunk]] = {
        "chunks": [
            {"timestamp": [0.0, 2.5], "text": "Hello"},
            {"timestamp": [2.5, 5.0], "text": "World"},
        ]
    }
    processed = audio_processor._process_transcription(raw_transcription)
    assert isinstance(processed, list), "Processed output should be a list"
    assert len(processed) == 2, "Expected 2 processed chunks"
    for chunk in processed:
        assert isinstance(chunk, dict), "Each processed chunk should be a dictionary"
        assert set(chunk.keys()) == {"start", "end", "text"}, "Incorrect keys in processed chunk"
        assert isinstance(chunk["start"], float), "Start time should be a float"
        assert isinstance(chunk["end"], float), "End time should be a float"
        assert isinstance(chunk["text"], str), "Text should be a string"


def test_segment_audio_output(audio_processor: AudioProcessor) -> None:
    """Test the output of audio segmentation.

    This test verifies that the _segment_audio method correctly segments
    audio data based on given chunks and returns the expected output format.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    audio = AudioSegment.silent(duration=5000)  # 5 seconds of silence
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    audio_data = buffer.getvalue()

    chunks: list[TranscriptionChunk] = [
        {"start": 0, "end": 2, "text": "Hello"},
        {"start": 2, "end": 5, "text": "World"},
    ]

    segmented = audio_processor._segment_audio(audio_data, chunks)
    assert isinstance(segmented, list)
    assert len(segmented) == 2
    assert all(isinstance(segment, np.ndarray) for segment in segmented)
    assert segmented[0].shape[0] == 2 * audio.frame_rate  # 2 seconds
    assert segmented[1].shape[0] == 3 * audio.frame_rate  # 3 seconds


def test_create_metadata_format(audio_processor: AudioProcessor) -> None:
    """Test the format of created metadata.

    This test checks that the _create_metadata method generates a DataFrame
    with the expected structure and content.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    chunks: list[TranscriptionChunk] = [
        {"start": 0, "end": 2, "text": "Hello"},
        {"start": 2, "end": 5, "text": "World"},
    ]
    file_path = "test.wav"

    metadata = audio_processor._create_metadata(chunks, file_path)
    assert isinstance(metadata, pd.DataFrame)
    assert set(metadata.columns) == {"segment_id", "start", "end", "text", "source_file"}
    assert len(metadata) == 2
    assert all(metadata["source_file"] == file_path)


def test_end_to_end_processing(
    audio_processor: AudioProcessor, sample_audio_file: AudioFile
) -> None:
    """Test end-to-end audio processing.

    This test verifies the complete audio processing pipeline by processing
    a sample audio file and checking the structure of the output.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        sample_audio_file (AudioFile): A sample audio file for testing.
    """
    result = audio_processor._process_single_audio(sample_audio_file)
    assert isinstance(result, ProcessedAudio)
    assert len(result.segmented_audio) > 0
    assert not result.metadata.empty
    assert set(result.metadata.columns) == {"segment_id", "start", "end", "text", "source_file"}
    assert all(isinstance(segment, np.ndarray) for segment in result.segmented_audio)


def test_error_handling_in_process(
    audio_processor: AudioProcessor, sample_audio_file: AudioFile
) -> None:
    """Test error handling in the process method.

    This test simulates an error during processing and verifies that the
    process method handles it gracefully by returning an empty list.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        sample_audio_file (AudioFile): A sample audio file for testing.
    """
    with patch.object(AudioProcessor, "_process_single_audio", side_effect=Exception("Test error")):
        results = audio_processor.process([sample_audio_file])
        assert len(results) == 0


def test_whisper_pipeline_initialization(audio_processor: AudioProcessor) -> None:
    """Test initialization of the Whisper pipeline.

    This test checks that the Whisper pipeline is correctly initialized
    and is callable.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    assert audio_processor.whisper_pipe is not None
    assert callable(audio_processor.whisper_pipe)


@pytest.mark.parametrize(
    "cuda_available,flash_attn_available",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_setup_whisper_pipeline_success(
    audio_processor: AudioProcessor, cuda_available: bool, flash_attn_available: bool
) -> None:
    """Test successful setup of the Whisper pipeline.

    This test verifies that the _setup_whisper_pipeline method correctly
    configures the pipeline based on CUDA and flash attention availability.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        cuda_available (bool): Whether CUDA is available.
        flash_attn_available (bool): Whether flash attention is available.
    """
    with (
        patch("speechnoodle.processors.audio_processor.pipeline") as mock_pipeline,
        patch(
            "speechnoodle.processors.audio_processor.torch.cuda.is_available",
            return_value=cuda_available,
        ),
        patch(
            "speechnoodle.processors.audio_processor.is_flash_attn_2_available",
            return_value=flash_attn_available,
        ),
    ):
        mock_pipeline.return_value = MagicMock()
        result = audio_processor._setup_whisper_pipeline()

        assert result is not None
        mock_pipeline.assert_called_once_with(
            task="automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=torch.float16,
            device="cuda" if cuda_available else "cpu",
            model_kwargs={
                "attn_implementation": "flash_attention_2" if flash_attn_available else "sdpa",
            },
            generate_kwargs={"language": "finnish", "task": "transcribe"},
        )


@pytest.mark.parametrize(
    "error",
    [
        Exception("Generic error"),
        ValueError("Invalid value"),
        RuntimeError("Runtime issue"),
        ImportError("Missing dependency"),
    ],
)
def test_setup_whisper_pipeline_error_handling(
    audio_processor: AudioProcessor, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """Test error handling in Whisper pipeline setup.

    This test simulates various errors during pipeline setup and verifies
    that they are correctly logged and handled.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        error (Exception): The error to simulate.
        caplog (pytest.LogCaptureFixture): Fixture to capture log output.
    """
    with patch("speechnoodle.processors.audio_processor.pipeline", side_effect=error):
        with pytest.raises(AudioProcessorError):
            audio_processor._setup_whisper_pipeline()

        assert f"Error setting up Whisper pipeline: {str(error)}" in caplog.text


def test_setup_whisper_pipeline_configuration(audio_processor: AudioProcessor) -> None:
    """Test configuration of the Whisper pipeline.

    This test verifies that the _setup_whisper_pipeline method correctly
    configures the pipeline with the expected parameters.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    with patch("speechnoodle.processors.audio_processor.pipeline") as mock_pipeline:
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        result = audio_processor._setup_whisper_pipeline()

        assert result == mock_pipe
        mock_pipeline.assert_called_once()
        _, kwargs = mock_pipeline.call_args
        assert kwargs["task"] == "automatic-speech-recognition"
        assert kwargs["model"] == "openai/whisper-large-v3"
        assert kwargs["torch_dtype"] == torch.float16
        assert kwargs["generate_kwargs"] == {"language": "finnish", "task": "transcribe"}

        # Test that the pipeline is cached
        audio_processor._setup_whisper_pipeline()
        assert mock_pipeline.call_count == 1


@pytest.mark.parametrize(
    "audio_length,expected_output",
    [
        (1000, {"chunks": [{"timestamp": [0.0, 0.1], "text": "Short"}]}),
        (80000, {"chunks": [{"timestamp": [0.0, 5.0], "text": "Medium"}]}),
        (
            1600000,
            {
                "chunks": [
                    {"timestamp": [0.0, 30.0], "text": "Long chunk 1"},
                    {"timestamp": [30.0, 60.0], "text": "Long chunk 2"},
                    {"timestamp": [60.0, 90.0], "text": "Long chunk 3"},
                    {"timestamp": [90.0, 100.0], "text": "Long chunk 4"},
                ]
            },
        ),
    ],
)
def test_transcribe_audio_various_lengths(
    audio_processor: AudioProcessor, audio_length: int, expected_output: dict[str, Any]
) -> None:
    """Test transcription of audio data of various lengths.

    This test verifies that the _transcribe_audio method correctly handles
    audio data of different lengths and returns the expected output.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        audio_length (int): Length of the test audio in milliseconds.
        expected_output (Dict[str, Any]): Expected output for the given audio length.
    """
    with patch.object(AudioProcessor, "_setup_whisper_pipeline") as mock_setup_pipeline:
        mock_pipe = MagicMock()
        mock_setup_pipeline.return_value = mock_pipe
        mock_pipe.return_value = expected_output

        audio_data = np.random.bytes(audio_length)
        result = audio_processor._transcribe_audio(audio_data)

        assert result == expected_output
        mock_pipe.assert_called_once_with(
            audio_data,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
            generate_kwargs={"language": "finnish"},
        )


def test_transcribe_audio_output_format(audio_processor: AudioProcessor) -> None:
    """Test the output format of the _transcribe_audio method.

    This test verifies that the _transcribe_audio method returns a dictionary
    with the expected structure and content.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    mock_pipe = MagicMock()
    audio_processor.whisper_pipe = mock_pipe

    expected_output = {
        "chunks": [
            {"timestamp": [0.0, 5.0], "text": "Test transcription"},
            {"timestamp": [5.0, 10.0], "text": "Another chunk"},
        ]
    }
    mock_pipe.return_value = expected_output

    result = audio_processor._transcribe_audio(b"audio_data")

    assert isinstance(result, dict)
    assert "chunks" in result
    assert isinstance(result["chunks"], list)
    for chunk in result["chunks"]:
        assert isinstance(chunk, dict)
        assert "timestamp" in chunk
        assert "text" in chunk
        assert isinstance(chunk["timestamp"], list)
        assert len(chunk["timestamp"]) == 2
        assert isinstance(chunk["timestamp"][0], float)
        assert isinstance(chunk["timestamp"][1], float)
        assert isinstance(chunk["text"], str)


@pytest.mark.parametrize(
    "error",
    [
        Exception("Generic error"),
        ValueError("Invalid audio data"),
        RuntimeError("Transcription failed"),
    ],
)
def test_transcribe_audio_error_handling(
    audio_processor: AudioProcessor, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """Test error handling in the _transcribe_audio method.

    This test simulates various errors during transcription and verifies
    that they are correctly logged and handled.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        error (Exception): The error to simulate.
        caplog (pytest.LogCaptureFixture): Fixture to capture log output.
    """
    mock_pipe = MagicMock(side_effect=error)
    audio_processor.whisper_pipe = mock_pipe

    result = audio_processor._transcribe_audio(b"audio_data")

    assert result is None
    assert f"Error transcribing audio: {str(error)}" in caplog.text


@pytest.mark.parametrize(
    "input_transcription,expected_output",
    [
        (
            {
                "chunks": [
                    {"timestamp": [0.0, 5.0], "text": "First chunk"},
                    {"timestamp": [5.0, 10.0], "text": "Second chunk"},
                ]
            },
            [
                {"start": 0.0, "end": 5.0, "text": "First chunk"},
                {"start": 5.0, "end": 10.0, "text": "Second chunk"},
            ],
        ),
        (
            {
                "chunks": [
                    {"timestamp": [0.0, 1.5], "text": "Short chunk"},
                    {"timestamp": [1.5, 20.0], "text": "Long chunk"},
                ]
            },
            [
                {"start": 0.0, "end": 1.5, "text": "Short chunk"},
                {"start": 1.5, "end": 20.0, "text": "Long chunk"},
            ],
        ),
        (
            {"chunks": [{"timestamp": [0.0, 0.5], "text": "Very short"}]},
            [{"start": 0.0, "end": 0.5, "text": "Very short"}],
        ),
    ],
)
def test_process_transcription_correct_parsing(
    audio_processor: AudioProcessor,
    input_transcription: dict[Literal["chunks"], list[RawTranscriptionChunk]],
    expected_output: list[TranscriptionChunk],
) -> None:
    """Test correct parsing of raw transcription data.

    This test verifies that the _process_transcription method correctly
    parses the raw transcription data into the expected structure.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        input_transcription (Dict[Literal['chunks'], List[RawTranscriptionChunk]]): Raw transcription data.
        expected_output (List[TranscriptionChunk]): Expected output.
    """
    result = audio_processor._process_transcription(input_transcription)
    assert result == expected_output, "Processed transcription does not match expected output"
    for chunk in result:
        assert isinstance(chunk["start"], float), "Start time should be a float"
        assert isinstance(chunk["end"], float), "End time should be a float"
        assert isinstance(chunk["text"], str), "Text should be a string"


@pytest.mark.parametrize(
    "empty_input",
    [
        {"chunks": []},
        {},
        {"chunks": None},
    ],
)
def test_process_transcription_empty_input(
    audio_processor: AudioProcessor,
    empty_input: dict[Literal["chunks"], list[RawTranscriptionChunk] | None],
) -> None:
    """Test handling of empty input in _process_transcription.

    This test verifies that the _process_transcription method returns
    an empty list when given empty input.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        empty_input (Dict[Literal['chunks'], Union[List[RawTranscriptionChunk], None]]): Empty input data.
    """
    result = audio_processor._process_transcription(empty_input)
    assert result == []


@pytest.mark.parametrize(
    "malformed_transcription",
    [
        {"chunks": [{"text": "No timestamp"}]},
        {"chunks": [{"timestamp": [5.0], "text": "Incomplete timestamp"}]},
        {"chunks": [{"timestamp": "invalid", "text": "Invalid timestamp type"}]},
        {"chunks": [{"timestamp": [0.0, 5.0], "text": 123}]},  # Non-string text
        {
            "chunks": [
                {"timestamp": [10.0, 15.0], "text": "Correct chunk"},
                {"timestamp": [5.0], "text": "Incomplete"},
                {"text": "No timestamp"},
            ]
        },
    ],
)
def test_process_transcription_malformed_data(
    audio_processor: AudioProcessor,
    malformed_transcription: dict[Literal["chunks"], list[dict[str, list[float] | str | int]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test handling of malformed input in _process_transcription.

    This test verifies that the _process_transcription method correctly
    handles malformed input data and logs appropriate warnings.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        malformed_transcription (Dict[Literal['chunks'], List[Dict[str, Union[List[float], str, int]]]]): Malformed input data.
        caplog (pytest.LogCaptureFixture): Fixture to capture log output.
    """
    result = audio_processor._process_transcription(malformed_transcription)

    assert "Skipping malformed chunk" in caplog.text

    # Check that only valid chunks are processed
    for chunk in result:
        assert isinstance(chunk, dict)
        assert "start" in chunk
        assert "end" in chunk
        assert "text" in chunk
        assert isinstance(chunk["start"], float)
        assert isinstance(chunk["end"], float)
        assert isinstance(chunk["text"], str)


def test_clear_pipeline(audio_processor: AudioProcessor) -> None:
    """Test clearing of the Whisper pipeline.

    This test verifies that the clear_pipeline method correctly removes
    the Whisper pipeline and clears CUDA cache.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    with patch(
        "speechnoodle.processors.audio_processor.torch.cuda.empty_cache"
    ) as mock_empty_cache:
        audio_processor.clear_pipeline()
        assert audio_processor.whisper_pipe is None
        mock_empty_cache.assert_called_once()


def test_get_audio_statistics_error_handling(audio_processor: AudioProcessor) -> None:
    """Test error handling in get_audio_statistics method.

    This test simulates an error during audio statistics retrieval and
    verifies that it's correctly handled and raised as an AudioFileError.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    invalid_audio_data = b"invalid_audio_data"
    with pytest.raises(AudioFileError):
        audio_processor.get_audio_statistics(invalid_audio_data)


@pytest.mark.parametrize(
    "config_params",
    [
        {"model_name": "openai/whisper-tiny"},
        {"language": "english"},
        {"chunk_length_s": 15},
        {"batch_size": 32},
        {"device": "cpu"},
        {"max_file_size_mb": 50},
        {"supported_formats": ["mp3", "wav"]},
    ],
)
def test_audio_processor_config(config_params: dict[str, str | int | list[str]]) -> None:
    """Test AudioProcessor configuration.

    This test verifies that the AudioProcessor correctly applies custom
    configuration parameters.

    Args:
        config_params (Dict[str, Union[str, int, List[str]]]): Custom configuration parameters.
    """
    config = AudioProcessorConfig(**config_params)
    processor = AudioProcessor(config)
    for key, value in config_params.items():
        assert getattr(processor.config, key) == value


def test_process_transcription_empty_input(audio_processor: AudioProcessor) -> None:
    """Test processing of empty transcription input.

    This test verifies that the _process_transcription method correctly
    handles empty input by returning an empty list.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    empty_transcription: dict[Literal["chunks"], list[RawTranscriptionChunk]] = {"chunks": []}
    result = audio_processor._process_transcription(empty_transcription)
    assert isinstance(result, list)
    assert len(result) == 0


def test_segment_audio_empty_chunks(
    audio_processor: AudioProcessor, sample_audio_file: AudioFile
) -> None:
    """Test audio segmentation with empty chunks.

    This test verifies that the _segment_audio method correctly handles
    empty chunks input by returning an empty list.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
        sample_audio_file (AudioFile): A sample audio file for testing.
    """
    empty_chunks: list[TranscriptionChunk] = []
    result = audio_processor._segment_audio(sample_audio_file.data, empty_chunks)
    assert isinstance(result, list)
    assert len(result) == 0


def test_create_metadata_empty_chunks(audio_processor: AudioProcessor) -> None:
    """Test metadata creation with empty chunks.

    This test verifies that the _create_metadata method correctly handles
    empty chunks input by returning an empty DataFrame.

    Args:
        audio_processor (AudioProcessor): The AudioProcessor instance.
    """
    empty_chunks: list[TranscriptionChunk] = []
    result = audio_processor._create_metadata(empty_chunks, "test.wav")
    assert isinstance(result, pd.DataFrame)
    assert result.empty
