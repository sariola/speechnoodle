import logging

import numpy as np

from speechnoodle.utils.exceptions import EnrichmentError

logger = logging.getLogger(__name__)


def add_speech_duration(
    batch: dict[str, dict[str, list[np.ndarray] | list[int]]],
) -> dict[str, list[float]]:
    """Add speech duration to the batch.

    Args:
        batch (Dict[str, Dict[str, Union[List[np.ndarray], List[int]]]]): A batch of examples from the dataset.
            Expected structure: {"audio": {"array": List[np.ndarray], "sampling_rate": List[int]}}

    Returns:
        Dict[str, List[float]]: A dictionary containing the speech durations.

    Raises:
        EnrichmentError: If there's an error calculating speech duration.
        KeyError: If the required keys are not present in the batch.
        ValueError: If the audio arrays and sampling rates have different lengths.
    """
    try:
        audio_arrays: list[np.ndarray] = batch["audio"]["array"]
        sampling_rates: list[int] = batch["audio"]["sampling_rate"]

        if len(audio_arrays) != len(sampling_rates):
            raise ValueError("Mismatch between number of audio arrays and sampling rates")

        durations: list[float] = []
        for audio, sampling_rate in zip(audio_arrays, sampling_rates):
            if not isinstance(audio, np.ndarray) or not isinstance(sampling_rate, int):
                raise TypeError("Invalid type for audio array or sampling rate")
            if sampling_rate <= 0:
                raise ValueError(f"Invalid sampling rate: {sampling_rate}")
            duration: float = len(audio) / sampling_rate
            durations.append(duration)

        return {"speech_duration": durations}
    except KeyError as e:
        logger.error(f"Missing key in batch: {str(e)}")
        raise EnrichmentError(f"Missing key in batch: {str(e)}") from e
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid data in batch: {str(e)}")
        raise EnrichmentError(f"Invalid data in batch: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error calculating speech duration: {str(e)}")
        raise EnrichmentError(f"Failed to calculate speech duration: {str(e)}") from e
