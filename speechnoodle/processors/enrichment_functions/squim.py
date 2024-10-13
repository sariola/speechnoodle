import logging

import numpy as np
import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE

from speechnoodle.utils.exceptions import EnrichmentError

logger = logging.getLogger(__name__)


def add_squim_quality_estimation(
    batch: dict[str, dict[str, list[np.ndarray] | list[int]]],
    squim_model: SQUIM_OBJECTIVE,
    device: str,
) -> dict[str, list[float]]:
    """Add SQUIM quality estimation to the batch.

    Args:
        batch (Dict[str, Dict[str, Union[List[np.ndarray], List[int]]]]): A batch of examples from the dataset.
            Expected structure: {"audio": {"array": List[np.ndarray], "sampling_rate": List[int]}}
        squim_model (SQUIM_OBJECTIVE): The SQUIM model for quality estimation.
        device (str): The device to use for computations.

    Returns:
        Dict[str, List[float]]: A dictionary containing the SQUIM quality estimations.

    Raises:
        EnrichmentError: If there's an error calculating SQUIM quality estimation.
        KeyError: If the required keys are not present in the batch.
        ValueError: If the audio arrays and sampling rates have different lengths or if the sampling rate is not 16000.
        TypeError: If the input types are incorrect.
    """
    try:
        audio_arrays: list[np.ndarray] = batch["audio"]["array"]
        sampling_rates: list[int] = batch["audio"]["sampling_rate"]

        if len(audio_arrays) != len(sampling_rates):
            raise ValueError("Mismatch between number of audio arrays and sampling rates")

        sdr_scores: list[float] = []
        stoi_scores: list[float] = []
        pesq_scores: list[float] = []

        for audio, sampling_rate in zip(audio_arrays, sampling_rates):
            if not isinstance(audio, np.ndarray) or not isinstance(sampling_rate, int):
                raise TypeError("Invalid type for audio array or sampling rate")
            if sampling_rate != 16000:
                raise ValueError(f"Expected sampling rate of 16000, but got {sampling_rate}")

            audio_tensor: torch.Tensor = (
                torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                metrics: dict[str, torch.Tensor] = squim_model(audio_tensor)

            sdr_scores.append(float(metrics["si_sdr"].item()))
            stoi_scores.append(float(metrics["stoi"].item()))
            pesq_scores.append(float(metrics["pesq"].item()))

        return {"squim_sdr": sdr_scores, "squim_stoi": stoi_scores, "squim_pesq": pesq_scores}
    except KeyError as e:
        logger.error(f"Missing key in batch: {str(e)}")
        raise EnrichmentError(f"Missing key in batch: {str(e)}") from e
    except ValueError as e:
        logger.error(f"Invalid data in batch: {str(e)}")
        raise EnrichmentError(f"Invalid data in batch: {str(e)}") from e
    except TypeError as e:
        logger.error(f"Invalid input type: {str(e)}")
        raise EnrichmentError(f"Invalid input type: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error calculating SQUIM quality estimation: {str(e)}")
        raise EnrichmentError(f"Failed to calculate SQUIM quality estimation: {str(e)}") from e
