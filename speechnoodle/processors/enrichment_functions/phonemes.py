import logging

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from speechnoodle.utils.exceptions import EnrichmentError

logger = logging.getLogger(__name__)


def add_phonemes(
    batch: dict[str, list[str]], phonemizer: EspeakBackend, num_workers: int
) -> dict[str, list[str]]:
    """Add phoneme transcriptions to the batch.

    Args:
        batch (Dict[str, List[str]]): A batch of examples from the dataset.
            Expected structure: {"text": List[str]}
        phonemizer (EspeakBackend): The phonemizer to use for text-to-phoneme conversion.
        num_workers (int): Number of workers to use for phonemization.

    Returns:
        Dict[str, List[str]]: A dictionary containing the phoneme transcriptions.

    Raises:
        EnrichmentError: If there's an error during phonemization.
        KeyError: If the required keys are not present in the batch.
        ValueError: If the num_workers is not a positive integer.
        TypeError: If the input types are incorrect.
    """
    try:
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError(f"num_workers must be a positive integer, got {num_workers}")

        texts: list[str] = batch["text"]
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("batch['text'] must be a list of strings")

        phonemized: list[str] = phonemizer.phonemize(
            texts,
            separator=Separator(word=" ", syllable="|", phone=None),
            strip=False,
            njobs=num_workers,
        )

        if not isinstance(phonemized, list) or not all(isinstance(p, str) for p in phonemized):
            raise TypeError("Phonemizer output must be a list of strings")

        return {"phonemes": phonemized}
    except KeyError as e:
        logger.error(f"Missing key in batch: {str(e)}")
        raise EnrichmentError(f"Missing key in batch: {str(e)}") from e
    except ValueError as e:
        logger.error(f"Invalid input value: {str(e)}")
        raise EnrichmentError(f"Invalid input value: {str(e)}") from e
    except TypeError as e:
        logger.error(f"Invalid input type: {str(e)}")
        raise EnrichmentError(f"Invalid input type: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error phonemizing text: {str(e)}")
        raise EnrichmentError(f"Failed to phonemize text: {str(e)}") from e
