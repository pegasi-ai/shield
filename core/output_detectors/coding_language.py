import logging
from typing import List, Optional

from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    TextClassificationPipeline,
)

from pegasi_shield_safeguards.utils import device

from pegasi_shield_safeguards.output_detectors.base_detector import Detector

log = logging.getLogger(__name__)
model_name = "huggingface/CodeBERTa-language-id"


class LanguageDetection(Detector):
    """
    A detector to check for specific languages in text using Huggingface Transformers.
    """

    def __init__(self):
        """
        Initializes LanguageDetection with a language detection model.
        """
        self._pipeline = TextClassificationPipeline(
            model=RobertaForSequenceClassification.from_pretrained(model_name),
            tokenizer=RobertaTokenizer.from_pretrained(model_name),
            device=device,
        )
        log.debug(f"Initialized language detection model: {model_name}")

    def detect_languages(
        self, text: str, allowed: List[str], denied: List[str]
    ) -> (bool, float):
        """
        Checks whether the text contains languages listed in the 'allowed' or 'denied' lists.

        Parameters:
            text (str): The text to check.
            allowed (List[str]): A list of allowed languages.
            denied (List[str]): A list of denied languages.

        Returns:
            bool: True if the text contains allowed languages or doesn't contain denied languages. False otherwise.
            float: Risk score, where 0 is no risk and 1 is the highest risk.
        """
        if text.strip() == "":
            return True, 0.0

        languages = self._pipeline(text)
        log.debug(f"Detected languages: {languages}")

        for language in languages:
            language_name = language["label"]
            score = round(language["score"], 2)

            if len(allowed) > 0 and language_name in allowed:
                log.debug(
                    f"Language {language_name} found in the allowed list with score {score}"
                )
                return True, 0.0

            if len(denied) > 0 and language_name in denied:
                log.warning(f"Language {language_name} is not allowed (score {score})")
                return False, score

        if len(allowed) > 0:
            log.warning(f"No allowed languages detected: {languages}")
            return False, 1.0

        log.debug(f"No denied languages detected: {languages}")

        return True, 0.0


class CodingLanguageOutput(Detector):
    """
    A class for detecting if the output includes code in specific programming languages.
    """

    def __init__(
        self, allowed: Optional[List[str]] = None, denied: Optional[List[str]] = None
    ):
        """
        Initializes CodingLanguage with the allowed and denied languages.

        Parameters:
            allowed (Optional[List[str]]): A list of allowed languages. Default is an empty list.
            denied (Optional[List[str]]): A list of denied languages. Default is an empty list.

        Raises:
            ValueError: If both 'allowed' and 'denied' lists are provided or if both are empty.
        """
        if not allowed:
            allowed = []

        if not denied:
            denied = []

        if len(allowed) > 0 and len(denied) > 0:
            raise ValueError("Provide either allowed or denied programming languages")

        if len(allowed) == 0 and len(denied) == 0:
            raise ValueError("No allowed or denied programming languages provided")

        self._allowed = allowed
        self._denied = denied

    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        language_detection = LanguageDetection()
        valid, score = language_detection.detect_languages(
            output, self._allowed, self._denied
        )
        return prompt, valid, score
