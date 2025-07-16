import logging
from ftlangdetect import detect
from pegasi_shield.input_detectors.base_detector import Detector

log = logging.getLogger(__name__)


class LanguageInput(Detector):

    def __init__(self, valid_languages=None, low_memory=True):
        self._valid_languages = valid_languages or ["en"]
        self._low_memory = low_memory
        log.debug(
            f"Initialized LanguageInput detector with valid languages: {self._valid_languages}, low_memory: {self._low_memory}"
        )

    def scan(self, prompt: str) -> (str, bool, float):
        if prompt.strip() == "":
            return prompt, True, 0.0

        result = detect(text=prompt, low_memory=self._low_memory)
        detected_language = result["lang"]
        confidence_score = result["score"]

        if detected_language not in self._valid_languages:
            log.warning(
                f"Detected invalid language: {detected_language}, confidence score: {confidence_score}"
            )
            return prompt, False, confidence_score

        log.debug(
            f"Detected valid language: {detected_language}, confidence score: {confidence_score}"
        )
        return prompt, True, confidence_score
