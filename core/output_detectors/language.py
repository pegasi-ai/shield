import logging
from typing import List

from pegasi_shield.input_detectors.language import LanguageInput
from pegasi_shield.output_detectors.base_detector import Detector

log = logging.getLogger(__name__)


class LanguageOutput(Detector):

    def __init__(self, valid_languages: List[str]):
        """
        Initializes the Language detector with a list of valid languages.

        Parameters:
            valid_languages (List[str]): A list of valid language codes.
        """

        self._detector = LanguageInput(valid_languages=valid_languages)

    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        return self._detector.scan(output)
