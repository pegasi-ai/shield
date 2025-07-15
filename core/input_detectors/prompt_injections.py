import logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)
from pegasi_shield_safeguards.utils import device
from pegasi_shield_safeguards.input_detectors.base_detector import Detector

_model_path_2 = "hubert233/GPTFuzz"
log = logging.getLogger(__name__)


class PromptInjections(Detector):

    def __init__(self, threshold: float = 0.75):
        model_2 = AutoModelForSequenceClassification.from_pretrained(_model_path_2)
        tokenizer_2 = AutoTokenizer.from_pretrained(_model_path_2)

        self._threshold = threshold
        self._text_classification_pipeline = TextClassificationPipeline(
            model=model_2,
            tokenizer=tokenizer_2,
            device=device,
        )
        log.debug(f"Initialized model: {_model_path_2} on device {device}")

    def scan(self, prompt: str) -> (str, bool, float):
        if prompt.strip() == "":
            return prompt, True, 0.0

        max_length = self._text_classification_pipeline.tokenizer.model_max_length

        if len(prompt) > max_length:
            log.warning(f"Prompt length exceeded max length. Truncating to fit.")
            prompt = prompt[:max_length]

        result = self._text_classification_pipeline(
            prompt, truncation=True, max_length=max_length
        )

        injection_score = round(
            (
                result[0]["score"]
                if result[0]["label"] == "INJECTION"
                else 1 - result[0]["score"]
            ),
            2,
        )

        if injection_score > self._threshold:
            log.warning(
                f"Detected prompt injection with score: {injection_score}, threshold: {self._threshold}"
            )
            return prompt, False, injection_score

        log.debug(
            f"No prompt injection detected (Score: {injection_score}, Threshold: {self._threshold})"
        )
        return prompt, True, 0.0
