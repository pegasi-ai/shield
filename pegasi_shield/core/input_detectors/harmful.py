import openai
import logging
from ..keys import load_api_key
from .base_detector import Detector

log = logging.getLogger(__name__)


class HarmfulInput(Detector):
    """
    A detector to detect harmful content using OpenAI's moderation API.
    """

    def __init__(self, thresholds=None):
        self._api_key = load_api_key("OPENAI_API_KEY")
        if thresholds == None:
            self.thresholds = {
                "hate": 0.5,
                "hate/threatening": 0.5,
                "maliciousness": 0.5,
                "controversiality": 0.5,
                "coherence": 0.5,
                "correctness": 0.5,
                "harmfulness": 0.5,
                "maliciousness": 0.5,
                "helpfulness": 0.5,
                "misogyny": 0.5,
                "criminality": 0.5,
                "insensitivity": 0.5,
            }
        else:
            self.thresholds = thresholds

    def _moderation(self, text: str) -> dict:
        openai.api_key = self._api_key
        client = openai.OpenAI()
        return client.moderations.create(input=text)

    def scan(self, text: str) -> (str, bool, dict):
        if text.strip() == "":
            return text, True, {}

        moderation_result = self._moderation(text)
        results = getattr(moderation_result, "results", None)

        if results is not None and len(results) > 0:
            category_scores = getattr(results[0], "category_scores", {})
            is_harmful = False
            scores = {}

            for category, threshold in self.thresholds.items():
                score = getattr(category_scores, "category", 0)
                scores[category] = score
                if score >= threshold:
                    is_harmful = True
                    log.warning(
                        f"Content is harmful in category {category} (Score: {score})"
                    )

            if not is_harmful:
                log.debug("Content is not harmful.")
                return text, True, scores

        log.warning("Content is harmful.")
        return (
            text,
            False,
            scores,
        )  # You can set the toxicity score to 1.0 for harmful content
