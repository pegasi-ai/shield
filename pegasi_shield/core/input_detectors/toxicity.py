import logging

from googleapiclient import discovery
from ..keys import load_api_key, init_perspective_key

from .base_detector import Detector

log = logging.getLogger(__name__)


class ToxicityInput(Detector):
    """
    A detector to detect and prevent toxic comments using the Perspective API.
    """

    def __init__(self):
        """
        Initializes the ToxicityDetector with a Perspective API key.

        Parameters:
            api_key (str): Your Perspective API key.
        """
        self._api_key = load_api_key("PERSPECTIVE_API_KEY")
        init_perspective_key()

        # Create a client for the Perspective API
        self._client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self._api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def _analyze_toxicity(self, text: str) -> float:
        """
        Analyzes the toxicity of a given text using the Perspective API.

        Parameters:
            text (str): The text to analyze for toxicity.

        Returns:
            float: The toxicity score (0.0 to 1.0) indicating the level of toxicity.
        """
        analyze_request = {
            "comment": {"text": text},
            "requestedAttributes": {"TOXICITY": {}},
        }

        response = self._client.comments().analyze(body=analyze_request).execute()
        toxicity_score = (
            response.get("attributeScores", {})
            .get("TOXICITY", {})
            .get("summaryScore", {})
            .get("value", 0.0)
        )
        return toxicity_score

    def scan(self, comment_text: str) -> (str, bool, float):
        if comment_text.strip() == "":
            return comment_text, True, 0.0

        toxicity_score = self._analyze_toxicity(comment_text)

        if toxicity_score < 0.5:
            log.debug(f"Comment is not toxic (Toxicity Score: {toxicity_score})")
            return comment_text, True, toxicity_score

        log.warning(f"Comment is toxic (Toxicity Score: {toxicity_score})")
        return comment_text, False, toxicity_score
