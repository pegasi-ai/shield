import logging
from genbit.genbit_metrics import GenBitMetrics
from pegasi_shield.output_detectors.base_detector import Detector

_language = "en"

log = logging.getLogger(__name__)


class Bias(Detector):
    """
    A detector to check for bias in text using GenBit metrics.
    """

    def __init__(self, language_code=_language):
        """
        Initializes BiasDetection with GenBit and a language code.

        Parameters:
            language_code (str): The language code for GenBit.
        """
        self._genbit_metrics = None  # Initialize GenBitMetrics to None
        self._language_code = language_code
        
        log.debug(f"Initialized Bias detector with language code: {language_code}")

    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        """
        Detects bias in the given text.

        Parameters:
            text (str): The text to check for bias.

        Returns:
            tuple: A tuple containing:
                - str: The input text.
                - bool: True if bias is not detected in the text, False otherwise.
                - dict: A dictionary containing bias-related information (e.g., scores, metrics).
        """
        if output.strip() == "":
            return output, True, {}

        # Reinitialize GenBitMetrics for each scan to clear data
        self._genbit_metrics = GenBitMetrics(
            self._language_code, context_window=5, distance_weight=0.95, percentile_cutoff=80
        )

        # Add text to GenBit for bias detection
        self._genbit_metrics.add_data([output], tokenized=False)

        # Generate gender bias metrics from GenBit
        metrics = self._genbit_metrics.get_metrics(
            output_statistics=True, output_word_list=True
        )
        log.debug(f"GenBit Bias metrics: {metrics}")

        # TensorFlow model inference removed - using only GenBit metrics

        # Check if bias is detected based on your criteria
        # You can customize this part based on your specific bias detection requirements
        bias_detected = False

        # Check if bias is detected based on GenBit metrics
        gender_metrics_score = metrics["genbit_score"]
        # Check your specific bias detection criteria here
        # If the gender bias score is above a certain threshold, consider it as bias detected
        if gender_metrics_score > 0.5:
            bias_detected = True

        if bias_detected:
            log.warning("Bias detected in the output.")
        else:
            log.debug("No bias detected in the output.")

        genbit_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in {"additional_metrics", "statistics", "token_based_metrics"}
        }

        result = {
            "bias_detected": bias_detected,
            "bias_score": gender_metrics_score,
            "genbit_metrics": genbit_metrics,
        }

        return output, not bias_detected, result
