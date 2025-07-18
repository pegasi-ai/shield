import logging
from genbit.genbit_metrics import GenBitMetrics
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from pegasi_shield.output_detectors.base_detector import Detector

_model_name = "d4data/bias-detection-model"
_language = "en"

log = logging.getLogger(__name__)


class Bias(Detector):
    """
    A detector to check for bias in text using GenBit and a Huggingface bias detection model.
    """

    def __init__(self, language_code=_language, model_name=_model_name):
        """
        Initializes BiasDetection with GenBit, a language code, and a Huggingface model for bias detection.

        Parameters:
            language_code (str): The language code for GenBit.
            model_name (str): The HuggingFace model name for bias detection.
        """
        self._genbit_metrics = None  # Initialize GenBitMetrics to None
        self._bias_detection_model = (
            TFAutoModelForSequenceClassification.from_pretrained(model_name)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        log.debug(f"Initialized HuggingFace model: {model_name}")

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
            _language, context_window=5, distance_weight=0.95, percentile_cutoff=80
        )

        # Add text to GenBit for bias detection
        self._genbit_metrics.add_data([output], tokenized=False)

        # Generate gender bias metrics from GenBit
        metrics = self._genbit_metrics.get_metrics(
            output_statistics=True, output_word_list=True
        )
        log.debug(f"GenBit Bias metrics: {metrics}")

        # Tokenize and prepare input for the HuggingFace model
        # inputs = self._tokenizer(output, return_tensors="tf", padding=True, truncation=True)  # Use "tf" for TensorFlow
        # logits = self._bias_detection_model(inputs)[0]
        # bias_score = tf.nn.softmax(logits, axis=-1).numpy()[0]  # Assuming the model is multi-class

        # Check if bias is detected based on your criteria
        # You can customize this part based on your specific bias detection requirements
        bias_detected = False

        # Example: Check if bias is detected based on GenBit and HuggingFace model
        gender_metrics_score = metrics["genbit_score"]
        # Check your specific bias detection criteria here
        # Example: If the gender bias score is above a certain threshold and the model predicts bias, consider it as bias detected
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
