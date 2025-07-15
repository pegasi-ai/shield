import logging
from pegasi_shield_safeguards.input_detectors.base_detector import Detector
from textstat import textstat

log = logging.getLogger(__name__)


class TextQualityInput(Detector):
    """
    A detector to assess the quality of text using various text quality metrics.
    """

    def __init__(self):
        self.logger = logging.getLogger("TextQuality")
        logging.basicConfig(level=logging.INFO)

    def _scale_to_01(self, value, min_value, max_value):
        """
        Scale a value to the range [0, 1].

        Parameters:
            value (float): The value to be scaled.
            min_value (float): The minimum possible value.
            max_value (float): The maximum possible value.

        Returns:
            float: The scaled value in the range [0, 1].
        """
        return (value - min_value) / (max_value - min_value)

    def _evaluate(self, prompt):
        text = prompt
        metrics = {
            "automated_readability_index": {
                "col_type": "String",
                "schema_name": None,
                "min": 1.0,
                "max": 14.0,
                "function": lambda text: textstat.automated_readability_index(text),
            },
            "dale_chall_readability_score": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "min": 0.0,
                "max": 9.9,
                "function": lambda text: textstat.dale_chall_readability_score(text),
            },
            "linsear_write_formula": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "min": 0.0,
                "max": 12.0,
                "function": lambda text: textstat.linsear_write_formula(text),
            },
            "gunning_fog": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "min": 1.0,
                "max": 17.0,
                "function": lambda text: textstat.gunning_fog(text),
            },
            "aggregate_reading_level": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: textstat.text_standard(
                    text, float_output=True
                ),
            },
            # Add more metrics here as needed
        }

        results = {}
        for metric, config in metrics.items():
            try:
                result = config["function"](text)
                self.logger.info(f"{metric}: {result}")
                # self.scale_to_01(result, min_value, max_value)  # Scale to [0, 1]
                results[metric] = result
            except Exception as e:
                self.logger.error(f"Error while evaluating {metric}: {e}")
                results[metric] = None

        return results

    def scan(self, prompt: str) -> (str, bool, float):
        # You can implement custom logic here to determine whether the text quality is acceptable or not
        # Example: If the average of scaled metrics is greater than 0.7, consider it high-quality text
        quality_metrics = self._evaluate(prompt)

        if quality_metrics["aggregate_reading_level"] > 20.0:
            is_high_quality = False
        elif quality_metrics["aggregate_reading_level"]:
            is_high_quality = True

        if is_high_quality:
            log.debug("Text is of high readability.")
            return prompt, True, quality_metrics
        else:
            log.warning("Text is of low readability.")
            return prompt, False, quality_metrics
