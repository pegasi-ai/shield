from typing import Protocol, Tuple


class Detector(Protocol):
    """
    Detector protocol defining the interface for detectors.

    Ensures detectors implement a common interface for interchangeable use.
    """

    def scan(self, prompt: str) -> Tuple[str, bool, float]:
        """
        Process and sanitize the input prompt according to the specific detector's implementation.

        Args:
            prompt (str): The input prompt to be processed.

        Returns:
            Tuple[str, bool, float]: A tuple containing:
                - str: The sanitized and processed prompt as per the detector's implementation.
                - bool: A flag indicating whether the prompt is valid or not.
                - float: Risk score where 0 means no risk and 1 means high risk.
        """
