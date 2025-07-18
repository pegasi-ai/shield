from typing import Protocol, Tuple


class Detector(Protocol):
    """
    Detector protocol defining the interface for detectors.

    Ensures detectors implement a common interface for interchangeable use.
    """

    def scan(
        self, prompt: str, output: str, context: str = None
    ) -> Tuple[str, bool, float]:
        """
        Process and sanitize the output according to the specific detector's implementation.

        Args:
            out (str): The output to be processed.

        Returns:
            Tuple[str, bool, float]: A tuple containing:
                - str: The sanitized and processed output as per the detector's implementation.
                - bool: A flag indicating whether the output is valid or not.
                - float: Risk score where 0 means no risk and 1 means high risk.
        """
