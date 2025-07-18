import logging
import nltk
import re
import torch
import json
import importlib.util
from typing import List, Dict, Any, Tuple, Union, Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging

from .base_detector import Detector
from pegasi_shield.utils import device

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

transformers_logging.set_verbosity_error()
logger = logging.getLogger(__name__)

MAX_LENGTH = 512


class HallucinationDetector(Detector):
    """
    HallucinationDetector Class:

    This class checks for hallucinations in the output in relation to a given context using lettucedetect modernbert.
    """

    def __init__(self, threshold=0.5, model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"):
        """
        Initializes an instance of the HallucinationDetector class.

        Parameters:
            threshold (float): The threshold used to determine hallucinations. Defaults to 0.5.
            model_path (str): The path to the ModernBERT model. Defaults to "KRLabsOrg/lettucedect-base-modernbert-en-v1".
        """
        self._threshold = threshold
        self._model_path = model_path
        self._detector = None
        self._lettucedetect_available = True
        try:
            spec = importlib.util.find_spec("lettucedetect")
        except ImportError:
            self._lettucedetect_available = False
            logger.warning(
                "The lettucedetect library is not available. "
            )

        logger.debug(f"Initialized HallucinationDetector with model {self._model_path}")

    def _load_detector(self):
        """
        Lazily load the lettucedetect HallucinationDetector.
        
        Returns:
            bool: True if the detector was loaded successfully, False otherwise.
        """
        if self._detector is not None:
            return True
            
        if not self._lettucedetect_available:
            return False
            
        try:
            # Import the library only when needed
            from lettucedetect.models.inference import HallucinationDetector as LettuceDetector
            
            self._detector = LettuceDetector(
                method="transformer", 
                model_path=self._model_path
            )

            logger.debug(f"Initialized ModernBERT model {self._model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ModernBERT model: {e}")
            return False

    @staticmethod
    def clean_text(text):
        """
        Helper function to simplify capitalization, spacing, and punctuation in a given text using NLP.

        :param text: String containing the text to be simplified.
        :return: Simplified string.
        """
        text = text.lower()
        words = nltk.word_tokenize(text)
        simplified_text = " ".join(re.sub(r"\s+'", "'", word) for word in words)
        return simplified_text

    def scan(self, prompt: str, output: str, context: str) -> Tuple[str, bool, float]:
        """
        Scans the output for hallucinations based on the given context using ModernBERT.

        Parameters:
            prompt (str): The prompt string.
            output (str): The output string to evaluate.
            context (str): The context string to check against.

        Returns:
            A tuple (output, is_clean, hallucination_score).
            - output: The original output text.
            - is_clean: Boolean indicating if the output is free of hallucinations.
            - hallucination_score: A float between 0 and 1 indicating the hallucination score.
        """
        if not context or context.strip() == "":
            logger.warning("Empty context provided, skipping hallucination detection")
            return output, True, 0.0

        # Try to load the detector if it's not already loaded
        if not self._load_detector():
            logger.warning("ModernBERT detector is not available, skipping hallucination detection")
            return output, True, 0.0

        try:
            # Clean the text inputs
            context = self.clean_text(context)
            output = self.clean_text(output)
            
            # Get span-level predictions indicating which parts of the answer are considered hallucinated
            predictions = self._detector.predict(
                context=[context], 
                question=prompt, 
                answer=output, 
                output_format="spans"
            )
            
            # Process the predictions to get a hallucination score
            hallucination_score = self._calculate_hallucination_score(predictions, output)
            
            # Determine if the output is clean based on the threshold
            is_clean = hallucination_score <= self._threshold
            
            if not is_clean:
                logger.warning(
                    f"Detected hallucination in the output with score: {hallucination_score}, threshold: {self._threshold}"
                )
            else:
                logger.debug(
                    f"No hallucination detected in the output. Score: {hallucination_score}, threshold: {self._threshold}"
                )
                
            return output, is_clean, hallucination_score
            
        except Exception as e:
            logger.error(f"Error in HallucinationDetector.scan: {e}")
            # Return the original output with a default score in case of error
            return output, True, 0.0
    
    def _calculate_hallucination_score(self, predictions: Any, output: str) -> float:
        """
        Calculate a hallucination score based on the span predictions.
        
        Parameters:
            predictions (Any): The predictions from the lettucedetect model.
            output (str): The original output text.
            
        Returns:
            float: A score between 0 and 1 indicating the hallucination level.
        """
        if not self._detector:
            return 0.0  # Detector not available
            
        try:
            # Handle different prediction formats
            spans = []
            if isinstance(predictions, dict):
                spans = predictions.get("spans", [])
            elif isinstance(predictions, list):
                spans = predictions
            else:
                logger.warning(f"Unexpected predictions format: {type(predictions)}")
                return 0.0
            
            # Extract hallucinated spans from the predictions
            # In lettucedetect, spans with higher confidence are more likely to be hallucinated
            # The confidence threshold can be adjusted based on the use case
            hallucinated_spans = []
            confidence_threshold = 0.5  # Spans with confidence > 0.5 are considered hallucinated
            
            for pred in spans:
                if isinstance(pred, dict):
                    confidence = pred.get("confidence", 0.0)
                    if confidence > confidence_threshold:
                        start = pred.get("start", 0)
                        end = pred.get("end", 0)
                        hallucinated_spans.append((start, end))
                elif hasattr(pred, 'confidence') and pred.confidence > confidence_threshold:
                    # Handle object-based predictions
                    start = getattr(pred, 'start', 0)
                    end = getattr(pred, 'end', 0)
                    hallucinated_spans.append((start, end))
            
            # Calculate the proportion of hallucinated text
            if not hallucinated_spans:
                return 0.0  # No hallucinations detected
                
            total_chars = len(output)
            if total_chars == 0:
                return 0.0  # Empty output
                
            hallucinated_chars = sum(end - start for start, end in hallucinated_spans)
            hallucination_score = min(1.0, hallucinated_chars / total_chars)
            
            return hallucination_score
            
        except Exception as e:
            logger.error(f"Error calculating hallucination score: {e}")
            return 0.0  # Default to no hallucination in case of error




