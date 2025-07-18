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
from ..utils import device

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

transformers_logging.set_verbosity_error()
logger = logging.getLogger(__name__)

MAX_LENGTH = 512


class HallucinationDetector(Detector):
    """
    HallucinationDetector Class:

    This class checks for hallucinations in the output in relation to a given context using a pretrained model.
    """

    def __init__(self, threshold=0.5):
        """
        Initializes an instance of the HallucinationDetector class.

        Parameters:
            threshold (float): The threshold used to determine hallucinations. Defaults to 0.5.
        """

        self._model_path = "vectara/hallucination_evaluation_model"
        self._tokenizer_path = "google/flan-t5-base"

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        self._model.eval()
        self._model.to(device)

        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        self._threshold = threshold

        logger.debug(f"Initialized model {self._model_path} on device {device}")
        logger.debug(f"Initialized tokenizer {self._tokenizer_path} on device {device}")

    def clean_text(self, text):
        """
        Function to simplify capitalization, spacing, and punctuation in a given text using NLP.

        :param text: String containing the text to be simplified.
        :return: Simplified string.
        """
        # Convert text to lowercase
        text = text.lower()

        # Tokenize the text into sentences
        sentences = nltk.tokenize.sent_tokenize(text)

        simplified_sentences = []
        for sentence in sentences:
            # Tokenize the sentence into words
            words = nltk.tokenize.word_tokenize(sentence)

            # Join the words back into a sentence with single space
            simplified_sentence = " ".join(words)

            # Remove spaces around apostrophes
            simplified_sentence = re.sub(r"\s+('\w+)\s+", r"\1 ", simplified_sentence)

            simplified_sentences.append(simplified_sentence)

        # Join the sentences back into text with single space
        simplified_text = " ".join(simplified_sentences)

        return simplified_text

    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        """
        Scans the output for hallucinations based on the given context.

        Parameters:
            context (str): The context string.
            output (str): The output string to evaluate.

        Returns:
            A tuple (output, is_hallucination, hallucination_score).
        """
        if context.strip() == "":
            return output, True, 0.0

        context = self.clean_text(context)
        output = self.clean_text(output)

        # Prepare input for the model
        inputs = self._tokenizer.encode_plus(
            context,
            output,
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Run the model
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits.cpu().detach().flatten()
            scores = torch.sigmoid(logits)  # Convert logits to probabilities
            scores = scores.tolist()  # Convert torch tensor to standard Python float

        # Assume the first score corresponds to hallucination
        # Inverse score as 0 is hallucination, 1 is truthful statement
        hallucination_score = 1 - scores[0]  # 1: hallucination, 0: truthful statement

        if hallucination_score > self._threshold:
            logger.warning(
                f"Detected hallucination in the output with score: {hallucination_score}, threshold: {self._threshold}"
            )
            return output, False, hallucination_score

        logger.debug(
            f"No hallucination detected in the output. Score: {hallucination_score}, threshold: {self._threshold}"
        )

        return output, True, hallucination_score


class ModernBERTGroundednessDetector(Detector):
    """
    ModernBERTGroundednessDetector Class:

    This class checks for hallucinations in the output in relation to a given context using the
    ModernBERT model from KRLabsOrg/lettucedect-large-modernbert-en-v1.
    
    It uses the lettucedetect library to detect span-level hallucinations in the output.
    The library is loaded lazily to avoid dependency issues when not in use.
    """

    def __init__(self, threshold=0.5, model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"):
        """
        Initializes an instance of the ModernBERTGroundednessDetector class.

        Parameters:
            threshold (float): The threshold used to determine hallucinations. Defaults to 0.5.
            model_path (str): The path to the ModernBERT model. Defaults to "KRLabsOrg/lettucedect-base-modernbert-en-v1".
        """
        self._threshold = threshold
        self._model_path = model_path
        self._detector = None
        self._lettucedetect_available = self._check_lettucedetect_available()
        
        if not self._lettucedetect_available:
            logger.warning(
                "The lettucedetect library is not available. "
                "Please install it with 'pip install lettucedetect' to use the ModernBERTGroundednessDetector."
            )
    
    def _check_lettucedetect_available(self):
        """
        Check if the lettucedetect library is available.
        
        Returns:
            bool: True if the library is available, False otherwise.
        """
        try:
            spec = importlib.util.find_spec("lettucedetect")
            return spec is not None
        except ImportError:
            return False
    
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

    def scan(self, prompt: str, output: str, context: str) -> Tuple[str, bool, float]:
        """
        Scans the output for hallucinations based on the given context using ModernBERT.

        Parameters:
            prompt (str): The prompt string (not used in this implementation).
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
                # You could optionally highlight or mark the hallucinated spans in the output
                # marked_output = self._mark_hallucinated_spans(output, predictions)
                # return marked_output, is_clean, hallucination_score
            else:
                logger.debug(
                    f"No hallucination detected in the output. Score: {hallucination_score}, threshold: {self._threshold}"
                )
                
            return output, is_clean, hallucination_score
            
        except Exception as e:
            logger.error(f"Error in ModernBERTGroundednessDetector.scan: {e}")
            # Return the original output with a default score in case of error
            return output, True, 0.0
    
    def _calculate_hallucination_score(self, predictions: Dict[str, Any], output: str) -> float:
        """
        Calculate a hallucination score based on the span predictions.
        
        Parameters:
            predictions (Dict[str, Any]): The predictions from the lettucedetect model.
            output (str): The original output text.
            
        Returns:
            float: A score between 0 and 1 indicating the hallucination level.
        """
        if not self._detector:
            return 0.0  # Detector not available
            
        try:
            # Extract hallucinated spans from the predictions
            hallucinated_spans = []
            for pred in predictions.get("spans", []):
                if pred.get("is_hallucinated", False):
                    hallucinated_spans.append((pred.get("start", 0), pred.get("end", 0)))
            
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
    
    def _mark_hallucinated_spans(self, output: str, predictions: Dict[str, Any]) -> str:
        """
        Mark hallucinated spans in the output text.
        
        Parameters:
            output (str): The original output text.
            predictions (Dict[str, Any]): The predictions from the lettucedetect model.
            
        Returns:
            str: The output text with hallucinated spans marked.
        """
        if not self._detector:
            return output  # Detector not available
            
        try:
            # Create a list of characters from the output
            chars = list(output)
            
            # Mark the beginning and end of each hallucinated span
            for pred in predictions.get("spans", []):
                if pred.get("is_hallucinated", False):
                    start = pred.get("start", 0)
                    end = pred.get("end", 0)
                    
                    # Ensure indices are within bounds
                    if 0 <= start < len(chars):
                        chars[start] = f"[HALLUCINATION_START]{chars[start]}"
                    if 0 <= end < len(chars):
                        chars[end-1] = f"{chars[end-1]}[HALLUCINATION_END]"
            
            # Join the characters back into a string
            marked_output = "".join(chars)
            return marked_output
            
        except Exception as e:
            logger.error(f"Error marking hallucinated spans: {e}")
            return output  # Return the original output in case of error
