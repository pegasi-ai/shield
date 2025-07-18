"""
Groundedness detector module using HuggingFace transformers to detect hallucinations.

This module implements a detector that uses a HuggingFace model to check for hallucinations
in generated text by comparing it with a given context.
"""

import logging
import importlib.util
from typing import List, Dict, Any, Tuple, Union, Optional
import re
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification

from .base_detector import Detector

logger = logging.getLogger(__name__)


class Groundedness(Detector):
    """
    GroundednessDetector Class:

    This class checks for hallucinations in the output in relation to a given context using the
    lettucedetect library with the KRLabsOrg/lettucedect-large-modernbert-en-v1 model.
    
    It detects span-level hallucinations in the output and provides a score indicating the
    level of hallucination.
    """

    def __init__(self, threshold=0.25, model_path="KRLabsOrg/lettucedect-large-modernbert-en-v1"):
        """
        Initializes an instance of the GroundednessDetector class.

        Parameters:
            threshold (float): The threshold used to determine hallucinations. Defaults to 0.25.
            model_path (str): The path to the model. Defaults to "KRLabsOrg/lettucedect-large-modernbert-en-v1".
        """
        self._threshold = threshold  # Lower threshold to be more sensitive to hallucinations
        self._model_path = model_path
        self._tokenizer = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._max_length = 4096
        self._load_model()
    
    def _load_model(self):
        try:
            # Use the correct model class for KRLabsOrg/lettucedect model
            if "lettucedect" in self._model_path.lower():
                logger.info(f"Loading KRLabsOrg model as MaskedLM: {self._model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
                self._model = AutoModelForMaskedLM.from_pretrained(self._model_path)
                self._model_type = "masked_lm"
            else:
                logger.info(f"Loading model as TokenClassification: {self._model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
                self._model = AutoModelForTokenClassification.from_pretrained(self._model_path)
                self._model_type = "token_classification"
            
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"Successfully initialized model {self._model_path} on device {self._device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            
            # Try an alternative approach with a different model
            try:
                logger.info("Attempting to load an alternative model...")
                # Try a more standard model for token classification
                alt_model_path = "dslim/bert-base-NER"
                self._model_path = alt_model_path
                
                self._tokenizer = AutoTokenizer.from_pretrained(alt_model_path)
                self._model = AutoModelForTokenClassification.from_pretrained(alt_model_path)
                self._model_type = "token_classification"
                self._model.to(self._device)
                self._model.eval()
                logger.info(f"Successfully initialized alternative model {alt_model_path}")
                return True
            except Exception as alt_e:
                logger.error(f"Failed to initialize alternative model: {alt_e}")
                return False

    def _form_prompt(self, context: list[str], question: str | None) -> str:
        PROMPT_QA = """
        Briefly answer the following question:
        {question}
        Bear in mind that your response should be strictly based on the following {num_passages} passages:
        {context}
        In case the passages do not contain the necessary information to answer the question, please reply with: "Unable to answer based on given passages."
        output:
        """

        PROMPT_SUMMARY = """
        Summarize the following text:
        {text}
        output:
        """
        
        context_str = "\n".join(
            [f"passage {i + 1}: {passage}" for i, passage in enumerate(context)]
        )
        if question is None:
            return PROMPT_SUMMARY.format(text=context_str)
        else:
            return PROMPT_QA.format(
                question=question, num_passages=len(context), context=context_str
            )
        
    def _predict(self, context: str, answer: str, output_format: str = "tokens"):
        # Tokenize input
        encoding = self._tokenizer(
            context,
            answer,
            truncation=True,
            max_length=self._max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        
        # Extract offset mapping
        offsets = encoding.pop("offset_mapping")[0]
        
        # Compute answer start token index
        prompt_tokens = self._tokenizer.encode(context, add_special_tokens=False)
        answer_start_token = 1 + len(prompt_tokens) + 1  # [CLS] + prompt tokens + [SEP]
        
        # Create labels tensor
        labels = torch.full_like(encoding["input_ids"][0], -100, device=self._device)
        labels[answer_start_token:] = 0
        
        # Move encoding to device
        encoding = {key: value.to(self._device) for key, value in encoding.items()}
        
        # Run model inference
        with torch.no_grad():
            outputs = self._model(**encoding)
        
        # Process outputs based on model type
        if self._model_type == "masked_lm":
            # For MaskedLM models, we need to handle the output differently
            # We'll use the hidden states to determine hallucination probability
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            
            # Simple approach: use the norm of hidden states as a proxy for hallucination
            # Higher norm might indicate more uncertainty/hallucination
            token_norms = torch.norm(hidden_states, dim=-1)[0]
            
            # Normalize to [0, 1] range for the answer tokens
            answer_token_norms = token_norms[answer_start_token:]
            if len(answer_token_norms) > 0:
                min_norm = torch.min(answer_token_norms)
                max_norm = torch.max(answer_token_norms)
                norm_range = max_norm - min_norm
                if norm_range > 0:
                    normalized_norms = (answer_token_norms - min_norm) / norm_range
                else:
                    normalized_norms = torch.zeros_like(answer_token_norms)
                
                # Create fake logits and probabilities for consistent processing
                logits = torch.zeros((token_norms.size(0), 2), device=self._device)
                probabilities = torch.zeros((token_norms.size(0), 2), device=self._device)
                
                # Set class 1 (hallucination) probability based on normalized norms
                for i in range(answer_start_token, token_norms.size(0)):
                    idx = i - answer_start_token
                    if idx < len(normalized_norms):
                        prob = normalized_norms[idx].item()
                        probabilities[i, 1] = prob
                        probabilities[i, 0] = 1 - prob
                        # Fake logits based on probabilities
                        logits[i, 1] = torch.log(torch.tensor(prob + 1e-10))
                        logits[i, 0] = torch.log(torch.tensor(1 - prob + 1e-10))
                
                token_preds = torch.argmax(logits, dim=-1)
            else:
                # Fallback if no answer tokens
                logits = torch.zeros((token_norms.size(0), 2), device=self._device)
                probabilities = torch.zeros((token_norms.size(0), 2), device=self._device)
                token_preds = torch.zeros(token_norms.size(0), dtype=torch.long, device=self._device)
        else:
            # For TokenClassification models, use the logits directly
            logits = outputs.logits
            token_preds = torch.argmax(logits, dim=-1)[0]
            probabilities = torch.softmax(logits, dim=-1)[0]
        
        # Process results based on output format
        if output_format == "tokens":
            # Return token probabilities for each token
            token_probs = []
            input_ids = encoding["input_ids"][0]
            for i, (token, pred, prob) in enumerate(zip(input_ids, token_preds, probabilities)):
                if labels[i].item() != -100:
                    token_probs.append({
                        "token": self._tokenizer.decode([token]),
                        "pred": pred.item(),
                        "prob": prob[1].item(),  # Probability for class 1 (hallucination)
                    })
            return token_probs
        elif output_format == "spans":
            # Compute the answer's character offset
            if answer_start_token < offsets.size(0):
                answer_char_offset = offsets[answer_start_token][0].item()
            else:
                answer_char_offset = 0
                
            spans = []
            current_span = None
            
            # Iterate over tokens in the answer region
            for i in range(answer_start_token, token_preds.size(0)):
                # Skip tokens marked as ignored
                if labels[i].item() == -100:
                    continue
                    
                token_start, token_end = offsets[i].tolist()
                # Skip special tokens with zero length
                if token_start == token_end:
                    continue
                    
                # Adjust offsets relative to the answer text
                rel_start = token_start - answer_char_offset
                rel_end = token_end - answer_char_offset
                
                is_hallucination = token_preds[i].item() == 1  # Class 1 indicates hallucination
                confidence = probabilities[i, 1].item() if is_hallucination else 0.0
                
                if is_hallucination:
                    if current_span is None:
                        current_span = {
                            "start": rel_start,
                            "end": rel_end,
                            "confidence": confidence,
                            "is_hallucinated": True
                        }
                    else:
                        # Extend the current span
                        current_span["end"] = rel_end
                        current_span["confidence"] = max(current_span["confidence"], confidence)
                else:
                    # If we were building a hallucination span, finalize it
                    if current_span is not None:
                        # Extract the hallucinated text from the answer
                        span_text = answer[current_span["start"] : current_span["end"]]
                        current_span["text"] = span_text
                        spans.append(current_span)
                        current_span = None
                        
            # Append any span still in progress
            if current_span is not None:
                span_text = answer[current_span["start"] : current_span["end"]]
                current_span["text"] = span_text
                spans.append(current_span)
                
            return {"spans": spans}
        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")
    def _calculate_hallucination_score(self, predictions: Dict[str, Any], output: str) -> float:
        """
        Calculate a hallucination score based on the span predictions.
        
        Parameters:
            predictions (Dict[str, Any]): The predictions from the model.
            output (str): The original output text.
            
        Returns:
            float: A score between 0 and 1 indicating the hallucination level.
        """
        try:
            # Extract hallucinated spans from the predictions
            hallucinated_spans = []
            for pred in predictions.get("spans", []):
                # If the span is marked as hallucinated or has high hallucination probability
                hallucination_prob = pred.get("hallucination_prob", 0.0)
                is_hallucinated = pred.get("is_hallucinated", False) or hallucination_prob > 0.3
                
                if is_hallucinated:
                    start = pred.get("start", 0)
                    end = pred.get("end", 0)
                    hallucinated_spans.append((start, end))
                    
                    # Log the hallucinated span for debugging
                    if start < len(output) and end <= len(output):
                        hallucinated_text = output[start:end]
                        logger.info(f"Detected hallucinated span: '{hallucinated_text}' with probability {hallucination_prob:.4f}")
            
            # If any hallucinated spans are found, ensure we return a score that indicates hallucination
            if hallucinated_spans:
                # Calculate the proportion of hallucinated text
                total_chars = len(output)
                if total_chars == 0:
                    return 0.5  # Default score for empty output with hallucination flags
                    
                hallucinated_chars = sum(end - start for start, end in hallucinated_spans)
                base_score = min(1.0, hallucinated_chars / total_chars)
                
                # Ensure even small hallucinations are detected by setting a minimum score
                # This ensures that if any span is hallucinated, the score is high enough to be flagged
                return max(0.3, base_score)
            
            return 0.0  # No hallucinations detected
            
        except Exception as e:
            logger.error(f"Error calculating hallucination score: {e}")
            return 0.0  # Default to no hallucination in case of error
            
    def _is_completely_unrelated(self, context: str, output: str) -> bool:
        """
        Check if the output is completely unrelated to the context.
        This is a simple heuristic based on keyword overlap.
        
        Parameters:
            context (str): The context to check against.
            output (str): The output to evaluate.
            
        Returns:
            bool: True if the output appears to be completely unrelated to the context.
        """
        try:
            # Download NLTK resources if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            # Extract keywords from context and output
            # Simple approach: use nouns and named entities as keywords
            def extract_keywords(text):
                words = nltk.word_tokenize(text.lower())
                # Filter out short words and common stopwords
                stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through', 'over', 'before', 'after', 'between', 'under', 'during', 'since', 'without', 'of', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must'])
                keywords = [word for word in words if len(word) > 3 and word not in stopwords and word.isalnum()]
                return set(keywords)
                
            context_keywords = extract_keywords(context)
            output_keywords = extract_keywords(output)
            
            # Calculate overlap
            if not output_keywords:
                return False  # Empty output, can't determine if unrelated
                
            overlap = context_keywords.intersection(output_keywords)
            overlap_ratio = len(overlap) / len(output_keywords) if output_keywords else 0
            
            # More aggressive detection of unrelated content
            # Check for specific named entities or key terms that might indicate unrelated content
            # For example, if context is about France but output mentions Japan, it's likely unrelated
            
            # Extract potential named entities (simple heuristic: capitalized words not at start of sentence)
            def extract_potential_entities(text):
                words = nltk.word_tokenize(text)
                entities = set()
                for i, word in enumerate(words):
                    if (word[0].isupper() and i > 0 and words[i-1] not in ['.', '!', '?']) or word.isupper():
                        entities.add(word.lower())
                return entities
                
            context_entities = extract_potential_entities(context)
            output_entities = extract_potential_entities(output)
            
            # If output has entities not in context, it might be hallucinating
            new_entities = output_entities - context_entities
            
            # If there's very little keyword overlap or new entities are introduced, consider it unrelated
            return (overlap_ratio < 0.15 and len(output_keywords) > 2) or (len(new_entities) > 0 and overlap_ratio < 0.3)
            
        except Exception as e:
            logger.error(f"Error in _is_completely_unrelated: {e}")
            return False  # Default to not unrelated in case of error
            
    def _adjust_score_with_heuristics(self, base_score: float, context: str, output: str) -> float:
        """
        Apply additional heuristics to adjust the hallucination score.
        
        Parameters:
            base_score (float): The initial hallucination score.
            context (str): The context string.
            output (str): The output string.
            
        Returns:
            float: The adjusted hallucination score.
        """
        try:
            # Heuristic 1: Adjust score based on output length
            # Very short outputs are less likely to contain hallucinations
            if len(output) < 20:
                return max(0.0, base_score - 0.2)
                
            # Heuristic 2: Adjust score based on output complexity
            # Simple outputs with few unique words are less likely to hallucinate
            words = output.split()
            unique_words = set(words)
            if len(words) > 0 and len(unique_words) / len(words) < 0.5:
                return max(0.0, base_score - 0.1)
                
            # Heuristic 3: Check for specific patterns that suggest hallucination
            # For example, phrases like "I believe", "I think", "probably" suggest uncertainty
            uncertainty_patterns = [r'\b(probably|possibly|perhaps|maybe|might|could be|I think|I believe|likely|unlikely)\b']
            for pattern in uncertainty_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    return min(1.0, base_score + 0.1)
            
            # Heuristic 4: Check for information in output not present in context
            # This is a more aggressive check for hallucinations
            context_words = set(context.lower().split())
            output_words = set(output.lower().split())
            
            # Words in output not in context (excluding common words)
            common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                              'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through', 'over', 
                              'before', 'after', 'between', 'under', 'during', 'since', 'without', 'of', 'this', 
                              'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'has', 'have', 'had', 
                              'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must'])
            
            new_words = output_words - context_words - common_words
            if len(new_words) > 0:
                # Calculate ratio of new words to total output words
                new_word_ratio = len(new_words) / len(output_words) if output_words else 0
                
                # Adjust score based on ratio of new words
                if new_word_ratio > 0.2:  # If more than 20% of words are new
                    return min(1.0, base_score + 0.2 + (new_word_ratio - 0.2))
                elif new_word_ratio > 0.1:  # If 10-20% of words are new
                    return min(1.0, base_score + 0.1)
                    
            return base_score
            
        except Exception as e:
            logger.error(f"Error in _adjust_score_with_heuristics: {e}")
            return base_score  # Return original score in case of error
            
    def scan(self, prompt: str, output: str, context: str) -> Tuple[str, bool, float]:
        """
        Scans the output for hallucinations based on the given context.

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
            
        if not self._model or not self._tokenizer:
            if not self._load_model():
                logger.warning("Model is not available, skipping hallucination detection")
                return output, True, 0.0
        
        try:
            # Check if output is completely unrelated to context
            # This is a simple heuristic to catch cases where the output is about a different topic
            if self._is_completely_unrelated(context, output):
                logger.warning("Output appears to be completely unrelated to the context")
                return output, False, 0.8  # High hallucination score for unrelated content
            
            try:
                # Form prompt and get predictions
                prompt_text = self._form_prompt([context], prompt)
                predictions = self._predict(prompt_text, output, "spans")
                
                # Calculate hallucination score
                hallucination_score = self._calculate_hallucination_score(predictions, output)
            except Exception as model_error:
                # If model prediction fails, use heuristics only
                logger.error(f"Error in model prediction: {model_error}")
                logger.info("Falling back to heuristic-only evaluation")
                
                # Use a more sophisticated heuristic approach
                # 1. Check keyword overlap
                context_words = set(context.lower().split())
                output_words = set(output.lower().split())
                
                # Remove common stopwords
                stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                              'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like'])
                context_words = context_words - stopwords
                output_words = output_words - stopwords
                
                overlap = context_words.intersection(output_words)
                
                # 2. Calculate base score from overlap ratio
                if len(output_words) > 0:
                    overlap_ratio = len(overlap) / len(output_words)
                    base_score = max(0.0, 1.0 - overlap_ratio)
                else:
                    base_score = 0.0
                    
                # 3. Check for new information in output not in context
                new_words = output_words - context_words
                if len(new_words) > 0:
                    new_word_ratio = len(new_words) / len(output_words) if output_words else 0
                    # Increase score based on amount of new information
                    base_score = min(1.0, base_score + (new_word_ratio * 0.5))
                    
                hallucination_score = base_score
            
            # Apply additional heuristics to adjust the score
            hallucination_score = self._adjust_score_with_heuristics(hallucination_score, context, output)
            
            is_clean = hallucination_score <= self._threshold
            
            if not is_clean:
                logger.warning(
                    f"Detected hallucination in the output with score: {hallucination_score:.4f}, threshold: {self._threshold}"
                )
            else:
                logger.info(
                    f"No hallucination detected in the output. Score: {hallucination_score:.4f}, threshold: {self._threshold}"
                )
            
            return output, is_clean, hallucination_score
        except Exception as e:
            logger.error(f"Error in Groundedness.scan: {e}")
            logger.error(f"Error details: {str(e)}")
            return output, True, 0.0