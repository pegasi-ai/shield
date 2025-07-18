import asyncio
import json
import logging
import re
import time
import numpy as np

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from ..base_detector import Detector


log = logging.getLogger(__name__)


class Equity(Detector):
    """
    A detector to detect and prevent toxic outputs using Prompt Guard model.
    """

    def __init__(self):
        """
        Initializes the EquityDetector
        """
        self.model_name = "gravitee-io/Llama-Prompt-Guard-2-22M-onnx"
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.model_name, 
            file_name="model.quant.onnx"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    async def create_chat_completion(self, messages, **kwargs):
        """
        Process the prompt through Prompt Guard model to detect harmful content
        """
        # Extract the content from messages
        content = messages[0]["content"] if messages else ""
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            content, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Run inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Convert to probabilities using softmax
        probs = np.exp(logits.numpy()) / np.sum(np.exp(logits.numpy()), axis=1, keepdims=True)
        
        # Get the harmful probability (assuming class 1 is harmful)
        harmful_prob = float(probs[0][1]) if probs.shape[1] > 1 else 0.0
        
        # Determine if content is harmful based on threshold
        is_harmful = harmful_prob > 0.5
        
        # Created a response in the expected JSON format
        equity_response = json.dumps({
            "harmful": is_harmful,
            "reasoning": f"Prompt Guard model detected {'harmful' if is_harmful else 'safe'} content with confidence {harmful_prob:.2f}",
            "confidence": harmful_prob
        })
        
        end_time = time.time()
        total_time = end_time - start_time
        log.info(f"Prompt Guard equity detector executed in {total_time:.4f} seconds")
        
        return equity_response

    async def analyze_equity(self, text: str) -> str:
        """
        Analyzes the equity of a given text using the ONNX model directly
        """
        # Create a message format for the model
        messages = [
            {
                "role": "user",
                "content": text,
            }
        ]
        response = await self.create_chat_completion(messages)
        return response

    def scan(self, prompt: str, response: str, context: str) -> (str, bool, float):
        """
        Scan the prompt and response for harmful content using only the ONNX model
        """
        if response.strip() == "" or prompt.strip() == "":
            return json.dumps({
                "harmful": False,
                "reasoning": "Empty prompt or response",
                "confidence": 0.0
            }), False, 0.0
        
        # Combine prompt and response for analysis
        input_text = f"prompt: {prompt}\nresponse: {response}"
        
        # Run the ONNX model analysis
        equity_response = asyncio.run(self.analyze_equity(input_text))
        
        # Parse the JSON response
        try:
            equity_response_dict = json.loads(equity_response)
            
            # Extract the values
            is_harmful = bool(equity_response_dict["harmful"])
            confidence = float(equity_response_dict["confidence"])
            
            if is_harmful:
                log.warning(f"Harm detected in the output with confidence {confidence:.2f}")
            else:
                log.debug(f"No harm detected in the output (confidence: {confidence:.2f})")
            
            return equity_response, is_harmful, confidence
            
        except Exception as e:
            log.error(f"Error decoding JSON response: {e}")
            # Return a safe default in case of error
            error_response = json.dumps({
                "harmful": True,
                "reasoning": f"Error processing response: {str(e)}",
                "confidence": 1.0
            })
            return error_response, True, 1.0