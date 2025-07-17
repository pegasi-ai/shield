from typing import Dict, List, Optional, Union
import logging

from pegasi_shield.input_detectors import (
    LanguageDetection,
    Secrets,
    TextQualityInput,
    ToxicityInput,
)
from pegasi_shield.output_detectors import (
    Bias,
    Contradictions,
    Deanonymize,
    FactualConsistency,
    HallucinationDetector,
    HarmfulOutput,
    PromptOutputRelevance,
    StopOutputSubstrings,
    TemporalMismatchDetector,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeguardsPipeline:
    def __init__(self):
        # Initialize input detectors
        self.language_detector = LanguageDetection()
        self.secrets_detector = Secrets()
        self.text_quality = TextQualityInput()
        self.toxicity = ToxicityInput()

        # Initialize output detectors
        self.bias_detector = Bias()
        self.contradiction_detector = Contradictions()
        self.factual_consistency = FactualConsistency()
        self.hallucination_detector = HallucinationDetector()
        self.harmful_output = HarmfulOutput()
        self.relevance = PromptOutputRelevance()
        self.temporal = TemporalMismatchDetector()
        
        logger.info("Initialized all detectors")

    def scan_input(self, prompt: str, allowed_languages: Optional[List[str]] = None) -> Dict[str, Union[bool, float]]:
        """
        Scan input prompt for potential issues.
        
        Args:
            prompt: Input text to analyze
            allowed_languages: List of allowed programming languages
            
        Returns:
            Dictionary containing scan results from each detector
        """
        results = {}
        
        # Language detection
        if allowed_languages:
            lang_result = self.language_detector.scan(prompt, allowed=allowed_languages)
            results["language"] = lang_result
            
        # Secrets detection
        secrets_result = self.secrets_detector.scan(prompt)
        results["secrets"] = secrets_result
        
        # Text quality
        quality_result = self.text_quality.scan(prompt)
        results["text_quality"] = quality_result
        
        # Toxicity
        toxicity_result = self.toxicity.scan(prompt)
        results["toxicity"] = toxicity_result
        
        return results

    def scan_output(
        self, 
        prompt: str, 
        output: str, 
        context: str = "",
        stop_substrings: Optional[List[str]] = None
    ) -> Dict[str, Union[bool, float]]:
        """
        Scan generated output for potential issues.
        
        Args:
            prompt: Original input prompt
            output: Generated output to analyze
            context: Additional context for relevance checking
            stop_substrings: List of substrings to block in output
            
        Returns:
            Dictionary containing scan results from each detector
        """
        results = {}
        
        # Check bias
        bias_result = self.bias_detector.scan(output)
        results["bias"] = bias_result
        
        # Check contradictions
        contradiction_result = self.contradiction_detector.scan(prompt, output, context)
        results["contradictions"] = contradiction_result
        
        # Check factual consistency
        factual_result = self.factual_consistency.scan(prompt, output)
        results["factual"] = factual_result
        
        # Check hallucinations
        hallucination_result = self.hallucination_detector.scan(prompt, output)
        results["hallucinations"] = hallucination_result
        
        # Check harmful content
        harmful_result = self.harmful_output.scan(output)
        results["harmful"] = harmful_result
        
        # Check relevance
        relevance_result = self.relevance.scan(prompt, output, context)
        results["relevance"] = relevance_result
        
        # Check temporal consistency
        temporal_result = self.temporal.scan(output)
        results["temporal"] = temporal_result
        
        # Check stop substrings if provided
        if stop_substrings:
            stop_detector = StopOutputSubstrings(substrings=stop_substrings)
            stop_result = stop_detector.scan(output)
            results["stop_substrings"] = stop_result
            
        return results

if __name__ == "__main__":
    # Example usage
    pipeline = SafeguardsPipeline()
    
    # Example input scan
    input_text = "Here is some input text"
    input_results = pipeline.scan_input(input_text, allowed_languages=["python", "javascript"])
    logger.info(f"Input scan results: {input_results}")
    
    # Example output scan
    output_text = "Here is some output text"
    output_results = pipeline.scan_output(
        prompt=input_text,
        output=output_text,
        context="Additional context",
        stop_substrings=["forbidden", "blocked"]
    )
    logger.info(f"Output scan results: {output_results}")
