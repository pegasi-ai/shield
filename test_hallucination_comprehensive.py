#!/usr/bin/env python3
"""
Comprehensive test script for HallucinationDetector to verify functionality step by step.
This test covers all aspects of the scan function with different scenarios.
"""

import sys
import os
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path so we can import pegasi_shield
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pegasi_shield.output_detectors.hallucinations import HallucinationDetector, ModernBERTGroundednessDetector

# Set up logging for better visibility
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestHallucinationDetector:
    """Test suite for HallucinationDetector class"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.detector = HallucinationDetector(threshold=0.5)
    
    def test_initialization(self):
        """Test that HallucinationDetector initializes properly"""
        logger.info("Testing HallucinationDetector initialization")
        
        # Test default initialization
        detector = HallucinationDetector()
        assert detector._threshold == 0.5
        assert detector._model_path == "KRLabsOrg/lettucedect-base-modernbert-en-v1"
        assert detector._detector is None
        
        # Test custom initialization
        custom_detector = HallucinationDetector(threshold=0.3, model_path="custom/model")
        assert custom_detector._threshold == 0.3
        assert custom_detector._model_path == "custom/model"
        
        logger.info("✓ Initialization test passed")
    
    def test_scan_empty_context(self):
        """Test scan function with empty context"""
        logger.info("Testing scan with empty context")
        
        prompt = "What is the capital of France?"
        output = "The capital of France is Paris."
        context = ""
        
        result_output, is_clean, score = self.detector.scan(prompt, output, context)
        
        assert result_output == output
        assert is_clean == True
        assert score == 0.0
        
        logger.info("✓ Empty context test passed")
    
    def test_scan_none_context(self):
        """Test scan function with None context"""
        logger.info("Testing scan with None context")
        
        prompt = "What is the capital of France?"
        output = "The capital of France is Paris."
        context = None
        
        result_output, is_clean, score = self.detector.scan(prompt, output, context)
        
        assert result_output == output
        assert is_clean == True
        assert score == 0.0
        
        logger.info("✓ None context test passed")
    
    def test_scan_whitespace_context(self):
        """Test scan function with whitespace-only context"""
        logger.info("Testing scan with whitespace context")
        
        prompt = "What is the capital of France?"
        output = "The capital of France is Paris."
        context = "   \t\n  "
        
        result_output, is_clean, score = self.detector.scan(prompt, output, context)
        
        assert result_output == output
        assert is_clean == True
        assert score == 0.0
        
        logger.info("✓ Whitespace context test passed")
    
    @patch('pegasi_shield.output_detectors.hallucinations.importlib.util.find_spec')
    def test_scan_without_lettucedetect(self, mock_find_spec):
        """Test scan function when lettucedetect is not available"""
        logger.info("Testing scan without lettucedetect library")
        
        # Mock that lettucedetect is not available
        mock_find_spec.return_value = None
        
        detector = HallucinationDetector()
        prompt = "What is the capital of France?"
        output = "The capital of France is Paris."
        context = "France is a country in Europe."
        
        result_output, is_clean, score = detector.scan(prompt, output, context)
        
        assert result_output == output
        assert is_clean == True
        assert score == 0.0
        
        logger.info("✓ No lettucedetect library test passed")
    
    @patch('pegasi_shield.output_detectors.hallucinations.HallucinationDetector._load_detector')
    def test_scan_detector_load_failure(self, mock_load_detector):
        """Test scan function when detector fails to load"""
        logger.info("Testing scan with detector load failure")
        
        mock_load_detector.return_value = False
        
        prompt = "What is the capital of France?"
        output = "The capital of France is Paris."
        context = "France is a country in Europe."
        
        result_output, is_clean, score = self.detector.scan(prompt, output, context)
        
        assert result_output == output
        assert is_clean == True
        assert score == 0.0
        
        logger.info("✓ Detector load failure test passed")
    
    @patch('pegasi_shield.output_detectors.hallucinations.HallucinationDetector._load_detector')
    @patch('pegasi_shield.output_detectors.hallucinations.HallucinationDetector.clean_text')
    def test_scan_with_mocked_detector(self, mock_clean_text, mock_load_detector):
        """Test scan function with mocked detector and text cleaning"""
        logger.info("Testing scan with mocked detector")
        
        # Mock successful detector loading
        mock_load_detector.return_value = True
        
        # Mock text cleaning
        mock_clean_text.side_effect = lambda x: x.lower().strip()
        
        # Mock the detector
        mock_detector = Mock()
        mock_predictions = {
            "spans": [
                {"start": 0, "end": 5, "is_hallucinated": True},
                {"start": 10, "end": 15, "is_hallucinated": False}
            ]
        }
        mock_detector.predict.return_value = mock_predictions
        self.detector._detector = mock_detector
        
        prompt = "What is the capital of France?"
        output = "Berlin is the capital of France."
        context = "France is a country in Europe. Paris is the capital of France."
        
        result_output, is_clean, score = self.detector.scan(prompt, output, context)
        
        # Verify detector was called with correct parameters
        mock_detector.predict.assert_called_once_with(
            context=[context.lower().strip()],
            question=prompt,
            answer=output.lower().strip(),
            output_format="spans"
        )
        
        assert result_output == output.lower().strip()
        assert isinstance(is_clean, bool)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        logger.info("✓ Mocked detector test passed")
    
    def test_calculate_hallucination_score_no_hallucinations(self):
        """Test _calculate_hallucination_score with no hallucinations"""
        logger.info("Testing hallucination score calculation with no hallucinations")
        
        # Mock detector to be available
        self.detector._detector = Mock()
        
        predictions = {
            "spans": [
                {"start": 0, "end": 5, "is_hallucinated": False},
                {"start": 10, "end": 15, "is_hallucinated": False}
            ]
        }
        output = "Paris is the capital of France."
        
        score = self.detector._calculate_hallucination_score(predictions, output)
        
        assert score == 0.0
        
        logger.info("✓ No hallucinations score test passed")
    
    def test_calculate_hallucination_score_with_hallucinations(self):
        """Test _calculate_hallucination_score with hallucinations"""
        logger.info("Testing hallucination score calculation with hallucinations")
        
        # Mock detector to be available
        self.detector._detector = Mock()
        
        predictions = {
            "spans": [
                {"start": 0, "end": 5, "is_hallucinated": True},  # 5 chars
                {"start": 10, "end": 15, "is_hallucinated": True}  # 5 chars
            ]
        }
        output = "0123456789012345678901234567890"  # 30 chars
        
        score = self.detector._calculate_hallucination_score(predictions, output)
        
        # 10 hallucinated chars out of 30 total = 10/30 = 0.333...
        expected_score = 10.0 / 30.0
        assert abs(score - expected_score) < 0.001
        
        logger.info("✓ With hallucinations score test passed")
    
    def test_calculate_hallucination_score_empty_output(self):
        """Test _calculate_hallucination_score with empty output"""
        logger.info("Testing hallucination score calculation with empty output")
        
        # Mock detector to be available
        self.detector._detector = Mock()
        
        predictions = {
            "spans": [
                {"start": 0, "end": 5, "is_hallucinated": True}
            ]
        }
        output = ""
        
        score = self.detector._calculate_hallucination_score(predictions, output)
        
        assert score == 0.0
        
        logger.info("✓ Empty output score test passed")
    
    def test_calculate_hallucination_score_no_detector(self):
        """Test _calculate_hallucination_score when detector is not available"""
        logger.info("Testing hallucination score calculation without detector")
        
        # Ensure detector is None
        self.detector._detector = None
        
        predictions = {"spans": []}
        output = "Some output text"
        
        score = self.detector._calculate_hallucination_score(predictions, output)
        
        assert score == 0.0
        
        logger.info("✓ No detector score test passed")
    
    def test_clean_text_function(self):
        """Test the clean_text static method"""
        logger.info("Testing clean_text function")
        
        # Test various text inputs
        test_cases = [
            ("Hello World!", "hello world !"),
            ("  MULTIPLE   SPACES  ", "multiple spaces"),
            ("It's a beautiful day", "it 's a beautiful day"),
            ("123 Numbers and Text!", "123 numbers and text !"),
            ("", "")
        ]
        
        for input_text, expected in test_cases:
            result = HallucinationDetector.clean_text(input_text)
            assert result == expected, f"Failed for input: '{input_text}'. Expected: '{expected}', Got: '{result}'"
        
        logger.info("✓ Clean text function test passed")
    
    @patch('pegasi_shield.output_detectors.hallucinations.HallucinationDetector._load_detector')
    def test_scan_exception_handling(self, mock_load_detector):
        """Test scan function exception handling"""
        logger.info("Testing scan exception handling")
        
        mock_load_detector.return_value = True
        
        # Mock detector that raises an exception
        mock_detector = Mock()
        mock_detector.predict.side_effect = Exception("Test exception")
        self.detector._detector = mock_detector
        
        prompt = "What is the capital of France?"
        output = "Paris is the capital of France."
        context = "France is a country in Europe."
        
        result_output, is_clean, score = self.detector.scan(prompt, output, context)
        
        # Should return defaults when exception occurs
        assert result_output == output
        assert is_clean == True
        assert score == 0.0
        
        logger.info("✓ Exception handling test passed")

class TestModernBERTGroundednessDetector:
    """Test suite for ModernBERTGroundednessDetector class"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.detector = ModernBERTGroundednessDetector(threshold=0.5)
    
    def test_initialization(self):
        """Test that ModernBERTGroundednessDetector initializes properly"""
        logger.info("Testing ModernBERTGroundednessDetector initialization")
        
        # Test default initialization
        detector = ModernBERTGroundednessDetector()
        assert detector._threshold == 0.5
        assert detector._model_path == "KRLabsOrg/lettucedect-base-modernbert-en-v1"
        assert detector._detector is None
        
        logger.info("✓ ModernBERT initialization test passed")
    
    def test_check_lettucedetect_available(self):
        """Test _check_lettucedetect_available method"""
        logger.info("Testing _check_lettucedetect_available")
        
        # This will test the actual availability - result may vary
        result = self.detector._check_lettucedetect_available()
        assert isinstance(result, bool)
        
        logger.info("✓ Check lettucedetect availability test passed")

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("Starting comprehensive HallucinationDetector tests...")
    
    # Create test instances
    test_hallucination = TestHallucinationDetector()
    test_modernbert = TestModernBERTGroundednessDetector()
    
    # Run HallucinationDetector tests
    test_methods = [
        (test_hallucination.setup_method, "setup_method"),
        (test_hallucination.test_initialization, "test_initialization"),
        (test_hallucination.test_scan_empty_context, "test_scan_empty_context"),
        (test_hallucination.test_scan_none_context, "test_scan_none_context"),
        (test_hallucination.test_scan_whitespace_context, "test_scan_whitespace_context"),
        (test_hallucination.test_calculate_hallucination_score_no_hallucinations, "test_calculate_hallucination_score_no_hallucinations"),
        (test_hallucination.test_calculate_hallucination_score_with_hallucinations, "test_calculate_hallucination_score_with_hallucinations"),
        (test_hallucination.test_calculate_hallucination_score_empty_output, "test_calculate_hallucination_score_empty_output"),
        (test_hallucination.test_calculate_hallucination_score_no_detector, "test_calculate_hallucination_score_no_detector"),
    ]
    
    # Run ModernBERTGroundednessDetector tests
    test_modernbert.setup_method()
    test_methods.extend([
        (test_modernbert.test_initialization, "test_modernbert_initialization"),
        (test_modernbert.test_check_lettucedetect_available, "test_check_lettucedetect_available"),
    ])
    
    passed = 0
    failed = 0
    
    for test_method, test_name in test_methods:
        try:
            test_method()
            logger.info(f"✓ {test_name} PASSED")
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED: {e}")
            failed += 1
    
    logger.info(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️  {failed} tests failed")
    
    return failed == 0

if __name__ == "__main__":
    # Run the comprehensive tests
    success = run_comprehensive_tests()
    
    # Also run some manual tests to demonstrate functionality
    logger.info("\n" + "="*50)
    logger.info("Manual Testing Examples:")
    logger.info("="*50)
    
    # Test with HallucinationDetector
    detector = HallucinationDetector()
    
    # Test case 1: Empty context
    logger.info("\n1. Testing with empty context:")
    result = detector.scan("What is the capital of France?", "Berlin", "")
    logger.info(f"   Result: {result}")
    
    # Test case 2: None context
    logger.info("\n2. Testing with None context:")
    result = detector.scan("What is the capital of France?", "Berlin", None)
    logger.info(f"   Result: {result}")
    
    # Test case 3: Valid context (will depend on lettucedetect availability)
    logger.info("\n3. Testing with valid context:")
    result = detector.scan(
        "What is the capital of France?", 
        "Berlin", 
        "The capital of France is Paris."
    )
    logger.info(f"   Result: {result}")
    
    # Test case 4: Text cleaning
    logger.info("\n4. Testing text cleaning:")
    cleaned = HallucinationDetector.clean_text("Hello World! It's a BEAUTIFUL day.")
    logger.info(f"   Cleaned text: '{cleaned}'")
    
    sys.exit(0 if success else 1)