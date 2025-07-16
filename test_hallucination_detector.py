#!/usr/bin/env python3
"""
Comprehensive hallucination detection tests with various scenarios.
Similar to FinanceBench examples with quantitative, qualitative, and creative test cases.
"""

import sys
import os
import importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pegasi_shield.output_detectors.hallucinations import HallucinationDetector

# Initialize the detector
detector = HallucinationDetector(threshold=0.5)

def check_dependencies():
    """Check if required dependencies are available"""
    lettucedetect_available = importlib.util.find_spec("lettucedetect") is not None
    return lettucedetect_available

def test_hallucination_scenarios():
    """Test various hallucination scenarios"""
    
    print("=" * 80)
    print("HALLUCINATION DETECTION TESTS")
    print("=" * 80)
    
    # Check dependencies
    lettucedetect_available = check_dependencies()
    print(f"lettucedetect library available: {lettucedetect_available}")
    
    if not lettucedetect_available:
        print("⚠️  WARNING: lettucedetect library not found. All tests will return risk score 0.0")
        print("Install with: pip install lettucedetect")
        print("Without this library, the detector cannot distinguish between hallucinated and factual content.")
    
    print(f"Detector threshold: {detector._threshold}")
    print("=" * 80)
    
    # Test 1: FinanceBench-style quantitative accuracy
    print("\n1. FinanceBench Quantitative Example")
    prompt = "Using only the information within the statement of income, what is the FY2017 net interest expense for Corning? Answer in USD millions."
    response = "$155.00"
    context = """
    Index Consolidated Statements of (Loss) Income Corning Incorporated and Subsidiary Companies Years ended December 31,
    (In millions, except per share amounts) 2017 2016 2015 Net sales $ 10,116 $ 9,390 $ 9,111 Cost of sales 6,084 5,644 5,458 Gross margin
    4,032 3,746 3,653 Operating expenses: Selling, general and administrative expenses 1,467 1,472 1,508 Research, development and engineering expenses
    860 742 769 Amortization of purchased intangibles 75 64 54 Restructuring, impairment and other charges (Note 2) 77 Operating income 1,630 1,391 1,322
    Equity in earnings of affiliated companies (Note 7) 361 284 299 Interest income 45 32 21 Interest expense (155) (159) (140) Translated earnings contract (loss) gain, net
    (121) (448) 80 Gain on realignment of equity investment 2,676 Other expense, net (103) (84) (96) Income before income taxes 1,657 3,692 1,486 (Provision) benefit for income taxes (Note 6)
    (2,154) 3 (147) Net (loss) income attributable to Corning Incorporated $ (497) $ 3,695 $ 1,339
    """
    
    result_output, is_clean, risk_score = detector.scan(prompt, response, context)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Risk Score: {risk_score}")
    print(f"Is Clean: {is_clean}")
    expected_result = "✅ ACCURATE" if lettucedetect_available else "❓ CANNOT DETECT (no lettucedetect)"
    print(f"Expected Detection: {expected_result}")
    
    # Test 2: Hallucinated quantitative answer
    print("\n2. Hallucinated Quantitative Answer")
    prompt = "What is the FY2017 net interest expense for Corning?"
    response = "$300.00"  # Incorrect, should be $155
    context = """Interest expense (155) (159) (140) for years 2017, 2016, 2015 respectively"""
    
    result_output, is_clean, risk_score = detector.scan(prompt, response, context)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Risk Score: {risk_score}")
    print(f"Is Clean: {is_clean}")
    expected_result = "🚨 HALLUCINATION DETECTED" if lettucedetect_available else "❓ CANNOT DETECT (no lettucedetect)"
    print(f"Expected Detection: {expected_result}")
    
    # Debug: Let's test what happens with a really obvious hallucination
    print("\n2b. Debug - Very Obvious Hallucination")
    prompt_debug = "What is the FY2017 net interest expense for Corning?"
    response_debug = "The answer is purple elephants and flying unicorns."
    context_debug = """Interest expense (155) (159) (140) for years 2017, 2016, 2015 respectively"""
    
    result_output_debug, is_clean_debug, risk_score_debug = detector.scan(prompt_debug, response_debug, context_debug)
    print(f"Prompt: {prompt_debug}")
    print(f"Response: {response_debug}")
    print(f"Risk Score: {risk_score_debug}")
    print(f"Is Clean: {is_clean_debug}")
    
    # Test 3: Medical membership example
    print("\n3. Medical Membership Example")
    prompt = "How many new medical memberships did UnitedHealthcare report in Q1 2023?"
    response = "Based on the information from the Earnings Call Transcripts, UnitedHealthcare increased the number of people served in the U.S. by 1.2 million in Q1 2023."
    context = """good morning and thank you all for joining us today. growth in the quarter was strong and well balanced across optum and unitedhealthcare with revenue increasing 15% to $92 billion. this broad-based growth, combined with the continued focus of our colleagues on tight execution, helped us deliver first quarter adjusted earnings per share of $6.26, up 14% over last year. year-to-date, unitedhealthcare increased the number of people served in the u.s. by 1.2 million, about half of this total within our commercial offerings."""
    
    result_output, is_clean, risk_score = detector.scan(prompt, response, context)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Risk Score: {risk_score}")
    print(f"Is Clean: {is_clean}")
    expected_result = "✅ ACCURATE" if lettucedetect_available else "❓ CANNOT DETECT (no lettucedetect)"
    print(f"Expected Detection: {expected_result}")
    
    # Test 4: Hallucinated membership numbers
    print("\n4. Hallucinated Membership Numbers")
    prompt = "How many new medical memberships did UnitedHealthcare report in Q1 2023?"
    response = "UnitedHealthcare reported 2.5 million new medical memberships in Q1 2023, with significant growth in Medicare Advantage."
    context = """year-to-date, unitedhealthcare increased the number of people served in the u.s. by 1.2 million, about half of this total within our commercial offerings."""
    
    result_output, is_clean, risk_score = detector.scan(prompt, response, context)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Risk Score: {risk_score}")
    print(f"Is Clean: {is_clean}")
    expected_result = "🚨 HALLUCINATION DETECTED" if lettucedetect_available else "❓ CANNOT DETECT (no lettucedetect)"
    print(f"Expected Detection: {expected_result}")
    


if __name__ == "__main__":
    test_hallucination_scenarios()
