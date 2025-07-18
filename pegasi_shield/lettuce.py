from lettucedetect.models.inference import HallucinationDetector

# For a transformer-based approach:
detector = HallucinationDetector(
    method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

contexts = ["France is a country in Europe. The capital of France is Paris. The population of France is 67 million.",]
question = "What is the capital of France? What is the population of France?"
answer = "The capital of France is Paris. The population of France is 69 million."
# Predictions: [{'start': 31, 'end': 71, 'confidence': 0.9891982674598694, 'text': ' The population of France is 69 million.'}]
# Get span-level predictions indicating which parts of the answer are considered hallucinated.
predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
print("Predictions:", predictions)