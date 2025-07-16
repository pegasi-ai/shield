from .base_relevance import BaseRelevanceDetector


class OutputContextRelevance(BaseRelevanceDetector):
    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        if output.strip() == "":
            return "Output & Context", True, 0.0

        # Calculate semantic similarity
        similarity = self._calculate_similarity(output, context)

        # Initialize is_relevant and risk_score
        is_relevant = similarity >= self._threshold
        risk_score = round(1 - similarity, 2)

        # Check for entity mismatches
        mismatch_score = self._check_entity_mismatch(output, context)

        # Adjust relevance and risk score based on entity mismatch
        if mismatch_score > 0.8:
            is_relevant = False  # High mismatch overrides similarity for relevance
            risk_score = mismatch_score  # Max risk due to high mismatch
        elif mismatch_score < 0.3:
            # Low mismatch might not significantly affect relevance
            risk_score = round(1 - similarity, 2)
        else:
            # For moderate mismatch, adjust the relevance and risk score accordingly
            is_relevant = similarity >= self._threshold and mismatch_score < 0.5
            risk_score = round(max(1 - similarity, mismatch_score), 2)

        return "Output & Context", is_relevant, risk_score
