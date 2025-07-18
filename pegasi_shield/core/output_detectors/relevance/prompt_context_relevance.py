from .base_relevance import BaseRelevanceDetector


class PromptContextRelevance(BaseRelevanceDetector):
    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        if context.strip() == "":
            return "Prompt & Context", True, 0.0

        # Calculate semantic similarity
        similarity = self._calculate_similarity(prompt, context)

        # Initialize is_relevant based on similarity
        is_relevant = similarity >= self._threshold

        # Initialize risk_score based on similarity
        risk_score = round(1 - similarity, 2)

        # Check for entity mismatches
        mismatch_score = self._check_entity_mismatch(prompt, context)

        # Consider both semantic similarity and entity mismatch for relevance
        # Adjust relevance and risk score based on entity mismatch
        if mismatch_score > 0.8:
            is_relevant = False  # High mismatch overrides similarity for relevance
            risk_score = mismatch_score  # Max risk due to high mismatch
        elif mismatch_score < 0.3:
            # Low mismatch might not significantly affect relevance
            # The risk score is primarily influenced by semantic similarity in this case
            risk_score = round(1 - similarity, 2)
        else:
            # For moderate mismatch, adjust the relevance and risk score accordingly
            is_relevant = similarity >= self._threshold and mismatch_score < 0.5
            # Adjust risk score to reflect the combined effect of similarity and mismatch
            risk_score = round(max(1 - similarity, mismatch_score), 2)

        return "Prompt & Context", is_relevant, risk_score
