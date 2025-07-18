import logging
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from ...utils import device
import torch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)
_transformer_name = "sentence-transformers/paraphrase-mpnet-base-v2"  # Upgraded model


class BaseRelevanceDetector:
    """
    Base class for relevance detection.
    """

    _ner_pipeline = None  # Class variable to hold the Hugging Face NER model

    def __init__(self, threshold: float = 0.85):
        self._threshold = threshold
        self._transformer_model = SentenceTransformer(_transformer_name, device=device)
        log.debug(
            f"Initialized sentence transformer {_transformer_name} on device {device}"
        )

        if BaseRelevanceDetector._ner_pipeline is None:
            # Initialize the Hugging Face NER pipeline
            model_name = (
                "dbmdz/bert-large-cased-finetuned-conll03-english"  # Upgraded NER model
            )
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            BaseRelevanceDetector._ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            log.debug(f"Hugging Face NER pipeline loaded with model {model_name}")

    @lru_cache(maxsize=128)
    def _encode_text(self, text: str):
        try:
            embedding = self._transformer_model.encode(text, convert_to_tensor=True)
        except Exception as e:
            log.error(f"Error encoding text: {e}")
            return None
        return embedding

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        embedding_1 = self._encode_text(text1)
        embedding_2 = self._encode_text(text2)
        if embedding_1 is None or embedding_2 is None:
            return 0.0
        similarity = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        return similarity

    def _aggregate_entities(self, ner_results):
        aggregated_entities = {}
        buffer_word = ""
        buffer_type = ""
        for entity in ner_results:
            word = entity["word"]
            entity_type = entity["entity"]

            # If current entity is a continuation of the previous one, merge it
            if word.startswith("##"):
                buffer_word += word.lstrip("#")
            else:
                # If there's something in the buffer, save it before starting a new word
                if buffer_word:
                    aggregated_entities[buffer_word] = buffer_type
                buffer_word = word
                buffer_type = entity_type

        # Don't forget to add the last buffered word if present
        if buffer_word:
            aggregated_entities[buffer_word] = buffer_type

        return aggregated_entities

    def _extract_entities(self, text: str):
        # Use the NER pipeline to extract named entities
        try:
            ner_results = BaseRelevanceDetector._ner_pipeline(text)
        except Exception as e:
            log.error(f"Error extracting entities: {e}")
            return {}
        # Filter entities by their types (person and organization)
        filtered_entities = [
            entity for entity in ner_results if entity["entity"] in ["B-PER", "B-ORG"]
        ]
        # Aggregate entities to handle subtokens and reconstruct full entity names
        entities = self._aggregate_entities(filtered_entities)
        return entities

    def _batch_extract_entities(self, texts: list):
        # Use batch processing for NER to handle large texts or multiple texts efficiently
        try:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._extract_entities, text): text
                    for text in texts
                }
                results = {}
                for future in as_completed(futures):
                    text = futures[future]
                    try:
                        entities = future.result()
                        results[text] = entities
                    except Exception as e:
                        log.error(
                            f"Error in batch extracting entities for text: {text}, error: {e}"
                        )
                        results[text] = {}
        except Exception as e:
            log.error(f"Error in batch processing: {e}")
            results = {text: {} for text in texts}
        return results

    def _calculate_mismatch_score(self, entities1, entities2):
        """
        Calculate a weighted mismatch score based on the symmetric difference
        of entities in the two sets, using the keys from the entity dictionaries.
        """
        entity_words1 = set(entities1.keys())
        entity_words2 = set(entities2.keys())

        total_entities = len(entity_words1.union(entity_words2))
        if total_entities == 0:
            return 0  # Avoid division by zero

        mismatched_entities = entity_words1.symmetric_difference(entity_words2)
        mismatch_score = len(mismatched_entities) / total_entities

        # Apply weights based on entity importance
        weight = 0.5  # Example weight, can be adjusted or dynamically determined
        weighted_mismatch_score = mismatch_score * weight

        return weighted_mismatch_score

    def _adjust_score_for_text_length(self, score, text1, text2, base_mismatch_score):
        """
        Conditionally adjust the mismatch score based on the length of the texts.
        This adjustment is minimized or not applied for high base mismatch scores
        to preserve the indicative value of clear mismatches.
        """
        # Minimize length adjustment for high base mismatch scores
        if base_mismatch_score > 0.5:
            return min(score, 1)  # Preserve higher mismatch scores

        length_ratio = len(text1) / (len(text2) + 1)  # Avoid division by zero
        length_adjustment = min(max(length_ratio, 0.1), 10)  # Clamp the ratio

        adjusted_score = score * length_adjustment
        return min(adjusted_score, 1)  # Ensure the score does not exceed 1

    def _check_entity_mismatch(self, text1: str, text2: str) -> float:
        entities1 = self._extract_entities(text1)
        entities2 = self._extract_entities(text2)

        base_score = self._calculate_mismatch_score(entities1, entities2)
        # Now passing base_score to _adjust_score_for_text_length
        adjusted_score = self._adjust_score_for_text_length(
            base_score, text1, text2, base_score
        )

        log.debug(f"Adjusted entity score: {adjusted_score}")
        log.debug(f"Entities in text1: {entities1}")
        log.debug(f"Entities in text2: {entities2}")
        return adjusted_score

    def detect_relevance(self, prompt: str, response: str, context: str) -> float:
        """
        Detect relevance between the prompt, response, and context.
        Combines similarity and entity mismatch scores for a final relevance score.
        """
        similarity_score = self._calculate_similarity(prompt + context, response)
        entity_mismatch_score = self._check_entity_mismatch(prompt + context, response)

        # Combine scores (weights can be adjusted based on importance)
        final_score = similarity_score * (1 - entity_mismatch_score)

        log.debug(f"Final relevance score: {final_score}")
        return final_score
