import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging

from pegasi_shield.utils import device

from .base_detector import Detector

_model_path = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
transformers_logging.set_verbosity_error()
log = logging.getLogger(__name__)

MAX_LENGTH = 512


class Contradictions(Detector):
    """
    Contradictions Class:

    This class checks for Contradictions between a given prompt and output using a pretrained NLI model.
    """

    def __init__(self, threshold=0.5):
        """
        Initializes an instance of the Contradictions class.

        Parameters:
            threshold (float): The threshold used to determine Contradictions. Defaults to 0.
        """

        self._model = AutoModelForSequenceClassification.from_pretrained(_model_path)
        self._model.eval()
        self._model.to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(_model_path)
        self._threshold = threshold

        log.debug(f"Initialized sentence transformer {_model_path} on device {device}")

    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        if context.strip() == "":
            print("No context provided...")
            return output, True, 0.0

        tokenized_input_seq_pair = self._tokenizer.encode_plus(
            context,
            output,
            max_length=MAX_LENGTH,
            return_token_type_ids=True,
            truncation=True,
        )

        input_ids = (
            torch.Tensor(tokenized_input_seq_pair["input_ids"])
            .long()
            .unsqueeze(0)
            .to(device)
        )
        # Remember, some models like BART don't have 'token_type_ids'. Remove the line below if you are using such a model.
        token_type_ids = (
            torch.Tensor(tokenized_input_seq_pair["token_type_ids"])
            .long()
            .unsqueeze(0)
            .to(device)
        )
        attention_mask = (
            torch.Tensor(tokenized_input_seq_pair["attention_mask"])
            .long()
            .unsqueeze(0)
            .to(device)
        )

        outputs = self._model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None,
        )

        predicted_probability = torch.softmax(outputs[0], dim=1)[
            0
        ].tolist()  # Batch size is only one

        log.debug(
            f"Detected entailment in the output with score: {predicted_probability[0]}, neutral: {predicted_probability[1]}, contradiction: {predicted_probability[2]}"
        )

        contradiction_score = round(predicted_probability[2], 2)
        if contradiction_score > self._threshold:
            log.warning(
                f"Detected contradiction in the output with score: {contradiction_score}, threshold: {self._threshold}"
            )

            return output, False, contradiction_score

        log.debug(
            f"No contradiction in the output. Max score: {contradiction_score}, threshold: {self._threshold}"
        )

        return output, True, 0.0
