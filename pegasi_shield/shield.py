# pegasi_shield/shield.py
from __future__ import annotations
from typing import Callable, Sequence, Dict, Any, Optional, Iterable, Union

class ShieldError(Exception):
    """Raised when a detector/policy blocks or sanitizes in a disallowed way."""

class Detector:
    """
    Minimal Detector interface.
    Implement either/both check_input and check_output; raise ShieldError if blocked.
    Return value from check_output may replace the response (e.g., sanitized).
    """
    def check_input(self, messages: Sequence[Dict[str, Any]]) -> None:
        return None
    def check_output(
        self,
        messages: Sequence[Dict[str, Any]],
        response: Any,
        content_text: Optional[str] = None
    ) -> Any:
        return response

class Shield:
    def __init__(
        self,
        input_detectors: Optional[Sequence[Detector]] = None,
        output_detectors: Optional[Sequence[Detector]] = None,
        policy: str = "default",
        fail_fast: bool = True,
    ):
        self.policy = policy
        self.fail_fast = fail_fast
        self.input_detectors = list(input_detectors or [])
        self.output_detectors = list(output_detectors or [])

    # ---------- Internal helpers ----------
    def _run_input(self, messages):
        for d in self.input_detectors:
            d.check_input(messages)

    def _run_output(self, messages, response, content_text: Optional[str]):
        for d in self.output_detectors:
            response = d.check_output(messages, response, content_text=content_text)
        return response

    # ---------- Public method ----------
    def chat_completion(
        self,
        llm_call: Callable[[], Any],
        *,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        stream: bool = False,
        collect_stream_text: bool = True,
        text_extractor: Optional[Callable[[Any], str]] = None,
    ) -> Union[Any, Iterable[Any]]:
        """
        Run an LLM call under Shield:
          - Pre-scan messages (if provided)
          - Execute user-supplied callable
          - (Optionally) stream through to caller while collecting content
          - Post-scan / sanitize the final result

        Parameters
        ----------
        llm_call: zero-argument callable that performs the raw model invocation.
        messages: original user messages (for input scanning).
        stream: whether the underlying call returns an iterator of chunks.
        collect_stream_text: if True, we concatenate chunk texts to run output detectors at end.
        text_extractor: optional override to extract final text from a non-standard response object.

        Returns
        -------
        Original response object (possibly sanitized) or an iterator (if streaming).
        """
        msgs = list(messages or [])

        if msgs:
            self._run_input(msgs)

        result = llm_call()

        if stream:
            # Assume iterator of chunks
            iterator = iter(result)
            if not collect_stream_text:
                # Pass-through
                for chunk in iterator:
                    yield chunk
                return
            collected_parts = []
            # Wrap streaming generator
            def generator():
                for chunk in iterator:
                    # Attempt to read chunk text for later scanning
                    try:
                        # OpenAI style: chunk.choices[0].delta.content
                        piece = getattr(chunk.choices[0].delta, "content", None)
                        if piece:
                            collected_parts.append(piece)
                    except Exception:
                        pass
                    yield chunk
            # Return a generator the caller can iterate over
            def final_iter():
                for ch in generator():
                    yield ch
                aggregated = "".join(collected_parts) if collected_parts else None
                # Run output detectors after stream finishes
                self._run_output(msgs, None, aggregated)
            return final_iter()

        # Non-stream: extract text for detectors if possible
        content_text = None
        if text_extractor is not None:
            try:
                content_text = text_extractor(result)
            except Exception:
                content_text = None
        else:
            # Default OpenAI Chat response extraction attempt
            try:
                content_text = result.choices[0].message.content
            except Exception:
                content_text = None

        scanned = self._run_output(msgs, result, content_text)
        return scanned

__all__ = ["Shield", "ShieldError", "Detector"]
