"""
Drop‑in OpenAI client protected by Pegasi Shield.

Example:
    from core.openai_wrapper import Shield
    shield = Shield(api_key="sk-...")
    resp = shield.chat_completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi!"}],
    )
"""
from openai import OpenAI
from .shield import Shield as _Core


class Shield(OpenAI):
    def chat_completion(self, *args, **kwargs):
        with _Core(policy="default") as guard:
            return guard.run(super().chat.completions.create, *args, **kwargs)


__all__ = ["Shield"]
