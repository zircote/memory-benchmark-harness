"""OpenAI-based LLM client for benchmark assessments.

This module provides an OpenAI client that implements the LLMClient protocol
expected by benchmark pipelines (LongMemEval, LoCoMo, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass(slots=True)
class LLMResponse:
    """Response from an LLM completion.

    Attributes:
        content: The generated text response
        model: The model that generated the response
        usage: Token usage statistics
    """

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class OpenAIClient:
    """OpenAI-based LLM client for benchmark assessments.

    Implements the LLMClient protocol expected by benchmark pipelines.

    Example:
        ```python
        client = OpenAIClient(model="gpt-4o")
        response = client.complete(
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.content)
        ```

    Attributes:
        model: The OpenAI model to use
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 1000,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: Model to use for completions
            max_tokens: Maximum tokens for completion responses
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self._call_count = 0

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            system: System prompt providing context
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            LLMResponse with the generated content
        """
        self._call_count += 1

        # Build messages with system prompt
        all_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        all_messages.extend(messages)

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
        )

    @property
    def call_count(self) -> int:
        """Number of API calls made."""
        return self._call_count
