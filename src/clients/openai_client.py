"""OpenAI-based LLM client for benchmark assessments.

This module provides an OpenAI client that implements the LLMClient protocol
expected by benchmark pipelines (LongMemEval, LoCoMo, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai import OpenAI

if TYPE_CHECKING:
    from git_notes_memory.observability import MetricsCollector


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
        client = OpenAIClient(model="gpt-5-mini")
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
        model: str = "gpt-5-mini",
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
        start_time = time.perf_counter()

        # Build messages with system prompt
        all_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        all_messages.extend(messages)

        # Call OpenAI API
        # Build kwargs - gpt-5 models have restrictions on temperature and max_tokens
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "max_completion_tokens": self.max_tokens,
        }

        # gpt-5 models only support temperature=1 (default), so skip the parameter
        if not self.model.startswith("gpt-5"):
            create_kwargs["temperature"] = temperature

        response = self.client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        # Record telemetry
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._record_telemetry(latency_ms, usage)

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
        )

    def _record_telemetry(self, latency_ms: float, usage: dict[str, int]) -> None:
        """Record LLM call telemetry to observability system."""
        try:
            from git_notes_memory.observability import get_metrics

            metrics = get_metrics()
            labels = {"model": self.model, "operation": "complete"}

            # Record latency histogram
            metrics.observe("llm_latency_ms", latency_ms, labels)

            # Record call counter
            metrics.increment("llm_calls_total", 1, labels)

            # Record token counters
            if usage:
                metrics.increment(
                    "llm_tokens_total",
                    usage.get("total_tokens", 0),
                    {**labels, "type": "total"},
                )
                metrics.increment(
                    "llm_tokens_total",
                    usage.get("prompt_tokens", 0),
                    {**labels, "type": "prompt"},
                )
                metrics.increment(
                    "llm_tokens_total",
                    usage.get("completion_tokens", 0),
                    {**labels, "type": "completion"},
                )
        except ImportError:
            pass  # Observability not available

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> str:
        """Generate a response (alternative interface).

        Some benchmark agents use generate() instead of complete().

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens (ignored, uses instance default)

        Returns:
            Generated response string
        """
        system = system_prompt or "You are a helpful assistant."
        messages = [{"role": "user", "content": prompt}]

        response = self.complete(
            system=system,
            messages=messages,
            temperature=0.0,
        )
        return response.content

    @property
    def call_count(self) -> int:
        """Number of API calls made."""
        return self._call_count
