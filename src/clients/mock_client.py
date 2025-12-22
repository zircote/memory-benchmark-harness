"""Mock LLM client for testing without API calls.

This module provides a mock LLM client that returns deterministic responses
for testing the benchmark pipeline without making external API calls.
"""

from __future__ import annotations

from src.clients.openai_client import LLMResponse


class MockLLMClient:
    """Mock LLM client for testing without API calls.

    Returns deterministic responses that allow the pipeline to complete
    without making external API calls. Useful for:
    - Testing the benchmark infrastructure
    - CI/CD smoke tests
    - Development without API key

    Example:
        ```python
        client = MockLLMClient()
        response = client.complete(
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.content)  # "This is a mock response."
        ```
    """

    def __init__(
        self,
        default_response: str = "This is a mock response for testing.",
        model: str = "mock-model",
    ) -> None:
        """Initialize the mock LLM client.

        Args:
            default_response: The response to return for all completions
            model: The model name to report
        """
        self.model = model
        self.default_response = default_response
        self._call_count = 0

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a mock completion.

        Args:
            system: System prompt (ignored in mock)
            messages: List of message dicts (ignored in mock)
            temperature: Sampling temperature (ignored in mock)

        Returns:
            LLMResponse with mock content
        """
        self._call_count += 1

        # Generate a response based on call count for some variation
        # This helps with debugging while still being deterministic
        response_text = f"{self.default_response} (call #{self._call_count})"

        return LLMResponse(
            content=response_text,
            model=self.model,
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
            },
        )

    @property
    def call_count(self) -> int:
        """Number of mock calls made."""
        return self._call_count
