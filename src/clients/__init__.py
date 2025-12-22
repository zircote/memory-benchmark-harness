"""LLM clients for benchmark assessments.

This module provides LLM client implementations that conform to the
LLMClient protocol expected by benchmark pipelines.
"""

from src.clients.mock_client import MockLLMClient
from src.clients.openai_client import LLMResponse, OpenAIClient

__all__ = ["MockLLMClient", "OpenAIClient", "LLMResponse"]
