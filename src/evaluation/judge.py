"""LLM-as-Judge evaluation module.

This module implements GPT-4o based judgment for benchmark evaluations.
It provides:
- Semantic correctness evaluation with configurable prompts
- Content-addressed caching for reproducibility and cost efficiency
- Exponential backoff retry logic for API resilience

See ADR-006 for design rationale on LLM-as-Judge approach.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


class JudgmentResult(Enum):
    """Possible judgment outcomes."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class Judgment:
    """Result of an LLM judgment evaluation.

    Attributes:
        result: The judgment outcome (correct/incorrect/partial/error)
        score: Numeric score (0.0-1.0), where 1.0 is fully correct
        reasoning: LLM's explanation for the judgment
        question: The original question being evaluated
        reference_answer: The expected/reference answer
        model_answer: The model's response being judged
        metadata: Additional judgment metadata (timing, model, etc.)
        cached: Whether this result was retrieved from cache
        timestamp: When the judgment was made
    """

    result: JudgmentResult
    score: float
    reasoning: str
    question: str
    reference_answer: str
    model_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate score is within valid range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize judgment to dictionary for caching."""
        return {
            "result": self.result.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "model_answer": self.model_answer,
            "metadata": self.metadata,
            "cached": self.cached,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, mark_cached: bool = True) -> Judgment:
        """Deserialize judgment from dictionary.

        Args:
            data: Serialized judgment dictionary
            mark_cached: If True, sets cached=True (use when loading from cache)

        Returns:
            Deserialized Judgment instance
        """
        return cls(
            result=JudgmentResult(data["result"]),
            score=data["score"],
            reasoning=data["reasoning"],
            question=data["question"],
            reference_answer=data["reference_answer"],
            model_answer=data["model_answer"],
            metadata=data.get("metadata", {}),
            cached=mark_cached,  # Always mark as cached when deserializing from storage
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class JudgmentCache:
    """Content-addressed cache for LLM judgments.

    Uses SHA-256 hash of (question, reference, answer, prompt) as cache key.
    Stores judgments as JSON files with configurable TTL.

    Attributes:
        cache_dir: Directory for cache storage
        ttl_days: Time-to-live in days (default 30)
    """

    def __init__(
        self,
        cache_dir: Path | str = ".cache/judgments",
        ttl_days: int = 30,
    ) -> None:
        """Initialize the judgment cache.

        Args:
            cache_dir: Directory for cache storage
            ttl_days: Cache entry expiration in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        self._stats = {"hits": 0, "misses": 0}

    def _compute_key(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        prompt_template: str,
    ) -> str:
        """Compute content-addressed cache key.

        Args:
            question: The evaluation question
            reference_answer: Expected answer
            model_answer: Model's response
            prompt_template: The judge prompt template

        Returns:
            SHA-256 hash string
        """
        content = json.dumps(
            {
                "question": question,
                "reference": reference_answer,
                "answer": model_answer,
                "prompt": prompt_template,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Use first 2 chars as subdirectory for filesystem efficiency
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.json"

    def get(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        prompt_template: str,
    ) -> Judgment | None:
        """Retrieve cached judgment if available and not expired.

        Args:
            question: The evaluation question
            reference_answer: Expected answer
            model_answer: Model's response
            prompt_template: The judge prompt template

        Returns:
            Cached Judgment or None if not found/expired
        """
        key = self._compute_key(question, reference_answer, model_answer, prompt_template)
        cache_path = self._cache_path(key)

        if not cache_path.exists():
            self._stats["misses"] += 1
            return None

        try:
            with cache_path.open("r") as f:
                data = json.load(f)

            # Check TTL
            cached_time = datetime.fromisoformat(data["timestamp"])
            age_days = (datetime.now(UTC) - cached_time).days
            if age_days > self.ttl_days:
                logger.debug(f"Cache expired for key {key[:8]}... (age: {age_days} days)")
                cache_path.unlink()
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            judgment = Judgment.from_dict(data)
            logger.debug(f"Cache hit for key {key[:8]}...")
            return judgment

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid cache entry for key {key[:8]}...: {e}")
            cache_path.unlink(missing_ok=True)
            self._stats["misses"] += 1
            return None

    def put(
        self,
        judgment: Judgment,
        prompt_template: str,
    ) -> None:
        """Store a judgment in the cache.

        Args:
            judgment: The judgment to cache
            prompt_template: The judge prompt template (for key computation)
        """
        key = self._compute_key(
            judgment.question,
            judgment.reference_answer,
            judgment.model_answer,
            prompt_template,
        )
        cache_path = self._cache_path(key)

        with cache_path.open("w") as f:
            json.dump(judgment.to_dict(), f, indent=2)

        logger.debug(f"Cached judgment for key {key[:8]}...")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hit/miss counts and hit rate
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "total": total,
            "hit_rate": hit_rate,
        }

    def clear(self) -> int:
        """Clear all cached judgments.

        Returns:
            Number of entries cleared
        """
        count = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
        self._stats = {"hits": 0, "misses": 0}
        return count


# Default judgment prompt template
DEFAULT_JUDGE_PROMPT = """You are an expert evaluator judging the correctness of AI responses.

Given a question and a reference answer, evaluate whether the model's answer is correct.

## Question
{question}

## Reference Answer
{reference_answer}

## Model's Answer
{model_answer}

## Evaluation Instructions
1. Compare the model's answer to the reference answer
2. Consider semantic equivalence, not just exact matching
3. Partial credit is appropriate for incomplete but correct information

Respond with a JSON object containing:
{{
    "result": "correct" | "incorrect" | "partial",
    "score": <float 0.0-1.0>,
    "reasoning": "<brief explanation>"
}}

IMPORTANT: Respond ONLY with the JSON object, no other text."""


class LLMJudge:
    """LLM-based judge for benchmark evaluations.

    Uses GPT-4o to evaluate model responses against reference answers.
    Includes caching for reproducibility and cost efficiency, plus
    exponential backoff retry logic for API resilience.

    Attributes:
        model: The OpenAI model to use for judging
        cache: JudgmentCache instance for caching results
        prompt_template: Template for judge prompts
    """

    # Retry configuration
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 1.0  # seconds
    MAX_BACKOFF = 60.0  # seconds
    BACKOFF_MULTIPLIER = 2.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5-nano",
        cache_dir: Path | str = ".cache/judgments",
        cache_ttl_days: int = 30,
        prompt_template: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        """Initialize the LLM judge.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: Model to use for judging
            cache_dir: Directory for judgment cache
            cache_ttl_days: Cache TTL in days
            prompt_template: Custom judge prompt template
            temperature: LLM temperature (0.0 for deterministic)
            max_tokens: Max tokens for judgment response
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache = JudgmentCache(cache_dir=cache_dir, ttl_days=cache_ttl_days)
        self.prompt_template = prompt_template or DEFAULT_JUDGE_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._call_count = 0
        self._error_count = 0

    def judge(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        *,
        skip_cache: bool = False,
    ) -> Judgment:
        """Judge a model's answer against a reference.

        Args:
            question: The evaluation question
            reference_answer: The expected/correct answer
            model_answer: The model's response to judge
            skip_cache: If True, bypass cache lookup (still caches result)

        Returns:
            Judgment with result, score, and reasoning

        Raises:
            RuntimeError: If all retries exhausted
        """
        # Check cache first
        if not skip_cache:
            cached = self.cache.get(question, reference_answer, model_answer, self.prompt_template)
            if cached is not None:
                return cached

        # Build the prompt
        prompt = self.prompt_template.format(
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer,
        )

        # Call the API with retry logic
        judgment = self._call_with_retry(
            prompt=prompt,
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer,
        )

        # Cache the result
        self.cache.put(judgment, self.prompt_template)

        return judgment

    def _call_with_retry(
        self,
        prompt: str,
        question: str,
        reference_answer: str,
        model_answer: str,
    ) -> Judgment:
        """Call OpenAI API with exponential backoff retry.

        Args:
            prompt: The formatted judge prompt
            question: Original question (for Judgment construction)
            reference_answer: Reference answer (for Judgment construction)
            model_answer: Model answer (for Judgment construction)

        Returns:
            Judgment from successful API call

        Raises:
            RuntimeError: If all retries exhausted
        """
        last_error: Exception | None = None
        backoff = self.INITIAL_BACKOFF

        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.monotonic()
                self._call_count += 1

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )

                elapsed = time.monotonic() - start_time
                content = response.choices[0].message.content

                if content is None:
                    raise ValueError("Empty response from API")

                # Parse the JSON response
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response: {e}") from e

                # Validate and construct Judgment
                result_str = data.get("result", "").lower()
                if result_str not in ("correct", "incorrect", "partial"):
                    raise ValueError(f"Invalid result value: {result_str}")

                result = JudgmentResult(result_str)
                score = float(data.get("score", 0.0))
                reasoning = data.get("reasoning", "")

                return Judgment(
                    result=result,
                    score=score,
                    reasoning=reasoning,
                    question=question,
                    reference_answer=reference_answer,
                    model_answer=model_answer,
                    metadata={
                        "model": self.model,
                        "latency_ms": int(elapsed * 1000),
                        "attempt": attempt + 1,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                            "completion_tokens": response.usage.completion_tokens
                            if response.usage
                            else 0,
                        },
                    },
                    cached=False,
                )

            except RateLimitError as e:
                last_error = e
                self._error_count += 1
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                    f"backing off {backoff:.1f}s"
                )
                time.sleep(backoff)
                backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)

            except APIConnectionError as e:
                last_error = e
                self._error_count += 1
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                    f"backing off {backoff:.1f}s: {e}"
                )
                time.sleep(backoff)
                backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)

            except APIStatusError as e:
                last_error = e
                self._error_count += 1
                # Don't retry on 4xx errors (except rate limits which are handled above)
                if 400 <= e.status_code < 500:
                    logger.error(f"Client error (not retrying): {e}")
                    return self._error_judgment(question, reference_answer, model_answer, str(e))
                # 5xx errors - retry with backoff
                logger.warning(
                    f"API error (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                    f"backing off {backoff:.1f}s: {e}"
                )
                time.sleep(backoff)
                backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)

            except ValueError as e:
                # Parse errors - don't retry
                last_error = e
                self._error_count += 1
                logger.error(f"Parse error (not retrying): {e}")
                return self._error_judgment(question, reference_answer, model_answer, str(e))

        # All retries exhausted
        error_msg = f"All {self.MAX_RETRIES} retries exhausted. Last error: {last_error}"
        logger.error(error_msg)
        return self._error_judgment(question, reference_answer, model_answer, error_msg)

    def _error_judgment(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        error: str,
    ) -> Judgment:
        """Create an error judgment when API call fails."""
        return Judgment(
            result=JudgmentResult.ERROR,
            score=0.0,
            reasoning=f"Judgment failed: {error}",
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer,
            metadata={"error": error, "model": self.model},
            cached=False,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get judge statistics.

        Returns:
            Dictionary with call counts, error counts, and cache stats
        """
        return {
            "model": self.model,
            "total_calls": self._call_count,
            "error_count": self._error_count,
            "cache": self.cache.get_stats(),
        }

    def batch_judge(
        self,
        evaluations: list[tuple[str, str, str]],
        *,
        skip_cache: bool = False,
    ) -> list[Judgment]:
        """Judge multiple evaluations.

        Args:
            evaluations: List of (question, reference_answer, model_answer) tuples
            skip_cache: If True, bypass cache lookup

        Returns:
            List of Judgments in same order as input
        """
        return [self.judge(q, ref, ans, skip_cache=skip_cache) for q, ref, ans in evaluations]
