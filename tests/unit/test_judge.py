"""Unit tests for LLM-as-Judge evaluation module.

Tests cover:
- Judgment dataclass serialization/validation
- JudgmentCache content-addressing and TTL
- LLMJudge retry logic and error handling
- Mock API responses for deterministic testing
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.judge import (
    DEFAULT_JUDGE_PROMPT,
    Judgment,
    JudgmentCache,
    JudgmentResult,
    LLMJudge,
)


class TestJudgmentResult:
    """Tests for JudgmentResult enum."""

    def test_enum_values(self) -> None:
        """Test all expected enum values exist."""
        assert JudgmentResult.CORRECT.value == "correct"
        assert JudgmentResult.INCORRECT.value == "incorrect"
        assert JudgmentResult.PARTIAL.value == "partial"
        assert JudgmentResult.ERROR.value == "error"

    def test_enum_from_string(self) -> None:
        """Test creating enum from string value."""
        assert JudgmentResult("correct") == JudgmentResult.CORRECT
        assert JudgmentResult("incorrect") == JudgmentResult.INCORRECT
        assert JudgmentResult("partial") == JudgmentResult.PARTIAL

    def test_invalid_enum_raises(self) -> None:
        """Test invalid value raises ValueError."""
        with pytest.raises(ValueError):
            JudgmentResult("invalid")


class TestJudgment:
    """Tests for Judgment dataclass."""

    def test_create_valid_judgment(self) -> None:
        """Test creating a valid judgment."""
        judgment = Judgment(
            result=JudgmentResult.CORRECT,
            score=0.95,
            reasoning="Answer matches reference semantically",
            question="What is 2+2?",
            reference_answer="4",
            model_answer="The answer is 4",
        )
        assert judgment.result == JudgmentResult.CORRECT
        assert judgment.score == 0.95
        assert judgment.cached is False

    def test_score_validation_lower_bound(self) -> None:
        """Test score cannot be below 0."""
        with pytest.raises(ValueError, match="score must be between"):
            Judgment(
                result=JudgmentResult.CORRECT,
                score=-0.1,
                reasoning="test",
                question="q",
                reference_answer="a",
                model_answer="a",
            )

    def test_score_validation_upper_bound(self) -> None:
        """Test score cannot be above 1."""
        with pytest.raises(ValueError, match="score must be between"):
            Judgment(
                result=JudgmentResult.CORRECT,
                score=1.1,
                reasoning="test",
                question="q",
                reference_answer="a",
                model_answer="a",
            )

    def test_score_boundary_values(self) -> None:
        """Test boundary values 0.0 and 1.0 are valid."""
        j0 = Judgment(
            result=JudgmentResult.INCORRECT,
            score=0.0,
            reasoning="test",
            question="q",
            reference_answer="a",
            model_answer="wrong",
        )
        assert j0.score == 0.0

        j1 = Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="test",
            question="q",
            reference_answer="a",
            model_answer="a",
        )
        assert j1.score == 1.0

    def test_to_dict_serialization(self) -> None:
        """Test serialization to dictionary."""
        now = datetime.now(UTC)
        judgment = Judgment(
            result=JudgmentResult.PARTIAL,
            score=0.5,
            reasoning="Partially correct",
            question="What color is the sky?",
            reference_answer="Blue",
            model_answer="Light blue during day",
            metadata={"model": "gpt-4o"},
            cached=False,
            timestamp=now,
        )
        data = judgment.to_dict()

        assert data["result"] == "partial"
        assert data["score"] == 0.5
        assert data["reasoning"] == "Partially correct"
        assert data["question"] == "What color is the sky?"
        assert data["reference_answer"] == "Blue"
        assert data["model_answer"] == "Light blue during day"
        assert data["metadata"] == {"model": "gpt-4o"}
        assert data["cached"] is False
        assert data["timestamp"] == now.isoformat()

    def test_from_dict_deserialization(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "result": "correct",
            "score": 0.9,
            "reasoning": "Correct",
            "question": "q",
            "reference_answer": "a",
            "model_answer": "a",
            "metadata": {"key": "value"},
            "cached": True,
            "timestamp": "2024-01-01T12:00:00+00:00",
        }
        judgment = Judgment.from_dict(data)

        assert judgment.result == JudgmentResult.CORRECT
        assert judgment.score == 0.9
        assert judgment.cached is True
        assert judgment.metadata == {"key": "value"}

    def test_round_trip_serialization(self) -> None:
        """Test serialization round-trip preserves data."""
        original = Judgment(
            result=JudgmentResult.CORRECT,
            score=0.85,
            reasoning="Good answer",
            question="Test Q",
            reference_answer="Test A",
            model_answer="Test Response",
            metadata={"attempt": 1},
        )
        data = original.to_dict()
        restored = Judgment.from_dict(data)

        assert restored.result == original.result
        assert restored.score == original.score
        assert restored.reasoning == original.reasoning
        assert restored.question == original.question


class TestJudgmentCache:
    """Tests for JudgmentCache."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create temporary cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def cache(self, cache_dir: Path) -> JudgmentCache:
        """Create cache instance with temp directory."""
        return JudgmentCache(cache_dir=cache_dir, ttl_days=30)

    def test_cache_miss_returns_none(self, cache: JudgmentCache) -> None:
        """Test cache miss returns None."""
        result = cache.get("q", "ref", "ans", "prompt")
        assert result is None

    def test_cache_put_and_get(self, cache: JudgmentCache) -> None:
        """Test putting and getting a judgment."""
        judgment = Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="Perfect",
            question="What is Python?",
            reference_answer="A programming language",
            model_answer="Python is a programming language",
        )
        prompt = "Test prompt"

        cache.put(judgment, prompt)
        retrieved = cache.get(
            judgment.question,
            judgment.reference_answer,
            judgment.model_answer,
            prompt,
        )

        assert retrieved is not None
        assert retrieved.result == JudgmentResult.CORRECT
        assert retrieved.score == 1.0
        assert retrieved.cached is True  # Should be marked as cached

    def test_content_addressing(self, cache: JudgmentCache) -> None:
        """Test that different inputs produce different cache keys."""
        judgment1 = Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="First",
            question="Q1",
            reference_answer="A1",
            model_answer="M1",
        )
        judgment2 = Judgment(
            result=JudgmentResult.INCORRECT,
            score=0.0,
            reasoning="Second",
            question="Q2",
            reference_answer="A2",
            model_answer="M2",
        )

        cache.put(judgment1, "prompt")
        cache.put(judgment2, "prompt")

        r1 = cache.get("Q1", "A1", "M1", "prompt")
        r2 = cache.get("Q2", "A2", "M2", "prompt")

        assert r1 is not None and r1.reasoning == "First"
        assert r2 is not None and r2.reasoning == "Second"

    def test_prompt_affects_cache_key(self, cache: JudgmentCache) -> None:
        """Test that different prompts produce different cache entries."""
        judgment = Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="Test",
            question="Q",
            reference_answer="A",
            model_answer="M",
        )

        cache.put(judgment, "prompt1")

        # Same Q/A/M but different prompt should miss
        result = cache.get("Q", "A", "M", "prompt2")
        assert result is None

        # Same prompt should hit
        result = cache.get("Q", "A", "M", "prompt1")
        assert result is not None

    def test_ttl_expiration(self, cache_dir: Path) -> None:
        """Test that expired entries are not returned."""
        # Create cache with 1 day TTL
        cache = JudgmentCache(cache_dir=cache_dir, ttl_days=1)

        # Create a judgment with old timestamp
        old_time = datetime.now(UTC) - timedelta(days=5)
        judgment = Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="Old",
            question="Q",
            reference_answer="A",
            model_answer="M",
            timestamp=old_time,
        )

        cache.put(judgment, "prompt")

        # Should return None due to expiration
        result = cache.get("Q", "A", "M", "prompt")
        assert result is None

    def test_stats_tracking(self, cache: JudgmentCache) -> None:
        """Test cache hit/miss statistics."""
        judgment = Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="Test",
            question="Q",
            reference_answer="A",
            model_answer="M",
        )

        # Initial state
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Miss
        cache.get("Q", "A", "M", "prompt")
        stats = cache.get_stats()
        assert stats["misses"] == 1

        # Put and hit
        cache.put(judgment, "prompt")
        cache.get("Q", "A", "M", "prompt")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear_cache(self, cache: JudgmentCache) -> None:
        """Test clearing all cached entries."""
        for i in range(5):
            judgment = Judgment(
                result=JudgmentResult.CORRECT,
                score=1.0,
                reasoning=f"Test {i}",
                question=f"Q{i}",
                reference_answer=f"A{i}",
                model_answer=f"M{i}",
            )
            cache.put(judgment, "prompt")

        count = cache.clear()
        assert count == 5

        # All entries should be gone
        for i in range(5):
            assert cache.get(f"Q{i}", f"A{i}", f"M{i}", "prompt") is None

    def test_invalid_cache_entry_handled(self, cache: JudgmentCache, cache_dir: Path) -> None:
        """Test that corrupted cache entries are handled gracefully."""
        # Manually create an invalid cache file
        cache_dir.mkdir(parents=True, exist_ok=True)
        subdir = cache_dir / "ab"
        subdir.mkdir(exist_ok=True)

        # Write invalid JSON
        invalid_file = subdir / "abcdef1234567890.json"
        invalid_file.write_text("not valid json{{{")

        # Should return None and not crash
        result = cache.get("Q", "A", "M", "prompt")
        assert result is None


class TestLLMJudge:
    """Tests for LLMJudge class."""

    @pytest.fixture
    def mock_openai_response(self) -> MagicMock:
        """Create a mock OpenAI API response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "result": "correct",
                "score": 0.95,
                "reasoning": "The answer correctly states the result.",
            }
        )
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        return mock_response

    @pytest.fixture
    def judge(self, tmp_path: Path) -> LLMJudge:
        """Create judge with mock client."""
        with patch("src.evaluation.judge.OpenAI"):
            return LLMJudge(
                api_key="test-key",  # pragma: allowlist secret
                cache_dir=tmp_path / "cache",
            )

    def test_judge_returns_judgment(self, judge: LLMJudge, mock_openai_response: MagicMock) -> None:
        """Test successful judgment returns correct structure."""
        judge.client.chat.completions.create.return_value = mock_openai_response

        result = judge.judge(
            question="What is 2+2?",
            reference_answer="4",
            model_answer="The answer is 4",
        )

        assert result.result == JudgmentResult.CORRECT
        assert result.score == 0.95
        assert result.cached is False
        assert "model" in result.metadata

    def test_cache_hit_returns_cached(
        self, judge: LLMJudge, mock_openai_response: MagicMock
    ) -> None:
        """Test that second call with same inputs returns cached result."""
        judge.client.chat.completions.create.return_value = mock_openai_response

        # First call
        result1 = judge.judge("Q", "A", "M")
        assert result1.cached is False
        assert judge._call_count == 1

        # Second call with same inputs - should hit cache
        result2 = judge.judge("Q", "A", "M")
        assert result2.cached is True
        assert judge._call_count == 1  # No additional API call

    def test_skip_cache_forces_api_call(
        self, judge: LLMJudge, mock_openai_response: MagicMock
    ) -> None:
        """Test skip_cache=True bypasses cache lookup."""
        judge.client.chat.completions.create.return_value = mock_openai_response

        # First call
        judge.judge("Q", "A", "M")
        assert judge._call_count == 1

        # Second call with skip_cache - should call API
        result = judge.judge("Q", "A", "M", skip_cache=True)
        assert result.cached is False
        assert judge._call_count == 2

    def test_retry_on_rate_limit(self, judge: LLMJudge, mock_openai_response: MagicMock) -> None:
        """Test retry logic on rate limit errors."""
        from openai import RateLimitError

        # First call fails with rate limit, second succeeds
        judge.client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            ),
            mock_openai_response,
        ]

        # Patch sleep to speed up test
        with patch("src.evaluation.judge.time.sleep"):
            result = judge.judge("Q", "A", "M")

        assert result.result == JudgmentResult.CORRECT
        assert result.metadata.get("attempt") == 2

    def test_max_retries_exhausted(self, judge: LLMJudge) -> None:
        """Test that exhausting retries returns error judgment."""
        from openai import RateLimitError

        # All calls fail
        judge.client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        with patch("src.evaluation.judge.time.sleep"):
            result = judge.judge("Q", "A", "M")

        assert result.result == JudgmentResult.ERROR
        assert "retries exhausted" in result.reasoning.lower()

    def test_client_error_no_retry(self, judge: LLMJudge) -> None:
        """Test that 4xx errors (except rate limit) don't retry."""
        from openai import APIStatusError

        # Create a mock response with status_code 400
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}
        mock_response.request = MagicMock()

        judge.client.chat.completions.create.side_effect = APIStatusError(
            message="Bad request",
            response=mock_response,
            body=None,
        )

        result = judge.judge("Q", "A", "M")

        assert result.result == JudgmentResult.ERROR
        assert judge._call_count == 1  # No retry

    def test_invalid_json_response(self, judge: LLMJudge, mock_openai_response: MagicMock) -> None:
        """Test handling of invalid JSON in API response."""
        mock_openai_response.choices[0].message.content = "not json"
        judge.client.chat.completions.create.return_value = mock_openai_response

        result = judge.judge("Q", "A", "M")

        assert result.result == JudgmentResult.ERROR
        assert "json" in result.reasoning.lower()

    def test_invalid_result_value(self, judge: LLMJudge, mock_openai_response: MagicMock) -> None:
        """Test handling of invalid result value in response."""
        mock_openai_response.choices[0].message.content = json.dumps(
            {
                "result": "maybe",  # Invalid
                "score": 0.5,
                "reasoning": "test",
            }
        )
        judge.client.chat.completions.create.return_value = mock_openai_response

        result = judge.judge("Q", "A", "M")

        assert result.result == JudgmentResult.ERROR

    def test_empty_response(self, judge: LLMJudge, mock_openai_response: MagicMock) -> None:
        """Test handling of empty API response."""
        mock_openai_response.choices[0].message.content = None
        judge.client.chat.completions.create.return_value = mock_openai_response

        result = judge.judge("Q", "A", "M")

        assert result.result == JudgmentResult.ERROR

    def test_get_stats(self, judge: LLMJudge, mock_openai_response: MagicMock) -> None:
        """Test statistics collection."""
        judge.client.chat.completions.create.return_value = mock_openai_response

        # Make some calls
        judge.judge("Q1", "A1", "M1")
        judge.judge("Q2", "A2", "M2")
        judge.judge("Q1", "A1", "M1")  # Cache hit

        stats = judge.get_stats()
        assert stats["model"] == "gpt-4o"
        assert stats["total_calls"] == 2
        assert stats["cache"]["hits"] == 1
        assert stats["cache"]["misses"] == 2

    def test_batch_judge(self, judge: LLMJudge, mock_openai_response: MagicMock) -> None:
        """Test batch judgment processing."""
        judge.client.chat.completions.create.return_value = mock_openai_response

        evaluations = [
            ("Q1", "A1", "M1"),
            ("Q2", "A2", "M2"),
            ("Q3", "A3", "M3"),
        ]

        results = judge.batch_judge(evaluations)

        assert len(results) == 3
        assert all(j.result == JudgmentResult.CORRECT for j in results)
        assert judge._call_count == 3

    def test_exponential_backoff(self, judge: LLMJudge, mock_openai_response: MagicMock) -> None:
        """Test that backoff increases exponentially."""
        from openai import RateLimitError

        sleep_times: list[float] = []

        def capture_sleep(seconds: float) -> None:
            sleep_times.append(seconds)

        # Fail 3 times then succeed
        judge.client.chat.completions.create.side_effect = [
            RateLimitError(message="", response=MagicMock(status_code=429), body=None),
            RateLimitError(message="", response=MagicMock(status_code=429), body=None),
            RateLimitError(message="", response=MagicMock(status_code=429), body=None),
            mock_openai_response,
        ]

        with patch("src.evaluation.judge.time.sleep", side_effect=capture_sleep):
            judge.judge("Q", "A", "M")

        # Check exponential growth: 1, 2, 4
        assert len(sleep_times) == 3
        assert sleep_times[0] == 1.0
        assert sleep_times[1] == 2.0
        assert sleep_times[2] == 4.0


class TestDefaultPrompt:
    """Tests for default judge prompt template."""

    def test_prompt_has_placeholders(self) -> None:
        """Test that default prompt has required placeholders."""
        assert "{question}" in DEFAULT_JUDGE_PROMPT
        assert "{reference_answer}" in DEFAULT_JUDGE_PROMPT
        assert "{model_answer}" in DEFAULT_JUDGE_PROMPT

    def test_prompt_requests_json(self) -> None:
        """Test that prompt requests JSON output."""
        assert "JSON" in DEFAULT_JUDGE_PROMPT or "json" in DEFAULT_JUDGE_PROMPT

    def test_prompt_formatting(self) -> None:
        """Test that prompt can be formatted without error."""
        formatted = DEFAULT_JUDGE_PROMPT.format(
            question="Test question",
            reference_answer="Test reference",
            model_answer="Test answer",
        )
        assert "Test question" in formatted
        assert "Test reference" in formatted
        assert "Test answer" in formatted
