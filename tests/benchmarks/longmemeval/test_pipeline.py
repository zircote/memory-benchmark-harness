"""Tests for LongMemEval benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from src.adapters.mock import MockAdapter
from src.benchmarks.longmemeval.dataset import (
    LongMemEvalDataset,
    LongMemEvalQuestion,
    LongMemEvalSession,
    Message,
    QuestionType,
)
from src.benchmarks.longmemeval.pipeline import (
    AssessmentResult,
    BenchmarkPipeline,
    QuestionResult,
)
from src.benchmarks.longmemeval.wrapper import LLMResponse, LongMemEvalAgent
from src.evaluation.judge import Judgment, JudgmentResult


@dataclass
class MockLLMClient:
    """Mock LLM client for testing."""

    default_response: str = "Test response"
    responses: dict[str, str] = field(default_factory=dict)
    call_count: int = 0

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Return a mock response."""
        self.call_count += 1
        user_content = messages[0]["content"] if messages else ""

        for key, response in self.responses.items():
            if key in user_content:
                return LLMResponse(
                    content=response,
                    model="mock-model",
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                )

        return LLMResponse(
            content=self.default_response,
            model="mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


@dataclass
class MockJudge:
    """Mock LLM judge for testing."""

    default_result: JudgmentResult = JudgmentResult.CORRECT
    default_score: float = 1.0
    call_count: int = 0
    batch_call_count: int = 0

    def judge(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        *,
        skip_cache: bool = False,
    ) -> Judgment:
        """Return a mock judgment."""
        self.call_count += 1
        return Judgment(
            result=self.default_result,
            score=self.default_score,
            reasoning="Mock reasoning",
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer,
            metadata={},
            cached=False,
            timestamp=datetime.now(),
        )

    def batch_judge(
        self,
        items: list[tuple[str, str, str]],
        *,
        skip_cache: bool = False,
    ) -> list[Judgment]:
        """Return mock judgments for a batch."""
        self.batch_call_count += 1
        return [
            Judgment(
                result=self.default_result,
                score=self.default_score,
                reasoning="Mock reasoning",
                question=q,
                reference_answer=ref,
                model_answer=ans,
                metadata={},
                cached=False,
                timestamp=datetime.now(),
            )
            for q, ref, ans in items
        ]


class TestQuestionResult:
    """Tests for QuestionResult dataclass."""

    @pytest.fixture
    def sample_judgment(self) -> Judgment:
        """Create a sample judgment."""
        return Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="Correct answer",
            question="What is 2+2?",
            reference_answer="4",
            model_answer="4",
            metadata={},
            cached=False,
            timestamp=datetime.now(),
        )

    def test_is_correct(self, sample_judgment: Judgment) -> None:
        """Test is_correct property."""
        result = QuestionResult(
            question_id="q1",
            question_text="What is 2+2?",
            question_type="multi-session",
            ground_truth=["4"],
            agent_answer="4",
            judgment=sample_judgment,
            is_abstention_expected=False,
            is_abstention_actual=False,
            latency_ms=100.0,
        )
        assert result.is_correct is True
        assert result.score == 1.0

    def test_is_partial(self, sample_judgment: Judgment) -> None:
        """Test is_partial property."""
        partial_judgment = Judgment(
            result=JudgmentResult.PARTIAL,
            score=0.5,
            reasoning="Partially correct",
            question="What is 2+2?",
            reference_answer="4",
            model_answer="about 4",
            metadata={},
            cached=False,
            timestamp=datetime.now(),
        )
        result = QuestionResult(
            question_id="q1",
            question_text="What is 2+2?",
            question_type="multi-session",
            ground_truth=["4"],
            agent_answer="about 4",
            judgment=partial_judgment,
            is_abstention_expected=False,
            is_abstention_actual=False,
            latency_ms=100.0,
        )
        assert result.is_partial is True
        assert result.is_correct is False

    def test_abstention_correct(self, sample_judgment: Judgment) -> None:
        """Test abstention_correct property."""
        # Expected abstention, actual abstention -> correct
        result1 = QuestionResult(
            question_id="q1",
            question_text="Unknown?",
            question_type="multi-session",
            ground_truth=["unknown"],
            agent_answer="I don't know",
            judgment=sample_judgment,
            is_abstention_expected=True,
            is_abstention_actual=True,
            latency_ms=100.0,
        )
        assert result1.abstention_correct is True

        # Expected abstention, no actual abstention -> incorrect
        result2 = QuestionResult(
            question_id="q2",
            question_text="Unknown?",
            question_type="multi-session",
            ground_truth=["unknown"],
            agent_answer="Some guess",
            judgment=sample_judgment,
            is_abstention_expected=True,
            is_abstention_actual=False,
            latency_ms=100.0,
        )
        assert result2.abstention_correct is False


class TestAssessmentResult:
    """Tests for AssessmentResult dataclass."""

    @pytest.fixture
    def sample_results(self) -> list[QuestionResult]:
        """Create sample question results."""
        results = []
        now = datetime.now()

        # 3 correct, 1 partial, 1 incorrect
        for i, (res, score) in enumerate(
            [
                (JudgmentResult.CORRECT, 1.0),
                (JudgmentResult.CORRECT, 1.0),
                (JudgmentResult.CORRECT, 1.0),
                (JudgmentResult.PARTIAL, 0.5),
                (JudgmentResult.INCORRECT, 0.0),
            ]
        ):
            judgment = Judgment(
                result=res,
                score=score,
                reasoning="Test",
                question=f"Q{i}",
                reference_answer="A",
                model_answer="A",
                metadata={},
                cached=False,
                timestamp=now,
            )
            qtype = "single-session-user" if i < 3 else "multi-session"
            results.append(
                QuestionResult(
                    question_id=f"q{i}",
                    question_text=f"Question {i}?",
                    question_type=qtype,
                    ground_truth=["A"],
                    agent_answer="A",
                    judgment=judgment,
                    is_abstention_expected=False,
                    is_abstention_actual=False,
                    latency_ms=100.0,
                )
            )
        return results

    def test_counts(self, sample_results: list[QuestionResult]) -> None:
        """Test correct/partial/incorrect counts."""
        now = datetime.now()
        result = AssessmentResult(
            dataset_subset="S",
            question_results=sample_results,
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        assert result.correct_count == 3
        assert result.partial_count == 1
        assert result.incorrect_count == 1

    def test_accuracy(self, sample_results: list[QuestionResult]) -> None:
        """Test accuracy calculation."""
        now = datetime.now()
        result = AssessmentResult(
            dataset_subset="S",
            question_results=sample_results,
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        assert result.accuracy == 0.6  # 3/5

    def test_mean_score(self, sample_results: list[QuestionResult]) -> None:
        """Test mean score calculation."""
        now = datetime.now()
        result = AssessmentResult(
            dataset_subset="S",
            question_results=sample_results,
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        # (1.0 + 1.0 + 1.0 + 0.5 + 0.0) / 5 = 0.7
        assert result.mean_score == 0.7

    def test_scores_by_type(self, sample_results: list[QuestionResult]) -> None:
        """Test score aggregation by question type."""
        now = datetime.now()
        result = AssessmentResult(
            dataset_subset="S",
            question_results=sample_results,
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        by_type = result.scores_by_type()
        assert "single-session-user" in by_type
        assert "multi-session" in by_type
        assert by_type["single-session-user"] == 1.0  # 3 correct
        assert by_type["multi-session"] == 0.25  # (0.5 + 0.0) / 2

    def test_get_summary(self, sample_results: list[QuestionResult]) -> None:
        """Test summary generation."""
        now = datetime.now()
        result = AssessmentResult(
            dataset_subset="S",
            question_results=sample_results,
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        summary = result.get_summary()
        assert summary["dataset_subset"] == "S"
        assert summary["total_questions"] == 5
        assert summary["correct"] == 3
        assert summary["accuracy"] == 0.6

    def test_abstention_accuracy(self) -> None:
        """Test abstention accuracy calculation."""
        now = datetime.now()
        judgment = Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="Test",
            question="Q",
            reference_answer="A",
            model_answer="A",
            metadata={},
            cached=False,
            timestamp=now,
        )

        # 2 abstention expected: 1 correct abstention, 1 incorrect
        results = [
            QuestionResult(
                question_id="q1",
                question_text="Unknown?",
                question_type="multi-session",
                ground_truth=["unknown"],
                agent_answer="I don't know",
                judgment=judgment,
                is_abstention_expected=True,
                is_abstention_actual=True,  # Correct
                latency_ms=100.0,
            ),
            QuestionResult(
                question_id="q2",
                question_text="Unknown?",
                question_type="multi-session",
                ground_truth=["unknown"],
                agent_answer="Guess",
                judgment=judgment,
                is_abstention_expected=True,
                is_abstention_actual=False,  # Incorrect
                latency_ms=100.0,
            ),
        ]

        result = AssessmentResult(
            dataset_subset="S",
            question_results=results,
            total_questions=2,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        assert result.abstention_accuracy == 0.5  # 1/2


class TestBenchmarkPipeline:
    """Tests for BenchmarkPipeline class."""

    @pytest.fixture
    def sample_dataset(self) -> LongMemEvalDataset:
        """Create a sample dataset for testing."""
        sessions = [
            LongMemEvalSession(
                session_id="s1",
                messages=[
                    Message(role="user", content="My name is Alice"),
                    Message(role="assistant", content="Nice to meet you, Alice!"),
                ],
            ),
            LongMemEvalSession(
                session_id="s2",
                messages=[
                    Message(role="user", content="I love reading books"),
                    Message(role="assistant", content="That's a great hobby!"),
                ],
            ),
        ]

        questions = [
            LongMemEvalQuestion(
                question_id="q1",
                question_text="name",  # Substring match for MockAdapter
                ground_truth=["Alice"],
                question_type=QuestionType.SINGLE_SESSION_USER,
                relevant_session_ids=["s1"],
            ),
            LongMemEvalQuestion(
                question_id="q2",
                question_text="reading",  # Substring match
                ground_truth=["reading books"],
                question_type=QuestionType.SINGLE_SESSION_USER,
                relevant_session_ids=["s2"],
            ),
        ]

        return LongMemEvalDataset(
            subset="S",
            sessions=sessions,
            questions=questions,
        )

    @pytest.fixture
    def pipeline(self) -> BenchmarkPipeline:
        """Create a pipeline with mock dependencies."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="Alice")
        judge = MockJudge()
        return BenchmarkPipeline(adapter, llm, judge, relevant_sessions_only=False)

    def test_pipeline_init(self) -> None:
        """Test pipeline initialization."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        judge = MockJudge()

        pipeline = BenchmarkPipeline(
            adapter,
            llm,
            judge,
            memory_search_limit=20,
            min_relevance_score=0.5,
        )

        assert pipeline._memory_search_limit == 20
        assert pipeline._min_relevance_score == 0.5

    def test_run_pipeline(
        self, pipeline: BenchmarkPipeline, sample_dataset: LongMemEvalDataset
    ) -> None:
        """Test running the complete pipeline."""
        result = pipeline.run(sample_dataset)

        assert result.dataset_subset == "S"
        assert result.total_questions == 2
        assert len(result.question_results) == 2
        assert result.ingestion_time_ms > 0
        assert result.assessment_time_ms > 0

    def test_run_pipeline_with_callback(
        self, pipeline: BenchmarkPipeline, sample_dataset: LongMemEvalDataset
    ) -> None:
        """Test pipeline with progress callback."""
        progress_calls: list[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        pipeline.run(sample_dataset, progress_callback=callback)

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2)
        assert progress_calls[1] == (2, 2)

    def test_run_pipeline_metadata(
        self, pipeline: BenchmarkPipeline, sample_dataset: LongMemEvalDataset
    ) -> None:
        """Test pipeline result metadata."""
        result = pipeline.run(sample_dataset)

        assert "memory_search_limit" in result.metadata
        assert "total_sessions" in result.metadata
        assert result.metadata["total_sessions"] == 2
        assert "cache_hits" in result.metadata
        assert "cache_misses" in result.metadata

    def test_run_single_question(self, sample_dataset: LongMemEvalDataset) -> None:
        """Test running a single question."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="Alice")
        judge = MockJudge()

        pipeline = BenchmarkPipeline(adapter, llm, judge, relevant_sessions_only=False)

        # Create and set up agent
        agent = LongMemEvalAgent(adapter, llm)
        agent.ingest_all_sessions(sample_dataset.sessions)

        # Run single question
        result = pipeline.run_single_question(agent, sample_dataset.questions[0])

        assert result.question_id == "q1"
        assert result.latency_ms > 0
        assert result.judgment.result == JudgmentResult.CORRECT


class TestBenchmarkPipelineIntegration:
    """Integration tests for the benchmark pipeline."""

    def test_full_pipeline_with_mock_components(self) -> None:
        """Test the full pipeline flow with all mock components."""
        # Set up components
        adapter = MockAdapter()
        llm = MockLLMClient(
            default_response="The user's name is Alice",
            responses={
                "reading": "The user enjoys reading books",
            },
        )
        judge = MockJudge(default_result=JudgmentResult.CORRECT, default_score=1.0)

        pipeline = BenchmarkPipeline(adapter, llm, judge, relevant_sessions_only=False)

        # Create dataset
        dataset = LongMemEvalDataset(
            subset="S",
            sessions=[
                LongMemEvalSession(
                    session_id="s1",
                    messages=[
                        Message(role="user", content="My name is Alice"),
                        Message(role="assistant", content="Hello Alice!"),
                        Message(role="user", content="I love reading"),
                    ],
                ),
            ],
            questions=[
                LongMemEvalQuestion(
                    question_id="q1",
                    question_text="name",  # Matches "My name is Alice"
                    ground_truth=["Alice"],
                    question_type=QuestionType.SINGLE_SESSION_USER,
                    relevant_session_ids=["s1"],
                ),
                LongMemEvalQuestion(
                    question_id="q2",
                    question_text="reading",  # Matches "I love reading"
                    ground_truth=["reading books"],
                    question_type=QuestionType.SINGLE_SESSION_USER,
                    relevant_session_ids=["s1"],
                ),
            ],
        )

        # Run pipeline
        result = pipeline.run(dataset)

        # Verify results
        assert result.accuracy == 1.0
        assert result.mean_score == 1.0
        assert result.correct_count == 2
        assert len(result.question_results) == 2

        # Verify judge was called
        assert judge.batch_call_count == 1

    def test_pipeline_with_partial_results(self) -> None:
        """Test pipeline with mixed correct/incorrect results."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="Answer")

        # Custom judge that returns different results
        class MixedJudge:
            def __init__(self) -> None:
                self.results = [
                    (JudgmentResult.CORRECT, 1.0),
                    (JudgmentResult.PARTIAL, 0.5),
                    (JudgmentResult.INCORRECT, 0.0),
                ]
                self.idx = 0

            def batch_judge(
                self, items: list[tuple[str, str, str]], *, skip_cache: bool = False
            ) -> list[Judgment]:
                judgments = []
                for q, ref, ans in items:
                    res, score = self.results[self.idx % len(self.results)]
                    self.idx += 1
                    judgments.append(
                        Judgment(
                            result=res,
                            score=score,
                            reasoning="Test",
                            question=q,
                            reference_answer=ref,
                            model_answer=ans,
                            metadata={},
                            cached=False,
                            timestamp=datetime.now(),
                        )
                    )
                return judgments

        judge = MixedJudge()
        pipeline = BenchmarkPipeline(adapter, llm, judge, relevant_sessions_only=False)

        dataset = LongMemEvalDataset(
            subset="S",
            sessions=[
                LongMemEvalSession(
                    session_id="s1",
                    messages=[Message(role="user", content="test content")],
                ),
            ],
            questions=[
                LongMemEvalQuestion(
                    question_id=f"q{i}",
                    question_text="test",
                    ground_truth=["answer"],
                    question_type=QuestionType.MULTI_SESSION,
                    relevant_session_ids=["s1"],
                )
                for i in range(3)
            ],
        )

        result = pipeline.run(dataset)

        assert result.correct_count == 1
        assert result.partial_count == 1
        assert result.incorrect_count == 1
        assert result.accuracy == pytest.approx(1 / 3)
        assert result.mean_score == pytest.approx(0.5)


class TestEmptyDataset:
    """Tests for edge cases with empty datasets."""

    def test_empty_questions(self) -> None:
        """Test pipeline with no questions."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        judge = MockJudge()

        pipeline = BenchmarkPipeline(adapter, llm, judge)

        dataset = LongMemEvalDataset(
            subset="S",
            sessions=[
                LongMemEvalSession(
                    session_id="s1",
                    messages=[Message(role="user", content="Hello")],
                ),
            ],
            questions=[],
        )

        result = pipeline.run(dataset)

        assert result.total_questions == 0
        assert result.accuracy == 0.0
        assert result.mean_score == 0.0

    def test_empty_sessions(self) -> None:
        """Test pipeline with no sessions."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="I don't know")
        judge = MockJudge()

        pipeline = BenchmarkPipeline(adapter, llm, judge)

        dataset = LongMemEvalDataset(
            subset="S",
            sessions=[],
            questions=[
                LongMemEvalQuestion(
                    question_id="q1",
                    question_text="What?",
                    ground_truth=["answer"],
                    question_type=QuestionType.MULTI_SESSION,
                    relevant_session_ids=[],
                ),
            ],
        )

        result = pipeline.run(dataset)

        assert result.total_questions == 1
        assert result.metadata["total_messages_ingested"] == 0
