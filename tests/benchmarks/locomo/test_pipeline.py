"""Tests for LoCoMo benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from src.adapters.mock import MockAdapter
from src.benchmarks.locomo.dataset import (
    LoCoMoConversation,
    LoCoMoDataset,
    LoCoMoQuestion,
    LoCoMoSession,
    LoCoMoTurn,
    QACategory,
)
from src.benchmarks.locomo.pipeline import (
    AssessmentResult,
    CategoryMetrics,
    ConversationMetrics,
    LoCoMoPipeline,
    QuestionResult,
)
from src.benchmarks.locomo.wrapper import LLMResponse, LoCoMoAgent
from src.evaluation.judge import Judgment, JudgmentResult


@dataclass
class MockLLMClient:
    """Mock LLM client for testing."""

    default_response: str = "Test response"
    responses: dict[str, str] = field(default_factory=dict)
    call_count: int = 0

    def complete(
        self,
        system: str,  # noqa: ARG002
        messages: list[dict[str, str]],
        temperature: float = 0.0,  # noqa: ARG002
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
        skip_cache: bool = False,  # noqa: ARG002
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
        skip_cache: bool = False,  # noqa: ARG002
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


def create_sample_conversation(conv_id: str = "conv_1") -> LoCoMoConversation:
    """Create a sample conversation for testing."""
    sessions = [
        LoCoMoSession(
            session_num=1,
            timestamp="2023-01-15 10:30:00",
            turns=[
                LoCoMoTurn(
                    dia_id="D1:1",
                    speaker="Alice",
                    text="Hi Bob, I'm Alice. I work at OpenAI.",
                    session_num=1,
                ),
                LoCoMoTurn(
                    dia_id="D1:2",
                    speaker="Bob",
                    text="Nice to meet you Alice! I work at Google.",
                    session_num=1,
                ),
            ],
            speaker_a="Alice",
            speaker_b="Bob",
        ),
        LoCoMoSession(
            session_num=2,
            timestamp="2023-02-20 14:00:00",
            turns=[
                LoCoMoTurn(
                    dia_id="D2:1",
                    speaker="Alice",
                    text="I went skiing last weekend.",
                    session_num=2,
                ),
                LoCoMoTurn(
                    dia_id="D2:2",
                    speaker="Bob",
                    text="That sounds fun! I prefer hiking.",
                    session_num=2,
                ),
            ],
            speaker_a="Alice",
            speaker_b="Bob",
        ),
    ]

    questions = [
        LoCoMoQuestion(
            question_id=f"{conv_id}_q1",
            conversation_id=conv_id,
            category=QACategory.IDENTITY,
            question="Where does Alice work?",
            answer="OpenAI",
            evidence=["D1:1"],
        ),
        LoCoMoQuestion(
            question_id=f"{conv_id}_q2",
            conversation_id=conv_id,
            category=QACategory.TEMPORAL,
            question="When did Alice go skiing?",
            answer="Last weekend",
            evidence=["D2:1"],
        ),
    ]

    return LoCoMoConversation(
        sample_id=conv_id,
        sessions=sessions,
        questions=questions,
    )


def create_adversarial_question(conv_id: str = "conv_1") -> LoCoMoQuestion:
    """Create an adversarial question for testing."""
    return LoCoMoQuestion(
        question_id=f"{conv_id}_adv",
        conversation_id=conv_id,
        category=QACategory.ADVERSARIAL,
        question="What did Alice say about her Microsoft job?",
        answer="Alice works at OpenAI, not Microsoft",
        evidence=["D1:1"],
        adversarial_answer="The premise is incorrect - Alice works at OpenAI",
    )


class TestQuestionResult:
    """Tests for QuestionResult dataclass."""

    @pytest.fixture
    def sample_judgment(self) -> Judgment:
        """Create a sample judgment."""
        return Judgment(
            result=JudgmentResult.CORRECT,
            score=1.0,
            reasoning="Correct answer",
            question="Where does Alice work?",
            reference_answer="OpenAI",
            model_answer="OpenAI",
            metadata={},
            cached=False,
            timestamp=datetime.now(),
        )

    def test_is_correct(self, sample_judgment: Judgment) -> None:
        """Test is_correct property."""
        result = QuestionResult(
            question_id="q1",
            conversation_id="conv_1",
            question_text="Where does Alice work?",
            category=QACategory.IDENTITY,
            ground_truth="OpenAI",
            agent_answer="OpenAI",
            judgment=sample_judgment,
            is_adversarial=False,
            is_abstention_actual=False,
            adversarial_handled=False,
            latency_ms=100.0,
            evidence_sessions=[1],
        )
        assert result.is_correct is True
        assert result.score == 1.0

    def test_is_partial(self, sample_judgment: Judgment) -> None:  # noqa: ARG002
        """Test is_partial property."""
        partial_judgment = Judgment(
            result=JudgmentResult.PARTIAL,
            score=0.5,
            reasoning="Partially correct",
            question="Where does Alice work?",
            reference_answer="OpenAI",
            model_answer="A tech company",
            metadata={},
            cached=False,
            timestamp=datetime.now(),
        )
        result = QuestionResult(
            question_id="q1",
            conversation_id="conv_1",
            question_text="Where does Alice work?",
            category=QACategory.IDENTITY,
            ground_truth="OpenAI",
            agent_answer="A tech company",
            judgment=partial_judgment,
            is_adversarial=False,
            is_abstention_actual=False,
            adversarial_handled=False,
            latency_ms=100.0,
            evidence_sessions=[1],
        )
        assert result.is_partial is True
        assert result.is_correct is False

    def test_category_name(self, sample_judgment: Judgment) -> None:
        """Test category_name property."""
        result = QuestionResult(
            question_id="q1",
            conversation_id="conv_1",
            question_text="Where does Alice work?",
            category=QACategory.IDENTITY,
            ground_truth="OpenAI",
            agent_answer="OpenAI",
            judgment=sample_judgment,
            is_adversarial=False,
            is_abstention_actual=False,
            adversarial_handled=False,
            latency_ms=100.0,
            evidence_sessions=[1],
        )
        assert result.category_name == "IDENTITY"

    def test_adversarial_properties(self, sample_judgment: Judgment) -> None:
        """Test adversarial question properties."""
        result = QuestionResult(
            question_id="q1",
            conversation_id="conv_1",
            question_text="What about her Microsoft job?",
            category=QACategory.ADVERSARIAL,
            ground_truth="Premise is incorrect",
            agent_answer="That's not accurate - she works at OpenAI",
            judgment=sample_judgment,
            is_adversarial=True,
            is_abstention_actual=False,
            adversarial_handled=True,
            latency_ms=100.0,
            evidence_sessions=[1],
        )
        assert result.is_adversarial is True
        assert result.adversarial_handled is True


class TestCategoryMetrics:
    """Tests for CategoryMetrics dataclass."""

    def test_accuracy_calculation(self) -> None:
        """Test accuracy property."""
        metrics = CategoryMetrics(
            category=QACategory.IDENTITY,
            total_questions=10,
            correct_count=7,
            partial_count=2,
            mean_score=0.8,
            mean_latency_ms=150.0,
        )
        assert metrics.accuracy == 0.7

    def test_accuracy_zero_questions(self) -> None:
        """Test accuracy with zero questions."""
        metrics = CategoryMetrics(
            category=QACategory.IDENTITY,
            total_questions=0,
            correct_count=0,
            partial_count=0,
            mean_score=0.0,
            mean_latency_ms=0.0,
        )
        assert metrics.accuracy == 0.0

    def test_category_name(self) -> None:
        """Test category_name property."""
        metrics = CategoryMetrics(
            category=QACategory.TEMPORAL,
            total_questions=5,
            correct_count=3,
            partial_count=1,
            mean_score=0.7,
            mean_latency_ms=100.0,
        )
        assert metrics.category_name == "TEMPORAL"


class TestConversationMetrics:
    """Tests for ConversationMetrics dataclass."""

    def test_accuracy_calculation(self) -> None:
        """Test accuracy property."""
        metrics = ConversationMetrics(
            conversation_id="conv_1",
            sessions_ingested=5,
            turns_ingested=50,
            questions_assessed=20,
            correct_count=15,
            mean_score=0.8,
        )
        assert metrics.accuracy == 0.75

    def test_accuracy_zero_questions(self) -> None:
        """Test accuracy with zero questions."""
        metrics = ConversationMetrics(
            conversation_id="conv_1",
            sessions_ingested=5,
            turns_ingested=50,
            questions_assessed=0,
            correct_count=0,
            mean_score=0.0,
        )
        assert metrics.accuracy == 0.0


class TestAssessmentResult:
    """Tests for AssessmentResult dataclass."""

    @pytest.fixture
    def sample_results(self) -> list[QuestionResult]:
        """Create sample question results with different categories."""
        results = []
        now = datetime.now()

        # Category 1 (IDENTITY): 2 correct, 1 partial
        # Category 2 (TEMPORAL): 1 correct, 1 incorrect
        test_data = [
            (QACategory.IDENTITY, JudgmentResult.CORRECT, 1.0, False, False),
            (QACategory.IDENTITY, JudgmentResult.CORRECT, 1.0, False, False),
            (QACategory.IDENTITY, JudgmentResult.PARTIAL, 0.5, False, False),
            (QACategory.TEMPORAL, JudgmentResult.CORRECT, 1.0, False, False),
            (QACategory.TEMPORAL, JudgmentResult.INCORRECT, 0.0, False, False),
        ]

        for i, (cat, res, score, is_adv, adv_handled) in enumerate(test_data):
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
            results.append(
                QuestionResult(
                    question_id=f"q{i}",
                    conversation_id="conv_1",
                    question_text=f"Question {i}?",
                    category=cat,
                    ground_truth="A",
                    agent_answer="A",
                    judgment=judgment,
                    is_adversarial=is_adv,
                    is_abstention_actual=False,
                    adversarial_handled=adv_handled,
                    latency_ms=100.0,
                    evidence_sessions=[1],
                )
            )
        return results

    def test_counts(self, sample_results: list[QuestionResult]) -> None:
        """Test correct/partial/incorrect counts."""
        now = datetime.now()
        result = AssessmentResult(
            question_results=sample_results,
            category_metrics={},
            conversation_metrics={},
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
            question_results=sample_results,
            category_metrics={},
            conversation_metrics={},
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
            question_results=sample_results,
            category_metrics={},
            conversation_metrics={},
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        # (1.0 + 1.0 + 0.5 + 1.0 + 0.0) / 5 = 0.7
        assert result.mean_score == 0.7

    def test_adversarial_accuracy_all_handled(self) -> None:
        """Test adversarial accuracy when all are handled."""
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

        results = [
            QuestionResult(
                question_id=f"q{i}",
                conversation_id="conv_1",
                question_text=f"Adversarial {i}?",
                category=QACategory.ADVERSARIAL,
                ground_truth="A",
                agent_answer="Premise is incorrect",
                judgment=judgment,
                is_adversarial=True,
                is_abstention_actual=False,
                adversarial_handled=True,
                latency_ms=100.0,
                evidence_sessions=[1],
            )
            for i in range(2)
        ]

        assessment = AssessmentResult(
            question_results=results,
            category_metrics={},
            conversation_metrics={},
            total_questions=2,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        assert assessment.adversarial_accuracy == 1.0

    def test_adversarial_accuracy_partial(self) -> None:
        """Test adversarial accuracy with partial handling."""
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

        results = [
            QuestionResult(
                question_id="q1",
                conversation_id="conv_1",
                question_text="Adversarial 1?",
                category=QACategory.ADVERSARIAL,
                ground_truth="A",
                agent_answer="Premise is incorrect",
                judgment=judgment,
                is_adversarial=True,
                is_abstention_actual=False,
                adversarial_handled=True,  # Handled
                latency_ms=100.0,
                evidence_sessions=[1],
            ),
            QuestionResult(
                question_id="q2",
                conversation_id="conv_1",
                question_text="Adversarial 2?",
                category=QACategory.ADVERSARIAL,
                ground_truth="A",
                agent_answer="Some wrong answer",
                judgment=judgment,
                is_adversarial=True,
                is_abstention_actual=False,
                adversarial_handled=False,  # Not handled
                latency_ms=100.0,
                evidence_sessions=[1],
            ),
        ]

        assessment = AssessmentResult(
            question_results=results,
            category_metrics={},
            conversation_metrics={},
            total_questions=2,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        assert assessment.adversarial_accuracy == 0.5

    def test_adversarial_accuracy_no_adversarial(
        self, sample_results: list[QuestionResult]
    ) -> None:
        """Test adversarial accuracy with no adversarial questions."""
        now = datetime.now()
        assessment = AssessmentResult(
            question_results=sample_results,  # No adversarial questions
            category_metrics={},
            conversation_metrics={},
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        assert assessment.adversarial_accuracy == 1.0  # Default when no adversarial

    def test_scores_by_category(self) -> None:
        """Test score aggregation by category."""
        now = datetime.now()
        category_metrics = {
            "IDENTITY": CategoryMetrics(
                category=QACategory.IDENTITY,
                total_questions=3,
                correct_count=2,
                partial_count=1,
                mean_score=0.833,
                mean_latency_ms=100.0,
            ),
            "TEMPORAL": CategoryMetrics(
                category=QACategory.TEMPORAL,
                total_questions=2,
                correct_count=1,
                partial_count=0,
                mean_score=0.5,
                mean_latency_ms=100.0,
            ),
        }

        assessment = AssessmentResult(
            question_results=[],
            category_metrics=category_metrics,
            conversation_metrics={},
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        by_cat = assessment.scores_by_category()
        assert "IDENTITY" in by_cat
        assert "TEMPORAL" in by_cat
        assert by_cat["IDENTITY"] == pytest.approx(0.833)
        assert by_cat["TEMPORAL"] == 0.5

    def test_accuracy_by_category(self) -> None:
        """Test accuracy aggregation by category."""
        now = datetime.now()
        category_metrics = {
            "IDENTITY": CategoryMetrics(
                category=QACategory.IDENTITY,
                total_questions=10,
                correct_count=8,
                partial_count=1,
                mean_score=0.85,
                mean_latency_ms=100.0,
            ),
            "ADVERSARIAL": CategoryMetrics(
                category=QACategory.ADVERSARIAL,
                total_questions=5,
                correct_count=3,
                partial_count=0,
                mean_score=0.6,
                mean_latency_ms=100.0,
            ),
        }

        assessment = AssessmentResult(
            question_results=[],
            category_metrics=category_metrics,
            conversation_metrics={},
            total_questions=15,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        by_cat = assessment.accuracy_by_category()
        assert by_cat["IDENTITY"] == 0.8  # 8/10
        assert by_cat["ADVERSARIAL"] == 0.6  # 3/5

    def test_get_summary(self, sample_results: list[QuestionResult]) -> None:
        """Test summary generation."""
        now = datetime.now()
        category_metrics = {
            "IDENTITY": CategoryMetrics(
                category=QACategory.IDENTITY,
                total_questions=3,
                correct_count=2,
                partial_count=1,
                mean_score=0.833,
                mean_latency_ms=100.0,
            ),
        }
        conversation_metrics = {
            "conv_1": ConversationMetrics(
                conversation_id="conv_1",
                sessions_ingested=2,
                turns_ingested=4,
                questions_assessed=5,
                correct_count=3,
                mean_score=0.7,
            ),
        }

        assessment = AssessmentResult(
            question_results=sample_results,
            category_metrics=category_metrics,
            conversation_metrics=conversation_metrics,
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=now,
            completed_at=now,
        )

        summary = assessment.get_summary()
        assert summary["total_questions"] == 5
        assert summary["correct"] == 3
        assert summary["accuracy"] == 0.6
        assert summary["conversations_assessed"] == 1
        assert "scores_by_category" in summary
        assert "accuracy_by_category" in summary


class TestLoCoMoPipeline:
    """Tests for LoCoMoPipeline class."""

    @pytest.fixture
    def sample_dataset(self) -> LoCoMoDataset:
        """Create a sample dataset for testing."""
        conv = create_sample_conversation("conv_1")
        return LoCoMoDataset(
            conversations=[conv],
            metadata={"source": "test"},
        )

    @pytest.fixture
    def pipeline(self) -> LoCoMoPipeline:
        """Create a pipeline with mock dependencies."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="OpenAI")
        judge = MockJudge()
        return LoCoMoPipeline(adapter, llm, judge)

    def test_pipeline_init(self) -> None:
        """Test pipeline initialization."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        judge = MockJudge()

        pipeline = LoCoMoPipeline(
            adapter,
            llm,
            judge,
            memory_search_limit=20,
            min_relevance_score=0.5,
            use_category_prompts=False,
            use_evidence_sessions=True,
        )

        assert pipeline._memory_search_limit == 20
        assert pipeline._min_relevance_score == 0.5
        assert pipeline._use_category_prompts is False
        assert pipeline._use_evidence_sessions is True

    def test_run_pipeline(self, pipeline: LoCoMoPipeline, sample_dataset: LoCoMoDataset) -> None:
        """Test running the complete pipeline."""
        result = pipeline.run(sample_dataset)

        assert result.total_questions == 2
        assert len(result.question_results) == 2
        assert result.ingestion_time_ms > 0
        assert result.assessment_time_ms > 0

    def test_run_pipeline_with_callback(
        self, pipeline: LoCoMoPipeline, sample_dataset: LoCoMoDataset
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
        self, pipeline: LoCoMoPipeline, sample_dataset: LoCoMoDataset
    ) -> None:
        """Test pipeline result metadata."""
        result = pipeline.run(sample_dataset)

        assert "memory_search_limit" in result.metadata
        assert "total_sessions" in result.metadata
        assert "total_turns_ingested" in result.metadata
        assert "total_conversations" in result.metadata
        assert result.metadata["total_conversations"] == 1
        assert result.metadata["total_sessions"] == 2
        assert result.metadata["total_turns_ingested"] == 4

    def test_run_pipeline_max_conversations(self, pipeline: LoCoMoPipeline) -> None:
        """Test pipeline with max_conversations limit."""
        # Create 3 conversations
        convs = [create_sample_conversation(f"conv_{i}") for i in range(3)]
        dataset = LoCoMoDataset(conversations=convs, metadata={})

        # Limit to 1 conversation
        result = pipeline.run(dataset, max_conversations=1)

        assert result.metadata["total_conversations"] == 1
        # Each conversation has 2 questions
        assert result.total_questions == 2

    def test_run_pipeline_category_filter(
        self, pipeline: LoCoMoPipeline, sample_dataset: LoCoMoDataset
    ) -> None:
        """Test pipeline with category filter."""
        # Filter to only IDENTITY questions
        result = pipeline.run(sample_dataset, categories=[QACategory.IDENTITY])

        # Only 1 question is IDENTITY in our sample
        assert result.total_questions == 1
        assert result.question_results[0].category == QACategory.IDENTITY

    def test_run_single_question(self, sample_dataset: LoCoMoDataset) -> None:
        """Test running a single question."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="OpenAI")
        judge = MockJudge()

        pipeline = LoCoMoPipeline(adapter, llm, judge)

        # Create and set up agent
        agent = LoCoMoAgent(adapter, llm)
        agent.ingest_all_conversations(sample_dataset.conversations)

        # Run single question
        question = sample_dataset.all_questions()[0]
        result = pipeline.run_single_question(agent, question)

        assert result.question_id == question.question_id
        assert result.latency_ms > 0
        assert result.judgment.result == JudgmentResult.CORRECT

    def test_category_metrics_computed(
        self, pipeline: LoCoMoPipeline, sample_dataset: LoCoMoDataset
    ) -> None:
        """Test that category metrics are computed correctly."""
        result = pipeline.run(sample_dataset)

        assert len(result.category_metrics) > 0
        assert "IDENTITY" in result.category_metrics
        assert "TEMPORAL" in result.category_metrics

        identity_metrics = result.category_metrics["IDENTITY"]
        assert identity_metrics.total_questions == 1

    def test_conversation_metrics_computed(
        self, pipeline: LoCoMoPipeline, sample_dataset: LoCoMoDataset
    ) -> None:
        """Test that conversation metrics are computed correctly."""
        result = pipeline.run(sample_dataset)

        assert len(result.conversation_metrics) == 1
        assert "conv_1" in result.conversation_metrics

        conv_metrics = result.conversation_metrics["conv_1"]
        assert conv_metrics.conversation_id == "conv_1"
        assert conv_metrics.sessions_ingested == 2
        assert conv_metrics.turns_ingested == 4
        assert conv_metrics.questions_assessed == 2


class TestAdversarialHandling:
    """Tests for adversarial question handling."""

    def test_check_adversarial_handling_detects_premise(self) -> None:
        """Test that adversarial handling detection works."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        judge = MockJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        # These should be detected as identifying the false premise
        detected_answers = [
            "The premise is incorrect - Alice works at OpenAI",
            "That's not accurate, Alice never worked at Microsoft",
            "This doesn't match the conversation - Alice said OpenAI",
            "Contrary to what you stated, she works at OpenAI",
            "Alice never mentioned working at Microsoft",
            "That's not correct - she works elsewhere",
            "Based on an incorrect assumption about her job",
        ]

        for answer in detected_answers:
            assert pipeline._check_adversarial_handling(answer) is True, f"Should detect: {answer}"

    def test_check_adversarial_handling_misses_normal(self) -> None:
        """Test that normal answers aren't flagged as adversarial handling."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        judge = MockJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        # These should NOT be detected
        normal_answers = [
            "Alice works at Microsoft as a software engineer",
            "She started at Microsoft in 2020",
            "I don't know where Alice works",
            "Based on the conversation, she is employed",
        ]

        for answer in normal_answers:
            assert pipeline._check_adversarial_handling(answer) is False, (
                f"Should not detect: {answer}"
            )

    def test_adversarial_question_flow(self) -> None:
        """Test full flow with adversarial questions."""
        adapter = MockAdapter()
        # LLM will identify the false premise
        llm = MockLLMClient(default_response="The premise is incorrect - Alice works at OpenAI")
        judge = MockJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        # Create conversation with adversarial question
        conv = create_sample_conversation("conv_1")
        adv_q = create_adversarial_question("conv_1")
        conv = LoCoMoConversation(
            sample_id=conv.sample_id,
            sessions=conv.sessions,
            questions=[*conv.questions, adv_q],
        )
        dataset = LoCoMoDataset(conversations=[conv], metadata={})

        result = pipeline.run(dataset)

        # Find the adversarial result
        adv_result = next(r for r in result.question_results if r.is_adversarial)
        assert adv_result.adversarial_handled is True


class TestLoCoMoPipelineIntegration:
    """Integration tests for the LoCoMo pipeline."""

    def test_full_pipeline_with_mock_components(self) -> None:
        """Test the full pipeline flow with all mock components."""
        adapter = MockAdapter()
        llm = MockLLMClient(
            default_response="OpenAI",
            responses={
                "skiing": "Last weekend",
            },
        )
        judge = MockJudge(default_result=JudgmentResult.CORRECT, default_score=1.0)

        pipeline = LoCoMoPipeline(adapter, llm, judge)

        conv = create_sample_conversation("conv_1")
        dataset = LoCoMoDataset(conversations=[conv], metadata={})

        result = pipeline.run(dataset)

        assert result.accuracy == 1.0
        assert result.mean_score == 1.0
        assert result.correct_count == 2
        assert len(result.question_results) == 2
        # Pipeline now uses individual judge calls for progress logging
        assert judge.call_count == 2

    def test_pipeline_with_multiple_conversations(self) -> None:
        """Test pipeline with multiple conversations."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="Test answer")
        judge = MockJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        # Create 3 conversations
        convs = [create_sample_conversation(f"conv_{i}") for i in range(3)]
        dataset = LoCoMoDataset(conversations=convs, metadata={})

        result = pipeline.run(dataset)

        assert result.metadata["total_conversations"] == 3
        assert len(result.conversation_metrics) == 3
        # Each conversation has 2 questions
        assert result.total_questions == 6

    def test_pipeline_with_mixed_results(self) -> None:
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

            def judge(
                self,
                question: str,
                reference_answer: str,
                model_answer: str,
                *,
                skip_cache: bool = False,  # noqa: ARG002
            ) -> Judgment:
                res, score = self.results[self.idx % len(self.results)]
                self.idx += 1
                return Judgment(
                    result=res,
                    score=score,
                    reasoning="Test",
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
                skip_cache: bool = False,  # noqa: ARG002
            ) -> list[Judgment]:
                return [self.judge(q, ref, ans) for q, ref, ans in items]

        judge = MixedJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        # Create 2 conversations = 4 questions (but we'll get 3 patterns cycling)
        convs = [create_sample_conversation(f"conv_{i}") for i in range(2)]
        dataset = LoCoMoDataset(conversations=convs, metadata={})

        result = pipeline.run(dataset)

        # With 4 questions cycling through 3 patterns:
        # correct, partial, incorrect, correct
        assert result.correct_count == 2
        assert result.partial_count == 1
        assert result.incorrect_count == 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_questions(self) -> None:
        """Test pipeline with no questions."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        judge = MockJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        conv = LoCoMoConversation(
            sample_id="conv_1",
            sessions=[
                LoCoMoSession(
                    session_num=1,
                    timestamp="2023-01-15 10:00:00",
                    turns=[
                        LoCoMoTurn(dia_id="D1:1", speaker="A", text="Hello", session_num=1),
                    ],
                    speaker_a="A",
                    speaker_b="B",
                ),
            ],
            questions=[],  # No questions
        )
        dataset = LoCoMoDataset(conversations=[conv], metadata={})

        result = pipeline.run(dataset)

        assert result.total_questions == 0
        assert result.accuracy == 0.0
        assert result.mean_score == 0.0

    def test_empty_sessions(self) -> None:
        """Test pipeline with no sessions."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="I don't know")
        judge = MockJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        conv = LoCoMoConversation(
            sample_id="conv_1",
            sessions=[],  # No sessions
            questions=[
                LoCoMoQuestion(
                    question_id="q1",
                    conversation_id="conv_1",
                    category=QACategory.IDENTITY,
                    question="Who is Alice?",
                    answer="Unknown",
                    evidence=[],
                ),
            ],
        )
        dataset = LoCoMoDataset(conversations=[conv], metadata={})

        result = pipeline.run(dataset)

        assert result.total_questions == 1
        assert result.metadata["total_turns_ingested"] == 0

    def test_empty_conversations(self) -> None:
        """Test pipeline with no conversations."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        judge = MockJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        dataset = LoCoMoDataset(conversations=[], metadata={})

        result = pipeline.run(dataset)

        assert result.total_questions == 0
        assert result.metadata["total_conversations"] == 0

    def test_single_question_method(self) -> None:
        """Test run_single_question with adversarial question."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="The premise is incorrect")
        judge = MockJudge()
        pipeline = LoCoMoPipeline(adapter, llm, judge)

        conv = create_sample_conversation("conv_1")
        agent = LoCoMoAgent(adapter, llm)
        agent.ingest_conversation(conv)

        adv_q = create_adversarial_question("conv_1")
        result = pipeline.run_single_question(agent, adv_q)

        assert result.is_adversarial is True
        assert result.adversarial_handled is True
