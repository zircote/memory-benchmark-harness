"""MemoryAgentBench agent wrapper.

This module provides the agent wrapper that integrates memory systems
with the MemoryAgentBench evaluation framework.

The wrapper handles:
1. Context ingestion into memory
2. Question answering with memory retrieval
3. Conflict resolution with version history (for git-notes)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.adapters.base import MemorySystemAdapter
from src.benchmarks.memoryagentbench.dataset import (
    Competency,
    MemoryAgentBenchQuestion,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM clients used in answering questions."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        ...


@dataclass(slots=True)
class AnswerResult:
    """Result of answering a question.

    Attributes:
        question_id: The question that was answered
        answer: The generated answer
        retrieved_memories: Memories used in generating the answer
        confidence: Confidence score if available
        metadata: Additional result metadata
    """

    question_id: str
    answer: str
    retrieved_memories: list[dict[str, Any]]
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def used_memory(self) -> bool:
        """Check if memory was used to answer."""
        return len(self.retrieved_memories) > 0


@dataclass(slots=True)
class MemoryAgentBenchAgent:
    """Agent wrapper for MemoryAgentBench evaluation.

    This agent integrates a memory system with LLM-based question answering,
    specifically supporting conflict resolution with version history.

    Attributes:
        adapter: Memory system adapter for storage/retrieval
        llm: LLM client for generating answers
        use_version_history: Whether to use git version history for conflicts
        retrieval_limit: Maximum memories to retrieve per query
        min_relevance_score: Minimum score for memory retrieval
    """

    adapter: MemorySystemAdapter
    llm: LLMClient
    use_version_history: bool = True
    retrieval_limit: int = 10
    min_relevance_score: float = 0.5

    def _chunk_context(self, context: str, chunk_size: int = 2000) -> list[str]:
        """Split context into chunks for memory ingestion.

        Args:
            context: Full context string
            chunk_size: Target size per chunk (in characters)

        Returns:
            List of context chunks
        """
        # Split on paragraph boundaries first
        paragraphs = context.split("\n\n")
        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def ingest_context(
        self,
        question: MemoryAgentBenchQuestion,
    ) -> int:
        """Ingest the question's context into memory.

        Args:
            question: The question containing context to ingest

        Returns:
            Number of memory chunks stored
        """
        # Clear any previous context
        self.adapter.clear()

        # Chunk and store the context
        chunks = self._chunk_context(question.context)

        stored = 0
        for i, chunk in enumerate(chunks):
            metadata = {
                "question_id": question.question_id,
                "competency": question.competency.value,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }

            # Add keypoints if available and relevant to this chunk
            if question.keypoints:
                relevant_keypoints = [
                    kp for kp in question.keypoints if kp.lower() in chunk.lower()
                ]
                if relevant_keypoints:
                    metadata["keypoints"] = relevant_keypoints

            # Add previous events if available
            if question.previous_events:
                relevant_events = [
                    ev for ev in question.previous_events if ev.lower() in chunk.lower()
                ]
                if relevant_events:
                    metadata["events"] = relevant_events

            result = self.adapter.add(chunk, metadata)
            if result.success:
                stored += 1
            else:
                logger.warning(f"Failed to store chunk {i}: {result.error}")

        logger.debug(f"Ingested {stored}/{len(chunks)} chunks for question {question.question_id}")
        return stored

    def _build_system_prompt(self, competency: Competency) -> str:
        """Build system prompt based on competency type."""
        base_prompt = (
            "You are a helpful AI assistant with access to a memory system. "
            "Use the provided context and retrieved memories to answer questions accurately. "
        )

        if competency == Competency.CONFLICT_RESOLUTION:
            return base_prompt + (
                "When you encounter conflicting information, prioritize the most recent "
                "or explicitly updated information. Pay attention to dates, corrections, "
                "and updates that supersede earlier information. If version history is "
                "available, use it to identify the current correct state."
            )
        elif competency == Competency.ACCURATE_RETRIEVAL:
            return base_prompt + (
                "Focus on finding and extracting precise information from the provided "
                "context. For multi-hop questions, trace connections between entities "
                "step by step."
            )
        elif competency == Competency.TEST_TIME_LEARNING:
            return base_prompt + (
                "Learn from any demonstrations or examples provided. Apply patterns "
                "and knowledge from examples to answer new questions."
            )
        elif competency == Competency.LONG_RANGE_UNDERSTANDING:
            return base_prompt + (
                "Form a comprehensive understanding of the entire context. "
                "Consider relationships and patterns across the full document, "
                "not just local fragments."
            )
        else:
            return base_prompt

    def _build_prompt(
        self,
        question: MemoryAgentBenchQuestion,
        retrieved_memories: list[dict[str, Any]],
    ) -> str:
        """Build the question prompt with retrieved context."""
        parts: list[str] = []

        # Add demonstration if available
        if question.demo:
            parts.append("## Example\n")
            parts.append(question.demo)
            parts.append("\n")

        # Add retrieved memories
        if retrieved_memories:
            parts.append("## Retrieved Context\n")
            for i, mem in enumerate(retrieved_memories, 1):
                content = mem.get("content", "")
                score = mem.get("score", 0)
                parts.append(f"### Memory {i} (relevance: {score:.2f})\n{content}\n")

            # For conflict resolution, add version history if available
            if question.competency == Competency.CONFLICT_RESOLUTION and self.use_version_history:
                for mem in retrieved_memories:
                    history = mem.get("version_history", [])
                    if history:
                        parts.append("\n### Version History\n")
                        for entry in history:
                            version = entry.get("version", "?")
                            date = entry.get("date", "unknown")
                            change = entry.get("content", entry.get("change", ""))
                            parts.append(f"- Version {version} ({date}): {change}\n")

        # Add previous events if relevant
        if question.previous_events:
            parts.append("\n## Previous Events\n")
            for event in question.previous_events:
                parts.append(f"- {event}\n")

        # Add the question
        parts.append("\n## Question\n")
        parts.append(question.question_text)

        # Add date context if available
        if question.question_date:
            parts.append(f"\n\n(Question date: {question.question_date})")

        parts.append("\n\n## Answer\n")
        parts.append("Please provide a concise, accurate answer based on the context above.")

        return "".join(parts)

    def retrieve_for_question(
        self,
        question: MemoryAgentBenchQuestion,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories for a question.

        Args:
            question: The question to retrieve memories for

        Returns:
            List of retrieved memories with content and scores
        """
        # Build retrieval query from question
        query = question.question_text

        # Add keypoints to query if available
        if question.keypoints:
            query += " " + " ".join(question.keypoints[:3])

        # Retrieve memories
        memories = self.adapter.search(
            query=query,
            limit=self.retrieval_limit,
            min_score=self.min_relevance_score,
        )

        # Convert to dictionaries
        retrieved: list[dict[str, Any]] = []
        for mem in memories:
            mem_dict = {
                "id": mem.memory_id,
                "content": mem.content,
                "score": mem.score,
                "metadata": mem.metadata,
            }

            # For git-notes adapter, try to get version history
            if self.use_version_history and hasattr(self.adapter, "get_history"):
                try:
                    history = self.adapter.get_history(mem.memory_id)
                    if history:
                        mem_dict["version_history"] = history
                except Exception as e:
                    logger.debug(f"Could not get version history: {e}")

            retrieved.append(mem_dict)

        return retrieved

    def answer_question(
        self,
        question: MemoryAgentBenchQuestion,
        ingest: bool = True,
    ) -> AnswerResult:
        """Answer a question using memory-augmented generation.

        Args:
            question: The question to answer
            ingest: Whether to ingest context first (default True)

        Returns:
            AnswerResult with the generated answer
        """
        # Ingest context if requested
        if ingest:
            self.ingest_context(question)

        # Retrieve relevant memories
        retrieved = self.retrieve_for_question(question)

        # Build prompts
        system_prompt = self._build_system_prompt(question.competency)
        prompt = self._build_prompt(question, retrieved)

        # Generate answer
        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=512,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating answer: {e}"

        # Clean answer
        answer = self._clean_answer(answer)

        return AnswerResult(
            question_id=question.question_id,
            answer=answer,
            retrieved_memories=retrieved,
            metadata={
                "competency": question.competency.value,
                "difficulty": question.difficulty.value,
                "context_length": question.context_length,
                "num_retrieved": len(retrieved),
            },
        )

    def _clean_answer(self, answer: str) -> str:
        """Clean and normalize the generated answer."""
        # Remove common prefixes
        prefixes = [
            "the answer is",
            "answer:",
            "based on the context,",
            "according to the information provided,",
        ]
        answer_lower = answer.lower().strip()
        for prefix in prefixes:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix) :].strip()
                break

        # Remove leading/trailing whitespace and quotes
        answer = answer.strip().strip('"').strip("'")

        return answer

    def answer_with_conflict_check(
        self,
        question: MemoryAgentBenchQuestion,
    ) -> AnswerResult:
        """Answer a conflict resolution question with explicit version checking.

        This method specifically handles conflict resolution by:
        1. Identifying potentially conflicting information
        2. Using version history to determine current state
        3. Providing the most up-to-date answer

        Args:
            question: A conflict resolution question

        Returns:
            AnswerResult with resolution details
        """
        if question.competency != Competency.CONFLICT_RESOLUTION:
            logger.warning(
                f"answer_with_conflict_check called for non-CR question: {question.question_id}"
            )

        # Ingest context
        self.ingest_context(question)

        # Retrieve with higher limit to catch conflicts
        original_limit = self.retrieval_limit
        self.retrieval_limit = min(original_limit * 2, 20)

        retrieved = self.retrieve_for_question(question)

        self.retrieval_limit = original_limit

        # Analyze for conflicts
        conflicts = self._detect_conflicts(retrieved)

        # Build enhanced prompt with conflict awareness
        system_prompt = self._build_system_prompt(question.competency)

        if conflicts:
            system_prompt += (
                f"\n\nIMPORTANT: {len(conflicts)} potential conflicts detected. "
                "Use version history or recency to resolve."
            )

        prompt = self._build_prompt(question, retrieved)

        # Generate answer
        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=512,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating answer: {e}"

        answer = self._clean_answer(answer)

        return AnswerResult(
            question_id=question.question_id,
            answer=answer,
            retrieved_memories=retrieved,
            metadata={
                "competency": question.competency.value,
                "difficulty": question.difficulty.value,
                "conflicts_detected": len(conflicts),
                "conflict_details": conflicts,
            },
        )

    def _detect_conflicts(
        self,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Detect potential conflicts between memories.

        This is a heuristic-based detection that looks for:
        - Same entities with different values
        - Temporal updates (was X, now Y)
        - Explicit contradictions

        Args:
            memories: Retrieved memories to analyze

        Returns:
            List of detected conflicts with details
        """
        conflicts: list[dict[str, Any]] = []

        # Pattern for finding entity-value pairs
        patterns = [
            r"(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(\w+(?:\s+\w+)?)",
            r"(\w+(?:\s+\w+)?)\s*:\s*(\w+(?:\s+\w+)?)",
        ]

        # Extract facts from each memory
        facts_by_entity: dict[str, list[tuple[str, str, int]]] = {}

        for idx, mem in enumerate(memories):
            content = mem.get("content", "")

            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for entity, value in matches:
                    entity = entity.lower().strip()
                    value = value.lower().strip()

                    if entity not in facts_by_entity:
                        facts_by_entity[entity] = []
                    facts_by_entity[entity].append((value, content[:100], idx))

        # Find entities with multiple different values
        for entity, facts in facts_by_entity.items():
            unique_values = {f[0] for f in facts}
            if len(unique_values) > 1:
                conflicts.append(
                    {
                        "entity": entity,
                        "values": list(unique_values),
                        "memory_indices": [f[2] for f in facts],
                    }
                )

        return conflicts
