"""Agentic LoCoMo wrapper with active memory management.

This module provides an agent that actively decides what to save and recall,
rather than passively ingesting all content. It gives the LLM control over:

1. **During ingestion**: Extract and save key facts rather than raw turns
2. **During Q&A**: Iteratively search and refine queries to find answers

This approach is inspired by how human memory works - we don't store
every word of every conversation, but rather extract and remember key facts.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.adapters.base import MemorySystemAdapter
    from src.benchmarks.locomo.dataset import LoCoMoConversation, LoCoMoQuestion
    from src.benchmarks.locomo.wrapper import LLMClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgenticAnswer:
    """Answer from the agentic wrapper.

    Attributes:
        question_id: The question being answered
        answer: The generated answer
        search_iterations: Number of search iterations performed
        memories_saved: Number of memories saved during ingestion
        is_abstention: Whether the agent abstained from answering
        latency_ms: Total response time in milliseconds
        metadata: Additional metadata (searches performed, etc.)
    """

    question_id: str
    answer: str
    search_iterations: int
    memories_saved: int = 0
    is_abstention: bool = False
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AgenticLoCoMoWrapper:
    """Agentic wrapper that lets the LLM control memory operations.

    Unlike the passive LoCoMoAgent that ingests every turn, this wrapper:
    1. Asks the LLM to extract important facts from conversations
    2. Allows the LLM to perform multiple searches when answering
    3. Uses structured tool calls for memory operations

    This simulates how MCP-based memory would work in practice.

    Example:
        ```python
        adapter = SubcogAdapter()
        llm = OpenAIClient(model="gpt-5-mini")
        wrapper = AgenticLoCoMoWrapper(adapter, llm)

        # Agent extracts and saves key facts
        wrapper.ingest_conversation(conversation)

        # Agent iteratively searches to answer
        answer = wrapper.answer_question(question)
        ```
    """

    # System prompt for fact extraction during ingestion
    EXTRACTION_PROMPT = '''You are a memory extraction agent. Your task is to extract important facts from a conversation that would be useful for answering future questions.

For each important fact, output a JSON object with:
- "fact": The extracted fact as a clear, self-contained statement
- "entities": List of people, places, or things mentioned
- "type": One of "identity", "temporal", "relationship", "activity", "preference"

Output one fact per line as JSON. Only extract facts that are:
1. Specific (names, dates, places, activities)
2. Memorable (would be useful for future questions)
3. Self-contained (understandable without context)

Examples of good facts:
{"fact": "Caroline moved from Sweden to the US 5 years ago", "entities": ["Caroline", "Sweden", "US"], "type": "temporal"}
{"fact": "Melanie has two children named Josh and Emily", "entities": ["Melanie", "Josh", "Emily"], "type": "identity"}
{"fact": "They went camping at Yellowstone in August 2023", "entities": ["Yellowstone"], "type": "activity"}

Only output JSON lines, no other text.'''

    # System prompt for answering with iterative search
    ANSWER_PROMPT = '''You are a question-answering agent with access to a memory search tool.

To search memories, output: SEARCH: <query>

You can search multiple times to find the answer. After each search, you'll see relevant memories.

When you have enough information, output: ANSWER: <your answer>

If you cannot find the information after 3 searches, output: ANSWER: I cannot find this information in the conversation history.

Strategy:
1. Start with key terms from the question
2. If not found, try synonyms or related terms
3. If still not found, try broader or narrower searches
4. Connect facts across multiple memories if needed'''

    def __init__(
        self,
        adapter: MemorySystemAdapter,
        llm: LLMClient,
        *,
        max_search_iterations: int = 3,
        memories_per_search: int = 10,
        batch_size: int = 1,  # Process one session at a time for better extraction
    ) -> None:
        """Initialize the agentic wrapper.

        Args:
            adapter: Memory system adapter for storage and retrieval
            llm: LLM client for extraction and answering
            max_search_iterations: Maximum searches per question
            memories_per_search: Memories to return per search
            batch_size: Sessions to process together for extraction
        """
        self._adapter = adapter
        self._llm = llm
        self._max_search_iterations = max_search_iterations
        self._memories_per_search = memories_per_search
        self._batch_size = batch_size
        self._total_facts_saved = 0

    def ingest_conversation(
        self,
        conversation: LoCoMoConversation,
        *,
        show_progress: bool = False,
    ) -> int:
        """Ingest a conversation using LLM-based fact extraction.

        Instead of storing every turn, asks the LLM to extract
        important facts that would be useful for future questions.

        Args:
            conversation: The LoCoMo conversation to ingest
            show_progress: Show progress during extraction

        Returns:
            Number of facts extracted and saved
        """
        total_saved = 0

        # Process sessions in batches
        for i in range(0, len(conversation.sessions), self._batch_size):
            batch = conversation.sessions[i : i + self._batch_size]

            # Format sessions for extraction
            context_parts = []
            for session in batch:
                timestamp = session.timestamp or f"Session {session.session_num}"
                turns_text = "\n".join(
                    f"{t.speaker}: {t.text}" for t in session.turns
                )
                context_parts.append(f"[{timestamp}]\n{turns_text}")

            context = "\n\n---\n\n".join(context_parts)

            # Ask LLM to extract facts (with retry for empty responses)
            response = self._llm.complete(
                system=self.EXTRACTION_PROMPT,
                messages=[{"role": "user", "content": context}],
                temperature=0.0,
            )

            # Retry once if empty response
            if not response.content.strip():
                logger.warning("Empty response from LLM, retrying...")
                response = self._llm.complete(
                    system=self.EXTRACTION_PROMPT,
                    messages=[{"role": "user", "content": context}],
                    temperature=0.0,
                )

            # Debug: Log first 500 chars of response
            if response.content:
                logger.debug(f"LLM extraction response (first 300 chars): {response.content[:300]}")
            else:
                logger.warning(f"Empty LLM response for sessions {i+1}-{i+len(batch)}")

            # Parse and save extracted facts
            facts_saved = self._save_extracted_facts(
                response.content,
                conversation.sample_id,
            )
            total_saved += facts_saved

            if show_progress:
                logger.info(
                    f"Sessions {i+1}-{min(i+self._batch_size, len(conversation.sessions))}: "
                    f"Extracted {facts_saved} facts"
                )

        self._total_facts_saved += total_saved
        logger.info(
            f"Ingested {conversation.sample_id}: "
            f"Extracted {total_saved} facts from {len(conversation.sessions)} sessions"
        )

        return total_saved

    def _save_extracted_facts(
        self,
        llm_output: str,
        conversation_id: str,
    ) -> int:
        """Parse LLM output and save extracted facts.

        Args:
            llm_output: Raw LLM response with JSON facts
            conversation_id: ID of the source conversation

        Returns:
            Number of facts successfully saved
        """
        saved_count = 0
        parse_errors = 0

        # Try to extract JSON objects from the output
        # Handle both line-by-line and mixed content
        lines = llm_output.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Find JSON objects in the line (may have surrounding text)
            json_matches = re.findall(r'\{[^{}]*\}', line)

            for json_str in json_matches:
                try:
                    fact_data = json.loads(json_str)
                    fact_text = fact_data.get("fact", "")
                    if not fact_text:
                        continue

                    entities = fact_data.get("entities", [])
                    fact_type = fact_data.get("type", "general")

                    # Build enriched content with entities
                    content = fact_text
                    if entities:
                        content += f"\n[Entities: {', '.join(str(e) for e in entities)}]"

                    # Save to memory
                    result = self._adapter.add(
                        content=content,
                        metadata={
                            "conversation_id": conversation_id,
                            "type": fact_type,
                            "entities": entities,
                            "namespace": "locomo_facts",
                        },
                    )

                    if result.success:
                        saved_count += 1
                        logger.debug(f"Saved fact: {fact_text[:50]}...")

                except json.JSONDecodeError as e:
                    parse_errors += 1
                    logger.debug(f"JSON parse error: {e} for: {json_str[:50]}...")
                    continue

        if parse_errors > 0:
            logger.debug(f"Total JSON parse errors: {parse_errors}")

        return saved_count

    def answer_question(
        self,
        question: LoCoMoQuestion,
    ) -> AgenticAnswer:
        """Answer a question using iterative search.

        Allows the LLM to perform multiple searches to find
        the answer, refining queries based on what it finds.

        Args:
            question: The LoCoMo question to answer

        Returns:
            AgenticAnswer with the result and metadata
        """
        start_time = time.perf_counter()

        searches_performed: list[dict[str, Any]] = []
        accumulated_context = ""
        answer = ""
        iterations = 0

        # Initial prompt with the question
        current_messages = [
            {
                "role": "user",
                "content": f"Question: {question.question}\n\nUse SEARCH: <query> to find relevant information, then ANSWER: <your answer>",
            }
        ]

        while iterations < self._max_search_iterations:
            iterations += 1

            # Get LLM response
            response = self._llm.complete(
                system=self.ANSWER_PROMPT,
                messages=current_messages,
                temperature=0.0,
            )

            response_text = response.content.strip()

            # Check for ANSWER
            answer_match = re.search(
                r"ANSWER:\s*(.+)",
                response_text,
                re.IGNORECASE | re.DOTALL,
            )
            if answer_match:
                answer = answer_match.group(1).strip()
                break

            # Check for SEARCH
            search_match = re.search(
                r"SEARCH:\s*(.+?)(?:\n|$)",
                response_text,
                re.IGNORECASE,
            )
            if search_match:
                query = search_match.group(1).strip()

                # Perform search
                memories = self._adapter.search(
                    query=query,
                    limit=self._memories_per_search,
                    metadata_filter={"conversation_id": question.conversation_id},
                )

                # Format results
                if memories:
                    memory_texts = [
                        f"- {m.content}" for m in memories
                    ]
                    search_results = "\n".join(memory_texts)
                else:
                    search_results = "(No results found for this query)"

                searches_performed.append({
                    "query": query,
                    "results_count": len(memories),
                })

                # Add results to conversation
                current_messages.append({
                    "role": "assistant",
                    "content": response_text,
                })
                current_messages.append({
                    "role": "user",
                    "content": f"Search results for '{query}':\n{search_results}\n\nContinue searching or provide your ANSWER:",
                })

                accumulated_context += f"\n{search_results}"
            else:
                # No search or answer found, force an answer
                answer = response_text
                break

        # If we exhausted iterations without an answer, use last response
        if not answer:
            answer = "I cannot find this information after multiple searches."

        latency_ms = (time.perf_counter() - start_time) * 1000

        return AgenticAnswer(
            question_id=question.question_id,
            answer=answer,
            search_iterations=iterations,
            memories_saved=self._total_facts_saved,
            is_abstention=self._is_abstention(answer),
            latency_ms=latency_ms,
            metadata={
                "searches": searches_performed,
                "conversation_id": question.conversation_id,
                "category": question.category.name if question.category else None,
            },
        )

    def _is_abstention(self, answer: str) -> bool:
        """Check if answer indicates abstention."""
        answer_lower = answer.lower()
        return any(
            phrase in answer_lower
            for phrase in [
                "cannot find",
                "can't find",
                "no information",
                "not mentioned",
                "not discussed",
                "unable to find",
            ]
        )

    def clear_memory(self) -> bool:
        """Clear all memories."""
        result = self._adapter.clear()
        if result.success:
            self._total_facts_saved = 0
        return result.success

    def get_stats(self) -> dict[str, Any]:
        """Get wrapper statistics."""
        return {
            "total_facts_saved": self._total_facts_saved,
            "max_search_iterations": self._max_search_iterations,
            "memories_per_search": self._memories_per_search,
            **self._adapter.get_stats(),
        }
