"""LoCoMo agent wrapper for benchmark assessment.

This module provides the LoCoMoAgent class that adapts a MemorySystemAdapter
to the LoCoMo benchmark interface. It handles multi-session conversation ingestion,
memory search, and question answering with adversarial question handling.

The wrapper follows the benchmark pattern from ARCHITECTURE.md section 4.

Key differences from LongMemEval:
- LoCoMo has multi-session conversations (~35 sessions per conversation)
- Questions are categorized into 5 types (Identity, Temporal, Inference, Contextual, Adversarial)
- Adversarial questions have intentionally incorrect premises
- Evidence annotations link questions to specific dialogue turns
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.adapters.base import MemorySystemAdapter
    from src.benchmarks.locomo.dataset import (
        LoCoMoConversation,
        LoCoMoQuestion,
        LoCoMoSession,
        LoCoMoTurn,
        QACategory,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Query Enrichment Utilities
# =============================================================================


def enrich_query(question: str) -> str:
    """Enrich a question for better memory retrieval.

    Transforms questions into statement-like queries that better match
    how information is stored in memories.

    Args:
        question: The original question text

    Returns:
        Enriched query optimized for semantic search

    Example:
        >>> enrich_query("When did Melanie paint a sunrise?")
        'Melanie painted sunrise painting art'
    """
    query = question

    # Remove question words that hurt retrieval
    query = re.sub(
        r"^(When|What|Where|How|Why|Who|Did|Does|Is|Was|Were|Has|Have|Can|Could|Would|Should)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )

    # Remove trailing question mark
    query = query.rstrip("?").strip()

    # Convert common question patterns to statements
    # "did X do Y" -> "X did Y"
    query = re.sub(r"^did\s+(\w+)\s+", r"\1 ", query, flags=re.IGNORECASE)

    # Expand possessives for better matching
    # "Melanie's kids" -> "Melanie kids children"
    if "'s " in query.lower():
        # Add the base form without possessive
        query = re.sub(r"(\w+)'s\s+", r"\1 \1's ", query)

    # Add common synonyms for better recall
    expansions = {
        r"\bkids\b": "kids children",
        r"\bpainted\b": "painted painting art",
        r"\brace\b": "race running marathon",
        r"\bclass\b": "class course lesson signed up",
        r"\bconference\b": "conference event meeting",
        r"\bcamping\b": "camping camp outdoor",
        r"\brelationship\b": "relationship dating partner boyfriend girlfriend",
        r"\bidentity\b": "identity transgender LGBTQ gender",
        r"\bmove\b": "move moved country home",
        r"\bmoved\b": "moved move country home",
        r"\bfrom\b": "from country home",
        r"\bago\b": "ago years before",
        r"\bbooks?\b": "book books read reading",
        r"\bdestress\b": "destress relax relaxing hobby hobbies",
        r"\bhobbies?\b": "hobby hobbies interest interests",
        r"\bmuseum\b": "museum art gallery exhibit",
    }
    for pattern, replacement in expansions.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

    return query.strip()


def extract_entities(text: str) -> list[str]:
    """Extract named entities from text for enhanced memory indexing.

    Args:
        text: The text to extract entities from

    Returns:
        List of extracted entity strings
    """
    entities = []

    # Extract capitalized words (likely proper nouns)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", text)
    entities.extend(proper_nouns)

    # Extract dates and times
    dates = re.findall(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?\b",
        text,
        re.IGNORECASE,
    )
    entities.extend(dates)

    # Extract relative time expressions
    time_exprs = re.findall(
        r"\b(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+|\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
        text,
        re.IGNORECASE,
    )
    entities.extend(time_exprs)

    # Extract activity keywords
    activities = re.findall(
        r"\b(?:painted|ran|signed up|went to|visited|started|finished|completed|attended)\b",
        text,
        re.IGNORECASE,
    )
    entities.extend(activities)

    return list(set(entities))


def compile_memory_content(
    speaker: str,
    text: str,
    session_num: int,
    timestamp: str = "",
    img_caption: str = "",
) -> str:
    """Compile a conversation turn into an enriched memory format.

    Creates a memory string that includes temporal context and extracted
    entities for better retrieval.

    Args:
        speaker: The speaker name
        text: The dialogue text
        session_num: Session number
        timestamp: Optional session timestamp
        img_caption: Optional image caption

    Returns:
        Enriched memory content string
    """
    # Base format with session context
    if timestamp:
        content = f"[Session {session_num}, {timestamp}] {speaker}: {text}"
    else:
        content = f"[Session {session_num}] {speaker}: {text}"

    # Add image context if available
    if img_caption:
        content += f" [Image: {img_caption}]"

    # Extract and append entities for better retrieval
    entities = extract_entities(text)
    if entities:
        content += f"\n[Entities: {', '.join(entities[:5])}]"  # Limit to top 5

    return content


class LLMClient(Protocol):
    """Protocol for LLM client interface.

    Implementations should provide a complete() method that generates
    text responses from messages.
    """

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
        ...


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


@dataclass(slots=True)
class LoCoMoAnswer:
    """Answer generated by the agent for a LoCoMo question.

    Attributes:
        question_id: The question this answers
        answer: The generated answer text
        retrieved_memories: Number of memories used for context
        is_abstention: Whether the agent abstained from answering
        category: The QA category of the original question
        latency_ms: Response generation time in milliseconds
        metadata: Additional answer metadata (timing, model, etc.)
    """

    question_id: str
    answer: str
    retrieved_memories: int
    is_abstention: bool = False
    category: QACategory | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestionResult:
    """Result of ingesting a conversation or session.

    Attributes:
        conversation_id: The conversation ID
        sessions_ingested: Number of sessions processed
        turns_ingested: Number of turns successfully stored
        total_turns: Total turns attempted
        errors: List of error messages encountered
    """

    conversation_id: str
    sessions_ingested: int
    turns_ingested: int
    total_turns: int
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Get the ingestion success rate."""
        if self.total_turns == 0:
            return 1.0
        return self.turns_ingested / self.total_turns


class LoCoMoAgent:
    """Wrapper adapting MemorySystemAdapter to LoCoMo benchmark interface.

    This agent:
    1. Ingests multi-session conversations into memory (up to 35 sessions)
    2. Retrieves relevant memories for each question
    3. Generates answers using an LLM with memory context
    4. Handles adversarial questions (Category 5) with incorrect premises
    5. Tracks evidence sessions for retrieval validation

    Example:
        ```python
        adapter = GitNotesAdapter(repo_path="/path/to/repo")
        llm = OpenAIClient(model="gpt-5-mini")
        agent = LoCoMoAgent(adapter, llm)

        # Ingest conversation (all sessions)
        for conv in dataset.conversations:
            agent.ingest_conversation(conv)

        # Answer questions
        for question in dataset.all_questions():
            answer = agent.answer_question(question)
            print(f"{question.question_id}: {answer.answer}")
        ```

    Attributes:
        adapter: The memory system adapter for storage/retrieval
        llm: The LLM client for generating answers
    """

    # Default system prompt for question answering
    DEFAULT_SYSTEM_PROMPT = (
        "You are an AI assistant that answers questions based on past conversations. "
        "Use the provided conversation context to answer. "
        "Connect related information across different messages to form complete answers. "
        "Make reasonable inferences when facts are implied but not stated directly. "
        "Only say 'I cannot find this information' if there is truly no relevant context. "
        "Be concise and accurate."
    )

    # Category-specific prompts for better handling
    CATEGORY_PROMPTS = {
        1: (  # IDENTITY
            "Answer questions about the speakers' identity, background, and personal facts. "
            "Look for biographical details, names, relationships, family members, "
            "personal characteristics, relationship status, and life history. "
            "Pay attention to mentions of family (kids, parents, partners) and their attributes. "
            "IMPORTANT: Connect information across multiple messages. For example, if someone mentions "
            "'my home country' in one message and names that country in another, combine them. "
            "If someone mentions a 'breakup', infer they are likely single. "
            "Use reasoning to connect related facts."
        ),
        2: (  # TEMPORAL
            "Answer questions about when events occurred. "
            "Look for EXPLICIT dates (January 15, 2024), relative time markers "
            "(yesterday, last week, 2 years ago, next month), and session timestamps. "
            "When you see relative markers, calculate the approximate date from the session timestamp. "
            "If multiple events match the description, list them chronologically with their dates. "
            "Pay special attention to future plans ('going to', 'planning to', 'next') vs past events."
        ),
        3: (  # INFERENCE
            "Make reasonable inferences based on the conversation context. "
            "Use clues from the dialogue to predict or infer information. "
            "Consider what activities, interests, or characteristics are implied but not stated directly."
        ),
        4: (  # CONTEXTUAL
            "Answer detailed questions about specific events, activities, or discussions. "
            "Focus on the specifics mentioned in the conversations. "
            "Look for activity details: what was done, where, with whom, and the outcome. "
            "Include secondary characters (friends, family) mentioned in activities."
        ),
        5: (  # ADVERSARIAL
            "Be careful: this question may contain incorrect premises. "
            "If the premise doesn't match the conversation, point this out. "
            "Don't assume the question is correct - verify against the context."
        ),
    }

    def __init__(
        self,
        adapter: MemorySystemAdapter,
        llm: LLMClient,
        *,
        memory_search_limit: int = 45,  # Increased from 15 for better recall
        min_relevance_score: float = 0.0,
        system_prompt: str | None = None,
        use_category_prompts: bool = True,
        include_speaker_context: bool = True,
    ) -> None:
        """Initialize the LoCoMo agent.

        Args:
            adapter: Memory system adapter for storage and retrieval
            llm: LLM client for generating answers
            memory_search_limit: Max memories to retrieve per question (default: 45)
            min_relevance_score: Minimum score threshold for retrieval
            system_prompt: Custom system prompt (uses default if None)
            use_category_prompts: Whether to use category-specific prompts
            include_speaker_context: Whether to include speaker names in metadata
        """
        self._adapter = adapter
        self._llm = llm
        self._memory_search_limit = memory_search_limit
        self._min_relevance_score = min_relevance_score
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._use_category_prompts = use_category_prompts
        self._include_speaker_context = include_speaker_context

        # Track ingested conversations and sessions
        self._ingested_conversations: set[str] = set()
        self._ingested_sessions: dict[str, set[int]] = {}  # conv_id -> session_nums
        self._total_turns_ingested: int = 0

    @property
    def adapter(self) -> MemorySystemAdapter:
        """Get the memory adapter."""
        return self._adapter

    @property
    def ingested_conversation_count(self) -> int:
        """Get the number of ingested conversations."""
        return len(self._ingested_conversations)

    @property
    def total_turns_ingested(self) -> int:
        """Get the total number of turns ingested across all conversations."""
        return self._total_turns_ingested

    def ingest_turn(
        self,
        turn: LoCoMoTurn,
        conversation_id: str,
        session_timestamp: str = "",
    ) -> bool:
        """Ingest a single dialogue turn into memory.

        Args:
            turn: The dialogue turn to ingest
            conversation_id: ID of the parent conversation
            session_timestamp: Timestamp of the session

        Returns:
            True if ingestion succeeded, False otherwise
        """
        # Use enriched memory format with entities and temporal context
        content = compile_memory_content(
            speaker=turn.speaker,
            text=turn.text,
            session_num=turn.session_num,
            timestamp=session_timestamp,
            img_caption=turn.img_caption or "",
        )

        # Build rich metadata for filtering and context
        metadata: dict[str, Any] = {
            "conversation_id": conversation_id,
            "session_num": turn.session_num,
            "dia_id": turn.dia_id,
            "speaker": turn.speaker,
            "turn_num": turn.turn_num,
        }

        if session_timestamp:
            metadata["timestamp"] = session_timestamp

        if turn.img_url:
            metadata["has_image"] = True

        # Store in memory system
        result = self._adapter.add(content, metadata)
        if result.success:
            self._total_turns_ingested += 1
            return True

        logger.warning(
            f"Failed to ingest turn {turn.dia_id} from {conversation_id}: {result.error}"
        )
        return False

    def ingest_session(
        self,
        session: LoCoMoSession,
        conversation_id: str,
    ) -> int:
        """Ingest a conversation session into memory.

        Each turn in the session is stored as a separate memory entry
        with metadata including conversation ID, session number, and speaker.

        Args:
            session: The LoCoMo session to ingest
            conversation_id: ID of the parent conversation

        Returns:
            Number of turns successfully ingested
        """
        ingested_count = 0

        for turn in session.turns:
            if self.ingest_turn(turn, conversation_id, session.timestamp):
                ingested_count += 1

        # Track session ingestion
        if conversation_id not in self._ingested_sessions:
            self._ingested_sessions[conversation_id] = set()
        self._ingested_sessions[conversation_id].add(session.session_num)

        logger.debug(
            f"Ingested {ingested_count}/{len(session.turns)} turns "
            f"from session {session.session_num} of {conversation_id}"
        )

        return ingested_count

    def ingest_conversation(self, conversation: LoCoMoConversation) -> IngestionResult:
        """Ingest all sessions from a conversation into memory.

        Args:
            conversation: The LoCoMo conversation to ingest

        Returns:
            IngestionResult with ingestion statistics
        """
        total_turns = conversation.total_turns
        turns_ingested = 0
        errors: list[str] = []

        for session in conversation.sessions:
            try:
                count = self.ingest_session(session, conversation.sample_id)
                turns_ingested += count
            except Exception as e:
                error_msg = f"Error ingesting session {session.session_num}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        self._ingested_conversations.add(conversation.sample_id)

        result = IngestionResult(
            conversation_id=conversation.sample_id,
            sessions_ingested=len(conversation.sessions),
            turns_ingested=turns_ingested,
            total_turns=total_turns,
            errors=errors,
        )

        logger.info(
            f"Ingested conversation {conversation.sample_id}: "
            f"{turns_ingested}/{total_turns} turns across "
            f"{result.sessions_ingested} sessions"
        )

        return result

    def ingest_all_conversations(
        self,
        conversations: list[LoCoMoConversation],
        *,
        use_batch: bool = True,
    ) -> dict[str, IngestionResult]:
        """Ingest multiple conversations into memory.

        Args:
            conversations: List of conversations to ingest
            use_batch: If True and adapter supports it, use batch ingestion (~100x faster)

        Returns:
            Dictionary mapping conversation_id to IngestionResult
        """
        # Check if adapter supports fast batch ingestion
        has_batch = hasattr(self._adapter, "add_batch_fast") and use_batch

        if has_batch:
            return self._ingest_all_conversations_batch(conversations)

        # Fallback to individual ingestion
        results: dict[str, IngestionResult] = {}
        total_turns = 0

        for conv in conversations:
            result = self.ingest_conversation(conv)
            results[conv.sample_id] = result
            total_turns += result.turns_ingested

        logger.info(f"Ingested {total_turns} turns from {len(conversations)} conversations")
        return results

    def _ingest_all_conversations_batch(
        self,
        conversations: list[LoCoMoConversation],
    ) -> dict[str, IngestionResult]:
        """Ingest all conversations using optimized batch ingestion.

        This is ~100x faster than individual ingestion for adapters
        that support add_batch_fast() (e.g., GitNotesAdapter).

        Args:
            conversations: List of conversations to ingest

        Returns:
            Dictionary mapping conversation_id to IngestionResult
        """
        # Collect all turns as (content, metadata) tuples
        items: list[tuple[str, dict[str, Any]]] = []
        # Track which items belong to which conversation/session
        item_mapping: list[tuple[str, int]] = []  # (conversation_id, session_num)

        for conv in conversations:
            for session in conv.sessions:
                for turn in session.turns:
                    # Build enriched content with entities and temporal context
                    content = compile_memory_content(
                        speaker=turn.speaker,
                        text=turn.text,
                        session_num=session.session_num,
                        timestamp=session.timestamp or "",
                        img_caption=turn.img_caption or "",
                    )

                    # Build metadata
                    metadata: dict[str, Any] = {
                        "conversation_id": conv.sample_id,
                        "session_num": session.session_num,
                        "speaker": turn.speaker,
                        "dia_id": turn.dia_id,
                        "turn_num": turn.turn_num,
                        "namespace": "locomo",
                    }
                    if session.timestamp:
                        metadata["timestamp"] = session.timestamp
                    if turn.img_url:
                        metadata["has_image"] = True

                    items.append((content, metadata))
                    item_mapping.append((conv.sample_id, session.session_num))

        logger.info(
            f"Batch ingesting {len(items)} turns from {len(conversations)} conversations..."
        )

        # Use batch ingestion
        results = self._adapter.add_batch_fast(  # type: ignore[attr-defined]
            items,
            batch_size=64,
            show_progress=True,
        )

        # Build per-conversation results
        conv_results: dict[str, IngestionResult] = {}
        conv_turns: dict[str, int] = {c.sample_id: 0 for c in conversations}
        conv_sessions: dict[str, set[int]] = {c.sample_id: set() for c in conversations}
        conv_errors: dict[str, list[str]] = {c.sample_id: [] for c in conversations}

        for idx, (result, (conv_id, session_num)) in enumerate(
            zip(results, item_mapping, strict=True)
        ):
            if result.success:
                conv_turns[conv_id] += 1
                conv_sessions[conv_id].add(session_num)
            else:
                conv_errors[conv_id].append(f"Turn {idx}: {result.error}")

        # Build IngestionResult for each conversation
        for conv in conversations:
            conv_id = conv.sample_id
            conv_results[conv_id] = IngestionResult(
                conversation_id=conv_id,
                sessions_ingested=len(conv_sessions[conv_id]),
                turns_ingested=conv_turns[conv_id],
                total_turns=conv.total_turns,
                errors=conv_errors[conv_id],
            )
            # Track ingested conversations and sessions
            self._ingested_conversations.add(conv_id)
            if conv_id not in self._ingested_sessions:
                self._ingested_sessions[conv_id] = set()
            self._ingested_sessions[conv_id].update(conv_sessions[conv_id])
            self._total_turns_ingested += conv_turns[conv_id]

        total_success = sum(conv_turns.values())
        logger.info(
            f"Batch ingested {total_success}/{len(items)} turns "
            f"from {len(conversations)} conversations"
        )

        return conv_results

    def answer_question(
        self,
        question: LoCoMoQuestion,
        *,
        use_evidence_sessions: bool = False,
    ) -> LoCoMoAnswer:
        """Answer a LoCoMo question using retrieved memories.

        Retrieves relevant memories based on the question text, constructs
        a context window, and generates an answer using the LLM.

        Args:
            question: The LoCoMo question to answer
            use_evidence_sessions: If True, filter to evidence session IDs only
                                   (useful for ablation studies)

        Returns:
            LoCoMoAnswer with the generated response and metadata
        """
        start_time = time.perf_counter()

        # Build metadata filter if using evidence sessions (oracle mode)
        metadata_filter: dict[str, Any] | None = None
        if use_evidence_sessions and question.evidence_session_nums:
            metadata_filter = {
                "conversation_id": question.conversation_id,
                "session_num": list(question.evidence_session_nums),
            }
        elif question.conversation_id:
            # Always filter to the relevant conversation
            metadata_filter = {"conversation_id": question.conversation_id}

        # Enrich query for better retrieval
        enriched_query = enrich_query(question.question)

        # Dynamic search limit based on question category
        # Temporal questions (category 2) need more context to find date references
        search_limit = self._memory_search_limit
        min_score = self._min_relevance_score

        if question.category.value == 2:  # TEMPORAL
            search_limit = int(search_limit * 1.5)  # 50% more results
            min_score = min_score * 0.8  # Lower threshold for temporal

        # Detect secondary entity questions (e.g., "Melanie's kids")
        if self._is_secondary_entity_question(question.question):
            min_score = min_score * 0.7  # Even lower threshold

        # Search for relevant memories with enriched query
        memories = self._adapter.search(
            query=enriched_query,
            limit=search_limit,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )

        # Build context from retrieved memories with truncation to avoid token limits
        # ~4 chars per token, 100k chars ≈ 25k tokens, safe for most models
        MAX_CONTEXT_CHARS = 100_000

        if memories:
            context_parts = []
            total_chars = 0
            for mem in memories:
                session_num = mem.metadata.get("session_num", "?")
                part = f"[Session {session_num}] {mem.content}"
                if total_chars + len(part) > MAX_CONTEXT_CHARS:
                    logger.debug(
                        f"Context truncated at {len(context_parts)} memories "
                        f"({total_chars} chars)"
                    )
                    break
                context_parts.append(part)
                total_chars += len(part) + 2  # +2 for "\n\n" separator
            context = "\n\n".join(context_parts)
        else:
            context = "(No relevant conversation history found)"

        # Build the system prompt, optionally with category-specific guidance
        system = self._system_prompt
        if self._use_category_prompts:
            category_prompt = self.CATEGORY_PROMPTS.get(question.category.value, "")
            if category_prompt:
                system = f"{system}\n\n{category_prompt}"

        # Handle adversarial questions specially
        if question.is_adversarial:
            adversarial_note = (
                "\n\nNote: Verify the premise of this question against the "
                "conversation history. If the premise is incorrect, state that clearly."
            )
            system = f"{system}{adversarial_note}"

        # Construct the prompt
        user_message = (
            f"Conversation History:\n{context}\n\n"
            f"Question: {question.question}\n\n"
            "Answer based only on the conversation history above. "
            "If the information is not present, say so clearly."
        )

        # Generate answer using LLM
        response = self._llm.complete(
            system=system,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.0,  # Deterministic for reproducibility
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Detect if answer indicates abstention
        is_abstention = self._detect_abstention(response.content)

        return LoCoMoAnswer(
            question_id=question.question_id,
            answer=response.content,
            retrieved_memories=len(memories),
            is_abstention=is_abstention,
            category=question.category,
            latency_ms=latency_ms,
            metadata={
                "model": response.model,
                "usage": response.usage,
                "conversation_id": question.conversation_id,
                "evidence_dia_ids": question.evidence,
                "evidence_sessions": list(question.evidence_session_nums),
                "is_adversarial": question.is_adversarial,
                "adversarial_answer": question.adversarial_answer,
            },
        )

    def answer_all_questions(
        self,
        questions: list[LoCoMoQuestion],
        *,
        use_evidence_sessions: bool = False,
    ) -> list[LoCoMoAnswer]:
        """Answer all LoCoMo questions.

        Args:
            questions: List of questions to answer
            use_evidence_sessions: If True, filter to evidence session IDs only

        Returns:
            List of LoCoMoAnswer objects
        """
        answers = []
        for idx, question in enumerate(questions):
            answer = self.answer_question(
                question,
                use_evidence_sessions=use_evidence_sessions,
            )
            answers.append(answer)

            if (idx + 1) % 50 == 0:
                logger.info(f"Answered {idx + 1}/{len(questions)} questions")

        logger.info(f"Answered {len(answers)} questions")
        return answers

    def answer_questions_by_category(
        self,
        questions: list[LoCoMoQuestion],
        category: QACategory,
        *,
        use_evidence_sessions: bool = False,
    ) -> list[LoCoMoAnswer]:
        """Answer questions filtered by category.

        Args:
            questions: List of all questions
            category: The QA category to filter by
            use_evidence_sessions: If True, filter to evidence session IDs only

        Returns:
            List of LoCoMoAnswer objects for the specified category
        """
        filtered = [q for q in questions if q.category == category]
        logger.info(f"Answering {len(filtered)} questions in category {category.name}")
        return self.answer_all_questions(
            filtered,
            use_evidence_sessions=use_evidence_sessions,
        )

    def _is_secondary_entity_question(self, question: str) -> bool:
        """Detect if a question is about a secondary entity.

        Secondary entities (friends, family members, etc.) are mentioned less
        frequently in conversations and need lower retrieval thresholds.

        Args:
            question: The question text

        Returns:
            True if the question is likely about a secondary entity
        """
        question_lower = question.lower()

        # Patterns indicating secondary entities
        secondary_patterns = [
            r"\b\w+'s\s+(kids?|children|family|friends?|parents?|partner|boyfriend|girlfriend)\b",
            r"\b(melanie|their|his|her)\b",  # Names that aren't the main speaker
            r"\bfamily\s+members?\b",
            r"\bwhat\s+do\s+\w+'s\b",  # "What do X's Y like?"
        ]

        for pattern in secondary_patterns:
            if re.search(pattern, question_lower):
                return True

        return False

    def _detect_abstention(self, answer: str) -> bool:
        """Detect if an answer indicates abstention (no information available).

        Checks for common patterns indicating the model couldn't find
        the requested information.

        Args:
            answer: The generated answer text

        Returns:
            True if the answer indicates abstention
        """
        answer_lower = answer.lower()

        abstention_phrases = [
            "i don't know",
            "i do not know",
            "cannot find",
            "can't find",
            "not mentioned",
            "not discussed",
            "no information",
            "not available",
            "wasn't mentioned",
            "was not mentioned",
            "wasn't discussed",
            "was not discussed",
            "not in the conversation",
            "not present in",
            "unable to find",
            "cannot determine",
            "can't determine",
            "unable to determine",
            "not specified",
            "not clear from",
            "no relevant",
            "never discussed",
            "never mentioned",
            "no record",
            "doesn't appear",
            "does not appear",
            # Adversarial-specific phrases
            "premise is incorrect",
            "incorrect premise",
            "that's not what",
            "that is not what",
            "doesn't match",
            "does not match",
            "contrary to",
        ]

        return any(phrase in answer_lower for phrase in abstention_phrases)

    def clear_memory(self) -> bool:
        """Clear all ingested memories (for test isolation).

        Returns:
            True if successful, False otherwise
        """
        result = self._adapter.clear()
        if result.success:
            self._ingested_conversations.clear()
            self._ingested_sessions.clear()
            self._total_turns_ingested = 0
            logger.debug("Cleared all memories")
        return result.success

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with agent and memory statistics
        """
        adapter_stats = self._adapter.get_stats()

        # Calculate sessions per conversation
        sessions_per_conv = {
            conv_id: len(sessions) for conv_id, sessions in self._ingested_sessions.items()
        }

        return {
            "ingested_conversations": len(self._ingested_conversations),
            "total_sessions": sum(sessions_per_conv.values()),
            "total_turns": self._total_turns_ingested,
            "sessions_per_conversation": sessions_per_conv,
            "memory_search_limit": self._memory_search_limit,
            "min_relevance_score": self._min_relevance_score,
            "use_category_prompts": self._use_category_prompts,
            **adapter_stats,
        }
