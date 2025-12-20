"""Context-Bench agent wrapper.

This module provides the agent wrapper that integrates memory systems
with the Context-Bench evaluation framework.

Context-Bench provides agents with two tools:
1. open_files: Read complete file contents
2. grep_files: Search for patterns in files

The wrapper integrates these operations with the memory system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from src.adapters.base import MemorySystemAdapter
from src.benchmarks.contextbench.dataset import (
    ContextBenchDataset,
    ContextBenchQuestion,
)

logger = logging.getLogger(__name__)


class FileOperationType(Enum):
    """Types of file operations."""

    OPEN = "open"
    GREP = "grep"


@dataclass(slots=True, frozen=True)
class FileOperation:
    """A file operation performed during question answering.

    Attributes:
        operation_type: Type of operation (open or grep)
        target: File path or grep pattern
        success: Whether operation succeeded
        result: Operation result (content or matches)
        metadata: Additional operation metadata
    """

    operation_type: FileOperationType
    target: str
    success: bool
    result: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OperationResult:
    """Result of answering a question with operation tracking.

    Attributes:
        question_id: ID of the question answered
        answer: Generated answer
        operations: All file operations performed
        retrieved_memories: Memories retrieved from memory system
        total_tokens_read: Estimated tokens read from files
        metadata: Additional result metadata
    """

    question_id: str
    answer: str
    operations: list[FileOperation]
    retrieved_memories: list[dict[str, Any]]
    total_tokens_read: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def operation_count(self) -> int:
        """Return number of operations performed."""
        return len(self.operations)

    @property
    def files_opened(self) -> int:
        """Return number of files opened."""
        return sum(1 for op in self.operations if op.operation_type == FileOperationType.OPEN)

    @property
    def greps_performed(self) -> int:
        """Return number of grep operations."""
        return sum(1 for op in self.operations if op.operation_type == FileOperationType.GREP)


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM clients used in answering questions."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM."""
        ...


@dataclass(slots=True)
class ContextBenchAgent:
    """Agent wrapper for Context-Bench evaluation.

    This agent provides a memory-augmented interface for Context-Bench
    file navigation tasks. It can:
    1. Index files into memory for semantic search
    2. Use memory to guide file operations
    3. Track all operations for cost analysis

    Attributes:
        adapter: Memory system adapter
        llm: LLM client for generating answers
        dataset: The Context-Bench dataset (file system)
        use_memory_for_navigation: Use memory to guide file selection
        max_operations: Maximum operations per question
        retrieval_limit: Max memories to retrieve
    """

    adapter: MemorySystemAdapter
    llm: LLMClient
    dataset: ContextBenchDataset
    use_memory_for_navigation: bool = True
    max_operations: int = 20
    retrieval_limit: int = 10

    def index_files(self) -> int:
        """Index all files into the memory system.

        This allows semantic search over file contents to guide
        which files to open.

        Returns:
            Number of files indexed
        """
        self.adapter.clear()

        indexed = 0
        for file in self.dataset.files:
            metadata = {
                "path": file.path,
                "entity_type": file.entity_type,
                "size": file.size,
            }

            # Store file content
            result = self.adapter.add(file.content, metadata)
            if result.success:
                indexed += 1

        logger.info(f"Indexed {indexed}/{self.dataset.file_count} files into memory")
        return indexed

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are an AI assistant navigating a file system to answer questions.
You have access to two tools:
1. OPEN_FILE(path): Opens a file and returns its contents
2. GREP(pattern): Searches all files for a pattern and returns matching lines

Use these tools to find information needed to answer the question.
After gathering enough information, provide your final answer.

Format your response as:
THOUGHT: <your reasoning>
ACTION: <OPEN_FILE(path) or GREP(pattern) or ANSWER(your answer)>
"""

    def _parse_action(self, response: str) -> tuple[str, str]:
        """Parse action from LLM response.

        Returns:
            Tuple of (action_type, action_target)
        """
        # Look for ACTION: line
        lines = response.split("\n")
        for line in lines:
            if line.strip().upper().startswith("ACTION:"):
                action_part = line.split(":", 1)[1].strip()

                if action_part.upper().startswith("OPEN_FILE"):
                    # Extract path
                    start = action_part.find("(")
                    end = action_part.find(")")
                    if start != -1 and end != -1:
                        path = action_part[start + 1 : end].strip().strip("'\"")
                        return ("open", path)

                elif action_part.upper().startswith("GREP"):
                    start = action_part.find("(")
                    end = action_part.find(")")
                    if start != -1 and end != -1:
                        pattern = action_part[start + 1 : end].strip().strip("'\"")
                        return ("grep", pattern)

                elif action_part.upper().startswith("ANSWER"):
                    start = action_part.find("(")
                    end = action_part.rfind(")")
                    if start != -1 and end != -1:
                        answer = action_part[start + 1 : end].strip()
                        return ("answer", answer)

        # Default to answer with the whole response
        return ("answer", response)

    def _execute_open(self, path: str) -> FileOperation:
        """Execute an open_file operation."""
        file = self.dataset.get_file(path)
        if file:
            return FileOperation(
                operation_type=FileOperationType.OPEN,
                target=path,
                success=True,
                result=file.content,
                metadata={"size": file.size},
            )
        else:
            return FileOperation(
                operation_type=FileOperationType.OPEN,
                target=path,
                success=False,
                result=f"File not found: {path}",
            )

    def _execute_grep(self, pattern: str) -> FileOperation:
        """Execute a grep operation."""
        results = self.dataset.grep_files(pattern)

        if results:
            output_lines: list[str] = []
            for file, lines in results[:10]:  # Limit to 10 files
                output_lines.append(f"=== {file.path} ===")
                output_lines.extend(lines[:5])  # Limit lines per file

            return FileOperation(
                operation_type=FileOperationType.GREP,
                target=pattern,
                success=True,
                result="\n".join(output_lines),
                metadata={"files_matched": len(results)},
            )
        else:
            return FileOperation(
                operation_type=FileOperationType.GREP,
                target=pattern,
                success=False,
                result=f"No matches found for: {pattern}",
            )

    def _suggest_files_from_memory(
        self,
        question: ContextBenchQuestion,
    ) -> list[str]:
        """Use memory to suggest relevant files."""
        if not self.use_memory_for_navigation:
            return []

        # Search memory for relevant content
        memories = self.adapter.search(
            query=question.question_text,
            limit=self.retrieval_limit,
        )

        # Extract file paths from metadata
        suggested: list[str] = []
        for mem in memories:
            path = mem.metadata.get("path")
            if path and path not in suggested:
                suggested.append(path)

        return suggested

    def answer_question(
        self,
        question: ContextBenchQuestion,
    ) -> OperationResult:
        """Answer a question using file navigation.

        Args:
            question: The question to answer

        Returns:
            OperationResult with answer and operation trace
        """
        operations: list[FileOperation] = []
        retrieved_memories: list[dict[str, Any]] = []
        total_tokens = 0

        # Get memory-suggested files
        suggested_files = self._suggest_files_from_memory(question)
        if suggested_files:
            logger.debug(f"Memory suggests files: {suggested_files}")

        # Build initial prompt
        system_prompt = self._build_system_prompt()

        context_parts: list[str] = []
        context_parts.append(f"Question: {question.question_text}")

        if suggested_files:
            context_parts.append(
                f"\nMemory suggests these files may be relevant: {suggested_files}"
            )

        context_parts.append("\nAvailable files:")
        for f in self.dataset.files[:20]:  # Show first 20 files
            context_parts.append(f"  - {f.path}")

        prompt = "\n".join(context_parts)

        # ReAct loop
        answer = ""
        for iteration in range(self.max_operations):
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=512,
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                break

            action_type, action_target = self._parse_action(response)

            if action_type == "answer":
                answer = action_target
                break

            elif action_type == "open":
                op = self._execute_open(action_target)
                operations.append(op)
                if op.success:
                    total_tokens += len(op.result) // 4
                    prompt += f"\n\n[Opened {action_target}]\n{op.result[:2000]}"
                else:
                    prompt += f"\n\n[Error: {op.result}]"

            elif action_type == "grep":
                op = self._execute_grep(action_target)
                operations.append(op)
                if op.success:
                    total_tokens += len(op.result) // 4
                    prompt += f"\n\n[Grep results for '{action_target}']\n{op.result}"
                else:
                    prompt += f"\n\n[No results for '{action_target}']"

            prompt += "\n\nContinue searching or provide your ANSWER."

        # If no answer yet, try to extract from last response
        if not answer:
            answer = "Unable to determine answer from available information."

        return OperationResult(
            question_id=question.question_id,
            answer=answer,
            operations=operations,
            retrieved_memories=retrieved_memories,
            total_tokens_read=total_tokens,
            metadata={
                "suggested_files": suggested_files,
                "iterations": len(operations),
            },
        )

    def answer_question_simple(
        self,
        question: ContextBenchQuestion,
    ) -> OperationResult:
        """Simple single-shot answer without ReAct loop.

        This directly retrieves relevant memories and generates an answer,
        useful for baseline comparisons.

        Args:
            question: The question to answer

        Returns:
            OperationResult with answer
        """
        # Retrieve relevant memories
        memories = self.adapter.search(
            query=question.question_text,
            limit=self.retrieval_limit,
        )

        retrieved: list[dict[str, Any]] = []
        context_parts: list[str] = []

        for mem in memories:
            retrieved.append(
                {
                    "content": mem.content,
                    "score": mem.score,
                    "metadata": mem.metadata,
                }
            )
            context_parts.append(f"[From {mem.metadata.get('path', 'unknown')}]")
            context_parts.append(mem.content[:1000])
            context_parts.append("")

        prompt = f"""Based on the following context, answer the question.

Context:
{chr(10).join(context_parts)}

Question: {question.question_text}

Answer concisely:"""

        try:
            answer = self.llm.generate(prompt=prompt, max_tokens=256)
        except Exception as e:
            answer = f"Error: {e}"

        return OperationResult(
            question_id=question.question_id,
            answer=answer,
            operations=[],
            retrieved_memories=retrieved,
            total_tokens_read=sum(len(m["content"]) // 4 for m in retrieved),
            metadata={"mode": "simple"},
        )
