"""Context-Bench dataset loader and data classes.

This module handles loading and parsing the Context-Bench benchmark dataset.
Context-Bench evaluates agents on multi-hop information retrieval through
file navigation tasks.

The benchmark uses procedurally generated data with:
- Fictional entities and relationships (contamination-proof)
- SQL database backing for verified ground-truth answers
- Semi-structured text files for navigation

Source: https://github.com/letta-ai/letta-evals
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class QuestionCategory(Enum):
    """Categories of questions in Context-Bench.

    Each category tests different retrieval capabilities:
    - DIRECT: Single-hop direct fact lookup
    - RELATIONSHIP: Finding relationships between entities
    - MULTI_HOP: Chained lookups across multiple files
    - AGGREGATION: Combining information from multiple sources
    - TEMPORAL: Time-based queries and ordering
    """

    DIRECT = "direct"
    RELATIONSHIP = "relationship"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str) -> QuestionCategory:
        """Parse category from string."""
        normalized = s.lower().replace("-", "_").replace(" ", "_")
        for cat in cls:
            if cat.value == normalized:
                return cat
        return cls.UNKNOWN


@dataclass(slots=True, frozen=True)
class ContextBenchFile:
    """A file in the Context-Bench file system.

    Attributes:
        path: File path (relative to root)
        content: File text content
        entity_type: Type of entities in this file (e.g., 'person', 'project')
        metadata: Additional file metadata
    """

    path: str
    content: str
    entity_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Return the file name without directory."""
        return Path(self.path).name

    @property
    def directory(self) -> str:
        """Return the parent directory."""
        return str(Path(self.path).parent)

    @property
    def size(self) -> int:
        """Return content size in characters."""
        return len(self.content)


@dataclass(slots=True, frozen=True)
class ContextBenchQuestion:
    """A question from the Context-Bench benchmark.

    Attributes:
        question_id: Unique identifier
        question_text: The natural language question
        answer: Ground truth answer from SQL
        category: Question category/type
        sql_query: Original SQL query (for reference)
        relevant_files: Files needed to answer the question
        hop_count: Number of hops required
        metadata: Additional question metadata
    """

    question_id: str
    question_text: str
    answer: str
    category: QuestionCategory
    sql_query: str | None = None
    relevant_files: list[str] = field(default_factory=list)
    hop_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_hop(self) -> bool:
        """Check if this is a multi-hop question."""
        return self.hop_count > 1


@dataclass(slots=True)
class ContextBenchDataset:
    """Complete Context-Bench dataset with files and questions.

    Attributes:
        files: All files in the virtual file system
        questions: All evaluation questions
        file_index: Index mapping path to file
        metadata: Dataset metadata
    """

    files: list[ContextBenchFile]
    questions: list[ContextBenchQuestion]
    file_index: dict[str, ContextBenchFile] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build file index."""
        if not self.file_index:
            self.file_index = {f.path: f for f in self.files}

    @property
    def file_count(self) -> int:
        """Return number of files."""
        return len(self.files)

    @property
    def question_count(self) -> int:
        """Return number of questions."""
        return len(self.questions)

    @property
    def total_file_size(self) -> int:
        """Return total content size across all files."""
        return sum(f.size for f in self.files)

    def get_file(self, path: str) -> ContextBenchFile | None:
        """Get a file by path."""
        return self.file_index.get(path)

    def grep_files(
        self,
        pattern: str,
        case_sensitive: bool = False,
    ) -> list[tuple[ContextBenchFile, list[str]]]:
        """Search files for pattern, returning matching files and lines.

        Args:
            pattern: Search pattern
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of (file, matching_lines) tuples
        """
        results: list[tuple[ContextBenchFile, list[str]]] = []

        for file in self.files:
            content = file.content
            if not case_sensitive:
                pattern_lower = pattern.lower()
                lines = [line for line in content.split("\n") if pattern_lower in line.lower()]
            else:
                lines = [line for line in content.split("\n") if pattern in line]

            if lines:
                results.append((file, lines))

        return results

    def questions_by_category(
        self,
        category: QuestionCategory,
    ) -> list[ContextBenchQuestion]:
        """Get questions of a specific category."""
        return [q for q in self.questions if q.category == category]

    def multi_hop_questions(self) -> list[ContextBenchQuestion]:
        """Get all multi-hop questions."""
        return [q for q in self.questions if q.is_multi_hop]

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        category_counts: dict[str, int] = {}
        hop_counts: dict[int, int] = {}

        for q in self.questions:
            cat = q.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

            hop = q.hop_count
            hop_counts[hop] = hop_counts.get(hop, 0) + 1

        entity_types: dict[str, int] = {}
        for f in self.files:
            et = f.entity_type or "unknown"
            entity_types[et] = entity_types.get(et, 0) + 1

        return {
            "file_count": self.file_count,
            "question_count": self.question_count,
            "total_file_size": self.total_file_size,
            "category_distribution": category_counts,
            "hop_distribution": hop_counts,
            "entity_types": entity_types,
        }


def _parse_file(raw: dict[str, Any]) -> ContextBenchFile:
    """Parse a file from raw format."""
    return ContextBenchFile(
        path=raw.get("path", raw.get("filename", "")),
        content=raw.get("content", raw.get("text", "")),
        entity_type=raw.get("entity_type", raw.get("type")),
        metadata=raw.get("metadata", {}),
    )


def _parse_question(raw: dict[str, Any], idx: int) -> ContextBenchQuestion:
    """Parse a question from raw format."""
    # Get category
    cat_str = raw.get("category", raw.get("type", "unknown"))
    category = QuestionCategory.from_string(cat_str)

    # Get relevant files
    relevant = raw.get("relevant_files", raw.get("files", []))
    if isinstance(relevant, str):
        relevant = [relevant]

    # Infer hop count from relevant files or explicit field
    hop_count = raw.get("hop_count", raw.get("hops", len(relevant)))
    if hop_count < 1:
        hop_count = 1

    return ContextBenchQuestion(
        question_id=raw.get("question_id", raw.get("id", f"q_{idx}")),
        question_text=raw.get("question", raw.get("query", "")),
        answer=raw.get("answer", raw.get("ground_truth", "")),
        category=category,
        sql_query=raw.get("sql_query", raw.get("sql")),
        relevant_files=relevant,
        hop_count=hop_count,
        metadata=raw.get("metadata", {}),
    )


def load_contextbench(
    data_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> ContextBenchDataset:
    """Load Context-Bench dataset.

    This attempts to load from a local letta-evals clone or download.

    Args:
        data_dir: Path to letta-evals data directory
        cache_dir: Cache directory for downloads

    Returns:
        ContextBenchDataset with files and questions

    Raises:
        FileNotFoundError: If data directory not found
        ImportError: If required packages not available
    """
    # Try to find data directory
    if data_dir is None:
        # Look for common locations
        candidates = [
            Path("./letta-evals"),
            Path("./data/letta-evals"),
            Path("../letta-evals"),
        ]
        for candidate in candidates:
            if candidate.exists():
                data_dir = candidate
                break

    if data_dir is None:
        raise FileNotFoundError(
            "Context-Bench data not found. Clone letta-evals repository: "
            "git clone https://github.com/letta-ai/letta-evals.git"
        )

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    logger.info(f"Loading Context-Bench from {data_path}")

    # Look for data files
    files: list[ContextBenchFile] = []
    questions: list[ContextBenchQuestion] = []

    # Try to find files.json or files directory
    files_path = data_path / "files.json"
    if files_path.exists():
        with open(files_path) as f:
            files_data = json.load(f)
        files = [_parse_file(fd) for fd in files_data]

    # Try files directory
    files_dir = data_path / "files"
    if files_dir.exists():
        for file_path in files_dir.rglob("*.txt"):
            rel_path = file_path.relative_to(files_dir)
            content = file_path.read_text()
            files.append(
                ContextBenchFile(
                    path=str(rel_path),
                    content=content,
                )
            )

    # Load questions
    questions_path = data_path / "questions.json"
    if questions_path.exists():
        with open(questions_path) as f:
            questions_data = json.load(f)
        questions = [_parse_question(qd, idx) for idx, qd in enumerate(questions_data)]

    # Try benchmark.json format
    benchmark_path = data_path / "benchmark.json"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            benchmark_data = json.load(f)

        if "files" in benchmark_data:
            files = [_parse_file(fd) for fd in benchmark_data["files"]]
        if "questions" in benchmark_data:
            questions = [
                _parse_question(qd, idx) for idx, qd in enumerate(benchmark_data["questions"])
            ]

    logger.info(f"Loaded Context-Bench: {len(files)} files, {len(questions)} questions")

    return ContextBenchDataset(
        files=files,
        questions=questions,
        metadata={
            "source": str(data_path),
            "loaded_at": datetime.now().isoformat(),
        },
    )


def load_contextbench_from_file(
    filepath: Path | str,
) -> ContextBenchDataset:
    """Load Context-Bench from a single JSON file.

    Args:
        filepath: Path to JSON file with benchmark data

    Returns:
        ContextBenchDataset with files and questions
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Benchmark file not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    files: list[ContextBenchFile] = []
    questions: list[ContextBenchQuestion] = []

    if "files" in data:
        files = [_parse_file(fd) for fd in data["files"]]
    if "questions" in data:
        questions = [_parse_question(qd, idx) for idx, qd in enumerate(data["questions"])]

    return ContextBenchDataset(
        files=files,
        questions=questions,
        metadata={
            "source": str(filepath),
            "loaded_at": datetime.now().isoformat(),
        },
    )


def generate_synthetic_dataset(
    n_files: int = 50,
    n_questions: int = 100,
    seed: int = 42,
) -> ContextBenchDataset:
    """Generate a synthetic Context-Bench dataset for testing.

    This creates a small synthetic dataset with known answers
    for unit testing and development.

    Args:
        n_files: Number of files to generate
        n_questions: Number of questions to generate
        seed: Random seed for reproducibility

    Returns:
        Synthetic ContextBenchDataset
    """
    import random

    random.seed(seed)

    # Generate fictional entities
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]
    projects = ["Project Alpha", "Project Beta", "Project Gamma", "Project Delta"]
    departments = ["Engineering", "Marketing", "Sales", "Research", "Operations"]

    files: list[ContextBenchFile] = []
    facts: list[dict[str, Any]] = []

    # Generate person files
    for i in range(min(n_files // 2, len(first_names) * len(last_names))):
        first = random.choice(first_names)
        last = random.choice(last_names)
        name = f"{first} {last}"
        dept = random.choice(departments)
        project = random.choice(projects)
        age = random.randint(25, 65)

        content = f"""Name: {name}
Department: {dept}
Project: {project}
Age: {age}
Email: {first.lower()}.{last.lower()}@example.com
"""
        files.append(
            ContextBenchFile(
                path=f"people/{first.lower()}_{last.lower()}.txt",
                content=content,
                entity_type="person",
            )
        )
        facts.append(
            {
                "name": name,
                "department": dept,
                "project": project,
                "age": age,
            }
        )

    # Generate project files
    for project in projects:
        members = random.sample(
            [f["name"] for f in facts if f["project"] == project],
            min(3, len([f for f in facts if f["project"] == project])),
        )
        content = f"""Project: {project}
Status: Active
Team Members:
{chr(10).join("- " + m for m in members)}
"""
        files.append(
            ContextBenchFile(
                path=f"projects/{project.lower().replace(' ', '_')}.txt",
                content=content,
                entity_type="project",
            )
        )

    # Generate questions
    questions: list[ContextBenchQuestion] = []
    for i in range(n_questions):
        if not facts:
            break

        fact = random.choice(facts)
        q_type = random.choice(["direct", "relationship", "multi_hop"])

        if q_type == "direct":
            questions.append(
                ContextBenchQuestion(
                    question_id=f"q_{i}",
                    question_text=f"What department does {fact['name']} work in?",
                    answer=fact["department"],
                    category=QuestionCategory.DIRECT,
                    hop_count=1,
                )
            )
        elif q_type == "relationship":
            questions.append(
                ContextBenchQuestion(
                    question_id=f"q_{i}",
                    question_text=f"What project is {fact['name']} working on?",
                    answer=fact["project"],
                    category=QuestionCategory.RELATIONSHIP,
                    hop_count=1,
                )
            )
        else:
            # Multi-hop: find colleague
            colleagues = [
                f["name"]
                for f in facts
                if f["project"] == fact["project"] and f["name"] != fact["name"]
            ]
            if colleagues:
                colleague = random.choice(colleagues)
                colleague_fact = next(f for f in facts if f["name"] == colleague)
                questions.append(
                    ContextBenchQuestion(
                        question_id=f"q_{i}",
                        question_text=(
                            f"What is the age of someone who works with "
                            f"{fact['name']} on the same project?"
                        ),
                        answer=str(colleague_fact["age"]),
                        category=QuestionCategory.MULTI_HOP,
                        hop_count=2,
                    )
                )

    return ContextBenchDataset(
        files=files,
        questions=questions,
        metadata={
            "synthetic": True,
            "seed": seed,
            "generated_at": datetime.now().isoformat(),
        },
    )
