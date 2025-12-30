#!/usr/bin/env python3
"""Test script comparing agentic vs passive memory approaches on LoCoMo.

This script compares:
1. Passive: Ingest all turns blindly, single search per question
2. Agentic: LLM extracts key facts, iterative search per question

Usage:
    python scripts/test_agentic_memory.py [--questions N] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.subcog import SubcogAdapter
from src.benchmarks.locomo.agentic_wrapper import AgenticLoCoMoWrapper
from src.benchmarks.locomo.dataset import load_locomo
from src.benchmarks.locomo.wrapper import LoCoMoAgent
from src.clients.openai_client import OpenAIClient
from src.evaluation.judge import LLMJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_passive_approach(
    conversation_id: str,
    questions_limit: int,
    llm: OpenAIClient,
    judge: LLMJudge,
) -> dict:
    """Run the passive (current) approach."""
    logger.info("=" * 60)
    logger.info("PASSIVE APPROACH (ingest all, single search)")
    logger.info("=" * 60)

    # Load dataset
    dataset = load_locomo()

    # Find conversation by ID
    conversation = None
    for conv in dataset.conversations:
        if conv.sample_id == conversation_id:
            conversation = conv
            break

    if not conversation:
        logger.error(f"Conversation {conversation_id} not found")
        return {}

    questions = conversation.questions[:questions_limit]

    # Setup adapter and agent
    adapter = SubcogAdapter()
    adapter.clear()

    agent = LoCoMoAgent(adapter, llm)

    # Ingest all turns
    logger.info("Ingesting conversation...")
    result = agent.ingest_conversation(conversation)
    logger.info(f"Ingested {result.turns_ingested} turns")

    # Answer questions
    correct = 0
    total = 0
    scores = []

    logger.info(f"Answering {len(questions)} questions...")
    for q in questions:
        answer = agent.answer_question(q)

        # Judge the answer (ensure answer is string for judge)
        answer_str = str(q.answer) if not isinstance(q.answer, str) else q.answer
        judgment = judge.judge(
            question=q.question,
            reference_answer=answer_str,
            model_answer=answer.answer,
        )

        is_correct = judgment.score >= 0.7
        if is_correct:
            correct += 1
        total += 1
        scores.append(judgment.score)

        logger.debug(
            f"Q: {q.question[:50]}... "
            f"Expected: {answer_str[:30]}... "
            f"Got: {answer.answer[:30]}... "
            f"Score: {judgment.score:.2f}"
        )

    accuracy = correct / total if total else 0
    avg_score = sum(scores) / len(scores) if scores else 0

    logger.info(f"Results: {correct}/{total} ({accuracy:.0%})")
    logger.info(f"Avg score: {avg_score:.2f}")

    adapter.cleanup()

    return {
        "approach": "passive",
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "turns_ingested": result.turns_ingested,
    }


def run_agentic_approach(
    conversation_id: str,
    questions_limit: int,
    llm: OpenAIClient,
    judge: LLMJudge,
) -> dict:
    """Run the agentic approach."""
    logger.info("=" * 60)
    logger.info("AGENTIC APPROACH (extract facts, iterative search)")
    logger.info("=" * 60)

    # Load dataset
    dataset = load_locomo()

    # Find conversation by ID
    conversation = None
    for conv in dataset.conversations:
        if conv.sample_id == conversation_id:
            conversation = conv
            break

    if not conversation:
        logger.error(f"Conversation {conversation_id} not found")
        return {}

    questions = conversation.questions[:questions_limit]

    # Setup adapter and wrapper
    adapter = SubcogAdapter()
    adapter.clear()

    wrapper = AgenticLoCoMoWrapper(
        adapter,
        llm,
        max_search_iterations=3,
        memories_per_search=10,
    )

    # Ingest with fact extraction
    logger.info("Extracting facts from conversation...")
    facts_saved = wrapper.ingest_conversation(conversation, show_progress=True)
    logger.info(f"Extracted {facts_saved} facts")

    # Answer questions
    correct = 0
    total = 0
    scores = []
    total_searches = 0

    logger.info(f"Answering {len(questions)} questions...")
    for q in questions:
        answer = wrapper.answer_question(q)
        total_searches += answer.search_iterations

        # Judge the answer (ensure answer is string for judge)
        answer_str = str(q.answer) if not isinstance(q.answer, str) else q.answer
        judgment = judge.judge(
            question=q.question,
            reference_answer=answer_str,
            model_answer=answer.answer,
        )

        is_correct = judgment.score >= 0.7
        if is_correct:
            correct += 1
        total += 1
        scores.append(judgment.score)

        logger.debug(
            f"Q: {q.question[:50]}... "
            f"Searches: {answer.search_iterations} "
            f"Score: {judgment.score:.2f}"
        )

    accuracy = correct / total if total else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_searches = total_searches / total if total else 0

    logger.info(f"Results: {correct}/{total} ({accuracy:.0%})")
    logger.info(f"Avg score: {avg_score:.2f}")
    logger.info(f"Avg searches per question: {avg_searches:.1f}")

    adapter.cleanup()

    return {
        "approach": "agentic",
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "facts_extracted": facts_saved,
        "avg_searches": avg_searches,
    }


def main() -> None:
    """Run comparison test."""
    parser = argparse.ArgumentParser(
        description="Compare agentic vs passive memory on LoCoMo"
    )
    parser.add_argument(
        "--questions", "-n",
        type=int,
        default=20,
        help="Number of questions to test (default: 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--conversation",
        type=str,
        default="conv-26",
        help="Conversation ID to test (default: conv-26)",
    )
    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    # Setup LLM and judge
    llm = OpenAIClient(model=args.model, max_tokens=2000)
    judge = LLMJudge(model=args.model, max_tokens=1000)

    print()
    print("=" * 60)
    print("AGENTIC vs PASSIVE MEMORY COMPARISON")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Conversation: {args.conversation}")
    print(f"Questions: {args.questions}")
    print()

    # Run both approaches
    passive_results = run_passive_approach(
        args.conversation,
        args.questions,
        llm,
        judge,
    )

    print()

    agentic_results = run_agentic_approach(
        args.conversation,
        args.questions,
        llm,
        judge,
    )

    # Print comparison
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Approach':<15} {'Correct':<10} {'Accuracy':<12} {'Avg Score':<10}")
    print("-" * 50)
    passive_acc = passive_results.get('accuracy', 0)
    agentic_acc = agentic_results.get('accuracy', 0)
    print(
        f"{'Passive':<15} "
        f"{passive_results.get('correct', 0)}/{passive_results.get('total', 0):<7} "
        f"{passive_acc:.0%}{'':>8} "
        f"{passive_results.get('avg_score', 0):.2f}"
    )
    print(
        f"{'Agentic':<15} "
        f"{agentic_results.get('correct', 0)}/{agentic_results.get('total', 0):<7} "
        f"{agentic_acc:.0%}{'':>8} "
        f"{agentic_results.get('avg_score', 0):.2f}"
    )
    print()
    print("Details:")
    print(f"  Passive: {passive_results.get('turns_ingested', 0)} turns ingested")
    print(f"  Agentic: {agentic_results.get('facts_extracted', 0)} facts extracted")
    print(f"  Agentic: {agentic_results.get('avg_searches', 0):.1f} avg searches/question")
    print()

    # Save results
    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": args.model,
        "conversation": args.conversation,
        "questions": args.questions,
        "passive": passive_results,
        "agentic": agentic_results,
    }

    results_dir = Path("results/agentic_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"comparison_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
