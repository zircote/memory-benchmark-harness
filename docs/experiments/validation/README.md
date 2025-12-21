# Validation

Validates that the benchmark harness is correctly configured and the OpenAI API integration works.

## Purpose

The validation step ensures:
1. OpenAI API key is set and functional
2. LLM Judge can evaluate answer correctness
3. All components are properly integrated

**Always run validation before executing full benchmarks.**

## Quick Start

```bash
uv run python scripts/minimal_validation.py
```

## Expected Output

```
============================================================
Minimal End-to-End Validation
============================================================

1. API Key: ...xxxx

2. Creating LLM Judge...
   LLMJudge created (using GPT-4o)

3. Testing judgment (calling OpenAI API)...

   Test 1: What is the capital of France?...
      Model answer: The capital of France is Paris....
      Judgment: correct ✓
      Score: 1.00
      Reasoning: The model's answer is semantically equivalent...

   Test 2: What is 2 + 2?...
      Model answer: The answer is 4....
      Judgment: correct ✓
      Score: 1.00
      Reasoning: The model's answer is semantically equivalent...

   Test 3: Who wrote Romeo and Juliet?...
      Model answer: I'm not sure who wrote it....
      Judgment: incorrect ✓
      Score: 0.00
      Reasoning: The model's answer does not provide...

============================================================
VALIDATION RESULTS: 3/3 tests passed
============================================================

✓ LLM Judge is working correctly
✓ OpenAI API integration verified

The benchmark pipeline is ready for experiments.

Results saved to: results/validation_test/minimal_validation.json
```

## Test Cases

| Question | Reference Answer | Test Type |
|----------|-----------------|-----------|
| What is the capital of France? | Paris | Correct answer |
| What is 2 + 2? | 4 | Correct answer |
| Who wrote Romeo and Juliet? | William Shakespeare | Incorrect answer |

## Output Files

### Location

```
results/validation_test/minimal_validation.json
```

### Schema

```json
{
  "timestamp": "2025-12-20T10:30:00Z",
  "tests_passed": 3,
  "tests_total": 3,
  "success": true,
  "results": [
    {
      "question": "What is the capital of France?",
      "model_answer": "The capital of France is Paris.",
      "judgment_result": "correct",
      "is_correct": true,
      "expected": true,
      "match": true,
      "score": 1.0
    }
  ]
}
```

## Troubleshooting

### API Key Not Set

```
ERROR: OPENAI_API_KEY not set
```

**Fix**: Set the environment variable:
```bash
export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
```

### API Connection Error

```
ERROR: Connection refused
```

**Possible causes**:
- Network connectivity issues
- OpenAI API outage
- Firewall blocking requests

**Fix**: Check connectivity:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Rate Limit Exceeded

```
ERROR: Rate limit exceeded
```

**Fix**: Wait a few minutes and retry, or upgrade your API plan.

### Invalid API Key

```
ERROR: Invalid API key
```

**Fix**: Verify your API key is correct and has sufficient credits.

## How It Works

1. **API Key Check**: Verifies `OPENAI_API_KEY` environment variable is set
2. **Judge Initialization**: Creates an `LLMJudge` instance (uses GPT-4o)
3. **Test Execution**: Runs 3 predefined test cases
4. **Judgment Evaluation**: Each answer is judged for correctness
5. **Match Verification**: Ensures judge agrees with expected outcomes
6. **Results Export**: Saves structured JSON output

## Components Tested

| Component | Test |
|-----------|------|
| `LLMJudge` | Instance creation and API call |
| OpenAI API | Connectivity and response parsing |
| `JudgmentResult` | Correct/incorrect classification |
| Result caching | Cache directory creation |

## Next Steps

After successful validation:

1. Run a single benchmark:
   ```bash
   uv run benchmark run longmemeval --adapter no_memory --trials 1
   ```

2. Run all benchmarks:
   ```bash
   uv run benchmark run-all --adapters git_notes,no_memory --trials 5
   ```

## See Also

- [Experiments Overview](../README.md)
- [LongMemEval Documentation](../longmemeval/README.md)
- [Troubleshooting Guide](../../publication/REPRODUCIBILITY.md#troubleshooting)
