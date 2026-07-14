# Task 3: SDFT on Synthetic Algebra Adapter

Train an adapter model using SDFT (reverse KL distillation) on the synthetic algebra dataset. The teacher sees the explicit symbol mapping; the student gets a vague description and must learn the mapping through distribution matching.

## Adapter Flow

```
User -> API -> Adapter -> API -> User
```

The adapter checks the API's answer:
- If correct: outputs `\boxed{CORRECT}` (pass-through to user)
- If wrong/missing: outputs guidance text (handed back to API for retry)

The adapter never computes answers. Only checks and provides correction guidance.

## Models

| Role | Model | Notes |
|------|-------|-------|
| Student | `Qwen/Qwen3-8B` | Full finetuning (no LoRA) |
| Teacher | `Qwen/Qwen3-8B` | EMA-tracked copy of student |
| Inference | `Qwen/Qwen3-8B` | vLLM server for rollouts |

## Dataset

Stored in MaaS format (same pattern as existing trainer):

```json
{
  "prompt": [
    {"from": "system", "value": "You are an adapter that checks an API model's arithmetic answer. If it is correct, respond \\boxed{CORRECT} and it will be sent directly to the user. If wrong or missing, provide guidance back to the API on how to fix it (do not compute the answer yourself)."},
    {"from": "human", "value": "The symbols θ, α, γ, β each represent one of +, -, ×, ÷. BODMAS applies.\n\nExamples:\nExpression: 3 θ 4 | API answer: 7 → \\boxed{CORRECT}\nExpression: 10 α 3 | API answer: 5 → The subtraction 10-3=7, you said 5. Recheck.\nExpression: 2 γ 6 | API answer: none → Use symbol mapping to compute.\n\nExpression: {expr} | API answer: {api_answer} →"}
  ],
  "user_response": {"value": "θ=+, α=-, γ=×, β=÷. Correct answer: {answer}"}
}
```

Splits: `data/train.jsonl` (1600: 800 custom + 800 standard), `data/test.jsonl` (400: 200 custom + 200 standard).

### API answer generation (static, per expression)

| Type | api_answer | Ratio | How generated |
|------|-----------|-------|---------------|
| CORRECT | `answer` (ground truth) | 30% | — |
| WRONG | plausible-wrong int | 50% | compute with wrong op, or random ±10 offset |
| NONE | `"none"` | 20% | — |

## Privileged Information

**Teacher** sees the explicit mapping + correct answer (appended to last user message via `user_response`):
```
θ=+, α=-, γ=×, β=÷. Correct answer: {answer}
```

**Student** sees only the vague hint: "each symbol maps to one operation, BODMAS applies."

## Thinking Client (Nemotron Budget Pattern)

Two-step vLLM generation (`vllm_utils.py`):

```
Step 1: v1/completions with enable_thinking=True, max_tokens=reasoning_budget (128)
        → if </think> missing, force-close
Step 2: v1/completions continue from closed thinking, max_tokens=remaining
        → guidance text or \boxed{CORRECT}
```

Total ceiling: 256 tokens (128 thinking + 128 answer). Thinking naturally brief on arithmetic (~50-100 tokens).

## How SDFT Applies

1. **Teacher has privileged info**: explicit symbol mapping + correct answer (student only gets vague hint)
2. **Student generates completion** (via vLLM rollout): `\boxed{CORRECT}` or guidance text
3. **Teacher computes log-probs** on `[conditional_prompt + completion]`: teacher sees same completion with explicit knowledge — distribution reflects whether CORRECT/guidance is appropriate
4. **Reverse KL loss**: `KL(p_student || p_teacher)` — student pulled toward teacher's distribution
5. **Weight sync**: student → teacher (EMA) + vLLM after each optimizer step

## Files

| File | Action | What |
|------|--------|------|
| `synthetic_algebra_dataset.py` | Modify | Add prompt + user_response generation with API answer sampling |
| `task_3/collator.py` | Create | Thin wrapper — prompts are pre-formatted, no chat template needed |
| `train_dir/src/vllm_utils.py` | Modify | Add `vllm_generate_with_thinking()` with budget control |
| `train_dir/src/config.py` | Modify | New env vars: `TRAIN_DATA_PATH`, `REASONING_BUDGET` |
| `train_dir/src/trainer.py` | Modify | Support adapter prompt format (skip chat template) |
| `task_3/launch.sh` | Create | GPU assignment + env vars |
| `task_3/eval.py` | Create | Mode 1: binary classification, Mode 2: two-pass guidance |

## Config

```bash
TRAIN_DATA_PATH=data/train.jsonl
TEST_DATA_PATH=data/test.jsonl
REASONING_BUDGET=128
GEN_MAX_NEW_TOKENS=256
HINDSIGHT_FIELD=user_response
LEARNING_RATE=5e-5
EMA_ALPHA=0.05
NUM_EPOCHS=10
```

## Evaluation (`task_3/eval.py`)

Runs against frozen checkpoints (not during training).

### Mode 1: Binary Classification

Can the adapter tell right from wrong?

| | API correct | API wrong/NONE |
|---|---|---|
| `\boxed{CORRECT}` | TN (pass-through ok) | FN (missed error) |
| guidance | FP (false alarm) | TP (caught error) |

Parse adapter output for `\boxed{CORRECT}` presence. Compare against `api_answer == ground_truth`.

Metrics: accuracy, precision, recall — broken down by custom/standard.

### Mode 2: Two-Pass Guidance Quality

Does the adapter's guidance actually help the API fix its mistake?

```
1. Adapter receives expression + api_answer → outputs CORRECT or guidance
2. If guidance: feed to API model →
   "Previous answer was {api_answer}. Checker says: {guidance}. Solve: {expression}. Answer with just the number."
3. Parse API retry response for integer, compare to ground truth
```

Metric: guidance-corrected accuracy = % of originally-wrong answers fixed after guidance.

Note: This metric is confounded — it mixes guidance quality with API model execution capability. The API model may fail to follow good guidance or self-correct despite bad guidance. Cannot disentangle without an oracle API model.

### Baselines

| Model | Custom acc | Standard acc |
|-------|-----------|-------------|
| Claude Haiku (no adapter) | 0% | 75.5% |
| GRPO adapter @ 1000 steps | 84% | 97.5% |
| Untrained Qwen3-8B | ~10-15% | — |

## Order of Work

1. Extend `synthetic_algebra_dataset.py`
2. Add `vllm_generate_with_thinking()`
3. Create `task_3/collator.py`
4. Wire config + trainer
5. `task_3/launch.sh` + `task_3/eval.py`
6. Test end-to-end
