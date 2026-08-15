"""Evaluate a model on MaaS SDFT test set.

For each question:
  1. Generate answer with question only
  2. Judge against golden answer
  3. Generate answer with question + context
  4. Judge against golden answer

Uses vLLM for inference and Anthropic Vertex (Sonnet) for judging.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv_vllm/bin/python scripts/eval_maas_sdft.py \
    --model Qwen/Qwen3-8B \
    --test_jsonl /home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/test_maas_sdft.jsonl \
    --output_dir ./eval_results/base_model
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


JUDGE_SYSTEM = """\
You are a strict factual correctness evaluator. You will be given a question, a golden (reference) answer, source documentation, and a candidate answer.

Your task:
- Judge whether the candidate answer is factually correct with respect to the source documentation AND the golden answer.
- The candidate does NOT need to be word-for-word identical to the golden answer. It must convey the same key facts.
- If the candidate contains correct facts from the source documentation that the golden answer omits, that is NOT a failure.
- If the candidate contradicts the source documentation, that IS a failure.
- Missing key facts from the golden answer IS a failure.
- Minor phrasing differences are acceptable.
- Respond with exactly one line: PASS or FAIL
- Then a brief rationale (1-2 sentences max)."""

JUDGE_USER_TEMPLATE = """\
Question:
{question}

Source Documentation:
{source_doc}

Golden Answer:
{golden_answer}

Candidate Answer:
{candidate_answer}

Verdict (PASS or FAIL):"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./eval_results")
    p.add_argument("--vllm_tp", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--judge_model", type=str, default="claude-sonnet-4-6@default")
    p.add_argument("--judge_workers", type=int, default=20)
    p.add_argument("--default-mode", action="store_true")
    return p.parse_args()


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_prompts(records, tokenizer):
    """Build 200 prompts: 100 question-only + 100 question+context."""
    no_ctx_prompts = []
    with_ctx_prompts = []

    for rec in records:
        question = rec["prompt"][0].get("value") or rec["prompt"][0].get("content")

        # Question only
        messages_no_ctx = [{"role": "user", "content": question}]
        no_ctx_prompts.append(tokenizer.apply_chat_template(
            messages_no_ctx, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

        # Question + context
        context = rec["enriched_user_response"].get("value") or rec["enriched_user_response"].get("content")
        content_with_ctx = f"{question}\n\nContext:\n{context}"
        messages_with_ctx = [{"role": "user", "content": content_with_ctx}]
        with_ctx_prompts.append(tokenizer.apply_chat_template(
            messages_with_ctx, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

    return no_ctx_prompts, with_ctx_prompts


def generate_all(model_path, all_prompts, args):
    """Load vLLM, generate for all prompts, return list of response texts."""
    from vllm import LLM, SamplingParams

    tp_size = args.vllm_tp
    print(f"[vLLM] Loading {model_path} (tp={tp_size})...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=8192,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"[vLLM] Generating {len(all_prompts)} completions...")
    outputs = llm.generate(all_prompts, sampling_params)
    texts = [o.outputs[0].text for o in outputs]

    return texts, tokenizer


def judge_single(client, model, question, golden_answer, candidate_answer, source_doc, max_retries=5):
    """Call Anthropic Vertex to judge a single answer. Returns (pass_bool, rationale)."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        candidate_answer=candidate_answer,
        source_doc=source_doc,
    )

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=256,
                temperature=0.0,
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
            passed = text.upper().startswith("PASS")
            return passed, text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2 ** attempt))
            else:
                return False, f"JUDGE_ERROR: {e}"


def judge_single_majority(client, model, question, golden_answer, candidate_answer, source_doc, num_votes=3):
    """Call judge 3 times, return majority vote. Returns (pass_bool, rationale, votes)."""
    votes = []
    rationales = []
    for _ in range(num_votes):
        passed, rationale = judge_single(client, model, question, golden_answer, candidate_answer, source_doc)
        votes.append(passed)
        rationales.append(rationale)
    
    pass_count = sum(votes)
    majority_pass = pass_count >= (num_votes // 2 + 1)
    # Use rationale from the majority side
    majority_rationale = next(r for v, r in zip(votes, rationales) if v == majority_pass)
    return majority_pass, majority_rationale, votes


def judge_all(records, no_ctx_answers, with_ctx_answers, args):
    """Judge all answers using ThreadPoolExecutor with majority voting (3x per answer)."""
    from anthropic import AnthropicVertex
    client = AnthropicVertex()

    results = []
    tasks = []  # (index, mode, question, golden, candidate, source_doc)

    for i, rec in enumerate(records):
        question = rec["prompt"][0].get("value") or rec["prompt"][0].get("content")
        golden = rec["user_response"].get("value") or rec["user_response"].get("content")
        source_doc = rec.get("enriched_user_response", {}).get("value") or rec.get("enriched_user_response", {}).get("content", "")
        tasks.append((i, "no_context", question, golden, no_ctx_answers[i], source_doc))
        if not args.default_mode: tasks.append((i, "with_context", question, golden, with_ctx_answers[i], source_doc))

    judgments = {}  # (index, mode) -> (passed, rationale, votes)

    total_calls = len(tasks) * 3  # 3 votes per task
    print(f"[Judge] Judging {len(tasks)} answers x3 votes = {total_calls} calls with {args.judge_model} ({args.judge_workers} workers)...")
    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        future_to_key = {}
        for idx, mode, q, g, c, s in tasks:
            fut = pool.submit(judge_single_majority, client, args.judge_model, q, g, c, s)
            future_to_key[fut] = (idx, mode)

        done = 0
        for fut in as_completed(future_to_key):
            key = future_to_key[fut]
            judgments[key] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(tasks)}]")

    # Assemble results
    for i, rec in enumerate(records):
        question = rec["prompt"][0].get("value") or rec["prompt"][0].get("content")
        golden = rec["user_response"].get("value") or rec["user_response"].get("content")

        no_ctx_pass, no_ctx_rationale, no_ctx_votes = judgments[(i, "no_context")]
        if not args.default_mode: with_ctx_pass, with_ctx_rationale, with_ctx_votes = judgments[(i, "with_context")]

        results.append({
            "question": question,
            "golden_answer": golden,
            "no_context_answer": no_ctx_answers[i],
            "no_context_pass": no_ctx_pass,
            "no_context_rationale": no_ctx_rationale,
            "no_context_votes": no_ctx_votes,
        })
        if not args.default_mode: results[-1].update({
            "with_context_answer": with_ctx_answers[i],
            "with_context_pass": with_ctx_pass,
            "with_context_rationale": with_ctx_rationale,
            "with_context_votes": with_ctx_votes,
        })

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_records(args.test_jsonl)
    n = len(records)
    print(f"Loaded {n} test records")

    # Step 1: Build prompts
    # We need the tokenizer before generation, but vLLM gives us one.
    # Build prompts inside generate_all after tokenizer is available.
    # Instead, build all 200 prompts, generate in one batch.

    from vllm import LLM, SamplingParams

    tp_size = args.vllm_tp
    print(f"[vLLM] Loading {args.model} (tp={tp_size})...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=8192,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    no_ctx_prompts, with_ctx_prompts = build_prompts(records, tokenizer)
    all_prompts = no_ctx_prompts if args.default_mode else no_ctx_prompts + with_ctx_prompts

    print(f"[vLLM] Generating {len(all_prompts)} completions...")
    outputs = llm.generate(all_prompts, sampling_params)
    all_texts = [o.outputs[0].text for o in outputs]

    no_ctx_answers = all_texts[:n]
    with_ctx_answers = all_texts[n:]

    # Free vLLM
    del llm
    import gc; gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Judge
    results = judge_all(records, no_ctx_answers, with_ctx_answers, args)

    # Step 3: Save results
    output_path = os.path.join(args.output_dir, "eval_results.jsonl")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Step 4: Print summary
    no_ctx_pass = sum(1 for r in results if r["no_context_pass"])
    if not args.default_mode: with_ctx_pass = sum(1 for r in results if r["with_context_pass"])

    print(f"\n{'='*40}")
    print(f"Model: {args.model}")
    print(f"{'='*40}")
    print(f"no_context:   {no_ctx_pass}/{n} ({100*no_ctx_pass/n:.1f}%)")
    if not args.default_mode: print(f"with_context: {with_ctx_pass}/{n} ({100*with_ctx_pass/n:.1f}%)")
    print(f"{'='*40}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
