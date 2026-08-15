"""Evaluate a LoRA adapter on the MaaS SDFT test set.

Same flow as eval_maas_sdft.py (generate question-only + question+context,
judge vs golden with Anthropic Vertex), but serves the base model + LoRA
adapter through vLLM's in-process LoRA support.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_maas_sdft_lora.py \
    --base-model Qwen/Qwen3-8B \
    --adapter-dir /workspace/output_smoke/step_100 \
    --test_jsonl /workspace/test_maas_sdft.jsonl \
    --output_dir ./eval_results/lora_step_100
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
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--adapter-dir", type=str, required=True)
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
    no_ctx_prompts = []
    with_ctx_prompts = []

    for rec in records:
        question = rec["prompt"][0].get("value") or rec["prompt"][0].get("content")
        messages_no_ctx = [{"role": "user", "content": question}]
        no_ctx_prompts.append(tokenizer.apply_chat_template(
            messages_no_ctx, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

        context = rec["enriched_user_response"].get("value") or rec["enriched_user_response"].get("content")
        content_with_ctx = f"{question}\n\nContext:\n{context}"
        messages_with_ctx = [{"role": "user", "content": content_with_ctx}]
        with_ctx_prompts.append(tokenizer.apply_chat_template(
            messages_with_ctx, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

    return no_ctx_prompts, with_ctx_prompts


def judge_single(client, model, question, golden_answer, candidate_answer, source_doc, max_retries=5):
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
    votes = []
    rationales = []
    for _ in range(num_votes):
        passed, rationale = judge_single(client, model, question, golden_answer, candidate_answer, source_doc)
        votes.append(passed)
        rationales.append(rationale)

    pass_count = sum(votes)
    majority_pass = pass_count >= (num_votes // 2 + 1)
    majority_rationale = next(r for v, r in zip(votes, rationales) if v == majority_pass)
    return majority_pass, majority_rationale, votes


def judge_all(records, no_ctx_answers, with_ctx_answers, args):
    from anthropic import AnthropicVertex
    client = AnthropicVertex()

    results = []
    tasks = []

    for i, rec in enumerate(records):
        question = rec["prompt"][0].get("value") or rec["prompt"][0].get("content")
        golden = rec["user_response"].get("value") or rec["user_response"].get("content")
        source_doc = rec.get("enriched_user_response", {}).get("value") or rec.get("enriched_user_response", {}).get("content", "")
        tasks.append((i, "no_context", question, golden, no_ctx_answers[i], source_doc))
        if not args.default_mode:
            tasks.append((i, "with_context", question, golden, with_ctx_answers[i], source_doc))

    judgments = {}

    total_calls = len(tasks) * 3
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

    for i, rec in enumerate(records):
        question = rec["prompt"][0].get("value") or rec["prompt"][0].get("content")
        golden = rec["user_response"].get("value") or rec["user_response"].get("content")

        no_ctx_pass, no_ctx_rationale, no_ctx_votes = judgments[(i, "no_context")]
        if not args.default_mode:
            with_ctx_pass, with_ctx_rationale, with_ctx_votes = judgments[(i, "with_context")]

        results.append({
            "question": question,
            "golden_answer": golden,
            "no_context_answer": no_ctx_answers[i],
            "no_context_pass": no_ctx_pass,
            "no_context_rationale": no_ctx_rationale,
            "no_context_votes": no_ctx_votes,
        })
        if not args.default_mode:
            results[-1].update({
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

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[vLLM] Loading {args.base_model} + LoRA {args.adapter_dir}...")
    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=args.vllm_tp,
        trust_remote_code=True,
        max_model_len=8192,
        enforce_eager=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    lora_request = LoRARequest("sdft-policy", 1, args.adapter_dir)

    no_ctx_prompts, with_ctx_prompts = build_prompts(records, tokenizer)
    all_prompts = no_ctx_prompts if args.default_mode else no_ctx_prompts + with_ctx_prompts

    print(f"[vLLM] Generating {len(all_prompts)} completions...")
    outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request)
    all_texts = [o.outputs[0].text for o in outputs]

    no_ctx_answers = all_texts[:n]
    with_ctx_answers = all_texts[n:]

    del llm
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = judge_all(records, no_ctx_answers, with_ctx_answers, args)

    output_path = os.path.join(args.output_dir, "eval_results.jsonl")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    no_ctx_pass = sum(1 for r in results if r["no_context_pass"])
    if not args.default_mode:
        with_ctx_pass = sum(1 for r in results if r["with_context_pass"])

    print(f"\n{'='*40}")
    print(f"Model: {args.base_model} + LoRA {args.adapter_dir}")
    print(f"{'='*40}")
    print(f"no_context:   {no_ctx_pass}/{n} ({100*no_ctx_pass/n:.1f}%)")
    if not args.default_mode:
        print(f"with_context: {with_ctx_pass}/{n} ({100*with_ctx_pass/n:.1f}%)")
    print(f"{'='*40}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
