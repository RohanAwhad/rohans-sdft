"""OPD-style diagnostic: student vs teacher token-level distributions for SDFT.

Student: pi(v | x, y_<t)      -- prompt only
Teacher: pi(v | x+o, y_<t)    -- prompt + golden answer (hindsight)

Standalone (no dependency on train_dir/src). Run with the existing
train_dir/.venv (has torch/transformers/accelerate already installed):

    CUDA_VISIBLE_DEVICES=6 train_dir/.venv/bin/python opd_diagnostic.py
"""

import copy
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/home/rohan/1_Projects/sdft_rag_experiment/data/maas_datasets/eshwar_datasets/sdg_hub_v3/combined_cut_sdft.jsonl",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "opd_diagnostic")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "10"))
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "256"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
STUDENT_MAX_PROMPT_LEN = int(os.environ.get("STUDENT_MAX_PROMPT_LEN", "2048"))
TEACHER_MAX_PROMPT_LEN = int(os.environ.get("TEACHER_MAX_PROMPT_LEN", "2048"))
TOP_K = 16
DEVICE = torch.device("cuda:0")  # respects CUDA_VISIBLE_DEVICES

HINDSIGHT_TEMPLATE = (
    "The following is the correct answer. "
    "Use this to guide your response: {o}"
)


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert 'from/value' (WildChat) format to 'role/content'."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    normalized = []
    for msg in messages:
        if "value" in msg and "content" not in msg:
            role = msg.get("from", "user")
            normalized.append({"role": role_map.get(role, role), "content": msg["value"]})
        else:
            normalized.append(msg)
    return normalized


def load_examples(path: str, n: int) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def build_prompts(tokenizer, example: dict) -> tuple[str, str, str, str]:
    """Returns (student_prompt, teacher_prompt, raw_question, golden_answer)."""
    clean_prompt = normalize_messages(example["prompt"])
    raw_question = clean_prompt[-1]["content"]
    answer_data = example["user_response"]
    golden_answer = (answer_data.get("value") or answer_data.get("content")).strip()

    student_prompt = tokenizer.apply_chat_template(
        clean_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    teacher_history = copy.deepcopy(clean_prompt)
    teacher_history[-1]["content"] += "\n\n" + HINDSIGHT_TEMPLATE.format(o=golden_answer)
    teacher_prompt = tokenizer.apply_chat_template(
        teacher_history, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    return student_prompt, teacher_prompt, raw_question, golden_answer


@torch.no_grad()
def generate_completion(model, tokenizer, prompt_text: str) -> list[int]:
    enc = tokenizer(
        prompt_text, add_special_tokens=False, return_tensors="pt",
        truncation=True, max_length=STUDENT_MAX_PROMPT_LEN,
    ).to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GEN_TEMPERATURE,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return out[0, enc["input_ids"].shape[1]:].tolist()


@torch.no_grad()
def get_completion_logits(
    model, tokenizer, prompt_text: str, completion_ids: list[int], max_length: int,
) -> torch.Tensor:
    """Forward pass via backbone + selective lm_head. Returns (C, V) logits."""
    prompt_enc = tokenizer(
        prompt_text, add_special_tokens=False, return_tensors="pt",
        truncation=True, max_length=max_length,
    ).to(DEVICE)
    prompt_ids = prompt_enc["input_ids"][0]
    prompt_len = prompt_ids.size(0)
    C = len(completion_ids)

    comp_ids_t = torch.tensor(completion_ids, device=DEVICE, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, comp_ids_t]).unsqueeze(0)
    attn_mask = torch.ones_like(input_ids)

    hidden = model.model(input_ids=input_ids, attention_mask=attn_mask)[0]  # (1, S, H)
    completion_hidden = hidden[0, prompt_len - 1 : prompt_len + C - 1, :]  # (C, H)
    return model.lm_head(completion_hidden)  # (C, V)


def token_diagnostics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_ids: list[int],
    tokenizer,
) -> tuple[list[dict], dict]:
    C = student_logits.size(0)
    s_log = F.log_softmax(student_logits.float(), dim=-1)
    t_log = F.log_softmax(teacher_logits.float(), dim=-1)
    s_prob = s_log.exp()
    t_prob = t_log.exp()

    s_entropy = -(s_prob * s_log).sum(dim=-1)  # (C,)
    t_entropy = -(t_prob * t_log).sum(dim=-1)  # (C,)

    s_top = torch.topk(s_prob, TOP_K, dim=-1)
    t_top = torch.topk(t_prob, TOP_K, dim=-1)

    tokens = []
    overlap_ratios, student_probs, teacher_probs = [], [], []

    for i in range(C):
        tok_id = completion_ids[i]
        s_top_ids = s_top.indices[i].tolist()
        t_top_ids = t_top.indices[i].tolist()
        overlap_ids = sorted(set(s_top_ids) & set(t_top_ids))
        overlap_ratio = len(overlap_ids) / TOP_K

        sp = s_prob[i, tok_id].item()
        tp = t_prob[i, tok_id].item()

        student_top16 = [
            {"token_id": tid, "token_str": tokenizer.decode([tid]), "prob": s_top.values[i, j].item()}
            for j, tid in enumerate(s_top_ids)
        ]
        teacher_top16 = [
            {"token_id": tid, "token_str": tokenizer.decode([tid]), "prob": t_top.values[i, j].item()}
            for j, tid in enumerate(t_top_ids)
        ]

        tokens.append({
            "position": i,
            "token_id": tok_id,
            "token_str": tokenizer.decode([tok_id]),
            "student_prob": sp,
            "teacher_prob": tp,
            "student_entropy": s_entropy[i].item(),
            "teacher_entropy": t_entropy[i].item(),
            "overlap_ratio": overlap_ratio,
            "overlap_token_ids": overlap_ids,
            "student_top16": student_top16,
            "teacher_top16": teacher_top16,
        })
        overlap_ratios.append(overlap_ratio)
        student_probs.append(sp)
        teacher_probs.append(tp)

    summary = {
        "mean_overlap_ratio": sum(overlap_ratios) / C,
        "mean_student_entropy": s_entropy.mean().item(),
        "mean_teacher_entropy": t_entropy.mean().item(),
        "mean_entropy_gap": (s_entropy - t_entropy).mean().item(),
        "mean_student_prob": sum(student_probs) / C,
        "mean_teacher_prob": sum(teacher_probs) / C,
    }
    return tokens, summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=DEVICE,
    )
    model.eval()

    examples = load_examples(DATA_PATH, NUM_SAMPLES)
    print(f"Loaded {len(examples)} examples from {DATA_PATH}")

    for i, example in enumerate(examples):
        student_prompt, teacher_prompt, raw_question, golden_answer = build_prompts(tokenizer, example)

        completion_ids = generate_completion(model, tokenizer, student_prompt)
        if len(completion_ids) == 0:
            print(f"[sample {i}] empty completion, skipping")
            continue

        student_logits = get_completion_logits(
            model, tokenizer, student_prompt, completion_ids, STUDENT_MAX_PROMPT_LEN,
        )
        teacher_logits = get_completion_logits(
            model, tokenizer, teacher_prompt, completion_ids, TEACHER_MAX_PROMPT_LEN,
        )

        tokens, summary = token_diagnostics(student_logits, teacher_logits, completion_ids, tokenizer)

        result = {
            "question": raw_question,
            "golden_answer": golden_answer,
            "model_name": MODEL_NAME,
            "completion_length": len(completion_ids),
            "summary": summary,
            "tokens": tokens,
        }

        out_path = os.path.join(OUTPUT_DIR, f"sample{i}_base.json")
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"[sample {i}] {len(completion_ids)} tokens, overlap={summary['mean_overlap_ratio']:.3f} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
