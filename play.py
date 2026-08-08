"""Collator verification: OLS 31-item run + old-format regression + family guard."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from datasets import Dataset

TOKENIZER_PATH = "/mnt/nvme5n1/rohan_patched_ckpts/hf-cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
OLS_DATA = "data/ols/train_sdft_mini.jsonl"
STUDENT_LIMIT = 14336
TEACHER_LIMIT = 15360
TOTAL_LIMIT = 16384
HINT_PREFIX = "The following is the correct answer. Use this to guide your response: "

GPT_OSS_PATH = "/mnt/nvme1n1/rawhad/.hf_cache/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/cc89b3e7fd423253264883a80a4fa5abc619649f"
ANALYSIS_CHANNEL = "<|channel|>analysis<|message|>"
FINAL_CHANNEL = "<|channel|>final<|message|>"


def reset(env_updates: dict):
    os.environ.update(env_updates)
    import megatron_trainer.config as config
    import megatron_trainer.collator as collator

    importlib.reload(config)
    importlib.reload(collator)
    return collator


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    examples = [json.loads(l) for l in open(OLS_DATA)]

    # ---------------- Phase 1: OLS, Qwen path ----------------
    collator = reset({"TRAIN_DATA_PATH": OLS_DATA, "MODEL_NAME": "Qwen/Qwen3-8B",
                      "STUDENT_MAX_PROMPT_LEN": "14336", "TEACHER_MAX_PROMPT_LEN": "15360"})
    assert collator.IS_QWEN and not collator.IS_GPT_OSS
    assert collator.TOOL_DEFS is not None and len(collator.TOOL_DEFS) == 28
    stripped = {"resources_get", "resources_list", "nodes_top"}
    all_names = {d["function"]["name"] for d in collator.TOOL_DEFS}
    assert all_names >= stripped
    assert all("description" not in d["function"] for d in collator.TOOL_DEFS)
    for d in collator.TOOL_DEFS:
        for p in d["function"].get("parameters", {}).get("properties", {}).values():
            assert "description" not in p
    print(f"[1] tool defs loaded + stripped: {len(collator.TOOL_DEFS)} defs, 0 descriptions")

    c = collator.SDFTCollator(tokenizer=tok, hindsight_field="user_response")
    out = c(examples)

    kept_ds = c.filter_dataset(Dataset.from_list(examples))
    assert len(kept_ds) == 31, "drop filter removed valid OLS examples"
    print("[1c] protected-set drop filter: 0/31 dropped at OLS budgets")

    assert len(out["prompt_texts"]) == len(out["conditional_texts"]) == 31
    assert all(x is not None for x in out["conditional_texts"])
    assert len(out["golden_answers"]) == 31 and len(out["raw_questions"]) == 31

    full_lens = [len(ex["prompt"]) for ex in examples]
    trunc_lens = [len(nm) for nm in out["normalized_messages"]]
    dropped = [f - t for f, t in zip(full_lens, trunc_lens)]
    print(f"[1b] truncation: {sum(1 for d in dropped if d)} items dropped messages "
          f"(max {max(dropped)} msgs); normalized_messages = truncated")
    assert all(0 <= d for d in dropped)

    p_toks = [len(tok.encode(p)) for p in out["prompt_texts"]]
    xo_toks = [len(tok.encode(x)) for x in out["conditional_texts"]]
    p_over = sum(1 for t in p_toks if t > STUDENT_LIMIT)
    p_over_total = sum(1 for t in p_toks if t > TOTAL_LIMIT)
    xo_over = sum(1 for t in xo_toks if t > TEACHER_LIMIT)
    xo_over_total = sum(1 for t in xo_toks if t > TOTAL_LIMIT)
    print(f"[2] prompt tokens: {min(p_toks)}-{max(p_toks)}, median {sorted(p_toks)[15]}"
          f" — {p_over}/31 > {STUDENT_LIMIT} (0 after truncation), {p_over_total}/31 > {TOTAL_LIMIT}")
    print(f"[3] conditional tokens: {min(xo_toks)}-{max(xo_toks)}, median {sorted(xo_toks)[15]}"
          f" — {xo_over}/31 > {TEACHER_LIMIT} (0 after truncation), {xo_over_total}/31 > {TOTAL_LIMIT}")
    assert p_over == 0 and xo_over == 0, "truncation failed to enforce budgets"
    assert p_over_total == 0 and xo_over_total == 0, "limit blowout"

    sys_complete = hint_complete = q_dropped = 0
    for i, ex in enumerate(examples):
        sys_msg = ex["prompt"][0]
        if sys_msg["role"] == "system":
            sys_content = sys_msg["content"]
            assert sys_content[:100] in out["prompt_texts"][i], "system head missing"
            assert sys_content[-100:] in out["prompt_texts"][i], "system tail missing"
            sys_complete += 1
        hint = HINT_PREFIX + out["golden_answers"][i]
        assert out["golden_answers"][i] in out["conditional_texts"][i], "hint incomplete"
        hint_complete += 1
        last_q = next(m["content"] for m in reversed(ex["prompt"]) if m["role"] == "user")
        if not any(m.get("content") == last_q for m in out["normalized_messages"][i]):
            q_dropped += 1
    print(f"[2b] completeness: system 31/31, hint {hint_complete}/31, questions dropped {q_dropped}/31")
    assert sys_complete == 31 and hint_complete == 31 and q_dropped == 0

    call_lines = 0
    for g in out["golden_answers"]:
        assert g, "empty target"
        call_lines += sum(1 for line in g.splitlines()
                          if line.split("(", 1)[0] in all_names)
    print(f"[4] targets: all non-empty; tool-call lines = {call_lines} (expect 88)")
    assert call_lines == 88

    merged = appended = 0
    for i, ex in enumerate(examples):
        hint = HINT_PREFIX + out["golden_answers"][i]
        cond = out["conditional_texts"][i]
        if ex["prompt"][-1]["role"] == "user":
            assert hint in cond and not cond.endswith(hint + "<|im_end|>"), "hint not merged inline"
            merged += 1
        else:
            idx_tool = cond.rindex("</tool_response>")
            idx_hint = cond.rindex(hint)
            assert idx_hint > idx_tool, "hint not after tool results"
            last_user = cond.rindex("<|im_start|>user\n")
            gen = cond.rindex("<|im_start|>assistant\n<think>")
            assert last_user > idx_tool
            assert cond[last_user:gen] == "<|im_start|>user\n" + hint + "<|im_end|>\n", "hint not the sole last user turn"
            appended += 1
    print(f"[5] hint placement: {merged} merged into last user msg, {appended} appended as new user turn")
    assert appended == 26 and merged == 5

    for i in range(31):
        if ex_last_is_tool := examples[i]["prompt"][-1]["role"] != "user":
            assert "<tool_response>" in out["prompt_texts"][i]
            break
    print("[6] tool_response body present in prompt render")

    # ---------------- Phase 2: family guard (gpt-oss + OLS shape) ----------------
    collator2 = reset({"TRAIN_DATA_PATH": OLS_DATA, "MODEL_NAME": "openai/gpt-oss-20b"})
    assert not collator2.IS_QWEN and collator2.IS_GPT_OSS
    try:
        collator2.SDFTCollator(tokenizer=tok, hindsight_field="user_response")(examples)
        raise AssertionError("guard did not fire")
    except ValueError as e:
        assert "not validated for model family" in str(e)
        print(f"[7] family guard fires for gpt-oss + OLS shape: ValueError — {e}")

    # gpt-oss + old shape (no tools file, no tool messages) must NOT fire
    old_dir = Path("/tmp/opencode/old_format")
    old_dir.mkdir(parents=True, exist_ok=True)
    old_ex = {"prompt": [{"from": "human", "value": "q"}],
              "user_response": {"value": "a"}}
    (old_dir / "train.jsonl").write_text(json.dumps(old_ex) + "\n")
    collator2b = reset({"TRAIN_DATA_PATH": str(old_dir / "train.jsonl"),
                        "MODEL_NAME": "openai/gpt-oss-20b"})
    assert collator2b.TOOL_DEFS is None
    out_old = collator2b.SDFTCollator(tokenizer=tok, hindsight_field="user_response")([old_ex])
    assert out_old["prompt_texts"][0] and out_old["golden_answers"][0] == "a"
    assert out_old["prompt_texts"][0].endswith(FINAL_CHANNEL), "gpt-oss prompt lacks final channel suffix"
    assert out_old["conditional_texts"][0].endswith(FINAL_CHANNEL), "gpt-oss conditional lacks final channel suffix"
    print("[8] gpt-oss + old format (no tool shape): no guard fire, renders, final channel appended")

    # ---------------- Phase 3: old-format regression (no tools file) ----------------
    collator3 = reset({"TRAIN_DATA_PATH": str(old_dir / "train.jsonl"),
                       "MODEL_NAME": "Qwen/Qwen3-8B"})
    assert collator3.TOOL_DEFS is None
    ex3 = {
        "prompt": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "what is 2+2?"},
            {"from": "gpt", "value": "it's 4"},
            {"from": "human", "value": "explain more"},
        ],
        "user_response": {"value": "4 because two plus two equals four"},
        "enriched_user_response": {"value": "basic arithmetic"},
    }
    c3 = collator3.SDFTCollator(tokenizer=tok, hindsight_field="enriched_user_response")
    out3 = c3([ex3])
    p3, xo3 = out3["prompt_texts"][0], out3["conditional_texts"][0]
    assert "<tools>" not in p3 and "<tools>" not in xo3, "tools block rendered with TOOL_DEFS None"
    assert "<tool_response>" not in p3
    assert "Documentation:\nbasic arithmetic" in xo3 and "Answer:\n4 because" in xo3
    assert out3["raw_questions"][0] == "explain more"
    assert out3["golden_answers"][0] == "4 because two plus two equals four"
    assert "explain more\n\nThe following is the relevant documentation" in xo3, "hint not merged into last user msg"
    assert "Documentation:\nbasic arithmetic\n\nAnswer:\n4 because two plus two equals four<|im_end|>" in xo3
    print("[9] old-format regression: renders, no tools block, hint merged into last user msg")

    # mirror-check: hand-render from _append_hint output equals conditional_texts
    from megatron_trainer import collator as col
    h = col._append_hint(out3["normalized_messages"][0], "hint")
    assert h[-1]["role"] == "user" and h[-1]["content"].endswith("hint")
    print("[10] _append_hint helper: merge path ok")

    # ---------------- Phase 4: drop filter (no raises on budgets) ----------------
    # [11] init budget check warns (no crash); [12] empty dataset is a hard error
    collator4 = reset({"TRAIN_DATA_PATH": OLS_DATA, "MODEL_NAME": "Qwen/Qwen3-8B",
                       "STUDENT_MAX_PROMPT_LEN": "100"})
    collator4.SDFTCollator(tokenizer=tok, hindsight_field="user_response")  # must not raise
    print("[11] init budget check warns (no raise) on tiny STUDENT_MAX_PROMPT_LEN")
    try:
        collator4.SDFTCollator(tokenizer=tok, hindsight_field="user_response").filter_dataset(
            Dataset.from_list(examples)
        )
        raise AssertionError("empty-dataset did not raise")
    except ValueError as e:
        assert "All examples dropped" in str(e)
        print(f"[12] empty filtered dataset raises: ValueError — {str(e)[:70]}...")

    # [13] partial drop: over-budget hints dropped, rest kept, warnings logged
    collator5 = reset({"TRAIN_DATA_PATH": str(old_dir / "train.jsonl"),
                       "MODEL_NAME": "Qwen/Qwen3-8B",
                       "STUDENT_MAX_PROMPT_LEN": "14336",
                       "TEACHER_MAX_PROMPT_LEN": "2048"})
    assert collator5.TOOL_DEFS is None
    c5 = collator5.SDFTCollator(tokenizer=tok, hindsight_field="enriched_user_response")
    legacy_big = {"prompt": [{"from": "human", "value": "q"}],
                  "user_response": {"value": "a"},
                  "enriched_user_response": {"value": "doc " * 3000}}
    legacy_small = {"prompt": [{"from": "human", "value": "q"}],
                    "user_response": {"value": "a"},
                    "enriched_user_response": {"value": "short"}}
    kept5 = c5.filter_dataset(Dataset.from_list([legacy_big, legacy_small, legacy_big]))
    assert len(kept5) == 1, "expected 2 over-budget hints dropped, 1 kept"
    out5 = c5(list(kept5))
    assert "short" in out5["conditional_texts"][0] and "doc " * 3000 not in out5["conditional_texts"][0]
    print("[13] drop filter: 2 over-budget hints dropped (warned), 1 kept")

    print("\nALL CHECKS PASSED")


def channel_compare():
    """gpt-oss 128-token generation: forced analysis channel vs forced final channel."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GPT_OSS_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        GPT_OSS_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    base = tok.apply_chat_template(
        [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome. Explain your approach in one paragraph."}],
        tokenize=False,
        add_generation_prompt=True,
    ).rstrip()
    assert base.endswith("<|start|>assistant")

    for name, suffix in [("analysis", ANALYSIS_CHANNEL), ("final", FINAL_CHANNEL)]:
        prompt = base + suffix
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=128, do_sample=False)
        new_ids = out[0][ids["input_ids"].shape[1]:]
        raw = tok.decode(new_ids, skip_special_tokens=False)
        text = tok.decode(new_ids, skip_special_tokens=True)
        channels = [c for c in ("analysis", "commentary", "final")
                    if f"<|channel|>{c}<|message|>" in raw]
        print(f"\n===== channel={name} | {new_ids.shape[0]} tokens | channels seen in output: {channels} =====")
        print(f"raw (special tokens): {raw!r}")
        print(f"clean completion: {text!r}")


if __name__ == "__main__":
    main()
    if os.environ.get("CHANNEL_COMPARE") == "1":
        channel_compare()
