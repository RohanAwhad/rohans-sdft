"""Benchmark: 32 sequential vs batched forward passes on Qwen3-8B."""

import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

DEVICE = "cuda:0"
MODEL = "Qwen/Qwen3-8B"
SEQ_LEN = 8192
N_SEQS = 32
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
PROMPT_LEN = SEQ_LEN // 2  # simulate half prompt, half completion

print(f"Loading {MODEL} in bf16 on {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE)
model.eval()
print("Model loaded.")

# Pre-generate random token IDs (same for all configs)
input_ids = torch.randint(0, model.config.vocab_size, (N_SEQS, SEQ_LEN), device=DEVICE)

# Warmup
print("Warmup...")
with torch.inference_mode():
    logits = model(input_ids=input_ids[:1], use_cache=False).logits[0]
    F.log_softmax(logits[PROMPT_LEN - 1 : SEQ_LEN - 1].float(), dim=-1)
torch.cuda.synchronize()
print("Warmup done.\n")

print(f"{'batch_size':>10}  {'calls':>5}  {'total':>8}  {'per_call':>8}")
print("-" * 40)

for bs in BATCH_SIZES:
    torch.cuda.synchronize()
    t0 = time.monotonic()
    with torch.inference_mode():
        for start in range(0, N_SEQS, bs):
            logits = model(
                input_ids=input_ids[start : start + bs],
                use_cache=False,
            ).logits
            # log_softmax on completion positions, matching server behavior
            for i in range(logits.size(0)):
                F.log_softmax(logits[i, PROMPT_LEN - 1 : SEQ_LEN - 1].float(), dim=-1)
    # torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    n_calls = N_SEQS // bs
    print(f"{bs:>10}  {n_calls:>5}  {elapsed:>7.2f}s  {elapsed / n_calls:>7.3f}s")
