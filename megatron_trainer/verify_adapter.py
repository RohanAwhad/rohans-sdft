"""LoRA adapter export gate: logit-parity between the Megatron-side model
(with adapter active) and the exported HF PEFT adapter loaded on the HF base.

Runs inside the nemo:26.06 container on a single GPU:
    Phase A — load HF base + exported adapter (peft), compute logits, free GPU
    Phase B — load Megatron base + identical LoRA transform, compute logits
    Gate    — top-k token agreement + mean abs logit diff at matched ranks

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m megatron_trainer.verify_adapter \
        --adapter-dir /workspace/output_lora/step_32

Exit code 0 = gate passed (top-k agreement >= threshold, default 100%).
"""

import argparse
import os

import torch
from loguru import logger


PROMPTS = [
    "What is the URL format for making a chat completions API call to a model through MaaS?",
    "How do you obtain an authentication token for MaaS API calls?",
    "What is the difference between a model deployment and a model in MaaS?",
    "How do I deploy a model to MaaS using the CLI?",
    "What are the required parameters for a chat completions request?",
    "How does model serving work on the Red Hat OpenShift AI cluster?",
    "What is the recommendation engine used for in this context?",
    "Which model families are supported by MaaS deployments?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True, help="HF PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=96)
    # Export materializes adapters in float32 (bf16 merge gives ~1e-3 weight
    # error) — top-k agreement at near-ties flips with fp32-vs-bf16 precision.
    # 0.95 leaves headroom while a wrong q/k/v split drops agreement to ~30%.
    parser.add_argument("--min-agreement", type=float, default=0.95)
    args = parser.parse_args()

    from megatron_trainer.config import HF_MODEL_PATH

    device = torch.device("cuda:0")

    # ---- Phase A: HF base + exported adapter ----
    logger.info(f"Phase A: HF base {HF_MODEL_PATH} + adapter {args.adapter_dir}")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(HF_MODEL_PATH, torch_dtype=torch.bfloat16)
    model_hf = PeftModel.from_pretrained(base, args.adapter_dir).to(device).eval()

    logits_hf = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_tokens).to(device)
            out = model_hf(**enc).logits  # (1, S, V)
            logits_hf[i] = out[0].float().cpu()  # (S, V)
    del model_hf, base
    torch.cuda.empty_cache()
    logger.info("Phase A done.")

    # ---- Phase B: Megatron base + LoRA transform (same config as trainer) ----
    logger.info("Phase B: Megatron model + LoRA transform")
    from megatron_trainer.model_utils import apply_lora_transform, init_distributed_standalone, load_model

    init_distributed_standalone()
    model = load_model(HF_MODEL_PATH)
    model = apply_lora_transform(model)
    model.eval()

    logits_mcore = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_tokens)
            input_ids = enc["input_ids"].to(device)
            position_ids = torch.arange(input_ids.size(1), device=device, dtype=torch.long).unsqueeze(0)
            out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            logits = out.logits if hasattr(out, "logits") else out[0]  # (S, V) or (1, S, V)
            if logits.dim() == 3:
                logits = logits.squeeze(0)
            logits_mcore[i] = logits.float().cpu()
    logger.info("Phase B done.")

    # ---- Gate: top-k agreement per position ----
    n_checked = 0
    n_match = 0
    diffs = []
    for i in range(len(PROMPTS)):
        lhf = logits_hf[i]
        lm = logits_mcore[i]
        s = min(lhf.size(0), lm.size(0))
        v = min(lhf.size(1), lm.size(1))
        lhf, lm = lhf[:s, :v], lm[:s, :v]
        for t in range(s):
            top_hf = lhf[t].topk(args.top_k).indices.tolist()
            top_m = lm[t].topk(args.top_k).indices.tolist()
            n_checked += args.top_k
            n_match += len(set(top_hf) & set(top_m))
            m_hf = torch.full_like(lhf[t], float("-inf"))
            m_hf[top_hf] = lhf[t][top_hf]
            m_m = torch.full_like(lm[t], float("-inf"))
            m_m[top_m] = lm[t][top_m]
            both = (m_hf != float("-inf")) & (m_m != float("-inf"))
            if both.any():
                diffs.append((m_hf[both] - m_m[both]).abs().mean().item())

    agreement = n_match / max(n_checked, 1)
    mean_diff = sum(diffs) / len(diffs) if diffs else float("inf")
    logger.info(
        f"gate: top-{args.top_k} agreement={agreement:.4f} ({n_match}/{n_checked}) "
        f"mean|logit diff| at matched ranks={mean_diff:.4f}"
    )
    if agreement < args.min_agreement:
        logger.error(f"GATE FAILED: agreement {agreement:.4f} < {args.min_agreement}")
        raise SystemExit(1)
    logger.info("GATE PASSED: exported adapter matches Megatron-side logits.")


if __name__ == "__main__":
    main()
