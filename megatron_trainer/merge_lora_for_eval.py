"""Merge an HF-PEFT LoRA adapter checkpoint (as produced by
save_hf_adapter_checkpoint / TRAIN_MODE=lora) into its base model, producing
a standalone HF checkpoint that eval_with_retrieval.py (or any vLLM
`LLM(model=...)` call) can load directly.

Standalone HF peft merge — NOT megatron.bridge's examples/peft/merge_lora.py,
which expects a Megatron-native dist_checkpointing checkpoint. Our trainer's
LoRA checkpoints are already in HF PEFT format (adapter_config.json +
adapter_model.safetensors), so a plain PeftModel.merge_and_unload() applies
directly — no Megatron/GPU-parallelism machinery needed.

Usage:
    python -m megatron_trainer.merge_lora_for_eval \
        --base-model unsloth/gpt-oss-20b-BF16 \
        --adapter-path /path/to/step_25 \
        --output-path /path/to/step_25_merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_checkpoint(base_model: str, adapter_path: str, output_path: str) -> None:
    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()
    print(f"Saving merged checkpoint: {output_path}")
    merged.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    merge_lora_checkpoint(args.base_model, args.adapter_path, args.output_path)
