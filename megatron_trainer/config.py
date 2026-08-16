"""SDFT training configuration (Megatron Bridge version). All overridable via environment variables."""

import os

MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME is required (e.g. MODEL_NAME=Qwen/Qwen3-8B)")
HF_MODEL_PATH = os.environ.get("HF_MODEL_PATH", MODEL_NAME)

# Optional frozen teacher. When set, the logprob server loads THIS model via
# plain HF transformers (mxfp4 checkpoints like openai/gpt-oss-120b auto-load
# in 4-bit) and the per-step student->teacher weight sync (NCCL + EMA) is
# disabled. When empty, current behavior: teacher = student (HF_MODEL_PATH,
# EMA-blended each step).
TEACHER_MODEL_PATH = os.environ.get("TEACHER_MODEL_PATH", "")

# gpt-oss models use a channel-based chat protocol (analysis/commentary/final).
# When True, the collator appends an explicit final-channel suffix to
# generation prompts and vLLM must return special tokens (skip_special_tokens=False).
IS_GPT_OSS = "gpt-oss" in MODEL_NAME.lower()

# Qwen-family models are the only ones validated for the new-format path
# (tools / tool_calls / tool_results). Non-Qwen models with that shape raise.
IS_QWEN = "qwen" in MODEL_NAME.lower()

# Student thinking mode ("1" = thinking on). Qwen renders with
# enable_thinking=True (native CoT, no empty <think> block); gpt-oss switches
# to the analysis channel. Applies to both student and teacher renders.
STUDENT_THINKING = os.environ.get("STUDENT_THINKING", "0") == "1"

if STUDENT_THINKING and not (IS_QWEN or IS_GPT_OSS):
    raise ValueError(
        "STUDENT_THINKING=1 is only validated for Qwen and gpt-oss model "
        f"families, got MODEL_NAME={MODEL_NAME}"
    )

# GPU assignment (physical GPU IDs, used in CUDA_VISIBLE_DEVICES)
GPU_VLLM = int(os.environ.get("GPU_VLLM", "0"))
GPU_TRAINER = int(os.environ.get("GPU_TRAINER", "1"))
GPU_LOGPROB_SERVER = int(os.environ.get("GPU_LOGPROB_SERVER", "2"))

# LoRA mode (see docs/megatron_trainer/lora.md). TRAIN_MODE=full keeps the
# existing full fine-tuning path; TRAIN_MODE=lora trains bridge LoRA adapters
# on a frozen base and serves them via vLLM hot-swap.
TRAIN_MODE = os.environ.get("TRAIN_MODE", "full")
if TRAIN_MODE not in ("full", "lora"):
    raise ValueError(f"TRAIN_MODE must be 'full' or 'lora', got {TRAIN_MODE!r}")
LORA_DIM = int(os.environ.get("LORA_DIM", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))
LORA_TARGET_MODULES = [
    m.strip()
    for m in os.environ.get(
        "LORA_TARGET_MODULES", "linear_qkv,linear_proj,linear_fc1,linear_fc2"
    ).split(",")
    if m.strip()
]
LORA_ADAPTER_NAME = os.environ.get("LORA_ADAPTER_NAME", "sdft-policy")

# vLLM LoRA slot ranks (vllm.lora.utils.MaxLoRARanks enum); --max-lora-rank
# must equal LORA_DIM exactly.
_VLLM_LORA_RANKS = {1, 8, 16, 32, 64, 128, 256, 320, 512}
if TRAIN_MODE == "lora" and LORA_DIM not in _VLLM_LORA_RANKS:
    raise ValueError(
        f"LORA_DIM must be one of {sorted(_VLLM_LORA_RANKS)} (vLLM max-lora-rank "
        f"enum), got {LORA_DIM}"
    )

# Training hyperparams
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
# LR schedule: "constant" (fixed LEARNING_RATE) or "cosine" (linear warmup of
# min(10% of total optimizer steps, 100), then cosine decay to 0)
LR_SCHEDULER = os.environ.get("LR_SCHEDULER", "constant")
if LR_SCHEDULER not in ("constant", "cosine"):
    raise ValueError(
        f"LR_SCHEDULER must be 'constant' or 'cosine', got {LR_SCHEDULER!r}"
    )
BATCH_SIZE = 1  # always 1; effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "32"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "10"))

# Async streaming rollouts (see docs/megatron_trainer/async_rollouts.md).
# ASYNC_ROLLOUT=1 moves generation to a rank-0 producer thread feeding a
# bounded queue; the main path consumes one microbatch (world_size samples)
# at a time. N_ASYNC bounds in-flight generations.
ASYNC_ROLLOUT = os.environ.get("ASYNC_ROLLOUT", "0") == "1"
N_ASYNC = int(os.environ.get("N_ASYNC", str(2 * GRAD_ACCUM_STEPS)))

# Determinism knobs for A/B verification runs. Both unset = default
# nondeterministic behavior. TRAINER_SEED fixes the dataloader shuffle;
# VLLM_SEED fixes vLLM sampling (per-request seed).
TRAINER_SEED = os.environ.get("TRAINER_SEED")
TRAINER_SEED = int(TRAINER_SEED) if TRAINER_SEED else None
VLLM_SEED = os.environ.get("VLLM_SEED")
VLLM_SEED = int(VLLM_SEED) if VLLM_SEED else None

MAX_GRAD_NORM = 1.0
EMA_ALPHA = float(os.environ.get("EMA_ALPHA", "0.05"))
STUDENT_MAX_PROMPT_LEN = int(os.environ.get("STUDENT_MAX_PROMPT_LEN", "2048"))
TEACHER_MAX_PROMPT_LEN = int(os.environ.get("TEACHER_MAX_PROMPT_LEN", "2048"))
MAX_TOTAL_LEN = int(os.environ.get("MAX_TOTAL_LEN", "8192"))

# Generation (vLLM rollout)
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "512"))
GEN_MAX_NEW_TOKENS = int(
    os.environ.get("GEN_MAX_NEW_TOKENS", str(MAX_TOTAL_LEN - STUDENT_MAX_PROMPT_LEN))
)
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "1.0"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "1.0"))
# Read timeout per vLLM completion request. 2048-token completions at ~20 tok/s
# (slow processed_logprobs path) run ~100s; 180s killed runs on tail-heavy
# prompts. Timeouts are retried 3x in vllm_generate, then the sample is skipped.
VLLM_COMPLETION_TIMEOUT = int(os.environ.get("VLLM_COMPLETION_TIMEOUT", "600"))

# Importance sampling (vLLM is the rollout engine; its proposal distribution
# can drift from the training policy via weight staleness or sampling params).
# Weights the reverse-KL loss by exp(policy_logp - rollout_logp), truncated at
# IS_CAP (Truncated Importance Sampling, same scheme as TRL DistilTrainer).
IS_WEIGHTING = os.environ.get("IS_WEIGHTING", "1") == "1"
IS_CAP = float(os.environ.get("IS_CAP", "5.0"))

if STUDENT_MAX_PROMPT_LEN + GEN_MAX_NEW_TOKENS > MAX_TOTAL_LEN:
    raise ValueError(
        "STUDENT_MAX_PROMPT_LEN + GEN_MAX_NEW_TOKENS must be <= MAX_TOTAL_LEN "
        f"({STUDENT_MAX_PROMPT_LEN} + {GEN_MAX_NEW_TOKENS} > {MAX_TOTAL_LEN})"
    )

# ---------------------------------------------------------------------------
# GRPO mode (see docs/megatron_trainer/grpo.md). LOSS_TYPE=sdft (default) is
# the existing reverse-KL distillation path, byte-identical. LOSS_TYPE=grpo
# replaces it with on-policy group-relative policy gradient against env
# verdicts (reflector PASS/FAIL) — no teacher, no distillation loss.
# ---------------------------------------------------------------------------
LOSS_TYPE = os.environ.get("LOSS_TYPE", "sdft")
if LOSS_TYPE not in ("sdft", "grpo"):
    raise ValueError(f"LOSS_TYPE must be 'sdft' or 'grpo', got {LOSS_TYPE!r}")

GRPO_GROUPS = int(os.environ.get("GRPO_GROUPS", "8"))  # G completions per prompt
GRPO_ADV = os.environ.get("GRPO_ADV", "mean")  # advantage estimator
GRPO_CLIP_LOW = float(os.environ.get("GRPO_CLIP_LOW", "0.2"))
GRPO_CLIP_HIGH = float(os.environ.get("GRPO_CLIP_HIGH", "0.28"))  # DAPO clip-higher
GRPO_OLD_LOGPS = os.environ.get("GRPO_OLD_LOGPS", "vllm")  # vllm | detached
GRPO_IS_C_MAX = float(os.environ.get("GRPO_IS_C_MAX", "3.0"))  # sequence-level TIS clamp
GRPO_KL_COEF = float(os.environ.get("GRPO_KL_COEF", "0.0"))  # beta; 0 = no reference forward
GRPO_FILTER_GROUPS = os.environ.get("GRPO_FILTER_GROUPS", "0") == "1"  # DAPO dynamic sampling
GRPO_MAX_GEN_BATCHES = int(os.environ.get("GRPO_MAX_GEN_BATCHES", "10"))  # resample cap (verl convention)
GRPO_LR = float(os.environ.get("GRPO_LR", "1e-6"))
GRPO_LR_WARMUP_STEPS = int(os.environ.get("GRPO_LR_WARMUP_STEPS", "15"))  # linear warmup, then constant
GRPO_GRAD_CLIP = float(os.environ.get("GRPO_GRAD_CLIP", "0.2"))
GRPO_MASK_TRUNCATED = os.environ.get("GRPO_MASK_TRUNCATED", "1") == "1"  # never punish length-truncated completions

# Whether the logprob server (teacher forward passes) is needed at all.
# GRPO with GRPO_KL_COEF=0 (default) needs no reference model — the whole
# logprob-server subsystem is skipped, freeing its GPU(s) for vLLM rollout
# capacity instead (G completions/prompt means G x the generation load).
USE_LOGPROB_SERVER = not (LOSS_TYPE == "grpo" and GRPO_KL_COEF == 0.0)

if LOSS_TYPE == "grpo":
    # v1 implementation scope — everything below is a hard requirement or an
    # explicit "not built yet" fence (fail fast at import time, never a silent
    # partial behavior). See docs/megatron_trainer/grpo.md for the full design.
    # TRAIN_MODE=lora is supported (as of issue #22's fix, PR #23): full FT's
    # FSDP-sharded AdamW state doesn't fit a 20B model on <4 trainer GPUs
    # (2-way shard: ~10B params/rank * 2 (bf16 exp_avg+exp_avg_sq) + params +
    # grads ~= 80GB, right at the H100 80GB ceiling) — LoRA only optimizes
    # adapter params, sidestepping that entirely, and the gpt-oss MoE adapter
    # export layout bug that previously blocked it (E046) is now fixed.
    if not ASYNC_ROLLOUT:
        raise ValueError(
            "LOSS_TYPE=grpo requires ASYNC_ROLLOUT=1 (group-atomic streaming producer)"
        )
    if GRPO_GROUPS < 2:
        raise ValueError(f"GRPO_GROUPS must be >= 2, got {GRPO_GROUPS}")
    if GRAD_ACCUM_STEPS % GRPO_GROUPS != 0:
        raise ValueError(
            f"GRAD_ACCUM_STEPS ({GRAD_ACCUM_STEPS}) must be divisible by "
            f"GRPO_GROUPS ({GRPO_GROUPS}) — groups are rank-local and must "
            f"divide evenly (world_size divisibility is checked at trainer "
            f"startup once world_size is known)."
        )
    if GRPO_ADV != "mean":
        raise NotImplementedError(
            f"GRPO_ADV={GRPO_ADV!r} not implemented yet (v1 only supports 'mean'; "
            "'zscore'/'median' are documented experiment knobs, not yet built)"
        )
    if GRPO_OLD_LOGPS not in ("vllm", "detached"):
        raise ValueError(f"GRPO_OLD_LOGPS must be 'vllm' or 'detached', got {GRPO_OLD_LOGPS!r}")
    if GRPO_KL_COEF > 0:
        raise NotImplementedError(
            "GRPO_KL_COEF > 0 (reference KL against an anchor) not implemented yet"
        )
    if GRPO_MAX_GEN_BATCHES < 1:
        raise ValueError(f"GRPO_MAX_GEN_BATCHES must be >= 1, got {GRPO_MAX_GEN_BATCHES}")

# vLLM server
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}"

# Multi-instance vLLM: comma-separated ports (e.g. "8001,8002,8003")
# Falls back to single VLLM_PORT if not set.
_vllm_ports_str = os.environ.get("VLLM_PORTS", "")
VLLM_BASE_URLS: list[str] = (
    [f"http://localhost:{p.strip()}" for p in _vllm_ports_str.split(",") if p.strip()]
    if _vllm_ports_str
    else [VLLM_BASE_URL]
)

# Logprob server (HTTP)
LOGPROB_PORT = int(os.environ.get("LOGPROB_PORT", "8010"))
LOGPROB_BASE_URL = f"http://localhost:{LOGPROB_PORT}"
# Logprob server (TCP — teacher logprob data plane; HTTP stays for health/weight-sync)
LOGPROB_TCP_PORT = int(os.environ.get("LOGPROB_TCP_PORT", "8011"))

# Dataset
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH")
if not TRAIN_DATA_PATH:
    raise ValueError(
        "TRAIN_DATA_PATH is required (path to the training .jsonl, "
        "e.g. /workspace/.../train_sdft.jsonl)"
    )

# Collator
HINDSIGHT_FIELD = os.environ.get("HINDSIGHT_FIELD", "enriched_user_response")

# Environment type: "rag" or "api_adapter"
ENV_TYPE = os.environ.get("ENV_TYPE", "rag")

# API-Adapter env
API_MODEL = os.environ.get("API_MODEL", "vertex_ai/claude-haiku-4-5@20251001")
MAX_ADAPTER_TURNS = int(os.environ.get("MAX_ADAPTER_TURNS", "5"))

# Reflector (used by RagEnv in online_feedback mode)
REFLECTOR_MODEL = os.environ.get("REFLECTOR_MODEL", "claude-sonnet-4-6@default")
REFLECTOR_REGION = os.environ.get("REFLECTOR_REGION", "us-east5")
REFLECTOR_PROJECT_ID = os.environ.get("REFLECTOR_PROJECT_ID", "")

# Wandb
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdft-online")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_NAME = os.environ.get("WANDB_NAME")

# Output
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "200"))
