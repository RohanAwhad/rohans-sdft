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
# at a time. ASYNC_IN_ORDER=1 keeps the deterministic batch-in-order producer
# (Layer 1 verification mode). N_ASYNC bounds in-flight generations
# (Magistral's conservative limit: 2 * step batch).
ASYNC_ROLLOUT = os.environ.get("ASYNC_ROLLOUT", "0") == "1"
ASYNC_IN_ORDER = os.environ.get("ASYNC_IN_ORDER", "0") == "1"
N_ASYNC = int(os.environ.get("N_ASYNC", str(2 * GRAD_ACCUM_STEPS)))

# Determinism knobs for A/B verification runs. Both unset = default
# nondeterministic behavior. TRAINER_SEED fixes the dataloader shuffle;
# VLLM_SEED fixes vLLM sampling (per-request seed).
TRAINER_SEED = os.environ.get("TRAINER_SEED")
TRAINER_SEED = int(TRAINER_SEED) if TRAINER_SEED else None
VLLM_SEED = os.environ.get("VLLM_SEED")
VLLM_SEED = int(VLLM_SEED) if VLLM_SEED else None

# Debug: log a hash of every rolled-out (prompt, completion) pair with its
# batch/step index — lets verification runs diff the exact data stream.
DEBUG_ROLLOUT_HASH = os.environ.get("DEBUG_ROLLOUT_HASH", "0") == "1"

# Rollout replay for Layer 1 verification: RECORD_ROLLOUT_PATH dumps every
# vLLM generation result (keyed by prompt hash) to a jsonl file;
# ROLLOUT_REPLAY_PATH replays those results instead of hitting vLLM — both
# modes then train on byte-identical rollout data, making the cross-mode
# comparison exact (vLLM's internal batching numerics are not cross-run
# reproducible, so replay is the only way to isolate the plumbing).
RECORD_ROLLOUT_PATH = os.environ.get("RECORD_ROLLOUT_PATH", "")
ROLLOUT_REPLAY_PATH = os.environ.get("ROLLOUT_REPLAY_PATH", "")
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
