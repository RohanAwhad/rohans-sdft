"""SDFT training configuration. All overridable via environment variables."""

import os

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")

# GPU assignment (physical GPU IDs, used in CUDA_VISIBLE_DEVICES)
GPU_VLLM = int(os.environ.get("GPU_VLLM", "0"))
GPU_TRAINER = int(os.environ.get("GPU_TRAINER", "1"))
GPU_LOGPROB_SERVER = int(os.environ.get("GPU_LOGPROB_SERVER", "2"))

# Training hyperparams
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
BATCH_SIZE = 1  # always 1; effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "32"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "10"))
MAX_GRAD_NORM = 10.0
EMA_ALPHA = float(os.environ.get("EMA_ALPHA", "0.05"))  # teacher EMA: phi = alpha*theta + (1-alpha)*phi

# Generation (vLLM rollout)
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "512"))
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "2048"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "0.95"))

# vLLM server
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}"

# NCCL (torch.distributed group for trainer <-> logprob server)
NCCL_MASTER_PORT = int(os.environ.get("NCCL_MASTER_PORT", "29500"))

# Dataset
TRAIN_DATA_PATH = os.environ.get(
    "TRAIN_DATA_PATH",
    "/home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/train_maas_sdft.jsonl",
)

# Collator
HINDSIGHT_FIELD = os.environ.get("HINDSIGHT_FIELD", "enriched_user_response")

# Reflector (used when HINDSIGHT_FIELD=online_feedback)
REFLECTOR_MODEL = os.environ.get("REFLECTOR_MODEL", "claude-sonnet-4-6@default")
REFLECTOR_REGION = os.environ.get("REFLECTOR_REGION", "us-east5")
REFLECTOR_PROJECT_ID = os.environ.get("REFLECTOR_PROJECT_ID", "")

# API-Adapter env
API_MODEL = os.environ.get("API_MODEL", "vertex_ai/claude-haiku-4-5@20251001")
MAX_ADAPTER_TURNS = int(os.environ.get("MAX_ADAPTER_TURNS", "5"))

# Wandb
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdft-online")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_NAME = os.environ.get("WANDB_NAME")

# Output
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "200"))  # save checkpoint every N optimizer steps
