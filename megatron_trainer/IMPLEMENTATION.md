# DDP SDFT Training — Implementation Notes

Based on thorough codebase exploration of the current single-rank trainer.
Reference: `DESIGN.md` for the high-level architecture.

---

## 1. Architecture: Three Fully Independent Processes

Three processes, each with its own `torch.distributed` world (or none).
No shared process group, no `new_group()`, no unified entry point.

```
vLLM server          Trainer ranks (torchrun)           Logprob server
─────────────        ──────────────────────────         ──────────────────
separate process     torch.distributed world_size=N     torch.distributed world_size=1
no torch.distributed (trainers only)                    (for Megatron model loading only)
NCCLWeightTransfer   Megatron DDP (wrap_with_ddp=True)  FastAPI HTTP server
Engine for weight    Megatron distributed optimizer      PyNcclCommunicator for weight sync
sync from trainer    (AdamW)                             (initialized via HTTP handshake)
```

**Why not a shared world?** `dist.new_group()` requires all processes to be
in the same `torch.distributed` world. That forces the logprob server into the
trainers' process group, complicating Megatron's parallel state (DP would be
N+1 instead of N). Keeping them separate means Megatron sees world_size=N for
trainers and correctly sets DP=N, TP=1, PP=1. The logprob server uses
world_size=1 just to satisfy Megatron/AutoBridge init requirements.

### Launch pattern

```bash
# GPU 0: vLLM (separate process, unchanged)
CUDA_VISIBLE_DEVICES=0 python start_vllm_patched.py --model ... --port 8004

# GPU 3: logprob server (separate standalone process)
CUDA_VISIBLE_DEVICES=3 python -m megatron_trainer.logprob_server --port 8010

# GPUs 1,2: trainer ranks (torchrun for Megatron DDP)
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 \
    -m megatron_trainer.trainer --logprob-url http://localhost:8010
```

Each process type has its own entry point:
- `trainer.py` — launched via `torchrun`, uses Megatron DDP
- `logprob_server.py` — standalone process, FastAPI + PyNcclCommunicator
- `start_vllm_patched.py` — vLLM server (unchanged)

No unified `main.py`. No rank-based branching.

---

## 2. Megatron DDP + Megatron Optimizer

### Why Megatron's DDP, not PyTorch DDP

Manual `MegatronDDP` wrapping puts us on Megatron's "blessed path":
- `main_grad` buffers are allocated (required by Megatron's optimizer)
- Gradient accumulation fusion works as designed
- Communication overlap (async allreduce) works as designed
- The distributed optimizer can shard optimizer state across DP ranks

With world_size=N (trainers only), Megatron correctly sets DP=N, TP=1, PP=1.
No DP group mismatch.

### Why Megatron's optimizer, not bitsandbytes

Current code (trainer.py:207-208):
```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
```

This is replaced by Megatron's distributed optimizer (AdamW). Benefits:
- Works with `main_grad` buffers (bitsandbytes doesn't know about them)
- `gradient_accumulation_fusion` can be re-enabled (currently disabled at
  model_utils.py:64 specifically because we weren't using a Megatron optimizer)
- Optimizer state is sharded across DP ranks (memory savings with N>1)
- No extra dependency (bitsandbytes)

```python
# Load model with wrap_with_ddp=False (existing load_model()), then manually wrap:
from megatron.core.distributed import DistributedDataParallel as MegatronDDP
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig()
ddp_model = MegatronDDP(config=model.config, ddp_config=ddp_config, module=model)

# Optimizer:
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

opt_config = OptimizerConfig(
    optimizer='adam',
    lr=LEARNING_RATE,
    bf16=True,
    clip_grad=MAX_GRAD_NORM,       # gradient clipping is internal — drop manual clip_grad_norm_
    use_distributed_optimizer=True, # shard optimizer state across DP ranks
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    weight_decay=0.01,
)
optimizer = get_megatron_optimizer(opt_config, model_chunks=[ddp_model])
```

### Why manual MegatronDDP wrapping instead of `wrap_with_ddp=True`

`provide_distributed_model(wrap_with_ddp=True)` re-creates the model from scratch
internally (calls `get_model()` → `_create_model()` → `MCoreGPTModel()`). During
exploration, this crashed with `output_layer_init_method is None` — likely because
`provider.finalize()` wasn't called in the test. It probably works with `finalize()`,
but manual wrapping is confirmed to work and has a practical benefit: `load_model()`
stays identical for both trainer and logprob server (always `wrap_with_ddp=False`),
and only the trainer adds the DDP wrapper after loading.

**Important**: `provider.finalize()` MUST be called before `provide_distributed_model()`.
It sets init methods and other config fields that Megatron requires. This is already
in our current `load_model()` code (model_utils.py:61).

---

## 3. Changes to `model_utils.py`

### `init_distributed()` — current (line 22-38)

```python
def init_distributed(rank: int, world_size: int = 2, master_port: int = 29500) -> None:
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(0)                          # <-- hardcoded to 0
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
```

### `init_distributed()` — new (trainer variant)

For trainers launched via `torchrun`, env vars are already set:

```python
def init_distributed_trainer() -> int:
    """Initialize torch.distributed for trainer DDP. Returns local_rank."""
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # torchrun sets RANK, WORLD_SIZE, MASTER_*
    return local_rank
```

### `init_distributed()` — new (logprob server variant)

The logprob server needs `torch.distributed` with world_size=1 for Megatron/AutoBridge
model loading, but doesn't participate in any DDP:

```python
def init_distributed_standalone() -> None:
    """Initialize torch.distributed world_size=1 for standalone Megatron model loading."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(get_free_port())  # any free port
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES already scoped to 1 GPU
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
```

### `load_model()` — current (line 41-71)

```python
provider.gradient_accumulation_fusion = False     # <-- was disabled for non-Megatron optimizer
provider.async_tensor_model_parallel_allreduce = False
models = provider.provide_distributed_model(wrap_with_ddp=False)
```

### `load_model()` — no changes needed

`load_model()` always uses `wrap_with_ddp=False` (current behavior). Both trainer
and logprob server call the same function. DDP wrapping is done separately in the
trainer after loading:

```python
model = load_model(HF_MODEL_PATH)  # returns raw MCore GPTModel
# Trainer only:
ddp_model = MegatronDDP(config=model.config, ddp_config=DistributedDataParallelConfig(), module=model)
```

### `DEVICE` — current

```python
DEVICE = torch.device("cuda:0")   # hardcoded everywhere
```

For trainers: `torch.device(f"cuda:{local_rank}")` where `local_rank` comes
from `torchrun` env var `LOCAL_RANK`.

For logprob server: `torch.device("cuda:0")` is correct since
`CUDA_VISIBLE_DEVICES` scopes to a single GPU.

### Functions that take `model` arg

`export_hf_weights_iter()`, `get_hf_weight_metadata()`, `save_hf_checkpoint()` all
receive the model and pass it to the AutoBridge. With Megatron DDP wrapping,
these need the **unwrapped** model:

```python
raw_model = model.module if hasattr(model, "module") else model
save_hf_checkpoint(raw_model, ckpt_dir, tokenizer)
```

Already partially handled — trainer.py line 190 does:
```python
unwrapped = model.module if hasattr(model, 'module') else model
```
Need to apply this pattern consistently to all call sites:
- `sync_weights_to_vllm(model.module, ...)` — pass unwrapped for HF conversion
- `save_hf_checkpoint(model.module, ...)` — pass unwrapped for HF export

---

## 4. Changes to `trainer.py`

### Signature: `train()` → `train()` (standalone entry point via torchrun)

No `ddp_group` or `logprob_group` args — the trainer's torch.distributed world
IS the DDP group (world_size=N, all trainers). Weight sync to logprob server
uses a separate standalone PyNcclCommunicator (see section 6).

### Data flow (replaces current lines 263-371)

```
CURRENT (single rank):
  for each GRAD_ACCUM_STEPS batch:
    1. Rollout via ApiAdapterEnv (ThreadPoolExecutor)
    2. For each micro_step:
       a. Encode completion → token IDs
       b. Get teacher logprobs via NCCL (request_teacher_log_probs)
       c. Student forward pass
       d. Compute KL loss, backward
    3. Optimizer step
    4. Sync weights (NCCL to logprob server + NCCL to vLLM)

NEW (Megatron DDP, rank 0 is master):
  for each GRAD_ACCUM_STEPS batch:
    1. [Rank 0] Rollout via ApiAdapterEnv (ThreadPoolExecutor) for all GRAD_ACCUM_STEPS items
    2. [Rank 0] Broadcast rollout data to all trainer ranks (dist.broadcast_object_list within DDP world)
    3. Each rank takes its slice: items[rank*M : (rank+1)*M]  where M = GRAD_ACCUM_STEPS // N
    4. For each local micro_step:
       a. Encode completion → token IDs
       b. Get teacher logprobs via HTTP (EACH rank calls logprob server independently for its own items)
       c. Student forward pass (Megatron DDP model)
       d. Compute KL loss, backward
       e. Use Megatron's gradient accumulation (or model.no_sync() equivalent for Megatron DDP)
    5. [All ranks] Optimizer step (Megatron DDP syncs gradients)
    6. [Rank 0] Sync weights to vLLM (NCCLWeightTransferEngine, unchanged)
    7. [Rank 0] Sync weights to logprob server (PyNcclCommunicator, background thread trick)
```

Key difference from current: each rank calls the logprob server HTTP endpoint
independently for its own items. No need to broadcast teacher logprobs from
rank 0 — each rank gets exactly what it needs directly.

### Rollout data broadcast

Rank 0 creates rollout results. Use `dist.broadcast_object_list()` to send to all ranks.
The DDP world contains only trainer ranks, so no scoping needed:

```python
if rank == 0:
    envs = [ApiAdapterEnv(...) for item in items]
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda e: e.run(), envs))

    rollout_data = [
        {
            "prompt_text": env.prompt_text,
            "completion_text": env.completion_text,
            "privileged_information_prompt": env.privileged_information_prompt,
            "episode_result": env.episode_result,
            "raw_question": env.raw_question,
            "golden_answer": env.golden_answer,
            "verdict": env.verdict,
            "feedback": env.feedback,
        }
        for env in envs
    ]
else:
    rollout_data = [None] * GRAD_ACCUM_STEPS

dist.broadcast_object_list(rollout_data, src=0)
```

Note: `broadcast_object_list` uses pickle under the hood (gloo backend).
Payload is small (~few KB of strings per item), negligible overhead.
No `group=` argument needed — the trainer's entire torch.distributed world
is the DDP group.

### Teacher logprobs via HTTP

Each trainer rank calls the logprob server HTTP endpoint independently:

```python
# Inside micro_step loop, each rank for its own items:
teacher_log_probs = request_teacher_log_probs_http(
    token_ids=cond_ids + completion_ids,
    prompt_len=len(cond_ids),
    vocab_size=vocab_size,
    device=device,
)
```

This replaces the NCCL-based `request_teacher_log_probs()` from `nccl_comm.py`.
Multiple ranks can call the HTTP endpoint concurrently — the logprob server
handles requests serially (GIL + single GPU), but requests from different
ranks won't conflict since each is a separate HTTP connection.

### Gradient accumulation scaling

Current (line 319):
```python
scaled_loss = loss / GRAD_ACCUM_STEPS
```

With Megatron DDP, each rank processes `M = GRAD_ACCUM_STEPS // num_trainers` micro-steps.
Megatron DDP averages gradients across ranks (divides by N). So:

```
effective_grad = (1/N) * sum_over_ranks( sum_over_M( grad / scale ) )
```

To get `(1/GRAD_ACCUM_STEPS) * sum_over_all(grad)`, we need `scale = M`:

```python
num_trainers = dist.get_world_size()
local_accum_steps = GRAD_ACCUM_STEPS // num_trainers
scaled_loss = loss / local_accum_steps
```

### Gradient accumulation with Megatron DDP

MegatronDDP has `no_sync()` — same API as PyTorch DDP:

```python
for i, data in enumerate(my_slice):
    ctx = ddp_model.no_sync() if i < local_accum_steps - 1 else nullcontext()
    with ctx:
        loss = compute_loss(data)
        (loss / local_accum_steps).backward()
```

Additional MegatronDDP methods available:
- `ddp_model.zero_grad_buffer()` — zero Megatron's internal grad buffers (call at start of accumulation window)
- `ddp_model.start_grad_sync(*unused)` — manually trigger all-reduce
- `ddp_model.finish_grad_sync(force_all_reduce=False)` — wait for all-reduce completion

### Optimizer step

```python
# Current (lines 207-208, 336-339):
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
optimizer.step()

# New:
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
opt_config = OptimizerConfig(optimizer='adam', lr=LEARNING_RATE, bf16=True,
                             clip_grad=MAX_GRAD_NORM, use_distributed_optimizer=True)
optimizer = get_megatron_optimizer(opt_config, model_chunks=[ddp_model])

# Per step (gradient clipping is internal to Megatron optimizer):
ddp_model.zero_grad_buffer()  # instead of optimizer.zero_grad()
# ... accumulation loop ...
optimizer.step()              # handles clip_grad internally
```

### Metrics and wandb logging

Only rank 0 should log to wandb. Other ranks accumulate metrics locally,
then reduce to rank 0 before logging:

```python
if rank == 0:
    wandb.log(log_dict, step=optimizer_step)
```

For aggregated metrics (loss, pass_rate), use `dist.reduce()` to sum across ranks,
then divide by num_trainers on rank 0.

### Success cache

Currently `success_cache` is a dict on rank 0. Since only rank 0 does rollouts,
it stays on rank 0. No change needed.

### vLLM weight sync (lines 242-245, 369-371)

Only rank 0 calls `init_vllm_weight_engine()` and `sync_weights_to_vllm()`.
No change needed — just gate with `if rank == 0`. Pass `model.module`
(unwrapped Megatron model) for HF format conversion.

---

## 5. Changes to `logprob_server.py`

### Current: pure NCCL command loop (lines 45-64)

```python
init_distributed(rank=1, world_size=2)
model = load_model(...)
while True:
    cmd = recv_command(device)
    if cmd == CMD_TEACHER_LOGPROBS: handle_teacher_log_probs(model, device)
    elif cmd == CMD_SYNC_WEIGHTS: broadcast_weights_ema(model, ...)
    elif cmd == CMD_SHUTDOWN: break
```

### New: HTTP server (FastAPI) + standalone PyNcclCommunicator

The logprob server is a fully independent process:
- Initializes its own `torch.distributed` with `world_size=1, rank=0`
  (just for Megatron/AutoBridge model loading)
- Runs a FastAPI HTTP server for logprob requests
- Uses a standalone `PyNcclCommunicator` for weight sync (initialized via HTTP
  handshake with trainer rank 0 — see section 6)

```python
def main(port: int = 8010) -> None:
    # Standalone torch.distributed for Megatron model loading
    init_distributed_standalone()

    model = load_model(HF_MODEL_PATH, wrap_with_ddp=False)
    model.eval()
    device = torch.device("cuda:0")

    # PyNcclCommunicator — initialized later via HTTP handshake
    logprob_nccl_comm: PyNcclCommunicator | None = None

    app = FastAPI()

    @app.post("/logprobs")
    def compute_logprobs(request: LogprobRequest):
        """HTTP endpoint: receives token_ids + prompt_len, returns log_probs as binary."""
        token_ids = torch.tensor(request.token_ids, device=device, dtype=torch.long)
        seq_len = len(request.token_ids)
        completion_len = seq_len - request.prompt_len

        with torch.no_grad():
            input_ids = token_ids.unsqueeze(0)
            pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            logits = model(input_ids=input_ids, position_ids=pos_ids, attention_mask=None)
            logits = logits[0]  # (S, V)

            comp_logits = logits[request.prompt_len - 1 : request.prompt_len + completion_len - 1]
            log_probs = F.log_softmax(comp_logits.float(), dim=-1).bfloat16()

        # Return as binary numpy (bf16 → float16 for numpy compatibility)
        return Response(
            content=log_probs.cpu().to(torch.float16).numpy().tobytes(),
            media_type="application/octet-stream",
        )

    @app.post("/init_weight_sync")
    def init_weight_sync(request: NCCLInitRequest):
        """HTTP handshake: trainer rank 0 sends NCCL init info, we create our communicator."""
        nonlocal logprob_nccl_comm
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import _stateless_init_process_group

        pg = _stateless_init_process_group(
            master_address=request.master_address,
            master_port=request.master_port,
            rank=1,  # logprob server is rank 1
            world_size=request.world_size,
            device=device,
        )
        logprob_nccl_comm = PyNcclCommunicator(group=pg, device=device)
        return {"status": "ok"}

    @app.post("/sync_weights")
    def sync_weights():
        """HTTP trigger for NCCL weight sync. Blocks until all weights received."""
        assert logprob_nccl_comm is not None
        for param in model.parameters():
            incoming = torch.empty_like(param.data)
            logprob_nccl_comm.broadcast(incoming, src=0)
            param.data.lerp_(incoming, EMA_ALPHA)
        return {"status": "ok"}

    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Logprob response format — performance consideration

Returning `log_probs` as JSON list is simple but slow for large vocab (151936 floats per token).
The endpoint returns raw binary instead:

- `log_probs.cpu().to(torch.float16).numpy().tobytes()` — compact binary
- Client decodes: `np.frombuffer(resp.content, dtype=np.float16).reshape(C, V)`

Size: C=500 tokens * V=151936 * 2 bytes (fp16) = ~145 MB per request.
Over localhost HTTP, this takes ~0.5-1s. Acceptable for research.

### Concurrent logprob requests

Multiple trainer ranks can call `/logprobs` concurrently. Since the logprob
server has a single GPU and runs inference under `torch.no_grad()`, requests
are effectively serialized by the GIL + CUDA stream. This is fine — the
logprob computation is the bottleneck regardless.

If throughput becomes an issue, options:
- Batch requests from multiple ranks into a single forward pass
- Use uvicorn with `--workers 1` (default) to keep it single-process

---

## 6. Logprob Weight Sync — PyNcclCommunicator Pattern

The logprob weight sync mirrors the existing vLLM weight sync in `vllm_utils.py:73-117`.
Both use a standalone NCCL communicator initialized via HTTP handshake, with a
background thread to handle the NCCL collective's requirement that both sides
call broadcast simultaneously.

### Why not `dist.broadcast()` with `group=`?

`dist.new_group()` requires both processes to be in the same `torch.distributed`
world. Our logprob server has its own world_size=1, so `new_group()` is not
possible. Instead, we use `PyNcclCommunicator` — the same mechanism vLLM uses
for its NCCL weight transfer engine.

### Init flow (at startup, trainer rank 0 only)

Mirrors `init_vllm_weight_engine()` at `vllm_utils.py:73-117`:

```
Trainer rank 0:                              Logprob server:
                                             (HTTP server running, waiting for /init_weight_sync)

1. master_address = get_ip()
   master_port = get_open_port()

2. t = Thread(POST /init_weight_sync         HTTP handler:
     {master_address, master_port,    ------>   comm = PyNcclCommunicator(
      world_size=2})                               rank=1, world_size=2,
   t.start()                                       master_addr, master_port)
                                                 return 200
3. comm = PyNcclCommunicator(
     rank=0, world_size=2,
     master_addr, master_port)

4. t.join() <-------------------------------

Both now have a shared standalone NCCL communicator.
```

The background thread trick is necessary because `PyNcclCommunicator.__init__`
blocks until all ranks connect — both sides must init simultaneously.

```python
# Trainer rank 0 — init logprob NCCL comm
def init_logprob_weight_engine(device: torch.device):
    """Create standalone NCCL communicator for logprob weight sync.
    
    Uses the same internal mechanism as NCCLWeightTransferEngine.trainer_init():
    _stateless_init_process_group() creates a StatelessProcessGroup,
    then PyNcclCommunicator wraps it.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import _stateless_init_process_group
    from vllm.utils.network_utils import get_ip, get_open_port

    master_address = get_ip()
    master_port = get_open_port()

    def _init_server_side():
        requests.post(
            f"{LOGPROB_BASE_URL}/init_weight_sync",
            json={
                "master_address": master_address,
                "master_port": master_port,
                "world_size": 2,
            },
            timeout=60,
        ).raise_for_status()

    t = threading.Thread(target=_init_server_side)
    t.start()

    # Trainer is rank 0 in this 2-process NCCL group
    pg = _stateless_init_process_group(
        master_address=master_address,
        master_port=master_port,
        rank=0,
        world_size=2,
        device=device,
    )
    comm = PyNcclCommunicator(group=pg, device=device)
    t.join()
    return comm
```

### Weight sync flow (every optimizer step, trainer rank 0 only)

Mirrors `sync_weights_to_vllm()` at `vllm_utils.py:120-163`:

```
Trainer rank 0:                         Logprob server:

t = Thread(POST /sync_weights) -------> HTTP handler enters NCCL receive loop:
t.start()                               for param in model.parameters():
                                           incoming = empty_like(param.data)
for param in model.module.parameters():    comm.broadcast(incoming, src=0)
  comm.broadcast(param.data, src=0)        param.data.lerp_(incoming, alpha)
                                         return HTTP 200
t.join() <------------------------------
```

Both sides iterate `model.parameters()` in the same order — both are Megatron
models loaded via the same AutoBridge path. Raw parameter broadcast, no
HF format conversion needed (unlike vLLM sync which converts Megatron→HF).

```python
# Trainer rank 0 — sync weights to logprob server
def sync_weights_to_logprob_server(
    model: torch.nn.Module,
    logprob_comm: PyNcclCommunicator,
) -> None:
    """Push trainer weights to logprob server via standalone NCCL."""

    def _trigger_recv():
        requests.post(f"{LOGPROB_BASE_URL}/sync_weights", timeout=300).raise_for_status()

    t = threading.Thread(target=_trigger_recv)
    t.start()

    # model.module to unwrap Megatron DDP
    raw_model = model.module if hasattr(model, "module") else model
    for param in raw_model.parameters():
        logprob_comm.broadcast(param.data, src=0)

    t.join()
```

### Key difference from vLLM sync

- **vLLM sync** uses `NCCLWeightTransferEngine` + `export_hf_weights_iter()` to convert
  Megatron→HF parameter names and shapes before sending. vLLM loads HF-format weights.
- **Logprob sync** uses `PyNcclCommunicator` + raw `model.parameters()` iteration.
  Both sides are Megatron models with identical parameter layout. No conversion needed.

---

## 7. Changes to `nccl_comm.py`

### Functions to REMOVE (replaced by HTTP + PyNcclCommunicator):
- `send_command()` — no more NCCL command protocol
- `recv_command()` — no more NCCL command protocol
- `request_teacher_log_probs()` — replaced by HTTP logprob request
- `handle_teacher_log_probs()` — moved into logprob server HTTP endpoint
- `CMD_TEACHER_LOGPROBS`, `CMD_SYNC_WEIGHTS`, `CMD_SHUTDOWN` constants

### Functions to KEEP (modified):
- `broadcast_weights_ema()` — the EMA blending logic stays, but now uses
  `PyNcclCommunicator.broadcast()` instead of `dist.broadcast()`:

```python
def broadcast_weights_ema(
    model: torch.nn.Module,
    comm: PyNcclCommunicator,
    alpha: float = 0.01,
    src: int = 0,
) -> None:
    """EMA weight sync via standalone NCCL communicator.

    On src rank: broadcasts own weights.
    On dst rank: receives and EMA-blends.
    """
    rank = comm.rank  # not dist.get_rank() — this is standalone NCCL
    for param in model.parameters():
        if rank == src:
            comm.broadcast(param.data, src=src)
        else:
            incoming = torch.empty_like(param.data)
            comm.broadcast(incoming, src=src)
            param.data.lerp_(incoming, alpha)
```

In practice, the trainer-side and server-side logic is split across
`sync_weights_to_logprob_server()` (trainer) and the `/sync_weights`
HTTP endpoint (server). The `broadcast_weights_ema()` function may be
inlined into those two places rather than kept as a shared function,
since the two sides can no longer share code (different processes,
different entry points).

---

## 8. Changes to `config.py`

### Current (line 31-32)

```python
NCCL_MASTER_PORT = int(os.environ.get("NCCL_MASTER_PORT", "29500"))
```

### New

```python
# Logprob server
LOGPROB_PORT = int(os.environ.get("LOGPROB_PORT", "8010"))
LOGPROB_BASE_URL = f"http://localhost:{LOGPROB_PORT}"

# NCCL_MASTER_PORT is no longer needed — torchrun manages its own master port
# for the trainer DDP group, and the logprob server uses a random free port
# for its standalone world_size=1 init.
# The PyNcclCommunicator for logprob weight sync uses a dynamically allocated port.
```

Keep `GPU_LOGPROB_SERVER` (line 11) — it's used for `CUDA_VISIBLE_DEVICES`
assignment in launch scripts.

---

## 9. New HTTP logprob client (in trainer)

Replace `request_teacher_log_probs()` from `nccl_comm.py`:

```python
def request_teacher_log_probs_http(
    token_ids: list[int],
    prompt_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Request teacher log-probs via HTTP from logprob server.

    Any trainer rank can call this independently — no rank 0 gating needed.
    """
    completion_len = len(token_ids) - prompt_len
    resp = requests.post(
        f"{LOGPROB_BASE_URL}/logprobs",
        json={
            "token_ids": token_ids,
            "prompt_len": prompt_len,
        },
        timeout=120,
    )
    resp.raise_for_status()

    # Decode binary response (float16 numpy bytes)
    log_probs_np = np.frombuffer(resp.content, dtype=np.float16)
    log_probs_np = log_probs_np.reshape(completion_len, vocab_size)
    return torch.from_numpy(log_probs_np).to(device=device, dtype=torch.bfloat16)
```

**Optimization**: For requests, JSON is fine (small payload: list of ints).
For responses, binary is required (145 MB per request at C=500, V=151936).

Could also use msgpack for the request body to avoid JSON overhead on the
token_ids list, but JSON is simpler and the request payload is small.

---

## 10. Megatron Parallel State Interaction

### Trainers: world_size=N, DP=N

With TP=1, PP=1, Megatron sets DP=world_size=N. Each trainer rank loads the
full model independently. Manual MegatronDDP wrapping handles gradient
all-reduce across these N ranks. The distributed optimizer shards optimizer
state across DP ranks.

### Logprob server: world_size=1, DP=1

The logprob server's standalone world_size=1 means Megatron sees DP=1, TP=1,
PP=1. Model loading works identically to single-rank mode. No cross-rank
communication during model loading.

### `save_hf_checkpoint()` — no barrier risk

Current code (model_utils.py:112-149) manually exports weights via
`export_hf_weights_iter` + safetensors to avoid bridge's `save_hf_pretrained()`
which uses barriers. This is already safe — no barriers in our save path.

**Only rank 0 should save checkpoints.** Gate with `if rank == 0:`.

---

## 11. Per-Step Data Flow (Detailed)

```
Step N:

1. [Rank 0] Rollout:
   - Create GRAD_ACCUM_STEPS ApiAdapterEnv instances
   - Run all via ThreadPoolExecutor(max_workers=16)
   - Collect rollout_data dicts (prompt_text, completion_text, etc.)

2. [Rank 0 → All] Broadcast rollout data:
   - dist.broadcast_object_list(rollout_data, src=0)
   - All trainer ranks now have the full rollout_data list

3. [Each rank] Slice data:
   - M = GRAD_ACCUM_STEPS // num_trainers
   - my_slice = rollout_data[rank * M : (rank + 1) * M]

4. [Each rank] Micro-step loop (M iterations):
   for i, data in enumerate(my_slice):
     a. completion_ids = tokenizer.encode(data["completion_text"])
     b. cond_ids = tokenizer.encode(data["privileged_information_prompt"])
     c. teacher_log_probs = request_teacher_log_probs_http(
            token_ids=cond_ids + completion_ids,
            prompt_len=len(cond_ids),
            vocab_size=vocab_size,
            device=device,
        )  # HTTP call — each rank independently
     d. student_logits = forward_student(model, tokenizer, ...)  # Megatron DDP model
     e. loss, metrics = compute_kl(student_logits, teacher_log_probs, ...)
     f. (loss / local_accum_steps).backward()
        # Megatron DDP: skip allreduce on all but last micro-step

5. [All ranks] Optimizer step:
   - Megatron DDP syncs gradients (allreduce on last backward)
   - optimizer.step()  # Megatron distributed AdamW
   - optimizer.zero_grad()

6. [Rank 0 only] Sync weights to vLLM:
   - sync_weights_to_vllm(model.module, device, vllm_group)
   - Uses NCCLWeightTransferEngine, Megatron→HF conversion (unchanged)

7. [Rank 0 only] Sync weights to logprob server:
   - sync_weights_to_logprob_server(model, logprob_comm)
   - Uses PyNcclCommunicator, raw parameter broadcast (no format conversion)
   - Background thread trick for NCCL collective
```

---

## 12. Launch Script Changes

### Current: 3 processes, manual everything

```bash
CUDA_VISIBLE_DEVICES=0 python start_vllm_patched.py ... &
CUDA_VISIBLE_DEVICES=2 python -m megatron_trainer.logprob_server &
CUDA_VISIBLE_DEVICES=1 python -m megatron_trainer.trainer
```

### New: 3 independent processes

```bash
# Terminal 1: vLLM (unchanged)
CUDA_VISIBLE_DEVICES=0 python start_vllm_patched.py --model Qwen/Qwen3-8B --port 8004

# Terminal 2: logprob server (standalone, own torch.distributed world_size=1)
CUDA_VISIBLE_DEVICES=3 python -m megatron_trainer.logprob_server --port 8010

# Terminal 3: trainers (torchrun, Megatron DDP world_size=2)
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 \
    -m megatron_trainer.trainer --logprob-url http://localhost:8010
```

### Parallel runs (different GPU set)

```bash
CUDA_VISIBLE_DEVICES=4 python start_vllm_patched.py --port 8005
CUDA_VISIBLE_DEVICES=7 python -m megatron_trainer.logprob_server --port 8011
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29501 \
    -m megatron_trainer.trainer --logprob-url http://localhost:8011 \
    --vllm-url http://localhost:8005
```

Note: `torchrun --master_port=29501` avoids port collision with the first run's
default master port (29500). The PyNcclCommunicator ports for vLLM and logprob
weight sync are dynamically allocated, so no manual port management needed.

---

## 13. Implementation Order

1. **`model_utils.py`**: Split `init_distributed()` into trainer/standalone variants,
   add `wrap_with_ddp` param to `load_model()`, handle local_rank for DEVICE
2. **`logprob_server.py`**: Rewrite as HTTP server (FastAPI) + PyNcclCommunicator
   weight sync endpoint, standalone `torch.distributed` init
3. **`nccl_comm.py`**: Remove NCCL command protocol functions, update
   `broadcast_weights_ema()` for PyNcclCommunicator (or inline into caller code)
4. **`trainer.py`**: Use torchrun env vars, manual MegatronDDP wrapping,
   Megatron optimizer, rollout broadcast, gradient accumulation fix, HTTP logprobs,
   PyNcclCommunicator init + weight sync for logprob server
5. **`config.py`**: Add LOGPROB_PORT, LOGPROB_BASE_URL, remove NCCL_MASTER_PORT
6. **Launch scripts**: Update for three independent processes
7. **Test**: Smoke test with N=2 trainers on 4 GPUs (1 vLLM + 2 trainers + 1 logprob)

---

## 14. Dependency Changes

New Python dependencies (already available or pip-installable in NeMo container):
- `fastapi` + `uvicorn` — for logprob HTTP server
- `numpy` — for binary serialization of log_probs (already available)
- `pynccl` / `vllm.distributed` — for PyNcclCommunicator (available via vLLM install
  in the trainer venv; logprob server may need its own install)

Removed:
- `bitsandbytes` — no longer needed (replaced by Megatron optimizer)

---

## 15. Backward Compatibility

Single-trainer mode: `torchrun --nproc_per_node=1` with one trainer rank.
Megatron DDP with world_size=1 is effectively no-op DDP. All the rank 0
gating (`if rank == 0`) passes through. No separate codepath needed.

---

## Appendix: API Exploration Notes

Findings from interactive exploration on the NeMo 26.06 container (GPU 7, rh-h100-01).

### MegatronDDP manual wrapping — confirmed working

```python
# After load_model() with wrap_with_ddp=False:
from megatron.core.distributed import DistributedDataParallel as MegatronDDP, DistributedDataParallelConfig
ddp_model = MegatronDDP(config=model.config, ddp_config=DistributedDataParallelConfig(), module=model)
# type: DistributedDataParallel
# .module: Float16Module
# .no_sync(): context manager ✓
# .start_grad_sync(*unused) ✓
# .finish_grad_sync(force_all_reduce=False) ✓
# .zero_grad_buffer() ✓
# Forward: ddp_model(input_ids=..., position_ids=..., attention_mask=None) → (B, S, V) ✓
```

### `provide_distributed_model(wrap_with_ddp=True)` — crashed in testing

Crashed with `ValueError: output_layer_init_method is None`. Root cause: test script
was missing `provider.finalize()` which sets init methods. Likely works with `finalize()`,
but manual wrapping is confirmed and preferred (same `load_model()` for trainer + logprob server).

### Megatron optimizer — confirmed API

```python
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
config = OptimizerConfig(optimizer='adam', lr=2e-6, bf16=True, clip_grad=1.0,
                         use_distributed_optimizer=True, weight_decay=0.01)
optimizer = get_megatron_optimizer(config, model_chunks=[ddp_model])
# Gradient clipping is internal (clip_grad field in OptimizerConfig)
# use_distributed_optimizer=True shards optimizer state across DP ranks
```

### PyNcclCommunicator — confirmed API

```python
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
# Constructor: PyNcclCommunicator(group: ProcessGroup | StatelessProcessGroup, device, library_path=None)
# Does NOT take rank/world_size directly — needs a process group object
# Use _stateless_init_process_group(master_address, master_port, rank, world_size, device) to create group
# Methods: broadcast(tensor, src, stream=None)
```

### DistributedDataParallelConfig defaults

Key defaults (all False/None unless noted):
- `overlap_grad_reduce = False`
- `overlap_param_gather = False`
- `use_distributed_optimizer = False`
- `bucket_size = None`
- `average_in_collective = False`
