# Evaluating HF Checkpoints on MaaS SDFT Test Set

How to run `eval_maas_sdft.py` (question-answering pass-rate eval, judged by
Claude via Vertex) against HF-format checkpoints, including on nodes with
older NVIDIA drivers that can't run the newest vLLM/torch.

## What it measures

For each of 100 test questions, generates two answers (no-context, +context)
via vLLM, judges each against a golden answer with Claude Sonnet (3x majority
vote), reports pass rate for both conditions.

Eval script (canonical, lives outside this repo):
`/mnt/nvme0n1/rawhad/self_distillation/aligning_lm_from_user_interaction/scripts/eval_maas_sdft.py`
on `rh-h100-01`. Copy it to wherever you're running eval.

```
python eval_maas_sdft.py \
  --model <hf_checkpoint_dir_or_hf_id> \
  --test_jsonl <path_to_test_maas_sdft.jsonl> \
  --output_dir <output_dir>
```

## Prerequisites

0. **Two known bugs/gotchas in the eval script** (found 2026-08-14 on
   rh-h100-12):
   - `--default-mode` is declared `action="store_true", default=20` — the
     `default=20` makes it **truthy**, so without the flag the script silently
     judges only the no-context answers (results lack `with_context_pass`).
     Fix: `default=20` → nothing (plain `store_true`).
   - vLLM spawns need `ninja` on PATH (same gotcha as training) — install
     `ninja` into the eval venv and prepend its `bin/` to PATH.
1. **Test data**: `test_maas_sdft.jsonl` — canonical copy at
   `/home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/test_maas_sdft.jsonl`
   on `rh-h100-01`. Verify with `md5sum` before assuming a copy elsewhere is current.
2. **AnthropicVertex judge auth** (on whichever node runs eval):
   - `gcloud auth application-default login` (interactive, one-time per node/user)
   - `export CLOUD_ML_REGION=us-east5`
   - `export ANTHROPIC_VERTEX_PROJECT_ID=itpc-gcp-ai-eng-claude`
   - Verify: `python -c "from anthropic import AnthropicVertex; print(AnthropicVertex().messages.create(model='claude-sonnet-4-6@default', messages=[{'role':'user','content':'say OK'}], max_tokens=10).content)"`
3. **vLLM + anthropic + transformers venv** — see driver compatibility below
   before picking versions.

## Driver/CUDA compatibility (check this first)

`nvidia-smi` on the target node shows the max CUDA version the driver
supports. vLLM's pinned torch version needs a wheel built for that CUDA (or
older). PyPI's default torch wheel targets the *newest* CUDA — it will
install fine but fail at runtime with:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12040) ...
```

| Driver / max CUDA | vLLM version | torch pin | Notes |
|---|---|---|---|
| Newer (e.g. driver 610.x / CUDA 13.x, `rh-h100-01`) | `vllm==0.24.0` (or latest) | `torch==2.11.0` | Default PyPI install works. |
| Older (e.g. driver 550.90.07 / CUDA 12.4, `rh-h100-05`) | `vllm==0.8.5.post1` | `torch==2.6.0+cu124` | Need explicit cu124 index; see below. |

To find a vLLM version whose torch pin has a wheel for your CUDA ceiling:
```bash
# check torch's requires-dist for a candidate vllm version
curl -s "https://pypi.org/pypi/vllm/<version>/json" | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print([r for r in d['info']['requires_dist'] if r.startswith('torch')])"

# check if that torch version has a wheel for your cuXXX
curl -s https://download.pytorch.org/whl/cuXXX/torch/ | grep -oE "torch-<version>\+cuXXX-cp312-cp312-linux_x86_64\.whl"
```

Install order matters — install torch from the CUDA-specific index *before*
vLLM, so vLLM doesn't pull the default (wrong-CUDA) torch build:

```bash
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu124
VIRTUAL_ENV=.venv uv pip install "vllm==0.8.5.post1" "transformers==4.51.3" "anthropic[vertex]"
```

`transformers` must be pinned to whatever vLLM's release-era expects
(check `requires-dist` for vllm — `transformers>=4.51.1` doesn't mean any
4.x/5.x works; newer transformers changed tokenizer internals and breaks
older vLLM's tokenizer loading, see gotcha below). `4.51.3` works with
`vllm==0.8.5.post1`.

**Cleanup gotcha**: if you `--reinstall` vllm to swap versions, orphaned
packages from the old install (e.g. `flashinfer`, `tvm_ffi`,
`torch_c_dlpack_ext`) can be left on disk with mismatched torch ABI, causing
`undefined symbol` errors at import time. If vLLM fails on an unrelated
import (not torch/CUDA), check `site-packages/` for such orphans and `rm -rf`
them — they're optional accelerators, not required deps.

## Checkpoint format gotchas

Checkpoints saved by `training-hub` may not load cleanly in an older
vLLM/transformers pinned for driver compatibility. Two known issues (both
fixed by `eval_scripts/patch_ckpts.py` — creates symlinked copies, doesn't
touch originals):

1. **`tokenizer_config.json` has a malformed `extra_special_tokens` field**
   (a flat list; transformers expects a `{name: token}` dict and calls
   `.keys()` on it). Fails with `AttributeError: 'list' object has no
   attribute 'keys'` regardless of transformers version. The real special
   tokens are already embedded correctly in `tokenizer.json`, so the field
   is safe to drop entirely.

2. **OSFT checkpoints' safetensors have extra `model.rotary_emb.inv_freq` /
   `model.rotary_emb.original_inv_freq` keys** with no corresponding vLLM
   model parameter (RoPE buffers are computed on-the-fly, not loaded from
   disk). Fails with `KeyError: 'rotary_emb.original_inv_freq'` during
   weight loading. Fix: strip those two keys from the affected safetensors
   shard + update `model.safetensors.index.json` accordingly. SFT
   checkpoints don't have this issue.

Run the patcher before eval:
```bash
CKPT_BASE=/mnt/nvme7n1/rawhad/sdft_knowledge_ingestion_experiment/models \
PATCHED_OUT=./patched_ckpts \
STEPS=400,800,1200,1600,2000,2400,2800,3200,3600,4000 \
python eval_scripts/patch_ckpts.py
```

If you hit a *new* checkpoint-loading error, diff `config.json` and the
`weight_map` keys in `model.safetensors.index.json` against a known-good
checkpoint (e.g. the base HF model on the hub) to spot the anomaly — that's
how both issues above were found.

## Running the full sweep

`eval_scripts/run_evals.sh` loops over `STEPS x KINDS`, skips any
checkpoint whose `eval_results/{kind}/samples_{step}/eval_results.jsonl`
already exists (safe to re-run/resume), and does **not** use `set -e` — a
single checkpoint failure is logged (`FAILED: {kind} samples_{step}`) and
the loop continues.

```bash
cd <eval_dir>   # must contain eval_maas_sdft.py, patched_ckpts/, eval_results/
STEPS="400 800 1200 1600 2000 2400 2800 3200 3600 4000" \
BASE=./patched_ckpts \
TEST=/path/to/test_maas_sdft.jsonl \
VENV=./.venv \
tmux new-session -d -s sdft_eval "bash eval_scripts/run_evals.sh > run_evals.log 2>&1"
```

Run in **tmux** — a 20-checkpoint sweep (vLLM load + 200 generations + 600
judge calls per checkpoint) takes hours; don't block an interactive shell.

Poll progress:
```bash
tmux capture-pane -t sdft_eval -p -S -3000 | grep -n "^===\|no_context:\|Results saved\|FAILED\|ALL EVALS DONE"
find eval_results -name eval_results.jsonl   # should match STEPS x KINDS count when done
```

## Reading results

```bash
python3 -c "
import json
recs = [json.loads(l) for l in open('eval_results/<kind>/samples_<step>/eval_results.jsonl')]
n = len(recs)
noctx = sum(1 for r in recs if r['no_context_pass'])
wctx = sum(1 for r in recs if r['with_context_pass'])
print(f'no_context={noctx}/{n} ({100*noctx/n:.1f}%) with_context={wctx}/{n} ({100*wctx/n:.1f}%)')
"
```

`no_context` pass rate reflects internalized knowledge (no retrieval given);
`with_context` reflects reading-comprehension-style QA given the source doc.
