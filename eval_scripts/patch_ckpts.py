"""Create eval-safe copies of checkpoints:
- symlink all files
- strip broken `extra_special_tokens` field from tokenizer_config.json
  (training-hub save artifact incompatible with AutoTokenizer loading)
- for osft checkpoints: strip `model.rotary_emb.inv_freq` and
  `model.rotary_emb.original_inv_freq` non-persistent buffer keys from the
  safetensors shard that contains them + update the index.json. These are
  RoPE buffers computed on-the-fly from rope_theta; vLLM's Qwen3 loader has
  no corresponding parameter to load them into, causing
  KeyError: 'rotary_emb.original_inv_freq' during weight loading.
"""
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

BASE = os.environ.get("CKPT_BASE", "/mnt/nvme7n1/rawhad/sdft_knowledge_ingestion_experiment/models")
OUT = os.environ.get("PATCHED_OUT", "/home/rohan/1_Projects/sdft_knowledge_ingestion_experiment/eval/patched_ckpts")
# Rewritten safetensors shards (multi-GB) are written here, not under OUT --
# OUT commonly lives on a small root disk (symlinks are cheap, full-shard
# rewrites are not). Point this at a spacious nvme mount.
SCRATCH = os.environ.get("PATCH_SCRATCH", "/mnt/nvme5n1/rohan_patched_ckpts_scratch")

BAD_ROTARY_KEYS = {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}

steps = [int(s) for s in os.environ.get("STEPS", "400,800,1200,1600,2000,2400,2800,3200,3600,4000").split(",")]
kinds = os.environ.get("KINDS", "sft,osft").split(",")

targets = []
for step in steps:
    if "sft" in kinds:
        targets.append((f"{BASE}/sft/ckpts/hf_format/samples_{step}", f"{OUT}/sft/samples_{step}", False))
    if "osft" in kinds:
        targets.append((f"{BASE}/osft/ckpts/hf_format/samples_{step}.0", f"{OUT}/osft/samples_{step}", True))

for src, dst, is_osft in targets:
    if not os.path.isdir(src):
        print(f"skip {src} (not found)")
        continue
    os.makedirs(dst, exist_ok=True)

    index_path = os.path.join(src, "model.safetensors.index.json")
    bad_shard = None
    if is_osft:
        with open(index_path) as f:
            index = json.load(f)
        wm = index["weight_map"]
        bad_keys_present = [k for k in BAD_ROTARY_KEYS if k in wm]
        if bad_keys_present:
            shards = {wm[k] for k in bad_keys_present}
            assert len(shards) == 1, f"expected bad keys in single shard, got {shards}"
            bad_shard = shards.pop()

    for fname in os.listdir(src):
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname)
        if os.path.lexists(dst_path):
            os.remove(dst_path)

        if fname == "tokenizer_config.json":
            with open(src_path) as f:
                cfg = json.load(f)
            cfg.pop("extra_special_tokens", None)
            with open(dst_path, "w") as f:
                json.dump(cfg, f, indent=2)

        elif fname == "model.safetensors.index.json" and bad_shard is not None:
            new_index = json.loads(json.dumps(index))  # deep copy
            for k in BAD_ROTARY_KEYS:
                new_index["weight_map"].pop(k, None)
            with open(dst_path, "w") as f:
                json.dump(new_index, f, indent=2)

        elif fname == bad_shard:
            os.makedirs(SCRATCH, exist_ok=True)
            scratch_path = os.path.join(SCRATCH, f"{os.path.basename(dst)}__{fname}")
            tensors = {}
            with safe_open(src_path, framework="pt") as f:
                meta = f.metadata()
                for k in f.keys():
                    if k in BAD_ROTARY_KEYS:
                        continue
                    tensors[k] = f.get_tensor(k)
            save_file(tensors, scratch_path, metadata=meta)
            os.symlink(scratch_path, dst_path)

        else:
            os.symlink(src_path, dst_path)

    print(f"patched {dst}" + (f" (stripped rotary keys from {bad_shard})" if bad_shard else ""))

print("DONE")
