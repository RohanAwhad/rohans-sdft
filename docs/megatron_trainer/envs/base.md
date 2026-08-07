# Rollout Environments (`env/`)

## Role

The rollout stage turns one training example into the `(prompt_text,
completion_text, privileged_information_prompt)` triple the trainer trains on.
It is abstracted behind `BaseEnv` (`env/base.py`) so `ENV_TYPE` selects the
rollout strategy:

| `ENV_TYPE` | Env | Spec |
|---|---|---|
| `rag` | `RagEnv` (`env/rag_env.py`) | `ragenv.md` |
| `api_adapter` | `ApiAdapterEnv` (`env/api_adapter_env.py`) | `api_adapter_env.md` |

## `BaseEnv` contract (`env/base.py:6`)

`BaseEnv(ABC)` — one example's rollout lifecycle. Subclasses must implement
`run()`, which populates:

| Field | Type | Meaning |
|---|---|---|
| `completion_text` | `str \| None` | model-generated completion |
| `privileged_information_prompt` | `str \| None` | teacher prompt with privileged info |
| `prompt_text` | `str` | student generation prompt |
| `vllm_base_url` | `str` | which vLLM instance to use |
| `episode_result` | `bool \| None` | rollout success (adapter verdicts) |

## Trainer ↔ env contract

The trainer (`trainer.py:222-285`) is the only caller:

1. Build one env per example (`GRAD_ACCUM_STEPS` of them) — `RagEnv` when
   `ENV_TYPE=rag`, `ApiAdapterEnv` when `ENV_TYPE=api_adapter`.
2. Run them **concurrently**: `ThreadPoolExecutor(max_workers=min(32, len(envs)))`.
3. Read outputs into the broadcast payload `{prompt_text, completion_text,
   privileged_information_prompt}` (`trainer.py:267-274`).
4. Compute `pass_rate` — `episode_result` for api_adapter, `reflector_result`
   verdicts for rag (`trainer.py:276-282`).
