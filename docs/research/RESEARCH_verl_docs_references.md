Yes. These four papers are much more interesting **together** than separately, because they expose two different layers of modern reasoning RL:

```text
                         Reasoning RL
                              │
             ┌────────────────┴────────────────┐
             │                                 │
      Algorithm / recipe                 Systems / runtime
             │                                 │
         Magistral                 ┌───────────┼────────────┐
                                   │           │            │
                                 AReaL      StreamRL     AsyncFlow
```

**Magistral asks:** how do we actually make large-scale RLVR produce a strong reasoning model?

The other three ask a different question:

> Once rollouts become 10K–30K tokens long, how do we stop half the cluster from sitting idle while doing RL?

And their answers differ substantially.

The most important takeaway from reading all four is that **“asynchronous RL” is not one design**. Magistral, AReaL, StreamRL, and AsyncFlow make very different choices about *how stale data may be, whether weights can change during a trajectory, what happens to the KV cache, and how aggressively generation and training are decoupled.*

---

# 1. The problem all four papers are running into

A conventional reasoning-RL loop looks roughly like:

```text
policy θ₀
   │
   ▼
generate 1024 / 4096 rollouts
   │
   │ wait for the longest rollout
   ▼
verify rewards
   │
   ▼
GRPO / PPO update
   │
   ▼
policy θ₁
   │
   ▼
generate again
```

The killer is:

```text
rollout lengths

sample 1   █████
sample 2   ████████████████
sample 3   ███
sample 4   ███████████████████████████
sample 5   ███████
                                    ↑
                         everyone waits for this
```

Reasoning models make this unusually bad because completion lengths vary enormously and tend to **grow during RL**.

AReaL explicitly observes that synchronous systems wait for the longest completion and therefore underutilize inference GPUs. Its motivation is to make rollout workers continuously generate while trainers independently consume completed trajectories. ([arXiv][1])

StreamRL calls the same phenomenon two different kinds of bubbles:

```text
pipeline bubble
    generation done
         ↓
    trainer waiting / generator waiting

skewness bubble
    most rollouts done
         ↓
    everybody waits for a handful of 20K-token rollouts
```

Its entire architecture is designed around eliminating both. ([arXiv][2])

---

# 2. Magistral: the most interesting paper algorithmically

Magistral is somewhat different from the other three.

It's primarily a **reasoning-model training paper**, but its asynchronous RL infrastructure turns out to be extremely relevant to the other papers.

Mistral trained:

* **Magistral Medium**: Mistral Medium 3 → pure RLVR, with **no reasoning-trace cold start**.
* **Magistral Small 24B**: Mistral Small 3 → SFT on Magistral Medium traces → RLVR. ([arXiv][3])

And they got a very large pure-RL improvement on Medium: AIME-24 pass@1 went from **26.8 → 73.6**, while LiveCodeBench v5 went from **29.1 → 59.4**. ([arXiv][3])

But the interesting part is *how*.

## Magistral's GRPO is not vanilla GRPO

They begin with GRPO, but make several modifications.

### 1. Remove the KL penalty entirely

Standard GRPO often contains:

[
-\beta D_{KL}(\pi_\theta || \pi_{ref})
]

Magistral simply removes it.

Their argument is pragmatic: the policy diverges anyway, and keeping another reference model around consumes substantial compute. ([arXiv][3])

So this:

```text
actor
reference model
reward/verifier
```

becomes closer to:

```text
actor
reward/verifier
```

That's a meaningful savings at this scale.

---

## 2. Loss normalization is over the entire group

Instead of effectively giving every sequence equal weight irrespective of length, they sum token losses over the group and divide by the **total number of generated tokens**.

This is meant to prevent length-related weighting artifacts between samples. ([arXiv][3])

---

## 3. They change advantage normalization

The first group-relative advantage is simply:

[
A_i=r_i-\mu_{\text{group}}
]

rather than:

[
\frac{r_i-\mu}{\sigma_{\text{group}}}
]

They subsequently normalize advantages over the minibatch. ([arXiv][3])

That distinction is easy to miss.

It means the *relative difficulty of questions* can continue to matter more than it would under per-group standard-deviation normalization.

---

## 4. Asymmetric clipping / Clip-Higher

This is especially interesting.

Standard PPO:

[
r_t \in [1-\epsilon,;1+\epsilon]
]

Magistral uses:

[
r_t \in [1-\epsilon_{\text{low}},;1+\epsilon_{\text{high}}]
]

with a larger upper clipping threshold.

Why?

Suppose a useful reasoning token currently has probability:

[
p=0.0005
]

A symmetric PPO trust region makes it difficult for that rare action suddenly to receive much more probability, even if it occurs in a successful trajectory.

The higher upper clip gives rare actions greater room to grow:

```text
rare reasoning strategy
      ↓
successful rollout
      ↓
allow probability to rise aggressively
      ↓
preserve exploration
```

They tuned (\epsilon_{\text{high}}) around **0.26–0.28** during Medium training to keep group entropy stable. ([arXiv][3])

This is basically an explicit response to **entropy collapse in reasoning RL**.

---

# 3. Their data curation may be as important as the RL algorithm

Magistral starts with roughly:

```text
699,000 math problems
       ↓
format/verifiability filtering
       ↓
501,000
       ↓
difficulty filtering
       ↓
38,000
```

Yes: **38K**, not hundreds of thousands. ([arXiv][3])

That tells us something important about RLVR.

More data is not automatically better.

They specifically want the region where:

```text
P(success) ≈ neither 0 nor 1
```

because:

* always-correct questions produce little useful learning signal;
* impossible questions produce little useful learning signal;
* partially solved questions produce useful group variance.

Their procedure is clever.

First:

```text
Mistral Large 2
    │
    ├─ sample 16 answers/problem
    │
    └─ remove trivial + impossible questions
```

Then they train a stronger 24B reasoning model.

Then:

```text
RL-trained 24B grader
    │
    ├─ sample another 16/problem
    │
    ├─ re-estimate difficulty
    │
    ├─ remove easy
    │
    ├─ remove still-impossible
    │
    └─ detect likely bad ground truths
```

If most model generations converge on the *same answer* but disagree with the dataset answer, they treat that as evidence that the dataset ground truth itself may be wrong. ([arXiv][3])

That is an excellent training-data trick.

---

# 4. Magistral's asynchronous architecture

Now we reach the connection to the other three papers.

Magistral has separate:

```text
┌──────────────┐
│ Generators   │
│              │
│ inference    │
└──────┬───────┘
       │ completions
       ▼
┌──────────────┐
│ Verifiers    │
└──────┬───────┘
       │ reward
       ▼
┌──────────────┐
│ Trainers     │
│              │
│ GRPO update  │
└──────┬───────┘
       │ weights
       └──────────────► Generators
```

But generation **never stops**.

Once the trainer produces (\theta_{i+1}), new weights are transmitted GPU→GPU using NCCL while rollouts are ongoing. Mistral reports the update taking under about five seconds even at large world sizes. ([arXiv][3])

And here comes the wild part.

### They replace weights in the middle of generation.

A trajectory could effectively look like:

```text
token 1 ───────────── token 2000 ───── token 4000
   θᵢ                      θᵢ₊₁              θᵢ₊₂
```

But they **do not recompute the KV cache**.

So after the switch:

```text
new model weights = θᵢ₊₁
KV cache           = hidden states produced using θᵢ
```

The state is technically inconsistent.

They say empirically this works and suggest PPO's off-policy correction may make it tolerable. ([arXiv][3])

This is one of the most fascinating engineering choices in all four papers.

---

# 5. Magistral discovers a very important async scaling law

They define three quantities:

[
n_{\text{async}}
]

= number of sequences simultaneously generating.

[
n_{\text{batch}}
]

= completions collected before a generator weight update.

[
n_{\text{minibatch}}
]

= sequences per optimizer update.

Then they observe something extremely useful.

If:

[
\frac{n_{\text{async}}}{n_{\text{batch}}}
]

becomes large, a rollout can span many policy versions.

For example:

```text
n_async = 8192
n_batch = 1024

≈ 8 policy updates can happen
while the pool of generations drains.
```

That makes data increasingly off-policy.

Their final recommendation was roughly:

[
\boxed{\frac{n_{\text{async}}}{n_{\text{batch}}}\le2}
]

and

[
\boxed{n_{\text{batch}}=n_{\text{minibatch}}}
]

for the final training setup. ([arXiv][3])

That is a very practical result.

It says:

> **You cannot increase rollout concurrency independently of optimizer batch size.**

More inference workers can actually change your RL semantics.

---

# 6. One other Magistral result I would take seriously

During training, they deliberately increased the allowed reasoning length:

```text
16K
 ↓
24K
 ↓
32K
```

while increasing task difficulty as the model got stronger. ([arXiv][3])

They later find approximately logarithmic scaling between average reasoning length and raw reward over part of the training trajectory. ([arXiv][3])

So their training curriculum is effectively:

```text
model improves
    ↓
current data gets too easy
    ↓
increase problem difficulty
    +
increase reasoning budget
    ↓
continue exploration
```

This looks much closer to a **closed-loop curriculum** than a fixed RL dataset.

---

# 7. AReaL: the strongest treatment of the actual off-policy problem

AReaL goes much further than Magistral in formally dealing with asynchronous RL.

The architecture is:

```text
             ┌───────────────────┐
             │ Rollout workers   │
             └─────────┬─────────┘
                       │
                       ▼
               ┌──────────────┐
               │ Reward       │
               │ service      │
               └──────┬───────┘
                      │
                      ▼
               ┌──────────────┐
               │ Replay       │
               │ buffer       │
               └──────┬───────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Trainer workers │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Parameter svc   │
             └────────┬────────┘
                      │
                      └──► rollout workers
```

Trajectories are used only once from the replay buffer. ([arXiv][1])

Unlike Magistral, AReaL makes **staleness a first-class state variable**.

---

# 8. AReaL's staleness parameter η

Suppose:

```text
current trainer policy = θ₁₀
```

and the replay buffer contains trajectories from:

```text
θ₁₀
θ₉
θ₈
θ₇
θ₆
...
```

AReaL defines maximum permitted staleness:

[
\eta
]

and rate-limits rollout generation to keep the queue inside that lag bound. Their controller uses a relationship between generated-trajectory count, batch size, trainer version, and (\eta) to prevent rollout workers from getting arbitrarily far ahead. ([arXiv][1])

Conceptually:

```text
η = 0
θ₁₀ data only
≈ synchronous

η = 1
θ₁₀ / θ₉ allowed

η = 4
θ₁₀ ... θ₆ allowed

η = ∞
anything in queue allowed
```

The clever systems insight is:

```text
small η
    ↓
excellent on-policyness
    ↓
generation sometimes throttled
    ↓
less throughput

large η
    ↓
rollout workers run freely
    ↓
better utilization
    ↓
harder RL optimization
```

Therefore the correct solution isn't simply “make η tiny.”

It's:

> Build an RL algorithm that tolerates larger η.

And that's exactly what they do.

---

# 9. Decoupled PPO is the core contribution of AReaL

Standard PPO effectively assumes:

[
\pi_{\text{old}}
]

serves two jobs.

It is simultaneously:

1. the **behavior policy** that generated your sample;
2. the **proximal policy** around which you enforce your trust region.

In synchronous RL that's natural because they're nearly the same thing.

In async RL they are not.

Suppose your rollout came from:

[
\pi_{\text{behav}}=\theta_{i-4}
]

but your current good policy is approximately:

[
\pi_{\text{prox}}=\theta_{i}
]

and you're training:

[
\pi_\theta.
]

AReaL splits the two roles.

The central ratio can be conceptually factorized as:

[
\frac{\pi_\theta}{\pi_{\text{behav}}}
=====================================

\frac{\pi_{\text{prox}}}{\pi_{\text{behav}}}
\frac{\pi_\theta}{\pi_{\text{prox}}}.
]

Now:

### First term

[
\frac{\pi_{\text{prox}}}{\pi_{\text{behav}}}
]

is an **importance-sampling correction** for the stale rollout.

### Second term

[
\frac{\pi_\theta}{\pi_{\text{prox}}}
]

is what gets constrained by PPO clipping.

This is a much better conceptual separation. ([arXiv][1])

Why?

If you clip around the ancient behavior policy:

```text
θᵢ₋₄ ←──── θᵢ ← θnew
  ↑
 PPO keeps pulling toward here
```

your optimizer is continually being pulled toward an outdated, lower-quality model.

AReaL instead says:

```text
behavior policy      = correct sampling distribution
proximal policy      = where trust region should be centered
```

Those are two different concepts.

That is probably the deepest algorithmic contribution among the three systems papers.

---

# 10. AReaL even allows multiple policies inside one sequence

Like Magistral, it can change weights while generation is underway.

But AReaL handles the KV cache differently.

When new weights arrive:

```text
interrupt generation
        ↓
discard stale KV cache
        ↓
load θnew
        ↓
recompute context/KV using θnew
        ↓
resume decoding
```

([arXiv][1])

So compare:

```text
Magistral
─────────
θ changes
KV stays old
generation continues


AReaL
─────
θ changes
KV discarded
KV recomputed
generation continues
```

AReaL is more expensive at the transition but much cleaner.

The paper also proves that a trajectory generated in segments by multiple policy versions can be treated as being generated by an equivalent behavior policy, allowing the importance-sampling formulation to remain coherent. ([arXiv][1])

---

# 11. How much staleness can you tolerate?

This is where AReaL provides the strongest experimental evidence.

For a 1.5B math model, **naive PPO collapses badly as staleness increases**.

At maximum staleness 4, for example, AIME24 falls dramatically without their decoupled objective, while the decoupled version remains approximately at the zero-staleness baseline. Moderate values around (\eta\le8) preserve most benchmark performance while unlocking significantly higher throughput. Unbounded staleness remains worse even with the improved objective. ([arXiv][1])

That's an important result:

[
\boxed{\text{off-policy correction does not mean unlimited stale data is free}}
]

There is still a finite useful stale-data window.

---

# 12. And the systems gains are real

AReaL adds:

* dynamic token-balanced microbatch allocation;
* interruptible generation;
* parallel reward computation;
* separate generation and training;
* direct parameter service;
* controlled replay-buffer staleness. ([arXiv][1])

Dynamic microbatching alone improves training throughput by roughly **30% on average** in their ablation.

Interruptible generation gives roughly **12–17%** generation-throughput improvement in the tested configurations. ([arXiv][1])

More importantly, end-to-end:

```text
1.5B math
Sync AReaL     41.0 h
Async AReaL    14.8 h

7B math
Sync AReaL     57.7 h
Async AReaL    25.4 h

14B coding
Sync AReaL     48.8 h
Async AReaL    21.9 h
```

with very similar final benchmark scores. ([arXiv][1])

They also demonstrate scaling to **512 GPUs**. ([arXiv][1])

One small paper-version detail: the latest AReaL PDF's abstract says “up to 2.77×” in one place while the introduction reports 2.57×; the concrete tables are more informative than relying on the headline maximum. ([arXiv][1])

---

# 13. StreamRL approaches the problem from the opposite direction

If AReaL's question is:

> “How stale can RL become while remaining mathematically stable?”

StreamRL's question is:

> **“Why do generation and training need to live on the same machines at all?”**

This is a very systems-oriented paper.

It challenges the usual colocated architecture:

```text
GPU cluster
│
├── rollout
├── unload / reshard
├── training
├── reshard
├── rollout
└── ...
```

and replaces it with:

```text
┌───────────────────────┐
│ Stream Generation     │
│ Service               │
│                       │
│ inference-optimized   │
└───────────┬───────────┘
            │ trajectories
            │
            ▼
┌───────────────────────┐
│ Trainer               │
│                       │
│ training-optimized    │
└───────────┬───────────┘
            │
            └──── weights ─────► SGS
```

The two clusters can even have **different GPU types or live in different datacenters**. ([arXiv][2])

---

# 14. Why disaggregation becomes attractive

Inference and training want very different hardware configurations.

Generation is often:

```text
memory bandwidth / KV-cache heavy
large inference batches
different TP/DP choice
```

Training is:

```text
compute heavy
optimizer states
gradients
FSDP / TP / PP
different memory requirements
```

A colocated framework forces one resource topology to serve both.

StreamRL calls this **resource coupling**. ([arXiv][2])

With disaggregation you can instead do something like:

```text
Generation
32 × H20

Training
16 × H800
```

if that maximizes dollars per sample.

And StreamRL actually evaluates such a setup.

---

# 15. Streaming is the key abstraction in StreamRL

Consider batch generation:

```text
g1 ✓
g2 ✓
g3 ------------------------✓
g4 ✓
g5 --------✓

                            ↓
                  now send whole batch
```

StreamRL instead immediately emits each finished sequence:

```text
g1 ✓ ─────────► trainer
g2 ✓ ─────────► trainer
g4 ✓ ─────────► trainer
g5      ✓ ────► trainer
g3                  ✓ ────► trainer
```

That lets downstream operations start before the rollout batch is complete. 

This sounds trivial, but it's a profound change in execution granularity:

```text
batch is no longer the unit of scheduling

trajectory/sample is.
```

---

# 16. StreamRL keeps async much safer than AReaL

This is an important distinction.

StreamRL deliberately stays at **one-step asynchronous RL**:

```text
generation θᵢ₊₁    ███████████
training data θᵢ   ███████
```

rather than allowing arbitrarily stale trajectories.

Their fully asynchronous pipeline streams old samples to training, overlaps parameter communication, and permits generation to continue on the previous version, but they explicitly say they do not introduce samples more than one step stale. ([arXiv][2])

Therefore StreamRL does **not need an AReaL-style decoupled PPO formulation**.

Their strategy is basically:

[
\boxed{\text{get most of the systems benefit with only one-step off-policy data}}
]

That's a very sensible engineering position.

---

# 17. The coolest part of StreamRL: predict which prompts will run long

StreamRL realizes that asynchronous execution doesn't completely solve this:

```text
normal prompt   → 4K
normal prompt   → 5K
normal prompt   → 3K
normal prompt   → 6K
evil prompt     → 20K
```

So they train a **small LLM that predicts completion length from the prompt**.

Training data:

```text
(prompt, observed output length)
```

They SFT a small model on those pairs. ([arXiv][2])

At runtime:

```text
prompts
  │
  ▼
length ranker
  │
  ├── expected normal
  │
  └── expected long-tail
```

The long-tail prompts are assigned separately.

Why?

Long sequences typically prefer a **smaller decode batch** because they occupy the batch for a long time.

So you can give the long tail dedicated resources rather than poisoning latency for everybody else.

Their ranker reaches up to about **87% recall on the longest 20% of samples** in their experiments. ([arXiv][2])

---

# 18. StreamRL also dynamically adds inference nodes

Reasoning length changes during RL.

Imagine initially:

```text
generation = 150 sec
training   = 150 sec
```

Perfect.

Later the model learns to think longer:

```text
generation = 260 sec
training   = 155 sec
```

Now training sits idle.

StreamRL monitors this gap and can elastically increase generation resources.

In one experiment, as output lengths increased from roughly 10K toward 20K, it automatically added an **8-GPU generation node** when the stages became imbalanced. ([arXiv][2])

Its resource planner essentially minimizes:

[
T_{\text{iteration}}
====================

\max(T_{\text{generation}},T_{\text{training}})
]

subject to the GPU budget. ([arXiv][2])

That's exactly the right objective once the two stages are fully overlapped.

---

# 19. StreamRL's cross-datacenter experiment is quite compelling

Their main environment used H800 GPUs connected via high-bandwidth RDMA.

They then moved the generation service to an H20 cluster in another datacenter while retaining training on H800s.

The datacenters were linked by an **80 Gbps connection**. ([arXiv][2])

Result:

**throughput per hardware cost improved roughly 1.23–1.31×**, because H20s were more economical for generation. Even a 72B model's cross-datacenter weight transfer took under roughly 10 seconds, less than 2% of iteration time in their setting. ([arXiv][2])

That's a powerful argument for disaggregation.

---

# 20. But StreamRL has an important experimental caveat

For their systems throughput tests, they use an internal CodeMath prompt set and DeepSeek-R1 responses as ground-truth output lengths.

To ensure all frameworks see exactly the same long-tail workload, they actually modify generation to produce outputs according to those predetermined lengths. ([arXiv][2])

That's defensible for a **systems benchmark** because it isolates scheduling differences.

But it means the headline throughput experiment isn't exactly the same thing as a natural evolving online RL workload.

Their separate PPO experiment is what supports the claim that one-step async maintains convergence. ([arXiv][2])

I would therefore interpret StreamRL primarily as:

> a very strong **distributed-systems paper**, not a definitive algorithmic study of off-policy reasoning RL.

---

# 21. AsyncFlow takes the abstraction one level higher

StreamRL primarily thinks in terms of:

```text
GENERATION
    ↓
TRAINING
```

AsyncFlow says an RL system actually consists of many tasks:

```text
actor rollout
      ↓
reference logprob
      ↓
reward
      ↓
actor forward/logprob
      ↓
advantage
      ↓
actor update
```

And these tasks don't necessarily need to execute as large monolithic stages.

So AsyncFlow creates a streaming distributed data layer called:

# `TransferQueue`

([arXiv][4])

---

# 22. TransferQueue is basically a dataflow operating system for RL

Conceptually:

```text
                         TransferQueue
                  ┌────────────────────────┐
                  │      Control plane     │
                  │                        │
                  │ Which samples have:    │
                  │ rollout? reward?       │
                  │ ref logprob? advantage?│
                  └────────────┬───────────┘
                               │ metadata
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  storage worker         storage worker         storage worker
```

Each downstream task asks:

```text
give me N samples for which:

rollout_ready = true
reward_ready = true
ref_logprob_ready = true
...
```

Instead of hardcoding:

```text
rollout worker 3
    ↓
reward worker 3
    ↓
trainer DP rank 3
```

the queue dynamically schedules whichever samples are ready. ([arXiv][4])

That makes task execution much more like a **stream-processing DAG**.

---

# 23. This gives AsyncFlow natural load balancing

Suppose:

```text
trainer DP 0 = fast
trainer DP 1 = fast
trainer DP 2 = slow
```

Static assignment gives:

```text
100 samples each
```

and everyone waits for DP2.

TransferQueue instead lets faster consumers continually request additional ready work:

```text
DP0  ─► request
DP1  ─► request
DP2  ─► request

DP0 finishes → request again
DP1 finishes → request again
```

The paper explicitly notes this can compensate both for hardware heterogeneity and variable response length. ([arXiv][4])

This is a very general idea.

---

# 24. AsyncFlow's async policy update is conservative like StreamRL

AsyncFlow's deployed strategy allows roughly a **one-step delay**.

When the trainer finishes new weights:

```text
rollout still generating using θᵢ
             │
             ├──────── continues
             │
new θᵢ₊₁ ────► CPU/host memory
             │
rollout iteration finishes
             │
             ▼
host → NPU
             │
             ▼
next rollout uses θᵢ₊₁
```

([arXiv][4])

This means expensive network transfer overlaps with computation.

Only the relatively fast H2D transfer remains exposed at the boundary.

Again:

[
\text{keep staleness ≈ 1 step}
]

rather than solving arbitrary off-policy RL.

---

# 25. AsyncFlow has an even finer-grained idea—but it wasn't implemented

The authors propose asynchronously updating rollout replicas one by one:

```text
rollout replicas:

R0 θnew   ← update
R1 θold   ← still generating
R2 θold   ← still generating
R3 θold   ← still generating

then

R0 θnew
R1 θnew   ← update
R2 θold
R3 θold
```

Thus some new data is already generated using the latest model while other workers keep the pipeline fed.

They call this effectively **sub-step asynchrony**. ([arXiv][4])

But this is important:

> The paper explicitly leaves the implementation of that mechanism as future work. ([arXiv][4])

So don't confuse Figure 8(d) with a measured production feature.

---

# 26. AsyncFlow's results

They evaluate:

* Qwen2.5-7B and 32B;
* GRPO;
* DeepScaleR;
* Ascend NPU clusters;
* 32 to 1024 NPUs;
* MindSpeed training;
* vLLM-Ascend inference. ([arXiv][4])

Against a port of verl, they report an average throughput improvement of about:

[
1.59\times
]

with a peak of:

[
2.03\times
]

for the 7B model at 256 NPUs. ([arXiv][4])

There's also a **2.74×** number in their ablation, but that comparison is against their deliberately sequential task-separated baseline:

```text
baseline                    1.00×
+ TransferQueue             2.01×
+ async optimizations       2.74×
```

not 2.74× versus verl. ([arXiv][4])

That's an important distinction.

---

# 27. AsyncFlow's algorithmic validation is weaker than AReaL's

AsyncFlow checks synchronous vs asynchronous training on a 7B model running on 16 NPUs and reports similar average reward and response-length behavior. ([arXiv][4])

That's useful evidence.

But compare the rigor:

### AReaL

```text
η = 0
η = 1
η = 2
η = 4
η = 8
η = 16
η = ∞

×
naive PPO vs decoupled PPO

×
multiple final benchmarks
```

### AsyncFlow

```text
sync vs async
reward curve
response-length curve
```

So from an **RL algorithm** perspective, AReaL has much stronger evidence.

AsyncFlow's contribution is really its **dataflow architecture**.

---

# 28. The four designs side by side

|                              | Magistral                       | AReaL                                       | StreamRL                         | AsyncFlow                         |
| ---------------------------- | ------------------------------- | ------------------------------------------- | -------------------------------- | --------------------------------- |
| Primary goal                 | Train reasoning models          | Fully async RL                              | Efficient disaggregated RL       | General task-separated RL runtime |
| Async granularity            | Continuous completions          | Continuous trajectories                     | Streaming samples                | Streaming task outputs            |
| Allowed staleness            | bounded indirectly              | explicit (\eta), potentially many steps     | ~1 step                          | ~1 step                           |
| Mid-rollout weight change    | **Yes**                         | **Yes**                                     | generally previous iteration     | update at iteration boundary      |
| KV after weight change       | **keep stale KV**               | **recompute KV**                            | avoids this issue                | avoids this issue                 |
| Special off-policy algorithm | no new objective                | **Decoupled PPO**                           | no, limits lag                   | no, limits lag                    |
| Long-tail handling           | continuous generation + packing | interruptible generation + dynamic batching | **length predictor + scheduler** | dynamic TransferQueue             |
| Elastic resource allocation  | not central                     | not central                                 | **Yes**                          | planner/load balancing            |
| Heterogeneous HW             | not focus                       | not focus                                   | **Yes**                          | backend abstraction               |
| Cross-DC                     | no                              | no                                          | **Yes**                          | architectural possibility         |
| Main strength                | RL recipe                       | algorithm-system co-design                  | systems/resource architecture    | generic dataflow/orchestration    |

This comparison comes directly from the execution models and mechanisms described across the four papers. ([arXiv][3])

---

# 29. The deepest insight: there are actually four independent axes

I wouldn't categorize RL systems simply as:

```text
sync
vs
async
```

That's too crude.

There are at least **four axes**.

### Axis 1 — stage concurrency

```text
rollout ──────────────
                     training ──────────

versus

rollout ───────────────────────────────
             training ─────────────
```

Do rollout and training execute simultaneously?

---

### Axis 2 — sample staleness

How old can a trajectory be?

```text
0 steps        synchronous
1 step         StreamRL / AsyncFlow
N bounded      AReaL
```

---

### Axis 3 — intra-trajectory policy consistency

Does one rollout have one policy?

```text
trajectory
θ₄ θ₄ θ₄ θ₄ θ₄ θ₄
```

or:

```text
trajectory
θ₄ θ₄ θ₄ θ₅ θ₅ θ₆
```

Magistral and AReaL explicitly enter the second regime. ([arXiv][3])

---

### Axis 4 — state consistency after hot-swapping

This is the subtle one:

```text
Magistral:
θnew + KVold

AReaL:
θnew + recomputed KVnew
```

So even two systems that both perform “mid-rollout weight updates” have different underlying sampling processes. ([arXiv][3])

That distinction is rarely captured by the word *asynchronous*.

---

# 30. There is an interesting spectrum here

I would arrange the papers like this:

```text
MORE CONSERVATIVE                            MORE AGGRESSIVE
OFF-POLICY                                      OFF-POLICY

StreamRL
    │
    │ 1-step stale
    │ whole-generation semantics
    ▼
AsyncFlow
    │
    │ 1-step + finer task streaming
    ▼
Magistral
    │
    │ weights can change inside trajectory
    │ stale KV tolerated
    ▼
AReaL
    │
    │ multiple-step staleness allowed
    │ behavior/proximal policies separated
    ▼
full asynchronous RL
```

Interestingly, AReaL is **algorithmically more careful** despite permitting more aggressive asynchrony.

Magistral is probably the more daring engineering choice because it says, roughly:

> Hot-swap the weights, don't even rebuild the KV cache, bound how far behind we get, and empirically it works.

AReaL says:

> If we're going this asynchronous, let's explicitly change the RL objective and model staleness.

---

# 31. What I would take from each paper

If I were building a reasoning RL stack, I would **not choose one paper**.

I'd steal different pieces.

From **Magistral**:

```text
✓ difficulty-adaptive dataset
✓ zero-variance group filtering
✓ Clip-Higher
✓ no unnecessary KL reference model
✓ token-balanced training microbatches
✓ nasync / nbatch staleness awareness
✓ dynamically increase max reasoning length
```

([arXiv][3])

From **AReaL**:

```text
✓ explicit policy-version metadata
✓ max-staleness constraint η
✓ behavior-policy logprobs
✓ decoupled behavior/proximal PPO
✓ interruptible rollouts
✓ dynamic token microbatching
```

([arXiv][1])

From **StreamRL**:

```text
✓ physically disaggregate inference and training
✓ stream completed responses immediately
✓ predict long-tail generations
✓ allocate long-tail requests separately
✓ balance Tgen ≈ Ttrain
✓ elastically scale rollout resources
✓ topology-aware weight broadcast
```

([arXiv][2])

From **AsyncFlow**:

```text
✓ treat RL as a dataflow DAG
✓ shared streaming experience store
✓ readiness metadata per sample
✓ consumer-driven dynamic scheduling
✓ engine-independent interfaces
✓ overlap weight transfer via CPU memory
```

([arXiv][4])

---

# 32. The architecture I think these papers are converging toward

All four point toward something approximately like:

```text
                       PROMPT SOURCE
                            │
                            ▼
                 ┌────────────────────┐
                 │ Difficulty / length│
                 │ prediction         │
                 └─────────┬──────────┘
                           │
                    dynamic dispatch
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      rollout pool A              rollout pool B
       normal jobs                 long-tail jobs
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                STREAMING EXPERIENCE BUS
          ┌────────────────────────────────┐
          │ prompt                         │
          │ response                       │
          │ reward                         │
          │ behavior logprobs              │
          │ policy version                 │
          │ reference logprobs             │
          │ advantage                      │
          │ readiness flags                │
          └──────┬───────────────┬─────────┘
                 │               │
          verifier pool      ref-logprob pool
                 │               │
                 └───────┬───────┘
                         ▼
                   TRAINER POOL
                         │
                     θᵢ → θᵢ₊₁
                         │
                         ▼
                  PARAMETER SERVER
                         │
                 async broadcast
                         │
                         ▼
                    rollout pools
```

Then add an AReaL-style controller:

```text
if policy_lag > η:
    throttle generation
```

and a StreamRL-style controller:

```text
if Tgen >> Ttrain:
    add rollout capacity

if Ttrain >> Tgen:
    add training capacity
```

That's basically the logical synthesis of the four systems.

---

# 33. One thing I would *not* copy blindly

I would be cautious about the Magistral choice:

```text
change weights mid-generation
+
retain KV cache from old weights
```

It is fascinating and apparently worked for them, but it means the hidden state corresponding to:

[
h_t=f_{\theta_i}(x_{\le t})
]

is subsequently used while decoding with:

[
f_{\theta_{i+1}}
]

so you are no longer sampling from the clean autoregressive distribution corresponding to **either** model.

That may work beautifully when parameter updates are tiny.

It may behave very differently with:

* higher learning rates;
* larger policy updates;
* aggressive GRPO;
* low-entropy models;
* LoRA adapters;
* sparse/MoE models;
* large numbers of updates during a single 32K rollout.

Magistral offers valuable empirical evidence, but not a general guarantee. Their own constraint on (n_{\text{async}}/n_{\text{batch}}) is effectively evidence that this approximation has limits. ([arXiv][3])

AReaL's KV recomputation is the safer design.

---

# 34. If I rank the papers by what they contribute

For **reasoning-model training science**:

**1. Magistral**
By far. Data difficulty, GRPO modifications, entropy, reasoning length, pure RL vs distillation, negative results.

For **asynchronous RL algorithm design**:

**1. AReaL**
The behavior-policy/proximal-policy distinction is the cleanest treatment of staleness here.

For **large-cluster systems architecture**:

**1. StreamRL**
Disaggregation, length prediction, heterogeneous GPUs, elasticity and cross-datacenter operation are excellent.

For **general framework architecture**:

**1. AsyncFlow**
TransferQueue is probably the most reusable software abstraction of the four.

---

# 35. And the overall lesson

The first generation of LLM RL frameworks largely optimized this:

```text
make PPO/GRPO run efficiently
```

These papers are moving toward:

```text
                 RL POST-TRAINING

       ┌──────────────────────────────┐
       │ distributed stream processor│
       └──────────────────────────────┘

generation             verification
    │                        │
    ├────── trajectories ────┤
    │                        │
reward                  logprobs
    │                        │
    └──────────┬─────────────┘
               │
            training
               │
            weights
               │
          generation
```

And the **batch barrier is disappearing**.

The trajectory/sample becomes the atomic scheduling unit.

Once that happens:

* generation doesn't need to wait for generation;
* training doesn't need to wait for the entire rollout batch;
* reward doesn't need to wait for every response;
* parameter transfer doesn't need to stop inference;
* generation and training don't even need to use the same type of GPU;
* eventually, they don't necessarily need to be in the same datacenter.

The price is that **on-policyness stops being binary and becomes a resource you budget**.

That's the unifying idea I take from all four:

[
\boxed{
\text{Throughput}
\quad\leftrightarrow\quad
\text{Policy freshness}
}
]

Magistral controls it with queue size and (n_{\text{async}}/n_{\text{batch}}).

StreamRL and AsyncFlow largely constrain it to one step.

AReaL explicitly parameterizes it as (\eta) and modifies PPO so it can safely move further along that frontier. ([arXiv][3])

**That, more than any individual 2× throughput number, is what I think these papers are really about.**

[1]: https://arxiv.org/pdf/2505.24298 "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"
[2]: https://arxiv.org/pdf/2504.15930 "StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation"
[3]: https://arxiv.org/pdf/2506.10910 "Magistral"
[4]: https://arxiv.org/pdf/2507.01663 "AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training"

