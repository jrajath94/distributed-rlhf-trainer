# Interview Prep: distributed-rlhf-trainer

## Elevator Pitch (30 seconds)

I built a minimal RLHF training framework that decomposes the monolithic RLHF loop into four independent, testable components -- reward model, policy, experience collector, and PPO trainer. Unlike OpenRLHF (50k+ LoC) or TRL, each component can be understood, replaced, or distributed independently, making it straightforward to debug, customize, and scale.

## Why I Built This

### The Real Motivation

Working with production RLHF systems, I kept hitting the same problem: when a reward signal degraded or PPO diverged, debugging required understanding the entire monolithic codebase. The coupling between experience collection, reward scoring, and policy optimization made it impossible to unit test components in isolation. I built this to prove that clean separation of concerns -- the same principle we use in backend systems -- applies equally well to ML training loops.

### Company-Specific Framing

| Company         | Why This Matters to Them                                                                                                                                                          |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Anthropic       | RLHF is core to Constitutional AI. Clean separation enables rapid iteration on reward modeling without touching the policy optimizer -- critical for alignment research velocity. |
| OpenAI          | At scale, RLHF components need independent teams. This architecture enables that organizational structure while maintaining correctness.                                          |
| DeepMind        | The mediator pattern here mirrors how DeepMind structures research -- independent components with clear interfaces enable ablation studies.                                       |
| NVIDIA          | Distributing RLHF across GPUs requires component-level placement. This architecture maps naturally to multi-GPU scheduling.                                                       |
| Google          | Large-scale training demands independent component scaling. Experience collection can scale horizontally while PPO runs on fewer, more powerful nodes.                            |
| Meta FAIR       | Open-source RLHF implementations need to be readable for the community. This is production-quality code that researchers can actually follow.                                     |
| Citadel/JS/2Sig | The same decomposition principle applies to backtesting systems -- separate signal generation from portfolio optimization from execution.                                         |

## Architecture Deep-Dive

The system follows a mediator pattern with four independent components:

**Data flow per iteration:**

1. `RLHFOrchestrator` samples a batch of prompts
2. `ExperienceCollector` generates responses via `PolicyModel.generate()` (autoregressive, top-k sampling)
3. `ExperienceCollector` scores (prompt, response) pairs via `RewardModel.forward()` (mean-pooled MLP)
4. `ExperienceCollector` packages everything into `Experience` dataclasses (query_ids, response_ids, old_log_probs, rewards, values)
5. `PPOTrainer` computes GAE advantages, normalizes, runs clipped PPO update with value loss + entropy bonus + KL penalty
6. Metrics are logged; loop repeats

### Key Design Decisions

| Decision                       | Why                                                                    | Alternative      | Tradeoff                                                   |
| ------------------------------ | ---------------------------------------------------------------------- | ---------------- | ---------------------------------------------------------- |
| Separate value head            | Standard actor-critic, enables independent value function debugging    | Shared backbone  | 2x head parameters, but cleaner gradient flow              |
| GAE with configurable lambda   | Lambda=0.95 is standard but some tasks need different bias-variance    | Fixed MC returns | Slightly more compute for the reverse sweep                |
| KL penalty + early stopping    | Double safety net prevents catastrophic policy collapse                | Just KL penalty  | Some compute wasted on KL check, but prevents NaN losses   |
| Pydantic BaseModel for configs | Validation at construction time catches bugs before expensive training | Plain dataclass  | Pydantic adds ~2ms import overhead, worth it for safety    |
| Experience as dataclass        | Lightweight, no validation overhead in the hot loop                    | Pydantic model   | No field validation on experiences, but speed matters here |

### Scaling Analysis

- **Current capacity:** 3.9 experiences/sec on CPU with batch_size=8, hidden_dim=128
- **10x strategy:** Move to GPU, increase batch_size to 64, pipeline experience collection with PPO updates (double buffering)
- **100x strategy:** Ray-based distributed experience collection across N workers, DeepSpeed ZeRO-3 for large policy models, vLLM for inference
- **Bottlenecks:** Autoregressive generation dominates (85% of iteration time); PPO update is fast
- **Cost estimate:** For 7B parameter policy on 8xA100: ~$50/hr on cloud, ~2000 iterations/hr

## 10 Deep-Dive Interview Questions

### Q1: Walk me through how a single RLHF iteration works end-to-end.

**A:** The orchestrator calls `_train_step()` which: (1) samples random prompts as a `(batch_size, 16)` tensor, (2) passes them to `ExperienceCollector.collect_batch()` which generates responses autoregressively via top-k sampling, scores them with the reward model's mean-pooled MLP, and packages query_ids, response_ids, old_log_probs, rewards, and value estimates into Experience objects, (3) passes the experiences to `PPOTrainer.update()` which computes per-step rewards (scalar reward placed at the last timestep), runs GAE with a reverse sweep, normalizes advantages to zero-mean unit-variance, then runs N PPO epochs with clipped surrogate objective, MSE value loss, entropy bonus, and KL penalty, (4) returns TrainingMetrics to the orchestrator for logging.

### Q2: Why separate the ExperienceCollector from the PPOTrainer instead of a single train loop?

**A:** Three reasons. First, testability: I can unit test experience collection by mocking the policy and reward model, and unit test PPO by providing synthetic experiences. Second, distribution: in production, you want experience collection on cheap inference nodes and PPO updates on expensive training nodes -- separation enables this without code changes. Third, replaceability: swap the reward model architecture without touching PPO, or replace PPO with REINFORCE++ without touching collection.

### Q3: What was the hardest bug you hit?

**A:** The numpy-torch interop issue. When converting advantages from PyTorch tensor to numpy for normalization, `tensor.numpy()` fails silently on some torch builds that weren't compiled with numpy C API support. The symptom was `RuntimeError: Numpy is not available` only on certain platforms. The fix was using `np.array(tensor.tolist())` instead, which always works because it goes through Python primitives. This taught me to never assume torch-numpy interop works -- always provide a fallback path.

### Q4: How would you scale this to 100x?

**A:** Replace `ExperienceCollector` with a Ray remote actor pool -- each actor has a copy of the policy model and runs inference independently. Use vLLM instead of autoregressive PyTorch generation for 5-10x speedup. For the policy model, wrap it in DeepSpeed ZeRO-3 to shard across GPUs. Use Ray's object store for zero-copy experience transfer between collector actors and the PPO trainer. The orchestrator becomes a Ray driver that coordinates the pipeline. Key insight: experience collection and PPO training can be double-buffered -- collect batch N+1 while training on batch N.

### Q5: What would you do differently with more time?

**A:** Three things. (1) Add a reference policy KL constraint that uses the original frozen model weights, not just the old policy from last step -- this is what real RLHF uses. (2) Implement minibatch splitting within PPO epochs for better GPU utilization with large batches. (3) Add a proper prompt dataset loader instead of random token generation, with stratified sampling to ensure diversity.

### Q6: How does this compare to OpenRLHF?

**A:** OpenRLHF is production-grade at 50k+ LoC with Ray integration, vLLM inference, and DeepSpeed training -- it's what you'd use to train a real LLM. This project is intentionally minimal (~400 LoC core) to demonstrate the architecture clearly. The key insight is that OpenRLHF's architecture is sound (it also separates components), but the implementation couples concerns through shared state and complex inheritance. This project shows the same architecture can be implemented cleanly enough that each component fits in one screen of code.

### Q7: What are the security implications?

**A:** The main attack surface is the reward model -- reward hacking (policy finds exploits in the reward function) is well-documented. Mitigations include KL penalty (limits policy drift), reward normalization (prevents reward magnitude exploits), and early stopping on KL divergence. For deployment, model checkpoints should only be loaded from trusted sources. The API surface is minimal -- no network endpoints, no user input handling.

### Q8: Explain your testing strategy.

**A:** Three layers. Unit tests verify each component independently: reward model output shape, policy generation, GAE computation, advantage normalization. Integration tests run the full orchestrator for 2-3 iterations and verify metrics are finite and have correct iteration numbers. Parametrized tests cover edge cases: different temperatures, invalid configs (rejected by Pydantic). Coverage is 86% -- uncovered code is primarily the CLI module and checkpoint I/O which require filesystem interaction.

### Q9: What are the failure modes?

**A:** (1) Policy collapse: PPO clip ratio becomes saturated, policy always generates the same tokens. Detected by entropy going to zero, mitigated by entropy bonus and KL early stopping. (2) Reward hacking: policy finds degenerate sequences that score high rewards. Detected by reward variance dropping to zero. (3) NaN losses: gradient explosion from large advantages. Mitigated by gradient clipping and advantage normalization. (4) KL divergence explosion: policy drifts too far from reference. Mitigated by KL penalty coefficient and early stopping threshold.

### Q10: Explain GAE (Generalized Advantage Estimation) from first principles.

**A:** The advantage function A(s,a) = Q(s,a) - V(s) measures "how much better is action a than average in state s." We need this for policy gradients. There are two extremes for estimating it: TD(0) uses one-step bootstrapping (low variance but biased because V might be wrong), and Monte Carlo uses full returns (unbiased but high variance because rewards are noisy). GAE interpolates between them with parameter lambda. At lambda=0, it's TD(0); at lambda=1, it's Monte Carlo. The implementation is a single reverse sweep: start from the last timestep, compute TD error delta_t = r_t + gamma*V(t+1) - V(t), then accumulate GAE_t = delta_t + gamma*lambda\*GAE(t+1). Lambda=0.95 is the standard setting -- it keeps most of the variance reduction from bootstrapping while staying close to unbiased.

## Complexity Analysis

- **Time:** O(B _ T _ V) per PPO epoch where B=batch, T=seq_len, V=vocab_size -- dominated by logit computation
- **Space:** O(B _ T _ V) for logit tensors -- the largest allocation in the forward pass
- **Network:** Zero network calls in single-node mode; in distributed mode, O(B \* T) per experience transfer
- **Disk:** O(P) per checkpoint where P = number of model parameters

## Metrics & Results

| Metric                | Value       | How Measured                           | Significance                               |
| --------------------- | ----------- | -------------------------------------- | ------------------------------------------ |
| Collection throughput | 3.9 exp/sec | bench_core.py, 10 iterations, 2 warmup | Dominated by autoregressive generation     |
| PPO update p50        | 300 ms      | bench_core.py, 10 iterations           | Fast enough for real-time training         |
| PPO update p99        | 452 ms      | bench_core.py, 10 iterations           | Tail latency from GC and gradient clipping |
| E2E iteration         | 1,855 ms    | bench_core.py, 5 full iterations       | Generation is 85% of time                  |
| Test coverage         | 86%         | pytest-cov                             | Uncovered: CLI, checkpoint I/O             |
| Tests passing         | 38/38       | pytest                                 | Full unit + integration suite              |

## Career Narrative

How this project fits my story:

- **JPMorgan (VP)** -> Built distributed systems at scale; this project applies the same decomposition principles to ML training infrastructure
- **Goldman Sachs (Quant)** -> Understood reward shaping and policy optimization from a quant research perspective; RLHF is PPO applied to language
- **NVIDIA** -> Deep understanding of GPU memory hierarchies and kernel optimization; the scaling strategy for this project leverages that knowledge
- **This project** -> Demonstrates ability to architect clean ML training systems, understand RLHF internals, and write production-quality code

## Interview Red Flags to Avoid

- NEVER say "I built this to learn RLHF" (sounds junior)
- NEVER be unable to explain GAE or PPO clipping
- NEVER claim the benchmark numbers are from GPU when they're from CPU
- NEVER badmouth OpenRLHF (it's excellent for production; this is for understanding)
- ALWAYS connect separation of concerns to the company's specific RLHF challenges
- ALWAYS mention what you'd improve (reference policy, minibatching, real prompts)
- ALWAYS discuss failure modes unprompted (policy collapse, reward hacking, NaN)
