# Architecture: distributed-rlhf-trainer

## Overview

The RLHF training loop is decomposed into four independent components following the Single Responsibility Principle. Each component owns one concern and communicates through well-defined data structures (`Experience`, `TrainingMetrics`).

## Component Responsibilities

### RewardModel

- **Input:** Token ID sequences (batch_size, seq_len)
- **Output:** Scalar reward per sequence (batch_size,)
- **Architecture:** Embedding -> MLP backbone -> Linear head
- **Key decision:** Mean-pooling over sequence dimension (position-agnostic reward)

### PolicyModel

- **Input:** Token ID sequences
- **Output:** Logits (batch_size, seq_len, vocab_size) + Values (batch_size, seq_len)
- **Architecture:** Embedding -> 2-layer MLP -> Separate LM head + Value head
- **Key decision:** Separate value head (actor-critic, standard for PPO)

### ExperienceCollector

- **Input:** Batch of prompts
- **Output:** List of Experience dataclasses
- **Process:** Generate responses -> Score with reward model -> Package with values + log-probs
- **Key decision:** Runs in `torch.no_grad()` mode for efficiency

### PPOTrainer

- **Input:** List of Experience objects
- **Output:** TrainingMetrics
- **Process:** Compute GAE -> Normalize advantages -> Clipped PPO loss + Value loss + Entropy bonus + KL penalty
- **Key decision:** Early stopping on KL divergence (prevents policy collapse)

### RLHFOrchestrator

- **Input:** RLHFConfig
- **Output:** List of TrainingMetrics (one per iteration)
- **Process:** Wire components -> Loop (collect -> update -> log)
- **Key decision:** Only component aware of all others (mediator pattern)

## Data Flow

```
Prompts -> ExperienceCollector -> [Experience] -> PPOTrainer -> TrainingMetrics
               |                                      |
               v                                      v
          PolicyModel.generate()              PolicyModel.update()
          RewardModel.forward()
```

## Configuration Hierarchy

```
RLHFConfig
    ├── PPOConfig (learning_rate, clip_range, gae_lambda, ...)
    ├── RewardModelConfig (hidden_dim, num_layers, dropout, ...)
    ├── batch_size, max_seq_length, num_iterations
    └── vocab_size, hidden_dim, seed
```

All configs use Pydantic `BaseModel` with field validators — invalid hyperparameters are caught at construction time, not during training.
