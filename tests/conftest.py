"""Shared test fixtures for distributed RLHF trainer."""

from __future__ import annotations

import pytest
import torch

from distributed_rlhf_trainer.core import (
    ExperienceCollector,
    PolicyModel,
    RewardModel,
)
from distributed_rlhf_trainer.models import (
    Experience,
    PPOConfig,
    RewardModelConfig,
    RLHFConfig,
)

VOCAB_SIZE = 100
HIDDEN_DIM = 64
BATCH_SIZE = 4
SEQ_LEN = 16
PROMPT_LEN = 8


@pytest.fixture
def ppo_config() -> PPOConfig:
    """Default PPO config for testing."""
    return PPOConfig(
        learning_rate=1e-3,
        ppo_epochs=2,
        clip_range=0.2,
    )


@pytest.fixture
def reward_config() -> RewardModelConfig:
    """Default reward model config for testing."""
    return RewardModelConfig(
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture
def rlhf_config() -> RLHFConfig:
    """Default RLHF config for testing."""
    return RLHFConfig(
        batch_size=BATCH_SIZE,
        max_seq_length=SEQ_LEN,
        num_iterations=2,
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        seed=42,
    )


@pytest.fixture
def policy_model() -> PolicyModel:
    """A small policy model for testing."""
    return PolicyModel(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)


@pytest.fixture
def reward_model(reward_config: RewardModelConfig) -> RewardModel:
    """A small reward model for testing."""
    return RewardModel(config=reward_config, vocab_size=VOCAB_SIZE)


@pytest.fixture
def sample_prompts() -> torch.Tensor:
    """Batch of random prompts."""
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, PROMPT_LEN))


@pytest.fixture
def sample_experiences(
    policy_model: PolicyModel,
    reward_model: RewardModel,
    rlhf_config: RLHFConfig,
    sample_prompts: torch.Tensor,
) -> list[Experience]:
    """Pre-collected experiences for PPO testing."""
    collector = ExperienceCollector(policy_model, reward_model, rlhf_config)
    return collector.collect_batch(sample_prompts)
