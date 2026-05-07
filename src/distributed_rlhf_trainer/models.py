"""Data models for RLHF training configuration and runtime state.

Separates data definitions from logic — every struct used in the RLHF loop
is defined here with full validation via Pydantic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_PPO_EPOCHS = 4
DEFAULT_CLIP_RANGE = 0.2
DEFAULT_VALUE_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_DISCOUNT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_KL_PENALTY_COEF = 0.1
MIN_CLIP_RANGE = 0.01
MAX_CLIP_RANGE = 0.5


class PPOConfig(BaseModel):
    """Configuration for Proximal Policy Optimization.

    Attributes:
        learning_rate: Optimizer learning rate for policy updates.
        ppo_epochs: Number of optimization epochs per batch.
        clip_range: PPO clipping parameter epsilon.
        value_coef: Coefficient for value function loss.
        entropy_coef: Coefficient for entropy bonus.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        gamma: Discount factor for future rewards.
        max_grad_norm: Maximum gradient norm for clipping.
        kl_penalty_coef: Coefficient for KL divergence penalty.
    """

    learning_rate: float = Field(default=DEFAULT_LEARNING_RATE, gt=0)
    ppo_epochs: int = Field(default=DEFAULT_PPO_EPOCHS, ge=1)
    clip_range: float = Field(default=DEFAULT_CLIP_RANGE, ge=MIN_CLIP_RANGE, le=MAX_CLIP_RANGE)
    value_coef: float = Field(default=DEFAULT_VALUE_COEF, ge=0)
    entropy_coef: float = Field(default=DEFAULT_ENTROPY_COEF, ge=0)
    gae_lambda: float = Field(default=DEFAULT_GAE_LAMBDA, ge=0, le=1)
    gamma: float = Field(default=DEFAULT_DISCOUNT_GAMMA, ge=0, le=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    kl_penalty_coef: float = Field(default=DEFAULT_KL_PENALTY_COEF, ge=0)


class RewardModelConfig(BaseModel):
    """Configuration for the reward model.

    Attributes:
        hidden_dim: Hidden layer dimension.
        num_layers: Number of transformer-style layers.
        dropout: Dropout probability.
        normalize_rewards: Whether to normalize reward scores.
    """

    hidden_dim: int = Field(default=256, ge=1)
    num_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    normalize_rewards: bool = True


class RLHFConfig(BaseModel):
    """Top-level configuration for the RLHF training loop.

    Attributes:
        ppo: PPO optimizer configuration.
        reward: Reward model configuration.
        batch_size: Number of experiences per training batch.
        max_seq_length: Maximum sequence length for generated text.
        num_iterations: Total RLHF iterations to run.
        vocab_size: Vocabulary size for the language model.
        hidden_dim: Hidden dimension of the policy model.
        checkpoint_interval: Save checkpoint every N iterations.
        log_interval: Log metrics every N iterations.
        seed: Random seed for reproducibility.
    """

    ppo: PPOConfig = Field(default_factory=PPOConfig)
    reward: RewardModelConfig = Field(default_factory=RewardModelConfig)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)
    max_seq_length: int = Field(default=DEFAULT_MAX_SEQ_LENGTH, ge=1)
    num_iterations: int = Field(default=100, ge=1)
    vocab_size: int = Field(default=1000, ge=1)
    hidden_dim: int = Field(default=256, ge=1)
    checkpoint_interval: int = Field(default=10, ge=1)
    log_interval: int = Field(default=1, ge=1)
    seed: int = 42

    @model_validator(mode="after")
    def validate_dimensions(self) -> RLHFConfig:
        """Ensure reward model hidden dim matches policy hidden dim."""
        if self.reward.hidden_dim != self.hidden_dim:
            logger.warning(
                "Reward hidden_dim (%d) differs from policy hidden_dim (%d). "
                "Aligning reward model to policy dimension.",
                self.reward.hidden_dim,
                self.hidden_dim,
            )
            self.reward.hidden_dim = self.hidden_dim
        return self


@dataclass
class Experience:
    """A single RLHF experience tuple.

    Stores all information needed for a PPO update step: the generated
    sequence, its log probability under the old policy, the reward score,
    and the value estimate from the critic.

    Attributes:
        query_ids: Input prompt token IDs.
        response_ids: Generated response token IDs.
        old_log_probs: Log probabilities under the policy that generated the response.
        rewards: Scalar reward from the reward model.
        values: Value estimates from the critic network.
        advantages: Computed GAE advantages (filled during training).
        returns: Computed discounted returns (filled during training).
    """

    query_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    rewards: float = 0.0
    values: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)


@dataclass
class TrainingMetrics:
    """Aggregated metrics from a training iteration.

    Attributes:
        iteration: Current training iteration number.
        policy_loss: Mean policy gradient loss.
        value_loss: Mean value function loss.
        entropy: Mean policy entropy.
        kl_divergence: Mean KL divergence from reference policy.
        mean_reward: Mean reward across the batch.
        clip_fraction: Fraction of updates that were clipped.
        approx_kl: Approximate KL divergence for early stopping.
        explained_variance: How well the value function explains returns.
    """

    iteration: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    mean_reward: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    explained_variance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a flat dictionary for logging.

        Returns:
            Dictionary of metric name to value mappings.
        """
        return {
            "iteration": self.iteration,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "kl_divergence": self.kl_divergence,
            "mean_reward": self.mean_reward,
            "clip_fraction": self.clip_fraction,
            "approx_kl": self.approx_kl,
            "explained_variance": self.explained_variance,
        }
