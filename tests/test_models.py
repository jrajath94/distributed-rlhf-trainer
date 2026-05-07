"""Tests for data models and configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distributed_rlhf_trainer.models import (
    Experience,
    PPOConfig,
    RewardModelConfig,
    RLHFConfig,
    TrainingMetrics,
)


class TestPPOConfig:
    """Tests for PPO configuration validation."""

    def test_default_values(self) -> None:
        """Default config should have reasonable PPO hyperparameters."""
        config = PPOConfig()
        assert config.clip_range == 0.2
        assert config.ppo_epochs == 4
        assert config.learning_rate == 1e-5

    @pytest.mark.parametrize(
        "field,value",
        [
            ("learning_rate", -1.0),
            ("clip_range", 0.0),
            ("clip_range", 0.6),
            ("ppo_epochs", 0),
            ("gae_lambda", -0.1),
            ("gae_lambda", 1.1),
        ],
    )
    def test_invalid_values_rejected(self, field: str, value: float) -> None:
        """Invalid hyperparameter values should raise ValidationError."""
        with pytest.raises(ValidationError):
            PPOConfig(**{field: value})

    def test_custom_values_accepted(self) -> None:
        """Valid custom values should be accepted."""
        config = PPOConfig(
            learning_rate=3e-4,
            ppo_epochs=8,
            clip_range=0.3,
        )
        assert config.learning_rate == 3e-4
        assert config.ppo_epochs == 8


class TestRewardModelConfig:
    """Tests for reward model configuration."""

    def test_default_values(self) -> None:
        """Default reward config should normalize rewards."""
        config = RewardModelConfig()
        assert config.normalize_rewards is True
        assert config.hidden_dim == 256

    def test_invalid_dropout_rejected(self) -> None:
        """Dropout >= 1.0 should be rejected."""
        with pytest.raises(ValidationError):
            RewardModelConfig(dropout=1.0)


class TestRLHFConfig:
    """Tests for top-level RLHF configuration."""

    def test_dimension_alignment(self) -> None:
        """Reward hidden_dim should be aligned to policy hidden_dim."""
        config = RLHFConfig(hidden_dim=128)
        assert config.reward.hidden_dim == 128

    def test_default_iteration_count(self) -> None:
        """Default config should specify 100 iterations."""
        config = RLHFConfig()
        assert config.num_iterations == 100


class TestExperience:
    """Tests for the Experience dataclass."""

    def test_default_creation(self) -> None:
        """Default experience should have empty lists and zero reward."""
        exp = Experience()
        assert exp.rewards == 0.0
        assert exp.query_ids == []
        assert exp.advantages == []

    def test_populated_experience(self) -> None:
        """Experience with data should store all fields."""
        exp = Experience(
            query_ids=[1, 2, 3],
            response_ids=[4, 5, 6],
            old_log_probs=[-0.5, -0.3, -0.7],
            rewards=1.5,
            values=[0.1, 0.2, 0.3],
        )
        assert len(exp.query_ids) == 3
        assert exp.rewards == 1.5


class TestTrainingMetrics:
    """Tests for training metrics."""

    def test_to_dict(self) -> None:
        """Metrics should serialize to a flat dictionary."""
        metrics = TrainingMetrics(
            iteration=5,
            policy_loss=0.5,
            mean_reward=1.2,
        )
        result = metrics.to_dict()
        assert result["iteration"] == 5
        assert result["policy_loss"] == 0.5
        assert result["mean_reward"] == 1.2
        assert len(result) == 9
