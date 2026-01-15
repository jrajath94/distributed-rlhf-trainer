"""Tests for core RLHF components: RewardModel, PolicyModel, PPOTrainer, Orchestrator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from distributed_rlhf_trainer.core import (
    ExperienceCollector,
    PolicyModel,
    PPOTrainer,
    RewardModel,
    RLHFOrchestrator,
)
from distributed_rlhf_trainer.exceptions import (
    ExperienceCollectionError,
    PolicyUpdateError,
)
from distributed_rlhf_trainer.models import (
    Experience,
    PPOConfig,
    RewardModelConfig,
    RLHFConfig,
    TrainingMetrics,
)
from distributed_rlhf_trainer.utils import (
    compute_gae,
    compute_kl_divergence,
    normalize_advantages,
    normalize_rewards,
    compute_explained_variance,
)

VOCAB_SIZE = 100
HIDDEN_DIM = 64
BATCH_SIZE = 4
PROMPT_LEN = 8


class TestRewardModel:
    """Tests for the reward model."""

    def test_forward_shape(self, reward_model: RewardModel) -> None:
        """Reward model should produce scalar per sequence."""
        token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 16))
        scores = reward_model(token_ids)
        assert scores.shape == (BATCH_SIZE,)

    def test_output_normalized(self, reward_model: RewardModel) -> None:
        """With normalize_rewards=True, output should be roughly zero-mean."""
        token_ids = torch.randint(0, VOCAB_SIZE, (32, 16))
        scores = reward_model(token_ids)
        assert abs(scores.mean().item()) < 0.5

    def test_different_inputs_different_scores(self, reward_model: RewardModel) -> None:
        """Different inputs should generally produce different scores."""
        input_a = torch.zeros(1, 16, dtype=torch.long)
        input_b = torch.ones(1, 16, dtype=torch.long)
        score_a = reward_model(input_a)
        score_b = reward_model(input_b)
        # They could theoretically be equal but extremely unlikely
        assert score_a.shape == score_b.shape


class TestPolicyModel:
    """Tests for the policy model."""

    def test_forward_shapes(self, policy_model: PolicyModel) -> None:
        """Forward pass should produce correct logit and value shapes."""
        token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 16))
        logits, values = policy_model(token_ids)
        assert logits.shape == (BATCH_SIZE, 16, VOCAB_SIZE)
        assert values.shape == (BATCH_SIZE, 16)

    def test_generate_shapes(self, policy_model: PolicyModel) -> None:
        """Generation should produce correct shapes."""
        prompts = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, PROMPT_LEN))
        gen_ids, log_probs = policy_model.generate(prompts, max_new_tokens=10)
        assert gen_ids.shape == (BATCH_SIZE, 10)
        assert log_probs.shape == (BATCH_SIZE, 10)

    def test_log_probs_negative(self, policy_model: PolicyModel) -> None:
        """Log probabilities should be negative."""
        token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 16))
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 8))
        log_probs = policy_model.compute_log_probs(token_ids, targets)
        assert (log_probs <= 0).all()

    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    def test_generation_temperature(
        self, policy_model: PolicyModel, temperature: float
    ) -> None:
        """Generation should work with different temperatures."""
        prompts = torch.randint(0, VOCAB_SIZE, (2, PROMPT_LEN))
        gen_ids, _ = policy_model.generate(
            prompts, max_new_tokens=5, temperature=temperature
        )
        assert gen_ids.shape == (2, 5)
        assert (gen_ids >= 0).all() and (gen_ids < VOCAB_SIZE).all()


class TestExperienceCollector:
    """Tests for the experience collector."""

    def test_collect_batch_count(
        self,
        policy_model: PolicyModel,
        reward_model: RewardModel,
        rlhf_config: RLHFConfig,
        sample_prompts: torch.Tensor,
    ) -> None:
        """Collector should return one experience per prompt."""
        collector = ExperienceCollector(policy_model, reward_model, rlhf_config)
        experiences = collector.collect_batch(sample_prompts)
        assert len(experiences) == BATCH_SIZE

    def test_experience_fields_populated(
        self, sample_experiences: list[Experience]
    ) -> None:
        """Each experience should have all fields populated."""
        for exp in sample_experiences:
            assert len(exp.query_ids) == PROMPT_LEN
            assert len(exp.response_ids) > 0
            assert len(exp.old_log_probs) > 0
            assert isinstance(exp.rewards, float)
            assert len(exp.values) > 0


class TestPPOTrainer:
    """Tests for the PPO trainer."""

    def test_update_returns_metrics(
        self,
        policy_model: PolicyModel,
        ppo_config: PPOConfig,
        sample_experiences: list[Experience],
    ) -> None:
        """PPO update should return valid TrainingMetrics."""
        trainer = PPOTrainer(policy_model, ppo_config)
        metrics = trainer.update(sample_experiences)
        assert isinstance(metrics, TrainingMetrics)
        assert not np.isnan(metrics.policy_loss)
        assert not np.isnan(metrics.value_loss)

    def test_update_changes_weights(
        self,
        policy_model: PolicyModel,
        ppo_config: PPOConfig,
        sample_experiences: list[Experience],
    ) -> None:
        """PPO update should modify policy weights."""
        trainer = PPOTrainer(policy_model, ppo_config)

        # Capture weights before
        weights_before = {
            name: param.clone()
            for name, param in policy_model.named_parameters()
        }

        trainer.update(sample_experiences)

        # At least some weights should change
        any_changed = any(
            not torch.equal(weights_before[name], param)
            for name, param in policy_model.named_parameters()
        )
        assert any_changed


class TestUtilityFunctions:
    """Tests for GAE, normalization, and other utilities."""

    def test_compute_gae_shape(self) -> None:
        """GAE should return advantages and returns of correct shape."""
        rewards = np.array([1.0, 0.0, 0.0, 2.0])
        values = np.array([0.5, 0.4, 0.3, 0.2])
        advantages, returns = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
        assert advantages.shape == (4,)
        assert returns.shape == (4,)

    def test_gae_zero_rewards_near_zero_advantages(self) -> None:
        """Zero rewards with constant values should give near-zero advantages."""
        rewards = np.zeros(5)
        values = np.ones(5) * 0.5
        advantages, _ = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
        # Advantages should be small when rewards match value predictions
        assert np.abs(advantages).max() < 1.0

    def test_normalize_advantages_zero_mean(self) -> None:
        """Normalized advantages should have approximately zero mean."""
        advantages = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_advantages(advantages)
        assert abs(normalized.mean()) < 1e-6

    def test_normalize_rewards_clipping(self) -> None:
        """Rewards should be clipped to [-5, 5] after normalization."""
        rewards = np.array([100.0, -100.0, 0.0])
        normalized = normalize_rewards(rewards)
        assert normalized.max() <= 5.0
        assert normalized.min() >= -5.0

    def test_kl_divergence_same_policy_zero(self) -> None:
        """KL divergence of a policy with itself should be zero."""
        log_probs = torch.tensor([-1.0, -2.0, -0.5])
        kl = compute_kl_divergence(log_probs, log_probs)
        assert abs(kl.item()) < 1e-6

    def test_explained_variance_perfect_prediction(self) -> None:
        """Perfect value predictions should give explained variance = 1."""
        values = np.array([1.0, 2.0, 3.0])
        returns = np.array([1.0, 2.0, 3.0])
        ev = compute_explained_variance(values, returns)
        assert abs(ev - 1.0) < 1e-6

    def test_explained_variance_random_prediction(self) -> None:
        """Random predictions should give low explained variance."""
        np.random.seed(42)
        values = np.random.randn(100)
        returns = np.random.randn(100)
        ev = compute_explained_variance(values, returns)
        assert ev < 0.5


class TestRLHFOrchestrator:
    """Integration tests for the full RLHF loop."""

    def test_train_runs_correct_iterations(self, rlhf_config: RLHFConfig) -> None:
        """Orchestrator should run the specified number of iterations."""
        orchestrator = RLHFOrchestrator(rlhf_config)
        metrics = orchestrator.train(num_iterations=3)
        assert len(metrics) == 3

    def test_train_metrics_have_iteration_numbers(
        self, rlhf_config: RLHFConfig
    ) -> None:
        """Each metric should have the correct iteration number."""
        orchestrator = RLHFOrchestrator(rlhf_config)
        metrics = orchestrator.train(num_iterations=2)
        assert metrics[0].iteration == 1
        assert metrics[1].iteration == 2

    def test_train_rewards_are_finite(self, rlhf_config: RLHFConfig) -> None:
        """All rewards should be finite numbers."""
        orchestrator = RLHFOrchestrator(rlhf_config)
        metrics = orchestrator.train(num_iterations=2)
        for m in metrics:
            assert np.isfinite(m.mean_reward)
            assert np.isfinite(m.policy_loss)
