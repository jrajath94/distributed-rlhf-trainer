"""Core RLHF components with clean separation of concerns.

The RLHF loop is decomposed into four independent components:
  1. RewardModel — scores (query, response) pairs
  2. PolicyModel — generates responses and provides log-probs
  3. ExperienceCollector — gathers rollouts into Experience tuples
  4. PPOTrainer — performs policy gradient updates
  5. RLHFOrchestrator — wires everything together

Each component has a single responsibility and can be tested, replaced,
or distributed independently. This is the key architectural decision that
differentiates this project from monolithic RLHF implementations.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from distributed_rlhf_trainer.exceptions import (
    ExperienceCollectionError,
    PolicyUpdateError,
    RewardModelError,
)
from distributed_rlhf_trainer.models import (
    Experience,
    PPOConfig,
    RewardModelConfig,
    RLHFConfig,
    TrainingMetrics,
)
from distributed_rlhf_trainer.utils import (
    compute_entropy,
    compute_explained_variance,
    compute_gae,
    compute_kl_divergence,
    format_metrics,
    normalize_advantages,
    normalize_rewards,
    set_seed,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

TEMPERATURE_DEFAULT = 1.0
TOP_K_DEFAULT = 50


class RewardModel(nn.Module):
    """Scores (query, response) pairs with a scalar reward.

    A lightweight MLP-based reward model that takes token embeddings,
    processes them through transformer-style layers, and outputs a
    scalar reward. In production, this would be a fine-tuned LM head;
    here we use a clean, testable architecture.

    Args:
        config: Reward model configuration.
        vocab_size: Size of the token vocabulary.
    """

    def __init__(self, config: RewardModelConfig, vocab_size: int) -> None:
        super().__init__()
        self._config = config
        self._embedding = nn.Embedding(vocab_size, config.hidden_dim)

        layers: list[nn.Module] = []
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
        self._backbone = nn.Sequential(*layers)
        self._head = nn.Linear(config.hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute reward score for a batch of sequences.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len).

        Returns:
            Scalar rewards of shape (batch_size,).

        Raises:
            RewardModelError: If forward pass produces NaN values.
        """
        embeddings = self._embedding(token_ids)
        # Mean-pool over sequence dimension
        pooled = embeddings.mean(dim=1)
        features = self._backbone(pooled)
        scores = self._head(features).squeeze(-1)

        if torch.isnan(scores).any():
            raise RewardModelError("Reward model produced NaN scores")

        if self._config.normalize_rewards:
            scores = (scores - scores.mean()) / (scores.std() + 1e-8)

        return scores


class PolicyModel(nn.Module):
    """Language model policy that generates responses and computes log-probs.

    Simplified policy network for demonstration. In production, this would
    wrap a pre-trained LLM. The architecture mirrors the key interfaces
    needed: generate(), log_probs(), and value().

    Args:
        vocab_size: Size of the token vocabulary.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._embedding = nn.Embedding(vocab_size, hidden_dim)
        self._transformer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self._lm_head = nn.Linear(hidden_dim, vocab_size)
        # Separate value head — standard actor-critic architecture
        self._value_head = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and values for input tokens.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len).

        Returns:
            Tuple of (logits, values) where logits has shape
            (batch_size, seq_len, vocab_size) and values has shape
            (batch_size, seq_len).
        """
        embeddings = self._embedding(token_ids)
        hidden = self._transformer(embeddings)
        logits = self._lm_head(hidden)
        values = self._value_head(hidden).squeeze(-1)
        return logits, values

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = TEMPERATURE_DEFAULT,
        top_k: int = TOP_K_DEFAULT,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate response tokens.

        Args:
            prompt_ids: Prompt token IDs of shape (batch_size, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Number of top tokens to consider for sampling.

        Returns:
            Tuple of (generated_ids, log_probs) where generated_ids has
            shape (batch_size, max_new_tokens) and log_probs has the same shape.
        """
        generated_ids = []
        log_probs_list = []
        current_input = prompt_ids

        for _ in range(max_new_tokens):
            logits, _ = self.forward(current_input)
            # Take logits for the last position only
            next_logits = logits[:, -1, :] / temperature
            next_logits = self._top_k_filter(next_logits, top_k)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_log_prob = torch.log(
                probs.gather(1, next_token) + 1e-10
            )

            generated_ids.append(next_token)
            log_probs_list.append(token_log_prob)
            current_input = torch.cat([current_input, next_token], dim=1)

        generated = torch.cat(generated_ids, dim=1)
        log_probs = torch.cat(log_probs_list, dim=1)
        return generated, log_probs

    def compute_log_probs(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities of target tokens given context.

        Args:
            token_ids: Full sequence token IDs (batch_size, seq_len).
            target_ids: Target token IDs to score (batch_size, target_len).

        Returns:
            Log probabilities of shape (batch_size, target_len).
        """
        logits, _ = self.forward(token_ids)
        # Align logits to target positions
        target_len = target_ids.shape[1]
        relevant_logits = logits[:, -target_len:, :]
        log_probs = F.log_softmax(relevant_logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Zero out logits below the top-k threshold.

        Args:
            logits: Raw logits of shape (batch_size, vocab_size).
            top_k: Number of top values to keep.

        Returns:
            Filtered logits with non-top-k values set to -inf.
        """
        top_k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits[logits < threshold] = float("-inf")
        return logits


class ExperienceCollector:
    """Collects rollout experiences by running the policy and reward model.

    This component handles the "data collection" phase of RLHF:
    generate responses with the current policy, score them with the
    reward model, and package everything as Experience tuples.

    Args:
        policy: The current policy model.
        reward_model: The trained reward model.
        config: RLHF configuration.
    """

    def __init__(
        self,
        policy: PolicyModel,
        reward_model: RewardModel,
        config: RLHFConfig,
    ) -> None:
        self._policy = policy
        self._reward_model = reward_model
        self._config = config

    @torch.no_grad()
    def collect_batch(
        self,
        prompts: torch.Tensor,
    ) -> list[Experience]:
        """Collect a batch of experiences from the current policy.

        Args:
            prompts: Batch of prompt token IDs (batch_size, prompt_len).

        Returns:
            List of Experience objects, one per prompt.

        Raises:
            ExperienceCollectionError: If generation or scoring fails.
        """
        try:
            # Set models to inference mode
            self._policy.training = False
            self._reward_model.training = False

            generated_ids, old_log_probs = self._policy.generate(
                prompts,
                max_new_tokens=self._config.max_seq_length,
            )

            full_sequences = torch.cat([prompts, generated_ids], dim=1)
            rewards = self._reward_model(full_sequences)

            _, values = self._policy.forward(full_sequences)
            response_values = values[:, prompts.shape[1]:]

            experiences = self._build_experiences(
                prompts, generated_ids, old_log_probs, rewards, response_values
            )

            logger.debug(
                "Collected %d experiences, mean reward: %.4f",
                len(experiences),
                rewards.mean().item(),
            )
            return experiences

        except Exception as exc:
            raise ExperienceCollectionError(
                f"Experience collection failed: {exc}"
            ) from exc

    def _build_experiences(
        self,
        prompts: torch.Tensor,
        generated_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> list[Experience]:
        """Convert tensors to a list of Experience dataclasses.

        Args:
            prompts: Prompt token IDs.
            generated_ids: Generated response token IDs.
            old_log_probs: Log probs under the generating policy.
            rewards: Scalar rewards for each sequence.
            values: Value estimates for response positions.

        Returns:
            List of Experience objects.
        """
        batch_size = prompts.shape[0]
        experiences = []

        for i in range(batch_size):
            exp = Experience(
                query_ids=prompts[i].cpu().tolist(),
                response_ids=generated_ids[i].cpu().tolist(),
                old_log_probs=old_log_probs[i].cpu().tolist(),
                rewards=rewards[i].item(),
                values=values[i].cpu().tolist(),
            )
            experiences.append(exp)

        return experiences


class PPOTrainer:
    """Performs PPO policy gradient updates.

    Implements the clipped PPO objective with value function loss,
    entropy bonus, and KL penalty. This is the core optimization
    component — it takes Experience tuples and updates the policy.

    Args:
        policy: The policy model to update.
        config: PPO configuration.
    """

    def __init__(self, policy: PolicyModel, config: PPOConfig) -> None:
        self._policy = policy
        self._config = config
        self._optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
        )

    def update(self, experiences: list[Experience]) -> TrainingMetrics:
        """Perform PPO update on a batch of experiences.

        Args:
            experiences: List of Experience objects from the collector.

        Returns:
            TrainingMetrics with aggregated loss and diagnostic values.

        Raises:
            PolicyUpdateError: If the update produces NaN losses.
        """
        self._policy.train()
        metrics = self._initialize_metrics()

        # Convert experiences to tensors
        batch = self._prepare_batch(experiences)

        for epoch in range(self._config.ppo_epochs):
            epoch_metrics = self._ppo_step(batch)
            self._accumulate_metrics(metrics, epoch_metrics)

            # Early stopping on KL divergence
            if epoch_metrics["approx_kl"] > 0.02:
                logger.info(
                    "Early stopping at epoch %d: KL=%.4f > 0.02",
                    epoch,
                    epoch_metrics["approx_kl"],
                )
                break

        self._finalize_metrics(metrics, self._config.ppo_epochs)
        return metrics

    def _prepare_batch(
        self, experiences: list[Experience]
    ) -> dict[str, torch.Tensor]:
        """Convert Experience list to padded tensor batch.

        Args:
            experiences: List of Experience objects.

        Returns:
            Dictionary of tensors ready for the PPO step.
        """
        response_ids = torch.tensor(
            [exp.response_ids for exp in experiences], dtype=torch.long
        )
        query_ids = torch.tensor(
            [exp.query_ids for exp in experiences], dtype=torch.long
        )
        old_log_probs = torch.tensor(
            [exp.old_log_probs for exp in experiences], dtype=torch.float32
        )

        # Compute GAE for each experience
        all_advantages = []
        all_returns = []
        rewards_array = np.array([exp.rewards for exp in experiences])

        if len(experiences) > 0:
            normalized_rewards = normalize_rewards(rewards_array)
        else:
            normalized_rewards = rewards_array

        for i, exp in enumerate(experiences):
            values_np = np.array(exp.values, dtype=np.float64)
            # Spread the scalar reward across timesteps
            seq_len = len(exp.response_ids)
            per_step_rewards = np.zeros(seq_len, dtype=np.float64)
            per_step_rewards[-1] = float(normalized_rewards[i])

            advantages, returns = compute_gae(
                per_step_rewards,
                values_np,
                gamma=self._config.gamma,
                gae_lambda=self._config.gae_lambda,
            )
            all_advantages.append(advantages)
            all_returns.append(returns)

        advantages = torch.tensor(
            np.stack(all_advantages), dtype=torch.float32
        )
        returns = torch.tensor(
            np.stack(all_returns), dtype=torch.float32
        )

        # Normalize advantages across the batch
        adv_np = np.array(advantages.tolist(), dtype=np.float32)
        advantages = torch.tensor(
            normalize_advantages(adv_np), dtype=torch.float32
        )

        return {
            "query_ids": query_ids,
            "response_ids": response_ids,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
        }

    def _ppo_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Execute a single PPO optimization step.

        Args:
            batch: Dictionary of tensors from _prepare_batch.

        Returns:
            Dictionary of per-step metric values.

        Raises:
            PolicyUpdateError: If loss is NaN.
        """
        full_ids = torch.cat(
            [batch["query_ids"], batch["response_ids"]], dim=1
        )
        new_log_probs = self._policy.compute_log_probs(
            full_ids, batch["response_ids"]
        )
        _, values = self._policy.forward(full_ids)
        response_values = values[:, batch["query_ids"].shape[1]:]

        # Policy loss (clipped surrogate objective)
        ratio = torch.exp(new_log_probs - batch["old_log_probs"])
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self._config.clip_range,
            1.0 + self._config.clip_range,
        )
        policy_loss = -torch.min(
            ratio * batch["advantages"],
            clipped_ratio * batch["advantages"],
        ).mean()

        # Value loss
        value_loss = F.mse_loss(response_values, batch["returns"])

        # Entropy bonus
        logits, _ = self._policy.forward(full_ids)
        response_logits = logits[:, batch["query_ids"].shape[1]:, :]
        log_probs_full = F.log_softmax(response_logits, dim=-1)
        entropy = compute_entropy(log_probs_full)

        # KL penalty
        kl_div = compute_kl_divergence(new_log_probs, batch["old_log_probs"])

        # Combined loss
        total_loss = (
            policy_loss
            + self._config.value_coef * value_loss
            - self._config.entropy_coef * entropy
            + self._config.kl_penalty_coef * kl_div
        )

        if torch.isnan(total_loss):
            raise PolicyUpdateError("PPO update produced NaN loss")

        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self._policy.parameters(),
            self._config.max_grad_norm,
        )
        self._optimizer.step()

        # Diagnostics
        clip_fraction = (
            (torch.abs(ratio - 1.0) > self._config.clip_range)
            .float()
            .mean()
            .item()
        )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl_divergence": kl_div.item(),
            "clip_fraction": clip_fraction,
            "approx_kl": kl_div.item(),
        }

    @staticmethod
    def _initialize_metrics() -> TrainingMetrics:
        """Create a zeroed TrainingMetrics instance."""
        return TrainingMetrics()

    @staticmethod
    def _accumulate_metrics(
        metrics: TrainingMetrics,
        step_metrics: dict[str, float],
    ) -> None:
        """Add step metrics to running totals."""
        metrics.policy_loss += step_metrics["policy_loss"]
        metrics.value_loss += step_metrics["value_loss"]
        metrics.entropy += step_metrics["entropy"]
        metrics.kl_divergence += step_metrics["kl_divergence"]
        metrics.clip_fraction += step_metrics["clip_fraction"]
        metrics.approx_kl += step_metrics["approx_kl"]

    @staticmethod
    def _finalize_metrics(
        metrics: TrainingMetrics,
        num_epochs: int,
    ) -> None:
        """Average accumulated metrics over epochs."""
        metrics.policy_loss /= num_epochs
        metrics.value_loss /= num_epochs
        metrics.entropy /= num_epochs
        metrics.kl_divergence /= num_epochs
        metrics.clip_fraction /= num_epochs
        metrics.approx_kl /= num_epochs


class RLHFOrchestrator:
    """Orchestrates the full RLHF training loop.

    Wires together the four independent components: policy, reward model,
    experience collector, and PPO trainer. This is the only component
    that knows about all the others.

    Args:
        config: Full RLHF configuration.
    """

    def __init__(self, config: RLHFConfig) -> None:
        self._config = config
        set_seed(config.seed)

        self._policy = PolicyModel(config.vocab_size, config.hidden_dim)
        self._reward_model = RewardModel(config.reward, config.vocab_size)
        self._collector = ExperienceCollector(
            self._policy, self._reward_model, config
        )
        self._ppo_trainer = PPOTrainer(self._policy, config.ppo)
        self._metrics_history: list[TrainingMetrics] = []

        logger.info(
            "RLHF Orchestrator initialized: vocab=%d, hidden=%d, batch=%d",
            config.vocab_size,
            config.hidden_dim,
            config.batch_size,
        )

    @property
    def policy(self) -> PolicyModel:
        """Access the current policy model."""
        return self._policy

    @property
    def reward_model(self) -> RewardModel:
        """Access the reward model."""
        return self._reward_model

    @property
    def metrics_history(self) -> list[TrainingMetrics]:
        """Access full training metrics history."""
        return self._metrics_history

    def train(
        self,
        num_iterations: int | None = None,
    ) -> list[TrainingMetrics]:
        """Run the full RLHF training loop.

        Args:
            num_iterations: Override for number of iterations (uses config default).

        Returns:
            List of TrainingMetrics, one per iteration.
        """
        iterations = num_iterations or self._config.num_iterations
        logger.info("Starting RLHF training for %d iterations", iterations)

        for iteration in range(1, iterations + 1):
            metrics = self._train_step(iteration)
            self._metrics_history.append(metrics)

            if iteration % self._config.log_interval == 0:
                logger.info(format_metrics(metrics))

        logger.info(
            "Training complete. Final mean reward: %.4f",
            self._metrics_history[-1].mean_reward if self._metrics_history else 0.0,
        )
        return self._metrics_history

    def _train_step(self, iteration: int) -> TrainingMetrics:
        """Execute a single RLHF iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            TrainingMetrics for this iteration.
        """
        # Generate random prompts (in production: sample from dataset)
        prompts = torch.randint(
            0,
            self._config.vocab_size,
            (self._config.batch_size, 16),
        )

        # Collect experiences
        experiences = self._collector.collect_batch(prompts)

        # PPO update
        metrics = self._ppo_trainer.update(experiences)
        metrics.iteration = iteration
        metrics.mean_reward = np.mean([exp.rewards for exp in experiences])

        # Compute explained variance
        all_values = np.concatenate([np.array(exp.values) for exp in experiences])
        all_returns = np.concatenate(
            [np.array(exp.returns) if exp.returns else np.array(exp.values)
             for exp in experiences]
        )
        if len(all_values) == len(all_returns):
            metrics.explained_variance = compute_explained_variance(
                all_values, all_returns
            )

        return metrics
