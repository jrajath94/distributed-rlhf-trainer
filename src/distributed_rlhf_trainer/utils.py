"""Utility functions for RLHF training.

Provides GAE computation, reward normalization, KL divergence estimation,
and reproducibility helpers. All numerically-intensive operations use
numpy vectorization.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distributed_rlhf_trainer.exceptions import CheckpointError
from distributed_rlhf_trainer.models import Experience, TrainingMetrics

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a bias-variance tradeoff for advantage estimation.
    Lambda=0 gives TD(0) (low variance, high bias). Lambda=1 gives
    Monte Carlo (high variance, low bias).

    Args:
        rewards: Array of shape (T,) with per-step rewards.
        values: Array of shape (T+1,) with value estimates (includes bootstrap).
        gamma: Discount factor for future rewards.
        gae_lambda: Lambda parameter controlling bias-variance tradeoff.

    Returns:
        Tuple of (advantages, returns) arrays each of shape (T,).
    """
    seq_len = len(rewards)
    advantages = np.zeros(seq_len, dtype=np.float64)
    last_gae = 0.0

    # Reverse sweep — accumulate advantages from the end
    for t in reversed(range(seq_len)):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:seq_len]
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """Normalize advantages to zero mean and unit variance.

    Args:
        advantages: Raw advantage values of any shape.

    Returns:
        Normalized advantages with mean~0 and std~1.
    """
    std = np.std(advantages)
    if std < 1e-8:
        return advantages - np.mean(advantages)
    return (advantages - np.mean(advantages)) / std


def normalize_rewards(rewards: np.ndarray) -> np.ndarray:
    """Normalize rewards using running statistics.

    Args:
        rewards: Raw reward values.

    Returns:
        Normalized rewards clipped to [-5, 5].
    """
    std = np.std(rewards)
    if std < 1e-8:
        return np.clip(rewards - np.mean(rewards), -5.0, 5.0)
    normalized = (rewards - np.mean(rewards)) / std
    return np.clip(normalized, -5.0, 5.0)


def compute_kl_divergence(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
) -> torch.Tensor:
    """Compute approximate KL divergence between two policies.

    Uses the approximation: KL(old || new) ~ exp(log_new - log_old) - 1 - (log_new - log_old)

    Args:
        log_probs_new: Log probabilities under the updated policy.
        log_probs_old: Log probabilities under the old policy.

    Returns:
        Scalar tensor with mean KL divergence estimate.
    """
    log_ratio = log_probs_new - log_probs_old
    approx_kl = torch.mean(torch.exp(log_ratio) - 1 - log_ratio)
    return approx_kl


def compute_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Compute policy entropy from log probabilities.

    Args:
        log_probs: Log probabilities of shape (batch, seq_len, vocab).

    Returns:
        Scalar mean entropy.
    """
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return torch.mean(entropy)


def compute_explained_variance(
    values: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Compute explained variance of value function predictions.

    Returns 1 if value function perfectly predicts returns, 0 if no
    better than predicting the mean, negative if worse.

    Args:
        values: Predicted values.
        returns: Actual returns.

    Returns:
        Explained variance ratio in (-inf, 1].
    """
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_returns)


def format_metrics(metrics: TrainingMetrics) -> str:
    """Format training metrics as a human-readable log line.

    Args:
        metrics: Training metrics from a single iteration.

    Returns:
        Formatted string suitable for logging.
    """
    return (
        f"iter={metrics.iteration:>4d} | "
        f"reward={metrics.mean_reward:>8.4f} | "
        f"policy_loss={metrics.policy_loss:>8.4f} | "
        f"value_loss={metrics.value_loss:>8.4f} | "
        f"entropy={metrics.entropy:>6.4f} | "
        f"kl={metrics.kl_divergence:>6.4f} | "
        f"clip_frac={metrics.clip_fraction:>5.3f}"
    )


def save_checkpoint(
    state: dict[str, Any],
    path: Path,
    iteration: int,
) -> Path:
    """Save a training checkpoint to disk.

    Args:
        state: Dictionary containing model state dicts and optimizer state.
        path: Directory to save the checkpoint in.
        iteration: Current iteration number for filename.

    Returns:
        Path to the saved checkpoint file.

    Raises:
        CheckpointError: If the checkpoint cannot be saved.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = path / f"checkpoint_{iteration:06d}.pt"
        torch.save(state, checkpoint_path)
        logger.info("Checkpoint saved: %s", checkpoint_path)
        return checkpoint_path
    except OSError as exc:
        raise CheckpointError(f"Failed to save checkpoint at {path}: {exc}") from exc


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a training checkpoint from disk.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Dictionary containing model state dicts and optimizer state.

    Raises:
        CheckpointError: If the checkpoint cannot be loaded.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("Checkpoint loaded: %s", path)
        return state
    except (OSError, RuntimeError) as exc:
        raise CheckpointError(f"Failed to load checkpoint from {path}: {exc}") from exc
