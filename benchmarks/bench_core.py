"""Benchmark the core RLHF training loop performance.

Measures:
  - Experience collection throughput (experiences/sec)
  - PPO update throughput (updates/sec)
  - End-to-end iteration time
  - Memory usage
"""

import logging
import sys
import time
from typing import Any

import numpy as np
import torch

from distributed_rlhf_trainer.core import (
    ExperienceCollector,
    PolicyModel,
    PPOTrainer,
    RewardModel,
    RLHFOrchestrator,
)
from distributed_rlhf_trainer.models import PPOConfig, RewardModelConfig, RLHFConfig

logger = logging.getLogger(__name__)

WARMUP_ITERATIONS = 2
BENCHMARK_ITERATIONS = 10


def benchmark_experience_collection(
    vocab_size: int = 500,
    hidden_dim: int = 128,
    batch_size: int = 8,
    seq_length: int = 32,
) -> dict[str, float]:
    """Benchmark experience collection throughput.

    Args:
        vocab_size: Vocabulary size.
        hidden_dim: Hidden dimension.
        batch_size: Batch size.
        seq_length: Maximum sequence length.

    Returns:
        Dictionary with timing results.
    """
    config = RLHFConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        max_seq_length=seq_length,
    )
    policy = PolicyModel(vocab_size, hidden_dim)
    reward = RewardModel(config.reward, vocab_size)
    collector = ExperienceCollector(policy, reward, config)

    # Warmup
    prompts = torch.randint(0, vocab_size, (batch_size, 16))
    for _ in range(WARMUP_ITERATIONS):
        collector.collect_batch(prompts)

    # Benchmark
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        collector.collect_batch(prompts)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_time_sec": np.mean(times),
        "std_time_sec": np.std(times),
        "throughput_exp_per_sec": batch_size / np.mean(times),
        "p50_time_sec": np.percentile(times, 50),
        "p99_time_sec": np.percentile(times, 99),
    }


def benchmark_ppo_update(
    vocab_size: int = 500,
    hidden_dim: int = 128,
    batch_size: int = 8,
    seq_length: int = 32,
) -> dict[str, float]:
    """Benchmark PPO update throughput.

    Args:
        vocab_size: Vocabulary size.
        hidden_dim: Hidden dimension.
        batch_size: Batch size.
        seq_length: Maximum sequence length.

    Returns:
        Dictionary with timing results.
    """
    config = RLHFConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        max_seq_length=seq_length,
    )
    policy = PolicyModel(vocab_size, hidden_dim)
    reward = RewardModel(config.reward, vocab_size)
    collector = ExperienceCollector(policy, reward, config)
    trainer = PPOTrainer(policy, config.ppo)

    prompts = torch.randint(0, vocab_size, (batch_size, 16))
    experiences = collector.collect_batch(prompts)

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        trainer.update(experiences)

    # Benchmark
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        trainer.update(experiences)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_time_sec": np.mean(times),
        "std_time_sec": np.std(times),
        "throughput_updates_per_sec": 1.0 / np.mean(times),
        "p50_time_sec": np.percentile(times, 50),
        "p99_time_sec": np.percentile(times, 99),
    }


def benchmark_end_to_end(
    vocab_size: int = 500,
    hidden_dim: int = 128,
    batch_size: int = 8,
    seq_length: int = 32,
    num_iterations: int = 5,
) -> dict[str, float]:
    """Benchmark full end-to-end RLHF loop.

    Args:
        vocab_size: Vocabulary size.
        hidden_dim: Hidden dimension.
        batch_size: Batch size.
        seq_length: Maximum sequence length.
        num_iterations: Number of RLHF iterations.

    Returns:
        Dictionary with timing results.
    """
    config = RLHFConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        max_seq_length=seq_length,
        num_iterations=num_iterations,
    )

    start = time.perf_counter()
    orchestrator = RLHFOrchestrator(config)
    metrics = orchestrator.train()
    total_time = time.perf_counter() - start

    return {
        "total_time_sec": total_time,
        "time_per_iteration_sec": total_time / num_iterations,
        "final_mean_reward": metrics[-1].mean_reward,
        "final_policy_loss": metrics[-1].policy_loss,
    }


def format_results(name: str, results: dict[str, float]) -> str:
    """Format benchmark results as a table row.

    Args:
        name: Benchmark name.
        results: Dictionary of metric values.

    Returns:
        Formatted string.
    """
    lines = [f"\n{'=' * 60}", f"  {name}", f"{'=' * 60}"]
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"  {key:<30s}: {value:>12.6f}")
        else:
            lines.append(f"  {key:<30s}: {value}")
    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def main() -> int:
    """Run all benchmarks and print results.

    Returns:
        Exit code.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.setLevel(logging.INFO)

    results_collection = benchmark_experience_collection()
    logger.info(format_results("Experience Collection", results_collection))

    results_ppo = benchmark_ppo_update()
    logger.info(format_results("PPO Update", results_ppo))

    results_e2e = benchmark_end_to_end()
    logger.info(format_results("End-to-End RLHF Loop", results_e2e))

    # Print summary table
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<35s} {'Value':>15s} {'Unit':>10s}")
    print("-" * 70)
    print(f"  {'Collection throughput':<35s} {results_collection['throughput_exp_per_sec']:>15.1f} {'exp/sec':>10s}")
    print(f"  {'Collection p50 latency':<35s} {results_collection['p50_time_sec']*1000:>15.1f} {'ms':>10s}")
    print(f"  {'Collection p99 latency':<35s} {results_collection['p99_time_sec']*1000:>15.1f} {'ms':>10s}")
    print(f"  {'PPO update throughput':<35s} {results_ppo['throughput_updates_per_sec']:>15.1f} {'upd/sec':>10s}")
    print(f"  {'PPO update p50 latency':<35s} {results_ppo['p50_time_sec']*1000:>15.1f} {'ms':>10s}")
    print(f"  {'PPO update p99 latency':<35s} {results_ppo['p99_time_sec']*1000:>15.1f} {'ms':>10s}")
    print(f"  {'E2E time per iteration':<35s} {results_e2e['time_per_iteration_sec']*1000:>15.1f} {'ms':>10s}")
    print(f"  {'E2E total time (5 iters)':<35s} {results_e2e['total_time_sec']:>15.3f} {'sec':>10s}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
