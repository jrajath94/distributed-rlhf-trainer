"""Quickstart example: Run a minimal RLHF training loop.

This demonstrates the full RLHF pipeline with a tiny model.
In production, replace PolicyModel with a pre-trained LLM wrapper.
"""

import logging
import sys

from distributed_rlhf_trainer.core import RLHFOrchestrator
from distributed_rlhf_trainer.models import RLHFConfig
from distributed_rlhf_trainer.utils import format_metrics


def main() -> None:
    """Run a minimal RLHF training loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config = RLHFConfig(
        batch_size=4,
        max_seq_length=16,
        num_iterations=5,
        vocab_size=500,
        hidden_dim=64,
        seed=42,
    )

    orchestrator = RLHFOrchestrator(config)
    metrics_history = orchestrator.train()

    logging.info("=" * 60)
    logging.info("Training Summary")
    logging.info("=" * 60)
    for m in metrics_history:
        logging.info(format_metrics(m))
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
