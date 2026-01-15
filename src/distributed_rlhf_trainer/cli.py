"""Command-line interface for the distributed RLHF trainer."""

from __future__ import annotations

import argparse
import logging
import sys

from distributed_rlhf_trainer.core import RLHFOrchestrator
from distributed_rlhf_trainer.models import RLHFConfig
from distributed_rlhf_trainer.utils import format_metrics

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Distributed RLHF Trainer — minimal, readable RLHF with separation of concerns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size per iteration"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=10, help="Number of RLHF iterations"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=1000, help="Vocabulary size"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=32, help="Max generated sequence length"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="PPO learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the RLHF training loop from CLI arguments.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = RLHFConfig(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )
    config.ppo.learning_rate = args.learning_rate

    logger.info("Starting RLHF training with config: %s", config.model_dump())

    orchestrator = RLHFOrchestrator(config)
    metrics_history = orchestrator.train()

    if metrics_history:
        logger.info("Final metrics: %s", format_metrics(metrics_history[-1]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
