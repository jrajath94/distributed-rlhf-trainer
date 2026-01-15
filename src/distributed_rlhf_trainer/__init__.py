"""Distributed RLHF Trainer — Minimal, readable RLHF with separation of concerns."""

__version__ = "0.1.0"

from distributed_rlhf_trainer.models import (
    RLHFConfig,
    PPOConfig,
    RewardModelConfig,
    Experience,
    TrainingMetrics,
)
from distributed_rlhf_trainer.core import (
    RewardModel,
    PolicyModel,
    PPOTrainer,
    ExperienceCollector,
    RLHFOrchestrator,
)

__all__ = [
    "RLHFConfig",
    "PPOConfig",
    "RewardModelConfig",
    "Experience",
    "TrainingMetrics",
    "RewardModel",
    "PolicyModel",
    "PPOTrainer",
    "ExperienceCollector",
    "RLHFOrchestrator",
]
