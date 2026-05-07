"""Distributed RLHF Trainer — Minimal, readable RLHF with separation of concerns."""

__version__ = "0.1.0"

from distributed_rlhf_trainer.core import (
    ExperienceCollector,
    PolicyModel,
    PPOTrainer,
    RewardModel,
    RLHFOrchestrator,
)
from distributed_rlhf_trainer.models import (
    Experience,
    PPOConfig,
    RewardModelConfig,
    RLHFConfig,
    TrainingMetrics,
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
