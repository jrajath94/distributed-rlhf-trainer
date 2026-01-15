"""Custom exceptions for distributed RLHF trainer."""


class RLHFError(Exception):
    """Base exception for all RLHF-related errors."""


class ConfigValidationError(RLHFError):
    """Raised when configuration parameters are invalid."""


class RewardModelError(RLHFError):
    """Raised when reward model encounters an error during scoring."""


class PolicyUpdateError(RLHFError):
    """Raised when policy gradient update fails."""


class ExperienceCollectionError(RLHFError):
    """Raised when experience collection encounters issues."""


class CheckpointError(RLHFError):
    """Raised when checkpoint save/load operations fail."""


class DistributedError(RLHFError):
    """Raised when distributed communication fails."""
