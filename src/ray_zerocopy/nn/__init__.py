"""
Neural network module for zero-copy model loading with regular PyTorch models.

This module provides utilities for:
- Loading models in Ray actors (actors.py)
- Core tensor operations (rewrite.py)
- High-level utility classes (utils.py)
"""

from .actors import (
    load_pipeline_for_actors,
    prepare_pipeline_for_actors,
)
from .rewrite import load_pipeline_for_tasks, prepare_pipeline, rewrite_pipeline
from .utils import ZeroCopyModel

__all__ = [
    # Actor-based functions
    "prepare_pipeline_for_actors",
    "load_pipeline_for_actors",
    # Task-based functions
    "load_pipeline_for_tasks",
    "rewrite_pipeline",  # Convenience function
    # Generic pipeline preparation
    "prepare_pipeline",
    # Utility classes
    "ZeroCopyModel",
]
