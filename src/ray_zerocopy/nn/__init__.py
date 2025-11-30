"""
Neural network module for zero-copy model loading with regular PyTorch models.

This module provides utilities for:
- Loading models in Ray actors (actors.py)
- Core tensor operations (rewrite.py)
- High-level utility classes (utils.py)
"""

from .actors import load_pipeline_for_actors
from .rewrite import model_info_to_model_refs, prepare_pipeline
from .tasks import load_pipeline_for_tasks, rewrite_pipeline

__all__ = [
    # Actor-based functions
    "load_pipeline_for_actors",
    # Task-based functions
    "load_pipeline_for_tasks",
    "rewrite_pipeline",  # Convenience function
    # Generic pipeline preparation
    "prepare_pipeline",
    "model_info_to_model_refs",
]
