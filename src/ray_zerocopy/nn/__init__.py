"""
Neural network module for zero-copy model loading with regular PyTorch models.

This module provides utilities for:
- Loading models in Ray actors (actors.py)
- Executing models via Ray tasks (tasks.py)
- Core tensor operations (rewrite.py)
- High-level utility classes (utils.py)
"""

from .actors import (
    load_model_in_actor,
    load_pipeline_in_actor,
    prepare_model_for_actors,
    rewrite_pipeline_for_actors,
)
from .rewrite import extract_tensors, replace_tensors
from .tasks import call_model, rewrite_pipeline
from .utils import ZeroCopyModel

__all__ = [
    # Actor-based functions
    "prepare_model_for_actors",
    "load_model_in_actor",
    "rewrite_pipeline_for_actors",
    "load_pipeline_in_actor",
    # Task-based functions
    "rewrite_pipeline",
    "call_model",
    # Core rewrite functions
    "extract_tensors",
    "replace_tensors",
    # Utility classes
    "ZeroCopyModel",
]
