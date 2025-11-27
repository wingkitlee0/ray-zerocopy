from .actor import (
    load_model_in_actor,
    load_pipeline_in_actor,
    prepare_model_for_actors,
    rewrite_pipeline_for_actors,
)
from .invoke import call_model, rewrite_pipeline
from .public import ZeroCopyModel
from .rewrite import extract_tensors, replace_tensors

__all__ = [
    # Original task-based functions
    "rewrite_pipeline",
    "call_model",
    "extract_tensors",
    "replace_tensors",
    "ZeroCopyModel",
    # New actor-based functions
    "prepare_model_for_actors",
    "load_model_in_actor",
    "rewrite_pipeline_for_actors",
    "load_pipeline_in_actor",
]
