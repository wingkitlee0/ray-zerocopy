from .actor import (
    load_model_in_actor,
    load_pipeline_in_actor,
    prepare_model_for_actors,
    rewrite_pipeline_for_actors,
)
from .invoke import call_model, rewrite_pipeline
from .public import ZeroCopyModel
from .rewrite import extract_tensors, replace_tensors
from .wrappers import ActorWrapper, JITActorWrapper, JITTaskWrapper, TaskWrapper

__all__ = [
    # New unified wrapper API (recommended)
    "TaskWrapper",
    "ActorWrapper",
    "JITTaskWrapper",
    "JITActorWrapper",
    # Original task-based functions (deprecated)
    "rewrite_pipeline",
    "call_model",
    "extract_tensors",
    "replace_tensors",
    "ZeroCopyModel",
    # Actor-based functions (deprecated)
    "prepare_model_for_actors",
    "load_model_in_actor",
    "rewrite_pipeline_for_actors",
    "load_pipeline_in_actor",
]
