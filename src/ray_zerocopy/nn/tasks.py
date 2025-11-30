import copy
from typing import Optional, Set, TypeVar

import ray

from ray_zerocopy._internal.zerocopy import _RemoteModelShim
from ray_zerocopy.nn.rewrite import prepare_pipeline

T = TypeVar("T")


def load_pipeline_for_tasks(
    pipeline_skeleton: T,
    model_info: dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]],
) -> T:
    """
    Load a pipeline for task-based execution by creating remote model shims.

    Only the methods (`method_names`) specified during the
    `prepare_pipeline` call will be exposed via remote tasks.

    Args:
        pipeline_skeleton: Pipeline skeleton from prepare_pipeline
        model_info: Model info dict from prepare_pipeline (with allowed_methods)

    Returns:
        Pipeline with models replaced by task-based shims

    Example:
        >>> # Prepare pipeline for tasks
        >>> skeleton, model_info = prepare_pipeline(
        ...     pipeline,
        ...     method_names=("__call__", "forward"),
        ...     filter_private=False
        ... )
        >>>
        >>> # Load for task execution
        >>> task_pipeline = load_pipeline_for_tasks(skeleton, model_info)
        >>> result = task_pipeline.process(data)  # Models run in Ray tasks
    """

    # Create a shallow copy and replace models with shims
    result = copy.copy(pipeline_skeleton)
    for attr_name, (model_ref, allowed_methods) in model_info.items():
        if allowed_methods is not None:  # Only create shims if methods were tracked
            shim = _RemoteModelShim(model_ref, allowed_methods)
            setattr(result, attr_name, shim)

    return result


def rewrite_pipeline(pipeline: T, method_names: tuple = ("__call__",)) -> T:
    """
    Convenience function that combines prepare_pipeline and load_pipeline_for_tasks.

    Args:
        pipeline: Pipeline object containing PyTorch models as attributes
        method_names: Names of model methods to expose via remote tasks

    Returns:
        Pipeline with models replaced by task-based shims

    Example:
        >>> # Simple one-step API
        >>> rewritten = rewrite_pipeline(pipeline)
        >>> result = rewritten.process(data)  # Models run in Ray tasks

        >>> # Equivalent to:
        >>> skeleton, model_info = prepare_pipeline(
        ...     pipeline, method_names=("__call__",), filter_private=False
        ... )
        >>> rewritten = load_pipeline_for_tasks(skeleton, model_info)
    """
    skeleton, model_info = prepare_pipeline(
        pipeline, method_names=method_names, filter_private=False
    )
    return load_pipeline_for_tasks(skeleton, model_info)
