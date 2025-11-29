#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Core generic pipeline preparation logic.

This module provides the generic prepare_pipeline() function that can be
customized for both task-based and actor-based execution patterns.
"""

import copy
from typing import Optional, Set, TypeVar

import ray
import torch

from ray_zerocopy._internal.zerocopy import extract_tensors

T = TypeVar("T")


def prepare_pipeline(
    pipeline: T,
    model_attr_names: Optional[list] = None,
    method_names: Optional[tuple] = None,
    filter_private: bool = False,
) -> tuple[T, dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]]]:
    """
    Generic pipeline preparation for both task-based and actor-based execution.

    This function extracts all PyTorch models from a pipeline object and stores
    them in Ray's object store. Returns a skeleton and model information dict
    that can be used to reconstruct the pipeline for either tasks or actors.

    :param pipeline: Pipeline object containing PyTorch models as attributes
    :param model_attr_names: List of attribute names that are models. If None,
                             auto-discovers all torch.nn.Module attributes
    :param method_names: Names of model methods to expose via remote tasks.
                        If None, no method tracking (actor mode).
                        If provided, tracks methods (task mode).
    :param filter_private: If True, exclude underscore-prefixed attributes
                          during auto-discovery. Typically True for actors,
                          False for tasks.
    :returns: Tuple of (pipeline_skeleton, model_info_dict) where model_info_dict
              maps attribute names to (model_ref, valid_methods) tuples.
              valid_methods is None in actor mode, Set[str] in task mode.

    Example - Task mode:
        >>> skeleton, model_info = prepare_pipeline(
        ...     pipeline,
        ...     method_names=("__call__", "forward"),
        ...     filter_private=False
        ... )

    Example - Actor mode:
        >>> skeleton, model_info = prepare_pipeline(
        ...     pipeline,
        ...     method_names=None,
        ...     filter_private=True
        ... )
    """
    # Auto-discover model attributes if not specified
    if model_attr_names is None:
        model_attr_names = [
            name
            for name in dir(pipeline)
            if (not filter_private or not name.startswith("_"))
            and isinstance(getattr(pipeline, name), torch.nn.Module)
        ]

    # Create a shallow copy of the pipeline
    pipeline_skeleton = copy.copy(pipeline)

    # Extract and store each model in the object store
    model_info = {}
    for attr_name in model_attr_names:
        model = getattr(pipeline, attr_name)
        if isinstance(model, torch.nn.Module):
            # Store extracted tensors in object store
            model_ref = ray.put(extract_tensors(model))

            # Track methods only if method_names provided (task mode)
            valid_methods = None
            if method_names is not None:
                valid_methods = {m for m in method_names if hasattr(model, m)}
                # Always include __call__ if the model is callable
                if hasattr(model, "__call__") and callable(model):
                    valid_methods.add("__call__")

            model_info[attr_name] = (model_ref, valid_methods)
            # Set the attribute to None in skeleton to save memory
            setattr(pipeline_skeleton, attr_name, None)

    return pipeline_skeleton, model_info


def load_pipeline_for_tasks(
    pipeline_skeleton: T,
    model_info: dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]],
) -> T:
    """
    Load a pipeline for task-based execution by creating remote model shims.

    This function takes the output from prepare_pipeline (with method_names specified)
    and creates a pipeline where model calls are executed as Ray tasks.

    :param pipeline_skeleton: Pipeline skeleton from prepare_pipeline
    :param model_info: Model info dict from prepare_pipeline (with valid_methods)
    :returns: Pipeline with models replaced by task-based shims

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
    import copy

    from ray_zerocopy._internal.zerocopy import _RemoteModelShim

    # Create a shallow copy and replace models with shims
    result = copy.copy(pipeline_skeleton)
    for attr_name, (model_ref, valid_methods) in model_info.items():
        if valid_methods is not None:  # Only create shims if methods were tracked
            shim = _RemoteModelShim(model_ref, valid_methods)
            setattr(result, attr_name, shim)

    return result


def rewrite_pipeline(pipeline: T, method_names: tuple = ("__call__",)) -> T:
    """
    Convenience function that combines prepare_pipeline and load_pipeline_for_tasks.

    This is a simplified API for task-based execution that matches the original
    IBM interface. For more control, use prepare_pipeline + load_pipeline_for_tasks
    separately.

    :param pipeline: Pipeline object containing PyTorch models as attributes
    :param method_names: Names of model methods to expose via remote tasks
    :returns: Pipeline with models replaced by task-based shims

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


__all__ = [
    "prepare_pipeline",
    "load_pipeline_for_tasks",
    "rewrite_pipeline",
]
