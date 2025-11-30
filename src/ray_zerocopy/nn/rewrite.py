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
    Prepare a Pipeline instance that contains PyTorch modules for zero-copy model loading.

    This function extracts all PyTorch modules (`nn.Modules`) from a pipeline and stores
    them in Ray's object store. To load and reconstruct the pipeline, use `load_pipeline_for_tasks` and
    `load_pipeline_for_actors` for task and actor mode respectively.

    Args:
        pipeline: Pipeline object containing PyTorch models as attributes

        model_attr_names: Explicit list of attribute names to treat as models.
            - If provided: Only these attributes will be extracted and prepared
            - If None (default): Auto-discovers all torch.nn.Module attributes
            Use explicit names when you want precise control over which models to prepare,
            or when auto-discovery picks up unwanted modules.

        method_names: Names of model methods to expose via remote tasks.
            - If None: No method tracking (actor mode - actors expose all methods)
            - If provided: Track only these methods (task mode - for _RemoteModelShim)
            Example: ("__call__", "forward", "generate")

        filter_private: Whether to exclude underscore-prefixed attributes during
            auto-discovery (only applies when model_attr_names is None).
            - False (default for tasks): Include all models, even those starting with '_'.
              This ensures all models are available regardless of naming convention.
            - True (typical for actors): Skip attributes starting with '_' to avoid
              exposing internal/private models as remote methods.

    Returns:
        Tuple of (pipeline_skeleton, model_info_dict) where model_info_dict
        maps attribute names to (model_ref, allowed_methods) tuples.

        allowed_methods specifies which methods can be called remotely:
        - In task mode: Set[str] containing method names that are allowed to be
          invoked remotely via _RemoteModelShim. This acts as an allowlist to
          restrict which model methods can be called as Ray tasks, ensuring only
          intended methods are remotely callable.
        - In actor mode: None (actors expose all methods by default)

    Example - Task mode (include all models):
        >>> skeleton, model_info = prepare_pipeline(
        ...     pipeline,
        ...     method_names=("__call__", "forward"),
        ...     filter_private=False
        ... )

    Example - Actor mode (exclude private models):
        >>> skeleton, model_info = prepare_pipeline(
        ...     pipeline,
        ...     method_names=None,
        ...     filter_private=True
        ... )

    Example - Explicit model selection:
        >>> skeleton, model_info = prepare_pipeline(
        ...     pipeline,
        ...     model_attr_names=["encoder", "decoder"],  # Only these two
        ...     method_names=("__call__",)
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
            allowed_methods = None
            if method_names is not None:
                allowed_methods = {m for m in method_names if hasattr(model, m)}
                # Always include __call__ if the model is callable
                if hasattr(model, "__call__") and callable(model):
                    allowed_methods.add("__call__")

            model_info[attr_name] = (model_ref, allowed_methods)
            # Set the attribute to None in skeleton to save memory
            setattr(pipeline_skeleton, attr_name, None)

    return pipeline_skeleton, model_info


def model_info_to_model_refs(
    model_info: dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]],
) -> dict[str, ray.ObjectRef]:
    """
    Convert model_info dict to model_refs dict by extracting just the object references.

    Args:
        model_info: Dictionary mapping attribute names to (model_ref, allowed_methods) tuples

    Returns:
        Dictionary mapping attribute names to model_refs (object references only)

    Example:
        >>> skeleton, model_info = prepare_pipeline(pipeline, method_names=None)
        >>> model_refs = model_info_to_model_refs(model_info)
    """
    return {attr_name: model_ref for attr_name, (model_ref, _) in model_info.items()}
