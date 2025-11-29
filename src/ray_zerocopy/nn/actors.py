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
Functions for zero-copy model loading in Ray actors, designed for use
with Ray Data map_batches and ActorPoolStrategy.
"""

import copy
import warnings
from typing import Optional, TypeVar

import ray

from ray_zerocopy._internal.zerocopy import (
    replace_tensors,
    replace_tensors_direct,
)

from .rewrite import prepare_pipeline

T = TypeVar("T")


def prepare_pipeline_for_actors(
    pipeline: T,
    model_attr_names: Optional[list] = None,
) -> tuple[T, dict[str, ray.ObjectRef]]:
    """
    Prepare a pipeline object with PyTorch models for use in Ray actors.

    This function extracts all PyTorch models from a pipeline object and stores
    them in Ray's object store. Returns a skeleton and model references dict
    that can be used to reconstruct the pipeline inside actors.

    Args:
        pipeline: Pipeline object containing PyTorch models as attributes
        model_attr_names: List of attribute names that are models. If None,
            auto-discovers all torch.nn.Module attributes

    Returns:
        Tuple of (pipeline_skeleton, model_refs_dict)

    Example:
        >>> class Pipeline:
        ...     def __init__(self):
        ...         self.encoder = EncoderModel()
        ...         self.decoder = DecoderModel()
        ...
        ...     def __call__(self, data):
        ...         encoded = self.encoder(data)
        ...         return self.decoder(encoded)
        >>>
        >>> # Prepare pipeline for actors
        >>> pipeline = Pipeline()
        >>> skeleton, model_refs = prepare_pipeline_for_actors(pipeline)
        >>>
        >>> class InferenceActor:
        ...     def __init__(self, skeleton, model_refs):
        ...         self.pipeline = load_pipeline_for_actors(skeleton, model_refs)
        ...
        ...     def __call__(self, batch):
        ...         return self.pipeline(batch["data"])
    """
    skeleton, model_info = prepare_pipeline(
        pipeline,
        model_attr_names=model_attr_names,
        method_names=None,  # No method tracking for actors
        filter_private=True,
    )
    # Convert to simpler dict format for actors (no method tracking needed)
    return skeleton, {k: ref for k, (ref, _) in model_info.items()}


def load_pipeline_for_actors(
    pipeline_skeleton: T,
    model_refs: dict[str, ray.ObjectRef],
    use_fast_load: bool = False,
) -> T:
    """
    Reconstruct a pipeline with its models inside a Ray actor.

    This function should be called in an actor's __init__ method to load
    the models from the object store using zero-copy.

    Args:
        pipeline_skeleton: Pipeline skeleton from prepare_pipeline_for_actors()
        model_refs: Model references dict from prepare_pipeline_for_actors()
        use_fast_load: Whether to use faster but slightly riskier loading method.
            If True, uses replace_tensors_direct. Default False.

    Returns:
        Pipeline object with models loaded and ready for inference

    Example:
        >>> # Inside actor's __init__
        >>> def __init__(self, skeleton, model_refs):
        ...     self.pipeline = load_pipeline_for_actors(
        ...         skeleton,
        ...         model_refs,
        ...     )
    """
    # Suppress PyTorch warnings about immutable tensors
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

    # Create a copy of the skeleton
    pipeline = copy.copy(pipeline_skeleton)

    # Load each model from the object store
    for attr_name, model_ref in model_refs.items():
        # Get the model skeleton and weights from the object store (zero-copy)
        model_skeleton, model_weights = ray.get(model_ref)

        # Reconstruct the model
        if use_fast_load:
            replace_tensors_direct(model_skeleton, model_weights)
        else:
            replace_tensors(model_skeleton, model_weights)

        # Ensure model is in eval mode
        model_skeleton.eval()

        setattr(pipeline, attr_name, model_skeleton)

    return pipeline
