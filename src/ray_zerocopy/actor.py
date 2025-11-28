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
from typing import Any, Optional

import ray
import torch

from ray_zerocopy.rewrite import extract_tensors, replace_tensors


def prepare_model_for_actors(model: torch.nn.Module) -> ray.ObjectRef:
    """
    Prepare a PyTorch model for zero-copy loading across multiple Ray actors.

    .. deprecated::
        This function is deprecated. Use :class:`ray_zerocopy.ActorWrapper` instead:
        
        Old API::
        
            model_ref = prepare_model_for_actors(model)
            class InferenceActor:
                def __init__(self, model_ref):
                    self.model = load_model_in_actor(model_ref)
        
        New API::
        
            from ray_zerocopy import ActorWrapper
            actor_wrapper = ActorWrapper(pipeline)
            class InferenceActor:
                def __init__(self, actor_wrapper):
                    self.pipeline = actor_wrapper.load()

    This function extracts the model weights and stores them in Ray's object store,
    enabling multiple actors to load the same model without duplicating memory.

    :param model: PyTorch model to prepare
    :returns: Ray ObjectRef containing the model skeleton and weights

    Example:
        >>> model = YourModel()
        >>> model_ref = prepare_model_for_actors(model)
        >>>
        >>> class InferenceActor:
        ...     def __init__(self, model_ref):
        ...         self.model = load_model_in_actor(model_ref)
        ...
        ...     def __call__(self, batch):
        ...         return self.model(batch["data"])
        >>>
        >>> # Use with Ray Data
        >>> ds.map_batches(
        ...     InferenceActor,
        ...     fn_constructor_kwargs={"model_ref": model_ref},
        ...     compute=ActorPoolStrategy(size=4)
        ... )
    """
    warnings.warn(
        "prepare_model_for_actors() is deprecated. Use ActorWrapper instead: "
        "from ray_zerocopy import ActorWrapper; wrapper = ActorWrapper(pipeline)",
        DeprecationWarning,
        stacklevel=2
    )
    return ray.put(extract_tensors(model))


def load_model_in_actor(
    model_ref: ray.ObjectRef, device: Optional[str] = None, use_fast_load: bool = False
) -> torch.nn.Module:
    """
    Load a model inside a Ray actor from the object store using zero-copy.

    This function reconstructs the model from the reference created by
    :func:`prepare_model_for_actors`. The model weights are loaded via zero-copy
    from Ray's object store (Plasma).

    :param model_ref: ObjectRef from :func:`prepare_model_for_actors`
    :param device: Device to move the model to (e.g., "cuda:0", "cpu").
                   If None, model stays on CPU
    :param use_fast_load: If True, use the faster but slightly riskier
                          replace_tensors_direct. Default False.
    :returns: Reconstructed PyTorch model ready for inference

    Example:
        >>> # Inside an actor's __init__
        >>> def __init__(self, model_ref):
        ...     self.model = load_model_in_actor(model_ref, device="cuda:0")
    """
    # Suppress PyTorch warnings about immutable tensors
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

    # Get the model skeleton and weights from the object store (zero-copy)
    model_skeleton, model_weights = ray.get(model_ref)

    # Reconstruct the model
    if use_fast_load:
        from ray_zerocopy.rewrite import replace_tensors_direct

        replace_tensors_direct(model_skeleton, model_weights)
    else:
        replace_tensors(model_skeleton, model_weights)

    # Move to specified device if needed
    if device is not None:
        model_skeleton = model_skeleton.to(device)

    # Ensure model is in eval mode
    model_skeleton.eval()

    return model_skeleton


def rewrite_pipeline_for_actors(
    pipeline: Any,
    model_attr_names: Optional[list] = None,
    device: Optional[str] = None,
    use_fast_load: bool = False,
) -> tuple[Any, dict[str, ray.ObjectRef]]:
    """
    Prepare a pipeline object with PyTorch models for use in Ray actors.

    .. deprecated::
        This function is deprecated. Use :class:`ray_zerocopy.ActorWrapper` instead:
        
        Old API::
        
            skeleton, model_refs = rewrite_pipeline_for_actors(pipeline)
            class InferenceActor:
                def __init__(self, skeleton, model_refs):
                    self.pipeline = load_pipeline_in_actor(skeleton, model_refs)
        
        New API::
        
            from ray_zerocopy import ActorWrapper
            actor_wrapper = ActorWrapper(pipeline)
            class InferenceActor:
                def __init__(self, actor_wrapper):
                    self.pipeline = actor_wrapper.load()

    This function extracts all PyTorch models from a pipeline object and stores
    them in Ray's object store. It returns a factory function that can reconstruct
    the pipeline with loaded models inside each actor.

    Unlike :func:`rewrite_pipeline` from invoke.py, this does NOT create nested
    remote tasks. Instead, models are loaded directly into each actor's memory
    space using zero-copy from the object store.

    :param pipeline: Pipeline object containing PyTorch models as attributes
    :param model_attr_names: List of attribute names that are models. If None,
                             auto-discovers all torch.nn.Module attributes
    :param device: Device to load models on in actors (e.g., "cuda:0")
    :param use_fast_load: Whether to use fast loading method
    :returns: Tuple of (pipeline_skeleton, model_refs_dict)

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
        >>> pipeline_skeleton, model_refs = rewrite_pipeline_for_actors(pipeline)
        >>>
        >>> class InferenceActor:
        ...     def __init__(self, pipeline_skeleton, model_refs):
        ...         self.pipeline = load_pipeline_in_actor(
        ...             pipeline_skeleton, model_refs
        ...         )
        ...
        ...     def __call__(self, batch):
        ...         return self.pipeline(batch["data"])
        >>>
        >>> # Use with Ray Data
        >>> ds.map_batches(
        ...     InferenceActor,
        ...     fn_constructor_kwargs={
        ...         "pipeline_skeleton": pipeline_skeleton,
        ...         "model_refs": model_refs
        ...     },
        ...     compute=ActorPoolStrategy(size=4)
        ... )
    """
    warnings.warn(
        "rewrite_pipeline_for_actors() is deprecated. Use ActorWrapper instead: "
        "from ray_zerocopy import ActorWrapper; wrapper = ActorWrapper(pipeline)",
        DeprecationWarning,
        stacklevel=2
    )
    # Auto-discover model attributes if not specified
    if model_attr_names is None:
        model_attr_names = [
            name
            for name in dir(pipeline)
            if not name.startswith("_")
            and isinstance(getattr(pipeline, name), torch.nn.Module)
        ]

    # Create a shallow copy of the pipeline
    pipeline_skeleton = copy.copy(pipeline)

    # Extract and store each model in the object store
    model_refs = {}
    for attr_name in model_attr_names:
        model = getattr(pipeline, attr_name)
        if isinstance(model, torch.nn.Module):
            model_refs[attr_name] = prepare_model_for_actors(model)
            # Set the attribute to None in skeleton to save memory
            setattr(pipeline_skeleton, attr_name, None)

    return pipeline_skeleton, model_refs


def load_pipeline_in_actor(
    pipeline_skeleton: Any,
    model_refs: dict[str, ray.ObjectRef],
    device: Optional[str] = None,
    use_fast_load: bool = False,
) -> Any:
    """
    Reconstruct a pipeline with its models inside a Ray actor.

    This function should be called in an actor's __init__ method to load
    the models from the object store using zero-copy.

    :param pipeline_skeleton: Pipeline skeleton from :func:`rewrite_pipeline_for_actors`
    :param model_refs: Model references dict from :func:`rewrite_pipeline_for_actors`
    :param device: Device to load models on (e.g., "cuda:0")
    :param use_fast_load: Whether to use fast loading method
    :returns: Pipeline object with models loaded and ready for inference

    Example:
        >>> # Inside actor's __init__
        >>> def __init__(self, pipeline_skeleton, model_refs):
        ...     self.pipeline = load_pipeline_in_actor(
        ...         pipeline_skeleton,
        ...         model_refs,
        ...         device="cuda:0"
        ...     )
    """
    # Create a copy of the skeleton
    pipeline = copy.copy(pipeline_skeleton)

    # Load each model from the object store
    for attr_name, model_ref in model_refs.items():
        model = load_model_in_actor(
            model_ref, device=device, use_fast_load=use_fast_load
        )
        setattr(pipeline, attr_name, model)

    return pipeline


# Note: No factory function needed!
# Just define your actor class and use fn_constructor_kwargs in map_batches.
# Example:
#
# class MyActor:
#     def __init__(self, model_ref, device="cuda:0"):
#         self.model = load_model_in_actor(model_ref, device=device)
#
#     def __call__(self, batch):
#         return self.model(batch["data"])
#
# ds.map_batches(
#     MyActor,
#     fn_constructor_kwargs={"model_ref": model_ref, "device": "cuda:0"},
#     compute=ActorPoolStrategy(size=4)
# )
