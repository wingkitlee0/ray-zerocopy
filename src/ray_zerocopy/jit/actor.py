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

"""
Functions for zero-copy TorchScript model loading in Ray actors.

This module provides actor-based support for TorchScript models, similar to
ray_zerocopy.actor but specifically for torch.jit.ScriptModule models.
"""

import copy
import warnings
from typing import Any, Optional

import ray
import torch

from .rewrite import extract_tensors, replace_tensors


def prepare_jit_model_for_actors(jit_model: torch.jit.ScriptModule) -> ray.ObjectRef:
    """
    Prepare a TorchScript model for zero-copy loading across multiple Ray actors.
    
    This function extracts the model structure and weights, storing them in Ray's
    object store. Multiple actors can then load the same model without duplicating
    memory.
    
    Args:
        jit_model: TorchScript model (torch.jit.ScriptModule) to prepare
    
    Returns:
        Ray ObjectRef containing the serialized model and weights
    
    Example:
        >>> import torch
        >>> 
        >>> # Trace your model
        >>> model = MyModel()
        >>> example = torch.randn(1, 3, 224, 224)
        >>> jit_model = torch.jit.trace(model, example)
        >>> 
        >>> # Prepare for actors
        >>> model_ref = prepare_jit_model_for_actors(jit_model)
        >>> 
        >>> class InferenceActor:
        ...     def __init__(self, model_ref):
        ...         self.model = load_jit_model_in_actor(model_ref)
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
    if not isinstance(jit_model, torch.jit.ScriptModule):
        raise TypeError(
            f"Expected torch.jit.ScriptModule, got {type(jit_model)}. "
            "Use ray_zerocopy.prepare_model_for_actors() for regular nn.Module models."
        )
    return ray.put(extract_tensors(jit_model))


def load_jit_model_in_actor(
    model_ref: ray.ObjectRef, device: Optional[str] = None
) -> torch.jit.ScriptModule:
    """
    Load a TorchScript model inside a Ray actor from the object store using zero-copy.
    
    This function reconstructs a TorchScript model from the reference created by
    prepare_jit_model_for_actors(). The model weights are loaded via zero-copy
    from Ray's object store.
    
    Args:
        model_ref: ObjectRef from prepare_jit_model_for_actors()
        device: Device to move the model to (e.g., "cuda:0", "cpu").
               If None, model stays on CPU
    
    Returns:
        Reconstructed TorchScript model ready for inference
    
    Example:
        >>> # Inside an actor's __init__
        >>> def __init__(self, model_ref):
        ...     self.model = load_jit_model_in_actor(model_ref, device="cuda:0")
    """
    # Suppress PyTorch warnings about immutable tensors
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
    
    # Get the model bytes and weights from the object store (zero-copy)
    model_bytes, model_weights = ray.get(model_ref)
    
    # Reconstruct the model
    model = replace_tensors(model_bytes, model_weights)
    
    # Move to specified device if needed
    if device is not None:
        model = model.to(device)
    
    # Ensure model is in eval mode
    model.eval()
    
    return model


def rewrite_pipeline_for_actors(
    pipeline: Any,
    model_attr_names: Optional[list] = None,
    device: Optional[str] = None,
) -> tuple[Any, dict[str, ray.ObjectRef]]:
    """
    Prepare a pipeline object with TorchScript models for use in Ray actors.
    
    This function extracts all TorchScript models from a pipeline object and stores
    them in Ray's object store. It returns a skeleton and model references that can
    be used to reconstruct the pipeline in each actor.
    
    Unlike jit.invoke.rewrite_pipeline(), this does NOT create nested remote tasks.
    Instead, models are loaded directly into each actor's memory space using
    zero-copy from the object store.
    
    Args:
        pipeline: Pipeline object containing TorchScript models as attributes
        model_attr_names: List of attribute names that are TorchScript models.
                         If None, auto-discovers all torch.jit.ScriptModule attributes
        device: Device to load models on in actors (e.g., "cuda:0")
    
    Returns:
        Tuple of (pipeline_skeleton, model_refs_dict)
    
    Example:
        >>> import torch
        >>> 
        >>> # Create pipeline with TorchScript models
        >>> class Pipeline:
        ...     def __init__(self):
        ...         encoder = EncoderModel()
        ...         decoder = DecoderModel()
        ...         self.encoder = torch.jit.trace(encoder, example_input)
        ...         self.decoder = torch.jit.trace(decoder, example_encoded)
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
    # Auto-discover TorchScript model attributes if not specified
    if model_attr_names is None:
        model_attr_names = [
            name
            for name in dir(pipeline)
            if not name.startswith("_")
            and isinstance(getattr(pipeline, name), torch.jit.ScriptModule)
        ]
    
    # Create a shallow copy of the pipeline
    pipeline_skeleton = copy.copy(pipeline)
    
    # Extract and store each TorchScript model in the object store
    model_refs = {}
    for attr_name in model_attr_names:
        model = getattr(pipeline, attr_name)
        if isinstance(model, torch.jit.ScriptModule):
            model_refs[attr_name] = prepare_jit_model_for_actors(model)
            # Set the attribute to None in skeleton to save memory
            setattr(pipeline_skeleton, attr_name, None)
    
    return pipeline_skeleton, model_refs


def load_pipeline_in_actor(
    pipeline_skeleton: Any,
    model_refs: dict[str, ray.ObjectRef],
    device: Optional[str] = None,
) -> Any:
    """
    Reconstruct a pipeline with TorchScript models inside a Ray actor.
    
    This function should be called in an actor's __init__ method to load
    the TorchScript models from the object store using zero-copy.
    
    Args:
        pipeline_skeleton: Pipeline skeleton from rewrite_pipeline_for_actors()
        model_refs: Model references dict from rewrite_pipeline_for_actors()
        device: Device to load models on (e.g., "cuda:0")
    
    Returns:
        Pipeline object with TorchScript models loaded and ready for inference
    
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
    
    # Load each TorchScript model from the object store
    for attr_name, model_ref in model_refs.items():
        model = load_jit_model_in_actor(model_ref, device=device)
        setattr(pipeline, attr_name, model)
    
    return pipeline
