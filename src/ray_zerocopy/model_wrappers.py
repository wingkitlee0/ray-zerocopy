"""
Improved wrapper API with clearer serialization/deserialization patterns.

This module provides an improved API design with:
- Clearer naming: from_model/to_actor/unwrap instead of __init__/load
- ModelWrapper that handles both standalone nn.Module and Pipeline objects
- Consistent patterns across all wrapper types
- Better separation of concerns between preparation and loading

The wrapper classes:
- ModelWrapper: Handles both nn.Module and Pipeline objects for actor usage
- TaskWrapper: For task-based execution (unchanged, for compatibility)
- ActorWrapper: Improved version with better API (backward compatible)
- JITModelWrapper: JIT version of ModelWrapper
- JITActorWrapper: Improved JIT actor wrapper

Key API Patterns:

1. **Serialization/Creation Pattern**:
   - Use `from_model()` class method to create wrapper from model/pipeline
   - This makes it clear we're creating a wrapper FROM a source

2. **Deserialization Pattern**:
   - Use `unwrap()` to get the actual model (inside actor)
   - This makes it clear we're unwrapping/deserializing

3. **Actor Preparation Pattern**:
   - Use `to_actor()` to get serialized form for actor constructor
   - This makes it clear we're preparing for actor usage

Usage Examples:

    # 1. ModelWrapper - Flexible wrapper for single models or pipelines
    >>> from ray_zerocopy import ModelWrapper
    >>>
    >>> # Option A: Wrap a standalone nn.Module
    >>> model = YourModel()
    >>> wrapper = ModelWrapper.from_model(model)
    >>>
    >>> # Option B: Wrap a Pipeline with multiple models
    >>> class Pipeline:
    ...     def __init__(self):
    ...         self.encoder = EncoderModel()
    ...         self.decoder = DecoderModel()
    ...     def __call__(self, x):
    ...         return self.decoder(self.encoder(x))
    >>>
    >>> pipeline = Pipeline()
    >>> wrapper = ModelWrapper.from_model(pipeline)
    >>>
    >>> # Use in Ray actors
    >>> class InferenceActor:
    ...     def __init__(self, model_wrapper):
    ...         self.model = model_wrapper.unwrap(device="cuda:0")
    ...     def __call__(self, batch):
    ...         return self.model(batch["data"])
    >>>
    >>> ds.map_batches(
    ...     InferenceActor,
    ...     fn_constructor_kwargs={"model_wrapper": wrapper},
    ...     compute=ActorPoolStrategy(size=4)
    ... )

    # 2. Alternative: More explicit actor preparation
    >>> wrapper = ModelWrapper.from_model(model)
    >>> actor_config = wrapper.to_actor()  # Get config dict for actor
    >>>
    >>> class InferenceActor:
    ...     def __init__(self, actor_config):
    ...         self.model = ModelWrapper.from_actor(**actor_config).unwrap(device="cuda:0")
    >>>
    >>> ds.map_batches(
    ...     InferenceActor,
    ...     fn_constructor_kwargs={"actor_config": actor_config},
    ...     compute=ActorPoolStrategy(size=4)
    ... )

    # 3. Backward compatible ActorWrapper (improved API)
    >>> from ray_zerocopy import ActorWrapper
    >>>
    >>> # Old way (still works)
    >>> wrapper = ActorWrapper(pipeline)
    >>> loaded = wrapper.load(device="cuda:0")
    >>>
    >>> # New way (clearer)
    >>> wrapper = ActorWrapper.from_model(pipeline)
    >>> loaded = wrapper.unwrap(device="cuda:0")
"""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar, Union

import ray
import torch

from ray_zerocopy import nn as rzc_nn
from ray_zerocopy._internal import WrapperMixin

T = TypeVar("T")


class _ModuleContainer:
    """Internal container to make standalone nn.Module look like a pipeline object."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def get_model(self) -> torch.nn.Module:
        """Get the model from the container."""
        return self.model


class ModelWrapper(WrapperMixin[T], Generic[T]):
    """
    A serializable wrapper with zero-copy loading for nn.Module and Pipeline objects.

    Args:
        skeleton: The skeleton of the model or pipeline
        model_refs: The model references for the model or pipeline
        is_standalone_module: Whether the model or pipeline is a standalone module

    Example - Standalone Model:
        >>> model = YourModel()
        >>> wrapper = ModelWrapper.from_model(model)
        >>>
        >>> class InferenceActor:
        ...     def __init__(self, model_wrapper):
        ...         self.model = model_wrapper.unwrap(device="cuda:0")
        ...     def __call__(self, batch):
        ...         return self.model(batch["data"])
        >>>
        >>> ds.map_batches(
        ...     InferenceActor,
        ...     fn_constructor_kwargs={"model_wrapper": wrapper},
        ...     compute=ActorPoolStrategy(size=4)
        ... )

    Example - Pipeline:
        >>> class Pipeline:
        ...     def __init__(self):
        ...         self.encoder = EncoderModel()
        ...         self.decoder = DecoderModel()
        ...     def __call__(self, x):
        ...         return self.decoder(self.encoder(x))
        >>>
        >>> pipeline = Pipeline()
        >>> wrapper = ModelWrapper.from_model(pipeline)
        >>> # Use same way as standalone model above
    """

    _skeleton: T
    _model_refs: dict[str, ray.ObjectRef]
    _is_standalone_module: bool

    def __init__(
        self,
        skeleton: T,
        model_refs: dict[str, ray.ObjectRef],
        is_standalone_module: bool = False,
    ):
        """
        Initialize ModelWrapper.

        Args:
            skeleton: The skeleton of the model or pipeline
            model_refs: The model references
            is_standalone_module: Whether this is a standalone module
        """
        self._skeleton = skeleton
        self._model_refs = model_refs
        self._is_standalone_module = is_standalone_module

    @classmethod
    def from_model(
        cls, model_or_pipeline: T, model_attr_names: Optional[list] = None
    ) -> "ModelWrapper[T]":
        """Instantiate a ModelWrapper from a model or pipeline (a class containing nn.Module attributes)

        Args:
            model_or_pipeline: The model or pipeline to wrap
            model_attr_names: The attribute names of the models in the pipeline

        Returns:
            A ModelWrapper instance
        """

        is_standalone = isinstance(model_or_pipeline, torch.nn.Module)

        _pipeline: Union[_ModuleContainer, T] = (
            _ModuleContainer(model_or_pipeline) if is_standalone else model_or_pipeline
        )

        skeleton, model_refs = rzc_nn.prepare_pipeline_for_actors(
            _pipeline,
            model_attr_names=model_attr_names,
        )

        _wrapper = cls(skeleton, model_refs, is_standalone)
        # Use skeleton instead of original pipeline to avoid capturing model reference
        _wrapper._configure_wrapper(skeleton)  # type: ignore[arg-type]

        return _wrapper

    def to_pipeline(
        self, device: Optional[str] = None, _use_fast_load: bool = False
    ) -> torch.nn.Module | T:
        """Unwrap the ModelWrapper to a pipeline.

        This function is to be called from within an actor's __init__ to deserialize and
        load the model from Ray's object store using zero-copy.

        Args:
            device: Device to move models to (e.g., "cuda:0", "cpu").
                If None, models remain on CPU. Defaults to None.
            _use_fast_load: Use faster but slightly riskier loading method. Defaults to False.

        Returns:
            The deserialized pipeline ready for inference

        Example:
            >>> class InferenceActor:
            ...     def __init__(self, model_wrapper):
            ...         # Unwrap and load on GPU
            ...         self.model = model_wrapper.to_pipeline(device="cuda:0")
            ...
            ...     def __call__(self, batch):
            ...         return self.model(batch["data"])
        """
        pipeline = rzc_nn.load_pipeline_for_actors(
            self._skeleton,
            self._model_refs,
            device=device,
            use_fast_load=_use_fast_load,
        )

        if self._is_standalone_module:
            loaded_container: _ModuleContainer = pipeline  # type: ignore[assignment]
            return loaded_container.get_model()
        else:
            return pipeline

    def __getstate__(self):
        """Return state for pickling."""
        return {
            "_skeleton": self._skeleton,
            "_model_refs": self._model_refs,
            "_is_standalone_module": self._is_standalone_module,
        }

    def __setstate__(self, state):
        """Restore state from pickling."""
        self._skeleton = state["_skeleton"]
        self._model_refs = state["_model_refs"]
        self._is_standalone_module = state.get("_is_standalone_module", False)

    @classmethod
    def deserialize(
        cls,
        skeleton: T,
        model_refs: dict[str, ray.ObjectRef],
        is_standalone_module: bool = False,
    ) -> "ModelWrapper[T]":
        """Deserialize a ModelWrapper from a skeleton and model references."""
        return cls(skeleton, model_refs, is_standalone_module)

    def serialize(self) -> dict[str, Any]:
        """Serialize the ModelWrapper to a dictionary."""
        return {
            "skeleton": self._skeleton,
            "model_refs": self._model_refs,
            "is_standalone_module": self._is_standalone_module,
        }
