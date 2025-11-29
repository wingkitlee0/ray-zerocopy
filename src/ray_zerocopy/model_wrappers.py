"""
Primary API for zero-copy model sharing with nn.Module models.

This module provides ModelWrapper, a unified wrapper that supports both task and actor
execution modes for zero-copy model sharing.

The wrapper classes:
- ModelWrapper: Handles both nn.Module and Pipeline objects for task and actor usage
- JITTaskWrapper/JITActorWrapper: For TorchScript (compiled) models (in wrappers.py)

Key API Patterns:

1. **Task Mode** - For ad-hoc inference with Ray tasks:
   - Use `ModelWrapper.for_tasks()` or `ModelWrapper.from_model(..., mode="task")`
   - Wrapper is immediately usable: `result = wrapped(data)`

2. **Actor Mode** - For Ray Data and long-running actors:
   - Use `ModelWrapper.from_model(..., mode="actor")`
   - Load in actor: `model = wrapper.load()`

Usage Examples:

    # 1. Task Mode - Immediate use
    >>> from ray_zerocopy import ModelWrapper
    >>> pipeline = MyPipeline()
    >>> wrapped = ModelWrapper.for_tasks(pipeline)
    >>> result = wrapped(data)  # Each call spawns a Ray task

    # 2. Actor Mode - For Ray Data
    >>> from ray_zerocopy import ModelWrapper
    >>> pipeline = MyPipeline()
    >>> wrapper = ModelWrapper.from_model(pipeline, mode="actor")
    >>>
    >>> class InferenceActor:
    ...     def __init__(self, model_wrapper):
    ...         self.model = model_wrapper.load()
    ...         self.model = self.model.to("cuda:0")  # Move to GPU if needed
    ...     def __call__(self, batch):
    ...         return self.model(batch["data"])
    >>>
    >>> ds.map_batches(
    ...     InferenceActor,
    ...     fn_constructor_kwargs={"model_wrapper": wrapper},
    ...     compute=ActorPoolStrategy(size=4)
    ... )

    # 3. Pipeline with multiple models
    >>> class Pipeline:
    ...     def __init__(self):
    ...         self.encoder = EncoderModel()
    ...         self.decoder = DecoderModel()
    ...     def __call__(self, x):
    ...         return self.decoder(self.encoder(x))
    >>>
    >>> pipeline = Pipeline()
    >>> wrapper = ModelWrapper.from_model(pipeline, mode="actor")
    >>> # All models are automatically detected and shared via zero-copy
"""

from __future__ import annotations

from typing import Any, Generic, Literal, Optional, Set, TypeVar, Union

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
    A unified serializable wrapper with zero-copy loading for nn.Module and Pipeline objects.

    Supports both task-based and actor-based execution modes:
    - Task mode: Models are executed via Ray tasks with zero-copy loading
    - Actor mode: Models are prepared for loading in Ray actors with zero-copy

    Args:
        skeleton: The skeleton of the model or pipeline
        model_refs: The model references for the model or pipeline
        is_standalone_module: Whether the model or pipeline is a standalone module
        mode: Execution mode - "task" or "actor"
        rewritten: For task mode, the immediately-usable rewritten pipeline
        model_info: For task mode, model info with method tracking

    Example - Task Mode (immediate use):
        >>> from ray_zerocopy import ModelWrapper
        >>> model = YourModel()
        >>> wrapper = ModelWrapper.from_model(model, mode="task")
        >>> result = wrapper(data)  # Ready to use immediately

    Example - Task Mode Shortcut:
        >>> wrapper = ModelWrapper.for_tasks(model)
        >>> result = wrapper(data)  # Equivalent to rewrite_pipeline()

    Example - Actor Mode:
        >>> wrapper = ModelWrapper.from_model(model, mode="actor")
        >>>
        >>> class InferenceActor:
        ...     def __init__(self, model_wrapper):
        ...         self.model = model_wrapper.load()
        ...         self.model = self.model.to("cuda:0")  # Move to GPU if needed
        ...     def __call__(self, batch):
        ...         return self.model(batch["data"])
        >>>
        >>> ds.map_batches(
        ...     InferenceActor,
        ...     fn_constructor_kwargs={"model_wrapper": wrapper},
        ...     compute=ActorPoolStrategy(size=4)
        ... )
    """

    _skeleton: T
    _model_refs: dict[str, ray.ObjectRef]
    _is_standalone_module: bool
    _mode: Literal["task", "actor"]
    _rewritten: Optional[T]  # For task mode
    _model_info: Optional[
        dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]]
    ]  # For task mode

    def __init__(
        self,
        skeleton: T,
        model_refs: dict[str, ray.ObjectRef],
        is_standalone_module: bool = False,
        mode: Literal["task", "actor"] = "actor",
        rewritten: Optional[T] = None,
        model_info: Optional[
            dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]]
        ] = None,
    ):
        """
        Initialize ModelWrapper.

        Args:
            skeleton: The skeleton of the model or pipeline
            model_refs: The model references
            is_standalone_module: Whether this is a standalone module
            mode: Execution mode ("task" or "actor")
            rewritten: For task mode, the rewritten pipeline
            model_info: For task mode, model info with method tracking
        """
        self._skeleton = skeleton
        self._model_refs = model_refs
        self._is_standalone_module = is_standalone_module
        self._mode = mode
        self._rewritten = rewritten
        self._model_info = model_info

    @classmethod
    def from_model(
        cls,
        model_or_pipeline: T,
        mode: Literal["task", "actor"] = "actor",
        model_attr_names: Optional[list] = None,
        method_names: Optional[tuple] = None,
    ) -> "ModelWrapper[T]":
        """Instantiate a ModelWrapper from a model or pipeline.

        Args:
            model_or_pipeline: The model or pipeline to wrap
            mode: Execution mode - "task" for immediate use, "actor" for actor loading
            model_attr_names: The attribute names of the models in the pipeline
            method_names: Model methods to expose via remote tasks (auto-selected if None)

        Returns:
            A ModelWrapper instance

        Example - Task mode:
            >>> wrapper = ModelWrapper.from_model(pipeline, mode="task")
            >>> result = wrapper.process(data)  # Ready to use immediately

        Example - Actor mode:
            >>> wrapper = ModelWrapper.from_model(pipeline, mode="actor")
            >>> # In actor: pipeline = wrapper.load()
        """
        is_standalone = isinstance(model_or_pipeline, torch.nn.Module)

        _pipeline: Union[_ModuleContainer, T] = (
            _ModuleContainer(model_or_pipeline) if is_standalone else model_or_pipeline
        )

        # Auto-select method_names if None
        if method_names is None:
            if mode == "task":
                method_names = ("__call__",)
            else:
                method_names = None  # No method tracking in actor mode

        if mode == "task":
            # Task mode: prepare + load immediately
            skeleton, model_info = rzc_nn.prepare_pipeline(
                _pipeline,
                method_names=method_names,
                filter_private=False,
            )

            # Load immediately for task execution
            rewritten = rzc_nn.load_pipeline_for_tasks(skeleton, model_info)

            # Extract simplified model_refs for pickling
            model_refs = {
                attr_name: model_ref for attr_name, (model_ref, _) in model_info.items()
            }

            _wrapper = cls(
                skeleton,
                model_refs,
                is_standalone,
                mode="task",
                rewritten=rewritten,
                model_info=model_info,
            )
            # Use skeleton to avoid capturing model reference
            _wrapper._configure_wrapper(skeleton)  # type: ignore[arg-type]

            # Preserve call signature from original pipeline
            _wrapper._preserve_call_signature(_pipeline)  # type: ignore[arg-type]

        else:
            # Actor mode: prepare only (no loading)
            skeleton, model_refs = rzc_nn.prepare_pipeline_for_actors(
                _pipeline,
                model_attr_names=model_attr_names,
            )

            _wrapper = cls(
                skeleton,
                model_refs,
                is_standalone,
                mode="actor",
                rewritten=None,
                model_info=None,
            )
            # Use skeleton to avoid capturing model reference
            _wrapper._configure_wrapper(skeleton)  # type: ignore[arg-type]

        return _wrapper

    @classmethod
    def for_tasks(
        cls,
        model_or_pipeline: T,
        method_names: Optional[tuple] = None,
    ) -> "ModelWrapper[T]":
        """Convenience shortcut for task mode - equivalent to rewrite_pipeline().

        This immediately prepares and loads the pipeline for task-based execution,
        making it ready to use right away.

        Args:
            model_or_pipeline: The model or pipeline to wrap
            method_names: Model methods to expose via remote tasks (defaults to ("__call__",))

        Returns:
            A ModelWrapper ready for immediate use

        Example:
            >>> wrapper = ModelWrapper.for_tasks(pipeline)
            >>> result = wrapper.process(data)  # Immediately usable
        """
        return cls.from_model(
            model_or_pipeline,
            mode="task",
            method_names=method_names,
        )

    def load(self, _use_fast_load: bool = False) -> torch.nn.Module | T:
        """Load the model/pipeline from the wrapper.

        This function is to be called from within an actor's __init__ to deserialize and
        load the model from Ray's object store using zero-copy.

        Note: For task mode wrappers, this returns the rewritten pipeline.
        For actor mode wrappers, this loads the pipeline from the object store.
        Models are loaded on CPU. Users should handle device placement themselves after loading.

        Args:
            _use_fast_load: Use faster but slightly riskier loading method. Defaults to False.
                           Only applies to actor mode.

        Returns:
            The deserialized pipeline ready for inference (on CPU)

        Example:
            >>> class InferenceActor:
            ...     def __init__(self, model_wrapper):
            ...         # Load model (on CPU)
            ...         self.model = model_wrapper.load()
            ...         # Move to GPU if needed
            ...         self.model = self.model.to("cuda:0")
            ...
            ...     def __call__(self, batch):
            ...         return self.model(batch["data"])
        """
        if self._mode == "task":
            # Task mode: return the rewritten pipeline
            if self._rewritten is None:
                raise ValueError("Task mode wrapper has no rewritten pipeline")

            if self._is_standalone_module:
                loaded_container: _ModuleContainer = self._rewritten  # type: ignore[assignment]
                return loaded_container.get_model()
            else:
                return self._rewritten
        else:
            # Actor mode: load from object store
            pipeline = rzc_nn.load_pipeline_for_actors(
                self._skeleton,
                self._model_refs,
                device=None,
                use_fast_load=_use_fast_load,
            )

            if self._is_standalone_module:
                loaded_container: _ModuleContainer = pipeline  # type: ignore[assignment]
                return loaded_container.get_model()
            else:
                return pipeline

    def to_pipeline(self, _use_fast_load: bool = False) -> torch.nn.Module | T:
        """Deprecated: Use load() instead.

        This method is deprecated and will be removed in a future version.
        Use load() instead.
        """
        import warnings

        warnings.warn(
            "to_pipeline() is deprecated. Use load() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load(_use_fast_load=_use_fast_load)

    def __call__(self, *args, **kwargs):
        """Forward calls to the rewritten pipeline (task mode only)."""
        if self._mode != "task":
            raise TypeError(
                "Cannot call actor mode wrapper directly. "
                "Use wrapper.load() in the actor's __init__ first."
            )
        if self._rewritten is None:
            raise ValueError("Task mode wrapper has no rewritten pipeline")

        # For standalone modules, unwrap from container
        if self._is_standalone_module:
            container: _ModuleContainer = self._rewritten  # type: ignore[assignment]
            return container.get_model()(*args, **kwargs)
        else:
            return self._rewritten(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward attribute access to the rewritten pipeline (task mode only)."""
        # Avoid infinite recursion for private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        if self._mode != "task":
            raise AttributeError(
                "Cannot access attributes on actor mode wrapper. "
                "Use wrapper.load() in the actor's __init__ first."
            )
        if self._rewritten is None:
            raise ValueError("Task mode wrapper has no rewritten pipeline")

        # For standalone modules, unwrap from container
        if self._is_standalone_module:
            container: _ModuleContainer = self._rewritten  # type: ignore[assignment]
            return getattr(container.get_model(), name)
        else:
            return getattr(self._rewritten, name)

    def __getstate__(self):
        """Return state for pickling."""
        state = {
            "_skeleton": self._skeleton,
            "_model_refs": self._model_refs,
            "_is_standalone_module": self._is_standalone_module,
            "_mode": self._mode,
        }

        # Include rewritten and model_info only for task mode
        if self._mode == "task":
            state["_rewritten"] = self._rewritten
            state["_model_info"] = self._model_info

        return state

    def __setstate__(self, state):
        """Restore state from pickling."""
        self._skeleton = state["_skeleton"]
        self._model_refs = state["_model_refs"]
        self._is_standalone_module = state.get("_is_standalone_module", False)
        self._mode = state.get("_mode", "actor")

        # Restore task mode fields if present
        if self._mode == "task":
            self._rewritten = state.get("_rewritten")
            self._model_info = state.get("_model_info")
        else:
            self._rewritten = None
            self._model_info = None

    @classmethod
    def deserialize(
        cls,
        skeleton: T,
        model_refs: dict[str, ray.ObjectRef],
        is_standalone_module: bool = False,
        mode: Literal["task", "actor"] = "actor",
        rewritten: Optional[T] = None,
        model_info: Optional[
            dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]]
        ] = None,
    ) -> "ModelWrapper[T]":
        """Deserialize a ModelWrapper from a skeleton and model references."""
        return cls(
            skeleton,
            model_refs,
            is_standalone_module,
            mode,
            rewritten,
            model_info,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the ModelWrapper to a dictionary."""
        result = {
            "skeleton": self._skeleton,
            "model_refs": self._model_refs,
            "is_standalone_module": self._is_standalone_module,
            "mode": self._mode,
        }

        # Include task mode fields if applicable
        if self._mode == "task":
            result["rewritten"] = self._rewritten
            result["model_info"] = self._model_info

        return result
