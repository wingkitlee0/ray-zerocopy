"""
Primary API for zero-copy model sharing with nn.Module models.

This module provides ModelWrapper, a unified wrapper that supports both task and actor
execution modes for zero-copy model sharing.

The wrapper classes:
- ModelWrapper: Handles both nn.Module and Pipeline objects for task and actor usage
- JITTaskWrapper/JITActorWrapper: For TorchScript (compiled) models (in wrappers.py)

Key API Patterns:

1. **Task Mode** - For ad-hoc inference with Ray tasks:
   - Use `ModelWrapper.for_tasks()` to get a callable rewritten pipeline
   - Or use `ModelWrapper.from_model(..., mode="task")` then call `.load()`
   - Pipeline is immediately usable: `result = rewritten(data)`

2. **Actor Mode** - For Ray Data and long-running actors:
   - Use `ModelWrapper.from_model(..., mode="actor")`
   - Load in actor: `model = wrapper.load()`

Usage Examples:

    # 1. Task Mode - Immediate use
    >>> from ray_zerocopy import ModelWrapper
    >>> pipeline = MyPipeline()
    >>> rewritten = ModelWrapper.for_tasks(pipeline)
    >>> result = rewritten(data)  # Each call spawns a Ray task

    # 2. Actor Mode - For Ray Data
    >>> from ray_zerocopy import ModelWrapper
    >>> pipeline = MyPipeline()
    >>> wrapper = ModelWrapper.from_model(pipeline, mode="actor")
    >>>
    >>> class InferenceActor:
    ...     def __init__(self, model_wrapper):
    ...         self.model = model_wrapper.load()
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

# Type aliases for model info formats
TaskModelInfo = dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]]
ActorModelInfo = dict[str, ray.ObjectRef]
ModelInfo = Union[TaskModelInfo, ActorModelInfo]


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

    Attributes:
        skeleton: The skeleton of the model or pipeline
        model_refs: A dict of Ray object references to the model tensors
        is_standalone_module: Whether the model or pipeline is a standalone module
        mode: Execution mode - "task" or "actor"
        model_info: Model info dict with method tracking (task mode) or just refs (actor mode)

    Examples:
        Task Mode (using for_tasks shortcut):

        >>> from ray_zerocopy import ModelWrapper
        >>> model = YourModel()
        >>> rewritten = ModelWrapper.for_tasks(model)
        >>> result = rewritten(data)  # Callable pipeline
        >>>

        Task Mode (using from_model + load):

        >>> wrapper = ModelWrapper.from_model(model, mode="task")
        >>> rewritten = wrapper.load()  # Get callable pipeline
        >>> result = rewritten(data)

        Actor Mode:

        >>> wrapper = ModelWrapper.from_model(model, mode="actor")
        >>>
        >>> class InferenceActor:
        ...     def __init__(self, model_wrapper):
        ...         self.model = model_wrapper.load()  # Inside an actor's __init__
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
    _model_info: ModelInfo
    _is_standalone_module: bool
    _mode: Literal["task", "actor"]

    def __init__(
        self,
        skeleton: T,
        model_info: ModelInfo,
        is_standalone_module: bool = False,
        mode: Literal["task", "actor"] = "actor",
    ):
        """
        Initialize ModelWrapper.

        Args:
            skeleton: The skeleton of the model or pipeline
            model_info: Model info dict. In task mode: dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]]
                (with method tracking). In actor mode: dict[str, ray.ObjectRef] (just refs).
            is_standalone_module: Whether this is a standalone module
            mode: Execution mode ("task" or "actor")
        """
        self._skeleton = skeleton
        self._is_standalone_module = is_standalone_module
        self._mode = mode
        self._model_info = model_info

    @property
    def model_refs(self) -> dict[str, ray.ObjectRef]:
        """Get model references, extracted from model_info."""
        if self._mode == "task":
            # Task mode: model_info is TaskModelInfo (dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]])
            model_info: TaskModelInfo = self._model_info  # type: ignore[assignment]
            return {
                attr_name: model_ref for attr_name, (model_ref, _) in model_info.items()
            }
        else:
            # Actor mode: model_info is ActorModelInfo (dict[str, ray.ObjectRef])
            model_info: ActorModelInfo = self._model_info  # type: ignore[assignment]
            return model_info

    @classmethod
    def from_model(
        cls,
        model_or_pipeline: T,
        mode: Literal["task", "actor"] = "actor",
        model_attr_names: Optional[list] = None,
        method_names: Optional[tuple] = None,
    ) -> "ModelWrapper[T]":
        """Instantiate a ModelWrapper from a model or pipeline.

        A ModelWrapper is serializable and can be put into Ray's object store by
        `ray.put()`.

        Args:
            model_or_pipeline: The model or pipeline to wrap
            mode: Execution mode - "task" for task-based execution, "actor" for actor loading
                Must be "task" or "actor". Defaults to "actor".
            model_attr_names: The attribute names of the models in the pipeline
            method_names: Model methods to expose via remote tasks (auto-selected if None)

        Returns:
            A ModelWrapper instance (not callable - use `.load()` to get the callable pipeline)

        Example - Task mode:
            >>> wrapper = ModelWrapper.from_model(pipeline, mode="task")
            >>> rewritten = wrapper.load()  # Get callable pipeline
            >>> result = rewritten(data)  # Use the pipeline

        Example - Actor mode:
            >>> wrapper = ModelWrapper.from_model(pipeline, mode="actor")
            >>> # In actor: pipeline = wrapper.load()
        """
        if mode not in ["task", "actor"]:
            raise ValueError(f"Invalid mode: {mode}")

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
            # Task mode: prepare only (no loading)
            skeleton, model_info = rzc_nn.prepare_pipeline(
                _pipeline,
                method_names=method_names,
                filter_private=False,
            )

            # model_info is TaskModelInfo in task mode
            _wrapper = cls(
                skeleton,
                model_info,
                is_standalone,
                mode="task",
            )
            # Use skeleton to avoid capturing model reference
            _wrapper._configure_wrapper(skeleton)  # type: ignore[arg-type]

        else:
            # Actor mode: prepare only (no loading)
            skeleton, model_info = rzc_nn.prepare_pipeline(
                _pipeline,
                model_attr_names=model_attr_names,
                method_names=None,  # No method tracking for actors
                filter_private=True,
            )
            # Convert to ActorModelInfo format (just refs, no method tracking)
            actor_model_info: ActorModelInfo = rzc_nn.model_info_to_model_refs(
                model_info
            )

            _wrapper = cls(
                skeleton,
                actor_model_info,
                is_standalone,
                mode="actor",
            )
            # Use skeleton to avoid capturing model reference
            _wrapper._configure_wrapper(skeleton)  # type: ignore[arg-type]

        return _wrapper

    @classmethod
    def for_tasks(
        cls,
        model_or_pipeline: T,
        method_names: Optional[tuple] = None,
    ) -> torch.nn.Module | T:
        """Convert a model or pipeline into a callable rewritten pipeline with
        zero-copy model loading.

        Note:
            Under the hood, this is a wrapper around `from_model()` and `load()`
            that immediately prepares and loads the converted pipeline. The returned
            pipeline will use a remote Ray task for execution.

        Args:
            model_or_pipeline: The model or pipeline to wrap
            method_names: Model methods to expose via remote tasks (defaults to ``("__call__",)``)

        Returns:
            A rewritten pipeline ready for immediate use (callable). Each call will spawn a Ray task.
        """
        wrapper = cls.from_model(
            model_or_pipeline,
            mode="task",
            method_names=method_names,
        )
        return wrapper.load()

    def load(self, _use_fast_load: bool = False) -> torch.nn.Module | T:
        """Load the model/pipeline from the wrapper.

        For task mode: Creates the rewritten pipeline on-demand with remote model shims.
        For actor mode: Loads the pipeline from Ray's object store using zero-copy.

        Models are loaded on CPU. Users should handle device placement themselves after loading.

        Args:
            _use_fast_load: Use faster but slightly riskier loading method. Defaults to False.
                           Only applies to actor mode.

        Returns:
            The deserialized pipeline ready for inference (on CPU)

        Example - Actor mode:
            >>> class InferenceActor:
            ...     def __init__(self, model_wrapper):
            ...         # Load model (on CPU)
            ...         self.model = model_wrapper.load()
            ...
            ...     def __call__(self, batch):
            ...         return self.model(batch["data"])

        Example - Task mode:
            >>> wrapper = ModelWrapper.from_model(pipeline, mode="task")
            >>> rewritten = wrapper.load()  # Get callable pipeline
            >>> result = rewritten(data)  # Use the pipeline
        """
        if self._mode == "task":
            # Task mode: create rewritten pipeline on-demand
            model_info: TaskModelInfo = self._model_info  # type: ignore[assignment]
            rewritten = rzc_nn.load_pipeline_for_tasks(self._skeleton, model_info)

            if self._is_standalone_module:
                loaded_container: _ModuleContainer = rewritten  # type: ignore[assignment]
                return loaded_container.get_model()
            else:
                return rewritten
        else:
            # Actor mode: load from object store
            pipeline = rzc_nn.load_pipeline_for_actors(
                self._skeleton,
                self.model_refs,  # Use property to get model_refs
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

    def __getstate__(self):
        """Return state for pickling."""
        return {
            "_skeleton": self._skeleton,
            "_model_info": self._model_info,
            "_is_standalone_module": self._is_standalone_module,
            "_mode": self._mode,
        }

    def __setstate__(self, state):
        """Restore state from pickling."""
        self._skeleton = state["_skeleton"]
        self._is_standalone_module = state.get("_is_standalone_module", False)
        self._mode = state.get("_mode", "actor")
        self._model_info = state["_model_info"]

    @classmethod
    def deserialize(
        cls,
        skeleton: T,
        model_info: ModelInfo,
        is_standalone_module: bool = False,
        mode: Literal["task", "actor"] = "actor",
    ) -> "ModelWrapper[T]":
        """Deserialize a ModelWrapper from a skeleton and model info.

        Args:
            skeleton: The skeleton of the model or pipeline
            model_info: Model info dict. Task mode: TaskModelInfo, Actor mode: ActorModelInfo
            is_standalone_module: Whether this is a standalone module
            mode: Execution mode ("task" or "actor")
        """
        return cls(
            skeleton,
            model_info,
            is_standalone_module,
            mode,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the ModelWrapper to a dictionary."""
        return {
            "skeleton": self._skeleton,
            "model_info": self._model_info,
            "is_standalone_module": self._is_standalone_module,
            "mode": self._mode,
        }
