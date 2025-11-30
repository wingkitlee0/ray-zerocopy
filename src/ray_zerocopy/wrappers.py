"""
Unified pipeline-based API for ray_zerocopy.

This module provides a consistent, easy-to-use API for zero-copy model loading
for TorchScript (JIT) models across different execution modes (tasks vs actors).

The wrapper classes:
- JITModelWrapper: Unified wrapper for TorchScript models supporting both task and actor modes

For nn.Module models, use ModelWrapper from ray_zerocopy.model_wrappers:
- ModelWrapper.for_tasks() or ModelWrapper.from_model(..., mode="task") for task-based execution
- ModelWrapper.from_model(..., mode="actor") for actor-based execution

Usage Examples:

    # 1. ModelWrapper Task Mode - Run nn.Module models in Ray tasks
    >>> from ray_zerocopy import ModelWrapper
    >>> pipeline = MyPipeline()  # Has .encoder, .decoder as nn.Module
    >>> wrapped = ModelWrapper.for_tasks(pipeline)
    >>> result = wrapped.process(data)  # Each model call spawns a Ray task

    # 2. ModelWrapper Actor Mode - Run nn.Module models in Ray actors
    >>> from ray_zerocopy import ModelWrapper
    >>> pipeline = MyPipeline()
    >>> model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")
    >>>
    >>> class MyActor:
    ...     def __init__(self, model_wrapper):
    ...         self.pipeline = model_wrapper.load()
    ...     def __call__(self, batch):
    ...         return self.pipeline(batch["data"])
    >>>
    >>> ds.map_batches(
    ...     MyActor,
    ...     fn_constructor_kwargs={"model_wrapper": model_wrapper},
    ...     compute=ActorPoolStrategy(size=4)
    ... )

    # 3. JITModelWrapper Task Mode - Run TorchScript models in Ray tasks
    >>> from ray_zerocopy import JITModelWrapper
    >>> jit_pipeline = torch.jit.trace(pipeline, example)
    >>> wrapped = JITModelWrapper.for_tasks(jit_pipeline)
    >>> result = wrapped(data)  # Each model call spawns a Ray task

    # 4. JITModelWrapper Actor Mode - Run TorchScript models in Ray actors
    >>> from ray_zerocopy import JITModelWrapper
    >>> jit_pipeline = torch.jit.trace(pipeline, example)
    >>> wrapper = JITModelWrapper.from_model(jit_pipeline, mode="actor")
    >>>
    >>> class MyJITActor:
    ...     def __init__(self, wrapper):
    ...         self.pipeline = wrapper.load()
    ...     def __call__(self, batch):
    ...         return self.pipeline(batch["data"])
    >>>
    >>> ds.map_batches(
    ...     MyJITActor,
    ...     fn_constructor_kwargs={"wrapper": wrapper},
    ...     compute=ActorPoolStrategy(size=4)
    ... )
"""

from __future__ import annotations

from typing import Any, Generic, Literal, Optional, Set, TypeVar, Union

import ray
import torch

from ray_zerocopy import jit as rzc_jit
from ray_zerocopy._internal import WrapperMixin

T = TypeVar("T")

# Type aliases for model info formats
TaskModelInfo = dict[str, tuple[ray.ObjectRef, Optional[Set[str]]]]
ActorModelInfo = dict[str, ray.ObjectRef]
ModelInfo = Union[TaskModelInfo, ActorModelInfo]


class _JITModuleContainer:
    """Internal container to make standalone torch.jit.ScriptModule look like a pipeline object."""

    def __init__(self, model: torch.jit.ScriptModule):
        self.model = model

    def get_model(self) -> torch.jit.ScriptModule:
        """Get the model from the container."""
        return self.model


class JITModelWrapper(WrapperMixin[T], Generic[T]):
    """
    A unified serializable wrapper with zero-copy loading for torch.jit.ScriptModule and Pipeline objects.

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

        >>> from ray_zerocopy import JITModelWrapper
        >>> jit_model = torch.jit.trace(model, example)
        >>> rewritten = JITModelWrapper.for_tasks(jit_model)
        >>> result = rewritten(data)  # Callable pipeline

        Task Mode (using from_model + load):

        >>> wrapper = JITModelWrapper.from_model(jit_model, mode="task")
        >>> rewritten = wrapper.load()  # Get callable pipeline
        >>> result = rewritten(data)

        Actor Mode:

        >>> wrapper = JITModelWrapper.from_model(jit_pipeline, mode="actor")
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
        Initialize JITModelWrapper.

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
    ) -> "JITModelWrapper[T]":
        """Instantiate a JITModelWrapper from a model or pipeline.

        A JITModelWrapper is serializable and can be put into Ray's object store by
        `ray.put()`.

        Args:
            model_or_pipeline: The TorchScript model or pipeline to wrap
            mode: Execution mode - "task" for task-based execution, "actor" for actor loading
                Must be "task" or "actor". Defaults to "actor".
            model_attr_names: The attribute names of the models in the pipeline
            method_names: Model methods to expose via remote tasks (auto-selected if None)

        Returns:
            A JITModelWrapper instance (not callable - use `.load()` to get the callable pipeline)

        Example - Task mode:
            >>> wrapper = JITModelWrapper.from_model(jit_pipeline, mode="task")
            >>> rewritten = wrapper.load()  # Get callable pipeline
            >>> result = rewritten(data)  # Use the pipeline

        Example - Actor mode:
            >>> wrapper = JITModelWrapper.from_model(jit_pipeline, mode="actor")
            >>> # In actor: pipeline = wrapper.load()
        """
        if mode not in ["task", "actor"]:
            raise ValueError(f"Invalid mode: {mode}")

        is_standalone = isinstance(model_or_pipeline, torch.jit.ScriptModule)

        _pipeline: Union[_JITModuleContainer, T] = (
            _JITModuleContainer(model_or_pipeline)
            if is_standalone
            else model_or_pipeline
        )

        # Auto-select method_names if None
        if method_names is None:
            if mode == "task":
                method_names = ("__call__", "forward")
            else:
                method_names = None  # No method tracking in actor mode

        if mode == "task":
            # Task mode: prepare only (no loading)
            skeleton, model_info = rzc_jit.prepare_pipeline_for_tasks(
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
            skeleton, model_refs = rzc_jit.prepare_pipeline_for_actors(
                _pipeline,
                model_attr_names=model_attr_names,
            )
            # Convert to ActorModelInfo format (just refs, no method tracking)
            actor_model_info: ActorModelInfo = model_refs

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
    ) -> torch.jit.ScriptModule | T:
        """Convert a model or pipeline into a callable rewritten pipeline with
        zero-copy model loading.

        Note:
            Under the hood, this is a wrapper around `from_model()` and `load()`
            that immediately prepares and loads the converted pipeline. The returned
            pipeline will use a remote Ray task for execution.

        Args:
            model_or_pipeline: The TorchScript model or pipeline to wrap
            method_names: Model methods to expose via remote tasks (defaults to ``("__call__", "forward")``)

        Returns:
            A rewritten pipeline ready for immediate use (callable). Each call will spawn a Ray task.
        """
        wrapper = cls.from_model(
            model_or_pipeline,
            mode="task",
            method_names=method_names,
        )
        return wrapper.load()

    def load(self) -> torch.jit.ScriptModule | T:
        """Load the model/pipeline from the wrapper.

        For task mode: Creates the rewritten pipeline on-demand with remote model shims.
        For actor mode: Loads the pipeline from Ray's object store using zero-copy.

        Models are loaded on CPU. Users should handle device placement themselves after loading.

        Args:
            (No arguments - kept for API consistency with ModelWrapper)

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
            >>> wrapper = JITModelWrapper.from_model(jit_pipeline, mode="task")
            >>> rewritten = wrapper.load()  # Get callable pipeline
            >>> result = rewritten(data)  # Use the pipeline
        """
        if self._mode == "task":
            # Task mode: create rewritten pipeline on-demand
            model_info: TaskModelInfo = self._model_info  # type: ignore[assignment]
            rewritten = rzc_jit.load_pipeline_for_tasks(self._skeleton, model_info)

            if self._is_standalone_module:
                loaded_container: _JITModuleContainer = rewritten  # type: ignore[assignment]
                return loaded_container.get_model()
            else:
                return rewritten
        else:
            # Actor mode: load from object store
            pipeline = rzc_jit.load_pipeline_for_actors(
                self._skeleton,
                self.model_refs,  # Use property to get model_refs
            )

            if self._is_standalone_module:
                loaded_container: _JITModuleContainer = pipeline  # type: ignore[assignment]
                return loaded_container.get_model()
            else:
                return pipeline

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
    ) -> "JITModelWrapper[T]":
        """Deserialize a JITModelWrapper from a skeleton and model info.

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
        """Serialize the JITModelWrapper to a dictionary."""
        return {
            "skeleton": self._skeleton,
            "model_info": self._model_info,
            "is_standalone_module": self._is_standalone_module,
            "mode": self._mode,
        }
