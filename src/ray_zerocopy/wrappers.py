"""
Unified pipeline-based API for ray_zerocopy.

This module provides a consistent, easy-to-use API for zero-copy model loading
for TorchScript (JIT) models across different execution modes (tasks vs actors).

The wrapper classes:
- JITTaskWrapper: TorchScript models executed via Ray tasks
- JITActorWrapper: TorchScript models loaded in Ray actors (for Ray Data)

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

    # 3. JITTaskWrapper - Run TorchScript models in Ray tasks
    >>> from ray_zerocopy import JITTaskWrapper
    >>> jit_pipeline = torch.jit.trace(pipeline, example)
    >>> wrapped = JITTaskWrapper(jit_pipeline)
    >>> result = wrapped.forward(data)  # Each model call spawns a Ray task

    # 4. JITActorWrapper - Run TorchScript models in Ray actors
    >>> from ray_zerocopy import JITActorWrapper
    >>> jit_pipeline = torch.jit.trace(pipeline, example)
    >>> actor_wrapper = JITActorWrapper(jit_pipeline)
    >>>
    >>> class MyJITActor:
    ...     def __init__(self, actor_wrapper):
    ...         self.pipeline = actor_wrapper.load()
    ...     def __call__(self, batch):
    ...         return self.pipeline(batch["data"])
    >>>
    >>> ds.map_batches(
    ...     MyJITActor,
    ...     fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    ...     compute=ActorPoolStrategy(size=4)
    ... )
"""

from typing import Generic, Optional, TypeVar

import ray

from ray_zerocopy import jit as rzc_jit
from ray_zerocopy._internal import WrapperMixin

T = TypeVar("T")


class JITTaskWrapper(WrapperMixin[T], Generic[T]):
    """
    Wrapper for zero-copy TorchScript inference via Ray tasks.

    This wrapper takes a pipeline object with torch.jit.ScriptModule attributes
    and rewrites it so that model inference calls are executed in Ray tasks with
    zero-copy model loading from the object store.

    Use this when you want to run TorchScript models via Ray tasks (not actors).
    For Ray Data with ActorPoolStrategy, use JITActorWrapper instead.

    Args:
        pipeline: Object containing torch.jit.ScriptModule models as attributes
        method_names: Tuple of model method names to expose via remote tasks.
                     Default is ("__call__", "forward")

    Example:
        >>> import torch
        >>>
        >>> # Create and trace your model
        >>> model = MyModel()
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> jit_model = torch.jit.trace(model, example_input)
        >>>
        >>> # Wrap in a pipeline object if needed
        >>> class Pipeline:
        ...     def __init__(self):
        ...         self.model = jit_model
        ...
        ...     def __call__(self, data):
        ...         return self.model(data)
        >>>
        >>> pipeline = Pipeline()
        >>> wrapped = JITTaskWrapper(pipeline)
        >>> result = wrapped(data)  # Model runs in Ray task
    """

    def __init__(self, pipeline: T, method_names: tuple = ("__call__", "forward")):
        """
        Initialize the JITTaskWrapper.

        Args:
            pipeline: Object containing torch.jit.ScriptModule models as attributes
            method_names: Model methods to expose via remote tasks
        """
        self._rewritten = rzc_jit.rewrite_pipeline(pipeline, method_names)

        self._configure_wrapper(pipeline)
        self._preserve_call_signature(pipeline)

    def __getstate__(self):
        """Return state for pickling."""
        return {
            "_rewritten": self._rewritten,
        }

    def __setstate__(self, state):
        """Restore state from pickling."""
        self._rewritten = state["_rewritten"]

    def __call__(self, *args, **kwargs):
        """Forward calls to the rewritten pipeline."""
        return self._rewritten(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward attribute access to the rewritten pipeline."""
        return getattr(self._rewritten, name)


class JITActorWrapper(WrapperMixin[T], Generic[T]):
    """
    Wrapper for zero-copy TorchScript inference in Ray actors.

    This wrapper prepares a pipeline with torch.jit.ScriptModule models for use
    in Ray actors, particularly with Ray Data's ActorPoolStrategy. Models are
    stored in Ray's object store and loaded with zero-copy in each actor.

    Use this when you want to run TorchScript models in Ray actors (e.g., with Ray Data).
    For simple Ray tasks, use JITTaskWrapper instead.

    Args:
        pipeline: Object containing torch.jit.ScriptModule models as attributes
        model_attr_names: List of attribute names that are models. If None,
                         auto-discovers all torch.jit.ScriptModule attributes.

    Example:
        >>> import torch
        >>>
        >>> # Create and trace your models
        >>> encoder = torch.jit.trace(EncoderModel(), example_input)
        >>> decoder = torch.jit.trace(DecoderModel(), example_encoded)
        >>>
        >>> class MyPipeline:
        ...     def __init__(self, encoder, decoder):
        ...         self.encoder = encoder
        ...         self.decoder = decoder
        ...
        ...     def __call__(self, data):
        ...         encoded = self.encoder(data)
        ...         return self.decoder(encoded)
        >>>
        >>> # Prepare pipeline for actors
        >>> pipeline = MyPipeline(encoder, decoder)
        >>> actor_wrapper = JITActorWrapper(pipeline)
        >>>
        >>> # Use in Ray Data
        >>> class InferenceActor:
        ...     def __init__(self, actor_wrapper):
        ...         # Specify device at load time
        ...         self.pipeline = actor_wrapper.load()
        ...
        ...     def __call__(self, batch):
        ...         return self.pipeline(batch["data"])
        >>>
        >>> ds = ray.data.read_parquet("data.parquet")
        >>> ds.map_batches(
        ...     InferenceActor,
        ...     fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
        ...     compute=ActorPoolStrategy(size=4)
        ... )
    """

    _skeleton: T
    _model_refs: dict[str, ray.ObjectRef]

    def __init__(
        self,
        pipeline: T,
        model_attr_names: Optional[list] = None,
    ):
        """
        Initialize the JITActorWrapper.

        Args:
            pipeline: Object containing torch.jit.ScriptModule models
            model_attr_names: List of model attribute names (auto-detected if None)
        """
        self._skeleton, self._model_refs = rzc_jit.prepare_pipeline_for_actors(
            pipeline,
            model_attr_names,
        )
        self._configure_wrapper(pipeline)

    def __getstate__(self):
        """Return state for pickling."""
        return {
            "_skeleton": self._skeleton,
            "_model_refs": self._model_refs,
        }

    def __setstate__(self, state):
        """Restore state from pickling."""
        self._skeleton = state["_skeleton"]
        self._model_refs = state["_model_refs"]

    def load(self) -> T:
        """
        Load the pipeline in an actor.

        Call this method from within an actor's __init__ method to reconstruct
        the pipeline with TorchScript models loaded from the object store using zero-copy.

        Returns:
            Pipeline object with TorchScript models loaded and ready for inference

        Example:
            >>> class MyActor:
            ...     def __init__(self, actor_wrapper):
            ...         # Move to GPU after loading from object store
            ...         self.pipeline = actor_wrapper.load()
            ...
            ...     def __call__(self, batch):
            ...         return self.pipeline(batch["data"])
        """

        return rzc_jit.load_pipeline_for_actors(
            self._skeleton,
            self._model_refs,
        )

    @property
    def constructor_kwargs(self) -> dict:
        """Get kwargs dict for Ray Data fn_constructor_kwargs."""
        return {
            "pipeline_skeleton": self._skeleton,
            "model_refs": self._model_refs,
        }
