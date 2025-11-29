"""
Unified pipeline-based API for ray_zerocopy.

This module provides a consistent, easy-to-use API for zero-copy model loading
across different execution modes (tasks vs actors) and model types (nn.Module vs JIT).

The four wrapper classes:
- TaskWrapper: nn.Module models executed via Ray tasks
- ActorWrapper: nn.Module models loaded in Ray actors (for Ray Data)
- JITTaskWrapper: TorchScript models executed via Ray tasks
- JITActorWrapper: TorchScript models loaded in Ray actors (for Ray Data)

Usage Examples:

    # 1. TaskWrapper - Run nn.Module models in Ray tasks
    >>> from ray_zerocopy import TaskWrapper
    >>> pipeline = MyPipeline()  # Has .encoder, .decoder as nn.Module
    >>> wrapped = TaskWrapper(pipeline)
    >>> result = wrapped.process(data)  # Each model call spawns a Ray task

    # 2. ActorWrapper - Run nn.Module models in Ray actors
    >>> from ray_zerocopy import ActorWrapper
    >>> pipeline = MyPipeline()
    >>> actor_wrapper = ActorWrapper(pipeline, device="cuda:0")
    >>>
    >>> class MyActor:
    ...     def __init__(self, actor_wrapper):
    ...         self.pipeline = actor_wrapper.load()
    ...     def __call__(self, batch):
    ...         return self.pipeline(batch["data"])
    >>>
    >>> ds.map_batches(
    ...     MyActor,
    ...     fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
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
    >>> actor_wrapper = JITActorWrapper(jit_pipeline, device="cuda:0")
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
from ray_zerocopy import nn as rzc_nn
from ray_zerocopy._internal import WrapperMixin

T = TypeVar("T")


class TaskWrapper(WrapperMixin[T], Generic[T]):
    """
    Wrapper for zero-copy nn.Module inference via Ray tasks.

    This wrapper takes a pipeline object (any object with torch.nn.Module attributes)
    and rewrites it so that model inference calls are executed in Ray tasks with
    zero-copy model loading from the object store.

    Use this when you want to run inference via Ray tasks (not actors).
    For Ray Data with ActorPoolStrategy, use ActorWrapper instead.

    Args:
        pipeline: Object containing torch.nn.Module models as attributes
        method_names: Tuple of model method names to expose via remote tasks.
                     Default is ("__call__",)

    Example:
        >>> class MyPipeline:
        ...     def __init__(self):
        ...         self.encoder = EncoderModel()
        ...         self.decoder = DecoderModel()
        ...
        ...     def process(self, data):
        ...         encoded = self.encoder(data)
        ...         return self.decoder(encoded)
        >>>
        >>> pipeline = MyPipeline()
        >>> wrapped = TaskWrapper(pipeline)
        >>> result = wrapped.process(data)  # Each model call spawns a Ray task
    """

    def __init__(self, pipeline: T, method_names: tuple = ("__call__",)):
        """
        Initialize the TaskWrapper.

        Args:
            pipeline: Object containing torch.nn.Module models as attributes
            method_names: Model methods to expose via remote tasks
        """
        # Use the new prepare_pipeline and load_pipeline_for_tasks API
        skeleton, model_info = rzc_nn.prepare_pipeline(
            pipeline, method_names=method_names, filter_private=False
        )

        self._rewritten = rzc_nn.load_pipeline_for_tasks(skeleton, model_info)

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


class ActorWrapper(WrapperMixin[T], Generic[T]):
    """
    Wrapper for zero-copy nn.Module inference in Ray actors.

    This wrapper prepares a pipeline with torch.nn.Module models for use in
    Ray actors, particularly with Ray Data's ActorPoolStrategy. Models are
    stored in Ray's object store and loaded with zero-copy in each actor.

    Use this when you want to run inference in Ray actors (e.g., with Ray Data).
    For simple Ray tasks, use TaskWrapper instead.

    Args:
        pipeline: Object containing torch.nn.Module models as attributes
        model_attr_names: List of attribute names that are models. If None,
                         auto-discovers all torch.nn.Module attributes.
        use_fast_load: Use faster but slightly riskier loading method.
                      Default is False.

    Example:
        >>> class MyPipeline:
        ...     def __init__(self):
        ...         self.encoder = EncoderModel()
        ...         self.decoder = DecoderModel()
        ...
        ...     def __call__(self, data):
        ...         encoded = self.encoder(data)
        ...         return self.decoder(encoded)
        >>>
        >>> # Prepare pipeline for actors
        >>> pipeline = MyPipeline()
        >>> actor_wrapper = ActorWrapper(pipeline)
        >>>
        >>> # Use in Ray Data
        >>> class InferenceActor:
        ...     def __init__(self, actor_wrapper):
        ...         # Specify device at load time
        ...         self.pipeline = actor_wrapper.load(device="cuda:0")
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
        Initialize the ActorWrapper.

        Args:
            pipeline: Object containing torch.nn.Module models
            model_attr_names: List of model attribute names (auto-detected if None)
            use_fast_load: Use faster but riskier loading method
        """
        self._skeleton, self._model_refs = rzc_nn.prepare_pipeline_for_actors(
            pipeline,
            model_attr_names,
        )
        self._configure_wrapper(pipeline)

    def load(self, device: Optional[str] = None, _use_fast_load: bool = False) -> T:
        """
        Load the pipeline in an actor.

        Call this method from within an actor's __init__ method to reconstruct
        the pipeline with models loaded from the object store using zero-copy.

        Args:
            device: Device to move models to after loading (e.g., "cuda:0", "cpu").
                   If None, no device transfer is performed (models remain on the
                   device they were reconstructed on, typically CPU from object store).
            _use_fast_load: Use faster but slightly riskier loading method.

        Returns:
            Pipeline object with models loaded and ready for inference

        Example:
            >>> class MyActor:
            ...     def __init__(self, actor_wrapper):
            ...         # Move to GPU after loading from object store
            ...         self.pipeline = actor_wrapper.load(device="cuda:0")
            ...
            ...     def __call__(self, batch):
            ...         return self.pipeline(batch["data"])
        """
        return rzc_nn.load_pipeline_for_actors(
            self._skeleton,
            self._model_refs,
            device=device,
            use_fast_load=_use_fast_load,
        )

    @property
    def constructor_kwargs(self) -> dict:
        """Get kwargs dict for Ray Data fn_constructor_kwargs."""
        return {
            "pipeline_skeleton": self._skeleton,
            "model_refs": self._model_refs,
        }


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
        ...         self.pipeline = actor_wrapper.load(device="cuda:0")
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

    def load(self, device: Optional[str] = None) -> T:
        """
        Load the pipeline in an actor.

        Call this method from within an actor's __init__ method to reconstruct
        the pipeline with TorchScript models loaded from the object store using zero-copy.

        Args:
            device: Device to move models to after loading (e.g., "cuda:0", "cpu").
                   If None, no device transfer is performed (models remain on the
                   device they were reconstructed on, typically CPU from object store).

        Returns:
            Pipeline object with TorchScript models loaded and ready for inference

        Example:
            >>> class MyActor:
            ...     def __init__(self, actor_wrapper):
            ...         # Move to GPU after loading from object store
            ...         self.pipeline = actor_wrapper.load(device="cuda:0")
            ...
            ...     def __call__(self, batch):
            ...         return self.pipeline(batch["data"])
        """

        return rzc_jit.load_pipeline_for_actors(
            self._skeleton,
            self._model_refs,
            device=device,
        )

    @property
    def constructor_kwargs(self) -> dict:
        """Get kwargs dict for Ray Data fn_constructor_kwargs."""
        return {
            "pipeline_skeleton": self._skeleton,
            "model_refs": self._model_refs,
        }
