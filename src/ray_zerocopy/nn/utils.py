from typing import Any, Protocol

import ray
import torch

from ray_zerocopy._internal.zerocopy import replace_tensors


class ModelContainerProtocol(Protocol):
    """A class that contains a nn.Module model or a list of nn.Module models."""


class ZeroCopyModel:
    """
    Utilities for working with zero-copy models.
    This class provides static methods to bridge the gap between
    high-level rewritten models (shims) and low-level Ray ObjectRefs.

    Note: For most use cases, prefer using TaskWrapper or ActorWrapper directly
    instead of these low-level utilities.
    """

    @staticmethod
    def to_object_ref(model_shim: Any) -> ray.ObjectRef:
        """
        Extract the underlying Ray ObjectRef from a rewritten model shim.
        This reference points to the zero-copy data (skeleton + weights) in Plasma.

        Args:
            model_shim: The rewritten model object returned by TaskWrapper

        Returns:
            ray.ObjectRef
        """
        if hasattr(model_shim, "_model_ref"):
            return model_shim._model_ref
        raise TypeError("The provided object is not a valid zero-copy model shim.")

    @staticmethod
    def from_object_ref(model_ref: ray.ObjectRef) -> torch.nn.Module:
        """
        Reconstruct a functional PyTorch model from a zero-copy ObjectRef.
        This operation maps shared memory and does not copy weight data.

        Args:
            model_ref: ray.ObjectRef pointing to (skeleton, weights) tuple

        Returns:
            torch.nn.Module ready for inference
        """
        model_skeleton, model_weights = model_ref
        replace_tensors(model_skeleton, model_weights)
        return model_skeleton

    @staticmethod
    def rewrite(pipeline: ModelContainerProtocol) -> ModelContainerProtocol:
        """
        DEPRECATED: Use TaskWrapper instead for task-based execution.

        This is a legacy wrapper that exists for backward compatibility.
        For new code, use:
            from ray_zerocopy import TaskWrapper
            wrapped = TaskWrapper(pipeline)
        """
        import warnings

        warnings.warn(
            "ZeroCopyModel.rewrite() is deprecated. Use TaskWrapper instead:\n"
            "  from ray_zerocopy import TaskWrapper\n"
            "  wrapped = TaskWrapper(pipeline)",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import here to avoid circular dependency
        from ray_zerocopy.wrappers import TaskWrapper

        return TaskWrapper(pipeline)
