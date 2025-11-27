from typing import Any, Protocol

import ray
import torch

from ray_zerocopy.invoke import replace_tensors, rewrite_pipeline


class ModelContainerProtocol(Protocol):
    """A class that contains a nn.Module model or a list of nn.Module models."""


class ZeroCopyModel:
    """
    Utilities for working with zero-copy models.
    This class provides a pair of static methods to bridge the gap between
    high-level rewritten models (shims) and low-level Ray ObjectRefs.
    """

    @staticmethod
    def to_object_ref(model_shim: Any) -> ray.ObjectRef:
        """
        Extract the underlying Ray ObjectRef from a rewritten model shim.
        This reference points to the zero-copy data (skeleton + weights) in Plasma.

        :param model_shim: The rewritten model object returned by rewrite_pipeline.
        :return: ray.ObjectRef
        """
        if hasattr(model_shim, "_model_ref"):
            return model_shim._model_ref
        raise TypeError("The provided object is not a valid zero-copy model shim.")

    @staticmethod
    def from_object_ref(model_ref: ray.ObjectRef) -> torch.nn.Module:
        """
        Reconstruct a functional PyTorch model from a zero-copy ObjectRef.
        This operation maps shared memory and does not copy weight data.

        :param model_ref: ray.ObjectRef pointing to (skeleton, weights) tuple.
        :return: torch.nn.Module (ready for inference)
        """
        model_skeleton, model_weights = model_ref
        replace_tensors(model_skeleton, model_weights)
        return model_skeleton

    @staticmethod
    def rewrite(pipeline: ModelContainerProtocol) -> ModelContainerProtocol:
        """A simple wrapper for the rewrite_pipeline function. For
        more advanced use cases, use rewrite_pipeline directly."""

        return rewrite_pipeline(pipeline)
