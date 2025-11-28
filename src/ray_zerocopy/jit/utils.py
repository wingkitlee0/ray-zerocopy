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
Public API for working with zero-copy TorchScript models.
"""

from typing import Any, Protocol

import ray
import torch

from .rewrite import replace_tensors
from .tasks import rewrite_pipeline


class JitModelContainerProtocol(Protocol):
    """A class that contains a torch.jit.ScriptModule model or list of models."""


class ZeroCopyModel:
    """
    Utilities for working with zero-copy TorchScript models.

    This class provides methods similar to ray_zerocopy.ZeroCopyModel but specifically
    designed for TorchScript (torch.jit) models, which have different
    internal structure compared to regular torch.nn.Module models.
    """

    @staticmethod
    def to_object_ref(model_shim: Any) -> ray.ObjectRef:
        """
        Extract the underlying Ray ObjectRef from a rewritten TorchScript model shim.
        This reference points to the zero-copy data (skeleton + weights) in Plasma.

        :param model_shim: The rewritten model object returned by rewrite_pipeline
        :return: ray.ObjectRef
        """
        if hasattr(model_shim, "_model_ref"):
            return model_shim._model_ref
        raise TypeError(
            "The provided object is not a valid zero-copy TorchScript model shim."
        )

    @staticmethod
    def from_object_ref(model_ref: ray.ObjectRef) -> torch.jit.ScriptModule:
        """
        Reconstruct a functional TorchScript model from a zero-copy ObjectRef.
        This operation maps shared memory and does not copy weight data.

        :param model_ref: ray.ObjectRef pointing to (model_bytes, weights) tuple
        :return: torch.jit.ScriptModule (ready for inference)
        """
        model_bytes, model_weights = model_ref
        return replace_tensors(model_bytes, model_weights)

    @staticmethod
    def rewrite(pipeline: JitModelContainerProtocol) -> JitModelContainerProtocol:
        """
        A simple wrapper for the rewrite_pipeline function.

        For more advanced use cases, use rewrite_pipeline directly.

        :param pipeline: An object containing TorchScript models
        :return: A rewritten version with zero-copy model loading
        """
        return rewrite_pipeline(pipeline)
