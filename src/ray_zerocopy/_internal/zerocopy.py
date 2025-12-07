#
#  Copyright (c) 2021, 2022 IBM Corp.
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
#
#  Modified by Kit Lee, 2025
#  Changes:
#  - Extended with additional utility functions (_make_tensor_from_array,
#  - _RemoteModelShim) and integration with ray_zerocopy module architecture.

"""
PyTorch model rewrites related to zero-copy model loading. These rewrites allow
users to separate a model into its weights and graph, so that the weights can
be loaded via a zero-copy mechanism such as `ray.get()`, then plugged into an
empty version of the graph.

This module contains the core IBM zero-copy implementation for tensor extraction
and replacement operations.
"""

import copy
import warnings
from typing import Any, Dict, List, Tuple

import os

import numpy as np
import ray
import torch

# Check if Ray is configured to support zero-copy tensor serialization
USE_DIRECT_TENSORS = os.environ.get("RAY_ENABLE_ZERO_COPY_TENSOR_SERIALIZATION") == "1"


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    """
    Remove the tensors from a PyTorch model, extract them as Tensors,
    and return the stripped model and tensors.

    Args:
        m: Root node of a PyTorch model encoded as a graph of subclasses of
            torch.nn.Module

    Returns:
        A tuple with two elements:
        * A deep copy of `m` in which all weight tensors have been
          replaced by `None`
        * The tensors that were removed from the copy of `m`, encoded as
          a list of dictionaries. Each dictionary holds the tensors
          associated with a single torch.nn.Module in the model's graph,
          indexed by parameter name. The dictionaries occur in the order
          returned by m.named_modules()
    """
    tensors = []
    for _, module in m.named_modules():
        # Store the tensors in Python dictionaries
        params = {}
        for name, param in module.named_parameters(recurse=False):
            if USE_DIRECT_TENSORS:
                params[name] = torch.clone(param).detach().cpu()
            else:
                params[name] = torch.clone(param).detach().numpy()

        buffers = {}
        for name, buf in module.named_buffers(recurse=False):
            if USE_DIRECT_TENSORS:
                buffers[name] = torch.clone(buf).detach().cpu()
            else:
                buffers[name] = torch.clone(buf).detach().numpy()

        tensors.append({"params": params, "buffers": buffers})

    # Make a copy of the original model and strip all tensors and
    # temporary buffers out of the copy.
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    # Make sure the copy is configured for inference.
    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict]):
    """
    The inverse operation of extract_tensors(). Restores the tensors that
    extract_tensors() stripped out of a PyTorch model. This restore operation
    involves zero copying of data and results in a model that can be immediately
    used for CPU-based inference. To avoid copying, this function modifies the target
    model in place.

    Args:
        m: Root node of a PyTorch model encoded as a graph of subclasses of
            torch.nn.Module. Usually this parameter contains a model that has been
            stripped of its weights by extract_tensors(). **Modified in place.**
            If any weights are present in `m`, this function will replace them.
        tensors: The tensors to be inserted into `m`, encoded as a list of
            dictionaries. Each dictionary holds the tensors associated with a single
            torch.nn.Module in the model's graph, indexed by parameter name.
            The dictionaries occur in the order returned by m.named_modules()
    """
    with torch.inference_mode():
        modules = [module for _, module in m.named_modules()]
        for module, tensor_dict in zip(modules, tensors):
            # There are separate APIs to set parameters and buffers.
            # There are separate APIs to set parameters and buffers.
            for name, tensor_or_array in tensor_dict["params"].items():
                if isinstance(tensor_or_array, np.ndarray):
                    tensor = _make_tensor_from_array(tensor_or_array)
                else:
                    tensor = tensor_or_array

                module.register_parameter(
                    name, torch.nn.Parameter(tensor, requires_grad=False)
                )
            for name, tensor_or_array in tensor_dict["buffers"].items():
                if isinstance(tensor_or_array, np.ndarray):
                    tensor = _make_tensor_from_array(tensor_or_array)
                else:
                    tensor = tensor_or_array

                module.register_buffer(name, tensor)


def replace_tensors_direct(m: torch.nn.Module, tensors: List[Dict]):
    """
    A version of replace_tensors() that takes a faster but slightly dangerous
    shortcut.

    Like replace_tensors(), this function restores the tensors that
    extract_tensors() stripped out of a PyTorch model. However, this function
    skips the step of wrapping the restored tensors in torch.nn.Parameters objects.
    Skipping this step makes the restore operation go about 20% faster in testing on
    bert-base-uncased, but **may impact the correctness of some models**.
    Be sure to test this method carefully before using it on a particular PyTorch model.

    Like replace_tensors(), this function modifies the model in place to avoid
    copying data.

    Args:
        m: Root node of a PyTorch model encoded as a graph of subclasses of
            torch.nn.Module. Usually this parameter contains a model that has been
            stripped of its weights by extract_tensors(). **Modified in place.**
            If any weights are present in `m`, this function will replace them.
        tensors: The tensors to be inserted into `m`, encoded as a list of
            dictionaries. Each dictionary holds the tensors associated with a single
            torch.nn.Module in the model's graph, indexed by parameter name.
            The dictionaries occur in the order returned by m.named_modules()
    """
    with torch.inference_mode():
        modules = [module for _, module in m.named_modules()]
        for module, tensor_dict in zip(modules, tensors):
            # There are separate APIs to set parameters and buffers.
            # There are separate APIs to set parameters and buffers.
            for name, tensor_or_array in tensor_dict["params"].items():
                if isinstance(tensor_or_array, np.ndarray):
                    tensor = _make_tensor_from_array(tensor_or_array)
                else:
                    tensor = tensor_or_array

                # Super fast, somewhat risky version avoids
                # wrapping parameters in Parameters objects.
                module._parameters[name] = tensor
            for name, tensor_or_array in tensor_dict["buffers"].items():
                if isinstance(tensor_or_array, np.ndarray):
                    tensor = _make_tensor_from_array(tensor_or_array)
                else:
                    tensor = tensor_or_array

                module.register_buffer(name, tensor)


@ray.remote
def call_model(
    model_ref: ray.ObjectRef,
    args: Any = (),
    kwargs: Any = None,
    method_name: str = "__call__",
) -> Any:
    """
    Ray task that uses zero-copy model loading to reconstitute a model
    from Plasma, then a method on the model.

    Args:
        model_ref: Object reference to a tuple of model skeleton
            and model weights, as returned by extract_tensors()
        args: Ordered arguments to pass to the model's method
        kwargs: Keyword arguments to pass to the model's method,
            or None to pass no keyword arguments
        method_name: Name of the method to call on the object

    Returns:
        Return value from calling the specified method
    """
    if kwargs is None:
        kwargs = {}

    # Suppress PyTorch warnings about immutable tensors
    import warnings

    warnings.filterwarnings("ignore")

    model_skeleton, model_weights = model_ref
    replace_tensors(model_skeleton, model_weights)
    with torch.no_grad():
        method = getattr(model_skeleton, method_name)
        return method(*args, **kwargs)


def rewrite_pipeline_original(pipeline: Any, method_names=("__call__",)) -> Any:
    """
    Original IBM implementation: Rewrites PyTorch models in a model processing
    pipeline into Ray tasks that load the model using zero-copy model loading.

    NOTE: This function is preserved as a reference implementation of the original
    IBM code. Modern code should use ModelWrapper.for_tasks() instead, which uses prepare_pipeline()
    from ray_zerocopy.nn for better modularity.

    This is a low-level API. For most use cases, consider using
    ray_zerocopy.ModelWrapper for a higher-level interface.

    Current limitatations:
    * Only models that are stored in fields of the top-level object will be
      rewritten. This method does *not* recursively traverse child objects.
    * Only models that are subclasses of torch.nn.Module will be rewritten.
    * If there are multiple pointers to the same model, they will be
      treated as separate models and loaded separately onto Plasma.
    * pipeline must be an object that will still work properly after
      being shallow-copied with copy.copy()

    Args:
        pipeline: Python object that wraps a model serving pipeline
        method_names: Names of model methods to forward to remote classes

    Returns:
        A **shallow** copy of pipeline in which all PyTorch models
        that are stored in fields of pipeline are replaced with wrapper
        objects that forward calls to Ray tasks.
    """
    # Find all the models hanging directly off of the pipeline object.
    model_attr_names = [
        name
        for name in dir(pipeline)
        if isinstance(getattr(pipeline, name), torch.nn.Module)
    ]

    # Shallow-copy the original pipeline
    result = copy.copy(pipeline)

    # Replace models with shims that push model inference onto Ray tasks.
    for name in model_attr_names:
        model = getattr(result, name)
        model_ref = ray.put(extract_tensors(model))

        # Determine which of the requested methods actually exist on this model
        allowed_methods = {m for m in method_names if hasattr(model, m)}

        # Always include __call__ if the model is callable, as pipeline methods
        # often call the model directly (e.g., self.model(x))
        if hasattr(model, "__call__") and callable(model):
            allowed_methods.add("__call__")

        # Create the shim
        shim = _RemoteModelShim(model_ref, allowed_methods)
        setattr(result, name, shim)

    return result


def _make_tensor_from_array(array):
    """
    Create a PyTorch tensor from a NumPy array, avoiding copies if possible.

    Attempts to create a tensor without copying using torch.as_tensor().
    If this fails (e.g., due to array being read-only or incompatible),
    falls back to creating a tensor from a copy of the array.

    Args:
        array: NumPy array to convert to a PyTorch tensor

    Returns:
        PyTorch tensor, either zero-copy or from a copy of the input
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            return torch.as_tensor(array)
    except Exception:
        # Fallback to copy if zero-copy conversion fails
        return torch.as_tensor(array.copy())


class _RemoteModelShim:
    """
    Shim object that forwards method calls to a remote Ray task.
    """

    def __init__(self, model_ref: ray.ObjectRef, allowed_methods: set[str]):
        self._model_ref = model_ref
        self._allowed_methods = allowed_methods

    def __getstate__(self):
        """Return state for pickling."""
        return {
            "_model_ref": self._model_ref,
            "_allowed_methods": self._allowed_methods,
        }

    def __setstate__(self, state):
        """Restore state from pickling."""
        self._model_ref = state["_model_ref"]
        self._allowed_methods = state["_allowed_methods"]

    def __getattr__(self, name: str):
        # Avoid recursion during pickling - raise AttributeError for private attributes
        # and any attributes not in allowed_methods
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Get _allowed_methods directly from __dict__ to avoid recursion
        allowed_methods = object.__getattribute__(self, "_allowed_methods")

        if name in allowed_methods:
            model_ref = object.__getattribute__(self, "_model_ref")

            def _proxy(*args, **kwargs):
                return ray.get(call_model.remote(model_ref, args, kwargs, name))

            return _proxy
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __call__(self, *args, **kwargs):
        if "__call__" in self._allowed_methods:
            return ray.get(call_model.remote(self._model_ref, args, kwargs, "__call__"))
