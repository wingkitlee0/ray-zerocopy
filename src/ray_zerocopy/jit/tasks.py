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
Functions for invoking TorchScript models that have been rewritten for
zero-copy loading.
"""

import copy
from typing import Any, Set

import ray
import torch

from .rewrite import extract_tensors, replace_tensors


@ray.remote
def call_model(
    model_ref: ray.ObjectRef,
    args: Any = (),
    kwargs: Any = None,
    method_name: str = "__call__",
) -> Any:
    """
    Ray task that uses zero-copy model loading to reconstitute a TorchScript
    model from Plasma, then calls a method on the model.

    Args:
        model_ref: Object reference to a tuple of (model_bytes, weights)
            as returned by extract_tensors()
        args: Ordered arguments to pass to the model's method
        kwargs: Keyword arguments to pass to the model's method,
            or None to pass no keyword arguments
        method_name: Name of the method to call on the object.
            For TorchScript models, this is typically "__call__"
            or "forward"

    Returns:
        Return value from calling the specified method
    """
    if kwargs is None:
        kwargs = {}

    # Suppress PyTorch warnings about immutable tensors
    import warnings

    warnings.filterwarnings("ignore")

    model_bytes, model_weights = model_ref
    model = replace_tensors(model_bytes, model_weights)

    with torch.no_grad():
        if method_name == "__call__":
            return model(*args, **kwargs)
        else:
            method = getattr(model, method_name)
            return method(*args, **kwargs)


class _RemoteModelShim:
    """A Shim object that forwards method calls to a remote Ray task"""

    def __init__(self, model_ref: ray.ObjectRef, allowed_methods: Set[str]):
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
        raise TypeError(f"'{type(self).__name__}' object is not callable")


def rewrite_pipeline(pipeline: Any, method_names=("__call__", "forward")) -> Any:
    """
    Rewrites TorchScript models in a model processing pipeline into Ray tasks
    that load the model using zero-copy model loading.

    This is a low-level API. For most use cases, consider using
    ray_zerocopy.JITTaskWrapper for a higher-level interface.

    Limitations:
    * Only models that are subclasses of torch.jit.ScriptModule will be rewritten.
    * If there are multiple pointers to the same model, they will be
      treated as separate models and loaded separately onto Plasma.
    * pipeline must be an object that will still work properly after
      being shallow-copied with copy.copy()

    Args:
        pipeline: Python object that wraps a model serving pipeline
        method_names: Names of model methods to forward to remote classes.
            Default is ("__call__", "forward") for TorchScript

    Returns:
        A shallow copy of pipeline in which all TorchScript models
        are replaced with wrapper objects that forward calls to Ray tasks
    """
    # Find all TorchScript models hanging directly off the pipeline object
    jit_model_attr_names = [
        name
        for name in dir(pipeline)
        if isinstance(getattr(pipeline, name), torch.jit.ScriptModule)
    ]

    # Shallow-copy the original pipeline
    result = copy.copy(pipeline)

    # Replace TorchScript models with shims
    for name in jit_model_attr_names:
        model = getattr(result, name)
        model_ref = ray.put(extract_tensors(model))

        # Determine which methods exist on this model
        allowed_methods = {m for m in method_names if hasattr(model, m)}

        # Always include __call__ if the model is callable, as pipeline methods
        # often call the model directly (e.g., self.model(x))
        if hasattr(model, "__call__") and callable(model):
            allowed_methods.add("__call__")

        # Create the shim
        shim = _RemoteModelShim(model_ref, allowed_methods)
        setattr(result, name, shim)

    return result
