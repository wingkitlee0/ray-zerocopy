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

    :param model_ref: Object reference to a tuple of (model_bytes, weights)
                      as returned by extract_tensors()
    :param args: Ordered arguments to pass to the model's method
    :param kwargs: Keyword arguments to pass to the model's method,
                   or None to pass no keyword arguments
    :param method_name: Name of the method to call on the object.
                        For TorchScript models, this is typically "__call__"
                        or "forward"

    :returns: Return value from calling the specified method
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
    """
    Shim object that forwards method calls to a remote Ray task for
    TorchScript models.
    """

    def __init__(self, model_ref: ray.ObjectRef, valid_methods: Set[str]):
        self._model_ref = model_ref
        self._valid_methods = valid_methods

    def __getattr__(self, name: str):
        if name in self._valid_methods:

            def _proxy(*args, **kwargs):
                return ray.get(call_model.remote(self._model_ref, args, kwargs, name))

            return _proxy
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __call__(self, *args, **kwargs):
        if "__call__" in self._valid_methods:
            return ray.get(call_model.remote(self._model_ref, args, kwargs, "__call__"))
        raise TypeError(f"'{type(self).__name__}' object is not callable")


def rewrite_pipeline(pipeline: Any, method_names=("__call__", "forward")) -> Any:
    """
    Rewrites TorchScript models in a model processing pipeline into Ray tasks
    that load the model using zero-copy model loading.

    .. deprecated::
        This function is deprecated. Use :class:`ray_zerocopy.JITTaskWrapper` instead:
        
        Old API::
        
            from ray_zerocopy.jit import rewrite_pipeline
            rewritten = rewrite_pipeline(pipeline)
            result = rewritten(data)
        
        New API::
        
            from ray_zerocopy import JITTaskWrapper
            wrapped = JITTaskWrapper(pipeline)
            result = wrapped(data)

    This is similar to ray_zerocopy.rewrite_pipeline() but specifically designed for
    TorchScript models (torch.jit.ScriptModule).

    Limitations:
    * Only TorchScript models (torch.jit.ScriptModule) will be rewritten
    * Only models stored in fields of the top-level object will be rewritten
    * Does not recursively traverse child objects
    * If there are multiple pointers to the same model, they will be
      treated as separate models
    * pipeline must work properly after being shallow-copied with copy.copy()

    :param pipeline: Python object that wraps a model serving pipeline
    :param method_names: Names of model methods to forward to remote classes.
                         Default is ("__call__", "forward") for TorchScript

    :returns: A shallow copy of pipeline in which all TorchScript models
              are replaced with wrapper objects that forward calls to Ray tasks
    """
    import warnings
    warnings.warn(
        "jit.rewrite_pipeline() is deprecated. Use JITTaskWrapper instead: "
        "from ray_zerocopy import JITTaskWrapper; wrapped = JITTaskWrapper(pipeline)",
        DeprecationWarning,
        stacklevel=2
    )
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
        valid_methods = {m for m in method_names if hasattr(model, m)}

        # Create the shim
        shim = _RemoteModelShim(model_ref, valid_methods)
        setattr(result, name, shim)

    return result
