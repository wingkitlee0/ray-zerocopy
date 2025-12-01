import io
import logging

import ray
import torch

logger = logging.getLogger(__name__)

"""Utility functions for passing and loading models in and out of the Ray object store."""


def ray_put_model(
    model: torch.nn.Module | torch.jit.ScriptModule, use_jit: bool = False
) -> ray.ObjectRef:
    if not use_jit:
        return ray.put(model)

    model_bytes = io.BytesIO()
    torch.jit.save(model, model_bytes)
    return ray.put(model_bytes.getvalue())


def ray_get_model(
    model_ref: ray.ObjectRef | bytes, use_jit: bool = False, **kwargs
) -> torch.nn.Module | torch.jit.ScriptModule:
    """Get a model from the Ray object store. To
    be called in a Ray task."""
    if not use_jit:
        if isinstance(model_ref, ray.ObjectRef):
            return ray.get(model_ref)  # type: ignore
        else:
            return model_ref  # type: ignore

    if isinstance(model_ref, bytes):
        model_bytes = io.BytesIO(model_ref)
    else:
        # This is a rare case.
        model_bytes = io.BytesIO(ray.get(model_ref))

    return torch.jit.load(model_bytes, **kwargs)
