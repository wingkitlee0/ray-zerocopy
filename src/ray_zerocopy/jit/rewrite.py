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
TorchScript (torch.jit) model rewrites for zero-copy model loading.

This module provides functions to separate a TorchScript model into its
weights and compiled graph, enabling zero-copy weight loading via Ray's
shared memory while preserving the TorchScript optimizations.
"""

import io
import copy
import torch
import warnings
from typing import Tuple
from collections import OrderedDict


def extract_tensors(
    m: torch.jit.ScriptModule,
) -> Tuple[bytes, OrderedDict]:
    """
    Remove tensors from a TorchScript model and return the model structure
    with its weights separately.

    Unlike regular PyTorch models, TorchScript models are compiled and have
    a frozen graph structure. This function:
    1. Extracts all parameters/buffers as NumPy arrays (for zero-copy via Ray)
    2. Serializes the model structure

    :param m: A TorchScript model (torch.jit.ScriptModule)
    :type m: torch.jit.ScriptModule

    :returns: A tuple with two elements:
              * Serialized model structure (bytes)
              * Dictionary mapping parameter names to NumPy arrays
    """
    if not isinstance(m, torch.jit.ScriptModule):
        raise TypeError(
            f"Expected torch.jit.ScriptModule, got {type(m)}. "
            "Use ray_zerocopy.extract_tensors() for regular torch.nn.Module models."
        )

    # Extract all parameters and buffers as numpy arrays
    state_dict = m.state_dict()
    tensors_as_numpy = OrderedDict()

    for name, tensor in state_dict.items():
        # Clone and detach to ensure we have our own copy
        tensors_as_numpy[name] = torch.clone(tensor).detach().cpu().numpy()

    # Serialize the model structure
    # Note: TorchScript serialization will include the current weights,
    # but we'll replace them when loading
    buffer = io.BytesIO()
    torch.jit.save(m, buffer)
    model_bytes = buffer.getvalue()

    return model_bytes, tensors_as_numpy


def _make_tensor_from_array(array):
    """
    Create a PyTorch tensor from a NumPy array, avoiding copies if possible.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        return torch.as_tensor(array)


def replace_tensors(
    model_bytes: bytes, tensors: OrderedDict
) -> torch.jit.ScriptModule:
    """
    Reconstruct a TorchScript model from serialized structure and weights.

    This is the inverse of extract_tensors(). It loads a TorchScript model
    from its serialized form and restores the actual weights using zero-copy
    operations when possible.

    :param model_bytes: Serialized TorchScript model structure (from torch.jit.save)
    :param tensors: Dictionary mapping parameter names to NumPy arrays

    :returns: A fully functional TorchScript model ready for inference
    """
    # Deserialize the model structure
    buffer = io.BytesIO(model_bytes)
    model = torch.jit.load(buffer)

    # Convert numpy arrays back to tensors (zero-copy when possible)
    state_dict = OrderedDict()
    for name, array in tensors.items():
        try:
            tensor = _make_tensor_from_array(array)
        except Exception:
            # Fallback to copy if zero-copy fails
            tensor = torch.as_tensor(array.copy())
        state_dict[name] = tensor

    # Load the real weights into the model
    model.load_state_dict(state_dict)

    # Set to eval mode for inference
    model.eval()

    return model


def extract_tensors_minimal(
    m: torch.jit.ScriptModule,
) -> Tuple[bytes, OrderedDict]:
    """
    A more aggressive version of extract_tensors() that creates a smaller
    skeleton by replacing weights with minimal tensors.

    WARNING: This may not work with all TorchScript models, especially those
    that have shape-dependent logic in the compiled graph. Use with caution
    and test thoroughly.

    :param m: A TorchScript model (torch.jit.ScriptModule)

    :returns: A tuple of (serialized model bytes, tensors as numpy arrays)
    """
    if not isinstance(m, torch.jit.ScriptModule):
        raise TypeError(
            f"Expected torch.jit.ScriptModule, got {type(m)}. "
            "Use ray_zerocopy.extract_tensors() for regular torch.nn.Module models."
        )

    # Extract all parameters and buffers as numpy arrays
    original_state_dict = m.state_dict()
    tensors_as_numpy = OrderedDict()

    for name, tensor in original_state_dict.items():
        tensors_as_numpy[name] = torch.clone(tensor).detach().cpu().numpy()

    # Create a state dict with minimal (bool) tensors to reduce skeleton size
    minimal_state_dict = OrderedDict()
    for name, tensor in original_state_dict.items():
        # Use bool dtype (1 byte per element) with same shape
        minimal_state_dict[name] = torch.zeros(
            tensor.shape, dtype=torch.bool, device="cpu"
        )

    # Try to load minimal state dict
    try:
        # Make a copy to avoid corrupting the original
        m_copy = copy.deepcopy(m)
        m_copy.load_state_dict(minimal_state_dict)
        buffer = io.BytesIO()
        torch.jit.save(m_copy, buffer)
        model_bytes = buffer.getvalue()
        return model_bytes, tensors_as_numpy
    except Exception as e:
        # Fall back to the regular method
        warnings.warn(
            f"Minimal skeleton creation failed: {e}. "
            "Falling back to extract_tensors()."
        )
        # Use the regular extraction method
        return extract_tensors(m)
