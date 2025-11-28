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
TorchScript (torch.jit) zero-copy support for Ray.

This submodule provides zero-copy model loading for TorchScript models,
similar to the main ray_zerocopy module but specifically designed for
compiled/traced PyTorch models.

Usage:
    from ray_zerocopy.jit import extract_tensors, replace_tensors

    # Extract from TorchScript model
    jit_model = torch.jit.trace(model, example)
    model_bytes, tensors = extract_tensors(jit_model)

    # Restore with zero-copy
    restored = replace_tensors(model_bytes, tensors)
"""

from . import actor
from .invoke import call_model, rewrite_pipeline
from .public import ZeroCopyModel
from .rewrite import extract_tensors, extract_tensors_minimal, replace_tensors

__all__ = [
    "extract_tensors",
    "replace_tensors",
    "extract_tensors_minimal",
    "call_model",
    "rewrite_pipeline",
    "ZeroCopyModel",
    "actor",
]
