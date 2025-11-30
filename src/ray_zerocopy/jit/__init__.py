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

from .actors import (
    load_jit_model_in_actor,
    load_pipeline_for_actors,
    model_info_to_model_refs,
    prepare_jit_model_for_actors,
    prepare_pipeline_for_actors,
)
from .rewrite import extract_tensors, extract_tensors_minimal, replace_tensors
from .tasks import (
    call_model,
    load_pipeline_for_tasks,
    prepare_pipeline_for_tasks,
    rewrite_pipeline,
)

__all__ = [
    # Actor-based functions
    "prepare_jit_model_for_actors",
    "load_jit_model_in_actor",
    "prepare_pipeline_for_actors",
    "load_pipeline_for_actors",
    "model_info_to_model_refs",
    # Task-based functions
    "prepare_pipeline_for_tasks",
    "load_pipeline_for_tasks",
    "rewrite_pipeline",
    "call_model",
    # Core rewrite functions
    "extract_tensors",
    "replace_tensors",
    "extract_tensors_minimal",
]
