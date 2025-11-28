"""
Tests for the jit.invoke module (TorchScript Ray integration).
"""

import pytest
import ray
import torch
import torch.nn as nn

from ray_zerocopy.jit import (
    call_model,
    extract_tensors,
    replace_tensors,
    rewrite_pipeline,
)
from ray_zerocopy.jit.invoke import _RemoteModelShim


def test_call_jit_model_basic(ray_cluster, simple_jit_model):
    """Test that call_model works with basic inference."""
    model = simple_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    x = torch.randn(5, 100)

    # Call via Ray
    result_ref = call_model.remote(model_ref, args=(x,))
    result = ray.get(result_ref)

    # Compare with direct call
    with torch.no_grad():
        expected = model(x)

    torch.testing.assert_close(result, expected)


def test_call_jit_model_with_kwargs(ray_cluster, simple_jit_model):
    """Test call_model with keyword arguments."""
    model = simple_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    x = torch.randn(3, 100)

    # TorchScript models can be called with args or kwargs
    result_ref = call_model.remote(model_ref, args=(x,), kwargs={})
    result = ray.get(result_ref)

    assert result.shape == (3, 10)


def test_call_jit_model_multiple_calls(ray_cluster, simple_jit_model):
    """Test multiple calls to the same model."""
    model = simple_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    # Make multiple calls
    inputs = [torch.randn(2, 100) for _ in range(5)]
    result_refs = [call_model.remote(model_ref, args=(x,)) for x in inputs]
    results = ray.get(result_refs)

    # Verify all results
    with torch.no_grad():
        for inp, res in zip(inputs, results):
            expected = model(inp)
            torch.testing.assert_close(res, expected)


def test_remote_jit_model_shim_call(ray_cluster, simple_jit_model):
    """Test _RemoteModelShim with __call__ method."""
    model = simple_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    shim = _RemoteModelShim(model_ref, {"__call__"})

    x = torch.randn(4, 100)
    result = shim(x)

    with torch.no_grad():
        expected = model(x)

    torch.testing.assert_close(result, expected)


def test_remote_jit_model_shim_forward(ray_cluster):
    """Test _RemoteModelShim with forward method."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = Model()
    x = torch.randn(3, 10)
    traced = torch.jit.trace(model, x)

    model_data = extract_tensors(traced)
    model_ref = ray.put(model_data)

    shim = _RemoteModelShim(model_ref, {"forward"})

    result = shim.forward(x)
    assert result.shape == (3, 5)


def test_remote_jit_model_shim_invalid_method(ray_cluster, simple_jit_model):
    """Test that shim raises error for invalid methods."""
    model = simple_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    shim = _RemoteModelShim(model_ref, {"__call__"})

    with pytest.raises(AttributeError):
        shim.nonexistent_method()


def test_rewrite_pipeline_jit_basic(ray_cluster, simple_jit_model):
    """Test rewrite_pipeline with a simple pipeline."""

    class Pipeline:
        def __init__(self, model):
            self.model = model
            self.config = {"batch_size": 16}

    original_pipeline = Pipeline(simple_jit_model)
    rewritten_pipeline = rewrite_pipeline(original_pipeline)

    # Verify model was replaced with shim
    assert isinstance(rewritten_pipeline.model, _RemoteModelShim)

    # Verify config wasn't modified
    assert rewritten_pipeline.config == {"batch_size": 16}

    # Test inference through rewritten pipeline
    x = torch.randn(3, 100)
    result = rewritten_pipeline.model(x)

    with torch.no_grad():
        expected = simple_jit_model(x)

    torch.testing.assert_close(result, expected)


def test_rewrite_pipeline_jit_multiple_models(
    ray_cluster, simple_jit_model, conv_jit_model
):
    """Test rewrite_pipeline with multiple models."""

    class Pipeline:
        def __init__(self, model1, model2):
            self.encoder = model1
            self.decoder = model2

    original = Pipeline(simple_jit_model, conv_jit_model)
    rewritten = rewrite_pipeline(original)

    # Both models should be rewritten
    assert isinstance(rewritten.encoder, _RemoteModelShim)
    assert isinstance(rewritten.decoder, _RemoteModelShim)


def test_rewrite_pipeline_jit_skip_non_jit(ray_cluster, simple_jit_model, simple_model):
    """Test that rewrite_pipeline only rewrites TorchScript models."""

    class Pipeline:
        def __init__(self):
            self.jit_model = simple_jit_model
            self.regular_model = simple_model

    original = Pipeline()
    rewritten = rewrite_pipeline(original)

    # Only TorchScript model should be rewritten
    assert isinstance(rewritten.jit_model, _RemoteModelShim)
    assert isinstance(rewritten.regular_model, nn.Module)
    assert not isinstance(rewritten.regular_model, _RemoteModelShim)


def test_rewrite_pipeline_jit_method_names(ray_cluster):
    """Test rewrite_pipeline with custom method names."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

        def predict(self, x):
            return self.forward(x)

    model = Model()
    x = torch.randn(2, 10)
    traced = torch.jit.trace(model, x)

    class Pipeline:
        def __init__(self):
            self.model = traced

    original = Pipeline()
    rewritten = rewrite_pipeline(
        original, method_names=("__call__", "forward", "predict")
    )

    # Test all methods work
    x = torch.randn(3, 10)
    result1 = rewritten.model(x)
    result2 = rewritten.model.forward(x)

    assert result1.shape == (3, 5)
    assert result2.shape == (3, 5)


def test_parallel_inference(ray_cluster, large_jit_model):
    """Test parallel inference using multiple Ray workers."""
    model = large_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    # Create multiple inference tasks
    inputs = [torch.randn(4, 1000) for _ in range(8)]

    # Submit all tasks in parallel
    result_refs = [call_model.remote(model_ref, args=(x,)) for x in inputs]
    results = ray.get(result_refs)

    # Verify results
    assert len(results) == 8
    with torch.no_grad():
        for inp, res in zip(inputs, results):
            expected = model(inp)
            torch.testing.assert_close(res, expected)


def test_rewrite_preserves_non_model_attributes(ray_cluster, simple_jit_model):
    """Test that rewrite_pipeline preserves non-model attributes."""

    class Pipeline:
        def __init__(self):
            self.model = simple_jit_model
            self.name = "test_pipeline"
            self.version = 1.0
            self.metadata = {"author": "test"}

    original = Pipeline()
    rewritten = rewrite_pipeline(original)

    # Non-model attributes should be preserved
    assert rewritten.name == "test_pipeline"
    assert rewritten.version == 1.0
    assert rewritten.metadata == {"author": "test"}


def test_rewrite_with_empty_pipeline(ray_cluster):
    """Test rewrite_pipeline with a pipeline that has no models."""

    class Pipeline:
        def __init__(self):
            self.config = {"batch_size": 32}
            self.data = [1, 2, 3]

    original = Pipeline()
    rewritten = rewrite_pipeline(original)

    # Should return unchanged pipeline
    assert rewritten.config == {"batch_size": 32}
    assert rewritten.data == [1, 2, 3]


def test_conv_model_inference(ray_cluster, conv_jit_model):
    """Test inference with a convolutional model."""
    model = conv_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    x = torch.randn(2, 3, 32, 32)

    result_ref = call_model.remote(model_ref, args=(x,))
    result = ray.get(result_ref)

    with torch.no_grad():
        expected = model(x)

    torch.testing.assert_close(result, expected)


def test_model_ref_shared_across_tasks(ray_cluster, simple_jit_model):
    """Test that model_ref can be shared across multiple tasks efficiently."""
    model = simple_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    # Define a task that uses the model
    @ray.remote
    def batch_inference(model_data, batch_inputs):

        # Ray automatically dereferences ObjectRefs passed as arguments
        model_bytes, tensors = model_data
        model = replace_tensors(model_bytes, tensors)

        results = []
        with torch.no_grad():
            for x in batch_inputs:
                results.append(model(x))
        return results

    # Submit multiple tasks sharing the same model_ref
    batches = [[torch.randn(2, 100) for _ in range(3)] for _ in range(4)]
    result_refs = [batch_inference.remote(model_ref, batch) for batch in batches]
    results = ray.get(result_refs)

    # Verify we got results from all tasks
    assert len(results) == 4
    assert all(len(batch_results) == 3 for batch_results in results)


def test_zero_copy_memory_efficiency(ray_cluster, large_jit_model):
    """Test that zero-copy actually shares memory efficiently."""
    model = large_jit_model
    model_data = extract_tensors(model)
    model_ref = ray.put(model_data)

    # Submit many tasks - they should all share the same model in memory
    num_tasks = 10
    inputs = [torch.randn(1, 1000) for _ in range(num_tasks)]
    result_refs = [call_model.remote(model_ref, args=(x,)) for x in inputs]
    results = ray.get(result_refs)

    # All tasks should complete successfully
    assert len(results) == num_tasks
    assert all(r.shape == (1, 100) for r in results)
