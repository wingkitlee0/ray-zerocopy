"""
Tests for the invoke module (rewrite_pipeline and call_model).
"""

import ray
import torch

from ray_zerocopy._internal.zerocopy import call_model, extract_tensors
from ray_zerocopy.nn import rewrite_pipeline


def test_rewrite_pipeline_basic(sample_pipeline, simple_model, ray_cluster):
    """Test that rewrite_pipeline replaces models with shims."""
    Pipeline = sample_pipeline
    pipeline = Pipeline(simple_model)

    # Verify original pipeline has a real model
    assert isinstance(pipeline.model, torch.nn.Module)

    # Rewrite the pipeline
    rewritten = rewrite_pipeline(pipeline)

    # Verify rewritten pipeline has a shim (not the original model)
    assert not isinstance(rewritten.model, torch.nn.Module)
    assert hasattr(rewritten.model, "__call__")

    # Verify original pipeline is unchanged
    assert isinstance(pipeline.model, torch.nn.Module)


def test_rewrite_pipeline_functionality(sample_pipeline, simple_model, ray_cluster):
    """Test that rewritten pipeline produces same results as original."""
    Pipeline = sample_pipeline
    pipeline = Pipeline(simple_model)
    rewritten = rewrite_pipeline(pipeline)

    x = torch.randn(5, 100)

    # Get outputs
    original_output = pipeline.model(x)
    rewritten_output = rewritten.model(x)

    # Verify outputs match
    torch.testing.assert_close(original_output, rewritten_output)


def test_rewrite_pipeline_large_model(sample_pipeline, large_model, ray_cluster):
    """Test rewrite_pipeline with a larger model."""
    Pipeline = sample_pipeline
    pipeline = Pipeline(large_model)
    rewritten = rewrite_pipeline(pipeline)

    x = torch.randn(10, 1000)

    original_output = pipeline.model(x)
    rewritten_output = rewritten.model(x)

    torch.testing.assert_close(original_output, rewritten_output)


def test_call_model_remote(simple_model, ray_cluster):
    """Test the call_model remote function directly."""
    model_ref = ray.put(extract_tensors(simple_model))
    x = torch.randn(5, 100)

    # Call the model remotely
    result_ref = call_model.remote(model_ref, (x,), None, "__call__")  # type: ignore
    output = ray.get(result_ref)

    # Verify output shape
    assert isinstance(output, torch.Tensor)
    assert output.shape == (5, 10)

    # Verify it matches local execution
    with torch.no_grad():
        expected_output = simple_model(x)
    torch.testing.assert_close(output, expected_output)


def test_rewrite_pipeline_custom_method(sample_pipeline, simple_model, ray_cluster):
    """Test rewrite_pipeline with custom method names."""

    class ModelWithCustomMethod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

        def predict(self, x):
            return self.forward(x)

    Pipeline = sample_pipeline
    model = ModelWithCustomMethod()
    pipeline = Pipeline(model)
    rewritten = rewrite_pipeline(pipeline, method_names=("__call__", "predict"))

    x = torch.randn(3, 10)

    # Test both methods
    assert hasattr(rewritten.model, "predict")
    output1 = rewritten.model(x)
    output2 = rewritten.model.predict(x)

    expected = model(x)
    torch.testing.assert_close(output1, expected)
    torch.testing.assert_close(output2, expected)


def test_rewrite_pipeline_multiple_models(ray_cluster):
    """Test rewrite_pipeline with multiple models in pipeline."""

    class MultiModelPipeline:
        def __init__(self, model1, model2):
            self.model1 = model1
            self.model2 = model2

    model1 = torch.nn.Linear(10, 20)
    model2 = torch.nn.Linear(20, 5)

    pipeline = MultiModelPipeline(model1, model2)
    rewritten = rewrite_pipeline(pipeline)

    x = torch.randn(3, 10)

    # Both models should be rewritten
    assert not isinstance(rewritten.model1, torch.nn.Module)
    assert not isinstance(rewritten.model2, torch.nn.Module)

    # Test functionality
    output1 = rewritten.model1(x)
    output2 = rewritten.model2(output1)

    expected = model2(model1(x))
    torch.testing.assert_close(output2, expected)
