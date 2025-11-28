"""
Tests for the new unified wrapper API.

Tests TaskWrapper, ActorWrapper, JITTaskWrapper, and JITActorWrapper.
"""

import pytest
import torch
import torch.nn as nn

from ray_zerocopy import ActorWrapper, JITActorWrapper, JITTaskWrapper, TaskWrapper


# Test models and pipelines
class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class SimplePipeline:
    """Simple pipeline with one model."""

    def __init__(self):
        self.model = SimpleModel()

    def __call__(self, x):
        return self.model(x)

    def predict(self, x):
        return self(x)


class MultiModelPipeline:
    """Pipeline with multiple models."""

    def __init__(self):
        self.encoder = nn.Sequential(nn.Linear(10, 8), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(8, 5))

    def __call__(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


# ============================================================================
# TaskWrapper Tests
# ============================================================================


def test_task_wrapper_basic():
    """Test basic TaskWrapper functionality."""
    pipeline = SimplePipeline()
    wrapped = TaskWrapper(pipeline)

    # Test inference
    test_input = torch.randn(3, 10)
    result = wrapped(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_task_wrapper_custom_method():
    """Test TaskWrapper with custom method names."""
    pipeline = SimplePipeline()
    wrapped = TaskWrapper(pipeline, method_names=("predict",))

    # Test custom method
    test_input = torch.randn(3, 10)
    result = wrapped.predict(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_task_wrapper_multi_model():
    """Test TaskWrapper with multiple models."""
    pipeline = MultiModelPipeline()
    wrapped = TaskWrapper(pipeline)

    test_input = torch.randn(3, 10)
    result = wrapped(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


# ============================================================================
# ActorWrapper Tests
# ============================================================================


def test_actor_wrapper_basic():
    """Test basic ActorWrapper functionality."""
    pipeline = SimplePipeline()
    actor_wrapper = ActorWrapper(pipeline)

    # Load in current process (simulating actor)
    loaded_pipeline = actor_wrapper.load(device="cpu")

    # Test inference
    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_actor_wrapper_multi_model():
    """Test ActorWrapper with multiple models."""
    pipeline = MultiModelPipeline()
    actor_wrapper = ActorWrapper(pipeline)

    # Load in current process
    loaded_pipeline = actor_wrapper.load(device="cpu")

    # Test inference
    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_actor_wrapper_constructor_kwargs():
    """Test ActorWrapper.constructor_kwargs property."""
    pipeline = SimplePipeline()
    actor_wrapper = ActorWrapper(pipeline, use_fast_load=False)

    kwargs = actor_wrapper.constructor_kwargs

    assert "pipeline_skeleton" in kwargs
    assert "model_refs" in kwargs
    assert "use_fast_load" in kwargs
    assert kwargs["use_fast_load"] is False


def test_actor_wrapper_device_override():
    """Test ActorWrapper with device specification in load()."""
    pipeline = SimplePipeline()
    actor_wrapper = ActorWrapper(pipeline)

    # Load with device specification
    loaded_pipeline = actor_wrapper.load(device="cpu")

    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5)


# ============================================================================
# JITTaskWrapper Tests
# ============================================================================


def test_jit_task_wrapper_basic():
    """Test basic JITTaskWrapper functionality."""
    # Create and trace a model
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    # Wrap in pipeline
    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def __call__(self, x):
            return self.model(x)

    pipeline = JITPipeline()
    wrapped = JITTaskWrapper(pipeline)

    # Test inference
    test_input = torch.randn(3, 10)
    result = wrapped(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_jit_task_wrapper_forward_method():
    """Test JITTaskWrapper with forward method."""
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    # Pipeline that uses forward method
    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def forward(self, x):
            return self.model(x)

        def __call__(self, x):
            return self.forward(x)

    pipeline = JITPipeline()
    wrapped = JITTaskWrapper(pipeline, method_names=("forward",))

    # Test forward method
    test_input = torch.randn(3, 10)
    result = wrapped.forward(test_input)

    assert result.shape == (3, 5)


def test_jit_task_wrapper_multi_model():
    """Test JITTaskWrapper with multiple TorchScript models."""
    # Trace encoder
    encoder = nn.Sequential(nn.Linear(10, 8), nn.ReLU())
    encoder.eval()
    jit_encoder = torch.jit.trace(encoder, torch.randn(1, 10))

    # Trace decoder
    decoder = nn.Sequential(nn.Linear(8, 5))
    decoder.eval()
    jit_decoder = torch.jit.trace(decoder, torch.randn(1, 8))

    # Create pipeline
    class JITPipeline:
        def __init__(self):
            self.encoder = jit_encoder
            self.decoder = jit_decoder

        def __call__(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)

    pipeline = JITPipeline()
    wrapped = JITTaskWrapper(pipeline)

    # Test inference
    test_input = torch.randn(3, 10)
    result = wrapped(test_input)

    assert result.shape == (3, 5)


# ============================================================================
# JITActorWrapper Tests
# ============================================================================


def test_jit_actor_wrapper_basic():
    """Test basic JITActorWrapper functionality."""
    # Create and trace a model
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    # Wrap in pipeline
    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def __call__(self, x):
            return self.model(x)

    pipeline = JITPipeline()
    actor_wrapper = JITActorWrapper(pipeline)

    # Load in current process (simulating actor)
    loaded_pipeline = actor_wrapper.load(device="cpu")

    # Test inference
    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_jit_actor_wrapper_multi_model():
    """Test JITActorWrapper with multiple models."""
    # Trace encoder
    encoder = nn.Sequential(nn.Linear(10, 8), nn.ReLU())
    encoder.eval()
    jit_encoder = torch.jit.trace(encoder, torch.randn(1, 10))

    # Trace decoder
    decoder = nn.Sequential(nn.Linear(8, 5))
    decoder.eval()
    jit_decoder = torch.jit.trace(decoder, torch.randn(1, 8))

    # Create pipeline
    class JITPipeline:
        def __init__(self):
            self.encoder = jit_encoder
            self.decoder = jit_decoder

        def __call__(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)

    pipeline = JITPipeline()
    actor_wrapper = JITActorWrapper(pipeline)

    # Load in current process
    loaded_pipeline = actor_wrapper.load(device="cpu")

    # Test inference
    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5)


def test_jit_actor_wrapper_constructor_kwargs():
    """Test JITActorWrapper.constructor_kwargs property."""
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def __call__(self, x):
            return self.model(x)

    pipeline = JITPipeline()
    actor_wrapper = JITActorWrapper(pipeline)

    kwargs = actor_wrapper.constructor_kwargs

    assert "pipeline_skeleton" in kwargs
    assert "model_refs" in kwargs


def test_jit_actor_wrapper_device_override():
    """Test JITActorWrapper with device specification in load()."""
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def __call__(self, x):
            return self.model(x)

    pipeline = JITPipeline()
    actor_wrapper = JITActorWrapper(pipeline)

    # Load with device specification
    loaded_pipeline = actor_wrapper.load(device="cpu")

    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5)


# ============================================================================
# Integration Tests
# ============================================================================


def test_wrapper_api_consistency():
    """Test that all wrappers have consistent APIs."""
    # Create pipelines
    nn_pipeline = SimplePipeline()

    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def __call__(self, x):
            return self.model(x)

    jit_pipeline = JITPipeline()

    # All wrappers should accept similar constructor args
    task_wrapper = TaskWrapper(nn_pipeline)
    actor_wrapper = ActorWrapper(nn_pipeline)
    jit_task_wrapper = JITTaskWrapper(jit_pipeline)
    jit_actor_wrapper = JITActorWrapper(jit_pipeline)

    # Actor wrappers should have load() method
    assert hasattr(actor_wrapper, "load")
    assert hasattr(jit_actor_wrapper, "load")

    # Actor wrappers should have constructor_kwargs property
    assert hasattr(actor_wrapper, "constructor_kwargs")
    assert hasattr(jit_actor_wrapper, "constructor_kwargs")


def test_wrapper_determinism():
    """Test that wrapped models produce same results as originals."""
    # Create pipeline
    pipeline = SimplePipeline()

    # Get original output
    test_input = torch.randn(3, 10)
    with torch.no_grad():
        original_output = pipeline(test_input)

    # Test TaskWrapper
    wrapped = TaskWrapper(pipeline)
    wrapped_output = wrapped(test_input)
    assert torch.allclose(original_output, wrapped_output, rtol=1e-4)

    # Test ActorWrapper
    actor_wrapper = ActorWrapper(pipeline)
    loaded = actor_wrapper.load(device="cpu")
    with torch.no_grad():
        actor_output = loaded(test_input)
    assert torch.allclose(original_output, actor_output, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
