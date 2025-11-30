"""
Tests for the new unified wrapper API.

Tests ModelWrapper.from_model(), for_tasks(), and JITModelWrapper.
"""

import pytest
import torch
import torch.nn as nn

from ray_zerocopy import JITModelWrapper, ModelWrapper


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


class JITPipeline:
    """Pipeline with a TorchScript model for testing."""

    def __init__(self, jit_model):
        self.model = jit_model
        self.custom_attr = "test_value"

    def __call__(self, x):
        return self.model(x)


# ============================================================================
# ModelWrapper Task Mode Tests
# ============================================================================


def test_model_wrapper_task_basic():
    """Test basic ModelWrapper task mode functionality."""
    pipeline = SimplePipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="task")

    # Load the rewritten pipeline
    rewritten = wrapper.load()

    # Test inference
    test_input = torch.randn(3, 10)
    result = rewritten(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_model_wrapper_for_tasks_shortcut():
    """Test ModelWrapper.for_tasks() shortcut."""
    pipeline = SimplePipeline()
    rewritten = ModelWrapper.for_tasks(pipeline)

    # Test inference
    test_input = torch.randn(3, 10)
    result = rewritten(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_model_wrapper_task_custom_method():
    """Test ModelWrapper task mode with custom method names."""
    pipeline = SimplePipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="task", method_names=("predict",))

    # Load the rewritten pipeline
    rewritten = wrapper.load()

    # Test custom method
    test_input = torch.randn(3, 10)
    result = rewritten.predict(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_task_multi_model():
    """Test ModelWrapper task mode with multiple models."""
    pipeline = MultiModelPipeline()
    rewritten = ModelWrapper.for_tasks(pipeline)

    test_input = torch.randn(3, 10)
    result = rewritten(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


# ============================================================================
# ModelWrapper Actor Mode Tests
# ============================================================================


def test_model_wrapper_actor_basic():
    """Test basic ModelWrapper actor mode functionality."""
    pipeline = SimplePipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="actor")

    # Load in current process (simulating actor)
    loaded_pipeline = wrapper.load()

    # Test inference
    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_model_wrapper_actor_multi_model():
    """Test ModelWrapper actor mode with multiple models."""
    pipeline = MultiModelPipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="actor")

    # Load in current process
    loaded_pipeline = wrapper.load()

    # Test inference
    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


# ============================================================================
# JITModelWrapper Task Mode Tests
# ============================================================================


def test_jit_model_wrapper_task_basic():
    """Test basic JITModelWrapper task mode functionality."""
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
    wrapped = JITModelWrapper.for_tasks(pipeline)

    # Test inference
    test_input = torch.randn(3, 10)
    result = wrapped(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_jit_model_wrapper_task_from_model():
    """Test JITModelWrapper.from_model() in task mode."""
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
    wrapper = JITModelWrapper.from_model(pipeline, mode="task")
    wrapped = wrapper.load()

    # Test inference
    test_input = torch.randn(3, 10)
    result = wrapped(test_input)

    assert result.shape == (3, 5)


def test_jit_model_wrapper_task_forward_method():
    """Test JITModelWrapper task mode with forward method."""
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
    wrapped = JITModelWrapper.for_tasks(pipeline, method_names=("forward",))

    # Test forward method
    test_input = torch.randn(3, 10)
    result = wrapped.forward(test_input)

    assert result.shape == (3, 5)


def test_jit_model_wrapper_task_multi_model():
    """Test JITModelWrapper task mode with multiple TorchScript models."""
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
    wrapped = JITModelWrapper.for_tasks(pipeline)

    # Test inference
    test_input = torch.randn(3, 10)
    result = wrapped(test_input)

    assert result.shape == (3, 5)


# ============================================================================
# JITModelWrapper Actor Mode Tests
# ============================================================================


def test_jit_model_wrapper_actor_basic():
    """Test basic JITModelWrapper actor mode functionality."""
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
    wrapper = JITModelWrapper.from_model(pipeline, mode="actor")

    # Load in current process (simulating actor)
    loaded_pipeline = wrapper.load()

    # Test inference
    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"
    assert isinstance(result, torch.Tensor), "Result should be a tensor"


def test_jit_model_wrapper_actor_multi_model():
    """Test JITModelWrapper actor mode with multiple models."""
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
    wrapper = JITModelWrapper.from_model(pipeline, mode="actor")

    # Load in current process
    loaded_pipeline = wrapper.load()

    # Test inference
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
    task_rewritten = ModelWrapper.for_tasks(
        nn_pipeline
    )  # Returns pipeline, not wrapper
    actor_wrapper = ModelWrapper.from_model(nn_pipeline, mode="actor")
    jit_task_rewritten = JITModelWrapper.for_tasks(jit_pipeline)
    jit_actor_wrapper = JITModelWrapper.from_model(jit_pipeline, mode="actor")

    # Actor wrappers should have load() method
    assert hasattr(jit_actor_wrapper, "load")

    # Task rewritten should be callable
    assert callable(task_rewritten)
    assert callable(jit_task_rewritten)


def test_wrapper_determinism():
    """Test that wrapped models produce same results as originals."""
    # Create pipeline
    pipeline = SimplePipeline()

    # Get original output
    test_input = torch.randn(3, 10)
    with torch.no_grad():
        original_output = pipeline(test_input)

    # Test ModelWrapper task mode
    rewritten = ModelWrapper.for_tasks(pipeline)
    wrapped_output = rewritten(test_input)
    assert torch.allclose(original_output, wrapped_output, rtol=1e-4)

    # Test ModelWrapper actor mode
    actor_wrapper = ModelWrapper.from_model(pipeline, mode="actor")
    loaded = actor_wrapper.load()
    with torch.no_grad():
        actor_output = loaded(test_input)
    assert torch.allclose(original_output, actor_output, rtol=1e-4)


def test_jit_model_wrapper_task_pickling():
    """Test JITModelWrapper task mode pickling (__getstate__ and __setstate__)."""
    import pickle

    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    pipeline = JITPipeline(jit_model)
    wrapper = JITModelWrapper.from_model(pipeline, mode="task")

    # Test pickling
    pickled = pickle.dumps(wrapper)
    unpickled = pickle.loads(pickled)

    # Test that unpickled wrapper still works
    wrapped = unpickled.load()
    test_input = torch.randn(3, 10)
    result = wrapped(test_input)
    assert result.shape == (3, 5)


def test_jit_model_wrapper_actor_pickling():
    """Test JITModelWrapper actor mode pickling (__getstate__ and __setstate__)."""
    import pickle

    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    pipeline = JITPipeline(jit_model)
    wrapper = JITModelWrapper.from_model(pipeline, mode="actor")

    # Test pickling
    pickled = pickle.dumps(wrapper)
    unpickled = pickle.loads(pickled)

    # Test that unpickled wrapper still works
    loaded_pipeline = unpickled.load()
    test_input = torch.randn(3, 10)
    result = loaded_pipeline(test_input)
    assert result.shape == (3, 5)


def test_jit_model_wrapper_getattr():
    """Test JITModelWrapper __getattr__ forwarding in task mode."""
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    pipeline = JITPipeline(jit_model)
    wrapped = JITModelWrapper.for_tasks(pipeline)

    # Test that custom attributes are accessible
    assert wrapped.custom_attr == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
