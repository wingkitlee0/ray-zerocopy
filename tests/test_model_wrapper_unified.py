"""
Tests for the unified ModelWrapper API with both task and actor modes.
"""

import ray
import torch
import torch.nn as nn

from ray_zerocopy import ModelWrapper


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def __call__(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class SimplePipeline:
    """Simple pipeline with multiple models."""

    def __init__(self):
        self.encoder = SimpleModel(10, 20, 15)
        self.decoder = SimpleModel(15, 20, 5)

    def __call__(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


def test_model_wrapper_task_mode_basic():
    """Test ModelWrapper in task mode with basic functionality."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="task")

    # Load the rewritten pipeline
    rewritten = wrapper.load()

    # Should be callable
    test_input = torch.randn(3, 10)
    result = rewritten(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_task_mode_pipeline():
    """Test ModelWrapper in task mode with a pipeline."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    pipeline = SimplePipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="task")

    # Load the rewritten pipeline
    rewritten = wrapper.load()

    # Should be callable
    test_input = torch.randn(3, 10)
    result = rewritten(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_actor_mode_basic():
    """Test ModelWrapper in actor mode with basic functionality."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="actor")

    # Should not be callable directly (ModelWrapper is never callable)
    try:
        test_input = torch.randn(3, 10)
        result = wrapper(test_input)
        assert False, "Should not be able to call ModelWrapper directly"
    except (TypeError, AttributeError):
        pass  # ModelWrapper is not callable

    # Should be loadable in actor
    loaded = wrapper.load()
    result = loaded(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_actor_mode_pipeline():
    """Test ModelWrapper in actor mode with a pipeline."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    pipeline = SimplePipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="actor")

    # Load the pipeline
    loaded = wrapper.load()

    # Test inference
    test_input = torch.randn(3, 10)
    result = loaded(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_pickling_task_mode():
    """Test that task mode wrapper can be pickled and unpickled."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    import pickle

    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="task")

    # Pickle and unpickle
    pickled = pickle.dumps(wrapper)
    restored = pickle.loads(pickled)

    # Load the rewritten pipeline and test
    rewritten = restored.load()
    test_input = torch.randn(3, 10)
    result = rewritten(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_pickling_actor_mode():
    """Test that actor mode wrapper can be pickled and unpickled."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    import pickle

    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="actor")

    # Pickle and unpickle
    pickled = pickle.dumps(wrapper)
    restored = pickle.loads(pickled)

    # Should still work after loading
    loaded = restored.load()
    test_input = torch.randn(3, 10)
    result = loaded(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_mode_validation():
    """Test that mode parameter is validated."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    model = SimpleModel()

    # Valid modes
    wrapper_task = ModelWrapper.from_model(model, mode="task")
    wrapper_actor = ModelWrapper.from_model(model, mode="actor")

    assert wrapper_task._mode == "task"
    assert wrapper_actor._mode == "actor"


def test_model_wrapper_task_mode_custom_methods():
    """Test ModelWrapper in task mode with custom method names."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    class ModelWithCustomMethod(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

        def predict(self, x):
            return self.forward(x)

    model = ModelWithCustomMethod()
    wrapper = ModelWrapper.from_model(
        model, mode="task", method_names=("forward", "predict")
    )

    # Load the rewritten pipeline
    rewritten = wrapper.load()

    # Should have access to the methods
    test_input = torch.randn(3, 10)
    result = rewritten.forward(test_input)
    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_standalone_vs_pipeline():
    """Test ModelWrapper handles both standalone models and pipelines correctly."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Test standalone model
    model = SimpleModel()
    wrapper_standalone = ModelWrapper.from_model(model, mode="task")
    assert wrapper_standalone._is_standalone_module is True

    # Test pipeline
    pipeline = SimplePipeline()
    wrapper_pipeline = ModelWrapper.from_model(pipeline, mode="task")
    assert wrapper_pipeline._is_standalone_module is False

    # Both should work after loading
    rewritten_standalone = wrapper_standalone.load()
    rewritten_pipeline = wrapper_pipeline.load()

    test_input = torch.randn(3, 10)
    result_standalone = rewritten_standalone(test_input)
    result_pipeline = rewritten_pipeline(test_input)

    assert result_standalone.shape == (3, 5)
    assert result_pipeline.shape == (3, 5)


def test_model_wrapper_serialize_deserialize():
    """Test ModelWrapper serialization and deserialization."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="actor")

    # Serialize
    serialized = wrapper.serialize()

    # Deserialize
    restored = ModelWrapper.deserialize(**serialized)

    # Should work after loading
    loaded = restored.load()
    test_input = torch.randn(3, 10)
    result = loaded(test_input)

    assert result.shape == (3, 5), "Output shape should be (3, 5)"


def test_model_wrapper_task_mode_attribute_access():
    """Test that task mode wrapper loads pipeline with correct attributes."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    pipeline = SimplePipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="task")

    # Load the rewritten pipeline
    rewritten = wrapper.load()

    # Should be able to access attributes from the rewritten pipeline
    # (though they'll be RemoteModelShims)
    assert hasattr(rewritten, "encoder")
    assert hasattr(rewritten, "decoder")


def test_model_wrapper_actor_mode_attribute_error():
    """Test that ModelWrapper doesn't forward attribute access."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    pipeline = SimplePipeline()
    wrapper = ModelWrapper.from_model(pipeline)

    # ModelWrapper doesn't forward attributes - it's not callable
    # Attributes should only be accessible after loading
    loaded = wrapper.load()
    assert hasattr(loaded, "encoder")
    assert hasattr(loaded, "decoder")
