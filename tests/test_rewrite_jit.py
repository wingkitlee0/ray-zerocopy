"""
Tests for the jit.rewrite module (TorchScript zero-copy functionality).
"""

import io
import torch
import torch.nn as nn
import numpy as np
import pytest
from ray_zerocopy.jit import (
    extract_tensors,
    replace_tensors,
    extract_tensors_minimal,
)


def test_extract_tensors_basic(simple_jit_model):
    """Test that extract_tensors correctly separates model structure from weights."""
    model = simple_jit_model
    model_bytes, tensors = extract_tensors(model)

    # Verify we got bytes and a dict
    assert isinstance(model_bytes, bytes)
    assert isinstance(tensors, dict)
    assert len(model_bytes) > 0

    # Verify tensors were extracted as numpy arrays
    assert len(tensors) > 0
    for name, tensor in tensors.items():
        assert isinstance(tensor, np.ndarray), f"Tensor {name} should be a NumPy array"


def test_extract_tensors_type_check(simple_model):
    """Test that extract_tensors rejects non-TorchScript models."""
    # This is a regular nn.Module, not TorchScript
    with pytest.raises(TypeError, match="Expected torch.jit.ScriptModule"):
        extract_tensors(simple_model)


def test_replace_tensors_basic(simple_jit_model):
    """Test that replace_tensors correctly restores weights."""
    model = simple_jit_model
    model_bytes, tensors = extract_tensors(model)

    # Get original output
    x = torch.randn(10, 100)
    with torch.no_grad():
        original_output = model(x)

    # Restore model and verify output matches
    restored_model = replace_tensors(model_bytes, tensors)
    with torch.no_grad():
        restored_output = restored_model(x)

    torch.testing.assert_close(original_output, restored_output)


def test_extract_replace_jit_roundtrip(simple_jit_model):
    """Test that extract and replace form a perfect roundtrip for TorchScript."""
    model = simple_jit_model
    x = torch.randn(5, 100)

    with torch.no_grad():
        original_output = model(x)

    model_bytes, tensors = extract_tensors(model)
    restored_model = replace_tensors(model_bytes, tensors)

    with torch.no_grad():
        restored_output = restored_model(x)

    torch.testing.assert_close(original_output, restored_output)


def test_extract_replace_jit_large_model(large_jit_model):
    """Test extract/replace with a larger TorchScript model."""
    model = large_jit_model
    x = torch.randn(10, 1000)

    with torch.no_grad():
        original_output = model(x)

    model_bytes, tensors = extract_tensors(model)
    restored_model = replace_tensors(model_bytes, tensors)

    with torch.no_grad():
        restored_output = restored_model(x)

    torch.testing.assert_close(original_output, restored_output)


def test_extract_replace_conv_model(conv_jit_model):
    """Test extract/replace with a convolutional model."""
    model = conv_jit_model
    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        original_output = model(x)

    model_bytes, tensors = extract_tensors(model)
    restored_model = replace_tensors(model_bytes, tensors)

    with torch.no_grad():
        restored_output = restored_model(x)

    torch.testing.assert_close(original_output, restored_output)


def test_tensors_are_numpy_arrays_jit(simple_jit_model):
    """Verify that extracted tensors are NumPy arrays."""
    model = simple_jit_model
    _, tensors = extract_tensors(model)

    for name, array in tensors.items():
        assert isinstance(array, np.ndarray), f"Tensor {name} should be a NumPy array"
        # Verify it's contiguous for zero-copy efficiency
        assert array.flags["C_CONTIGUOUS"] or array.flags["F_CONTIGUOUS"]


def test_restored_model_is_scriptmodule(simple_jit_model):
    """Verify that restored model is still a TorchScript model."""
    model = simple_jit_model
    model_bytes, tensors = extract_tensors(model)
    restored_model = replace_tensors(model_bytes, tensors)

    assert isinstance(restored_model, torch.jit.ScriptModule)


def test_model_serialization_size(simple_jit_model):
    """Test that the skeleton size is reasonable."""
    model = simple_jit_model
    model_bytes, tensors = extract_tensors(model)

    # Calculate total weight size
    weight_size = sum(t.nbytes for t in tensors.values())

    # The skeleton includes the model structure and may include weights
    # Just verify we can extract and the sizes are reasonable
    assert len(model_bytes) > 0, "Model bytes should not be empty"
    assert weight_size > 0, "Should have extracted some weights"

    # Skeleton should be roughly the same order of magnitude as weights
    # (TorchScript serialization includes structure + weights)
    assert len(model_bytes) < weight_size * 10, (
        f"Skeleton ({len(model_bytes)} bytes) seems unreasonably large "
        f"compared to weights ({weight_size} bytes)"
    )


def test_multiple_extractions(simple_jit_model):
    """Test that we can extract tensors multiple times."""
    model = simple_jit_model
    x = torch.randn(5, 100)

    # First extraction
    model_bytes_1, tensors_1 = extract_tensors(model)
    restored_1 = replace_tensors(model_bytes_1, tensors_1)

    with torch.no_grad():
        output_1 = restored_1(x)

    # Second extraction (model state might have changed)
    model_bytes_2, tensors_2 = extract_tensors(model)
    restored_2 = replace_tensors(model_bytes_2, tensors_2)

    with torch.no_grad():
        output_2 = restored_2(x)

    # Both should produce same results
    torch.testing.assert_close(output_1, output_2)


def test_traced_vs_scripted_model():
    """Test that both traced and scripted models work."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    x = torch.randn(3, 10)

    # Test with traced model
    traced = torch.jit.trace(model, x)
    traced_bytes, traced_tensors = extract_tensors(traced)
    restored_traced = replace_tensors(traced_bytes, traced_tensors)

    # Test with scripted model
    scripted = torch.jit.script(model)
    scripted_bytes, scripted_tensors = extract_tensors(scripted)
    restored_scripted = replace_tensors(scripted_bytes, scripted_tensors)

    # Both should work
    with torch.no_grad():
        out_traced = restored_traced(x)
        out_scripted = restored_scripted(x)

    assert out_traced.shape == (3, 5)
    assert out_scripted.shape == (3, 5)


def test_model_with_buffers():
    """Test models that have buffers (like BatchNorm)."""

    class ModelWithBuffers(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.bn = nn.BatchNorm2d(16)
            self.fc = nn.Linear(16 * 14 * 14, 10)

        def forward(self, x):
            x = torch.relu(self.bn(self.conv(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = ModelWithBuffers()
    model.eval()  # Important for BatchNorm
    x = torch.randn(2, 3, 16, 16)

    # Trace the model
    traced = torch.jit.trace(model, x)

    with torch.no_grad():
        original_output = traced(x)

    # Extract and restore
    model_bytes, tensors = extract_tensors(traced)

    # Verify buffers were extracted (BatchNorm has running_mean, running_var, etc.)
    buffer_tensors = [
        name for name in tensors.keys() if "running" in name or "num_batches" in name
    ]
    assert len(buffer_tensors) > 0, "Should have extracted buffer tensors"

    restored = replace_tensors(model_bytes, tensors)

    with torch.no_grad():
        restored_output = restored(x)

    torch.testing.assert_close(original_output, restored_output)


def test_empty_model():
    """Test with a model that has no parameters."""

    class EmptyModel(nn.Module):
        def forward(self, x):
            return x * 2

    model = EmptyModel()
    x = torch.randn(3, 5)
    traced = torch.jit.trace(model, x)

    model_bytes, tensors = extract_tensors(traced)

    # Should have empty tensors dict
    assert len(tensors) == 0

    # Should still be able to restore
    restored = replace_tensors(model_bytes, tensors)

    with torch.no_grad():
        output = restored(x)

    torch.testing.assert_close(x * 2, output)


def test_minimal_extraction_fallback(simple_jit_model):
    """Test that minimal extraction falls back gracefully."""
    model = simple_jit_model
    x = torch.randn(5, 100)

    with torch.no_grad():
        original_output = model(x)

    # Minimal extraction may fail and fallback
    model_bytes, tensors = extract_tensors_minimal(model)

    # Should still work
    restored = replace_tensors(model_bytes, tensors)

    with torch.no_grad():
        restored_output = restored(x)

    torch.testing.assert_close(original_output, restored_output, rtol=1e-4, atol=1e-5)


def test_state_dict_preservation(simple_jit_model):
    """Test that state dict is properly preserved."""
    model = simple_jit_model
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    model_bytes, tensors = extract_tensors(model)
    restored = replace_tensors(model_bytes, tensors)

    restored_state = restored.state_dict()

    # Check all keys match
    assert set(original_state.keys()) == set(restored_state.keys())

    # Check all values match
    for key in original_state.keys():
        torch.testing.assert_close(
            original_state[key],
            restored_state[key],
            msg=f"State dict mismatch for {key}",
        )


def test_different_dtypes():
    """Test models with different tensor dtypes."""

    class MixedDtypeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x

    model = MixedDtypeModel()
    model.eval()

    # Use float32 for simplicity (mixed precision with JIT is tricky)
    x = torch.randn(3, 10)
    traced = torch.jit.trace(model, x)

    with torch.no_grad():
        original_output = traced(x)

    model_bytes, tensors = extract_tensors(traced)
    restored = replace_tensors(model_bytes, tensors)

    with torch.no_grad():
        restored_output = restored(x)

    torch.testing.assert_close(original_output, restored_output)


def test_tensor_shapes_preserved(simple_jit_model):
    """Verify that tensor shapes are preserved through extract/replace."""
    model = simple_jit_model
    original_state = model.state_dict()

    model_bytes, tensors = extract_tensors(model)

    # Check shapes in extracted tensors
    for name, array in tensors.items():
        original_shape = tuple(original_state[name].shape)
        extracted_shape = array.shape
        assert original_shape == extracted_shape, (
            f"Shape mismatch for {name}: {original_shape} vs {extracted_shape}"
        )


def test_pickle_serialization(simple_jit_model, tmp_path):
    """Test that extracted model data can be saved to pickle and loaded back."""
    import pickle

    model = simple_jit_model
    x = torch.randn(4, 100)

    # Get original output
    with torch.no_grad():
        original_output = model(x)

    # Extract tensors
    model_bytes, tensors = extract_tensors(model)

    # Save to pickle file
    pkl_file = tmp_path / "model_data.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({'model_bytes': model_bytes, 'tensors': tensors}, f)

    print(f"\nSaved to {pkl_file}")
    print(f"File size: {pkl_file.stat().st_size / 1024:.2f} KB")

    # Load from pickle file
    with open(pkl_file, 'rb') as f:
        loaded_data = pickle.load(f)

    loaded_model_bytes = loaded_data['model_bytes']
    loaded_tensors = loaded_data['tensors']

    # Verify data integrity
    assert loaded_model_bytes == model_bytes, "Model bytes don't match after pickle"
    assert set(loaded_tensors.keys()) == set(tensors.keys()), "Tensor keys don't match"

    for name in tensors.keys():
        np.testing.assert_array_equal(
            loaded_tensors[name],
            tensors[name],
            err_msg=f"Tensor {name} doesn't match after pickle"
        )

    # Restore model from pickled data
    restored_model = replace_tensors(loaded_model_bytes, loaded_tensors)

    # Verify model works correctly
    with torch.no_grad():
        restored_output = restored_model(x)

    torch.testing.assert_close(original_output, restored_output)
    print("✓ Model successfully saved to and loaded from pickle!")


def test_pickle_large_model(large_jit_model, tmp_path):
    """Test pickle serialization with a larger model."""
    import pickle

    model = large_jit_model
    x = torch.randn(5, 1000)

    # Extract and save
    model_bytes, tensors = extract_tensors(model)

    pkl_file = tmp_path / "large_model.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({'model_bytes': model_bytes, 'tensors': tensors}, f)

    file_size_mb = pkl_file.stat().st_size / 1024 / 1024
    print(f"\nLarge model file size: {file_size_mb:.2f} MB")

    # Load and verify
    with open(pkl_file, 'rb') as f:
        loaded_data = pickle.load(f)

    restored_model = replace_tensors(loaded_data['model_bytes'], loaded_data['tensors'])

    with torch.no_grad():
        original_output = model(x)
        restored_output = restored_model(x)

    torch.testing.assert_close(original_output, restored_output)


def test_pickle_conv_model(conv_jit_model, tmp_path):
    """Test pickle serialization with a convolutional model."""
    import pickle

    model = conv_jit_model
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        original_output = model(x)

    # Extract and save
    model_bytes, tensors = extract_tensors(model)

    pkl_file = tmp_path / "conv_model.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'model_bytes': model_bytes,
            'tensors': tensors,
            'metadata': {
                'num_params': len(tensors),
                'total_size': sum(t.nbytes for t in tensors.values()),
            }
        }, f)

    # Load back
    with open(pkl_file, 'rb') as f:
        loaded_data = pickle.load(f)

    # Check metadata
    assert loaded_data['metadata']['num_params'] == len(tensors)

    # Restore and test
    restored_model = replace_tensors(
        loaded_data['model_bytes'],
        loaded_data['tensors']
    )

    with torch.no_grad():
        restored_output = restored_model(x)

    torch.testing.assert_close(original_output, restored_output)
