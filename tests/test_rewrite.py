"""
Tests for the rewrite module (extract_tensors and replace_tensors).
"""

import numpy as np
import torch

from ray_zerocopy._internal.zerocopy import (
    extract_tensors,
    replace_tensors,
    replace_tensors_direct,
)


def test_extract_tensors_basic(simple_model):
    """Test that extract_tensors correctly separates model structure from weights."""
    model = simple_model
    model_skeleton, tensors = extract_tensors(model)

    # Verify skeleton is a different object
    assert model_skeleton is not model

    # Verify skeleton has no parameters
    for _, module in model_skeleton.named_modules():
        for name, param in module.named_parameters(recurse=False):
            assert getattr(module, name) is None, f"Parameter {name} should be None"

    # Verify tensors were extracted
    assert len(tensors) > 0
    assert all("params" in t for t in tensors)
    assert all("buffers" in t for t in tensors)

    # Verify original model still has parameters
    assert sum(p.numel() for p in model.parameters()) > 0


def test_replace_tensors_basic(simple_model):
    """Test that replace_tensors correctly restores weights to skeleton."""
    model = simple_model
    model_skeleton, tensors = extract_tensors(model)

    # Get original output
    x = torch.randn(10, 100)
    with torch.no_grad():
        original_output = model(x)

    # Replace tensors and verify output matches
    replace_tensors(model_skeleton, tensors)
    with torch.no_grad():
        restored_output = model_skeleton(x)

    torch.testing.assert_close(original_output, restored_output)


def test_replace_tensors_direct_basic(simple_model):
    """Test that replace_tensors_direct correctly restores weights to skeleton."""
    model = simple_model
    model_skeleton, tensors = extract_tensors(model)

    # Get original output
    x = torch.randn(10, 100)
    with torch.no_grad():
        original_output = model(x)

    # Replace tensors directly (faster path)
    replace_tensors_direct(model_skeleton, tensors)
    with torch.no_grad():
        restored_output = model_skeleton(x)

    torch.testing.assert_close(original_output, restored_output)


def test_extract_replace_roundtrip(simple_model):
    """Test that extract and replace form a perfect roundtrip."""
    model = simple_model
    x = torch.randn(5, 100)

    with torch.no_grad():
        original_output = model(x)
    model_skeleton, tensors = extract_tensors(model)
    replace_tensors(model_skeleton, tensors)
    with torch.no_grad():
        restored_output = model_skeleton(x)

    torch.testing.assert_close(original_output, restored_output)


def test_extract_replace_large_model(large_model):
    """Test extract/replace with a larger model."""
    model = large_model
    x = torch.randn(10, 1000)

    with torch.no_grad():
        original_output = model(x)
    model_skeleton, tensors = extract_tensors(model)
    replace_tensors(model_skeleton, tensors)
    with torch.no_grad():
        restored_output = model_skeleton(x)

    torch.testing.assert_close(original_output, restored_output)


def test_extract_replace_transformer_model(transformer_like_model):
    """Test extract/replace with a transformer-like model."""
    model = transformer_like_model
    x = torch.randint(0, 10000, (4, 128))  # batch_size=4, seq_len=128

    with torch.no_grad():
        original_output = model(x)
    model_skeleton, tensors = extract_tensors(model)
    replace_tensors(model_skeleton, tensors)
    with torch.no_grad():
        restored_output = model_skeleton(x)

    torch.testing.assert_close(original_output, restored_output)


def test_tensors_are_numpy_arrays(simple_model):
    """Verify that extracted tensors are NumPy arrays, not PyTorch tensors."""
    model = simple_model
    _, tensors = extract_tensors(model)

    for tensor_dict in tensors:
        for name, param_array in tensor_dict["params"].items():
            assert isinstance(param_array, np.ndarray), (
                f"Parameter {name} should be a NumPy array"
            )
        for name, buf_array in tensor_dict["buffers"].items():
            assert isinstance(buf_array, np.ndarray), (
                f"Buffer {name} should be a NumPy array"
            )


def test_model_skeleton_is_in_eval_mode(simple_model):
    """Verify that extracted skeleton is in eval mode."""
    model = simple_model
    model.train()  # Set to training mode
    model_skeleton, _ = extract_tensors(model)

    assert not model_skeleton.training, "Skeleton should be in eval mode"


def test_replace_tensors_inference_mode(simple_model):
    """Test that replace_tensors works correctly in inference mode."""
    model = simple_model
    model_skeleton, tensors = extract_tensors(model)

    # This should not raise an error
    replace_tensors(model_skeleton, tensors)

    # Verify model works
    x = torch.randn(5, 100)
    with torch.no_grad():
        output = model_skeleton(x)
    assert output.shape == (5, 10)
