#!/usr/bin/env python
"""
Quick test script to verify TorchScript zero-copy functionality.
"""

import sys

import torch
import torch.nn as nn

print("Testing TorchScript zero-copy implementation...")
print("=" * 60)

# Test 1: Basic import
print("\n✓ Test 1: Import modules")
try:
    from ray_zerocopy.jit import (
        extract_tensors,
        replace_tensors,
    )

    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Extract and replace tensors
print("\n✓ Test 2: Extract and replace tensors")
try:
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    model.eval()

    # Convert to TorchScript
    example_input = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example_input)

    # Test input
    test_input = torch.randn(3, 10)
    with torch.no_grad():
        original_output = jit_model(test_input)

    # Extract tensors
    model_bytes, tensors = extract_tensors(jit_model)
    print(f"  ✓ Extracted {len(tensors)} tensors")
    print(f"  ✓ Model skeleton size: {len(model_bytes):,} bytes")

    # Replace tensors
    restored_model = replace_tensors(model_bytes, tensors)
    with torch.no_grad():
        restored_output = restored_model(test_input)

    # Verify outputs match
    if torch.allclose(original_output, restored_output):
        print("  ✓ Outputs match perfectly!")
    else:
        print("  ✗ Outputs don't match")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Type checking
print("\n✓ Test 3: Type checking")
try:
    regular_model = nn.Linear(10, 5)
    try:
        extract_tensors(regular_model)
        print("  ✗ Should have raised TypeError")
        sys.exit(1)
    except TypeError as e:
        if "Expected torch.jit.ScriptModule" in str(e):
            print("  ✓ Type checking works correctly")
        else:
            print(f"  ✗ Wrong error message: {e}")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Model with BatchNorm (buffers)
print("\n✓ Test 4: Model with buffers (BatchNorm)")
try:

    class ModelWithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3)
            self.bn = nn.BatchNorm2d(8)

        def forward(self, x):
            return self.bn(self.conv(x))

    model = ModelWithBN()
    model.eval()

    x = torch.randn(2, 3, 8, 8)
    jit_model = torch.jit.trace(model, x)

    with torch.no_grad():
        original = jit_model(x)

    model_bytes, tensors = extract_tensors(jit_model)

    # Check for buffers
    buffer_count = sum(
        1 for name in tensors.keys() if "running" in name or "num_batches" in name
    )
    print(f"  ✓ Found {buffer_count} buffer tensors")

    restored = replace_tensors(model_bytes, tensors)
    with torch.no_grad():
        restored_out = restored(x)

    if torch.allclose(original, restored_out):
        print("  ✓ Model with buffers works correctly")
    else:
        print("  ✗ Outputs don't match")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Ray integration (if Ray is available)
print("\n✓ Test 5: Ray integration")
try:
    import ray

    from ray_zerocopy.jit import call_model

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create a model
    model = nn.Sequential(nn.Linear(10, 5))
    model.eval()
    example = torch.randn(1, 10)
    jit_model = torch.jit.trace(model, example)

    # Put in Ray object store
    model_data = extract_tensors(jit_model)
    model_ref = ray.put(model_data)

    # Call via Ray
    test_input = torch.randn(2, 10)
    result_ref = call_model.remote(model_ref, args=(test_input,))
    result = ray.get(result_ref)

    # Compare with direct call
    with torch.no_grad():
        expected = jit_model(test_input)

    if torch.allclose(result, expected):
        print("  ✓ Ray integration works correctly")
    else:
        print("  ✗ Ray outputs don't match")
        sys.exit(1)

    ray.shutdown()
except ImportError:
    print("  ⊘ Ray not available, skipping")
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    if ray.is_initialized():
        ray.shutdown()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
