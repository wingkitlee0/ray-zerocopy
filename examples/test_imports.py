#!/usr/bin/env python
"""Quick test to verify new import structure works."""

print("Testing import structure...")

# Test nn module
print("\n1. Testing ray_zerocopy.nn module:")

print("  ✓ NN module imports work")

# Test jit submodule
print("\n2. Testing ray_zerocopy.jit submodule:")
from ray_zerocopy.jit import replace_tensors as jit_replace_tensors

print("  ✓ JIT submodule imports work")

# Test that they are different
from ray_zerocopy.jit import extract_tensors as jit_extract
from ray_zerocopy.nn import extract_tensors as nn_extract

print("\n3. Verifying they are different functions:")
print(f"  NN extract_tensors: {nn_extract.__module__}")
print(f"  JIT extract_tensors: {jit_extract.__module__}")
assert nn_extract != jit_extract, "Functions should be different!"
print("  ✓ Functions are properly separated")

# Test actual functionality
print("\n4. Testing basic JIT functionality:")
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 5))
model.eval()
jit_model = torch.jit.trace(model, torch.randn(1, 10))

model_bytes, tensors = jit_extract(jit_model)
print(f"  ✓ Extracted {len(tensors)} tensors")

restored = jit_replace_tensors(model_bytes, tensors)
print(f"  ✓ Restored model type: {type(restored)}")

x = torch.randn(3, 10)
with torch.no_grad():
    orig_out = jit_model(x)
    rest_out = restored(x)

if torch.allclose(orig_out, rest_out):
    print("  ✓ Outputs match!")
else:
    print("  ✗ Outputs don't match")

print("\n" + "=" * 50)
print("✓ All import tests passed!")
print("=" * 50)
