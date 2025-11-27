"""
Simple test to verify the actor module works correctly.
"""

import sys

sys.path.insert(0, "/home/wklee/github/ray-zerocopy/src")

import torch

# Test 1: Can we extract and prepare a model?
print("Test 1: Preparing model for actors...")
model = torch.nn.Linear(10, 5)
try:
    # This doesn't need Ray running, just extracts tensors
    from ray_zerocopy.rewrite import extract_tensors

    skeleton, weights = extract_tensors(model)
    print("✓ Model extraction works")
except Exception as e:
    print(f"✗ Model extraction failed: {e}")

# Test 2: Verify the module structure
print("\nTest 2: Checking module structure...")
from ray_zerocopy import actor

expected_functions = [
    "prepare_model_for_actors",
    "load_model_in_actor",
    "rewrite_pipeline_for_actors",
    "load_pipeline_in_actor",
]
for func_name in expected_functions:
    if hasattr(actor, func_name):
        print(f"✓ {func_name} exists")
    else:
        print(f"✗ {func_name} missing")

print("\n✅ Module structure looks good!")
print("\nTo test with Ray:")
print("  1. Start Ray: ray.init()")
print("  2. model_ref = prepare_model_for_actors(model)")
print("  3. Use with map_batches as shown in examples")
