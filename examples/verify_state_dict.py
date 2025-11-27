#!/usr/bin/env python
"""Quick test to verify TorchScript has state_dict API."""

import torch
import torch.nn as nn

# Create a simple model
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
model.eval()

# Trace it
example = torch.randn(1, 10)
jit_model = torch.jit.trace(model, example)

print(f"Model type: {type(jit_model)}")
print(f"Has state_dict: {hasattr(jit_model, 'state_dict')}")
print(f"Has load_state_dict: {hasattr(jit_model, 'load_state_dict')}")

# Try calling state_dict
print("\nCalling state_dict()...")
state = jit_model.state_dict()
print(f"State dict keys: {list(state.keys())}")
print(f"State dict type: {type(state)}")

# Try creating a new state dict
print("\nTrying to modify and reload state_dict...")
new_state = {k: v.clone() for k, v in state.items()}
jit_model.load_state_dict(new_state)
print("✓ load_state_dict() works!")

# Test inference still works
x = torch.randn(3, 10)
output = jit_model(x)
print(f"\n✓ Inference works! Output shape: {output.shape}")

# Test with scripted model too
print("\n" + "=" * 50)
print("Testing with torch.jit.script...")
scripted = torch.jit.script(model)
print(f"Has state_dict: {hasattr(scripted, 'state_dict')}")
print(f"Has load_state_dict: {hasattr(scripted, 'load_state_dict')}")

state2 = scripted.state_dict()
print(f"State dict keys: {list(state2.keys())}")
scripted.load_state_dict(state2)
print("✓ Scripted model also has state_dict API!")
