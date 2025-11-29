#!/usr/bin/env python3
"""
Quick verification script to demonstrate the new unified ModelWrapper API.
"""

import torch
import torch.nn as nn
import ray

from ray_zerocopy import ModelWrapper


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def __call__(self, x):
        return self.fc(x)


def verify_task_mode():
    """Verify task mode works correctly."""
    print("✓ Testing task mode...")
    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="task")
    
    test_input = torch.randn(3, 10)
    result = wrapper(test_input)
    
    assert result.shape == (3, 5), f"Expected shape (3, 5), got {result.shape}"
    print("  ✓ Task mode works correctly")


def verify_for_tasks_shortcut():
    """Verify for_tasks() shortcut works."""
    print("✓ Testing for_tasks() shortcut...")
    model = SimpleModel()
    wrapper = ModelWrapper.for_tasks(model)
    
    test_input = torch.randn(3, 10)
    result = wrapper(test_input)
    
    assert result.shape == (3, 5), f"Expected shape (3, 5), got {result.shape}"
    print("  ✓ for_tasks() shortcut works correctly")


def verify_actor_mode():
    """Verify actor mode works correctly."""
    print("✓ Testing actor mode...")
    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="actor")
    
    # Load the model
    loaded = wrapper.to_pipeline(device="cpu")
    
    test_input = torch.randn(3, 10)
    result = loaded(test_input)
    
    assert result.shape == (3, 5), f"Expected shape (3, 5), got {result.shape}"
    print("  ✓ Actor mode works correctly")


def verify_backward_compatibility():
    """Verify backward compatibility with old API."""
    print("✓ Testing backward compatibility...")
    
    from ray_zerocopy import TaskWrapper, ActorWrapper
    
    # TaskWrapper
    model = SimpleModel()
    task_wrapper = TaskWrapper(model)
    test_input = torch.randn(3, 10)
    result = task_wrapper(test_input)
    assert result.shape == (3, 5)
    print("  ✓ TaskWrapper backward compatibility works")
    
    # ActorWrapper
    actor_wrapper = ActorWrapper(model)
    loaded = actor_wrapper.load(device="cpu")
    result = loaded(test_input)
    assert result.shape == (3, 5)
    print("  ✓ ActorWrapper backward compatibility works")


def main():
    print("=" * 60)
    print("Verifying Unified ModelWrapper Implementation")
    print("=" * 60)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        verify_task_mode()
        verify_for_tasks_shortcut()
        verify_actor_mode()
        verify_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("✓ All verifications passed!")
        print("=" * 60)
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
