"""
Example demonstrating zero-copy loading with TorchScript models.

This example shows how to use the TorchScript-specific functions to achieve
zero-copy model loading with compiled PyTorch models.
"""

import torch
import torch.nn as nn
import ray

from ray_zerocopy.jit import (
    extract_tensors,
    replace_tensors,
    rewrite_pipeline,
    ZeroCopyModel,
)


# Example 1: Simple model extraction and restoration
def example_basic_jit():
    """Basic example of extracting and replacing tensors in a TorchScript model."""
    print("=" * 60)
    print("Example 1: Basic TorchScript Zero-Copy")
    print("=" * 60)

    # Create a simple PyTorch model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # Create and trace the model to get TorchScript version
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    scripted_model = torch.jit.trace(model, example_input)

    print(f"Original model type: {type(scripted_model)}")

    # Test the original model
    test_input = torch.randn(5, 10)
    original_output = scripted_model(test_input)
    print(f"Original output shape: {original_output.shape}")

    # Extract tensors (separating weights from structure)
    model_bytes, tensors = extract_tensors_jit(scripted_model)
    print(f"\nModel skeleton size: {len(model_bytes):,} bytes")
    print(f"Number of tensors: {len(tensors)}")
    print(f"Tensor names: {list(tensors.keys())}")

    # Restore the model (zero-copy operation)
    restored_model = replace_tensors_jit(model_bytes, tensors)
    restored_output = restored_model(test_input)

    print(f"\nRestored output shape: {restored_output.shape}")
    print(f"Outputs match: {torch.allclose(original_output, restored_output)}")
    print()


# Example 2: Using with Ray for distributed inference
def example_ray_jit():
    """Example of using TorchScript zero-copy with Ray."""
    print("=" * 60)
    print("Example 2: TorchScript Zero-Copy with Ray")
    print("=" * 60)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create a TorchScript model
    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = ConvModel()
    model.eval()
    example_input = torch.randn(1, 3, 32, 32)
    scripted_model = torch.jit.trace(model, example_input)

    # Extract and put in Ray object store
    model_data = extract_tensors_jit(scripted_model)
    model_ref = ray.put(model_data)

    print(f"Model stored in Ray object store: {model_ref}")

    # Define a Ray remote function that uses the model
    @ray.remote
    def run_inference(model_data, input_data):
        # Ray automatically dereferences ObjectRefs passed as arguments
        model_bytes, tensors = model_data
        model = replace_tensors_jit(model_bytes, tensors)
        with torch.no_grad():
            return model(input_data)

    # Run inference in parallel on multiple Ray workers
    test_inputs = [torch.randn(2, 3, 32, 32) for _ in range(4)]
    futures = [run_inference.remote(model_ref, inp) for inp in test_inputs]
    results = ray.get(futures)

    print(f"\nRan inference on {len(results)} batches")
    print(f"First result shape: {results[0].shape}")
    print()


# Example 3: Using the high-level API with pipeline
def example_pipeline_jit():
    """Example using the high-level pipeline API."""
    print("=" * 60)
    print("Example 3: TorchScript Pipeline API")
    print("=" * 60)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create a pipeline object with TorchScript models
    class ModelPipeline:
        def __init__(self):
            # Create and script the model
            model = nn.Sequential(
                nn.Linear(20, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
            )
            model.eval()
            example_input = torch.randn(1, 20)
            self.model = torch.jit.trace(model, example_input)

        def predict(self, x):
            return self.model(x)

    # Original pipeline
    pipeline = ModelPipeline()
    test_input = torch.randn(3, 20)
    original_result = pipeline.predict(test_input)

    print(f"Original pipeline prediction shape: {original_result.shape}")

    # Rewrite pipeline for zero-copy loading
    rewritten_pipeline = rewrite_pipeline_jit(pipeline)

    print(f"Pipeline rewritten for zero-copy")
    print(f"Model type: {type(rewritten_pipeline.model)}")

    # Use the rewritten pipeline (calls happen via Ray)
    rewritten_result = rewritten_pipeline.predict(test_input)

    print(f"Rewritten pipeline prediction shape: {rewritten_result.shape}")
    print(
        f"Results match: {torch.allclose(original_result, rewritten_result, rtol=1e-4)}"
    )
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TorchScript Zero-Copy Examples")
    print("=" * 60 + "\n")

    example_basic_jit()
    example_ray_jit()
    example_pipeline_jit()

    if ray.is_initialized():
        ray.shutdown()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
