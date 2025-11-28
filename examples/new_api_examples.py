"""
Comprehensive examples demonstrating the new unified wrapper API.

This file shows all four wrapper classes side-by-side with realistic examples.
"""

import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy

# Initialize Ray (if not already initialized)
if not ray.is_initialized():
    ray.init()


# ============================================================================
# Example Models and Pipelines
# ============================================================================


class SimpleEncoder(nn.Module):
    """Simple encoder model for demonstration."""

    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))


class SimpleDecoder(nn.Module):
    """Simple decoder model for demonstration."""

    def __init__(self, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class Pipeline:
    """Pipeline combining encoder and decoder."""

    def __init__(self):
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()

    def __call__(self, data):
        """Process data through encoder then decoder."""
        encoded = self.encoder(data)
        return self.decoder(encoded)

    def process_batch(self, batch_data):
        """Alternative method name for batch processing."""
        return self(batch_data)


# ============================================================================
# Example 1: TaskWrapper - nn.Module with Ray Tasks
# ============================================================================


def example_task_wrapper():
    """
    TaskWrapper executes nn.Module models in Ray tasks with zero-copy loading.

    Use this when:
    - You want simple parallel inference via Ray tasks
    - Models are regular nn.Module (not TorchScript)
    - You don't need stateful actors
    """
    print("\n" + "=" * 70)
    print("Example 1: TaskWrapper (nn.Module + Ray Tasks)")
    print("=" * 70)

    from ray_zerocopy import TaskWrapper

    # Create and wrap pipeline
    pipeline = Pipeline()
    wrapped = TaskWrapper(pipeline)

    # Generate sample data
    sample_data = torch.randn(32, 784)

    # Call wrapped pipeline - model inference runs in Ray tasks!
    result = wrapped(sample_data)

    print(f"Input shape: {sample_data.shape}")
    print(f"Output shape: {result.shape}")
    print("✓ Models ran in Ray tasks with zero-copy loading")

    # Can also call other methods
    result2 = wrapped.process_batch(sample_data)
    print(f"✓ Also works with custom methods: {result2.shape}")


# ============================================================================
# Example 2: ActorWrapper - nn.Module with Ray Actors
# ============================================================================


def example_actor_wrapper():
    """
    ActorWrapper prepares nn.Module models for Ray actors (e.g., Ray Data).

    Use this when:
    - Using Ray Data with ActorPoolStrategy
    - Want stateful actors that load models once
    - Models are regular nn.Module (not TorchScript)
    """
    print("\n" + "=" * 70)
    print("Example 2: ActorWrapper (nn.Module + Ray Actors)")
    print("=" * 70)

    from ray_zerocopy import ActorWrapper

    # Create and wrap pipeline
    pipeline = Pipeline()
    actor_wrapper = ActorWrapper(
        pipeline, device="cpu"
    )  # Use "cuda:0" if GPU available

    # Define actor class
    class InferenceActor:
        def __init__(self, actor_wrapper):
            # Load pipeline inside actor
            self.pipeline = actor_wrapper.load()

        def __call__(self, batch):
            # Process batch through pipeline
            return {"predictions": self.pipeline(batch["data"])}

    # Create synthetic dataset
    def generate_data(batch):
        return {"data": torch.randn(10, 784)}

    ds = ray.data.range(100).map_batches(generate_data)

    # Run inference with actor pool
    results = ds.map_batches(
        InferenceActor,
        fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
        compute=ActorPoolStrategy(size=2),  # 2 actors in pool
        batch_size=10,
    )

    print(f"✓ Processed {results.count()} batches with actor pool")
    sample = results.take(1)[0]
    print(f"✓ Sample output shape: {sample['predictions'].shape}")


# ============================================================================
# Example 3: JITTaskWrapper - TorchScript with Ray Tasks
# ============================================================================


def example_jit_task_wrapper():
    """
    JITTaskWrapper executes TorchScript models in Ray tasks with zero-copy.

    Use this when:
    - Models are traced/scripted with torch.jit
    - Want simple parallel inference via Ray tasks
    - Don't need stateful actors
    """
    print("\n" + "=" * 70)
    print("Example 3: JITTaskWrapper (TorchScript + Ray Tasks)")
    print("=" * 70)

    from ray_zerocopy import JITTaskWrapper

    # Create pipeline and trace it
    pipeline = Pipeline()

    # Trace encoder and decoder separately
    example_input = torch.randn(1, 784)
    jit_encoder = torch.jit.trace(pipeline.encoder, example_input)

    example_encoded = jit_encoder(example_input)
    jit_decoder = torch.jit.trace(pipeline.decoder, example_encoded)

    # Create JIT pipeline
    class JITPipeline:
        def __init__(self):
            self.encoder = jit_encoder
            self.decoder = jit_decoder

        def __call__(self, data):
            encoded = self.encoder(data)
            return self.decoder(encoded)

    jit_pipeline = JITPipeline()

    # Wrap with JITTaskWrapper
    wrapped = JITTaskWrapper(jit_pipeline)

    # Run inference
    sample_data = torch.randn(32, 784)
    result = wrapped(sample_data)

    print(f"Input shape: {sample_data.shape}")
    print(f"Output shape: {result.shape}")
    print("✓ TorchScript models ran in Ray tasks with zero-copy loading")


# ============================================================================
# Example 4: JITActorWrapper - TorchScript with Ray Actors (NEW!)
# ============================================================================


def example_jit_actor_wrapper():
    """
    JITActorWrapper prepares TorchScript models for Ray actors.

    This is NEW functionality! Previously, TorchScript didn't work with actors.

    Use this when:
    - Using Ray Data with ActorPoolStrategy
    - Models are traced/scripted with torch.jit
    - Want compiled model performance + actor benefits
    """
    print("\n" + "=" * 70)
    print("Example 4: JITActorWrapper (TorchScript + Ray Actors) - NEW!")
    print("=" * 70)

    from ray_zerocopy import JITActorWrapper

    # Create pipeline and trace it
    pipeline = Pipeline()

    # Trace encoder and decoder
    example_input = torch.randn(1, 784)
    jit_encoder = torch.jit.trace(pipeline.encoder, example_input)

    example_encoded = jit_encoder(example_input)
    jit_decoder = torch.jit.trace(pipeline.decoder, example_encoded)

    # Create JIT pipeline
    class JITPipeline:
        def __init__(self):
            self.encoder = jit_encoder
            self.decoder = jit_decoder

        def __call__(self, data):
            encoded = self.encoder(data)
            return self.decoder(encoded)

    jit_pipeline = JITPipeline()

    # Wrap with JITActorWrapper
    actor_wrapper = JITActorWrapper(jit_pipeline, device="cpu")

    # Define actor class
    class JITInferenceActor:
        def __init__(self, actor_wrapper):
            # Load JIT pipeline inside actor
            self.pipeline = actor_wrapper.load()

        def __call__(self, batch):
            # Process batch through JIT pipeline
            return {"predictions": self.pipeline(batch["data"])}

    # Create synthetic dataset
    def generate_data(batch):
        return {"data": torch.randn(10, 784)}

    ds = ray.data.range(100).map_batches(generate_data)

    # Run inference with actor pool
    results = ds.map_batches(
        JITInferenceActor,
        fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
        compute=ActorPoolStrategy(size=2),
        batch_size=10,
    )

    print(f"✓ Processed {results.count()} batches with JIT actor pool")
    sample = results.take(1)[0]
    print(f"✓ Sample output shape: {sample['predictions'].shape}")
    print("✓ This is NEW! TorchScript now works with Ray actors!")


# ============================================================================
# Comparison: Old API vs New API
# ============================================================================


def comparison_old_vs_new():
    """
    Side-by-side comparison of old and new APIs.
    """
    print("\n" + "=" * 70)
    print("Comparison: Old API vs New API")
    print("=" * 70)

    pipeline = Pipeline()

    print("\n--- OLD API (deprecated) ---")
    print("from ray_zerocopy import rewrite_pipeline")
    print("rewritten = rewrite_pipeline(pipeline)")

    print("\n--- NEW API (recommended) ---")
    print("from ray_zerocopy import TaskWrapper")
    print("wrapped = TaskWrapper(pipeline)")

    print("\n✓ New API is more explicit and consistent across use cases")


# ============================================================================
# Main: Run all examples
# ============================================================================


def main():
    """Run all examples demonstrating the new API."""
    print("=" * 70)
    print("Ray ZeroCopy - New Unified Wrapper API Examples")
    print("=" * 70)

    # Run each example
    example_task_wrapper()
    example_actor_wrapper()
    example_jit_task_wrapper()
    example_jit_actor_wrapper()
    comparison_old_vs_new()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nFor more details, see:")
    print("- Migration guide: docs/migration_guide.md")
    print("- API documentation: help(ray_zerocopy.TaskWrapper)")
    print("=" * 70)


if __name__ == "__main__":
    main()
