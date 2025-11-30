"""
Example demonstrating the unified ModelWrapper API.

This example shows how to use the new ModelWrapper class that supports
both task-based and actor-based execution modes.
"""

import ray
import torch
import torch.nn as nn

from ray_zerocopy import ModelWrapper


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def __call__(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Define a pipeline with multiple models
class EncoderDecoderPipeline:
    def __init__(self):
        self.encoder = SimpleModel(128, 256, 64)
        self.decoder = SimpleModel(64, 128, 32)

    def __call__(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


def example_task_mode():
    """Example 1: Task mode - Immediate execution via Ray tasks."""
    print("\n=== Example 1: Task Mode ===")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create a model and wrap it for task execution
    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="task")

    # The wrapper is immediately usable
    test_input = torch.randn(5, 128)
    result = wrapper(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {result.shape}")
    print("✓ Task mode: Model executes via Ray tasks with zero-copy loading")


def example_task_mode_shortcut():
    """Example 2: Task mode using for_tasks() convenience method."""
    print("\n=== Example 2: Task Mode (Shortcut) ===")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Use the convenience method - equivalent to rewrite_pipeline()
    model = SimpleModel()
    wrapper = ModelWrapper.for_tasks(model)

    # Immediately usable
    test_input = torch.randn(5, 128)
    result = wrapper(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {result.shape}")
    print("✓ for_tasks() provides a simple one-line API")


def example_actor_mode():
    """Example 3: Actor mode - Load models in Ray actors."""
    print("\n=== Example 3: Actor Mode ===")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create a model and wrap it for actor usage
    model = SimpleModel()
    wrapper = ModelWrapper.from_model(model, mode="actor")

    # Define an actor that loads the model
    @ray.remote
    class InferenceActor:
        def __init__(self, model_wrapper):
            # Load the model with zero-copy in the actor (on CPU)
            self.model = model_wrapper.load()

        def predict(self, x):
            return self.model(x)

    # Create actors and run inference
    actors = [InferenceActor.remote(wrapper) for _ in range(3)]

    # Run parallel inference
    test_input = torch.randn(5, 128)
    futures = [actor.predict.remote(test_input) for actor in actors]
    results = ray.get(futures)

    print(f"Input shape: {test_input.shape}")
    print(f"Number of actors: {len(actors)}")
    print(f"Output shape from each actor: {results[0].shape}")
    print("✓ Actor mode: Models loaded with zero-copy in each actor")

    # Cleanup
    for actor in actors:
        ray.kill(actor)


def example_pipeline_task_mode():
    """Example 4: Pipeline in task mode."""
    print("\n=== Example 4: Pipeline Task Mode ===")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create a pipeline and wrap it for task execution
    pipeline = EncoderDecoderPipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="task")

    # Immediately usable
    test_input = torch.randn(5, 128)
    result = wrapper(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {result.shape}")
    print("✓ Pipeline executes with each model call as a Ray task")


def example_pipeline_actor_mode():
    """Example 5: Pipeline in actor mode."""
    print("\n=== Example 5: Pipeline Actor Mode ===")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create a pipeline and wrap it for actor usage
    pipeline = EncoderDecoderPipeline()
    wrapper = ModelWrapper.from_model(pipeline, mode="actor")

    # Define an actor that loads the pipeline
    @ray.remote
    class PipelineActor:
        def __init__(self, pipeline_wrapper):
            # Load the pipeline with zero-copy (on CPU)
            self.pipeline = pipeline_wrapper.load()

        def process(self, x):
            return self.pipeline(x)

    # Create actors
    actors = [PipelineActor.remote(wrapper) for _ in range(2)]

    # Run inference
    test_input = torch.randn(5, 128)
    futures = [actor.process.remote(test_input) for actor in actors]
    results = ray.get(futures)

    print(f"Input shape: {test_input.shape}")
    print(f"Number of actors: {len(actors)}")
    print(f"Output shape from each actor: {results[0].shape}")
    print("✓ Pipeline loaded with zero-copy in each actor")

    # Cleanup
    for actor in actors:
        ray.kill(actor)


def example_mode_comparison():
    """Example 6: Comparing task vs actor mode."""
    print("\n=== Example 6: Mode Comparison ===")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    model = SimpleModel()

    # Task mode - for distributed inference with automatic scaling
    task_wrapper = ModelWrapper.from_model(model, mode="task")
    print("Task mode: Best for dynamic workloads with automatic task scheduling")

    # Actor mode - for stateful inference with dedicated resources
    actor_wrapper = ModelWrapper.from_model(model, mode="actor")
    print("Actor mode: Best for stateful workloads with persistent actors")

    # Demonstrate task mode usage
    test_input = torch.randn(5, 128)
    task_result = task_wrapper(test_input)
    print(f"✓ Task mode result shape: {task_result.shape}")

    # Demonstrate actor mode usage
    loaded_model = actor_wrapper.load()
    actor_result = loaded_model(test_input)
    print(f"✓ Actor mode result shape: {actor_result.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Unified ModelWrapper API Examples")
    print("=" * 60)

    example_task_mode()
    example_task_mode_shortcut()
    example_actor_mode()
    example_pipeline_task_mode()
    example_pipeline_actor_mode()
    example_mode_comparison()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

    # Cleanup
    ray.shutdown()
