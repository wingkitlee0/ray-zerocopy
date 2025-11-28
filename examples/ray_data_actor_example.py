"""
Example: Using zero-copy model loading with Ray Data actors

This example demonstrates how to use the actor-based zero-copy loading
for efficient model inference with Ray Data's map_batches and ActorPoolStrategy.
"""

import numpy as np
import ray
import torch
from ray.data import ActorPoolStrategy

from ray_zerocopy.actor import (
    load_model_in_actor,
    load_pipeline_in_actor,
    prepare_model_for_actors,
    rewrite_pipeline_for_actors,
)


# Example 1: Simple model with manual actor class
def example_simple_model():
    """Example using a simple model with manual actor implementation."""

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )

    # Prepare model for actors (stores in Ray object store)
    model_ref = prepare_model_for_actors(model)

    # Define actor class
    class InferenceActor:
        def __init__(self, model_ref):
            # Load model from object store (zero-copy)
            self.model = load_model_in_actor(model_ref, device="cpu")

        def __call__(self, batch):
            # Convert batch to tensor
            inputs = torch.tensor(batch["data"], dtype=torch.float32)

            # Run inference
            with torch.no_grad():
                outputs = self.model(inputs)

            # Return results
            return {"predictions": outputs.numpy()}

    # Create sample dataset
    ds = ray.data.range(1000).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 100)}, batch_size=32
    )

    # Run inference using actor pool (4 actors share the same model via zero-copy)
    results = ds.map_batches(
        InferenceActor,
        fn_constructor_kwargs={"model_ref": model_ref},
        batch_size=32,
        compute=ActorPoolStrategy(size=4),
    )

    print("Simple model example:")
    print(f"Total batches processed: {results.count()}")
    first_batch = results.take_batch(1)
    print(f"First batch keys: {list(first_batch.keys())}")
    print(
        f"Prediction shape (first batch): {first_batch['predictions'].shape if hasattr(first_batch['predictions'], 'shape') else len(first_batch['predictions'])}"
    )


# Example 2: Pipeline with multiple models
def example_pipeline():
    """Example using a pipeline with multiple models."""

    class EncoderDecoderPipeline:
        def __init__(self):
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(100, 64),
                torch.nn.ReLU(),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(64, 10),
                torch.nn.Softmax(dim=1),
            )

        def __call__(self, inputs):
            encoded = self.encoder(inputs)
            return self.decoder(encoded)

    # Create pipeline
    pipeline = EncoderDecoderPipeline()

    # Prepare pipeline for actors
    pipeline_skeleton, model_refs = rewrite_pipeline_for_actors(pipeline)

    # Define actor class
    class PipelineActor:
        def __init__(self, pipeline_skeleton, model_refs):
            # Load pipeline with all models (zero-copy for each model)
            self.pipeline = load_pipeline_in_actor(
                pipeline_skeleton, model_refs, device="cpu"
            )

        def __call__(self, batch):
            inputs = torch.tensor(batch["data"], dtype=torch.float32)

            with torch.no_grad():
                outputs = self.pipeline(inputs)

            return {"predictions": outputs.numpy()}

    # Create sample dataset
    ds = ray.data.range(1000).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 100)}, batch_size=32
    )

    # Run inference
    results = ds.map_batches(
        PipelineActor,
        fn_constructor_kwargs={
            "pipeline_skeleton": pipeline_skeleton,
            "model_refs": model_refs,
        },
        batch_size=32,
        compute=ActorPoolStrategy(size=4),
    )

    print("\nPipeline example:")
    print(f"Total batches processed: {results.count()}")
    first_batch = results.take_batch(1)
    print(f"First batch keys: {list(first_batch.keys())}")


# Example 3: Actor with preprocessing/postprocessing
def example_with_processing():
    """Example showing preprocessing and postprocessing in the actor."""

    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(50, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 5),
            )

        def forward(self, x):
            return self.layers(x)

    model = SimpleModel()

    # Prepare for actors
    model_ref = prepare_model_for_actors(model)

    # Actor with pre/post processing built-in
    class ProcessingActor:
        def __init__(self, model_ref, device="cpu"):
            self.model = load_model_in_actor(model_ref, device=device)

        def __call__(self, batch):
            # Preprocess
            inputs = torch.tensor(batch["data"], dtype=torch.float32)

            # Inference
            with torch.no_grad():
                outputs = self.model(inputs)

            # Postprocess
            predictions = outputs.argmax(dim=1).numpy()
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()

            return {"predictions": predictions, "probabilities": probabilities}

    # Create dataset
    ds = ray.data.range(500).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 50)}, batch_size=32
    )

    # Run inference with custom device
    results = ds.map_batches(
        ProcessingActor,
        fn_constructor_kwargs={"model_ref": model_ref, "device": "cpu"},
        batch_size=32,
        compute=ActorPoolStrategy(size=2),
    )

    print("\nPreprocessing example:")
    print(f"Total batches processed: {results.count()}")
    first_batch = results.take_batch(1)
    print(f"Sample result keys: {first_batch.keys()}")
    print(f"Predictions type: {type(first_batch['predictions'])}")
    if hasattr(first_batch["predictions"], "shape"):
        print(f"Predictions shape: {first_batch['predictions'].shape}")
    else:
        print(
            f"Predictions (first few): {first_batch['predictions'][:5] if len(first_batch['predictions']) > 5 else first_batch['predictions']}"
        )


# Example 4: GPU-based inference
def example_gpu():
    """Example using GPU for inference (if available)."""

    if not torch.cuda.is_available():
        print("\nGPU example: Skipping (no GPU available)")
        return

    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )

    # Prepare for actors
    model_ref = prepare_model_for_actors(model)

    # Actor with GPU
    class GPUInferenceActor:
        def __init__(self, model_ref):
            # Load model directly onto GPU
            self.model = load_model_in_actor(model_ref, device="cuda:0")

        def __call__(self, batch):
            # Move data to GPU
            inputs = torch.tensor(batch["data"], dtype=torch.float32, device="cuda:0")

            with torch.no_grad():
                outputs = self.model(inputs)

            # Move results back to CPU
            return {"predictions": outputs.cpu().numpy()}

    # Create dataset
    ds = ray.data.range(1000).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 100)}, batch_size=32
    )

    # Run with GPU actors (1 actor per GPU)
    results = ds.map_batches(
        GPUInferenceActor,
        fn_constructor_kwargs={"model_ref": model_ref},
        batch_size=32,
        compute=ActorPoolStrategy(size=1),  # Typically 1 actor per GPU
        num_gpus=1,  # Request GPU resources
    )

    print("\nGPU example:")
    print(f"Total batches processed: {results.count()}")


if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Run examples
        example_simple_model()
        example_pipeline()
        example_with_processing()
        example_gpu()

        print("\n✅ All examples completed successfully!")

    finally:
        ray.shutdown()
