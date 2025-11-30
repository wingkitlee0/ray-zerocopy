"""
Example: Using zero-copy model loading with Ray Data actors

This example demonstrates how to use ModelWrapper for
efficient model inference with Ray Data's map_batches and ActorPoolStrategy.
"""

import numpy as np
import ray
import torch
from ray.data import ActorPoolStrategy

from ray_zerocopy import ModelWrapper


# Example 1: Simple model with manual actor class
def example_simple_model():
    """Example using a simple model with ModelWrapper actor mode."""

    # Create a simple pipeline (wrapper around model)
    class SimplePipeline:
        def __init__(self):
            self.model = torch.nn.Sequential(
                torch.nn.Linear(100, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 10),
            )

        def __call__(self, inputs):
            return self.model(inputs)

    # Create and wrap pipeline for actors
    pipeline = SimplePipeline()
    model_wrapper = ModelWrapper.from_model(pipeline)

    class InferenceActor:
        def __init__(self, model_wrapper):
            self.pipeline = model_wrapper.load()

        def __call__(self, batch):
            # Convert batch to tensor
            inputs = torch.tensor(batch["data"], dtype=torch.float32)

            # Run inference
            with torch.no_grad():
                outputs = self.pipeline(inputs)

            # Return results
            return {"predictions": outputs.numpy()}

    # Create sample dataset
    ds = ray.data.range(1000).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 100)}, batch_size=32
    )

    # Run inference using actor pool (4 actors share the same model via zero-copy)
    results = ds.map_batches(
        InferenceActor,
        fn_constructor_kwargs={"model_wrapper": model_wrapper},
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

    # Wrap for actors - single line!
    model_wrapper = ModelWrapper.from_model(pipeline)

    # Define actor class - much cleaner!
    class PipelineActor:
        def __init__(self, model_wrapper):
            # Load pipeline with all models (zero-copy for each model)
            self.pipeline = model_wrapper.load()

        def __call__(self, batch):
            inputs = torch.tensor(batch["data"], dtype=torch.float32)

            with torch.no_grad():
                outputs = self.pipeline(inputs)

            return {"predictions": outputs.numpy()}

    # Create sample dataset
    ds = ray.data.range(1000).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 100)}, batch_size=32
    )

    # Run inference - simpler kwargs!
    results = ds.map_batches(
        PipelineActor,
        fn_constructor_kwargs={"model_wrapper": model_wrapper},
        batch_size=32,
        compute=ActorPoolStrategy(size=4),
    )

    print("\nPipeline example:")
    print(f"Total batches processed: {results.count()}")
    first_batch = results.take_batch(1)
    print(f"First batch keys: {list(first_batch.keys())}")


# Example 3: Actor with preprocessing/postprocessing
def example_with_processing():
    """Example showing preprocessing and postprocessing in the actor - NEW API."""

    # Create a simple pipeline
    class SimplePipeline:
        def __init__(self):
            self.model = torch.nn.Sequential(
                torch.nn.Linear(50, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 5),
            )

        def __call__(self, x):
            return self.model(x)

    pipeline = SimplePipeline()

    # Wrap for actors
    model_wrapper = ModelWrapper.from_model(pipeline)

    # Actor with pre/post processing built-in
    class ProcessingActor:
        def __init__(self, model_wrapper):
            self.pipeline = model_wrapper.load()

        def __call__(self, batch):
            # Preprocess
            inputs = torch.tensor(batch["data"], dtype=torch.float32)

            # Inference
            with torch.no_grad():
                outputs = self.pipeline(inputs)

            # Postprocess
            predictions = outputs.argmax(dim=1).numpy()
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()

            return {"predictions": predictions, "probabilities": probabilities}

    # Create dataset
    ds = ray.data.range(500).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 50)}, batch_size=32
    )

    # Run inference
    results = ds.map_batches(
        ProcessingActor,
        fn_constructor_kwargs={"model_wrapper": model_wrapper},
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
    """Example using GPU for inference (if available) - NEW API."""

    if not torch.cuda.is_available():
        print("\nGPU example: Skipping (no GPU available)")
        return

    # Create pipeline
    class GPUPipeline:
        def __init__(self):
            self.model = torch.nn.Sequential(
                torch.nn.Linear(100, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 10),
            )

        def __call__(self, x):
            return self.model(x)

    pipeline = GPUPipeline()

    # Wrap for actors with GPU device
    model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

    # Actor with GPU
    class GPUInferenceActor:
        def __init__(self, model_wrapper: ModelWrapper):
            self.pipeline = model_wrapper.load()

        def __call__(self, batch):
            # Move data to GPU
            inputs = torch.tensor(batch["data"], dtype=torch.float32)

            with torch.no_grad():
                outputs = self.pipeline(inputs)

            # Move results back to CPU
            return {"predictions": outputs.cpu().numpy()}

    # Create dataset
    ds = ray.data.range(1000).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 100)}, batch_size=32
    )

    # Run with GPU actors (1 actor per GPU)
    results = ds.map_batches(
        GPUInferenceActor,
        fn_constructor_kwargs={"model_wrapper": model_wrapper},
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
