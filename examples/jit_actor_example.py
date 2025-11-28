"""
Example: Using zero-copy TorchScript model loading with Ray Data actors (NEW!)

This example demonstrates the NEW JITActorWrapper for efficient TorchScript
model inference with Ray Data's map_batches and ActorPoolStrategy.

This functionality was NOT previously available - TorchScript models can now
be used with Ray actors just like regular nn.Module models!
"""

import numpy as np
import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy

from ray_zerocopy import JITActorWrapper


# Example 1: Simple JIT model with actors
def example_simple_jit_actor():
    """Example using a simple TorchScript model with actors."""
    print("=" * 70)
    print("Example 1: Simple JIT Model with Actors (NEW!)")
    print("=" * 70)

    # Create a simple PyTorch model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # Create and trace the model
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 100)
    jit_model = torch.jit.trace(model, example_input)

    # Wrap the TorchScript model in a pipeline
    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def __call__(self, x):
            return self.model(x)

    # Create and wrap pipeline for actors
    pipeline = JITPipeline()
    actor_wrapper = JITActorWrapper(pipeline, device="cpu")

    # Define actor class
    class JITInferenceActor:
        def __init__(self, actor_wrapper):
            # Load JIT pipeline from wrapper (zero-copy)
            self.pipeline = actor_wrapper.load()

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

    # Run inference using actor pool
    results = ds.map_batches(
        JITInferenceActor,
        fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
        batch_size=32,
        compute=ActorPoolStrategy(size=4),
    )

    print(f"✓ Processed {results.count()} batches with JIT actor pool")
    first_batch = results.take_batch(1)
    print(f"✓ First batch keys: {list(first_batch.keys())}")
    print(f"✓ Prediction shape: {first_batch['predictions'].shape}")
    print("✓ TorchScript models now work with Ray actors!")


# Example 2: Multi-model JIT pipeline with actors
def example_jit_pipeline_actors():
    """Example using a pipeline with multiple TorchScript models."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Model JIT Pipeline with Actors")
    print("=" * 70)

    # Create encoder model
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.fc(x)

    # Create decoder model
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(64, 10),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            return self.fc(x)

    # Trace both models
    encoder = Encoder()
    encoder.eval()
    encoder_input = torch.randn(1, 100)
    jit_encoder = torch.jit.trace(encoder, encoder_input)

    decoder = Decoder()
    decoder.eval()
    decoder_input = torch.randn(1, 64)
    jit_decoder = torch.jit.trace(decoder, decoder_input)

    # Create pipeline with both JIT models
    class EncoderDecoderPipeline:
        def __init__(self):
            self.encoder = jit_encoder
            self.decoder = jit_decoder

        def __call__(self, inputs):
            encoded = self.encoder(inputs)
            return self.decoder(encoded)

    # Wrap for actors
    pipeline = EncoderDecoderPipeline()
    actor_wrapper = JITActorWrapper(pipeline, device="cpu")

    # Define actor class
    class PipelineActor:
        def __init__(self, actor_wrapper):
            self.pipeline = actor_wrapper.load()

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
        fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
        batch_size=32,
        compute=ActorPoolStrategy(size=4),
    )

    print(f"✓ Processed {results.count()} batches")
    print("✓ Multiple TorchScript models loaded with zero-copy in each actor")


# Example 3: JIT actor with preprocessing and postprocessing
def example_jit_with_processing():
    """Example showing preprocessing and postprocessing with JIT actors."""
    print("\n" + "=" * 70)
    print("Example 3: JIT Actor with Pre/Post Processing")
    print("=" * 70)

    # Create a classification model
    class ClassificationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(),
                nn.Linear(128, 5),
            )

        def forward(self, x):
            return self.layers(x)

    # Trace the model
    model = ClassificationModel()
    model.eval()
    example_input = torch.randn(1, 50)
    jit_model = torch.jit.trace(model, example_input)

    # Wrap in pipeline
    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def __call__(self, x):
            return self.model(x)

    pipeline = JITPipeline()
    actor_wrapper = JITActorWrapper(pipeline, device="cpu")

    # Actor with pre/post processing
    class ProcessingActor:
        def __init__(self, actor_wrapper):
            self.pipeline = actor_wrapper.load()

        def __call__(self, batch):
            # Preprocess
            inputs = torch.tensor(batch["data"], dtype=torch.float32)

            # Inference with JIT model
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
        fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
        batch_size=32,
        compute=ActorPoolStrategy(size=2),
    )

    print(f"✓ Processed {results.count()} batches")
    first_batch = results.take_batch(1)
    print(f"✓ Result keys: {first_batch.keys()}")
    print(f"✓ Predictions shape: {first_batch['predictions'].shape}")
    print("✓ JIT models support full pre/post processing pipelines!")


# Example 4: GPU-based JIT inference (if available)
def example_jit_gpu():
    """Example using GPU for JIT inference (if available)."""
    print("\n" + "=" * 70)
    print("Example 4: JIT GPU Inference")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("Skipping (no GPU available)")
        return

    # Create a larger model for GPU
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            return self.layers(x)

    # Trace the model
    model = LargeModel()
    model.eval()
    example_input = torch.randn(1, 100)
    jit_model = torch.jit.trace(model, example_input)

    # Wrap in pipeline
    class JITPipeline:
        def __init__(self):
            self.model = jit_model

        def __call__(self, x):
            return self.model(x)

    pipeline = JITPipeline()
    actor_wrapper = JITActorWrapper(pipeline, device="cuda:0")

    # GPU actor
    class GPUJITActor:
        def __init__(self, actor_wrapper):
            # Load JIT pipeline directly onto GPU
            self.pipeline = actor_wrapper.load()

        def __call__(self, batch):
            # Move data to GPU
            inputs = torch.tensor(batch["data"], dtype=torch.float32, device="cuda:0")

            with torch.no_grad():
                outputs = self.pipeline(inputs)

            # Move results back to CPU
            return {"predictions": outputs.cpu().numpy()}

    # Create dataset
    ds = ray.data.range(1000).map_batches(
        lambda batch: {"data": np.random.randn(len(batch["id"]), 100)}, batch_size=32
    )

    # Run with GPU actor
    results = ds.map_batches(
        GPUJITActor,
        fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
        batch_size=32,
        compute=ActorPoolStrategy(size=1),
        num_gpus=1,
    )

    print(f"✓ Processed {results.count()} batches on GPU")
    print("✓ TorchScript models work great with GPU actors!")


def main():
    """Run all JIT actor examples."""
    print("\n" + "=" * 70)
    print("Ray ZeroCopy - JIT Actor Examples (NEW FUNCTIONALITY!)")
    print("=" * 70)
    print("\nThese examples demonstrate TorchScript support for Ray actors,")
    print("which was NOT previously available in ray_zerocopy!")
    print("=" * 70)

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Run examples
        example_simple_jit_actor()
        example_jit_pipeline_actors()
        example_jit_with_processing()
        example_jit_gpu()

        print("\n" + "=" * 70)
        print("✅ All JIT actor examples completed successfully!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("- TorchScript models can now be used with Ray actors")
        print("- Zero-copy loading works the same as nn.Module")
        print("- JITActorWrapper provides the same clean API as ActorWrapper")
        print("- Supports GPU inference just like regular models")
        print("=" * 70)

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
