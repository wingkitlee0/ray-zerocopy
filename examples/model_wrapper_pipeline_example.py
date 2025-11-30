"""
Example: Using ModelWrapper with a Pipeline and Ray Data

This example demonstrates using ModelWrapper with a pipeline object that contains
multiple models. This is useful for more complex inference workflows like:
- Encoder-Decoder architectures
- Multi-stage processing pipelines
- Feature extraction + classification pipelines
"""

import numpy as np
import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy

from ray_zerocopy import ModelWrapper


# Define individual model components
class Encoder(nn.Module):
    """Encoder network that compresses input."""

    def __init__(self, input_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    """Decoder network that processes encoded features."""

    def __init__(self, latent_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.ReLU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, x):
        return self.network(x)


# Define a pipeline that uses multiple models
class EncoderDecoderPipeline:
    """
    Pipeline combining encoder and decoder models.

    This demonstrates how ModelWrapper can handle objects with multiple
    nn.Module attributes, enabling zero-copy loading of complex pipelines.
    """

    def __init__(
        self, input_dim: int = 128, latent_dim: int = 64, output_dim: int = 10
    ):
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

        # Pipeline can also have non-model attributes
        self.temperature = 1.0

    def __call__(self, x):
        """Run the full pipeline."""
        # Encode
        latent = self.encoder(x)
        # Decode
        output = self.decoder(latent)
        # Apply temperature scaling
        return output / self.temperature

    def set_eval(self):
        """Set all models to eval mode."""
        self.encoder.eval()
        self.decoder.eval()


# Define the inference actor
class PipelineInferenceActor:
    """Actor for distributed inference using a ModelWrapper-wrapped pipeline."""

    def __init__(self, model_wrapper: ModelWrapper[EncoderDecoderPipeline]):
        """
        Initialize the actor with a wrapped pipeline.

        Args:
            model_wrapper: ModelWrapper containing the serialized pipeline
        """
        # Unwrap the pipeline (zero-copy loading of both encoder and decoder)
        self.pipeline = model_wrapper.load()
        self.pipeline.set_eval()

        print("Actor initialized with pipeline:")
        print(f"  - Encoder device: {next(self.pipeline.encoder.parameters()).device}")
        print(f"  - Decoder device: {next(self.pipeline.decoder.parameters()).device}")

    def __call__(self, batch: dict) -> dict:
        """Run inference through the full pipeline."""
        with torch.no_grad():
            inputs = torch.tensor(np.vstack(batch["data"]), dtype=torch.float32)
            outputs = self.pipeline(inputs)
            return {"predictions": outputs.numpy()}


def main():
    """Main function demonstrating pipeline usage with ModelWrapper."""

    ray.init(ignore_reinit_error=True)

    try:
        print("=" * 80)
        print("ModelWrapper Pipeline Example with Ray Data")
        print("=" * 80)

        # Step 1: Create the pipeline
        print("\n1. Creating encoder-decoder pipeline...")
        pipeline = EncoderDecoderPipeline(input_dim=128, latent_dim=64, output_dim=10)
        pipeline.set_eval()

        encoder_params = sum(p.numel() for p in pipeline.encoder.parameters())
        decoder_params = sum(p.numel() for p in pipeline.decoder.parameters())
        total_params = encoder_params + decoder_params

        print("   Pipeline created:")
        print(f"   - Encoder: {encoder_params:,} parameters")
        print(f"   - Decoder: {decoder_params:,} parameters")
        print(f"   - Total: {total_params:,} parameters")

        # Step 2: Wrap the entire pipeline with ModelWrapper
        print("\n2. Wrapping pipeline with ModelWrapper...")
        wrapper = ModelWrapper.from_model(pipeline)
        print("   ✓ Pipeline wrapped successfully")
        print(f"   - Models detected: {list(wrapper._model_refs.keys())}")
        print(f"   - Is standalone module: {wrapper._is_standalone_module}")

        # Step 3: Create dataset
        print("\n3. Creating synthetic dataset...")
        import numpy as np

        num_samples = 500
        data = [{"data": np.random.randn(128).tolist()} for _ in range(num_samples)]
        ds = ray.data.from_items(data)
        print(f"   Created dataset with {ds.count()} samples")

        # Step 4: Run distributed inference
        print("\n4. Running distributed inference with 3 actors...")
        print("   (Each actor loads the complete pipeline using zero-copy)")

        result_ds = ds.map_batches(
            PipelineInferenceActor,
            fn_constructor_kwargs={"model_wrapper": wrapper},
            batch_size=25,
            compute=ActorPoolStrategy(size=3),
        )

        # Step 5: Collect and display results
        print("\n5. Collecting results...")
        results = result_ds.take_all()
        print(f"   ✓ Processed {len(results)} batches")

        print("\n6. Sample predictions:")
        for i, batch in enumerate(results[:3]):
            predictions = batch["predictions"]
            print(
                f"   Batch {i}: shape={predictions.shape}, "
                f"mean={predictions.mean():.4f}, std={predictions.std():.4f}"
            )

        print("\n" + "=" * 80)
        print("Pipeline example completed successfully!")
        print("=" * 80)
        print("\nKey features demonstrated:")
        print("  • Pipeline support: Wrap objects with multiple nn.Module attributes")
        print(
            "  • Auto-discovery: ModelWrapper automatically finds all models in the pipeline"
        )
        print("  • Zero-copy: All actors share the same encoder & decoder weights")
        print(
            "  • Preserve state: Non-model attributes (like temperature) are preserved"
        )

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
