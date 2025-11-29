"""
Example: Using ModelWrapper with Ray Data map_batches

This example demonstrates the new ModelWrapper API for zero-copy model loading
with Ray Data. It shows:
1. Wrapping a standalone PyTorch model with ModelWrapper
2. Using it with Ray Data's map_batches and ActorPoolStrategy
3. Zero-copy loading across multiple actors for memory efficiency
"""

import argparse

import numpy as np
import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy

from ray_zerocopy import ModelWrapper
from ray_zerocopy.benchmark import (
    create_large_model,
    estimate_model_size_mb,
    monitor_memory_context,
)


# Define a simple PyTorch model
class SimpleClassifier(nn.Module):
    """A simple neural network for demonstration."""

    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 10
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


# Define the inference actor that will use the wrapped model
class InferenceActor:
    """Actor for distributed inference using ModelWrapper."""

    def __init__(self, model_wrapper: ModelWrapper[nn.Module]):
        self.model = model_wrapper.load()
        self.model.eval()
        print(
            f"Actor initialized with model on device: {next(self.model.parameters()).device}"
        )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        with torch.no_grad():
            arr = np.vstack(batch["data"])
            outputs = self.model(torch.tensor(arr, dtype=torch.float32))
            return {"predictions": outputs.numpy()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False

    # Step 1: Create and prepare the model
    print("\n1. Creating model...")
    # model = SimpleClassifier(input_dim=args.input_dim, hidden_dim=256, output_dim=10)
    model = create_large_model()
    model.eval()

    model_size_mb = estimate_model_size_mb(model)
    print(f"Model size: {model_size_mb:.1f} MB")

    # Step 2: Wrap the model with ModelWrapper
    wrapper = ModelWrapper.from_model(model)

    # Create Ray Dataset
    with monitor_memory_context() as memory_stats:
        results = (
            ray.data.range(args.num_samples)
            .map(lambda batch: {"data": np.random.randn(args.input_dim).tolist()})
            .map_batches(
                InferenceActor,
                fn_constructor_kwargs={"model_wrapper": wrapper},
                batch_size=32,
                compute=ActorPoolStrategy(size=args.workers),
            )
            .take_all()
        )


if __name__ == "__main__":
    main()
