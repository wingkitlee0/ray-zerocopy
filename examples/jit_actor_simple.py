"""
Simple working example of JITModelWrapper with Ray Data.
"""

import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy

from ray_zerocopy import JITModelWrapper

# Initialize Ray
ray.init(ignore_reinit_error=True)


# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


# Trace the model
model = SimpleModel()
model.eval()
jit_model = torch.jit.trace(model, torch.randn(1, 10))


# Wrap in a simple pipeline
class SimplePipeline:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(x)


pipeline = SimplePipeline(jit_model)

# Create wrapper
wrapper = JITModelWrapper.from_model(pipeline, mode="actor")


# Define actor class
class InferenceActor:
    def __init__(self, wrapper):
        self.pipeline = wrapper.load()

    def __call__(self, batch):
        # Convert numpy to tensor
        data = torch.tensor(batch["data"], dtype=torch.float32)
        with torch.no_grad():
            result = self.pipeline(data)
            return {"predictions": result.numpy()}


# Create dataset
def generate_data(batch):
    import numpy as np

    return {"data": np.random.randn(len(batch["id"]), 10).astype(np.float32)}


ds = ray.data.range(50).map_batches(generate_data, batch_size=10)

# Run inference with actor pool
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"wrapper": wrapper},
    batch_size=10,
    compute=ActorPoolStrategy(size=2),
)

print(f"✓ Processed {results.count()} rows")
print("✓ JITModelWrapper works with Ray Data!")

ray.shutdown()
