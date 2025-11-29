# Tutorial: Basic Inference

This tutorial shows how to use ray-zerocopy for basic inference with a single PyTorch model and Ray Data.

## Goal

Process 1000 samples using 4 actors that share a single model via zero-copy, saving ~75% memory compared to loading 4 copies.

## Complete Code

```python
import numpy as np
import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper

# Step 1: Define a PyTorch model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
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

# Step 2: Create and wrap the model
model = SimpleClassifier(input_dim=128, hidden_dim=256, output_dim=10)
model.eval()

# Wrap for zero-copy sharing
model_wrapper = ModelWrapper.from_model(model, mode="actor")

# Step 3: Define the inference actor
class InferenceActor:
    def __init__(self, model_wrapper):
        # Load model once per actor (zero-copy, on CPU)
        self.model = model_wrapper.load()
        self.model.eval()

    def __call__(self, batch):
        # Convert batch data to tensor
        inputs = torch.tensor(batch["data"], dtype=torch.float32)

        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)

        # Return predictions
        return {"predictions": outputs.numpy()}

# Step 4: Create dataset
ds = ray.data.range(1000).map_batches(
    lambda batch: {
        "data": np.random.randn(len(batch["id"]), 128)
    },
    batch_size=32
)

# Step 5: Run inference with 4 actors
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),  # 4 actors share the model
)

# Step 6: Collect results
print(f"Total batches: {results.count()}")
first_batch = results.take_batch(1)
print(f"Prediction shape: {first_batch['predictions'].shape}")
```

## Step-by-Step Explanation

### Step 1: Define Your Model

Create a standard PyTorch model:

```python
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
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
```

Nothing special here - just a regular PyTorch model.

### Step 2: Wrap the Model

Create a `ModelWrapper` to enable zero-copy sharing:

```python
model = SimpleClassifier()
model.eval()  # Set to eval mode

model_wrapper = ModelWrapper.from_model(model, mode="actor")
```

The wrapper:
- Stores model weights in Ray's object store
- Prepares for zero-copy loading across actors
- Models are loaded on CPU; users handle device placement themselves

### Step 3: Define the Inference Actor

The actor loads the model and runs inference:

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        # Load model (zero-copy from object store, on CPU)
        self.model = model_wrapper.load()
        self.model.eval()

    def __call__(self, batch):
        # Process batch
        inputs = torch.tensor(batch["data"], dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(inputs)

        return {"predictions": outputs.numpy()}
```

Key points:
- `.load()` loads the model using zero-copy (no duplication!)
- Load happens in `__init__`, so it's done once per actor
- `__call__` runs for each batch

### Step 4: Create a Dataset

Use Ray Data to create your input dataset:

```python
ds = ray.data.range(1000).map_batches(
    lambda batch: {
        "data": np.random.randn(len(batch["id"]), 128)
    },
    batch_size=32
)
```

In practice, you'd load from files:
```python
ds = ray.data.read_parquet("s3://my-bucket/data/")
```

### Step 5: Run Inference

Use `map_batches` with `ActorPoolStrategy`:

```python
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),
)
```

This creates 4 actors that:
- Each loads the model once (zero-copy)
- Process batches in parallel
- Share the same underlying model weights

### Step 6: Collect Results

Get your predictions:

```python
print(f"Total batches: {results.count()}")
first_batch = results.take_batch(1)
print(f"Prediction shape: {first_batch['predictions'].shape}")
```

Or write to storage:
```python
results.write_parquet("s3://my-bucket/results/")
```

## Memory Savings

**Without zero-copy:**
- Actor 1: 200MB model
- Actor 2: 200MB model
- Actor 3: 200MB model
- Actor 4: 200MB model
- **Total: 800MB**

**With zero-copy:**
- Object store: 200MB (shared)
- Actor 1-4: references only
- **Total: ~200MB**

**75% memory reduction!**

## Next Steps

- Try the [Pipeline Tutorial](pipeline_example.md) for multi-model inference
- Learn about [Ray Data Integration](../user_guide/ray_data_integration.md)
- Explore the [ModelWrapper API](../api_reference/model_wrappers.md)
