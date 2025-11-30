# ModelWrapper Guide

ModelWrapper enables zero-copy model sharing for PyTorch inference in Ray. It supports two execution modes: **task-based** for ad-hoc inference and **actor-based** for batch processing with Ray Data or Ray Actors.

## What is Zero-Copy?

Zero-copy means sharing model weights across Ray workers without duplicating them in memory:

1. Model weights are stored **once** in Ray's object store
2. Multiple workers **reference** the same memory location via object references
3. No duplication = significant memory savings

**Example:** 4 actors with a 5GB model use ~5GB total (not 20GB) because they share the same model weights from the object store.

## Quick Start

### Installation

```bash
pip install ray-zerocopy
```

**Requirements:**
- Python 3.11+
- PyTorch 2.0+
- Ray 2.43+

## Task Mode

Use task mode for ad-hoc inference calls. Each call spawns a Ray task with zero-copy model loading.

### Example: Task-Based Inference

```python
import torch
import torch.nn as nn
from ray_zerocopy import ModelWrapper

# Define your model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.network(x)

# Create and wrap model
model = SimpleClassifier()
model.eval()
wrapped = ModelWrapper.for_tasks(model)

# Use immediately - each call spawns a Ray task
result = wrapped(torch.randn(1, 128))
```

## Actor Mode

Use actor mode for batch processing with Ray Data or long-running Ray Actors. The same pattern works for both.

### Example: Actor-Based Inference

```python
import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper

# Define your model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.network(x)

# Create and wrap model
model = SimpleClassifier()
model.eval()
model_wrapper = ModelWrapper.from_model(model, mode="actor")

# Define inference actor
class InferenceActor:
    def __init__(self, model_wrapper):
        # Load model once per actor (zero-copy, on CPU)
        self.model = model_wrapper.load()
        self.model.eval()

    def __call__(self, batch):
        inputs = torch.tensor(batch["data"], dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(inputs)
        return {"predictions": outputs.numpy()}

# Use with Ray Data
ds = ray.data.from_items([{"data": [0.1] * 128} for _ in range(100)])
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),  # 4 actors share the model
)

# Or use with Ray Actors
@ray.remote
class RayInferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()
        self.model.eval()

    def predict(self, data):
        with torch.no_grad():
            return self.model(torch.tensor(data, dtype=torch.float32))

actors = [RayInferenceActor.remote(model_wrapper) for _ in range(4)]
results = ray.get([actor.predict.remote([0.1] * 128) for actor in actors])
```

**Note:** The same actor pattern works for both Ray Data `map_batches` and Ray Actors. The only difference is whether you use `@ray.remote` decorator.

## Pipelines with Multiple Models

ModelWrapper automatically detects and shares all `nn.Module` attributes in your pipeline:

```python
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()  # nn.Module - shared
        self.decoder = DecoderModel()  # nn.Module - shared
        self.config = {"temp": 1.0}    # Regular attribute - copied

    def __call__(self, x):
        return self.decoder(self.encoder(x))

pipeline = MyPipeline()
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# In actor: both encoder and decoder are zero-copy shared
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load()  # Loads both models
```

## Device Placement

Models are loaded on CPU by default. Move them to GPU after loading if needed:

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()  # Loaded on CPU
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
```

## API Reference

### ModelWrapper.from_model()

```python
ModelWrapper.from_model(
    model_or_pipeline,
    mode="actor",  # "task" or "actor"
    model_attr_names=None,  # Optional: specify model attributes
    method_names=None,  # Optional: for task mode
)
```

### ModelWrapper.for_tasks()

Convenience shortcut for task mode:

```python
wrapped = ModelWrapper.for_tasks(model)
# Equivalent to: ModelWrapper.from_model(model, mode="task")
```

### model_wrapper.load()

Load the model in an actor (actor mode only):

```python
model = model_wrapper.load()  # Returns model on CPU
```

## When to Use What

| Scenario | Use This |
|----------|----------|
| Ad-hoc inference calls | `ModelWrapper.for_tasks()` |
| Ray Data batch inference | `ModelWrapper.from_model(..., mode="actor")` |
| Ray Actors (long-running) | `ModelWrapper.from_model(..., mode="actor")` |
| Batch processing workloads | `ModelWrapper.from_model(..., mode="actor")` |

## Memory Savings

**Without zero-copy:**
- Each actor loads its own copy: 4 actors × 5GB = 20GB

**With zero-copy:**
- Model stored once in object store: 5GB
- All actors reference the same memory: ~5GB total

## Next Steps

- See [JIT Wrappers](jit_wrappers.md) for TorchScript support (under development)
- Check [API Reference](../api_reference/model_wrappers.md) for detailed API docs
