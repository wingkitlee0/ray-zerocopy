# Getting Started

## Installation

### From PyPI (when published)

```bash
pip install ray-zerocopy
```

### From Source

```bash
git clone https://github.com/yourusername/ray-zerocopy.git
cd ray-zerocopy
pip install -e .
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Ray 2.43+
- NumPy

## Quick Start

### For Ray Data Actor UDFs (Recommended for Batch Inference)

```python
from ray.data import ActorPoolStrategy
from ray_zerocopy import ActorWrapper

# 1. Create your pipeline (a class with nn.Module attributes)
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, data):
        encoded = self.encoder(data)
        return self.decoder(encoded)

pipeline = MyPipeline()

# 2. Wrap with ActorWrapper for zero-copy sharing
actor_wrapper = ActorWrapper(pipeline)

# 3. Define actor UDF that loads the pipeline
class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()

    def __call__(self, batch):
        with torch.no_grad():
            return self.pipeline(batch["data"])

# 4. Use with Ray Data
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4),  # 4 actors share the model
)
```

### For Ray Actors (General Purpose)

```python
import ray
from ray_zerocopy import ActorWrapper

# Wrap pipeline for actors
pipeline = MyPipeline()
actor_wrapper = ActorWrapper(pipeline)

# Define inference actor
@ray.remote
class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()

    def predict(self, data):
        with torch.no_grad():
            return self.pipeline(data)

# Create actors that share the model
actors = [InferenceActor.remote(actor_wrapper) for _ in range(4)]
results = ray.get([actor.predict.remote(data) for actor in actors])
```

### For Ray Tasks (Ad-hoc Inference)

```python
from ray_zerocopy import TaskWrapper

# A Pipeline is a class with nn.Module attributes
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, data):
        encoded = self.encoder(data)
        return self.decoder(encoded)

pipeline = MyPipeline()
wrapped = TaskWrapper(pipeline)

# Each call spawns a Ray task with zero-copy model loading
result = wrapped(data)
```

## When to Use What

| Scenario | Use This |
|----------|----------|
| Ray Data `map_batches` batch inference | `ActorWrapper` with Ray Data Actor UDF |
| High-throughput batch inference | `ActorWrapper` with Ray Data Actor UDF |
| Long-running inference service | `ActorWrapper` with Ray Actor |
| Ad-hoc task-based inference | `TaskWrapper` with Ray Task |
| Sporadic inference calls | `TaskWrapper` with Ray Task |

## Next Steps

- Learn about [Core Concepts](user_guide/core_concepts.md)
- Explore [Tutorials](tutorials/index.md)
- Check the [API Reference](api_reference/index.md)
